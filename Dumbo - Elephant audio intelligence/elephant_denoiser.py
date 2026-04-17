"""
🐘 ElephantVoices Rumble Denoiser — HackSMU VII
Isolates elephant rumble vocalizations from mechanical noise.
"""

import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as signal
from scipy.ndimage import gaussian_filter, uniform_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, json, argparse, base64, time
from io import BytesIO
from pathlib import Path

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    print("⚠️  pip install librosa")

try:
    import noisereduce as nr
    HAS_NR = True
except ImportError:
    HAS_NR = False
    print("⚠️  pip install noisereduce  (strongly recommended for best results)")

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


class Config:
    SAMPLE_RATE = 48000
    NFFT = 8192
    HOP_LENGTH = 2048
    WIN_LENGTH = 8192
    FUNDAMENTAL_LOW = 8
    FUNDAMENTAL_HIGH = 25
    MAX_HARMONIC_FREQ = 800      # Lowered — most useful harmonics are below 800Hz
    MAX_HARMONIC_ORDER = 60
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
    GEMINI_MODEL = "gemini-2.0-flash"
    ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY", "")
    ELEVENLABS_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"


# ═══════════════════════════════════════════════════════════════
# AUDIO I/O
# ═══════════════════════════════════════════════════════════════

def load_audio(filepath):
    if HAS_LIBROSA:
        y, sr = librosa.load(filepath, sr=None, mono=True)
    else:
        sr, y = wav.read(filepath)
        if y.ndim > 1: y = y.mean(axis=1)
        y = y.astype(np.float32)
        if np.max(np.abs(y)) > 1.0: y = y / np.max(np.abs(y))
    print(f"✅ Loaded: {filepath} | SR={sr} Hz | {len(y)/sr:.1f}s")
    return y, sr

def compute_stft(y, sr):
    if HAS_LIBROSA:
        S = librosa.stft(y, n_fft=Config.NFFT, hop_length=Config.HOP_LENGTH,
                         win_length=Config.WIN_LENGTH)
    else:
        _, _, S = signal.stft(y, fs=sr, nperseg=Config.WIN_LENGTH,
                               noverlap=Config.WIN_LENGTH - Config.HOP_LENGTH,
                               nfft=Config.NFFT)
    return np.abs(S), np.angle(S), S

def reconstruct_audio(mag, phase, sr):
    S = mag * np.exp(1j * phase)
    if HAS_LIBROSA:
        y = librosa.istft(S, hop_length=Config.HOP_LENGTH, win_length=Config.WIN_LENGTH)
    else:
        _, y = signal.istft(S, fs=sr, nperseg=Config.WIN_LENGTH,
                             noverlap=Config.WIN_LENGTH - Config.HOP_LENGTH,
                             nfft=Config.NFFT)
    return y / (np.max(np.abs(y)) + 1e-10)

def save_audio(y, sr, fp):
    wav.write(fp, sr, np.int16(y * 32767))
    print(f"💾 Saved: {fp}")

def t2f(t, sr):
    return int(t * sr / Config.HOP_LENGTH)

def t2s(t, sr):
    """Time in seconds to sample index."""
    return int(t * sr)


# ═══════════════════════════════════════════════════════════════
# SPECTROGRAM PLOTTING
# ═══════════════════════════════════════════════════════════════

def plot_spectrogram(mag, sr, title="", max_freq=1500, save_path=None,
                     vmin_db=None, vmax_db=None, call_start=None, call_end=None):
    fig, ax = plt.subplots(figsize=(14, 6))
    freq = np.linspace(0, sr/2, mag.shape[0])
    mb = np.searchsorted(freq, max_freq)
    nf = mag.shape[1]
    tsec = np.linspace(0, nf * Config.HOP_LENGTH / sr, nf)
    db = 20 * np.log10(mag[:mb] + 1e-10)
    if vmin_db is None: vmin_db = np.percentile(db, 5)
    if vmax_db is None: vmax_db = np.percentile(db, 99)
    ax.imshow(db, aspect='auto', origin='lower',
              extent=[tsec[0], tsec[-1], 0, max_freq],
              cmap='inferno', vmin=vmin_db, vmax=vmax_db)
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Freq (Hz)'); ax.set_title(title)
    if call_start is not None:
        ax.axvline(x=call_start, color='lime', lw=1.5, alpha=.7)
    if call_end is not None:
        ax.axvline(x=call_end, color='lime', lw=1.5, alpha=.7)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Saved: {save_path}")
    plt.close(fig)
    return vmin_db, vmax_db


def spectrogram_to_b64(mag, sr):
    fig = plt.figure(figsize=(8, 4))
    freq = np.linspace(0, sr/2, mag.shape[0])
    mb = np.searchsorted(freq, 1500)
    db = 20*np.log10(mag[:mb]+1e-10)
    plt.imshow(db, aspect='auto', origin='lower', extent=[0,mag.shape[1],0,1500], cmap='inferno')
    plt.xlabel('Frame'); plt.ylabel('Hz'); plt.colorbar(label='dB'); plt.tight_layout()
    buf = BytesIO(); fig.savefig(buf, format='png', dpi=72); plt.close(fig); buf.seek(0)
    return base64.b64encode(buf.read()).decode()


# ═══════════════════════════════════════════════════════════════
# GEMINI CLASSIFICATION
# ═══════════════════════════════════════════════════════════════

def classify_noise_gemini(mag, sr):
    if not Config.GEMINI_API_KEY:
        print("⚠️  GEMINI_API_KEY not set.")
        return {"noise_type": "unknown", "confidence": 0.0}

    img = spectrogram_to_b64(mag, sr)
    prompt = """Analyze this elephant recording spectrogram (0-1500Hz).
Elephant rumbles: fundamental 10-20Hz, parallel harmonic bands.
Airplane: broadband sweeping. Car/vehicle: broadband low-freq steady rumble.
Generator: tonal fixed-freq hum. Respond ONLY valid JSON, no markdown:
{"noise_type":"airplane","noise_description":"...","recommended_approach":"...","confidence":0.8}
noise_type must be: airplane, car, or generator."""

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{Config.GEMINI_MODEL}:generateContent?key={Config.GEMINI_API_KEY}"
    payload = {
        "contents": [{"parts": [
            {"text": prompt},
            {"inline_data": {"mime_type": "image/png", "data": img}}
        ]}],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 256}
    }

    for attempt in range(3):
        try:
            r = requests.post(url, json=payload, timeout=30).json()
            if "error" in r:
                err = r["error"].get("message", "")
                print(f"⚠️  Gemini: {err[:100]}")
                if "RESOURCE_EXHAUSTED" in err or "quota" in err.lower():
                    return {"noise_type": "unknown", "confidence": 0.0}
                if "429" in str(r["error"].get("code", "")):
                    time.sleep(5 * (attempt + 1)); continue
                return {"noise_type": "unknown", "confidence": 0.0}
            if "candidates" not in r:
                print("⚠️  Gemini: no candidates")
                return {"noise_type": "unknown", "confidence": 0.0}
            c = r["candidates"][0]
            if "content" not in c:
                print(f"⚠️  Gemini: no content, reason={c.get('finishReason')}")
                return {"noise_type": "unknown", "confidence": 0.0}
            t = c["content"]["parts"][0]["text"].strip().strip('`')
            if t.startswith('json'): t = t[4:].strip()
            a = json.loads(t)
            print(f"🤖 Gemini: {a.get('noise_type')} | {a.get('noise_description', '')[:50]}")
            return a
        except json.JSONDecodeError:
            print(f"⚠️  Gemini: bad JSON")
            return {"noise_type": "unknown", "confidence": 0.0}
        except Exception as e:
            print(f"⚠️  Gemini: {e}")
            return {"noise_type": "unknown", "confidence": 0.0}
    return {"noise_type": "unknown", "confidence": 0.0}


# ═══════════════════════════════════════════════════════════════
# CORE DENOISING — AGGRESSIVE ELEPHANT ISOLATION
# ═══════════════════════════════════════════════════════════════

def get_noise_clip(y, sr, call_start=None, call_end=None):
    """Extract noise-only audio segment for noisereduce reference."""
    if call_start is not None and call_start > 1.0:
        # Use audio BEFORE the call (leave 0.5s margin)
        end_sample = t2s(call_start - 0.5, sr)
        if end_sample > sr:  # at least 1 second
            return y[:end_sample]
    if call_end is not None:
        # Use audio AFTER the call
        start_sample = t2s(call_end + 0.5, sr)
        if start_sample < len(y) - sr:
            return y[start_sample:]
    # Fallback: use first 15% and last 15%
    n = max(sr, len(y) // 7)
    return np.concatenate([y[:n], y[-n:]])


def detect_fundamental(mag, sr, call_start=None, call_end=None):
    """Find elephant fundamental frequency."""
    freq = np.linspace(0, sr/2, mag.shape[0])
    sf = t2f(call_start, sr) if call_start else 0
    ef = t2f(call_end, sr) if call_end else mag.shape[1]
    avg = np.mean(mag[:, sf:ef], axis=1)

    lo = np.searchsorted(freq, Config.FUNDAMENTAL_LOW)
    hi = np.searchsorted(freq, Config.FUNDAMENTAL_HIGH)
    region = avg[lo:hi]
    if len(region) == 0:
        return 14.0
    f0 = freq[lo + np.argmax(region)]

    # Validate with harmonics
    best_f0, best_score = f0, 0
    for candidate_f0 in [f0] + [10, 12, 14, 16, 18, 20]:
        score = 0
        for h in range(2, 10):
            hbin = np.searchsorted(freq, candidate_f0 * h)
            if hbin < len(avg) - 2:
                nb = avg[max(0,hbin-2):hbin+3]
                if np.max(nb) > np.median(avg) * 1.5:
                    score += 1
        if score > best_score:
            best_score = score
            best_f0 = candidate_f0

    print(f"🎵 Fundamental: {best_f0:.1f} Hz (harmonic score: {best_score})")
    return best_f0


def build_tight_harmonic_mask(mag, sr, f0, call_start=None, call_end=None,
                               tol_hz=2.5):
    """
    TIGHT harmonic mask — narrow bands ONLY at exact harmonic frequencies.
    Everything between harmonics is ZEROED. This is the key to removing
    mechanical noise while keeping elephant calls.
    """
    freq = np.linspace(0, sr/2, mag.shape[0])
    freq_res = freq[1] - freq[0]
    nf = mag.shape[1]
    mask = np.zeros_like(mag)

    sf = t2f(call_start, sr) if call_start else 0
    ef = min(t2f(call_end, sr) if call_end else nf, nf)

    nh = 0
    for h in range(1, Config.MAX_HARMONIC_ORDER + 1):
        hf = f0 * h
        if hf > Config.MAX_HARMONIC_FREQ:
            break
        nh = h
        hbin = np.searchsorted(freq, hf)
        if hbin >= mag.shape[0]:
            break

        # TIGHT tolerance — only ±tol_hz around exact harmonic
        tol_bins = max(1, int(tol_hz / freq_res))

        for b in range(max(0, hbin - tol_bins), min(mag.shape[0], hbin + tol_bins + 1)):
            dist = abs(freq[b] - hf)
            # Sharp Gaussian — drops off fast
            val = np.exp(-0.5 * (dist / (tol_hz * 0.8))**2)
            mask[b, sf:ef] = np.maximum(mask[b, sf:ef], val)

    # Fade in/out at call edges (0.2s)
    fade = max(2, int(0.2 * sr / Config.HOP_LENGTH))
    for i in range(min(fade, sf)):
        mask[:, sf - fade + i] = mask[:, sf] * (i / fade)
    for i in range(min(fade, nf - ef)):
        mask[:, ef + i] = mask[:, ef-1] * (1 - i / fade)

    # Very light smoothing — keep it tight
    mask = gaussian_filter(mask, sigma=(0.5, 1))
    mask = np.clip(mask, 0, 1)

    pct = (mask > 0.1).mean() * 100
    print(f"🎵 Tight harmonic mask: {nh} harmonics of {f0:.1f}Hz, {pct:.1f}% active")
    return mask


def denoise(y, mag, phase, sr, noise_type="unknown", call_start=None, call_end=None):
    """
    AGGRESSIVE denoising pipeline:
    1. noisereduce library (time-domain spectral gating) — removes bulk noise
    2. Power spectral subtraction — removes residual noise
    3. TIGHT harmonic mask — isolates only elephant harmonic bands
    4. Hard frequency cutoff above 800Hz
    """
    cs = f"{call_start:.1f}-{call_end:.1f}s" if call_start else "unknown"
    print(f"\n🔧 AGGRESSIVE denoising | noise={noise_type} | call={cs}")

    # ── STAGE 1: noisereduce (time-domain) — MULTIPLE PASSES ──
    if HAS_NR:
        noise_clip = get_noise_clip(y, sr, call_start, call_end)
        print(f"   Stage 1: noisereduce (noise ref: {len(noise_clip)/sr:.1f}s)")

        # Pass 1: Stationary noise removal (100% removal)
        y_nr = nr.reduce_noise(
            y=y, sr=sr,
            y_noise=noise_clip,
            prop_decrease=1.0,           # Remove 100% of detected noise
            n_fft=Config.NFFT,
            hop_length=Config.HOP_LENGTH,
            n_std_thresh_stationary=0.1, # Extremely aggressive - gates almost everything
            stationary=True
        )
        print("   Pass 1: Stationary removal (100%) done")

        # Pass 2: Non-stationary noise removal (catches time-varying noise)
        y_nr = nr.reduce_noise(
            y=y_nr, sr=sr,
            y_noise=noise_clip,
            prop_decrease=1.0,
            n_fft=Config.NFFT,
            hop_length=Config.HOP_LENGTH,
            stationary=False
        )
        print("   Pass 2: Non-stationary removal done")

        # Pass 3: Extra pass for airplane (sweeping broadband noise is hardest)
        if noise_type == "airplane":
            y_nr = nr.reduce_noise(
                y=y_nr, sr=sr,
                y_noise=noise_clip,
                prop_decrease=1.0,
                n_fft=2048,              # Smaller FFT for better time resolution
                hop_length=512,
                stationary=False
            )
            print("   Pass 3: Extra airplane pass (small FFT) done")

        # Pass 4: Apply bandpass filter 8-800Hz (elephant range only)
        from scipy.signal import butter, sosfilt
        sos = butter(6, [8, Config.MAX_HARMONIC_FREQ], btype='bandpass', fs=sr, output='sos')
        y_nr = sosfilt(sos, y_nr)
        print(f"   Bandpass filter: 8-{Config.MAX_HARMONIC_FREQ}Hz applied")

        # Recompute STFT from noise-reduced audio
        mag_nr, phase_nr, _ = compute_stft(y_nr, sr)
        print(f"   noisereduce energy reduction: "
              f"{10*np.log10(np.mean(mag**2)/(np.mean(mag_nr**2)+1e-10)):.1f} dB")
    else:
        print("   Stage 1: SKIP noisereduce (not installed)")
        mag_nr = mag.copy()
        phase_nr = phase.copy()

    # ── STAGE 2: Power spectral subtraction ──
    # Estimate remaining noise from non-call regions of noise-reduced signal
    nf = mag_nr.shape[1]
    if call_start is not None and call_end is not None:
        sf = t2f(call_start, sr)
        ef = t2f(call_end, sr)
        margin = max(5, int(0.5 * sr / Config.HOP_LENGTH))
        parts = []
        if sf - margin > 10: parts.append(mag_nr[:, :sf - margin])
        if ef + margin < nf - 10: parts.append(mag_nr[:, ef + margin:])
        if parts:
            noise_spec = np.median(np.concatenate(parts, axis=1), axis=1, keepdims=True)
        else:
            noise_spec = np.median(mag_nr[:, :max(10, nf//10)], axis=1, keepdims=True)
    else:
        n = max(10, nf//7)
        noise_spec = np.median(np.concatenate([mag_nr[:,:n], mag_nr[:,-n:]], axis=1),
                                axis=1, keepdims=True)

    # Aggressive oversubtraction
    oversub = {"generator": 5.0, "airplane": 6.0, "car": 4.0}.get(noise_type, 4.0)
    clean_pow = mag_nr**2 - oversub * noise_spec**2
    cleaned = np.sqrt(np.maximum(clean_pow, 1e-7))
    print(f"   Stage 2: Power subtraction (oversub={oversub}x)")

    # ── STAGE 3: Wiener filter ──
    snr = np.maximum(cleaned**2 / (noise_spec**2 + 1e-10) - 1, 0)
    snr = uniform_filter(snr, size=(1, 5))
    gain = np.clip(snr / (snr + 1), 0, 1)
    cleaned = cleaned * gain
    print("   Stage 3: Wiener filter")

    # ── STAGE 4: TIGHT harmonic mask ──
    f0 = detect_fundamental(mag, sr, call_start, call_end)
    tol = {"generator": 2.0, "airplane": 1.5, "car": 1.5}.get(noise_type, 2.0)
    hmask = build_tight_harmonic_mask(mag_nr, sr, f0, call_start, call_end, tol_hz=tol)
    cleaned = cleaned * hmask
    print("   Stage 4: Tight harmonic mask applied")

    # ── STAGE 5: INTRA-HARMONIC NOISE SUBTRACTION ──
    # Key insight: even within harmonic bands, noise energy exists.
    # Measure the noise level at each harmonic frequency from non-call regions,
    # then subtract that exact amount from the call region.
    freq = np.linspace(0, sr/2, mag.shape[0])
    freq_res = freq[1] - freq[0]
    sf_call = t2f(call_start, sr) if call_start else 0
    ef_call = min(t2f(call_end, sr) if call_end else nf, nf)

    for h in range(1, Config.MAX_HARMONIC_ORDER + 1):
        hf = f0 * h
        if hf > Config.MAX_HARMONIC_FREQ:
            break
        hbin = np.searchsorted(freq, hf)
        tol_bins = max(1, int(tol / freq_res))
        lo_b = max(0, hbin - tol_bins)
        hi_b = min(cleaned.shape[0], hbin + tol_bins + 1)

        # Measure noise energy at this harmonic freq from non-call regions
        noise_at_harmonic_parts = []
        if sf_call > 10:
            noise_at_harmonic_parts.append(mag_nr[lo_b:hi_b, :sf_call])
        if ef_call < nf - 10:
            noise_at_harmonic_parts.append(mag_nr[lo_b:hi_b, ef_call:])
        if noise_at_harmonic_parts:
            noise_at_h = np.median(np.concatenate(noise_at_harmonic_parts, axis=1),
                                    axis=1, keepdims=True)
        else:
            noise_at_h = noise_spec[lo_b:hi_b]

        # Subtract noise energy at this harmonic during call region
        h_region = cleaned[lo_b:hi_b, sf_call:ef_call]
        h_clean = h_region**2 - (oversub * 0.8) * noise_at_h**2
        cleaned[lo_b:hi_b, sf_call:ef_call] = np.sqrt(np.maximum(h_clean, 1e-8))

    print("   Stage 5: Intra-harmonic noise subtraction")

    # ── STAGE 6: Hard frequency cutoff ──
    cutbin = np.searchsorted(freq, Config.MAX_HARMONIC_FREQ)
    cleaned[cutbin:, :] = 0
    print(f"   Stage 6: Hard cutoff above {Config.MAX_HARMONIC_FREQ}Hz")

    # ── STAGE 7: Zero outside call region ──
    if call_start is not None:
        sf = t2f(call_start, sr)
        ef = min(t2f(call_end, sr) if call_end else nf, nf)
        # Keep small margin
        margin = max(2, int(0.3 * sr / Config.HOP_LENGTH))
        cleaned[:, :max(0, sf - margin)] = 0
        cleaned[:, min(nf, ef + margin):] = 0
        print(f"   Stage 7: Zeroed outside call region")

    cleaned = np.maximum(cleaned, 0)

    red = 10 * np.log10(np.mean(mag**2) / (np.mean(cleaned**2) + 1e-10))
    print(f"   ✅ Total energy reduction: {red:.1f} dB")
    return cleaned, phase_nr


# ═══════════════════════════════════════════════════════════════
# ELEVENLABS NARRATION
# ═══════════════════════════════════════════════════════════════

def narrate(analysis, noise_type, inp, out):
    if not Config.ELEVENLABS_API_KEY:
        print("⚠️  ELEVENLABS_API_KEY not set."); return None
    text = (f"Analysis complete for {os.path.basename(inp)}. "
            f"Noise type detected: {noise_type}. "
            f"{analysis.get('noise_description','')} "
            f"The elephant rumble has been successfully isolated using "
            f"spectral gating, power subtraction, and harmonic masking.")
    try:
        r = requests.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{Config.ELEVENLABS_VOICE_ID}",
            json={"text": text, "model_id": "eleven_multilingual_v2",
                  "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}},
            headers={"Accept": "audio/mpeg", "Content-Type": "application/json",
                     "xi-api-key": Config.ELEVENLABS_API_KEY}, timeout=30)
        r.raise_for_status()
        p = out.replace('.wav', '_narration.mp3')
        with open(p, 'wb') as f: f.write(r.content)
        print(f"🔊 Narration: {p}"); return p
    except Exception as e:
        print(f"⚠️  ElevenLabs: {e}"); return None


# ═══════════════════════════════════════════════════════════════
# FULL PIPELINE
# ═══════════════════════════════════════════════════════════════

def process_recording(input_path, output_path=None, call_start=None,
                      call_end=None, noise_type_override=None):
    if output_path is None:
        output_path = str(Path(input_path).parent / f"{Path(input_path).stem}_cleaned.wav")

    print("=" * 60)
    print(f"🐘 {input_path}")
    print("=" * 60)

    # Load
    y, sr = load_audio(input_path)
    Config.SAMPLE_RATE = sr

    # STFT
    print(f"📊 STFT (n_fft={Config.NFFT}, freq_res={sr/Config.NFFT:.1f} Hz/bin)...")
    mag, phase, S = compute_stft(y, sr)
    print(f"   Shape: {mag.shape}")

    # Original spectrogram
    vmin, vmax = plot_spectrogram(mag, sr, "Original",
                                   save_path=output_path.replace('.wav', '_original.png'),
                                   call_start=call_start, call_end=call_end)

    # Classify noise
    if noise_type_override:
        noise_type = noise_type_override
        analysis = {"noise_type": noise_type}
    else:
        fname = os.path.basename(input_path).lower()
        detected = ("car" if "vehicle" in fname or "car" in fname else
                    "airplane" if "airplane" in fname or "plane" in fname else
                    "generator" if "generator" in fname else None)
        analysis = classify_noise_gemini(mag, sr)
        if analysis["noise_type"] == "unknown" and detected:
            noise_type = detected; analysis["noise_type"] = detected
            print(f"🏷️  From filename: {noise_type}")
        else:
            noise_type = analysis["noise_type"]

    # DENOISE (now passes raw audio y for noisereduce)
    cleaned_mag, cleaned_phase = denoise(y, mag, phase, sr, noise_type,
                                          call_start, call_end)

    # Cleaned spectrogram (same dB scale)
    plot_spectrogram(cleaned_mag, sr, "Cleaned — Elephant Only",
                     save_path=output_path.replace('.wav', '_cleaned.png'),
                     vmin_db=vmin, vmax_db=vmax,
                     call_start=call_start, call_end=call_end)

    # Comparison
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(20, 6))
    freq = np.linspace(0, sr/2, mag.shape[0])
    mb = np.searchsorted(freq, 1500)
    tsec = np.linspace(0, mag.shape[1]*Config.HOP_LENGTH/sr, mag.shape[1])
    for ax, m, t in [(a1, mag, "Original"), (a2, cleaned_mag, "Cleaned — Elephant Only")]:
        ax.imshow(20*np.log10(m[:mb]+1e-10), aspect='auto', origin='lower',
                  extent=[tsec[0], tsec[-1], 0, 1500], cmap='inferno', vmin=vmin, vmax=vmax)
        ax.set_title(t); ax.set_xlabel('Time (s)'); ax.set_ylabel('Freq (Hz)')
        if call_start: ax.axvline(x=call_start, color='lime', lw=1.5, alpha=.7)
        if call_end: ax.axvline(x=call_end, color='lime', lw=1.5, alpha=.7)
    plt.suptitle(f'{os.path.basename(input_path)} — noise: {noise_type}', fontweight='bold')
    plt.tight_layout()
    fig.savefig(output_path.replace('.wav', '_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"📊 Comparison saved")

    # Reconstruct & save
    y_clean = reconstruct_audio(cleaned_mag, cleaned_phase, sr)
    save_audio(y_clean, sr, output_path)

    # Narration
    narrate(analysis, noise_type, input_path, output_path)

    print(f"\n✅ DONE | {noise_type} | {output_path}\n")
    return {"input": input_path, "output": output_path,
            "noise_type": noise_type, "analysis": analysis}


# ═══════════════════════════════════════════════════════════════
# BATCH
# ═══════════════════════════════════════════════════════════════

def batch_process(audio_dir, spreadsheet_path=None, output_dir=None):
    import glob
    if output_dir is None:
        output_dir = os.path.join(audio_dir, "cleaned")
    os.makedirs(output_dir, exist_ok=True)

    call_times = {}
    if spreadsheet_path and os.path.exists(spreadsheet_path):
        try:
            import pandas as pd
            df = pd.read_csv(spreadsheet_path)
            for fname, g in df.groupby("Sound_file"):
                fname = str(fname).strip()
                call_times[fname] = {
                    "start": float(g["Start_time"].min()),
                    "end": float(g["End_time"].max()),
                    "num_calls": len(g),
                }
            print(f"📋 {len(call_times)} files, {len(df)} calls loaded")
        except Exception as e:
            print(f"⚠️  Spreadsheet: {e}")

    wavs = sorted(glob.glob(os.path.join(audio_dir, "*.wav")))
    if not wavs: wavs = sorted(glob.glob(os.path.join(audio_dir, "*.WAV")))
    print(f"🔍 {len(wavs)} recordings\n")

    results = []
    for i, fp in enumerate(wavs):
        fn = os.path.basename(fp)
        op = os.path.join(output_dir, fn.replace('.wav', '_cleaned.wav').replace('.WAV', '_cleaned.wav'))
        t = call_times.get(fn, {})
        if t:
            print(f"[{i+1}/{len(wavs)}] {fn}: {t['num_calls']} calls, {t['start']:.1f}-{t['end']:.1f}s")
        try:
            results.append(process_recording(fp, op, t.get("start"), t.get("end")))
        except Exception as e:
            print(f"❌ {fn}: {e}")
            import traceback; traceback.print_exc()
            results.append({"input": fp, "error": str(e)})
        time.sleep(1)

    ok = sum(1 for r in results if "error" not in r)
    print(f"\n{'='*60}\n📊 Batch: {ok}/{len(results)} done\n{'='*60}")
    return results


def main():
    p = argparse.ArgumentParser(description="🐘 ElephantVoices Denoiser")
    p.add_argument("--input", "-i", required=True)
    p.add_argument("--output", "-o", default=None)
    p.add_argument("--noise-type", choices=["airplane", "car", "generator"])
    p.add_argument("--call-start", type=float)
    p.add_argument("--call-end", type=float)
    p.add_argument("--spreadsheet", default=None)
    p.add_argument("--batch", action="store_true")
    a = p.parse_args()
    if a.batch or os.path.isdir(a.input):
        batch_process(a.input, a.spreadsheet, a.output)
    else:
        process_recording(a.input, a.output, a.call_start, a.call_end, a.noise_type)

if __name__ == "__main__":
    main()
    '''
"""
🐘 ElephantVoices Rumble Denoiser — HackSMU VII
Isolates elephant rumble vocalizations from mechanical noise.
"""
import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as signal
from scipy.ndimage import gaussian_filter, uniform_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, json, argparse, base64, time
from io import BytesIO
from pathlib import Path

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    print("⚠️  pip install librosa")

try:
    import noisereduce as nr
    HAS_NR = True
except ImportError:
    HAS_NR = False
    print("⚠️  pip install noisereduce  (strongly recommended for best results)")

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


class Config:
    SAMPLE_RATE = 48000
    NFFT = 8192
    HOP_LENGTH = 2048
    WIN_LENGTH = 8192
    FUNDAMENTAL_LOW = 8
    FUNDAMENTAL_HIGH = 25
    MAX_HARMONIC_FREQ = 800
    MAX_HARMONIC_ORDER = 60
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
    GEMINI_MODEL = "gemini-2.0-flash"
    ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY", "")
    ELEVENLABS_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"


def load_audio(filepath):
    if HAS_LIBROSA:
        y, sr = librosa.load(filepath, sr=None, mono=True)
    else:
        sr, y = wav.read(filepath)
        if y.ndim > 1: y = y.mean(axis=1)
        y = y.astype(np.float32)
        if np.max(np.abs(y)) > 1.0: y = y / np.max(np.abs(y))
    print(f"✅ Loaded: {filepath} | SR={sr} Hz | {len(y)/sr:.1f}s")
    return y, sr

def compute_stft(y, sr):
    if HAS_LIBROSA:
        S = librosa.stft(y, n_fft=Config.NFFT, hop_length=Config.HOP_LENGTH,
                         win_length=Config.WIN_LENGTH)
    else:
        _, _, S = signal.stft(y, fs=sr, nperseg=Config.WIN_LENGTH,
                               noverlap=Config.WIN_LENGTH - Config.HOP_LENGTH,
                               nfft=Config.NFFT)
    return np.abs(S), np.angle(S), S

def reconstruct_audio(mag, phase, sr):
    S = mag * np.exp(1j * phase)
    if HAS_LIBROSA:
        y = librosa.istft(S, hop_length=Config.HOP_LENGTH, win_length=Config.WIN_LENGTH)
    else:
        _, y = signal.istft(S, fs=sr, nperseg=Config.WIN_LENGTH,
                             noverlap=Config.WIN_LENGTH - Config.HOP_LENGTH,
                             nfft=Config.NFFT)
    return y / (np.max(np.abs(y)) + 1e-10)

def amplify_audio(y, gain_db=20):
    gain = 10 ** (gain_db / 20)
    y_loud = np.tanh(y * gain)
    y_loud = y_loud * 0.95 / (np.max(np.abs(y_loud)) + 1e-10)
    print(f"🔊 Amplified by {gain_db} dB")
    return y_loud

def save_audio(y, sr, fp):
    wav.write(fp, sr, np.int16(y * 32767))
    print(f"💾 Saved: {fp}")

def t2f(t, sr):
    return int(t * sr / Config.HOP_LENGTH)

def t2s(t, sr):
    return int(t * sr)


def plot_spectrogram(mag, sr, title="", max_freq=1500, save_path=None,
                     vmin_db=None, vmax_db=None, call_start=None, call_end=None):
    fig, ax = plt.subplots(figsize=(14, 6))
    freq = np.linspace(0, sr/2, mag.shape[0])
    mb = np.searchsorted(freq, max_freq)
    nf = mag.shape[1]
    tsec = np.linspace(0, nf * Config.HOP_LENGTH / sr, nf)
    db = 20 * np.log10(mag[:mb] + 1e-10)
    if vmin_db is None: vmin_db = np.percentile(db, 5)
    if vmax_db is None: vmax_db = np.percentile(db, 99)
    ax.imshow(db, aspect='auto', origin='lower',
              extent=[tsec[0], tsec[-1], 0, max_freq],
              cmap='inferno', vmin=vmin_db, vmax=vmax_db)
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Freq (Hz)'); ax.set_title(title)
    if call_start is not None:
        ax.axvline(x=call_start, color='lime', lw=1.5, alpha=.7)
    if call_end is not None:
        ax.axvline(x=call_end, color='lime', lw=1.5, alpha=.7)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Saved: {save_path}")
    plt.close(fig)
    return vmin_db, vmax_db


def spectrogram_to_b64(mag, sr):
    fig = plt.figure(figsize=(8, 4))
    freq = np.linspace(0, sr/2, mag.shape[0])
    mb = np.searchsorted(freq, 1500)
    db = 20*np.log10(mag[:mb]+1e-10)
    plt.imshow(db, aspect='auto', origin='lower', extent=[0,mag.shape[1],0,1500], cmap='inferno')
    plt.xlabel('Frame'); plt.ylabel('Hz'); plt.colorbar(label='dB'); plt.tight_layout()
    buf = BytesIO(); fig.savefig(buf, format='png', dpi=72); plt.close(fig); buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def classify_noise_gemini(mag, sr):
    if not Config.GEMINI_API_KEY:
        print("⚠️  GEMINI_API_KEY not set.")
        return {"noise_type": "unknown", "confidence": 0.0}

    img = spectrogram_to_b64(mag, sr)
    prompt = """Analyze this elephant recording spectrogram (0-1500Hz).
Elephant rumbles: fundamental 10-20Hz, parallel harmonic bands.
Airplane: broadband sweeping. Car/vehicle: broadband low-freq steady rumble.
Generator: tonal fixed-freq hum. Respond ONLY valid JSON, no markdown:
{"noise_type":"airplane","noise_description":"...","recommended_approach":"...","confidence":0.8}
noise_type must be: airplane, car, or generator."""

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{Config.GEMINI_MODEL}:generateContent?key={Config.GEMINI_API_KEY}"
    payload = {
        "contents": [{"parts": [
            {"text": prompt},
            {"inline_data": {"mime_type": "image/png", "data": img}}
        ]}],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 256}
    }

    for attempt in range(3):
        try:
            r = requests.post(url, json=payload, timeout=30).json()
            if "error" in r:
                err = r["error"].get("message", "")
                print(f"⚠️  Gemini: {err[:100]}")
                if "RESOURCE_EXHAUSTED" in err or "quota" in err.lower():
                    return {"noise_type": "unknown", "confidence": 0.0}
                if "429" in str(r["error"].get("code", "")):
                    time.sleep(5 * (attempt + 1)); continue
                return {"noise_type": "unknown", "confidence": 0.0}
            if "candidates" not in r:
                return {"noise_type": "unknown", "confidence": 0.0}
            c = r["candidates"][0]
            if "content" not in c:
                return {"noise_type": "unknown", "confidence": 0.0}
            t = c["content"]["parts"][0]["text"].strip().strip('`')
            if t.startswith('json'): t = t[4:].strip()
            a = json.loads(t)
            print(f"🤖 Gemini: {a.get('noise_type')} | {a.get('noise_description', '')[:50]}")
            return a
        except json.JSONDecodeError:
            return {"noise_type": "unknown", "confidence": 0.0}
        except Exception as e:
            print(f"⚠️  Gemini: {e}")
            return {"noise_type": "unknown", "confidence": 0.0}
    return {"noise_type": "unknown", "confidence": 0.0}


def get_noise_clip(y, sr, call_start=None, call_end=None):
    if call_start is not None and call_start > 1.0:
        end_sample = t2s(call_start - 0.5, sr)
        if end_sample > sr:
            return y[:end_sample]
    if call_end is not None:
        start_sample = t2s(call_end + 0.5, sr)
        if start_sample < len(y) - sr:
            return y[start_sample:]
    n = max(sr, len(y) // 7)
    return np.concatenate([y[:n], y[-n:]])


def detect_fundamental(mag, sr, call_start=None, call_end=None):
    freq = np.linspace(0, sr/2, mag.shape[0])
    sf = t2f(call_start, sr) if call_start else 0
    ef = t2f(call_end, sr) if call_end else mag.shape[1]
    avg = np.mean(mag[:, sf:ef], axis=1)

    lo = np.searchsorted(freq, Config.FUNDAMENTAL_LOW)
    hi = np.searchsorted(freq, Config.FUNDAMENTAL_HIGH)
    region = avg[lo:hi]
    if len(region) == 0:
        return 14.0
    f0 = freq[lo + np.argmax(region)]

    best_f0, best_score = f0, 0
    for candidate_f0 in [f0] + [10, 12, 14, 16, 18, 20]:
        score = 0
        for h in range(2, 10):
            hbin = np.searchsorted(freq, candidate_f0 * h)
            if hbin < len(avg) - 2:
                nb = avg[max(0,hbin-2):hbin+3]
                if np.max(nb) > np.median(avg) * 1.5:
                    score += 1
        if score > best_score:
            best_score = score
            best_f0 = candidate_f0

    print(f"🎵 Fundamental: {best_f0:.1f} Hz (harmonic score: {best_score})")
    return best_f0


def build_tight_harmonic_mask(mag, sr, f0, call_start=None, call_end=None, tol_hz=2.5):
    freq = np.linspace(0, sr/2, mag.shape[0])
    freq_res = freq[1] - freq[0]
    nf = mag.shape[1]
    mask = np.zeros_like(mag)

    sf = t2f(call_start, sr) if call_start else 0
    ef = min(t2f(call_end, sr) if call_end else nf, nf)

    nh = 0
    for h in range(1, Config.MAX_HARMONIC_ORDER + 1):
        hf = f0 * h
        if hf > Config.MAX_HARMONIC_FREQ:
            break
        nh = h
        hbin = np.searchsorted(freq, hf)
        if hbin >= mag.shape[0]:
            break
        tol_bins = max(1, int(tol_hz / freq_res))
        for b in range(max(0, hbin - tol_bins), min(mag.shape[0], hbin + tol_bins + 1)):
            dist = abs(freq[b] - hf)
            val = np.exp(-0.5 * (dist / (tol_hz * 0.8))**2)
            mask[b, sf:ef] = np.maximum(mask[b, sf:ef], val)

    fade = max(2, int(0.2 * sr / Config.HOP_LENGTH))
    for i in range(min(fade, sf)):
        mask[:, sf - fade + i] = mask[:, sf] * (i / fade)
    for i in range(min(fade, nf - ef)):
        mask[:, ef + i] = mask[:, ef-1] * (1 - i / fade)

    mask = gaussian_filter(mask, sigma=(0.5, 1))
    mask = np.clip(mask, 0, 1)
    pct = (mask > 0.1).mean() * 100
    print(f"🎵 Tight harmonic mask: {nh} harmonics of {f0:.1f}Hz, {pct:.1f}% active")
    return mask


def denoise(y, mag, phase, sr, noise_type="unknown", call_start=None, call_end=None):
    cs = f"{call_start:.1f}-{call_end:.1f}s" if call_start else "unknown"
    print(f"\n🔧 AGGRESSIVE denoising | noise={noise_type} | call={cs}")

    if HAS_NR:
        noise_clip = get_noise_clip(y, sr, call_start, call_end)
        print(f"   Stage 1: noisereduce (noise ref: {len(noise_clip)/sr:.1f}s)")
        y_nr = nr.reduce_noise(y=y, sr=sr, y_noise=noise_clip, prop_decrease=1.0,
                                n_fft=Config.NFFT, hop_length=Config.HOP_LENGTH,
                                n_std_thresh_stationary=0.1, stationary=True)
        print("   Pass 1 done")
        y_nr = nr.reduce_noise(y=y_nr, sr=sr, y_noise=noise_clip, prop_decrease=1.0,
                                n_fft=Config.NFFT, hop_length=Config.HOP_LENGTH, stationary=False)
        print("   Pass 2 done")
        if noise_type == "airplane":
            y_nr = nr.reduce_noise(y=y_nr, sr=sr, y_noise=noise_clip, prop_decrease=1.0,
                                    n_fft=2048, hop_length=512, stationary=False)
            print("   Pass 3 (airplane) done")
        from scipy.signal import butter, sosfilt
        sos = butter(6, [8, Config.MAX_HARMONIC_FREQ], btype='bandpass', fs=sr, output='sos')
        y_nr = sosfilt(sos, y_nr)
        print(f"   Bandpass 8-{Config.MAX_HARMONIC_FREQ}Hz")
        mag_nr, phase_nr, _ = compute_stft(y_nr, sr)
        print(f"   NR energy reduction: {10*np.log10(np.mean(mag**2)/(np.mean(mag_nr**2)+1e-10)):.1f} dB")
    else:
        mag_nr, phase_nr = mag.copy(), phase.copy()

    nf = mag_nr.shape[1]
    if call_start is not None and call_end is not None:
        sf = t2f(call_start, sr); ef = t2f(call_end, sr)
        margin = max(5, int(0.5 * sr / Config.HOP_LENGTH))
        parts = []
        if sf - margin > 10: parts.append(mag_nr[:, :sf - margin])
        if ef + margin < nf - 10: parts.append(mag_nr[:, ef + margin:])
        noise_spec = np.median(np.concatenate(parts, axis=1), axis=1, keepdims=True) if parts else np.median(mag_nr[:, :max(10, nf//10)], axis=1, keepdims=True)
    else:
        n = max(10, nf//7)
        noise_spec = np.median(np.concatenate([mag_nr[:,:n], mag_nr[:,-n:]], axis=1), axis=1, keepdims=True)

    oversub = {"generator": 5.0, "airplane": 6.0, "car": 4.0}.get(noise_type, 4.0)
    cleaned = np.sqrt(np.maximum(mag_nr**2 - oversub * noise_spec**2, 1e-7))
    print(f"   Stage 2: Power subtraction (oversub={oversub}x)")

    snr = np.maximum(cleaned**2 / (noise_spec**2 + 1e-10) - 1, 0)
    snr = uniform_filter(snr, size=(1, 5))
    cleaned = cleaned * np.clip(snr / (snr + 1), 0, 1)
    print("   Stage 3: Wiener filter")

    f0 = detect_fundamental(mag, sr, call_start, call_end)
    tol = {"generator": 2.0, "airplane": 1.5, "car": 1.5}.get(noise_type, 2.0)
    hmask = build_tight_harmonic_mask(mag_nr, sr, f0, call_start, call_end, tol_hz=tol)
    cleaned = cleaned * hmask
    print("   Stage 4: Harmonic mask")

    freq = np.linspace(0, sr/2, mag.shape[0])
    freq_res = freq[1] - freq[0]
    sf_call = t2f(call_start, sr) if call_start else 0
    ef_call = min(t2f(call_end, sr) if call_end else nf, nf)
    for h in range(1, Config.MAX_HARMONIC_ORDER + 1):
        hf = f0 * h
        if hf > Config.MAX_HARMONIC_FREQ: break
        hbin = np.searchsorted(freq, hf)
        tol_bins = max(1, int(tol / freq_res))
        lo_b, hi_b = max(0, hbin - tol_bins), min(cleaned.shape[0], hbin + tol_bins + 1)
        nparts = []
        if sf_call > 10: nparts.append(mag_nr[lo_b:hi_b, :sf_call])
        if ef_call < nf - 10: nparts.append(mag_nr[lo_b:hi_b, ef_call:])
        noise_at_h = np.median(np.concatenate(nparts, axis=1), axis=1, keepdims=True) if nparts else noise_spec[lo_b:hi_b]
        h_region = cleaned[lo_b:hi_b, sf_call:ef_call]
        cleaned[lo_b:hi_b, sf_call:ef_call] = np.sqrt(np.maximum(h_region**2 - (oversub*0.8)*noise_at_h**2, 1e-8))
    print("   Stage 5: Intra-harmonic subtraction")

    cutbin = np.searchsorted(freq, Config.MAX_HARMONIC_FREQ)
    cleaned[cutbin:, :] = 0
    print(f"   Stage 6: Cutoff above {Config.MAX_HARMONIC_FREQ}Hz")

    if call_start is not None:
        sf = t2f(call_start, sr)
        ef = min(t2f(call_end, sr) if call_end else nf, nf)
        margin = max(2, int(0.3 * sr / Config.HOP_LENGTH))
        cleaned[:, :max(0, sf - margin)] = 0
        cleaned[:, min(nf, ef + margin):] = 0
        print("   Stage 7: Zeroed outside call")

    cleaned = np.maximum(cleaned, 0)
    red = 10 * np.log10(np.mean(mag**2) / (np.mean(cleaned**2) + 1e-10))
    print(f"   ✅ Total reduction: {red:.1f} dB")
    return cleaned, phase_nr


def narrate(analysis, noise_type, inp, out):
    if not Config.ELEVENLABS_API_KEY:
        print("⚠️  ELEVENLABS_API_KEY not set."); return None
    text = (f"Analysis complete for {os.path.basename(inp)}. "
            f"Noise type: {noise_type}. "
            f"{analysis.get('noise_description','')} "
            f"Elephant rumble isolated via spectral gating, power subtraction, and harmonic masking.")
    try:
        r = requests.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{Config.ELEVENLABS_VOICE_ID}",
            json={"text": text, "model_id": "eleven_multilingual_v2",
                  "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}},
            headers={"Accept": "audio/mpeg", "Content-Type": "application/json",
                     "xi-api-key": Config.ELEVENLABS_API_KEY}, timeout=30)
        r.raise_for_status()
        p = out.replace('.wav', '_narration.mp3')
        with open(p, 'wb') as f: f.write(r.content)
        print(f"🔊 Narration: {p}"); return p
    except Exception as e:
        print(f"⚠️  ElevenLabs: {e}"); return None


def process_recording(input_path, output_path=None, call_start=None,
                      call_end=None, noise_type_override=None):
    if output_path is None:
        output_path = str(Path(input_path).parent / f"{Path(input_path).stem}_cleaned.wav")

    print("=" * 60)
    print(f"🐘 {input_path}")
    print("=" * 60)

    y, sr = load_audio(input_path)
    Config.SAMPLE_RATE = sr
    mag, phase, S = compute_stft(y, sr)
    print(f"   STFT: {mag.shape}")

    spec_base = output_path.replace('.wav', '')
    vmin, vmax = plot_spectrogram(mag, sr, "Original", save_path=f"{spec_base}_original.png",
                                   call_start=call_start, call_end=call_end)

    if noise_type_override:
        noise_type = noise_type_override
        analysis = {"noise_type": noise_type}
    else:
        fname = os.path.basename(input_path).lower()
        detected = ("car" if "vehicle" in fname or "car" in fname else
                    "airplane" if "airplane" in fname or "plane" in fname else
                    "generator" if "generator" in fname else None)
        analysis = classify_noise_gemini(mag, sr)
        if analysis["noise_type"] == "unknown" and detected:
            noise_type = detected; analysis["noise_type"] = detected
            print(f"🏷️  From filename: {noise_type}")
        else:
            noise_type = analysis["noise_type"]

    cleaned_mag, cleaned_phase = denoise(y, mag, phase, sr, noise_type, call_start, call_end)

    plot_spectrogram(cleaned_mag, sr, "Cleaned — Elephant Only",
                     save_path=f"{spec_base}_cleaned.png",
                     vmin_db=vmin, vmax_db=vmax, call_start=call_start, call_end=call_end)

    # Comparison
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(20, 6))
    freq = np.linspace(0, sr/2, mag.shape[0])
    mb = np.searchsorted(freq, 1500)
    tsec = np.linspace(0, mag.shape[1]*Config.HOP_LENGTH/sr, mag.shape[1])
    for ax, m, t in [(a1, mag, "Original"), (a2, cleaned_mag, "Cleaned — Elephant Only")]:
        ax.imshow(20*np.log10(m[:mb]+1e-10), aspect='auto', origin='lower',
                  extent=[tsec[0], tsec[-1], 0, 1500], cmap='inferno', vmin=vmin, vmax=vmax)
        ax.set_title(t); ax.set_xlabel('Time (s)'); ax.set_ylabel('Freq (Hz)')
        if call_start: ax.axvline(x=call_start, color='lime', lw=1.5, alpha=.7)
        if call_end: ax.axvline(x=call_end, color='lime', lw=1.5, alpha=.7)
    plt.suptitle(f'{os.path.basename(input_path)} → {os.path.basename(output_path)} | noise: {noise_type}', fontweight='bold')
    plt.tight_layout()
    fig.savefig(f"{spec_base}_comparison.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

    y_clean = reconstruct_audio(cleaned_mag, cleaned_phase, sr)
    y_clean = amplify_audio(y_clean, gain_db=20)
    save_audio(y_clean, sr, output_path)
    narrate(analysis, noise_type, input_path, output_path)

    print(f"\n✅ DONE | {noise_type} | {output_path}\n")
    return {"input": input_path, "output": output_path, "noise_type": noise_type, "analysis": analysis}


# ═══════════════════════════════════════════════════════════════
# BATCH — Each call → elephant1.wav, elephant2.wav, ...
# ═══════════════════════════════════════════════════════════════

def batch_process(audio_dir, spreadsheet_path=None, output_dir=None):
    import glob

    if output_dir is None:
        output_dir = os.path.join(audio_dir, "cleaned")
    os.makedirs(output_dir, exist_ok=True)

    calls = []
    if spreadsheet_path and os.path.exists(spreadsheet_path):
        try:
            import pandas as pd
            df = pd.read_csv(spreadsheet_path)
            for _, row in df.iterrows():
                calls.append({
                    "sound_file": str(row["Sound_file"]).strip(),
                    "start": float(row["Start_time"]),
                    "end": float(row["End_time"]),
                    "call_type": str(row.get("Call_type", "rumble")).strip(),
                })
            print(f"📋 Loaded {len(calls)} individual elephant calls from spreadsheet")
        except Exception as e:
            print(f"⚠️  Spreadsheet error: {e}")

    if calls:
        # Each row in CSV = one elephant call = one output file
        results = []
        for i, call in enumerate(calls):
            elephant_num = i + 1
            input_path = os.path.join(audio_dir, call["sound_file"])
            if not os.path.exists(input_path):
                print(f"❌ Not found: {input_path}")
                results.append({"input": input_path, "error": "Not found"})
                continue

            output_path = os.path.join(output_dir, f"elephant{elephant_num}.wav")
            print(f"\n[{elephant_num}/{len(calls)}] {call['sound_file']} | "
                  f"{call['start']:.1f}-{call['end']:.1f}s → elephant{elephant_num}.wav")

            try:
                result = process_recording(input_path, output_path,
                                            call_start=call["start"], call_end=call["end"])
                results.append(result)
            except Exception as e:
                print(f"❌ elephant{elephant_num}: {e}")
                import traceback; traceback.print_exc()
                results.append({"input": input_path, "error": str(e)})
            time.sleep(0.5)
    else:
        # No spreadsheet — one file per WAV
        wavs = sorted(glob.glob(os.path.join(audio_dir, "*.wav")) +
                       glob.glob(os.path.join(audio_dir, "*.WAV")))
        print(f"🔍 {len(wavs)} recordings\n")
        results = []
        for i, fp in enumerate(wavs):
            elephant_num = i + 1
            output_path = os.path.join(output_dir, f"elephant{elephant_num}.wav")
            try:
                results.append(process_recording(fp, output_path))
            except Exception as e:
                print(f"❌ elephant{elephant_num}: {e}")
                results.append({"input": fp, "error": str(e)})
            time.sleep(0.5)

    ok = sum(1 for r in results if "error" not in r)
    print(f"\n{'='*60}")
    print(f"📊 Done: {ok}/{len(results)} elephant calls isolated")
    print(f"   Output: {output_dir}/elephant1.wav → elephant{len(results)}.wav")
    print(f"{'='*60}")
    return results


def main():
    p = argparse.ArgumentParser(description="🐘 ElephantVoices Denoiser")
    p.add_argument("--input", "-i", required=True)
    p.add_argument("--output", "-o", default=None)
    p.add_argument("--noise-type", choices=["airplane", "car", "generator"])
    p.add_argument("--call-start", type=float)
    p.add_argument("--call-end", type=float)
    p.add_argument("--spreadsheet", default=None)
    p.add_argument("--batch", action="store_true")
    a = p.parse_args()
    if a.batch or os.path.isdir(a.input):
        batch_process(a.input, a.spreadsheet, a.output)
    else:
        process_recording(a.input, a.output, a.call_start, a.call_end, a.noise_type)

if __name__ == "__main__":
    main()
 '''