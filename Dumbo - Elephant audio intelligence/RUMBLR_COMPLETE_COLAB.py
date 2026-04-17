"""
╔══════════════════════════════════════════════════════════════╗
║         RUMBLR — Complete Elephant Call Pipeline             ║
║         HackSMU VII  |  ElephantVoices Challenge            ║
║                                                              ║
║  Run locally:  python rumblr.py                              ║
║                                                              ║
║  Sponsor Integrations:                                       ║
║    ✅ Gemini API — noise classification + call interpretation║
║    ✅ ElevenLabs — voice narration of results                ║
║    ✅ Auth0 — secured API (api_server.py)                    ║
║    ✅ DigitalOcean — deployment target                       ║
╚══════════════════════════════════════════════════════════════╝

PROJECT STRUCTURE (put your files here):
  files/
  ├── data/
  │   ├── Audio/                ← your 44 .wav files
  │   └── Audio_files.csv       ← the spreadsheet
  ├── elephant voice/           ← OUTPUT (auto-created)
  │   ├── clean_clips/          → elephant1.wav, elephant2.wav, ...
  │   ├── spectrograms/         → spectrogram images
  │   ├── narrations/           → ElevenLabs voice narrations
  │   └── model/                → trained ML model
  └── rumblr.py                 ← THIS FILE
"""

import numpy as np
import pandas as pd
import os
import json
import time
import pickle
import warnings
import argparse
import base64
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import librosa
import librosa.display
import soundfile as sf

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

warnings.filterwarnings('ignore')


# ══════════════════════════════════════════════════════════════
# ██  CONFIGURATION — EDIT THESE                              ██
# ══════════════════════════════════════════════════════════════

AUDIO_DIR    = 'data/Audio'
SPREADSHEET  = 'data/Audio_files.csv'

# Free Gemini key: https://aistudio.google.com/apikey
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyDt68rUvLeTJsWUSB607Zp40Ioi0uF-JxI")

# ElevenLabs key (use MLH promo code): https://elevenlabs.io
ELEVENLABS_API_KEY  = os.environ.get("ELEVENLABS_API_KEY", "sk_b21a9c745a13836602894569479447db6297450d464bb20e")
ELEVENLABS_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"

# Output folders
OUTPUT_ROOT   = 'elephant voice'
CLEAN_DIR     = os.path.join(OUTPUT_ROOT, 'clean_clips')
SPEC_DIR      = os.path.join(OUTPUT_ROOT, 'spectrograms')
MODEL_DIR     = os.path.join(OUTPUT_ROOT, 'model')
NARRATION_DIR = os.path.join(OUTPUT_ROOT, 'narrations')

# Audio parameters
SAMPLE_RATE        = 22050
NOISE_WINDOW_SEC   = 2.0
NOISE_FLOOR        = 0.01
DENOISE_THRESHOLD  = 2.0


# ══════════════════════════════════════════════════════════════
# STEP 1 — NOISE REMOVAL + CLIP EXTRACTION
# ══════════════════════════════════════════════════════════════

def get_noise_profile(y_full, start_sec, sr, window_sec=2.0):
    """Get noise fingerprint from audio BEFORE the elephant call."""
    noise_start = max(0, int((start_sec - window_sec) * sr))
    noise_end   = int(start_sec * sr)
    if noise_end <= noise_start + 512:
        noise_end = min(int(0.5 * sr), len(y_full))
        noise_start = 0
    y_noise = y_full[noise_start:noise_end]
    D_noise = librosa.stft(y_noise, n_fft=2048, hop_length=512)
    return np.mean(np.abs(D_noise), axis=1, keepdims=True)


def wiener_denoise(y_segment, noise_profile):
    """Wiener filter — gentle on low frequencies, great for rumbles."""
    D     = librosa.stft(y_segment, n_fft=2048, hop_length=512)
    mag   = np.abs(D)
    phase = np.angle(D)
    signal_power = mag ** 2
    noise_power  = (noise_profile ** 2) + 1e-10
    snr   = np.maximum(signal_power - noise_power, 0) / noise_power
    gain  = snr / (snr + 1.0)
    D_clean = gain * mag * np.exp(1j * phase)
    return librosa.istft(D_clean, hop_length=512)


def spectral_subtract(y_segment, noise_profile, threshold=2.0, floor=0.01):
    """More aggressive noise removal via spectral subtraction."""
    D     = librosa.stft(y_segment, n_fft=2048, hop_length=512)
    mag   = np.abs(D)
    phase = np.angle(D)
    mag_clean = np.maximum(mag - threshold * noise_profile, floor * mag)
    D_clean = mag_clean * np.exp(1j * phase)
    return librosa.istft(D_clean, hop_length=512)


def extract_and_clean_clip(audio_path, start_time, end_time,
                           out_path, method='wiener'):
    """Extract elephant call from recording and remove background noise."""
    y_full, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    noise_profile = get_noise_profile(y_full, start_time, sr, NOISE_WINDOW_SEC)

    buffer     = int(0.1 * sr)
    call_start = max(0,           int(start_time * sr) - buffer)
    call_end   = min(len(y_full), int(end_time   * sr) + buffer)
    y_segment  = y_full[call_start:call_end]

    if method == 'wiener':
        y_clean = wiener_denoise(y_segment, noise_profile)
    else:
        y_clean = spectral_subtract(y_segment, noise_profile, DENOISE_THRESHOLD)

    if np.max(np.abs(y_clean)) > 0:
        y_clean = y_clean / np.max(np.abs(y_clean)) * 0.9

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sf.write(out_path, y_clean, sr)
    return True


# ══════════════════════════════════════════════════════════════
# STEP 2 — SPECTROGRAM GENERATION
# ══════════════════════════════════════════════════════════════

def make_spectrogram(clean_path, spec_path, call_type, elephant_num):
    """Convert clean audio clip to a spectrogram image."""
    y, sr = librosa.load(clean_path, sr=SAMPLE_RATE)
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=128, fmin=10, fmax=4000,
        n_fft=2048, hop_length=256
    )
    S_db = librosa.power_to_db(S, ref=np.max)

    fig, ax = plt.subplots(figsize=(4, 3))
    librosa.display.specshow(S_db, sr=sr, hop_length=256,
                              x_axis='time', y_axis='mel',
                              fmin=10, fmax=4000,
                              cmap='inferno', ax=ax)
    ax.set_title(f"{call_type}  elephant{elephant_num}", fontsize=9)
    ax.set_xlabel(''); ax.set_ylabel('')
    plt.tight_layout(pad=0.2)
    plt.savefig(spec_path, dpi=80, bbox_inches='tight')
    plt.close()


# ══════════════════════════════════════════════════════════════
# STEP 3 — FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════

def extract_features(audio_path):
    """Turn one audio clip into numeric features for ML."""
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    feats = {}

    feats['duration'] = len(y) / sr

    rms = librosa.feature.rms(y=y)
    feats['rms_mean'] = float(np.mean(rms))
    feats['rms_std']  = float(np.std(rms))

    try:
        f0, voiced, _ = librosa.pyin(y, fmin=12, fmax=1000,
                                      frame_length=4096, hop_length=512)
        voiced_f0 = f0[voiced] if voiced is not None else np.array([])
        feats['f0_mean']     = float(np.nanmean(voiced_f0)) if len(voiced_f0) > 0 else 0.0
        feats['f0_std']      = float(np.nanstd(voiced_f0))  if len(voiced_f0) > 0 else 0.0
        feats['voiced_frac'] = float(np.mean(voiced))       if voiced is not None else 0.0
    except:
        feats['f0_mean'] = feats['f0_std'] = feats['voiced_frac'] = 0.0

    centroid  = librosa.feature.spectral_centroid(y=y, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff   = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)
    zcr       = librosa.feature.zero_crossing_rate(y)

    feats['centroid_mean']  = float(np.mean(centroid))
    feats['centroid_std']   = float(np.std(centroid))
    feats['bandwidth_mean'] = float(np.mean(bandwidth))
    feats['rolloff_mean']   = float(np.mean(rolloff))
    feats['zcr_mean']       = float(np.mean(zcr))

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=2048, hop_length=512)
    for k in range(13):
        feats[f'mfcc_{k+1}_mean'] = float(np.mean(mfcc[k]))
        feats[f'mfcc_{k+1}_std']  = float(np.std(mfcc[k]))

    return feats


# ══════════════════════════════════════════════════════════════
# STEP 4 — GEMINI: INTERPRET WHAT THE ELEPHANT IS SAYING
# ══════════════════════════════════════════════════════════════

ETHOGRAM_CONTEXT = """You are an expert elephant behavioural bioacoustician with knowledge of
the ElephantVoices Elephant Ethogram — the world's most complete catalogue
of elephant behaviour (322 behaviours, 23 call contexts).

CALL TYPE MEANINGS:
- rumble: low frequency (14-35Hz), long duration. Contact, greetings,
  mother-calf communication, group coordination. Travels up to 10km.
- trumpet: high frequency (500Hz+), short burst. Alarm, excitement, aggression.
- bark-rumble: mild alarm followed by social communication.
- roar: broadband, medium duration. Distress, protest, separation anxiety.
- roar-rumble: high arousal + social signal. Separation or confrontation.
- rumble-roar-rumble: complex emotional state, social tension.
- trumpet-rumble: intense excitement + social contact, reunion behaviour.

RESPONSE FORMAT — return ONLY valid JSON, no markdown fences:
{
  "what_elephant_is_saying": "plain English, 1-2 sentences max",
  "emotional_state": "one word or short phrase",
  "social_context": "one sentence about the likely social situation",
  "conservation_alert": true or false,
  "alert_reason": "string or null",
  "confidence_note": "one sentence about what would confirm this"
}"""


def interpret_call_gemini(call_type, features_row):
    """Use free Gemini API to interpret what the elephant is communicating."""
    if not GEMINI_API_KEY or not HAS_REQUESTS:
        return None

    prompt = f"""{ETHOGRAM_CONTEXT}

Call type detected by AI model: {call_type}
Duration: {features_row.get('duration', 0):.2f} seconds
Pitch (F0): {features_row.get('f0_mean', 0):.1f} Hz
Spectral centroid: {features_row.get('centroid_mean', 0):.0f} Hz
Energy (RMS): {features_row.get('rms_mean', 0):.4f}
Zero crossing rate: {features_row.get('zcr_mean', 0):.4f}

Interpret this elephant call. Return ONLY valid JSON."""

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

    try:
        r = requests.post(url, json={
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.3, "maxOutputTokens": 400}
        }, timeout=30)
        r.raise_for_status()
        data = r.json()

        if "error" in data:
            print(f"  ⚠️  Gemini: {data['error'].get('message', '')[:80]}")
            return None
        if "candidates" not in data or not data["candidates"]:
            return None
        c = data["candidates"][0]
        if "content" not in c:
            return None

        text = c["content"]["parts"][0]["text"].strip().strip('`')
        if text.startswith('json'):
            text = text[4:].strip()
        return json.loads(text)
    except json.JSONDecodeError:
        print(f"  ⚠️  Gemini returned invalid JSON")
        return None
    except Exception as e:
        print(f"  ⚠️  Gemini: {e}")
        return None


def mock_interpret(call_type):
    """Fallback when Gemini API is unavailable."""
    templates = {
        'rumble': {
            "what_elephant_is_saying": "A contact rumble — calling to family members, likely saying 'I'm here, where are you?'",
            "emotional_state": "calm, social",
            "social_context": "Long-distance communication between separated family members.",
            "conservation_alert": False, "alert_reason": None,
            "confidence_note": "A response rumble would confirm this is a contact call."
        },
        'trumpet': {
            "what_elephant_is_saying": "An alarm trumpet — warning others of a threat.",
            "emotional_state": "alarmed",
            "social_context": "Warning signal broadcast to the herd.",
            "conservation_alert": True,
            "alert_reason": "Alarm call — possible human-elephant conflict",
            "confidence_note": "Others fleeing would confirm alarm."
        },
        'roar': {
            "what_elephant_is_saying": "A distress roar — pain, fear, or separation from family.",
            "emotional_state": "distressed",
            "social_context": "High-intensity negative emotional state.",
            "conservation_alert": True,
            "alert_reason": "Distress call — possible injury or poaching",
            "confidence_note": "Sustained roaring without response suggests isolation."
        },
    }
    base_key = call_type.split('-')[0]
    return templates.get(base_key, templates['rumble'])


# ══════════════════════════════════════════════════════════════
# STEP 5 — ELEVENLABS VOICE NARRATION
# ══════════════════════════════════════════════════════════════

def generate_narration(elephant_num, call_type, interpretation):
    """Generate ElevenLabs voice narration for a call."""
    if not ELEVENLABS_API_KEY or not HAS_REQUESTS:
        return None

    text = (
        f"Elephant call number {elephant_num}. "
        f"Call type: {call_type}. "
        f"{interpretation.get('what_elephant_is_saying', '')} "
        f"Emotional state: {interpretation.get('emotional_state', 'unknown')}. "
        f"{interpretation.get('social_context', '')}"
    )
    if interpretation.get('conservation_alert'):
        text += f" Conservation alert: {interpretation.get('alert_reason', '')}."

    try:
        r = requests.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}",
            json={
                "text": text,
                "model_id": "eleven_multilingual_v2",
                "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}
            },
            headers={
                "Accept": "audio/mpeg",
                "Content-Type": "application/json",
                "xi-api-key": ELEVENLABS_API_KEY
            },
            timeout=30
        )
        r.raise_for_status()
        narr_path = os.path.join(NARRATION_DIR, f"elephant{elephant_num}_narration.mp3")
        with open(narr_path, 'wb') as f:
            f.write(r.content)
        return narr_path
    except Exception as e:
        print(f"  ⚠️  ElevenLabs elephant{elephant_num}: {e}")
        return None


# ══════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════

def main():
    # ── Create output dirs ──
    for d in [CLEAN_DIR, SPEC_DIR, MODEL_DIR, NARRATION_DIR]:
        os.makedirs(d, exist_ok=True)

    print("═" * 60)
    print("  RUMBLR — Elephant Call Pipeline")
    print("═" * 60)
    print(f"  Audio dir  : {AUDIO_DIR}")
    print(f"  Spreadsheet: {SPREADSHEET}")
    print(f"  Output     : {OUTPUT_ROOT}/")
    print(f"  Gemini API : {'✅ Set' if GEMINI_API_KEY else '⚠️  Not set'}")
    print(f"  ElevenLabs : {'✅ Set' if ELEVENLABS_API_KEY else '⚠️  Not set (optional)'}")
    print("═" * 60)

    # ══════════════════════════════════════════════════════════
    # 1. LOAD SPREADSHEET
    # ══════════════════════════════════════════════════════════
    print("\n📋 Loading spreadsheet...")

    if SPREADSHEET.endswith('.csv'):
        df = pd.read_csv(SPREADSHEET)
    else:
        df = pd.read_excel(SPREADSHEET, engine='openpyxl')

    df['Sound_file'] = df['Sound_file'].astype(str).str.strip()
    df['duration'] = df['End_time'] - df['Start_time']

    print(f"   {len(df)} calls across {df['Sound_file'].nunique()} files")
    print(f"   Call types: {dict(df['Call_type'].value_counts())}")

    # Check which files exist
    available = []
    for fname in df['Sound_file'].unique():
        if os.path.exists(os.path.join(AUDIO_DIR, fname)):
            available.append(fname)
        else:
            print(f"   ⚠️  Missing: {fname}")

    df_available = df[df['Sound_file'].isin(available)].reset_index(drop=True)
    print(f"   Processing {len(df_available)} calls from {len(available)} files")

    # Dataset overview chart
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    df['Call_type'].value_counts().plot(kind='barh', ax=axes[0],
                                        color='#534AB7', title='Call type counts')
    df['duration'].hist(bins=20, ax=axes[1], color='#1D9E75')
    axes[1].set_title('Call durations (seconds)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_ROOT, 'dataset_overview.png'), dpi=100)
    plt.close()

    # ══════════════════════════════════════════════════════════
    # 2. DENOISE + EXTRACT → elephant1.wav, elephant2.wav, ...
    # ══════════════════════════════════════════════════════════
    print(f"\n🔊 Denoising {len(df_available)} calls → {CLEAN_DIR}/elephant1.wav ...")

    manifest = []
    for i, row in df_available.iterrows():
        audio_path   = os.path.join(AUDIO_DIR, row['Sound_file'])
        elephant_num = i + 1
        out_path     = os.path.join(CLEAN_DIR, f"elephant{elephant_num}.wav")

        try:
            extract_and_clean_clip(audio_path, row['Start_time'],
                                   row['End_time'], out_path)
            status = 'ok'
        except Exception as e:
            status = f'error: {e}'

        manifest.append({
            'elephant_num': elephant_num,
            'call_id'     : int(row['Selection']),
            'call_type'   : row['Call_type'],
            'sound_file'  : row['Sound_file'],
            'start_time'  : row['Start_time'],
            'end_time'    : row['End_time'],
            'duration'    : row['duration'],
            'clean_path'  : out_path if status == 'ok' else None,
            'status'      : status
        })

        if (i + 1) % 30 == 0 or (i + 1) == len(df_available):
            ok = sum(1 for m in manifest if m['status'] == 'ok')
            print(f"   [{i+1:>3}/{len(df_available)}]  ✅ {ok} clips")

    manifest_df = pd.DataFrame(manifest)
    manifest_df.to_csv(os.path.join(OUTPUT_ROOT, 'manifest.csv'), index=False)
    ok_clips = manifest_df[manifest_df['status'] == 'ok'].reset_index(drop=True)
    print(f"   ✅ {len(ok_clips)} clean clips saved")

    # ══════════════════════════════════════════════════════════
    # 3. SPECTROGRAMS
    # ══════════════════════════════════════════════════════════
    print(f"\n🖼️  Generating spectrograms...")

    for i, row in ok_clips.iterrows():
        spec_name = f"elephant{row['elephant_num']}_{row['call_type']}.png"
        spec_path = os.path.join(SPEC_DIR, spec_name)
        try:
            make_spectrogram(row['clean_path'], spec_path,
                              row['call_type'], row['elephant_num'])
            ok_clips.at[i, 'spec_path'] = spec_path
        except Exception as e:
            print(f"   ⚠️  elephant{row['elephant_num']}: {e}")

        if (i + 1) % 40 == 0 or (i + 1) == len(ok_clips):
            print(f"   [{i+1:>3}/{len(ok_clips)}] spectrograms")

    ok_clips.to_csv(os.path.join(OUTPUT_ROOT, 'manifest_with_specs.csv'), index=False)
    print(f"   ✅ Spectrograms saved to {SPEC_DIR}/")

    # ══════════════════════════════════════════════════════════
    # 4. FEATURE EXTRACTION
    # ══════════════════════════════════════════════════════════
    print(f"\n📊 Extracting acoustic features...")

    all_features = []
    for i, row in ok_clips.iterrows():
        try:
            f = extract_features(row['clean_path'])
            f['elephant_num'] = row['elephant_num']
            f['call_id']      = row['call_id']
            f['call_type']    = row['call_type']
            all_features.append(f)
        except Exception as e:
            print(f"   ⚠️  elephant{row['elephant_num']}: {e}")

        if (i + 1) % 40 == 0 or (i + 1) == len(ok_clips):
            print(f"   [{i+1:>3}/{len(ok_clips)}] features")

    features_df = pd.DataFrame(all_features)
    features_df.to_csv(os.path.join(OUTPUT_ROOT, 'features.csv'), index=False)
    print(f"   ✅ {features_df.shape[0]} calls × {features_df.shape[1]} features")

    # ══════════════════════════════════════════════════════════
    # 5. TRAIN ML CLASSIFIER
    # ══════════════════════════════════════════════════════════
    print(f"\n🧠 Training Random Forest classifier...")

    feat_df = features_df.copy()
    le = LabelEncoder()
    feat_df['label'] = le.fit_transform(feat_df['call_type'])

    meta_cols = {'elephant_num', 'call_id', 'call_type', 'label'}
    feat_cols = [c for c in feat_df.columns if c not in meta_cols]

    X = feat_df[feat_cols].fillna(0).values
    y_labels = feat_df['label'].values

    print("   Classes:")
    for i, ct in enumerate(le.classes_):
        n = (y_labels == i).sum()
        print(f"     {ct:<25} {n:>3} samples")

    model = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(
            n_estimators=200, max_depth=None,
            class_weight='balanced', random_state=42
        ))
    ])

    skf    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y_labels, cv=skf, scoring='accuracy')
    print(f"\n   5-fold CV accuracy: {scores.mean():.1%} ± {scores.std():.1%}")

    model.fit(X, y_labels)

    # Feature importance
    rf  = model.named_steps['clf']
    imp = pd.Series(rf.feature_importances_, index=feat_cols).sort_values(ascending=False)
    print("\n   Top 10 features:")
    for feat, score in imp.head(10).items():
        bar = "█" * int(score * 100)
        print(f"     {feat:<25} {score:.3f}  {bar}")

    fig, ax = plt.subplots(figsize=(8, 5))
    imp.head(15).plot(kind='barh', ax=ax, color='#534AB7')
    ax.set_title('Feature importance (top 15)', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_ROOT, 'feature_importance.png'), dpi=100)
    plt.close()

    with open(os.path.join(MODEL_DIR, 'rumblr_model.pkl'), 'wb') as f:
        pickle.dump({'model': model, 'label_encoder': le,
                     'feature_cols': feat_cols}, f)
    print(f"   ✅ Model saved")

    # ══════════════════════════════════════════════════════════
    # 6. EVALUATION
    # ══════════════════════════════════════════════════════════
    print(f"\n📊 Evaluating model...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_labels, test_size=0.2, random_state=42, stratify=y_labels
    )
    model_eval = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(n_estimators=200,
                                        class_weight='balanced', random_state=42))
    ])
    model_eval.fit(X_train, y_train)
    y_pred = model_eval.predict(X_test)

    print(classification_report(y_test, y_pred, target_names=le.classes_))

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(len(le.classes_)))
    ax.set_yticks(range(len(le.classes_)))
    ax.set_xticklabels(le.classes_, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(le.classes_, fontsize=9)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title('Confusion Matrix', fontweight='bold')
    for i in range(len(le.classes_)):
        for j in range(len(le.classes_)):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black',
                    fontsize=11, fontweight='bold')
    plt.colorbar(im); plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_ROOT, 'confusion_matrix.png'), dpi=100)
    plt.close()

    # ══════════════════════════════════════════════════════════
    # 7. GEMINI INTERPRETATION
    # ══════════════════════════════════════════════════════════
    print(f"\n🤖 Interpreting calls with Gemini AI...")

    use_gemini = bool(GEMINI_API_KEY) and HAS_REQUESTS
    if use_gemini:
        print("   ✅ Using Gemini 2.0 Flash (free)")
    else:
        print("   ⚠️  No GEMINI_API_KEY — using built-in templates")
        print("   Get free key: https://aistudio.google.com/apikey")

    X_all = features_df[feat_cols].fillna(0).values
    y_pred_all = model.predict(X_all)
    features_df['predicted_type'] = le.inverse_transform(y_pred_all)

    interpretations = []
    for i, row in features_df.iterrows():
        call_type    = row['predicted_type']
        elephant_num = int(row['elephant_num'])

        interp = None
        if use_gemini:
            interp = interpret_call_gemini(call_type, row.to_dict())
            time.sleep(0.5)

        if interp is None:
            interp = mock_interpret(call_type)

        interpretations.append({
            'elephant_num'   : elephant_num,
            'call_id'        : row['call_id'],
            'actual_type'    : row['call_type'],
            'predicted_type' : call_type,
            'correct'        : row['call_type'] == call_type,
            'duration'       : row['duration'],
            'f0_hz'          : row['f0_mean'],
            **interp
        })

        if (i + 1) % 30 == 0 or (i + 1) == len(features_df):
            alerts = sum(1 for r in interpretations if r.get('conservation_alert'))
            print(f"   [{i+1:>3}/{len(features_df)}]  🚨 alerts: {alerts}")

    results_df = pd.DataFrame(interpretations)
    results_df.to_csv(os.path.join(OUTPUT_ROOT, 'final_results.csv'), index=False)

    alerts = results_df[results_df['conservation_alert'] == True]

    # ══════════════════════════════════════════════════════════
    # 8. ELEVENLABS NARRATIONS
    # ══════════════════════════════════════════════════════════
    if ELEVENLABS_API_KEY and HAS_REQUESTS:
        print(f"\n🔊 Generating ElevenLabs voice narrations...")
        narr_count = 0
        for _, row in results_df.iterrows():
            interp = {k: row.get(k) for k in [
                'what_elephant_is_saying', 'emotional_state',
                'social_context', 'conservation_alert', 'alert_reason'
            ]}
            result = generate_narration(int(row['elephant_num']),
                                         row['predicted_type'], interp)
            if result:
                narr_count += 1
            time.sleep(0.3)
            if narr_count % 20 == 0 and narr_count > 0:
                print(f"   {narr_count} narrations...")
        print(f"   ✅ {narr_count} narrations saved to {NARRATION_DIR}/")
    else:
        print(f"\n⚠️  Skipping ElevenLabs (no key set)")

    # ══════════════════════════════════════════════════════════
    # 9. FINAL SUMMARY
    # ══════════════════════════════════════════════════════════
    print()
    print("═" * 60)
    print("  RUMBLR — FINAL RESULTS")
    print("═" * 60)
    print(f"  Calls processed    : {len(results_df)}")
    print(f"  Denoising          : Wiener filter")
    print(f"  ML model           : Random Forest (200 trees)")
    print(f"  CV accuracy        : {scores.mean():.1%}")
    print(f"  Conservation alerts: {len(alerts)}")
    print(f"  Interpreter        : {'Gemini 2.0 Flash' if use_gemini else 'Built-in'}")
    print(f"  Narrations         : {'ElevenLabs' if ELEVENLABS_API_KEY else 'Skipped'}")
    print()

    print("  SPONSOR INTEGRATIONS:")
    print(f"    {'✅' if use_gemini else '⬜'} Gemini API  — call interpretation")
    print(f"    {'✅' if ELEVENLABS_API_KEY else '⬜'} ElevenLabs — voice narration")
    print(f"    ⬜ Auth0      — see api_server.py")
    print(f"    ⬜ DigitalOcean— deploy on Droplet")
    print()

    print("  Sample interpretations:")
    for ct in results_df['predicted_type'].unique()[:4]:
        sample = results_df[results_df['predicted_type'] == ct].iloc[0]
        print(f"\n  [{ct.upper()}] elephant{int(sample['elephant_num'])}")
        print(f"  Says : {sample.get('what_elephant_is_saying','')}")
        print(f"  State: {sample.get('emotional_state','')}")
        alert = sample.get('conservation_alert')
        print(f"  Alert: {'🚨 ' + str(sample.get('alert_reason','')) if alert else 'No'}")

    print(f"\n{'═'*60}")
    print(f"\n📁 Output: {OUTPUT_ROOT}/")
    print(f"   clean_clips/      → elephant1.wav to elephant{len(manifest)}.wav")
    print(f"   spectrograms/     → spectrogram images")
    print(f"   narrations/       → voice narrations (if ElevenLabs set)")
    print(f"   model/            → rumblr_model.pkl")
    print(f"   manifest.csv      → elephantN ↔ original file + timestamps")
    print(f"   features.csv      → acoustic features")
    print(f"   final_results.csv → predictions + interpretations")
    print(f"\n✅ ALL DONE!")


if __name__ == "__main__":
    main()