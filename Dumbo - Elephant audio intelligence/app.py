import numpy as np, os, io, pickle, base64, warnings, uuid, json
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import librosa, librosa.display
from flask import Flask, request, render_template_string, jsonify
warnings.filterwarnings('ignore')

MODEL_PATH='elephant voice/model/classifier.pkl'
UPLOAD_DIR='elephant voice/uploads'
SR=22050
GEMINI_API_KEY=os.environ.get("GEMINI_API_KEY","")
ELEVENLABS_API_KEY="sk_0ed780c95727224bc28a053362fdf00aecd9c000664abab1"
ELEVENLABS_VOICE_ID="21m00Tcm4TlvDq8ikWAM"

os.makedirs(UPLOAD_DIR,exist_ok=True)
app=Flask(__name__)
app.config['MAX_CONTENT_LENGTH']=100*1024*1024

print("🧠 Loading classifier...")
with open(MODEL_PATH,'rb') as f: md=pickle.load(f)
model,scaler,le,feat_cols=md['model'],md['scaler'],md['label_encoder'],md['feature_cols']
classes=md['classes'];test_acc=md.get('test_accuracy',0);val_acc=md.get('val_accuracy',0)
best_name=md.get('best_model_name','XGBoost')
print(f"    {best_name} | {classes} | Test:{test_acc:.0%} Val:{val_acc:.0%}")
print(f"   Gemini: {'' if GEMINI_API_KEY else '️  not set'}")
print(f"   ElevenLabs:  Ready")

# ── Emotion mapping: call type + features → valence/arousal ──
EMOTION_ZONES = {
    'Rumble':  {'base_valence': 0.55, 'base_arousal': 0.25,
                'label': 'Contact / Social', 'emoji': '💚',
                'desc': 'Calm social communication — family bonding, location sharing'},
    'Trumpet': {'base_valence': -0.5, 'base_arousal': 0.75,
                'label': 'Alarm / Excitement', 'emoji': '⚡',
                'desc': 'High-energy signal — warning, surprise, or confrontation'},
    'Roar':    {'base_valence': -0.75, 'base_arousal': 0.85,
                'label': 'Distress / Protest', 'emoji': '🚨',
                'desc': 'Intense negative emotion — pain, fear, separation anxiety'},
}

def compute_emotion(call_type, features):
    zone = EMOTION_ZONES.get(call_type, EMOTION_ZONES['Rumble'])
    v = zone['base_valence']
    a = zone['base_arousal']
    energy = features.get('rms_mean', 0)
    f0 = features.get('f0_mean', 0)
    centroid = features.get('centroid_mean', 0)
    duration = features.get('duration', 0)
    zcr = features.get('zcr_mean', 0)
    a += np.clip(energy * 2 - 0.3, -0.15, 0.15)
    if f0 > 200: a += 0.1; v -= 0.1
    if f0 > 500: a += 0.1; v -= 0.1
    if centroid > 1000: v -= 0.05
    if duration > 3: a -= 0.05; v += 0.05
    if zcr > 0.1: a += 0.05
    v = float(np.clip(v, -1, 1))
    a = float(np.clip(a, 0, 1))
    if a > 0.6 and v < -0.3:
        emotion = "Distressed" if v < -0.5 else "Alarmed"
    elif a > 0.6 and v >= -0.3:
        emotion = "Excited" if v > 0.2 else "Agitated"
    elif a <= 0.6 and v >= 0:
        emotion = "Content" if v > 0.3 else "Calm"
    elif a <= 0.6 and v < 0:
        emotion = "Uneasy" if v > -0.4 else "Anxious"
    else:
        emotion = "Neutral"
    return {
        'valence': v, 'arousal': a,
        'emotion': emotion,
        'zone_label': zone['label'],
        'zone_emoji': zone['emoji'],
        'zone_desc': zone['desc'],
        'is_alert': call_type == 'Roar' or (a > 0.7 and v < -0.4),
        'alert_reason': 'Distress call detected — possible injury, separation, or threat' if call_type == 'Roar' else
                        'High-arousal negative call — monitor situation' if (a > 0.7 and v < -0.4) else None
    }


def gemini_interpret(call_type, features, emotion):
    if not GEMINI_API_KEY: return None
    try:
        import requests
        prompt = f"""You are an elephant bioacoustician. Interpret this elephant call.

Call type: {call_type}
Duration: {features.get('duration',0):.2f}s
Pitch: {features.get('f0_mean',0):.1f} Hz
Energy: {features.get('rms_mean',0):.4f}
Emotion: {emotion['emotion']} (valence={emotion['valence']:.2f}, arousal={emotion['arousal']:.2f})

Respond in valid JSON only, no markdown:
{{"translation": "What the elephant is saying in 1 sentence, as if it's speaking English — conversational and vivid",
  "behavior": "What behavior this likely accompanies in 1 sentence",
  "fun_fact": "One surprising fact about this type of elephant call"}}"""

        r = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}",
            json={"contents":[{"parts":[{"text":prompt}]}],
                  "generationConfig":{"temperature":0.5,"maxOutputTokens":300}},
            timeout=15).json()
        t = r["candidates"][0]["content"]["parts"][0]["text"].strip().strip('`')
        if t.startswith('json'): t=t[4:].strip()
        return json.loads(t)
    except:
        return None


def denoise_audio(y,sr):
    ne=min(int(.5*sr),len(y)//4)
    if ne<512:ne=min(len(y)//2,2048)
    Dn=librosa.stft(y[:ne],n_fft=2048,hop_length=512)
    np_=np.mean(np.abs(Dn),axis=1,keepdims=True)
    D=librosa.stft(y,n_fft=2048,hop_length=512);m,p=np.abs(D),np.angle(D)
    snr=np.maximum(m**2-(np_**2+1e-10),0)/(np_**2+1e-10)
    yc=librosa.istft((snr/(snr+1.))*m*np.exp(1j*p),hop_length=512)
    if np.max(np.abs(yc))>0:yc=yc/np.max(np.abs(yc))*.9
    return yc

def extract_features(y,sr):
    f={}; f['duration']=len(y)/sr
    rms=librosa.feature.rms(y=y);f['rms_mean']=float(np.mean(rms));f['rms_std']=float(np.std(rms));f['rms_max']=float(np.max(rms));f['rms_min']=float(np.min(rms))
    mid=len(rms[0])//2;f['energy_ratio']=float(np.mean(rms[0][:mid])/(np.mean(rms[0][mid:])+1e-10)) if mid>0 else 1.
    try:
        f0,v,_=librosa.pyin(y,fmin=12,fmax=2000,frame_length=4096,hop_length=512);vf=f0[v] if v is not None else np.array([])
        f['f0_mean']=float(np.nanmean(vf)) if len(vf)>0 else 0.;f['f0_std']=float(np.nanstd(vf)) if len(vf)>0 else 0.
        f['f0_min']=float(np.nanmin(vf)) if len(vf)>0 else 0.;f['f0_max']=float(np.nanmax(vf)) if len(vf)>0 else 0.
        f['f0_range']=f['f0_max']-f['f0_min'];f['f0_median']=float(np.nanmedian(vf)) if len(vf)>0 else 0.
        f['voiced_frac']=float(np.mean(v)) if v is not None else 0.
    except:
        for k in['f0_mean','f0_std','f0_min','f0_max','f0_range','f0_median','voiced_frac']:f[k]=0.
    ce=librosa.feature.spectral_centroid(y=y,sr=sr);bw=librosa.feature.spectral_bandwidth(y=y,sr=sr)
    ro=librosa.feature.spectral_rolloff(y=y,sr=sr,roll_percent=.85);r25=librosa.feature.spectral_rolloff(y=y,sr=sr,roll_percent=.25)
    co=librosa.feature.spectral_contrast(y=y,sr=sr,n_bands=6);fl=librosa.feature.spectral_flatness(y=y);zc=librosa.feature.zero_crossing_rate(y)
    f['centroid_mean']=float(np.mean(ce));f['centroid_std']=float(np.std(ce));f['centroid_max']=float(np.max(ce))
    f['bandwidth_mean']=float(np.mean(bw));f['bandwidth_std']=float(np.std(bw))
    f['rolloff85_mean']=float(np.mean(ro));f['rolloff25_mean']=float(np.mean(r25))
    f['flatness_mean']=float(np.mean(fl));f['flatness_std']=float(np.std(fl))
    f['zcr_mean']=float(np.mean(zc));f['zcr_std']=float(np.std(zc))
    for b in range(min(6,co.shape[0])):f[f'contrast_{b}_mean']=float(np.mean(co[b]));f[f'contrast_{b}_std']=float(np.std(co[b]))
    ch=librosa.feature.chroma_stft(y=y,sr=sr,n_fft=2048)
    for c in range(12):f[f'chroma_{c}_mean']=float(np.mean(ch[c]))
    f['chroma_std_mean']=float(np.mean(np.std(ch,axis=1)))
    try:
        tn=librosa.feature.tonnetz(y=librosa.effects.harmonic(y),sr=sr)
        for t in range(6):f[f'tonnetz_{t}_mean']=float(np.mean(tn[t]))
    except:
        for t in range(6):f[f'tonnetz_{t}_mean']=0.
    try:
        tp,_=librosa.beat.beat_track(y=y,sr=sr);f['tempo']=float(tp) if not hasattr(tp,'__len__') else float(tp[0])
    except:f['tempo']=0.
    mc=librosa.feature.mfcc(y=y,sr=sr,n_mfcc=13,n_fft=2048,hop_length=512)
    for k in range(13):f[f'mfcc_{k+1}_mean']=float(np.mean(mc[k]));f[f'mfcc_{k+1}_std']=float(np.std(mc[k]));f[f'mfcc_{k+1}_min']=float(np.min(mc[k]));f[f'mfcc_{k+1}_max']=float(np.max(mc[k]))
    dm=librosa.feature.delta(mc)
    for k in range(13):f[f'delta_mfcc_{k+1}_mean']=float(np.mean(dm[k]));f[f'delta_mfcc_{k+1}_std']=float(np.std(dm[k]))
    d2=librosa.feature.delta(mc,order=2)
    for k in range(13):f[f'delta2_mfcc_{k+1}_mean']=float(np.mean(d2[k]))
    return f

def make_spec_b64(y,sr,title=""):
    S=librosa.feature.melspectrogram(y=y,sr=sr,n_mels=128,fmin=10,fmax=4000,n_fft=2048,hop_length=256)
    S_db=librosa.power_to_db(S,ref=np.max)
    fig,ax=plt.subplots(figsize=(10,4))
    librosa.display.specshow(S_db,sr=sr,hop_length=256,x_axis='time',y_axis='mel',fmin=10,fmax=4000,cmap='inferno',ax=ax)
    ax.set_title(title,fontsize=13,fontweight='bold',color='#E8E0D4')
    ax.set_xlabel('Time (s)',color='#E8E0D4');ax.set_ylabel('Frequency (Hz)',color='#E8E0D4')
    ax.tick_params(colors='#E8E0D4');fig.patch.set_facecolor('#111');ax.set_facecolor('#111')
    plt.tight_layout();buf=io.BytesIO();fig.savefig(buf,format='png',dpi=120,bbox_inches='tight',facecolor='#111');plt.close(fig);buf.seek(0)
    return base64.b64encode(buf.read()).decode()


# ══════════════════════════════════════════════════════════════
# HTML — Redesigned "Savanna Observatory" UI — DUMBO
# ══════════════════════════════════════════════════════════════

HTML=r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>DUMBO — Elephant Call Classifier</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=IBM+Plex+Mono:wght@300;400;500;600&family=Sora:wght@200;300;400;500;600;700&display=swap" rel="stylesheet">
<style>
/* ═══════════════════════════════════════════════════════════
   DUMBO — "Savanna Observatory" Design System
   ═══════════════════════════════════════════════════════════ */

:root {
  --bg: #0B0E0C;
  --bg-warm: #0F1210;
  --surface: rgba(22, 28, 24, 0.75);
  --surface-elevated: rgba(30, 38, 32, 0.85);
  --accent: #D4A574;
  --accent-soft: rgba(212, 165, 116, 0.12);
  --accent-glow: rgba(212, 165, 116, 0.25);
  --text: #E4DDD4;
  --text-secondary: #8A8278;
  --text-muted: #5C564E;
  --rumble: #5BA4E6;
  --rumble-bg: rgba(91, 164, 230, 0.08);
  --trumpet: #E8B44A;
  --trumpet-bg: rgba(232, 180, 74, 0.08);
  --roar: #E05555;
  --roar-bg: rgba(224, 85, 85, 0.08);
  --green: #3DD68C;
  --border: rgba(212, 165, 116, 0.06);
  --border-hover: rgba(212, 165, 116, 0.15);
  --radius: 20px;
  --radius-sm: 12px;
  --radius-xs: 8px;
  --shadow-card: 0 4px 40px rgba(0,0,0,0.3), 0 0 0 1px var(--border);
  --shadow-glow: 0 0 80px rgba(212,165,116,0.08);
  --font-display: 'Instrument Serif', Georgia, serif;
  --font-body: 'Sora', sans-serif;
  --font-mono: 'IBM Plex Mono', monospace;
  --ease-out: cubic-bezier(0.16, 1, 0.3, 1);
  --ease-spring: cubic-bezier(0.34, 1.56, 0.64, 1);
}

*, *::before, *::after { margin: 0; padding: 0; box-sizing: border-box; }

html { scroll-behavior: smooth; }

body {
  background: var(--bg);
  color: var(--text);
  font-family: var(--font-body);
  font-weight: 300;
  min-height: 100vh;
  overflow-x: hidden;
  line-height: 1.6;
  -webkit-font-smoothing: antialiased;
}

/* ─── Topographic texture overlay ─── */
body::before {
  content: '';
  position: fixed;
  inset: 0;
  z-index: 0;
  pointer-events: none;
  background:
    radial-gradient(ellipse 80% 60% at 20% 80%, rgba(212,165,116,0.03) 0%, transparent 70%),
    radial-gradient(ellipse 60% 50% at 85% 20%, rgba(91,164,230,0.02) 0%, transparent 60%);
}

/* Grain texture */
body::after {
  content: '';
  position: fixed;
  inset: 0;
  z-index: 1;
  pointer-events: none;
  opacity: 0.3;
  background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.04'/%3E%3C/svg%3E");
}

/* ─── Particle + Confetti canvases ─── */
#particles { position: fixed; inset: 0; z-index: 0; pointer-events: none; }
#confetti  { position: fixed; inset: 0; z-index: 100; pointer-events: none; display: none; }

/* ─── Walking elephants ─── */
.ew {
  position: fixed;
  bottom: -5px;
  z-index: 1;
  pointer-events: none;
  font-size: 3rem;
  opacity: 0.04;
  animation: walk 50s linear infinite;
  filter: grayscale(1);
}
.ew:nth-child(2) { animation-duration: 65s; animation-delay: -22s; font-size: 2.4rem; bottom: 12px; opacity: 0.03; }
.ew:nth-child(3) { animation-duration: 40s; animation-delay: -12s; font-size: 3.6rem; bottom: -8px; opacity: 0.035; }
@keyframes walk { 0% { transform: translateX(-100px) scaleX(-1); } 100% { transform: translateX(calc(100vw + 100px)) scaleX(-1); } }

/* ─── Main container ─── */
.wrap {
  max-width: 860px;
  margin: 0 auto;
  padding: 64px 24px 48px;
  position: relative;
  z-index: 10;
}

/* ═══════════════════════════════════════════════════════════
   HEADER
   ═══════════════════════════════════════════════════════════ */
.hdr {
  text-align: center;
  margin-bottom: 64px;
  animation: headerIn 1.4s var(--ease-out) both;
}

.hdr h1 {
  font-family: var(--font-display);
  font-size: clamp(3.2rem, 8vw, 5.5rem);
  font-weight: 400;
  font-style: italic;
  color: var(--accent);
  letter-spacing: 0.12em;
  line-height: 1;
  position: relative;
  display: inline-block;
}

/* Subtle underline flourish */
.hdr h1::after {
  content: '';
  position: absolute;
  bottom: -8px;
  left: 15%;
  width: 70%;
  height: 1px;
  background: linear-gradient(90deg, transparent, var(--accent), transparent);
  opacity: 0.3;
}

.hdr .sub {
  font-family: var(--font-mono);
  font-size: 0.7rem;
  font-weight: 400;
  color: var(--text-muted);
  letter-spacing: 0.35em;
  text-transform: uppercase;
  margin-top: 18px;
  animation: fadeUp 0.9s var(--ease-out) 0.25s both;
}

.hdr .pills {
  display: inline-flex;
  gap: 1px;
  margin-top: 24px;
  background: var(--border);
  border-radius: 100px;
  overflow: hidden;
  font-family: var(--font-mono);
  font-size: 0.65rem;
  font-weight: 400;
  animation: fadeUp 0.9s var(--ease-out) 0.4s both;
}

.hdr .pills > div {
  padding: 8px 20px;
  background: var(--surface);
  color: var(--text-muted);
  letter-spacing: 0.06em;
}

.hdr .pills span {
  color: var(--accent);
  font-weight: 500;
}

/* ═══════════════════════════════════════════════════════════
   DROP ZONE
   ═══════════════════════════════════════════════════════════ */
.drop {
  position: relative;
  border: 1.5px solid var(--border);
  border-radius: var(--radius);
  background: var(--surface);
  backdrop-filter: blur(24px);
  padding: 72px 48px;
  text-align: center;
  cursor: pointer;
  transition: all 0.6s var(--ease-out);
  overflow: hidden;
  animation: fadeUp 0.9s var(--ease-out) 0.55s both;
}

/* Animated gradient border on hover */
.drop::before {
  content: '';
  position: absolute;
  inset: -1.5px;
  border-radius: var(--radius);
  background: conic-gradient(
    from 0deg,
    transparent 0%,
    var(--accent) 10%,
    transparent 20%,
    transparent 50%,
    var(--accent) 60%,
    transparent 70%
  );
  opacity: 0;
  transition: opacity 0.6s;
  z-index: -2;
  animation: borderSpin 6s linear infinite;
}

/* Inner background to mask the border gradient */
.drop::after {
  content: '';
  position: absolute;
  inset: 1.5px;
  border-radius: calc(var(--radius) - 1.5px);
  background: var(--surface);
  z-index: -1;
  transition: background 0.6s;
}

.drop:hover::before,
.drop.over::before { opacity: 0.6; }

.drop:hover,
.drop.over {
  border-color: transparent;
  transform: translateY(-3px);
  box-shadow: var(--shadow-glow);
}

.drop:hover::after,
.drop.over::after {
  background: var(--surface-elevated);
}

@keyframes borderSpin { to { transform: rotate(360deg); } }

.drop .ei {
  font-size: 4rem;
  display: block;
  margin-bottom: 20px;
  filter: drop-shadow(0 0 30px rgba(212,165,116,0.2));
  animation: elephantFloat 4s ease-in-out infinite;
}

@keyframes elephantFloat {
  0%, 100% { transform: translateY(0) rotate(0deg); }
  25% { transform: translateY(-8px) rotate(-2deg); }
  75% { transform: translateY(-4px) rotate(1deg); }
}

.drop h2 {
  font-family: var(--font-display);
  font-size: 1.5rem;
  font-weight: 400;
  margin-bottom: 8px;
  color: var(--text);
}

.drop p {
  color: var(--text-muted);
  font-size: 0.8rem;
  font-weight: 300;
  max-width: 380px;
  margin: 0 auto;
  line-height: 1.7;
}

.drop .fmt {
  display: inline-flex;
  gap: 8px;
  margin-top: 20px;
}

.drop .fmt span {
  background: var(--accent-soft);
  border: 1px solid rgba(212,165,116,0.1);
  border-radius: 100px;
  padding: 4px 14px;
  font-family: var(--font-mono);
  font-size: 0.62rem;
  font-weight: 500;
  color: var(--accent);
  letter-spacing: 0.04em;
}

.drop input {
  position: absolute;
  inset: 0;
  opacity: 0;
  cursor: pointer;
  z-index: 5;
}

/* ═══════════════════════════════════════════════════════════
   LOADING STATE
   ═══════════════════════════════════════════════════════════ */
.ld {
  display: none;
  text-align: center;
  padding: 80px 40px;
  animation: fadeUp 0.5s var(--ease-out);
}
.ld.on { display: block; }

.sw {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 5px;
  height: 64px;
  margin-bottom: 28px;
}

.sw .b {
  width: 3px;
  border-radius: 100px;
  background: linear-gradient(to top, var(--accent), rgba(212,165,116,0.3));
  animation: soundWave 1.2s ease-in-out infinite;
}

.sw .b:nth-child(1)  { height: 18px; }
.sw .b:nth-child(2)  { height: 32px; animation-delay: .08s; }
.sw .b:nth-child(3)  { height: 48px; animation-delay: .16s; }
.sw .b:nth-child(4)  { height: 38px; animation-delay: .24s; }
.sw .b:nth-child(5)  { height: 22px; animation-delay: .32s; }
.sw .b:nth-child(6)  { height: 42px; animation-delay: .40s; }
.sw .b:nth-child(7)  { height: 28px; animation-delay: .48s; }
.sw .b:nth-child(8)  { height: 52px; animation-delay: .56s; }
.sw .b:nth-child(9)  { height: 34px; animation-delay: .64s; }
.sw .b:nth-child(10) { height: 18px; animation-delay: .72s; }

@keyframes soundWave {
  0%, 100% { transform: scaleY(0.3); opacity: 0.3; }
  50%      { transform: scaleY(1);   opacity: 1; }
}

.ld p {
  font-family: var(--font-display);
  font-size: 1.1rem;
  font-style: italic;
  color: var(--text-secondary);
}

.ld .stg {
  font-family: var(--font-mono);
  font-size: 0.68rem;
  font-weight: 400;
  color: var(--accent);
  margin-top: 10px;
  letter-spacing: 0.08em;
  animation: stagePulse 2s ease-in-out infinite;
}

@keyframes stagePulse {
  0%, 100% { opacity: 0.4; }
  50%      { opacity: 1; }
}

/* ═══════════════════════════════════════════════════════════
   RESULTS
   ═══════════════════════════════════════════════════════════ */
.res { display: none; }
.res.on { display: block; }

/* ─── Card base ─── */
.rc {
  background: var(--surface);
  backdrop-filter: blur(24px);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 36px;
  margin-bottom: 20px;
  box-shadow: var(--shadow-card);
  animation: cardReveal 0.7s var(--ease-out) both;
}

.rc:nth-child(2) { animation-delay: 0.1s; }
.rc:nth-child(3) { animation-delay: 0.2s; }
.rc:nth-child(4) { animation-delay: 0.3s; }
.rc:nth-child(5) { animation-delay: 0.4s; }
.rc:nth-child(6) { animation-delay: 0.5s; }

@keyframes cardReveal {
  from {
    opacity: 0;
    transform: translateY(24px) scale(0.98);
    filter: blur(4px);
  }
  to {
    opacity: 1;
    transform: translateY(0) scale(1);
    filter: blur(0);
  }
}

/* ─── Result header ─── */
.rh {
  display: flex;
  align-items: center;
  gap: 24px;
  margin-bottom: 28px;
  flex-wrap: wrap;
}

/* ─── Call type badge ─── */
.badge {
  font-family: var(--font-display);
  font-size: 2.6rem;
  font-weight: 400;
  font-style: italic;
  padding: 10px 36px;
  border-radius: var(--radius-sm);
  text-transform: capitalize;
  letter-spacing: 0.04em;
  position: relative;
  overflow: hidden;
  animation: badgePop 0.6s var(--ease-spring) 0.2s both;
}

.badge::before {
  content: '';
  position: absolute;
  inset: 0;
  background: linear-gradient(135deg, rgba(255,255,255,0.08), transparent 60%);
  border-radius: var(--radius-sm);
}

@keyframes badgePop {
  0%   { transform: scale(0) rotate(-8deg); opacity: 0; }
  60%  { transform: scale(1.05) rotate(1deg); }
  100% { transform: scale(1) rotate(0); opacity: 1; }
}

.badge.Rumble {
  background: linear-gradient(145deg, rgba(91,164,230,0.18), rgba(91,164,230,0.04));
  color: var(--rumble);
  border: 1px solid rgba(91,164,230,0.2);
  box-shadow: 0 0 40px rgba(91,164,230,0.08), inset 0 0 30px rgba(91,164,230,0.03);
}

.badge.Trumpet {
  background: linear-gradient(145deg, rgba(232,180,74,0.18), rgba(232,180,74,0.04));
  color: var(--trumpet);
  border: 1px solid rgba(232,180,74,0.2);
  box-shadow: 0 0 40px rgba(232,180,74,0.08), inset 0 0 30px rgba(232,180,74,0.03);
}

.badge.Roar {
  background: linear-gradient(145deg, rgba(224,85,85,0.18), rgba(224,85,85,0.04));
  color: var(--roar);
  border: 1px solid rgba(224,85,85,0.2);
  box-shadow: 0 0 40px rgba(224,85,85,0.08), inset 0 0 30px rgba(224,85,85,0.03);
}

/* ─── Confidence ─── */
.conf {
  font-family: var(--font-mono);
  font-size: 0.75rem;
  font-weight: 400;
  color: var(--text-muted);
  letter-spacing: 0.03em;
}

.conf strong {
  font-family: var(--font-body);
  font-weight: 600;
  color: var(--accent);
  font-size: 2rem;
  display: block;
  line-height: 1.1;
  letter-spacing: -0.02em;
}

/* ─── Probability bars ─── */
.pg { display: grid; gap: 10px; margin-bottom: 28px; }

.pr {
  display: grid;
  grid-template-columns: 100px 1fr 52px;
  align-items: center;
  gap: 14px;
}

.pl {
  font-family: var(--font-mono);
  font-size: 0.72rem;
  font-weight: 500;
  color: var(--text-secondary);
  text-transform: uppercase;
  letter-spacing: 0.06em;
}

.pbg {
  height: 6px;
  background: rgba(255,255,255,0.03);
  border-radius: 100px;
  overflow: hidden;
}

.pb {
  height: 100%;
  border-radius: 100px;
  transition: width 1.2s var(--ease-out);
  background: linear-gradient(90deg, var(--accent), rgba(212,165,116,0.6));
}

.pb.lo {
  background: rgba(255,255,255,0.08);
}

.pv {
  font-family: var(--font-mono);
  font-size: 0.72rem;
  font-weight: 400;
  color: var(--text-muted);
  text-align: right;
}

/* ─── Feature grid ─── */
.fg {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 12px;
}

.fi {
  background: rgba(255,255,255,0.015);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  padding: 20px 16px;
  text-align: center;
  transition: all 0.4s var(--ease-out);
}

.fi:hover {
  background: var(--accent-soft);
  border-color: var(--border-hover);
  transform: translateY(-2px);
}

.fi .fv {
  font-family: var(--font-mono);
  font-size: 1.3rem;
  font-weight: 600;
  color: var(--accent);
  letter-spacing: -0.02em;
}

.fi .fl {
  font-family: var(--font-mono);
  font-size: 0.6rem;
  font-weight: 400;
  color: var(--text-muted);
  text-transform: uppercase;
  margin-top: 6px;
  letter-spacing: 0.1em;
}

/* ═══════════════════════════════════════════════════════════
   CONSERVATION ALERT
   ═══════════════════════════════════════════════════════════ */
.alert-banner {
  background: linear-gradient(135deg, rgba(224,85,85,0.1), rgba(224,85,85,0.03));
  border: 1px solid rgba(224,85,85,0.2);
  border-left: 3px solid var(--roar);
  border-radius: var(--radius);
  padding: 24px 28px;
  margin-bottom: 20px;
  display: flex;
  align-items: flex-start;
  gap: 16px;
  animation: alertGlow 3s ease-in-out infinite;
}

@keyframes alertGlow {
  0%, 100% { box-shadow: 0 0 20px rgba(224,85,85,0.05); }
  50%      { box-shadow: 0 0 50px rgba(224,85,85,0.15); }
}

.alert-banner .alert-icon { font-size: 1.6rem; flex-shrink: 0; line-height: 1.4; }

.alert-banner .alert-text h4 {
  font-family: var(--font-mono);
  font-size: 0.7rem;
  font-weight: 600;
  color: var(--roar);
  letter-spacing: 0.12em;
  text-transform: uppercase;
  margin-bottom: 6px;
}

.alert-banner .alert-text p {
  font-size: 0.82rem;
  font-weight: 300;
  color: var(--text-secondary);
  line-height: 1.6;
}

/* ═══════════════════════════════════════════════════════════
   GEMINI TRANSLATION
   ═══════════════════════════════════════════════════════════ */
.translation {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 32px;
  margin-bottom: 20px;
  position: relative;
  overflow: hidden;
}

/* Decorative quote mark */
.translation::before {
  content: '\201C';
  position: absolute;
  top: 8px;
  left: 20px;
  font-family: var(--font-display);
  font-size: 5rem;
  color: rgba(212,165,116,0.08);
  line-height: 1;
  pointer-events: none;
}

/* Soft side glow */
.translation::after {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  width: 3px;
  height: 100%;
  background: linear-gradient(to bottom, var(--accent), transparent);
  opacity: 0.3;
  border-radius: 0 0 0 var(--radius);
}

.translation .tq {
  font-family: var(--font-display);
  font-size: 1.2rem;
  font-style: italic;
  color: var(--text);
  padding-left: 32px;
  line-height: 1.7;
}

.translation .tmeta {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
  margin-top: 24px;
  padding-left: 32px;
  padding-top: 20px;
  border-top: 1px solid var(--border);
}

.translation .tmeta div {
  font-size: 0.78rem;
  font-weight: 300;
  color: var(--text-secondary);
  line-height: 1.6;
}

.translation .tmeta div strong {
  font-family: var(--font-mono);
  font-weight: 500;
  color: var(--accent);
  display: block;
  font-size: 0.6rem;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  margin-bottom: 6px;
}

/* ═══════════════════════════════════════════════════════════
   EMOTION MAP
   ═══════════════════════════════════════════════════════════ */
.emap {
  position: relative;
  width: 100%;
  aspect-ratio: 1.6 / 1;
  border-radius: var(--radius-sm);
  overflow: hidden;
  border: 1px solid var(--border);
  background: rgba(0,0,0,0.3);
}

.emap canvas {
  width: 100%;
  height: 100%;
  display: block;
}

.emap-legend {
  display: flex;
  gap: 20px;
  flex-wrap: wrap;
  justify-content: center;
  margin-top: 16px;
}

.emap-legend .el {
  display: flex;
  align-items: center;
  gap: 8px;
  font-family: var(--font-mono);
  font-size: 0.65rem;
  font-weight: 400;
  color: var(--text-muted);
  letter-spacing: 0.04em;
}

.emap-legend .el .dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
}

/* ═══════════════════════════════════════════════════════════
   SPECTROGRAM
   ═══════════════════════════════════════════════════════════ */
.ss {
  background: var(--surface);
  backdrop-filter: blur(24px);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 28px;
  margin-bottom: 20px;
  box-shadow: var(--shadow-card);
}

.ss h3 {
  font-family: var(--font-mono);
  font-size: 0.65rem;
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.15em;
  color: var(--text-muted);
  margin-bottom: 16px;
}

.ss img {
  width: 100%;
  border-radius: var(--radius-xs);
  display: block;
  transition: transform 0.4s var(--ease-out), box-shadow 0.4s;
}

.ss img:hover {
  transform: scale(1.015);
  box-shadow: 0 8px 40px rgba(0,0,0,0.4);
}

.sg {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 24px;
}

/* ═══════════════════════════════════════════════════════════
   VOICE PLAYBACK (ElevenLabs)
   ═══════════════════════════════════════════════════════════ */
/* Styled via inline styles in JS — no changes needed */

/* ═══════════════════════════════════════════════════════════
   ANALYZE ANOTHER
   ═══════════════════════════════════════════════════════════ */
.ta {
  text-align: center;
  margin-top: 36px;
}

.ta button {
  position: relative;
  background: transparent;
  border: 1px solid rgba(212,165,116,0.25);
  color: var(--accent);
  padding: 14px 40px;
  border-radius: 100px;
  font-family: var(--font-body);
  font-size: 0.88rem;
  font-weight: 400;
  letter-spacing: 0.06em;
  cursor: pointer;
  transition: all 0.5s var(--ease-out);
  overflow: hidden;
}

.ta button::before {
  content: '';
  position: absolute;
  inset: 0;
  background: var(--accent);
  border-radius: 100px;
  transform: scaleX(0);
  transform-origin: right;
  transition: transform 0.5s var(--ease-out);
  z-index: -1;
}

.ta button:hover {
  color: var(--bg);
  border-color: var(--accent);
  box-shadow: 0 0 40px rgba(212,165,116,0.15);
}

.ta button:hover::before {
  transform: scaleX(1);
  transform-origin: left;
}

/* ═══════════════════════════════════════════════════════════
   FOOTER
   ═══════════════════════════════════════════════════════════ */
.foot {
  text-align: center;
  margin-top: 56px;
  padding: 28px 24px;
  border-top: 1px solid var(--border);
  color: var(--text-muted);
  font-family: var(--font-mono);
  font-size: 0.58rem;
  font-weight: 400;
  letter-spacing: 0.2em;
  text-transform: uppercase;
  position: relative;
  z-index: 10;
}

/* ═══════════════════════════════════════════════════════════
   KEYFRAMES
   ═══════════════════════════════════════════════════════════ */
@keyframes headerIn {
  from { opacity: 0; transform: translateY(-20px); filter: blur(8px); }
  to   { opacity: 1; transform: translateY(0);    filter: blur(0); }
}

@keyframes fadeUp {
  from { opacity: 0; transform: translateY(16px); }
  to   { opacity: 1; transform: translateY(0); }
}

/* ═══════════════════════════════════════════════════════════
   RESPONSIVE
   ═══════════════════════════════════════════════════════════ */
@media (max-width: 640px) {
  .wrap { padding: 40px 16px 32px; }
  .hdr { margin-bottom: 40px; }
  .hdr h1 { letter-spacing: 0.06em; }
  .hdr .pills { flex-wrap: wrap; justify-content: center; }
  .drop { padding: 48px 24px; }
  .rc { padding: 24px; }
  .sg { grid-template-columns: 1fr; }
  .fg { grid-template-columns: repeat(2, 1fr); }
  .pr { grid-template-columns: 80px 1fr 44px; }
  .translation .tmeta { grid-template-columns: 1fr; }
  .rh { gap: 16px; }
  .badge { font-size: 2rem; padding: 8px 24px; }
  .emap-legend { gap: 12px; }
}

@media (max-width: 380px) {
  .fg { grid-template-columns: 1fr; }
  .hdr .pills > div { padding: 6px 14px; font-size: 0.6rem; }
}

/* ─── Scrollbar ─── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
  background: rgba(212,165,116,0.15);
  border-radius: 100px;
}
::-webkit-scrollbar-thumb:hover { background: rgba(212,165,116,0.3); }

/* ─── Selection ─── */
::selection {
  background: rgba(212,165,116,0.2);
  color: var(--text);
}
</style>
</head>
<body>
<div class="ew">🐘</div><div class="ew">🐘</div><div class="ew">🐘</div>
<canvas id="particles"></canvas><canvas id="confetti"></canvas>

<div class="wrap">
  <div class="hdr">
    <h1>Dumbo</h1>
    <p class="sub">Elephant Call Classifier & Emotion Mapper</p>
  </div>

  <div class="drop" id="dz">
    <span class="ei">🐘</span>
    <h2>Drop an elephant recording here</h2>
    <p>Denoises, classifies, maps emotions, and translates what the elephant is saying</p>
    <div class="fmt"><span>.wav</span><span>Rumble</span><span>Trumpet</span><span>Roar</span></div>
    <input type="file" id="fi" accept=".wav,.WAV">
  </div>

  <div class="ld" id="ld">
    <div class="sw"><div class="b"></div><div class="b"></div><div class="b"></div><div class="b"></div><div class="b"></div><div class="b"></div><div class="b"></div><div class="b"></div><div class="b"></div><div class="b"></div></div>
    <p>Listening to the elephant...</p>
    <div class="stg" id="stg">removing background noise</div>
  </div>

  <div class="res" id="res"></div>
</div>

<div class="foot">HackSMU VII &mdash; ElephantVoices &mdash; Gemini · ElevenLabs · Auth0 · DigitalOcean</div>

<script>
// ─── PARTICLES ───
const pc=document.getElementById('particles'),ctx=pc.getContext('2d');let W,H,dots=[];
function rsz(){W=pc.width=innerWidth;H=pc.height=innerHeight}rsz();addEventListener('resize',rsz);
for(let i=0;i<80;i++)dots.push({x:Math.random()*W,y:Math.random()*H,r:Math.random()*1.5+.5,vx:(Math.random()-.5)*.3,vy:(Math.random()-.5)*.3,o:Math.random()*.12+.03});
function dp(){ctx.clearRect(0,0,W,H);dots.forEach(d=>{d.x+=d.vx;d.y+=d.vy;if(d.x<0)d.x=W;if(d.x>W)d.x=0;if(d.y<0)d.y=H;if(d.y>H)d.y=0;ctx.beginPath();ctx.arc(d.x,d.y,d.r,0,Math.PI*2);ctx.fillStyle=`rgba(212,165,116,${d.o})`;ctx.fill()});for(let i=0;i<dots.length;i++)for(let j=i+1;j<dots.length;j++){const dx=dots[i].x-dots[j].x,dy=dots[i].y-dots[j].y,ds=Math.sqrt(dx*dx+dy*dy);if(ds<120){ctx.beginPath();ctx.moveTo(dots[i].x,dots[i].y);ctx.lineTo(dots[j].x,dots[j].y);ctx.strokeStyle=`rgba(212,165,116,${.03*(1-ds/120)})`;ctx.lineWidth=.5;ctx.stroke()}}requestAnimationFrame(dp)}dp();

// ─── CONFETTI ───
const cc=document.getElementById('confetti'),cx=cc.getContext('2d');let cf=[],cfA=false;
function rcf(){cc.width=innerWidth;cc.height=innerHeight}rcf();addEventListener('resize',rcf);
function boom(){cc.style.display='block';cfA=true;cf=[];const co=['#D4A574','#5BA4E6','#E8B44A','#E05555','#3DD68C','#E4DDD4'];for(let i=0;i<150;i++)cf.push({x:innerWidth/2+(Math.random()-.5)*200,y:innerHeight/2,vx:(Math.random()-.5)*15,vy:Math.random()*-18-5,r:Math.random()*6+3,c:co[~~(Math.random()*co.length)],rot:Math.random()*360,rv:(Math.random()-.5)*10,g:.4+Math.random()*.3,l:1});acf();setTimeout(()=>cfA=false,3000)}
function acf(){cx.clearRect(0,0,cc.width,cc.height);let al=false;cf.forEach(c=>{c.vy+=c.g;c.x+=c.vx;c.y+=c.vy;c.rot+=c.rv;c.l-=.008;if(c.l<=0)return;al=true;cx.save();cx.translate(c.x,c.y);cx.rotate(c.rot*Math.PI/180);cx.globalAlpha=c.l;cx.fillStyle=c.c;cx.fillRect(-c.r/2,-c.r/2,c.r,c.r*.6);cx.restore()});if(al||cfA)requestAnimationFrame(acf);else cc.style.display='none'}

// ─── EMOTION MAP RENDERER ───
function drawEmotionMap(canvasId, valence, arousal, callType) {
  const c = document.getElementById(canvasId);
  const x = c.getContext('2d');
  const w = c.width = c.offsetWidth * 2;
  const h = c.height = c.offsetHeight * 2;
  x.scale(2, 2);
  const W = w/2, H = h/2;

  // Background gradient quadrants
  const zones = [
    {x:0,y:0,w:W/2,h:H/2,color:'rgba(91,164,230,.05)'},
    {x:W/2,y:0,w:W/2,h:H/2,color:'rgba(224,85,85,.05)'},
    {x:0,y:H/2,w:W/2,h:H/2,color:'rgba(61,214,140,.05)'},
    {x:W/2,y:H/2,w:W/2,h:H/2,color:'rgba(232,180,74,.05)'},
  ];
  zones.forEach(z => {
    x.fillStyle = z.color;
    x.fillRect(z.x, z.y, z.w, z.h);
  });

  x.strokeStyle = 'rgba(255,255,255,.03)';
  x.lineWidth = 1;
  for (let i = 1; i < 4; i++) {
    x.beginPath(); x.moveTo(W * i/4, 0); x.lineTo(W * i/4, H); x.stroke();
    x.beginPath(); x.moveTo(0, H * i/4); x.lineTo(W, H * i/4); x.stroke();
  }

  x.strokeStyle = 'rgba(255,255,255,.08)'; x.lineWidth = 1;
  x.beginPath(); x.moveTo(W/2, 0); x.lineTo(W/2, H); x.stroke();
  x.beginPath(); x.moveTo(0, H/2); x.lineTo(W, H/2); x.stroke();

  x.fillStyle = 'rgba(255,255,255,.2)';
  x.font = '500 9px "IBM Plex Mono"';
  x.textAlign = 'center';
  x.fillText('HIGH AROUSAL', W/2, 14);
  x.fillText('LOW AROUSAL', W/2, H - 6);
  x.save(); x.translate(12, H/2); x.rotate(-Math.PI/2);
  x.fillText('POSITIVE', 0, 0); x.restore();
  x.save(); x.translate(W - 6, H/2); x.rotate(Math.PI/2);
  x.fillText('NEGATIVE', 0, 0); x.restore();

  x.fillStyle = 'rgba(255,255,255,.06)';
  x.font = '600 10px Sora';
  x.fillText('Content', W*0.25, H*0.75);
  x.fillText('Anxious', W*0.75, H*0.75);
  x.fillText('Excited', W*0.25, H*0.25);
  x.fillText('Distressed', W*0.75, H*0.25);

  const refs = [
    {v:0.55,a:0.25,color:'rgba(91,164,230,.15)',label:'Rumble zone'},
    {v:-0.5,a:0.75,color:'rgba(232,180,74,.15)',label:'Trumpet zone'},
    {v:-0.75,a:0.85,color:'rgba(224,85,85,.15)',label:'Roar zone'},
  ];
  refs.forEach(r => {
    const px = W/2 + (-r.v) * (W/2) * 0.85;
    const py = H - r.a * H * 0.85 - H*0.075;
    x.beginPath(); x.arc(px, py, 24, 0, Math.PI*2);
    x.fillStyle = r.color; x.fill();
    x.fillStyle = 'rgba(255,255,255,.1)';
    x.font = '400 7px "IBM Plex Mono"';
    x.fillText(r.label, px, py + 34);
  });

  const px = W/2 + (-valence) * (W/2) * 0.85;
  const py = H - arousal * H * 0.85 - H*0.075;

  const colors = {Rumble:'#5BA4E6',Trumpet:'#E8B44A',Roar:'#E05555'};
  const col = colors[callType] || '#D4A574';

  const grd = x.createRadialGradient(px, py, 0, px, py, 44);
  grd.addColorStop(0, col + '30');
  grd.addColorStop(1, 'transparent');
  x.fillStyle = grd; x.beginPath(); x.arc(px, py, 44, 0, Math.PI*2); x.fill();

  x.beginPath(); x.arc(px, py, 12, 0, Math.PI*2);
  x.strokeStyle = col; x.lineWidth = 1.5; x.stroke();

  x.beginPath(); x.arc(px, py, 5, 0, Math.PI*2);
  x.fillStyle = col; x.fill();

  x.strokeStyle = col + '30'; x.lineWidth = 1; x.setLineDash([3,5]);
  x.beginPath(); x.moveTo(px, 0); x.lineTo(px, H); x.stroke();
  x.beginPath(); x.moveTo(0, py); x.lineTo(W, py); x.stroke();
  x.setLineDash([]);
}

// ─── FILE HANDLING ───
const dz=document.getElementById('dz'),fi=document.getElementById('fi'),ld=document.getElementById('ld'),res=document.getElementById('res'),stg=document.getElementById('stg');
dz.addEventListener('dragover',e=>{e.preventDefault();dz.classList.add('over')});dz.addEventListener('dragleave',()=>dz.classList.remove('over'));
dz.addEventListener('drop',e=>{e.preventDefault();dz.classList.remove('over');if(e.dataTransfer.files.length)go(e.dataTransfer.files[0])});
fi.addEventListener('change',e=>{if(e.target.files.length)go(e.target.files[0])});

const stages=['removing background noise','extracting acoustic features','mapping emotional state','classifying call type','asking Gemini for translation'];
let si=0,st;

function go(f){
  if(!f.name.toLowerCase().endsWith('.wav')){alert('Upload a .wav file');return}
  dz.style.display='none';ld.classList.add('on');res.classList.remove('on');res.innerHTML='';
  si=0;stg.textContent=stages[0];st=setInterval(()=>{si=(si+1)%stages.length;stg.textContent=stages[si]},1500);
  const fd=new FormData();fd.append('file',f);
  fetch('/classify',{method:'POST',body:fd}).then(r=>r.json()).then(d=>{clearInterval(st);ld.classList.remove('on');res.classList.add('on');show(d);boom()}).catch(e=>{clearInterval(st);ld.classList.remove('on');dz.style.display='block';alert(e.message)});
}

function show(d){
  const t=d.predicted_type, em=d.emotion;
  let bars='';d.probabilities.forEach(p=>{const top=p.class===t;bars+=`<div class="pr"><div class="pl">${p.class}</div><div class="pbg"><div class="pb ${top?'':'lo'}" style="width:0%" data-w="${p.probability*100}"></div></div><div class="pv">${(p.probability*100).toFixed(1)}%</div></div>`});

  let alert='';
  if(em.is_alert){
    alert=`<div class="alert-banner"><div class="alert-icon">🚨</div><div class="alert-text"><h4>Conservation Alert</h4><p>${em.alert_reason}</p></div></div>`;
  }

  let trans='';
  if(d.gemini){
    trans=`<div class="translation rc">
      <div class="tq">${d.gemini.translation}</div>
      <div class="tmeta">
        <div><strong>Behavior</strong>${d.gemini.behavior}</div>
        <div><strong>Fun Fact</strong>${d.gemini.fun_fact}</div>
      </div>
    </div>`;
  }

  res.innerHTML=`
    ${alert}
    <div class="rc">
      <div class="rh"><div class="badge ${t}">${t}</div><div class="conf"><strong>${(d.confidence*100).toFixed(0)}%</strong>confidence</div></div>
      <div class="pg">${bars}</div>
      <div class="fg">
        <div class="fi"><div class="fv">${d.features.duration.toFixed(2)}s</div><div class="fl">Duration</div></div>
        <div class="fi"><div class="fv">${d.features.f0_mean.toFixed(1)} Hz</div><div class="fl">Pitch (F0)</div></div>
        <div class="fi"><div class="fv">${d.features.centroid_mean.toFixed(0)} Hz</div><div class="fl">Spectral Centroid</div></div>
      </div>
    </div>
    ${trans}
    <div class="rc">
      <h3 style="font-family:var(--font-mono);font-size:.65rem;font-weight:500;text-transform:uppercase;letter-spacing:.15em;color:var(--text-muted);margin-bottom:8px">Emotion Map</h3>
      <p style="font-size:.82rem;font-weight:300;color:var(--text-secondary);margin-bottom:20px">${em.zone_emoji} <strong style="color:var(--text);font-weight:500">${em.emotion}</strong> — ${em.zone_desc}</p>
      <div class="emap"><canvas id="emapCanvas"></canvas></div>
      <div class="emap-legend">
        <div class="el"><div class="dot" style="background:var(--rumble)"></div>Rumble zone</div>
        <div class="el"><div class="dot" style="background:var(--trumpet)"></div>Trumpet zone</div>
        <div class="el"><div class="dot" style="background:var(--roar)"></div>Roar zone</div>
        <div class="el"><div class="dot" style="background:var(--accent);box-shadow:0 0 8px var(--accent)"></div>This call</div>
      </div>
    </div>
    <div class="ss rc"><div class="sg"><div><h3>Original Audio</h3><img src="data:image/png;base64,${d.spec_original}"></div><div><h3>After Denoising</h3><img src="data:image/png;base64,${d.spec_cleaned}"></div></div></div>
    <div class="rc" id="voiceCard" style="display:flex;align-items:center;gap:18px;flex-wrap:wrap">
      <button id="speakBtn" onclick="speakResult()" style="
        background:linear-gradient(135deg,rgba(212,165,116,.1),rgba(212,165,116,.03));
        border:1px solid rgba(212,165,116,.2);color:var(--accent);
        padding:13px 26px;border-radius:var(--radius-sm);cursor:pointer;font-family:var(--font-body);
        font-size:.85rem;font-weight:400;display:flex;align-items:center;gap:10px;transition:all .4s var(--ease-out);
        white-space:nowrap;flex-shrink:0;letter-spacing:.03em;
      ">
        <span id="speakIcon">🔊</span> <span id="speakText">Listen to Analysis</span>
      </button>
      <audio id="narrationAudio" style="display:none"></audio>
      <div style="font-family:var(--font-mono);font-size:.62rem;font-weight:400;color:var(--text-muted);letter-spacing:.06em">
        Powered by ElevenLabs AI Voice
      </div>
    </div>
    <div class="ta"><button onclick="reset()">Analyze Another Recording</button></div>`;

  setTimeout(()=>{document.querySelectorAll('.pb[data-w]').forEach(b=>b.style.width=b.dataset.w+'%')},100);
  setTimeout(()=>drawEmotionMap('emapCanvas',em.valence,em.arousal,t),200);

  window._lastResult = d;
  setTimeout(()=>speakResult(), 1200);
}

function speakResult(){
  const d = window._lastResult;
  if(!d) return;
  const btn=document.getElementById('speakBtn');
  const icon=document.getElementById('speakIcon');
  const txt=document.getElementById('speakText');
  btn.disabled=true; icon.textContent='⏳'; txt.textContent='Generating voice...';
  btn.style.opacity='0.5';

  const body = {
    call_type: d.predicted_type,
    emotion: d.emotion ? d.emotion.emotion : 'unknown',
    confidence: Math.round(d.confidence * 100),
    translation: d.gemini ? d.gemini.translation : '',
    alert: d.emotion && d.emotion.is_alert ? d.emotion.alert_reason : ''
  };

  fetch('/speak',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})
    .then(r=>r.json())
    .then(resp=>{
      if(resp.audio){
        const audio=document.getElementById('narrationAudio');
        audio.src='data:audio/mpeg;base64,'+resp.audio;
        audio.style.display='block';
        audio.play();
        icon.textContent='🔊'; txt.textContent='Playing...';
        btn.style.opacity='1';
        audio.onended=()=>{icon.textContent='🔊';txt.textContent='Play Again';btn.disabled=false};
      } else {
        icon.textContent='⚠️'; txt.textContent='Voice unavailable';
        btn.disabled=false; btn.style.opacity='1';
      }
    })
    .catch(e=>{
      icon.textContent='⚠️'; txt.textContent='Voice unavailable';
      btn.disabled=false; btn.style.opacity='1';
    });
}

function reset(){res.classList.remove('on');res.innerHTML='';dz.style.display='block';fi.value='';window._lastResult=null}
</script>
</body>
</html>"""


@app.route('/')
def index():
    return render_template_string(HTML,model_name=best_name,test_acc=f"{test_acc:.0%}",val_acc=f"{val_acc:.0%}")

@app.route('/classify',methods=['POST'])
def classify():
    if 'file' not in request.files: return jsonify({'error':'No file'}),400
    file=request.files['file']
    if not file.filename.lower().endswith('.wav'): return jsonify({'error':'Only .wav'}),400
    fp=os.path.join(UPLOAD_DIR,f"{uuid.uuid4().hex[:8]}.wav");file.save(fp)
    try:
        yo,sr=librosa.load(fp,sr=SR,mono=True)
        so=make_spec_b64(yo,sr,"Original")
        yc=denoise_audio(yo,sr)
        sc=make_spec_b64(yc,sr,"Denoised")
        ft=extract_features(yc,sr)
        X=np.nan_to_num(np.array([[ft.get(c,0) for c in feat_cols]]),nan=0.,posinf=0.,neginf=0.)
        Xs=scaler.transform(X)
        pi=model.predict(Xs)[0];pp=model.predict_proba(Xs)[0];pt=le.inverse_transform([pi])[0]
        probs=sorted([{'class':le.classes_[i],'probability':float(pp[i])} for i in range(len(le.classes_))],key=lambda x:x['probability'],reverse=True)

        emotion=compute_emotion(pt, ft)
        gemini_result = gemini_interpret(pt, ft, emotion)

        return jsonify({
            'predicted_type':pt,'confidence':float(pp.max()),'probabilities':probs,
            'features':{'duration':ft.get('duration',0),'f0_mean':ft.get('f0_mean',0),'centroid_mean':ft.get('centroid_mean',0)},
            'emotion': emotion,
            'gemini': gemini_result,
            'spec_original':so,'spec_cleaned':sc
        })
    except Exception as e:
        import traceback;traceback.print_exc();return jsonify({'error':str(e)}),500
    finally:
        if os.path.exists(fp):os.remove(fp)

@app.route('/speak',methods=['POST'])
def speak():
    try:
        import requests as req
        data = request.get_json()
        call_type = data.get('call_type','unknown')
        emotion = data.get('emotion','unknown')
        confidence = data.get('confidence',0)
        translation = data.get('translation','')
        alert = data.get('alert','')

        text = f"Elephant call classified as {call_type}, with {int(confidence)}% confidence. "
        text += f"Emotional state: {emotion}. "
        if translation:
            text += f"The elephant is saying: {translation} "
        if alert:
            text += f"Conservation alert: {alert}"

        r = req.post(
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
        audio_b64 = base64.b64encode(r.content).decode('utf-8')
        return jsonify({'audio': audio_b64})
    except Exception as e:
        print(f"  ElevenLabs: {e}")
        return jsonify({'error': str(e)}), 500

if __name__=='__main__':
    print(f"\n{'═'*60}\n  DUMBO — {best_name} + Emotion Map + Gemini + ElevenLabs\n{'═'*60}")
    print(f"   http://localhost:5000\n{'═'*60}\n")
    app.run(host='0.0.0.0',port=5000,debug=False)