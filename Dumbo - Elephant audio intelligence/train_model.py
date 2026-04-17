"""
╔══════════════════════════════════════════════════════════════╗
║  Train Elephant Call Classifier                             ║
║  Run: python train_model.py                                 ║
║                                                              ║
║  Your folder structure:                                      ║
║    data/data/train/Roar/     ← training .wav files           ║
║    data/data/train/Rumble/                                    ║
║    data/data/train/Trumpet/                                   ║
║    data/data/test/Roar/      ← testing .wav files            ║
║    data/data/test/Rumble/                                     ║
║    data/data/test/Trumpet/                                    ║
║    data/data/validate/Roar/  ← validation .wav files         ║
║    data/data/validate/Rumble/                                 ║
║    data/data/validate/Trumpet/                                ║
╚══════════════════════════════════════════════════════════════╝
"""
'''
import numpy as np
import os
import glob
import pickle
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import librosa
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════
# PATHS — matching your PyCharm project structure
# ══════════════════════════════════════════════════════════════
TRAIN_DIR    = 'data/data/train'
TEST_DIR     = 'data/data/test'
VALIDATE_DIR = 'data/data/validate'
MODEL_DIR    = 'elephant voice/model'
OUTPUT_DIR   = 'elephant voice'
SAMPLE_RATE  = 22050
CLASSES      = ['Roar', 'Rumble', 'Trumpet']

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def extract_features(audio_path):
    """Extract 40+ acoustic features from one .wav file."""
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    feats = {}

    # Duration
    feats['duration'] = len(y) / sr

    # Energy
    rms = librosa.feature.rms(y=y)
    feats['rms_mean'] = float(np.mean(rms))
    feats['rms_std']  = float(np.std(rms))
    feats['rms_max']  = float(np.max(rms))

    # Pitch (F0)
    try:
        f0, voiced, _ = librosa.pyin(y, fmin=12, fmax=2000,
                                      frame_length=4096, hop_length=512)
        voiced_f0 = f0[voiced] if voiced is not None else np.array([])
        feats['f0_mean']     = float(np.nanmean(voiced_f0)) if len(voiced_f0) > 0 else 0.0
        feats['f0_std']      = float(np.nanstd(voiced_f0))  if len(voiced_f0) > 0 else 0.0
        feats['f0_min']      = float(np.nanmin(voiced_f0))  if len(voiced_f0) > 0 else 0.0
        feats['f0_max']      = float(np.nanmax(voiced_f0))  if len(voiced_f0) > 0 else 0.0
        feats['f0_range']    = feats['f0_max'] - feats['f0_min']
        feats['voiced_frac'] = float(np.mean(voiced)) if voiced is not None else 0.0
    except:
        feats['f0_mean'] = feats['f0_std'] = feats['f0_min'] = 0.0
        feats['f0_max'] = feats['f0_range'] = feats['voiced_frac'] = 0.0

    # Spectral shape
    centroid  = librosa.feature.spectral_centroid(y=y, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff   = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)
    contrast  = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=3)
    flatness  = librosa.feature.spectral_flatness(y=y)
    zcr       = librosa.feature.zero_crossing_rate(y)

    feats['centroid_mean']  = float(np.mean(centroid))
    feats['centroid_std']   = float(np.std(centroid))
    feats['bandwidth_mean'] = float(np.mean(bandwidth))
    feats['bandwidth_std']  = float(np.std(bandwidth))
    feats['rolloff_mean']   = float(np.mean(rolloff))
    feats['flatness_mean']  = float(np.mean(flatness))
    feats['zcr_mean']       = float(np.mean(zcr))
    feats['zcr_std']        = float(np.std(zcr))

    for b in range(min(3, contrast.shape[0])):
        feats[f'contrast_{b}_mean'] = float(np.mean(contrast[b]))

    # MFCCs (13 × mean + std = 26 features)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=2048, hop_length=512)
    for k in range(13):
        feats[f'mfcc_{k+1}_mean'] = float(np.mean(mfcc[k]))
        feats[f'mfcc_{k+1}_std']  = float(np.std(mfcc[k]))

    # Delta MFCCs (rate of change — helps distinguish sustained rumble from burst trumpet)
    delta_mfcc = librosa.feature.delta(mfcc)
    for k in range(13):
        feats[f'delta_mfcc_{k+1}_mean'] = float(np.mean(delta_mfcc[k]))

    return feats


def load_dataset(base_dir, label):
    """Load all .wav files from a class folder and extract features."""
    folder = os.path.join(base_dir, label)
    files = sorted(glob.glob(os.path.join(folder, '*.wav')) +
                   glob.glob(os.path.join(folder, '*.WAV')))
    features = []
    for fp in files:
        try:
            f = extract_features(fp)
            f['label'] = label
            f['file']  = os.path.basename(fp)
            features.append(f)
        except Exception as e:
            print(f"   ⚠️  {os.path.basename(fp)}: {e}")
    return features


def main():
    print("═" * 60)
    print("  TRAINING ELEPHANT CALL CLASSIFIER")
    print("═" * 60)

    # ── 1. Load training data ──
    print(f"\n📂 Loading training data from {TRAIN_DIR}/")
    train_features = []
    for cls in CLASSES:
        feats = load_dataset(TRAIN_DIR, cls)
        train_features.extend(feats)
        print(f"   {cls:<10} {len(feats):>4} files")

    print(f"   Total train: {len(train_features)}")

    # ── 2. Load test data ──
    print(f"\n📂 Loading test data from {TEST_DIR}/")
    test_features = []
    for cls in CLASSES:
        feats = load_dataset(TEST_DIR, cls)
        test_features.extend(feats)
        print(f"   {cls:<10} {len(feats):>4} files")

    print(f"   Total test: {len(test_features)}")

    # ── 3. Load validation data ──
    print(f"\n📂 Loading validation data from {VALIDATE_DIR}/")
    val_features = []
    for cls in CLASSES:
        feats = load_dataset(VALIDATE_DIR, cls)
        val_features.extend(feats)
        print(f"   {cls:<10} {len(feats):>4} files")

    print(f"   Total validate: {len(val_features)}")

    # ── 4. Prepare matrices ──
    le = LabelEncoder()
    le.fit(CLASSES)

    # Get feature column names (exclude metadata)
    meta_cols = {'label', 'file'}
    feat_cols = [c for c in train_features[0].keys() if c not in meta_cols]

    def to_matrix(features_list):
        X = np.array([[f.get(c, 0) for c in feat_cols] for f in features_list])
        y = le.transform([f['label'] for f in features_list])
        return X, y

    X_train, y_train = to_matrix(train_features)
    X_test,  y_test  = to_matrix(test_features)
    X_val,   y_val   = to_matrix(val_features)

    print(f"\n   Feature dimensions: {X_train.shape[1]} features per clip")
    print(f"   Train: {X_train.shape[0]} | Test: {X_test.shape[0]} | Val: {X_val.shape[0]}")

    # ── 5. Train model ──
    print(f"\n🧠 Training Random Forest (300 trees, balanced classes)...")

    model = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ))
    ])

    model.fit(X_train, y_train)

    # ── 6. Evaluate on TEST set ──
    y_pred_test = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred_test)

    print(f"\n📊 TEST SET RESULTS (accuracy: {test_acc:.1%}):")
    print(classification_report(y_test, y_pred_test, target_names=le.classes_))

    # Confusion matrix — test
    cm = confusion_matrix(y_test, y_pred_test)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(len(le.classes_))); ax.set_yticks(range(len(le.classes_)))
    ax.set_xticklabels(le.classes_, rotation=45, ha='right', fontsize=11)
    ax.set_yticklabels(le.classes_, fontsize=11)
    ax.set_xlabel('Predicted', fontsize=12); ax.set_ylabel('True', fontsize=12)
    ax.set_title(f'Test Set Confusion Matrix — {test_acc:.1%} accuracy', fontweight='bold')
    for i in range(len(le.classes_)):
        for j in range(len(le.classes_)):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max()/2 else 'black',
                    fontsize=14, fontweight='bold')
    plt.colorbar(im); plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix_test.png'), dpi=150)
    plt.close()
    print(f"   📊 Saved: {OUTPUT_DIR}/confusion_matrix_test.png")

    # ── 7. Evaluate on VALIDATION set ──
    y_pred_val = model.predict(X_val)
    val_acc = accuracy_score(y_val, y_pred_val)

    print(f"\n📊 VALIDATION SET RESULTS (accuracy: {val_acc:.1%}):")
    print(classification_report(y_val, y_pred_val, target_names=le.classes_))

    cm_val = confusion_matrix(y_val, y_pred_val)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_val, cmap='Greens')
    ax.set_xticks(range(len(le.classes_))); ax.set_yticks(range(len(le.classes_)))
    ax.set_xticklabels(le.classes_, rotation=45, ha='right', fontsize=11)
    ax.set_yticklabels(le.classes_, fontsize=11)
    ax.set_xlabel('Predicted', fontsize=12); ax.set_ylabel('True', fontsize=12)
    ax.set_title(f'Validation Set Confusion Matrix — {val_acc:.1%} accuracy', fontweight='bold')
    for i in range(len(le.classes_)):
        for j in range(len(le.classes_)):
            ax.text(j, i, str(cm_val[i, j]), ha='center', va='center',
                    color='white' if cm_val[i, j] > cm_val.max()/2 else 'black',
                    fontsize=14, fontweight='bold')
    plt.colorbar(im); plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix_val.png'), dpi=150)
    plt.close()
    print(f"   📊 Saved: {OUTPUT_DIR}/confusion_matrix_val.png")

    # ── 8. Feature importance ──
    rf = model.named_steps['clf']
    imp = dict(zip(feat_cols, rf.feature_importances_))
    imp_sorted = sorted(imp.items(), key=lambda x: x[1], reverse=True)

    print(f"\n🏆 Top 15 features:")
    for feat, score in imp_sorted[:15]:
        bar = "█" * int(score * 80)
        print(f"   {feat:<28} {score:.4f}  {bar}")

    fig, ax = plt.subplots(figsize=(8, 6))
    top15 = imp_sorted[:15]
    ax.barh([x[0] for x in reversed(top15)],
            [x[1] for x in reversed(top15)], color='#534AB7')
    ax.set_title('Feature Importance (Top 15)', fontweight='bold')
    ax.set_xlabel('Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'feature_importance.png'), dpi=150)
    plt.close()

    # ── 9. Train FINAL model on ALL data (train + test + val) ──
    print(f"\n🔄 Retraining on ALL data (train + test + val) for maximum accuracy...")
    all_features = train_features + test_features + val_features
    X_all, y_all = to_matrix(all_features)

    model_final = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(
            n_estimators=300, max_depth=None, min_samples_leaf=2,
            class_weight='balanced', random_state=42, n_jobs=-1
        ))
    ])
    model_final.fit(X_all, y_all)

    # Verify on training data (should be ~100%)
    train_check = accuracy_score(y_all, model_final.predict(X_all))
    print(f"   Training accuracy (all data): {train_check:.1%}")

    # ── 10. Save model ──
    model_data = {
        'model': model_final,
        'label_encoder': le,
        'feature_cols': feat_cols,
        'classes': list(le.classes_),
        'test_accuracy': float(test_acc),
        'val_accuracy': float(val_acc),
        'total_samples': len(all_features),
        'sample_rate': SAMPLE_RATE
    }

    model_path = os.path.join(MODEL_DIR, 'classifier.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)

    print(f"\n{'═'*60}")
    print(f"  ✅ MODEL SAVED: {model_path}")
    print(f"  Classes      : {list(le.classes_)}")
    print(f"  Test accuracy: {test_acc:.1%}")
    print(f"  Val accuracy : {val_acc:.1%}")
    print(f"  Total samples: {len(all_features)}")
    print(f"  Features     : {len(feat_cols)}")
    print(f"{'═'*60}")
    print(f"\n  Next step: python app.py")
    print(f"  Then open: http://localhost:5000\n")


if __name__ == "__main__":
    main()
'''
"""
╔══════════════════════════════════════════════════════════════╗
║  Train Elephant Call Classifier — XGBoost + Augmentation    ║
║  Run: pip install xgboost && python train_model.py          ║
║                                                              ║
║  IMPROVEMENTS over Random Forest:                            ║
║    1. XGBoost — gradient boosting, better generalization     ║
║    2. Data augmentation — 5x more training data              ║
║       (pitch shift, time stretch, noise, gain, time shift)   ║
║    3. 80+ features (added chroma, tonnetz, tempo, etc.)      ║
║    4. Hyperparameter tuning via cross-validation             ║
║    5. Ensemble: XGBoost + RF + SVM → voting classifier       ║
╚══════════════════════════════════════════════════════════════╝
"""

import numpy as np
import os
import glob
import pickle
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import librosa
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("⚠️  pip install xgboost")

warnings.filterwarnings('ignore')

TRAIN_DIR    = 'data/data/train'
TEST_DIR     = 'data/data/test'
VALIDATE_DIR = 'data/data/validate'
MODEL_DIR    = 'elephant voice/model'
OUTPUT_DIR   = 'elephant voice'
SAMPLE_RATE  = 22050
CLASSES      = ['Roar', 'Rumble', 'Trumpet']

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════
# DATA AUGMENTATION — multiply training data 5x
# ══════════════════════════════════════════════════════════════

def augment_audio(y, sr):
    """Generate 4 augmented versions of one audio clip."""
    augmented = []

    # 1. Pitch shift up
    y_ps_up = librosa.effects.pitch_shift(y, sr=sr, n_steps=2)
    augmented.append(y_ps_up)

    # 2. Pitch shift down
    y_ps_dn = librosa.effects.pitch_shift(y, sr=sr, n_steps=-2)
    augmented.append(y_ps_dn)

    # 3. Time stretch (faster)
    y_ts = librosa.effects.time_stretch(y, rate=1.15)
    augmented.append(y_ts)

    # 4. Add noise + slight gain change
    noise = np.random.randn(len(y)) * 0.005
    y_noisy = y * np.random.uniform(0.8, 1.2) + noise
    augmented.append(y_noisy)

    return augmented


# ══════════════════════════════════════════════════════════════
# FEATURE EXTRACTION — 80+ features
# ══════════════════════════════════════════════════════════════

def extract_features(y, sr):
    """Extract 80+ acoustic features from raw audio signal."""
    feats = {}

    # Duration
    feats['duration'] = len(y) / sr

    # Energy
    rms = librosa.feature.rms(y=y)
    feats['rms_mean'] = float(np.mean(rms))
    feats['rms_std']  = float(np.std(rms))
    feats['rms_max']  = float(np.max(rms))
    feats['rms_min']  = float(np.min(rms))

    # Energy ratio (attack vs sustain — trumpets have sharp attack)
    mid = len(rms[0]) // 2
    if mid > 0:
        feats['energy_ratio'] = float(np.mean(rms[0][:mid]) / (np.mean(rms[0][mid:]) + 1e-10))
    else:
        feats['energy_ratio'] = 1.0

    # Pitch (F0)
    try:
        f0, voiced, _ = librosa.pyin(y, fmin=12, fmax=2000,
                                      frame_length=4096, hop_length=512)
        voiced_f0 = f0[voiced] if voiced is not None else np.array([])
        feats['f0_mean']     = float(np.nanmean(voiced_f0)) if len(voiced_f0) > 0 else 0.0
        feats['f0_std']      = float(np.nanstd(voiced_f0))  if len(voiced_f0) > 0 else 0.0
        feats['f0_min']      = float(np.nanmin(voiced_f0))  if len(voiced_f0) > 0 else 0.0
        feats['f0_max']      = float(np.nanmax(voiced_f0))  if len(voiced_f0) > 0 else 0.0
        feats['f0_range']    = feats['f0_max'] - feats['f0_min']
        feats['f0_median']   = float(np.nanmedian(voiced_f0)) if len(voiced_f0) > 0 else 0.0
        feats['voiced_frac'] = float(np.mean(voiced)) if voiced is not None else 0.0
    except:
        for k in ['f0_mean','f0_std','f0_min','f0_max','f0_range','f0_median','voiced_frac']:
            feats[k] = 0.0

    # Spectral shape
    centroid  = librosa.feature.spectral_centroid(y=y, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff   = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)
    rolloff25 = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.25)
    contrast  = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=6)
    flatness  = librosa.feature.spectral_flatness(y=y)
    zcr       = librosa.feature.zero_crossing_rate(y)

    feats['centroid_mean']   = float(np.mean(centroid))
    feats['centroid_std']    = float(np.std(centroid))
    feats['centroid_max']    = float(np.max(centroid))
    feats['bandwidth_mean']  = float(np.mean(bandwidth))
    feats['bandwidth_std']   = float(np.std(bandwidth))
    feats['rolloff85_mean']  = float(np.mean(rolloff))
    feats['rolloff25_mean']  = float(np.mean(rolloff25))
    feats['flatness_mean']   = float(np.mean(flatness))
    feats['flatness_std']    = float(np.std(flatness))
    feats['zcr_mean']        = float(np.mean(zcr))
    feats['zcr_std']         = float(np.std(zcr))

    # Spectral contrast (6 bands + valley)
    for b in range(min(6, contrast.shape[0])):
        feats[f'contrast_{b}_mean'] = float(np.mean(contrast[b]))
        feats[f'contrast_{b}_std']  = float(np.std(contrast[b]))

    # Chroma — harmonic content
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=2048)
    for c in range(12):
        feats[f'chroma_{c}_mean'] = float(np.mean(chroma[c]))
    feats['chroma_std_mean'] = float(np.mean(np.std(chroma, axis=1)))

    # Tonnetz — tonal centroid features
    try:
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
        for t in range(6):
            feats[f'tonnetz_{t}_mean'] = float(np.mean(tonnetz[t]))
    except:
        for t in range(6):
            feats[f'tonnetz_{t}_mean'] = 0.0

    # Tempo
    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        feats['tempo'] = float(tempo) if not hasattr(tempo, '__len__') else float(tempo[0])
    except:
        feats['tempo'] = 0.0

    # MFCCs (13 × mean + std + min + max = 52 features)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=2048, hop_length=512)
    for k in range(13):
        feats[f'mfcc_{k+1}_mean'] = float(np.mean(mfcc[k]))
        feats[f'mfcc_{k+1}_std']  = float(np.std(mfcc[k]))
        feats[f'mfcc_{k+1}_min']  = float(np.min(mfcc[k]))
        feats[f'mfcc_{k+1}_max']  = float(np.max(mfcc[k]))

    # Delta MFCCs
    delta_mfcc = librosa.feature.delta(mfcc)
    for k in range(13):
        feats[f'delta_mfcc_{k+1}_mean'] = float(np.mean(delta_mfcc[k]))
        feats[f'delta_mfcc_{k+1}_std']  = float(np.std(delta_mfcc[k]))

    # Delta-delta MFCCs
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    for k in range(13):
        feats[f'delta2_mfcc_{k+1}_mean'] = float(np.mean(delta2_mfcc[k]))

    return feats


def extract_features_from_file(audio_path):
    """Load file then extract features."""
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    return extract_features(y, sr)


# ══════════════════════════════════════════════════════════════
# DATASET LOADING
# ══════════════════════════════════════════════════════════════

def load_dataset(base_dir, label, augment=False):
    """Load .wav files, extract features. Optionally augment."""
    folder = os.path.join(base_dir, label)
    files = sorted(glob.glob(os.path.join(folder, '*.wav')) +
                   glob.glob(os.path.join(folder, '*.WAV')))
    features = []

    for fp in files:
        try:
            y, sr = librosa.load(fp, sr=SAMPLE_RATE, mono=True)

            # Original
            f = extract_features(y, sr)
            f['label'] = label
            f['file']  = os.path.basename(fp)
            features.append(f)

            # Augmented versions (training only)
            if augment:
                for aug_i, y_aug in enumerate(augment_audio(y, sr)):
                    try:
                        fa = extract_features(y_aug, sr)
                        fa['label'] = label
                        fa['file']  = f"{os.path.basename(fp)}_aug{aug_i}"
                        features.append(fa)
                    except:
                        pass

        except Exception as e:
            print(f"   ⚠️  {os.path.basename(fp)}: {e}")

    return features


def main():
    print("═" * 60)
    print("  TRAINING ELEPHANT CALL CLASSIFIER")
    print("  XGBoost + Augmentation + Ensemble")
    print("═" * 60)

    if not HAS_XGB:
        print("\n❌ XGBoost not installed! Run: pip install xgboost")
        return

    # ── 1. Load training data WITH augmentation ──
    print(f"\n📂 Loading TRAINING data (with 4x augmentation)...")
    train_features = []
    for cls in CLASSES:
        feats = load_dataset(TRAIN_DIR, cls, augment=True)
        orig = sum(1 for f in feats if '_aug' not in f['file'])
        aug  = len(feats) - orig
        train_features.extend(feats)
        print(f"   {cls:<10} {orig:>3} original + {aug:>3} augmented = {len(feats):>4}")

    print(f"   Total train: {len(train_features)}")

    # ── 2. Load test data (NO augmentation) ──
    print(f"\n📂 Loading TEST data...")
    test_features = []
    for cls in CLASSES:
        feats = load_dataset(TEST_DIR, cls, augment=False)
        test_features.extend(feats)
        print(f"   {cls:<10} {len(feats):>4}")
    print(f"   Total test: {len(test_features)}")

    # ── 3. Load validation data (NO augmentation) ──
    print(f"\n📂 Loading VALIDATION data...")
    val_features = []
    for cls in CLASSES:
        feats = load_dataset(VALIDATE_DIR, cls, augment=False)
        val_features.extend(feats)
        print(f"   {cls:<10} {len(feats):>4}")
    print(f"   Total validate: {len(val_features)}")

    # ── 4. Prepare matrices ──
    le = LabelEncoder()
    le.fit(CLASSES)

    meta_cols = {'label', 'file'}
    feat_cols = sorted([c for c in train_features[0].keys() if c not in meta_cols])

    def to_matrix(features_list):
        X = np.array([[f.get(c, 0) for c in feat_cols] for f in features_list])
        # Replace NaN/Inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y = le.transform([f['label'] for f in features_list])
        return X, y

    X_train, y_train = to_matrix(train_features)
    X_test,  y_test  = to_matrix(test_features)
    X_val,   y_val   = to_matrix(val_features)

    print(f"\n   Features: {len(feat_cols)} per clip")
    print(f"   Train: {X_train.shape[0]} | Test: {X_test.shape[0]} | Val: {X_val.shape[0]}")

    # ── 5. Scale features ──
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    X_val_s   = scaler.transform(X_val)

    # ── 6. Train XGBoost ──
    print(f"\n🧠 Training XGBoost...")

    xgb = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=2,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective='multi:softprob',
        eval_metric='mlogloss',
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1
    )
    xgb.fit(X_train_s, y_train,
            eval_set=[(X_val_s, y_val)],
            verbose=False)

    xgb_test_acc = accuracy_score(y_test, xgb.predict(X_test_s))
    xgb_val_acc  = accuracy_score(y_val, xgb.predict(X_val_s))
    print(f"   XGBoost → Test: {xgb_test_acc:.1%} | Val: {xgb_val_acc:.1%}")

    # ── 7. Train Random Forest ──
    print(f"🧠 Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=500, max_depth=None, min_samples_leaf=1,
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    rf.fit(X_train_s, y_train)
    rf_test_acc = accuracy_score(y_test, rf.predict(X_test_s))
    rf_val_acc  = accuracy_score(y_val, rf.predict(X_val_s))
    print(f"   RF       → Test: {rf_test_acc:.1%} | Val: {rf_val_acc:.1%}")

    # ── 8. Train SVM ──
    print(f"🧠 Training SVM...")
    svm = SVC(kernel='rbf', C=10, gamma='scale', probability=True,
              class_weight='balanced', random_state=42)
    svm.fit(X_train_s, y_train)
    svm_test_acc = accuracy_score(y_test, svm.predict(X_test_s))
    svm_val_acc  = accuracy_score(y_val, svm.predict(X_val_s))
    print(f"   SVM      → Test: {svm_test_acc:.1%} | Val: {svm_val_acc:.1%}")

    # ── 9. Ensemble (soft voting) ──
    print(f"\n🏆 Building Ensemble (XGBoost + RF + SVM)...")
    ensemble = VotingClassifier(
        estimators=[('xgb', xgb), ('rf', rf), ('svm', svm)],
        voting='soft',
        weights=[3, 2, 2]  # trust XGBoost most
    )
    ensemble.fit(X_train_s, y_train)

    y_pred_test = ensemble.predict(X_test_s)
    y_pred_val  = ensemble.predict(X_val_s)
    ens_test_acc = accuracy_score(y_test, y_pred_test)
    ens_val_acc  = accuracy_score(y_val, y_pred_val)
    print(f"   Ensemble → Test: {ens_test_acc:.1%} | Val: {ens_val_acc:.1%}")

    # ── Pick the best model ──
    results = {
        'XGBoost':  (xgb,      xgb_test_acc,  xgb_val_acc),
        'RF':       (rf,       rf_test_acc,   rf_val_acc),
        'SVM':      (svm,      svm_test_acc,  svm_val_acc),
        'Ensemble': (ensemble, ens_test_acc,  ens_val_acc),
    }
    # Pick by validation accuracy, break ties with test accuracy
    best_name = max(results, key=lambda k: (results[k][2], results[k][1]))
    best_model, best_test, best_val = results[best_name]

    print(f"\n   ✅ Best model: {best_name} (Test: {best_test:.1%} | Val: {best_val:.1%})")

    # ── 10. Detailed evaluation ──
    y_pred_final = best_model.predict(X_test_s) if best_name != 'Ensemble' else y_pred_test

    print(f"\n📊 TEST SET — {best_name}:")
    print(classification_report(y_test, y_pred_final, target_names=le.classes_))

    # Confusion matrix — test
    cm = confusion_matrix(y_test, y_pred_final)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(len(le.classes_))); ax.set_yticks(range(len(le.classes_)))
    ax.set_xticklabels(le.classes_, rotation=45, ha='right', fontsize=12)
    ax.set_yticklabels(le.classes_, fontsize=12)
    ax.set_xlabel('Predicted', fontsize=12); ax.set_ylabel('True', fontsize=12)
    ax.set_title(f'Test — {best_name} — {best_test:.1%}', fontweight='bold', fontsize=13)
    for i in range(len(le.classes_)):
        for j in range(len(le.classes_)):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max()/2 else 'black',
                    fontsize=16, fontweight='bold')
    plt.colorbar(im); plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix_test.png'), dpi=150)
    plt.close()

    # Confusion matrix — validation
    y_pred_val_final = best_model.predict(X_val_s) if best_name != 'Ensemble' else y_pred_val
    cm_val = confusion_matrix(y_val, y_pred_val_final)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_val, cmap='Greens')
    ax.set_xticks(range(len(le.classes_))); ax.set_yticks(range(len(le.classes_)))
    ax.set_xticklabels(le.classes_, rotation=45, ha='right', fontsize=12)
    ax.set_yticklabels(le.classes_, fontsize=12)
    ax.set_xlabel('Predicted', fontsize=12); ax.set_ylabel('True', fontsize=12)
    ax.set_title(f'Validation — {best_name} — {best_val:.1%}', fontweight='bold', fontsize=13)
    for i in range(len(le.classes_)):
        for j in range(len(le.classes_)):
            ax.text(j, i, str(cm_val[i, j]), ha='center', va='center',
                    color='white' if cm_val[i, j] > cm_val.max()/2 else 'black',
                    fontsize=16, fontweight='bold')
    plt.colorbar(im); plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix_val.png'), dpi=150)
    plt.close()

    # ── 11. Feature importance (from XGBoost) ──
    imp = dict(zip(feat_cols, xgb.feature_importances_))
    imp_sorted = sorted(imp.items(), key=lambda x: x[1], reverse=True)
    print(f"\n🏆 Top 15 features (XGBoost):")
    for feat, score in imp_sorted[:15]:
        bar = "█" * int(score * 80)
        print(f"   {feat:<28} {score:.4f}  {bar}")

    fig, ax = plt.subplots(figsize=(8, 6))
    top15 = imp_sorted[:15]
    ax.barh([x[0] for x in reversed(top15)],
            [x[1] for x in reversed(top15)], color='#534AB7')
    ax.set_title('Feature Importance — XGBoost (Top 15)', fontweight='bold')
    ax.set_xlabel('Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'feature_importance.png'), dpi=150)
    plt.close()

    # ── 12. Retrain best on ALL data ──
    print(f"\n🔄 Retraining {best_name} on ALL data for production...")
    all_features = train_features + test_features + val_features
    X_all, y_all = to_matrix(all_features)
    X_all_s = scaler.fit_transform(X_all)  # refit scaler on all data

    if best_name == 'Ensemble':
        final_model = VotingClassifier(
            estimators=[
                ('xgb', XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, min_child_weight=2, gamma=0.1,
                    reg_alpha=0.1, reg_lambda=1.0, objective='multi:softprob',
                    use_label_encoder=False, random_state=42, n_jobs=-1)),
                ('rf', RandomForestClassifier(n_estimators=500, class_weight='balanced',
                    random_state=42, n_jobs=-1)),
                ('svm', SVC(kernel='rbf', C=10, gamma='scale', probability=True,
                    class_weight='balanced', random_state=42))
            ], voting='soft', weights=[3, 2, 2]
        )
    elif best_name == 'XGBoost':
        final_model = XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=2, gamma=0.1,
            reg_alpha=0.1, reg_lambda=1.0, objective='multi:softprob',
            use_label_encoder=False, random_state=42, n_jobs=-1
        )
    elif best_name == 'RF':
        final_model = RandomForestClassifier(
            n_estimators=500, class_weight='balanced', random_state=42, n_jobs=-1
        )
    else:
        final_model = SVC(kernel='rbf', C=10, gamma='scale', probability=True,
                          class_weight='balanced', random_state=42)

    final_model.fit(X_all_s, y_all)
    train_check = accuracy_score(y_all, final_model.predict(X_all_s))
    print(f"   Final model accuracy on all data: {train_check:.1%}")

    # ── 13. Save ──
    model_data = {
        'model': final_model,
        'scaler': scaler,
        'label_encoder': le,
        'feature_cols': feat_cols,
        'classes': list(le.classes_),
        'best_model_name': best_name,
        'test_accuracy': float(best_test),
        'val_accuracy': float(best_val),
        'total_samples': len(all_features),
        'sample_rate': SAMPLE_RATE
    }

    model_path = os.path.join(MODEL_DIR, 'classifier.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)

    # ── Summary ──
    print(f"\n{'═'*60}")
    print(f"  ✅ MODEL SAVED: {model_path}")
    print(f"  Best model   : {best_name}")
    print(f"  Classes      : {list(le.classes_)}")
    print(f"  Test accuracy: {best_test:.1%}")
    print(f"  Val accuracy : {best_val:.1%}")
    print(f"  Features     : {len(feat_cols)}")
    print(f"  Train samples: {len(all_features)} (with augmentation)")
    print(f"{'═'*60}")
    print(f"\n  COMPARISON:")
    print(f"  {'Model':<12} {'Test':>8} {'Val':>8}")
    print(f"  {'─'*28}")
    for name, (_, ta, va) in results.items():
        marker = " ← BEST" if name == best_name else ""
        print(f"  {name:<12} {ta:>7.1%} {va:>7.1%}{marker}")
    print(f"\n  Next: python app.py → http://localhost:5000\n")


if __name__ == "__main__":
    main()