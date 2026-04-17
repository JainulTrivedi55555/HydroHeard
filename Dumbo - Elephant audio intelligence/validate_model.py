"""
╔══════════════════════════════════════════════════════════════╗
║  RUMBLR — Complete Validation & Metrics Framework           ║
║                                                              ║
║  Run AFTER train_model.py:                                   ║
║    python validate_model.py                                  ║
║                                                              ║
║  What it does:                                               ║
║    1. Loads train/test/validate data from your folders        ║
║    2. Trains BOTH Random Forest and XGBoost                  ║
║    3. Runs 7 validation checks per model with PASS/WARN/FAIL ║
║    4. Side-by-side comparison table                          ║
║    5. Ensemble agreement check                               ║
║    6. Generates confusion matrix + comparison charts          ║
║    7. Final GO / NO-GO deployment verdict                    ║
║    8. Saves production model with dual-model ensemble        ║
║                                                              ║
║  Your folder structure:                                      ║
║    data/data/train/Roar/     data/data/test/Roar/            ║
║    data/data/train/Rumble/   data/data/test/Rumble/          ║
║    data/data/train/Trumpet/  data/data/test/Trumpet/         ║
║    data/data/validate/Roar/  etc.                            ║
╚══════════════════════════════════════════════════════════════╝
"""

import numpy as np
import os
import glob
import pickle
import time
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import librosa
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, cohen_kappa_score, brier_score_loss,
    accuracy_score
)
from sklearn.svm import SVC
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════
# PATHS
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

# ══════════════════════════════════════════════════════════════
# METRIC THRESHOLDS — pass/warn/fail gates
# ══════════════════════════════════════════════════════════════
THRESHOLDS = {
    'cv_accuracy_pass'    : 0.82,
    'cv_accuracy_warn'    : 0.70,
    'cv_std_pass'         : 0.05,
    'cv_std_warn'         : 0.10,
    'macro_f1_pass'       : 0.80,
    'macro_f1_warn'       : 0.65,
    'kappa_pass'          : 0.75,
    'kappa_warn'          : 0.60,
    'overfit_gap_pass'    : 0.05,
    'overfit_gap_warn'    : 0.15,
    'conservation_recall' : 0.75,   # recall for Roar and Trumpet
    'brier_pass'          : 0.15,
    'confidence_pass'     : 0.75,   # avg confidence should be > 75%
}

CONSERVATION_CLASSES = ['Roar', 'Trumpet']  # these matter most scientifically

# Color codes for terminal
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
RESET  = "\033[0m"
BOLD   = "\033[1m"


def verdict(value, pass_thresh, warn_thresh, higher_is_better=True):
    if higher_is_better:
        if value >= pass_thresh: return "PASS", GREEN
        if value >= warn_thresh: return "WARN", YELLOW
        return "FAIL", RED
    else:
        if value <= pass_thresh: return "PASS", GREEN
        if value <= warn_thresh: return "WARN", YELLOW
        return "FAIL", RED


# ══════════════════════════════════════════════════════════════
# FEATURE EXTRACTION (must match train_model.py + app.py)
# ══════════════════════════════════════════════════════════════

def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    f = {}
    f['duration'] = len(y) / sr

    rms = librosa.feature.rms(y=y)
    f['rms_mean'] = float(np.mean(rms)); f['rms_std'] = float(np.std(rms))
    f['rms_max'] = float(np.max(rms)); f['rms_min'] = float(np.min(rms))
    mid = len(rms[0]) // 2
    f['energy_ratio'] = float(np.mean(rms[0][:mid]) / (np.mean(rms[0][mid:]) + 1e-10)) if mid > 0 else 1.0

    try:
        f0, voiced, _ = librosa.pyin(y, fmin=12, fmax=2000, frame_length=4096, hop_length=512)
        vf = f0[voiced] if voiced is not None else np.array([])
        f['f0_mean']     = float(np.nanmean(vf)) if len(vf) > 0 else 0.0
        f['f0_std']      = float(np.nanstd(vf))  if len(vf) > 0 else 0.0
        f['f0_min']      = float(np.nanmin(vf))  if len(vf) > 0 else 0.0
        f['f0_max']      = float(np.nanmax(vf))  if len(vf) > 0 else 0.0
        f['f0_range']    = f['f0_max'] - f['f0_min']
        f['f0_median']   = float(np.nanmedian(vf)) if len(vf) > 0 else 0.0
        f['voiced_frac'] = float(np.mean(voiced)) if voiced is not None else 0.0
    except:
        for k in ['f0_mean','f0_std','f0_min','f0_max','f0_range','f0_median','voiced_frac']:
            f[k] = 0.0

    ce = librosa.feature.spectral_centroid(y=y, sr=sr)
    bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    ro = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)
    r25 = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.25)
    co = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=6)
    fl = librosa.feature.spectral_flatness(y=y)
    zc = librosa.feature.zero_crossing_rate(y)

    f['centroid_mean'] = float(np.mean(ce)); f['centroid_std'] = float(np.std(ce))
    f['centroid_max'] = float(np.max(ce))
    f['bandwidth_mean'] = float(np.mean(bw)); f['bandwidth_std'] = float(np.std(bw))
    f['rolloff85_mean'] = float(np.mean(ro)); f['rolloff25_mean'] = float(np.mean(r25))
    f['flatness_mean'] = float(np.mean(fl)); f['flatness_std'] = float(np.std(fl))
    f['zcr_mean'] = float(np.mean(zc)); f['zcr_std'] = float(np.std(zc))

    for b in range(min(6, co.shape[0])):
        f[f'contrast_{b}_mean'] = float(np.mean(co[b]))
        f[f'contrast_{b}_std'] = float(np.std(co[b]))

    ch = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=2048)
    for c in range(12):
        f[f'chroma_{c}_mean'] = float(np.mean(ch[c]))
    f['chroma_std_mean'] = float(np.mean(np.std(ch, axis=1)))

    try:
        tn = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
        for t in range(6):
            f[f'tonnetz_{t}_mean'] = float(np.mean(tn[t]))
    except:
        for t in range(6):
            f[f'tonnetz_{t}_mean'] = 0.0

    try:
        tp, _ = librosa.beat.beat_track(y=y, sr=sr)
        f['tempo'] = float(tp) if not hasattr(tp, '__len__') else float(tp[0])
    except:
        f['tempo'] = 0.0

    mc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=2048, hop_length=512)
    for k in range(13):
        f[f'mfcc_{k+1}_mean'] = float(np.mean(mc[k]))
        f[f'mfcc_{k+1}_std'] = float(np.std(mc[k]))
        f[f'mfcc_{k+1}_min'] = float(np.min(mc[k]))
        f[f'mfcc_{k+1}_max'] = float(np.max(mc[k]))

    dm = librosa.feature.delta(mc)
    for k in range(13):
        f[f'delta_mfcc_{k+1}_mean'] = float(np.mean(dm[k]))
        f[f'delta_mfcc_{k+1}_std'] = float(np.std(dm[k]))

    d2 = librosa.feature.delta(mc, order=2)
    for k in range(13):
        f[f'delta2_mfcc_{k+1}_mean'] = float(np.mean(d2[k]))

    return f


def load_dataset(base_dir):
    """Load all .wav files from train/test/validate folders."""
    features = []
    for cls in CLASSES:
        folder = os.path.join(base_dir, cls)
        files = sorted(glob.glob(os.path.join(folder, '*.wav')) +
                       glob.glob(os.path.join(folder, '*.WAV')))
        for fp in files:
            try:
                feat = extract_features(fp)
                feat['label'] = cls
                feat['file'] = os.path.basename(fp)
                features.append(feat)
            except Exception as e:
                print(f"   ⚠️  {os.path.basename(fp)}: {e}")
    return features


def augment_and_extract(base_dir):
    """Load training data with 4x augmentation."""
    features = []
    for cls in CLASSES:
        folder = os.path.join(base_dir, cls)
        files = sorted(glob.glob(os.path.join(folder, '*.wav')) +
                       glob.glob(os.path.join(folder, '*.WAV')))
        for fp in files:
            try:
                y, sr = librosa.load(fp, sr=SAMPLE_RATE, mono=True)

                # Original
                feat = extract_features(fp)
                feat['label'] = cls
                feat['file'] = os.path.basename(fp)
                features.append(feat)

                # Augmentations
                augments = [
                    librosa.effects.pitch_shift(y, sr=sr, n_steps=2),
                    librosa.effects.pitch_shift(y, sr=sr, n_steps=-2),
                    librosa.effects.time_stretch(y, rate=1.15),
                    y * np.random.uniform(0.8, 1.2) + np.random.randn(len(y)) * 0.005,
                ]
                for ai, ya in enumerate(augments):
                    try:
                        # Write temp, extract, delete
                        tmp = f'/tmp/aug_{cls}_{os.path.basename(fp)}_{ai}.wav'
                        import soundfile as sf
                        sf.write(tmp, ya, sr)
                        fa = extract_features(tmp)
                        fa['label'] = cls
                        fa['file'] = f"{os.path.basename(fp)}_aug{ai}"
                        features.append(fa)
                        os.remove(tmp)
                    except:
                        pass
            except Exception as e:
                print(f"   ⚠️  {os.path.basename(fp)}: {e}")
    return features


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    print("═" * 60)
    print(f"  {BOLD}RUMBLR — Complete Validation Framework{RESET}")
    print("═" * 60)

    # ── Load data ──
    print(f"\n📂 Loading TRAINING data (with augmentation)...")
    train_feats = augment_and_extract(TRAIN_DIR)
    for cls in CLASSES:
        n = sum(1 for f in train_feats if f['label'] == cls)
        print(f"   {cls:<10} {n:>4} (with augmentation)")
    print(f"   Total: {len(train_feats)}")

    print(f"\n📂 Loading TEST data...")
    test_feats = load_dataset(TEST_DIR)
    for cls in CLASSES:
        n = sum(1 for f in test_feats if f['label'] == cls)
        print(f"   {cls:<10} {n:>4}")

    print(f"\n📂 Loading VALIDATION data...")
    val_feats = load_dataset(VALIDATE_DIR)
    for cls in CLASSES:
        n = sum(1 for f in val_feats if f['label'] == cls)
        print(f"   {cls:<10} {n:>4}")

    # ── Prepare matrices ──
    le = LabelEncoder()
    le.fit(CLASSES)

    meta_cols = {'label', 'file'}
    feat_cols = sorted([c for c in train_feats[0].keys() if c not in meta_cols])

    def to_matrix(feats_list):
        X = np.array([[f.get(c, 0) for c in feat_cols] for f in feats_list])
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y = le.transform([f['label'] for f in feats_list])
        return X, y

    X_train, y_train = to_matrix(train_feats)
    X_test,  y_test  = to_matrix(test_feats)
    X_val,   y_val   = to_matrix(val_feats)

    print(f"\n   Features: {len(feat_cols)} per clip")
    print(f"   Train: {X_train.shape[0]} | Test: {X_test.shape[0]} | Val: {X_val.shape[0]}")

    class_names = list(le.classes_)

    # ── Define models ──
    models = {
        'Random Forest': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(
                n_estimators=300, max_depth=15, min_samples_leaf=2,
                class_weight='balanced', random_state=42, n_jobs=-1
            ))
        ]),
        'XGBoost': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', XGBClassifier(
                n_estimators=500, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=2,
                gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
                objective='multi:softprob', use_label_encoder=False,
                eval_metric='mlogloss', random_state=42, n_jobs=-1
            ))
        ]),
    }

    results_summary = {}

    for model_name, model in models.items():
        print("\n" + "─" * 60)
        print(f"  {BOLD}MODEL: {model_name}{RESET}")
        print("─" * 60)

        # ═══════════════════════════════════════════════════════
        # [1] STRATIFIED 5-FOLD CROSS VALIDATION
        # ═══════════════════════════════════════════════════════
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        t0 = time.time()
        cv_acc = cross_val_score(model, X_train, y_train, cv=skf, scoring='accuracy', n_jobs=-1)
        cv_f1  = cross_val_score(model, X_train, y_train, cv=skf, scoring='f1_macro', n_jobs=-1)
        cv_time = time.time() - t0

        cv_acc_mean = cv_acc.mean()
        cv_acc_std  = cv_acc.std()
        cv_f1_mean  = cv_f1.mean()

        v_acc, c_acc = verdict(cv_acc_mean, THRESHOLDS['cv_accuracy_pass'], THRESHOLDS['cv_accuracy_warn'])
        v_std, c_std = verdict(cv_acc_std, THRESHOLDS['cv_std_pass'], THRESHOLDS['cv_std_warn'], False)
        v_f1, c_f1   = verdict(cv_f1_mean, THRESHOLDS['macro_f1_pass'], THRESHOLDS['macro_f1_warn'])

        print(f"\n  [1] Cross-Validation (5-fold Stratified on training data)")
        print(f"    CV Accuracy : {cv_acc_mean:.4f} ± {cv_acc_std:.4f}  {c_acc}[{v_acc}]{RESET}")
        print(f"    CV Std      : {cv_acc_std:.4f}              {c_std}[{v_std}]{RESET}")
        print(f"    Macro F1    : {cv_f1_mean:.4f}              {c_f1}[{v_f1}]{RESET}")
        print(f"    Time        : {cv_time:.1f}s")

        print(f"\n    Fold-by-fold accuracy:")
        for i, s in enumerate(cv_acc):
            bar  = "█" * int(s * 20)
            flag = " ← outlier" if abs(s - cv_acc_mean) > 2 * cv_acc_std else ""
            print(f"      Fold {i+1}: {s:.4f}  {bar}{flag}")

        # ═══════════════════════════════════════════════════════
        # [2] OVERFITTING CHECK
        # ═══════════════════════════════════════════════════════
        model.fit(X_train, y_train)
        train_acc   = accuracy_score(y_train, model.predict(X_train))
        overfit_gap = train_acc - cv_acc_mean

        v_ov, c_ov = verdict(overfit_gap, THRESHOLDS['overfit_gap_pass'], THRESHOLDS['overfit_gap_warn'], False)

        print(f"\n  [2] Overfitting Check")
        print(f"    Train accuracy : {train_acc:.4f}")
        print(f"    CV accuracy    : {cv_acc_mean:.4f}")
        print(f"    Gap            : {overfit_gap:.4f}  {c_ov}[{v_ov}]{RESET}")
        if overfit_gap > THRESHOLDS['overfit_gap_warn']:
            if model_name == 'Random Forest':
                print(f"    FIX: Reduce max_depth (try 8–12), increase min_samples_leaf")
            else:
                print(f"    FIX: Reduce learning_rate to 0.01, increase min_child_weight")

        # ═══════════════════════════════════════════════════════
        # [3] TEST SET RESULTS
        # ═══════════════════════════════════════════════════════
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        test_acc    = accuracy_score(y_test, y_pred)
        macro_f1    = f1_score(y_test, y_pred, average='macro', zero_division=0)
        weighted_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        kappa       = cohen_kappa_score(y_test, y_pred)
        cm          = confusion_matrix(y_test, y_pred)

        v_tf1, c_tf1 = verdict(macro_f1, THRESHOLDS['macro_f1_pass'], THRESHOLDS['macro_f1_warn'])
        v_kap, c_kap = verdict(kappa, THRESHOLDS['kappa_pass'], THRESHOLDS['kappa_warn'])

        print(f"\n  [3] Test Set Results (data/data/test/)")
        print(f"    Test Accuracy  : {test_acc:.4f}")
        print(f"    Macro F1       : {macro_f1:.4f}  {c_tf1}[{v_tf1}]{RESET}")
        print(f"    Weighted F1    : {weighted_f1:.4f}")
        print(f"    Cohen's Kappa  : {kappa:.4f}  {c_kap}[{v_kap}]{RESET}")

        # ═══════════════════════════════════════════════════════
        # [4] PER-CLASS REPORT + CONSERVATION CHECK
        # ═══════════════════════════════════════════════════════
        print(f"\n  [4] Per-Class Report")
        report = classification_report(y_test, y_pred, target_names=class_names,
                                        output_dict=True, zero_division=0)

        print(f"    {'Class':<15} {'Precision':>9} {'Recall':>7} {'F1':>7} {'Support':>8}")
        print(f"    {'─'*48}")
        conservation_fails = []
        for cls in class_names:
            if cls not in report:
                continue
            r = report[cls]
            prec, rec, f1v, sup = r['precision'], r['recall'], r['f1-score'], r['support']
            flag = ""
            if cls in CONSERVATION_CLASSES:
                if rec < THRESHOLDS['conservation_recall']:
                    flag = f"  ← {RED}ALERT: low recall{RESET}"
                    conservation_fails.append(cls)
                else:
                    flag = f"  ← {GREEN}OK{RESET}"
            print(f"    {cls:<15} {prec:>9.3f} {rec:>7.3f} {f1v:>7.3f} {sup:>8}  {flag}")

        if conservation_fails:
            print(f"\n    {RED}Conservation alert: {conservation_fails} have low recall.{RESET}")
            print(f"    FIX: Add more training samples for these classes or increase augmentation.")

        # ═══════════════════════════════════════════════════════
        # [5] VALIDATION SET RESULTS
        # ═══════════════════════════════════════════════════════
        y_pred_val  = model.predict(X_val)
        y_proba_val = model.predict_proba(X_val)
        val_acc     = accuracy_score(y_val, y_pred_val)
        val_f1      = f1_score(y_val, y_pred_val, average='macro', zero_division=0)
        val_kappa   = cohen_kappa_score(y_val, y_pred_val)
        cm_val      = confusion_matrix(y_val, y_pred_val)

        print(f"\n  [5] Validation Set Results (data/data/validate/)")
        print(f"    Val Accuracy   : {val_acc:.4f}")
        print(f"    Val Macro F1   : {val_f1:.4f}")
        print(f"    Val Kappa      : {val_kappa:.4f}")

        # ═══════════════════════════════════════════════════════
        # [6] MODEL CALIBRATION (Brier Score)
        # ═══════════════════════════════════════════════════════
        brier_scores = []
        for cls_i in range(len(class_names)):
            y_bin = (y_test == cls_i).astype(int)
            prob  = y_proba[:, cls_i]
            bs    = brier_score_loss(y_bin, prob)
            brier_scores.append(bs)
        avg_brier = np.mean(brier_scores)

        v_bs, c_bs = verdict(avg_brier, THRESHOLDS['brier_pass'], THRESHOLDS['brier_pass'] * 1.5, False)

        print(f"\n  [6] Calibration (Brier Score — lower is better)")
        print(f"    Avg Brier Score : {avg_brier:.4f}  {c_bs}[{v_bs}]{RESET}")
        print(f"    (0.00 = perfect | 0.25 = random guessing)")

        # ═══════════════════════════════════════════════════════
        # [7] CONFIDENCE DISTRIBUTION + INFERENCE SPEED
        # ═══════════════════════════════════════════════════════
        max_proba = y_proba.max(axis=1)
        high_conf = (max_proba >= 0.80).mean()
        low_conf  = (max_proba < 0.50).mean()
        avg_conf  = max_proba.mean()

        v_cn, c_cn = verdict(avg_conf, THRESHOLDS['confidence_pass'], 0.60)

        print(f"\n  [7] Prediction Confidence")
        print(f"    Avg confidence    : {avg_conf:.1%}  {c_cn}[{v_cn}]{RESET}")
        print(f"    High conf (≥80%) : {high_conf:.1%}")
        print(f"    Low conf  (<50%) : {low_conf:.1%}  ← flag as 'uncertain' in web app")

        # Inference speed
        _ = model.predict(X_test)  # warm up
        t0 = time.time()
        for _ in range(100):
            model.predict(X_test[:1])
        inference_ms = (time.time() - t0) / 100 * 1000

        v_inf = "PASS" if inference_ms < 100 else "WARN"
        c_inf = GREEN if v_inf == "PASS" else YELLOW
        print(f"\n    Inference Speed : {inference_ms:.2f}ms/sample  {c_inf}[{v_inf}]{RESET}")

        # Store results
        results_summary[model_name] = {
            'cv_acc_mean': cv_acc_mean, 'cv_acc_std': cv_acc_std,
            'cv_f1_mean': cv_f1_mean, 'test_acc': test_acc,
            'macro_f1': macro_f1, 'weighted_f1': weighted_f1,
            'kappa': kappa, 'overfit_gap': overfit_gap,
            'avg_brier': avg_brier, 'inference_ms': inference_ms,
            'val_acc': val_acc, 'val_f1': val_f1, 'val_kappa': val_kappa,
            'avg_conf': avg_conf,
            'y_pred': y_pred, 'y_proba': y_proba, 'cm': cm,
            'y_pred_val': y_pred_val, 'cm_val': cm_val,
            'model': model,
        }
        print()

    # ══════════════════════════════════════════════════════════
    # SIDE-BY-SIDE COMPARISON
    # ══════════════════════════════════════════════════════════
    print("\n" + "═" * 60)
    print(f"  {BOLD}FINAL MODEL COMPARISON{RESET}")
    print("═" * 60)
    print(f"  {'Metric':<28} {'Random Forest':>14} {'XGBoost':>14}")
    print(f"  {'─'*58}")

    metrics_to_compare = [
        ('CV accuracy (mean)',    'cv_acc_mean',  True),
        ('CV std (lower=better)', 'cv_acc_std',   False),
        ('CV Macro F1',           'cv_f1_mean',   True),
        ('Test accuracy',         'test_acc',     True),
        ('Test Macro F1',         'macro_f1',     True),
        ('Val accuracy',          'val_acc',      True),
        ('Val Macro F1',          'val_f1',       True),
        ("Cohen's Kappa",         'kappa',        True),
        ('Overfit gap',           'overfit_gap',  False),
        ('Brier score',           'avg_brier',    False),
        ('Avg confidence',        'avg_conf',     True),
        ('Inference (ms)',        'inference_ms', False),
    ]

    for label, key, higher_better in metrics_to_compare:
        rf_val  = results_summary['Random Forest'][key]
        xgb_val = results_summary['XGBoost'][key]
        if higher_better:
            winner = 'RF' if rf_val > xgb_val else 'XGB'
        else:
            winner = 'RF' if rf_val < xgb_val else 'XGB'
        m_rf  = " ★" if winner == 'RF'  else ""
        m_xgb = " ★" if winner == 'XGB' else ""
        print(f"  {label:<28} {rf_val:>12.4f}{m_rf:<2}  {xgb_val:>12.4f}{m_xgb:<2}")

    # ══════════════════════════════════════════════════════════
    # ENSEMBLE AGREEMENT
    # ══════════════════════════════════════════════════════════
    rf_pred  = results_summary['Random Forest']['y_pred']
    xgb_pred = results_summary['XGBoost']['y_pred']
    agreement = (rf_pred == xgb_pred).mean()

    print(f"\n  Ensemble agreement: {agreement:.1%} of test predictions match")
    if agreement >= 0.85:
        print(f"  {GREEN}PASS{RESET} — High agreement. Use XGB primary, flag if RF disagrees.")
    elif agreement >= 0.70:
        print(f"  {YELLOW}WARN{RESET} — Moderate. Flag low-agreement predictions as 'uncertain'.")
    else:
        print(f"  {RED}FAIL{RESET} — Low agreement. Check features.")

    disagreements = np.where(rf_pred != xgb_pred)[0]
    if len(disagreements) > 0:
        print(f"\n  Disagreements ({len(disagreements)} samples):")
        for idx in disagreements[:10]:
            true = class_names[y_test[idx]]
            rf_c = class_names[rf_pred[idx]]
            xg_c = class_names[xgb_pred[idx]]
            print(f"    True: {true:<12} RF: {rf_c:<12} XGB: {xg_c}")

    # ══════════════════════════════════════════════════════════
    # PLOTS
    # ══════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("RUMBLR — Validation Results", fontsize=14, fontweight='bold')

    for ax_i, (mn, res) in enumerate(results_summary.items()):
        if ax_i >= 2:
            break
        ax = axes[ax_i]
        cm_plot = res['cm']
        cms = cm_plot.astype(float)
        row_sums = cms.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_norm = cms / row_sums

        im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=10)
        ax.set_yticklabels(class_names, fontsize=10)
        ax.set_title(f"{mn}\nTest acc: {res['test_acc']:.3f}  F1: {res['macro_f1']:.3f}", fontsize=11)
        ax.set_xlabel('Predicted'); ax.set_ylabel('True')
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                count = cm_plot[i, j]
                pct   = cm_norm[i, j]
                color = 'white' if pct > 0.6 else 'black'
                ax.text(j, i, f"{pct:.0%}\n({count})", ha='center', va='center',
                        fontsize=10, color=color, fontweight='bold')

    # Bar chart comparison
    ax = axes[2]
    metric_labels = ['CV acc', 'CV F1', 'Test acc', 'Test F1', 'Val acc', 'Kappa']
    rf_vals = [results_summary['Random Forest'][k] for k in
               ['cv_acc_mean','cv_f1_mean','test_acc','macro_f1','val_acc','kappa']]
    xgb_vals = [results_summary['XGBoost'][k] for k in
                ['cv_acc_mean','cv_f1_mean','test_acc','macro_f1','val_acc','kappa']]

    x = np.arange(len(metric_labels))
    w = 0.35
    ax.bar(x - w/2, rf_vals,  w, label='Random Forest', color='#534AB7', alpha=0.85)
    ax.bar(x + w/2, xgb_vals, w, label='XGBoost',       color='#1D9E75', alpha=0.85)
    ax.axhline(0.80, linestyle='--', color='#D85A30', linewidth=1, label='Pass (0.80)')
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=9)
    ax.set_ylabel('Score')
    ax.set_title('RF vs XGBoost', fontsize=11)
    ax.legend(fontsize=8)
    for i, (rv, xv) in enumerate(zip(rf_vals, xgb_vals)):
        ax.text(i - w/2, rv + 0.01, f'{rv:.2f}', ha='center', fontsize=7, color='#534AB7')
        ax.text(i + w/2, xv + 0.01, f'{xv:.2f}', ha='center', fontsize=7, color='#1D9E75')

    plt.tight_layout()
    chart_path = os.path.join(OUTPUT_DIR, 'validation_results.png')
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  📊 Chart saved → {chart_path}")

    # ══════════════════════════════════════════════════════════
    # DEPLOYMENT VERDICT
    # ══════════════════════════════════════════════════════════
    print("\n" + "═" * 60)
    print(f"  {BOLD}DEPLOYMENT VERDICT{RESET}")
    print("═" * 60)

    best_model_name = max(results_summary, key=lambda m: results_summary[m]['macro_f1'])
    best = results_summary[best_model_name]

    gates = [
        ("CV accuracy ≥ 0.82",       best['cv_acc_mean'] >= 0.82),
        ("CV std ≤ 0.05",            best['cv_acc_std']  <= 0.05),
        ("Macro F1 ≥ 0.80",          best['macro_f1']    >= 0.80),
        ("Cohen's Kappa ≥ 0.75",     best['kappa']       >= 0.75),
        ("Overfit gap ≤ 0.05",       best['overfit_gap'] <= 0.05),
        ("Brier score ≤ 0.15",       best['avg_brier']   <= 0.15),
        ("Val accuracy ≥ 0.80",      best['val_acc']     >= 0.80),
        ("Avg confidence ≥ 75%",     best['avg_conf']    >= 0.75),
        ("Ensemble agreement ≥ 85%", agreement           >= 0.85),
    ]

    passed = sum(1 for _, p in gates if p)
    for gate, passed_gate in gates:
        symbol = f"{GREEN}✓{RESET}" if passed_gate else f"{RED}✗{RESET}"
        print(f"  {symbol} {gate}")

    print()
    if passed == len(gates):
        print(f"  {GREEN}{BOLD}GO — All {len(gates)} gates passed. Ready to deploy.{RESET}")
    elif passed >= 6:
        print(f"  {YELLOW}{BOLD}CONDITIONAL GO — {passed}/{len(gates)} gates passed.{RESET}")
        print(f"  Fix failing gates before production.")
    else:
        print(f"  {RED}{BOLD}NO-GO — Only {passed}/{len(gates)} gates passed.{RESET}")
        print(f"  Model needs more work.")

    print(f"\n  Best model: {best_model_name}")
    print(f"  Strategy: XGBoost predicts, RF confirms. If they disagree → 'uncertain'.")

    # ══════════════════════════════════════════════════════════
    # SAVE PRODUCTION MODEL (dual-model ensemble)
    # ══════════════════════════════════════════════════════════
    print(f"\n🔄 Retraining both models on ALL data for production...")

    # Combine all data
    all_feats = train_feats + test_feats + val_feats
    X_all, y_all = to_matrix(all_feats)

    scaler = StandardScaler()
    X_all_s = scaler.fit_transform(X_all)

    rf_final = RandomForestClassifier(
        n_estimators=300, max_depth=15, min_samples_leaf=2,
        class_weight='balanced', random_state=42, n_jobs=-1)
    rf_final.fit(X_all_s, y_all)

    xgb_final = XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=2,
        gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
        objective='multi:softprob', use_label_encoder=False,
        eval_metric='mlogloss', random_state=42, n_jobs=-1)
    xgb_final.fit(X_all_s, y_all)

    model_package = {
        'model': xgb_final,            # primary model (for app.py)
        'rf_model': rf_final,           # secondary model (ensemble)
        'scaler': scaler,
        'label_encoder': le,
        'feature_cols': feat_cols,
        'classes': list(le.classes_),
        'best_model_name': best_model_name,
        'test_accuracy': float(best['test_acc']),
        'val_accuracy': float(best['val_acc']),
        'total_samples': len(all_feats),
        'sample_rate': SAMPLE_RATE,
        'validation': {
            'cv_accuracy': float(best['cv_acc_mean']),
            'macro_f1': float(best['macro_f1']),
            'kappa': float(best['kappa']),
            'ensemble_agreement': float(agreement),
            'brier_score': float(best['avg_brier']),
            'gates_passed': passed,
            'gates_total': len(gates),
        },
        'conservation_classes': CONSERVATION_CLASSES,
    }

    model_path = os.path.join(MODEL_DIR, 'classifier.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model_package, f)

    print(f"\n  ✅ Production model saved → {model_path}")
    print(f"     Contains: XGBoost (primary) + RF (ensemble)")
    print(f"     Scaler + features + validation metadata")

    print(f"\n{'═'*60}")
    print(f"  HOW TO RUN:")
    print(f"  1. python validate_model.py    ← you just did this")
    print(f"  2. python app.py               ← launches web app")
    print(f"  3. Open http://localhost:5000")
    print(f"  4. Upload a .wav → see classification + emotion map")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
