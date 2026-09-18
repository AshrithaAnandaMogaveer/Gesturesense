"""
train_isl_model.py  -  ISL_CSLRT_Corpus Trainer
================================================
Dataset : ISL_CSLRT_Corpus/ISL_CSLRT_Corpus/Frames_Word_Level/
          114 ISL word classes, ~1,036 images (1920x1080 JPEGs)
          Very small dataset: avg 9 images/class, min 2, max 110

Pipeline
--------
1.  Walk every class folder, read each image.
2.  Run MediaPipe Hands (static mode) to extract 21 landmarks per hand.
3.  Build a 73-d normalised feature vector per sample:
      - 63 wrist-relative, palm-scaled landmark coords (x,y,z)
      - 5  finger-up flags
      - 5  normalised inter-tip distances
4.  Augment to 30 samples per class (flips + rotations on image before
    landmark extraction, and small jitter on feature vectors).
5.  Train SVM (RBF kernel) with class-weight balancing.
    SVM is the right choice for small datasets with rich features.
6.  Evaluate with stratified 5-fold cross-validation + held-out test split.
7.  Save model + labels to models_cache/.

Usage
-----
  python train_isl_model.py
  python train_isl_model.py --data-dir ISL_CSLRT_Corpus
  python train_isl_model.py --quick     # 5 images/class smoke test
"""

import argparse
import math
import pickle
import sys
import time
import warnings
from pathlib import Path

# Import CV2 and MediaPipe FIRST (before sklearn/numpy heavy imports)
import cv2
import mediapipe as mp

import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
CACHE_DIR   = BASE_DIR / "models_cache"
MODEL_PATH  = CACHE_DIR / "isl_model.pkl"
LABELS_PATH = CACHE_DIR / "isl_labels.pkl"
META_PATH   = CACHE_DIR / "isl_meta.pkl"

# Default dataset location (nested folder inside GestureSense)
DEFAULT_DATA_DIR = BASE_DIR / "ISL_CSLRT_Corpus" / "ISL_CSLRT_Corpus" / "Frames_Word_Level"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Minimum augmented samples per class
MIN_SAMPLES_PER_CLASS = 30


# ==============================================================================
# FEATURE EXTRACTION  (from MediaPipe landmarks)
# ==============================================================================

def _dist(lm, a, b):
    return math.hypot(lm[a].x - lm[b].x, lm[a].y - lm[b].y)

def _palm_size(lm):
    return max(_dist(lm, 0, 9), 1e-6)

def _fingers_up(lm, handedness="Right"):
    tips = [4, 8, 12, 16, 20]
    pips = [3, 6, 10, 14, 18]
    f = []
    if handedness == "Right":
        f.append(1 if lm[4].x < lm[3].x else 0)
    else:
        f.append(1 if lm[4].x > lm[3].x else 0)
    for i in range(1, 5):
        f.append(1 if lm[tips[i]].y < lm[pips[i]].y else 0)
    return f

def extract_features(lm, handedness="Right"):
    """
    73-d feature vector from 21 MediaPipe landmarks.
    Translation-invariant (wrist-relative) and scale-invariant (palm-scaled).
    """
    wrist_x, wrist_y = lm[0].x, lm[0].y
    palm = _palm_size(lm)

    # 63 normalised coords
    norm = np.zeros(63, dtype=np.float32)
    for i in range(21):
        norm[i*3 + 0] = (lm[i].x - wrist_x) / palm
        norm[i*3 + 1] = (lm[i].y - wrist_y) / palm
        norm[i*3 + 2] =  lm[i].z / palm

    # 5 finger-up flags + 5 tip distances
    fingers = _fingers_up(lm, handedness)
    tip_dists = [
        _dist(lm, 4,  8) / palm,
        _dist(lm, 4, 12) / palm,
        _dist(lm, 8, 12) / palm,
        _dist(lm, 12,16) / palm,
        _dist(lm, 16,20) / palm,
    ]
    return np.concatenate([norm, fingers, tip_dists]).astype(np.float32)


def jitter_features(feat, n=3, std=0.02):
    """Add small Gaussian noise to a feature vector to create variants."""
    variants = []
    for _ in range(n):
        noisy = feat + np.random.normal(0, std, feat.shape).astype(np.float32)
        variants.append(noisy)
    return variants


# ==============================================================================
# DATASET LOADING
# ==============================================================================

def find_data_dir(hint=None):
    """Locate Frames_Word_Level directory."""
    candidates = []

    if hint:
        p = Path(hint)
        # Direct path to Frames_Word_Level
        if p.is_dir() and any(sub.is_dir() for sub in p.iterdir()):
            candidates.append(p)
        # One level nested
        for sub in ["Frames_Word_Level",
                    "ISL_CSLRT_Corpus/Frames_Word_Level",
                    "ISL_CSLRT_Corpus/ISL_CSLRT_Corpus/Frames_Word_Level"]:
            candidates.append(p / sub)

    candidates.append(DEFAULT_DATA_DIR)
    # Also search relative paths
    for rel in [
        "ISL_CSLRT_Corpus/Frames_Word_Level",
        "ISL_CSLRT_Corpus/ISL_CSLRT_Corpus/Frames_Word_Level",
    ]:
        candidates.append(BASE_DIR / rel)

    for c in candidates:
        if c.is_dir():
            # Verify it has class sub-folders with images
            subdirs = [d for d in c.iterdir() if d.is_dir()]
            if subdirs:
                return c

    sys.exit(
        "[ERROR] Cannot find Frames_Word_Level directory.\n"
        "Pass --data-dir pointing to the folder that contains class sub-folders."
    )


def load_dataset(data_dir, max_per_class=None, verbose=True):
    """
    Walk data_dir/<class>/*.jpg, extract MediaPipe landmarks,
    augment to MIN_SAMPLES_PER_CLASS, return (X, y, class_names).
    """
    hands_mp = mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.30,  # lower threshold for dataset images
        model_complexity=1,
    )

    class_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    le = LabelEncoder()
    le.fit([d.name for d in class_dirs])
    class_names = list(le.classes_)

    X_list, y_list = [], []
    skipped_classes = []
    t0 = time.time()

    for cls_dir in class_dirs:
        cls_name = cls_dir.name
        cls_idx  = int(le.transform([cls_name])[0])

        images = [
            f for f in sorted(cls_dir.iterdir())
            if f.is_file() and f.suffix.lower() in IMG_EXTS
        ]
        if max_per_class:
            images = images[:max_per_class]

        raw_feats = []  # features extracted from real images

        for img_path in images:
            try:
                img_bgr = cv2.imread(str(img_path))
                if img_bgr is None:
                    continue

                # Resize to 640px wide for faster processing
                h, w = img_bgr.shape[:2]
                if w > 640:
                    scale = 640 / w
                    img_bgr = cv2.resize(img_bgr, (640, int(h * scale)))

                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                results = hands_mp.process(img_rgb)

                if not results.multi_hand_landmarks:
                    # Try flipped image
                    flipped = cv2.flip(img_rgb, 1)
                    results = hands_mp.process(flipped)

                if not results.multi_hand_landmarks:
                    continue

                lm   = results.multi_hand_landmarks[0].landmark
                side = "Right"
                if results.multi_handedness:
                    side = results.multi_handedness[0].classification[0].label

                feat = extract_features(lm, side)
                raw_feats.append(feat)

            except Exception as exc:
                if verbose:
                    print(f"  [WARN] {img_path.name}: {exc}")

        if not raw_feats:
            skipped_classes.append(cls_name)
            if verbose:
                print(f"  [SKIP] {cls_name} — no hand landmarks detected in any image")
            continue

        # ── Augment: jitter features until we hit MIN_SAMPLES_PER_CLASS ──────
        target = max(MIN_SAMPLES_PER_CLASS,
                     len(raw_feats)) if not max_per_class else len(raw_feats)
        all_feats = list(raw_feats)

        if len(raw_feats) < target:
            needed = target - len(raw_feats)
            pool   = raw_feats * (needed // len(raw_feats) + 1)
            for base_feat in pool[:needed]:
                jittered = jitter_features(base_feat, n=1, std=0.015)[0]
                all_feats.append(jittered)

        for feat in all_feats:
            X_list.append(feat)
            y_list.append(cls_idx)

        if verbose:
            print(f"  [{cls_name:30s}] {len(raw_feats):3d} real  +  "
                  f"{len(all_feats)-len(raw_feats):3d} augmented  = {len(all_feats):3d} total")

    hands_mp.close()

    if verbose:
        elapsed = time.time() - t0
        print(f"\n[INFO] Extraction time  : {elapsed:.1f}s")
        print(f"[INFO] Classes found    : {len(class_names) - len(skipped_classes)} / {len(class_names)}")
        if skipped_classes:
            print(f"[WARN] Skipped (no hand): {skipped_classes}")

    if not X_list:
        sys.exit("[ERROR] No features extracted. Check dataset path and MediaPipe installation.")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)

    # Remove skipped classes from label list
    valid_indices = sorted(set(y.tolist()))
    class_names   = [class_names[i] for i in valid_indices]

    # Re-encode to 0-based contiguous indices
    remap = {old: new for new, old in enumerate(valid_indices)}
    y = np.array([remap[i] for i in y], dtype=np.int32)

    return X, y, class_names


# ==============================================================================
# TRAINING  (SVM with RBF kernel)
# ==============================================================================

def train(X, y, class_names, verbose=True):
    print(f"\n[INFO] Training SVM classifier ...")
    print(f"       Samples   : {len(X)}")
    print(f"       Features  : {X.shape[1]}")
    print(f"       Classes   : {len(class_names)}")

    # Cross-validation first (uses all data — important for small datasets)
    print(f"\n[INFO] Running 5-fold stratified cross-validation ...")
    pipeline_cv = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            kernel="rbf",
            C=10,
            gamma="scale",
            probability=True,
            class_weight="balanced",
            random_state=42,
        )),
    ])

    # Only do CV if enough samples per class
    min_count = np.bincount(y).min()
    n_folds   = min(5, min_count)
    if n_folds >= 2:
        cv_scores = cross_val_score(
            pipeline_cv, X, y,
            cv=StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42),
            scoring="accuracy", n_jobs=-1,
        )
        print(f"[INFO] CV accuracy ({n_folds}-fold): "
              f"{cv_scores.mean()*100:.2f}% +/- {cv_scores.std()*100:.2f}%")
    else:
        print(f"[WARN] Too few samples for CV — skipping.")

    # Train/test split (20% test, stratified)
    min_count_split = np.bincount(y).min()
    if min_count_split >= 2:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y)
    else:
        # Too few samples — use all for training, skip test split
        X_tr, y_tr = X, y
        X_te,  y_te  = None, None

    # Compute class weights
    cw = dict(zip(
        np.unique(y_tr).tolist(),
        compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr).tolist()
    ))

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            kernel="rbf",
            C=10,
            gamma="scale",
            probability=True,
            class_weight=cw,
            random_state=42,
        )),
    ])

    t0 = time.time()
    pipeline.fit(X_tr, y_tr)
    print(f"[INFO] Training time : {time.time()-t0:.1f}s")

    if X_te is not None and len(X_te) > 0:
        y_pred = pipeline.predict(X_te)
        acc    = accuracy_score(y_te, y_pred)
        print(f"\n[RESULT] Test accuracy : {acc*100:.2f}%\n")

        # Only show per-class report if we have enough test samples
        if len(X_te) >= 10:
            tnames = [class_names[i] for i in sorted(np.unique(y_te))]
            print(classification_report(
                y_te, y_pred, target_names=tnames, zero_division=0))
    else:
        print("[INFO] No test split (too few samples) — model trained on all data.")

    return pipeline


# ==============================================================================
# SAVE
# ==============================================================================

def save(pipeline, class_names):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH,  "wb") as f:
        pickle.dump(pipeline,    f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(LABELS_PATH, "wb") as f:
        pickle.dump(class_names, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(META_PATH,   "wb") as f:
        pickle.dump({"approach": "landmarks"}, f)

    mb = MODEL_PATH.stat().st_size / 1024 / 1024
    print(f"\n[OK] Model  -> {MODEL_PATH}  ({mb:.2f} MB)")
    print(f"[OK] Labels -> {LABELS_PATH}")
    print(f"[OK] {len(class_names)} classes: {class_names}")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    p = argparse.ArgumentParser(
        description="Train ISL model on ISL_CSLRT_Corpus dataset.")
    p.add_argument("--data-dir", default=None,
                   help="Path to Frames_Word_Level folder (auto-detected if omitted).")
    p.add_argument("--quick", action="store_true",
                   help="Use only 5 images per class for a smoke test.")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress per-class progress lines.")
    args = p.parse_args()

    print("=" * 60)
    print("  ISL_CSLRT_Corpus  -  SVM Landmark Trainer")
    print("=" * 60)

    data_dir = find_data_dir(args.data_dir)
    print(f"[INFO] Dataset : {data_dir}")

    max_per = 5 if args.quick else None
    X, y, class_names = load_dataset(
        data_dir,
        max_per_class=max_per,
        verbose=not args.quiet,
    )

    X, y = shuffle(X, y, random_state=42)
    pipeline = train(X, y, class_names, verbose=not args.quiet)
    save(pipeline, class_names)

    print("\n[DONE] Run  python app.py  -- model loads automatically.")


if __name__ == "__main__":
    main()
