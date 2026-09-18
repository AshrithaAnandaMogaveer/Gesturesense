"""
isl_classifier.py  -  ISL_CSLRT_Corpus landmark classifier
===========================================================
Loads the SVM model trained on MediaPipe landmark features.

Feature vector (73-d, matches train_isl_model.py exactly)
----------------------------------------------------------
  63  wrist-relative, palm-scaled landmark coords (x,y,z for each of 21 pts)
   5  finger-up flags  [thumb, index, middle, ring, pinky]
   5  normalised inter-tip distances

Public API
----------
  clf = get_isl_classifier()
  label, conf = clf.predict(hand_results)
  available   = clf.is_available()
  labels      = clf.labels           # list of ISL word strings
"""

from __future__ import annotations

import math
import pickle
import threading
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Model paths ────────────────────────────────────────────────────────────────
_BASE       = Path(__file__).parent.parent
MODEL_PATH  = _BASE / "models_cache" / "isl_model.pkl"
LABELS_PATH = _BASE / "models_cache" / "isl_labels.pkl"
META_PATH   = _BASE / "models_cache" / "isl_meta.pkl"

# Confidence gate  -  only override rule-based when ML is this confident
ISL_CONFIDENCE_THRESHOLD = 0.55


# ==============================================================================
# FEATURE EXTRACTION  (must match train_isl_model.py exactly)
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

def build_feature_vector(hand_results) -> Optional[np.ndarray]:
    """
    Build 73-d feature vector from a MediaPipe Hands result object.
    Uses the primary (first) detected hand.
    Returns None if no hand is detected.
    """
    if not hand_results or not hand_results.multi_hand_landmarks:
        return None

    lm   = hand_results.multi_hand_landmarks[0].landmark
    side = "Right"
    if hand_results.multi_handedness:
        side = hand_results.multi_handedness[0].classification[0].label

    wrist_x, wrist_y = lm[0].x, lm[0].y
    palm = _palm_size(lm)

    # 63 normalised coords
    norm = np.zeros(63, dtype=np.float32)
    for i in range(21):
        norm[i*3 + 0] = (lm[i].x - wrist_x) / palm
        norm[i*3 + 1] = (lm[i].y - wrist_y) / palm
        norm[i*3 + 2] =  lm[i].z / palm

    # 5 finger flags + 5 tip distances
    fingers = _fingers_up(lm, side)
    tip_dists = [
        _dist(lm, 4,  8) / palm,
        _dist(lm, 4, 12) / palm,
        _dist(lm, 8, 12) / palm,
        _dist(lm, 12,16) / palm,
        _dist(lm, 16,20) / palm,
    ]
    return np.concatenate([norm, fingers, tip_dists]).astype(np.float32)


# ==============================================================================
# CLASSIFIER
# ==============================================================================

class ISLClassifier:
    """Thread-safe SVM wrapper. Lazy-loads on first call."""

    def __init__(self):
        self._lock    = threading.Lock()
        self._model   = None
        self._labels: Optional[list[str]] = None
        self._loaded  = False
        self._broken  = False

    def is_available(self) -> bool:
        self._ensure_loaded()
        return self._loaded and not self._broken

    @property
    def labels(self) -> list[str]:
        self._ensure_loaded()
        return self._labels or []

    # Stubs for compatibility with dual-model interface used in gesture_engine
    @property
    def angles_available(self) -> bool:
        return self.is_available()

    @property
    def mnist_available(self) -> bool:
        return False

    def predict(self,
                hand_results,
                frame_bgr=None) -> Tuple[str, float, dict]:
        """
        Predict ISL word from a MediaPipe Hands result.

        Returns
        -------
        label  : str    predicted ISL word
        conf   : float  probability in [0, 1]
        detail : dict   {"angles": (label, conf), "mnist": ("Unknown", 0.0)}
        """
        empty = ("Unknown", 0.0, {"angles": ("Unknown", 0.0),
                                   "mnist":  ("Unknown", 0.0)})

        if not self.is_available():
            return empty

        feat = build_feature_vector(hand_results)
        if feat is None:
            return empty

        with self._lock:
            try:
                proba = self._model.predict_proba(feat.reshape(1, -1))[0]
            except Exception as exc:
                logger.error("ISLClassifier.predict: %s", exc)
                return empty

        idx   = int(np.argmax(proba))
        conf  = float(proba[idx])
        label = self._labels[idx]
        detail = {"angles": (label, conf), "mnist": ("Unknown", 0.0)}
        return label, conf, detail

    def _ensure_loaded(self):
        if self._loaded or self._broken:
            return
        with self._lock:
            if self._loaded or self._broken:
                return
            if not MODEL_PATH.exists():
                logger.warning(
                    "ISL model not found at %s -- "
                    "run  python train_isl_model.py  to train it.", MODEL_PATH)
                self._broken = True
                return
            try:
                with open(MODEL_PATH,  "rb") as f:
                    self._model  = pickle.load(f)
                with open(LABELS_PATH, "rb") as f:
                    self._labels = pickle.load(f)
                self._loaded = True
                logger.info("ISL model loaded -- %d classes: %s",
                            len(self._labels), self._labels)
            except Exception as exc:
                logger.error("Failed to load ISL model: %s", exc)
                self._broken = True


# ── Singleton ──────────────────────────────────────────────────────────────────
_instance: Optional[ISLClassifier] = None
_lock = threading.Lock()

def get_isl_classifier() -> ISLClassifier:
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = ISLClassifier()
    return _instance
