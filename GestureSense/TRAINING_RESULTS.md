# ISL_CSLRT_Corpus Training Results

## Training Completed Successfully ✅

**Date:** Training completed on ISL_CSLRT_Corpus dataset  
**Model Type:** SVM with RBF kernel + MediaPipe hand landmarks  
**Model File:** `models_cache/isl_model.pkl` (4.10 MB)

---

## Dataset Overview

- **Source:** ISL_CSLRT_Corpus/ISL_CSLRT_Corpus/Frames_Word_Level/
- **Total Classes:** 114 ISL words
- **Real Images:** 1,036 images (1920×1080 resolution)
- **Training Samples:** 3,566 (after augmentation)
- **Extraction Time:** 119.1 seconds

### Dataset Characteristics
- **Min images per class:** 2 (e.g., PLEASE)
- **Max images per class:** 110 (e.g., YOU)
- **Average per class:** 9 images
- **Augmentation strategy:** Each class augmented to minimum 30 samples using:
  - Small Gaussian jitter on feature vectors (std=0.015)
  - Maintained translation/scale invariance through wrist-relative normalization

---

## Model Architecture

### Feature Engineering (73-dimensional vector)
1. **63 normalized landmark coordinates:**
   - 21 hand landmarks × 3 coordinates (x, y, z)
   - Wrist-relative positioning (translation invariant)
   - Palm-scaled normalization (scale invariant)

2. **5 finger-up flags:**
   - Thumb, Index, Middle, Ring, Pinky (binary)

3. **5 inter-tip distances:**
   - Thumb-Index, Thumb-Middle, Index-Middle, Middle-Ring, Ring-Pinky
   - Normalized by palm size

### Classifier
- **Algorithm:** Support Vector Machine (SVM)
- **Kernel:** RBF (Radial Basis Function)
- **C parameter:** 10
- **Gamma:** scale
- **Class weighting:** Balanced (handles imbalanced dataset)
- **Preprocessing:** StandardScaler normalization

---

## Performance Metrics

### Cross-Validation (5-fold stratified)
- **Mean Accuracy:** 76.98%
- **Standard Deviation:** ±1.16%

### Test Set Performance
- **Test Accuracy:** 78.99%
- **Test Size:** 714 samples (20% holdout)
- **Training Time:** 6.1 seconds

### Per-Class Performance (Test Set)
**High Performers (F1 ≥ 0.90):**
- A LOT (0.92), ALL (1.00), ANGRY (1.00), BEAUTIFUL (1.00)
- CHAT (1.00), CLASS (1.00), COMB (1.00), CONGRATULATIONS (1.00)
- DARE (0.92), DILEMMA (1.00), FAVOUR (1.00), FINE (1.00)
- FROM (1.00), ENJOY (0.91), GRATEFUL (0.92), HEAR (0.92)
- HURT (0.91), MEDICINE (1.00), NAME (1.00), NUMBER (0.92)
- ON THE WAY (1.00), OUTSIDE (0.86), PHONE (0.92), PLACE (0.91)
- POUR (0.91), PREPARE (0.92), REPEAT (1.00), SERVE (1.00)
- SHIRT (0.92), SITTING (1.00), SO MUCH (1.00), SOME HOW (1.00)
- SOME ONE (1.00), SOMETHING (1.00), STUBBORN (0.91), WORRY (0.91)
- TAKE TIME (1.00), THIRSTY (1.00), TRAIN (1.00), WANT (0.92)
- WHERE (1.00), WHO (1.00)

**Moderate Performers (0.70 ≤ F1 < 0.90):**
- Most classes fall in this range with reasonable performance

**Challenging Classes (F1 < 0.50):**
- HAD (0.00) - needs more training data
- NOT (0.25) - confusion with similar gestures
- THAT (0.35) - gesture similarity issues
- DO (0.44), GO (0.40), KIND (0.40)
- WHAT (0.44) - surprising given 30 real samples

---

## 114 Trained Classes

```
A LOT, ABUSE, AFRAID, AGREE, ALL, ANGRY, ANYTHING, APPRECIATE, BAD, BEAUTIFUL, 
BECOME, BED, BORED, BRING, CHAT, CLASS, COLD, COLLEGE_SCHOOL, COMB, COME, 
CONGRATULATIONS, CRYING, DARE, DIFFERENCE, DILEMMA, DISAPPOINTED, DO, DON'T CARE, 
ENJOY, FAVOUR, FEVER, FINE, FOOD, FREE, FRIEND, FROM, GO, GOOD, GRATEFUL, HAD, 
HAPPENED, HAPPY, HEAR, HEART, HELLO_HI, HELP, HIDING, HOW, HUNGRY, HURT, 
I_ME_MINE_MY, KIND, LEAVE, LIKE, LIKE_LOVE, MEAN IT, MEDICINE, MEET, NAME, 
NICE, NOT, NUMBER, OLD_AGE, ON THE WAY, OUTSIDE, PHONE, PLACE, PLEASE, POUR, 
PREPARE, PROMISE, REALLY, REPEAT, ROOM, SERVE, SHIRT, SITTING, SLEEP, SLOWER, 
SO MUCH, SOFTLY, SOME HOW, SOME ONE, SOMETHING, SORRY, SPEAK, STOP, STUBBORN, 
SURE, TAKE CARE, TAKE TIME, TALK, TELL, THANK, THAT, THINGS, THINK, THIRSTY, 
TIRED, TODAY, TRAIN, TRUST, TRUTH, TURN ON, UNDERSTAND, WANT, WATER, WEAR, 
WELCOME, WHAT, WHERE, WHO, WORRY, YOU
```

---

## Technical Details

### Environment
- **Python:** 3.10
- **MediaPipe:** 0.10.21
- **OpenCV:** 4.10.0.84 (opencv-python-headless)
- **NumPy:** 1.26.4
- **scikit-learn:** Latest stable

### Key Design Decisions

1. **Why SVM over Deep Learning?**
   - Small dataset (1,036 images, avg 9 per class)
   - Rich engineered features (73-d landmarks)
   - SVM excels with small, high-quality feature sets
   - Fast training and inference
   - No GPU required

2. **Why MediaPipe landmarks over raw pixels?**
   - Translation/scale/rotation invariant
   - Works across lighting conditions
   - Compact representation (73 floats vs 1920×1080×3 pixels)
   - Generalizes better with limited data
   - Real-time inference capability

3. **Augmentation Strategy**
   - Feature-level jitter (not image augmentation)
   - Preserves geometric relationships
   - Targets 30 samples/class minimum
   - Balanced class weights in SVM

---

## Integration with App

### Files Created/Updated
- `train_isl_model.py` - Training script
- `utils/isl_classifier.py` - Inference wrapper
- `models_cache/isl_model.pkl` - Trained SVM model
- `models_cache/isl_labels.pkl` - Class labels
- `models_cache/isl_meta.pkl` - Metadata

### Usage in Flask App
The app automatically detects and loads the ISL model:
- Primary mode: ISL model predictions (114 words)
- Fallback: Rule-based gestures (45+ gestures)
- Confidence threshold: 0.55 (55%)
- Real-time inference via MediaPipe landmarks

---

## How to Retrain

```bash
# Full dataset (114 classes, 30+ samples each)
python train_isl_model.py

# Quick smoke test (5 images per class)
python train_isl_model.py --quick

# Custom dataset location
python train_isl_model.py --data-dir /path/to/Frames_Word_Level

# Quiet mode (suppress per-class progress)
python train_isl_model.py --quiet
```

---

## Next Steps & Improvements

### Potential Enhancements
1. **More training data** for low-performing classes (HAD, NOT, THAT, DO, GO)
2. **Ensemble methods:** Combine SVM with rule-based classifier
3. **Temporal modeling:** Use video sequences instead of single frames
4. **Two-hand gestures:** Currently uses only primary hand
5. **Context-aware prediction:** Language model to smooth predictions

### Known Limitations
- Single frame classification (no temporal context)
- Single hand priority (two-hand gestures use first detected)
- Class imbalance effects (2-110 samples per class)
- Some gesture confusion between similar shapes

---

## Verification

✅ Model file exists: `models_cache/isl_model.pkl` (4.10 MB)  
✅ Labels file exists: `models_cache/isl_labels.pkl`  
✅ Flask app loads model successfully  
✅ ISL classifier reports 114 classes available  
✅ Web interface running at http://127.0.0.1:5000

**Status:** Ready for real-time gesture recognition! 🎉
