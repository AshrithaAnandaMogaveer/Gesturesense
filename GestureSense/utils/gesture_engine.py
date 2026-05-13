"""
GestureEngine - Hand sign + Face expression recognition system.

Processes BOTH:
  - Hand signs  : 40+ gestures via MediaPipe Hands (21 landmarks)
  - Face expressions : 10+ expressions via MediaPipe Face Mesh (468 landmarks)

Both detectors run in the same background thread on the same frame.
"""

import cv2
import mediapipe as mp
import numpy as np
import threading
import time
import math
import base64
from collections import deque, Counter
from typing import Generator, Tuple, List, Optional


# =============================================================================
# GEOMETRY HELPERS
# =============================================================================

def _dist(lm, a: int, b: int) -> float:
    """Euclidean distance between two landmarks (normalised coords)."""
    return math.hypot(lm[a].x - lm[b].x, lm[a].y - lm[b].y)


def _angle(lm, a: int, b: int, c: int) -> float:
    """Angle in degrees at joint b, formed by points a-b-c."""
    v1 = np.array([lm[a].x - lm[b].x, lm[a].y - lm[b].y])
    v2 = np.array([lm[c].x - lm[b].x, lm[c].y - lm[b].y])
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 180.0
    return math.degrees(math.acos(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)))


def _fingers_up(lm, handedness: str = "Right") -> List[int]:
    """
    Returns [Thumb, Index, Middle, Ring, Pinky] as 1=extended / 0=curled.
    Thumb uses x-axis (direction flipped for left hand).
    Other fingers: tip.y < PIP.y means extended (y increases downward).
    """
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


def _palm_size(lm) -> float:
    """Reference scale: wrist (0) to middle-finger MCP (9)."""
    return max(_dist(lm, 0, 9), 0.01)


# =============================================================================
# FACE EXPRESSION CLASSIFIER  (MediaPipe Face Mesh - 468 landmarks)
# =============================================================================
#
# Key landmark indices used (normalised x,y,z coords):
#   Mouth corners : 61 (left), 291 (right)
#   Upper lip     : 13
#   Lower lip     : 14
#   Mouth top     : 0
#   Mouth bottom  : 17
#   Left eye      : outer=33, inner=133, top=159, bottom=145
#   Right eye     : outer=362, inner=263, top=386, bottom=374
#   Left brow     : 70 (inner), 105 (outer)
#   Right brow    : 300 (inner), 334 (outer)
#   Nose tip      : 1
#   Chin          : 152
#   Forehead      : 10
# =============================================================================

def _fdist(lm, a: int, b: int) -> float:
    """Distance between two face mesh landmarks."""
    return math.hypot(lm[a].x - lm[b].x, lm[a].y - lm[b].y)


def classify_face_expression(lm) -> Tuple[str, float]:
    """
    Classify facial expression from 468 MediaPipe Face Mesh landmarks.
    Returns (expression_name, confidence 0.0-1.0).

    y increases DOWNWARD in MediaPipe normalised coords.
    All metrics normalised by face height for scale-independence.

    Key landmarks used:
      Forehead:10  Chin:152  Nose tip:1
      Mouth corners: 61(left) 291(right)
      Upper lip:13  Lower lip:14
      Left eye:  top=159 bottom=145 outer=33 inner=133
      Right eye: top=386 bottom=374 outer=362 inner=263
      Left brow inner:70   Right brow inner:300
    """
    # ── Face scale ─────────────────────────────────────────────────────────────
    face_h = max(_fdist(lm, 10, 152), 0.01)   # forehead -> chin

    # ── Mouth metrics ──────────────────────────────────────────────────────────
    mouth_w = _fdist(lm, 61, 291)                  # corner width
    mouth_h = _fdist(lm, 13, 14)                   # lip gap (vertical)
    mar     = mouth_h / max(mouth_w, 0.01)         # Mouth Aspect Ratio

    # Mouth width normalised by face height
    # Neutral ~0.35-0.40, smile widens to ~0.42-0.50
    mouth_w_norm = mouth_w / face_h

    # ── Smile metric: corner height relative to lower lip ─────────────────────
    # In a smile, corners rise ABOVE the lower lip level.
    # lower_lip_y = lm[14].y  (lower lip centre)
    # corner_avg_y = average of left/right corner y
    # smile_score = (lower_lip_y - corner_avg_y) / face_h
    #   positive -> corners are ABOVE lower lip -> smile
    #   negative -> corners are BELOW lower lip -> frown
    lower_lip_y  = lm[14].y
    corner_avg_y = (lm[61].y + lm[291].y) / 2.0
    smile_score  = (lower_lip_y - corner_avg_y) / face_h

    # Secondary: corner height relative to upper lip
    upper_lip_y   = lm[13].y
    corner_vs_ulip = (upper_lip_y - corner_avg_y) / face_h
    # positive -> corners above upper lip (big smile)

    # ── Eye metrics ────────────────────────────────────────────────────────────
    l_ear    = _fdist(lm, 159, 145) / max(_fdist(lm, 33, 133), 0.01)
    r_ear    = _fdist(lm, 386, 374) / max(_fdist(lm, 362, 263), 0.01)
    avg_ear  = (l_ear + r_ear) / 2.0
    ear_diff = abs(l_ear - r_ear)

    # ── Brow metrics ───────────────────────────────────────────────────────────
    # Brow raise: how far brow is above eye top (positive = raised)
    l_brow_raise = (lm[159].y - lm[70].y)  / face_h
    r_brow_raise = (lm[386].y - lm[300].y) / face_h
    avg_brow_raise = (l_brow_raise + r_brow_raise) / 2.0

    brow_inner_dist = _fdist(lm, 70, 300) / face_h

    # ── Mouth asymmetry ────────────────────────────────────────────────────────
    mouth_asym = abs(lm[61].y - lm[291].y) / face_h

    # =========================================================================
    # EXPRESSION RULES  (most specific first)
    # =========================================================================

    # 1. Surprise - mouth wide open + eyes wide + brows raised
    if mar > 0.40 and avg_ear > 0.32 and avg_brow_raise > 0.04:
        return "Surprised 😲", 0.93

    # 2. Mouth wide open - yawn / shock
    if mar > 0.45:
        return "Mouth Open 😮", 0.90

    # 3. Wink - one eye clearly more closed
    if ear_diff > 0.12 and avg_ear > 0.10:
        closed_side = "Left" if l_ear < r_ear else "Right"
        return "Winking 😉 (" + closed_side + ")", 0.88

    # 4. Eyes closed - sleeping / blinking
    if avg_ear < 0.10:
        return "Eyes Closed 😌", 0.87

    # 5. Raised eyebrows - questioning
    if avg_brow_raise > 0.05 and mar < 0.20:
        return "Raised Eyebrows 🤨", 0.86

    # 6. Frown - corners BELOW lower lip + brows furrowed
    if smile_score < -0.02 and brow_inner_dist < 0.32:
        return "Frowning 😠", 0.87

    # 7. Sad - corners below lower lip, brows slightly raised
    if smile_score < -0.01 and avg_brow_raise > 0.02:
        return "Sad 😢", 0.84

    # 8. Big Smile - corners above upper lip + mouth open
    if corner_vs_ulip > 0.01 and mar > 0.08:
        return "Big Smile 😁", 0.93

    # 9. Happy / Smile - corners clearly above lower lip
    if smile_score > 0.02:
        return "Happy / Smiling 😊", 0.91

    # 10. Slight smile - corners just above lower lip or wide mouth
    if smile_score > 0.005:
        return "Slight Smile 🙂", 0.85

    # 11. Squinting - eyes narrowed + mouth closed
    if avg_ear < 0.20 and mar < 0.12:
        return "Squinting 😑", 0.83

    # 12. Disgust - upper lip raised + corners neutral/down
    upper_lip_raise = (lm[0].y - lm[13].y) / face_h
    if upper_lip_raise > 0.02 and smile_score < 0.0:
        return "Disgusted 🤢", 0.82

    # 13. Thinking - mouth asymmetry + neutral eyes
    if mouth_asym > 0.012 and avg_ear > 0.18:
        return "Thinking 🤔", 0.80

    # 14. Neutral fallback
    return "Neutral 😐", 0.85

def classify_gesture(lm, handedness="Right"):
    """Classify hand sign. Returns (name, confidence 0-1). 45+ gestures."""
    from collections import namedtuple
    f    = _fingers_up(lm, handedness)
    palm = _palm_size(lm)

    ti = _dist(lm, 4,  8)  / palm
    tm = _dist(lm, 4, 12)  / palm
    im = _dist(lm, 8, 12)  / palm
    mr = _dist(lm, 12,16)  / palm
    rp = _dist(lm, 16,20)  / palm

    wy  = lm[0].y
    ty  = lm[4].y
    py9 = lm[9].y

    # RELAXED thresholds for real webcam use
    thumb_up    = ty < wy - 0.02
    thumb_down  = ty > wy + 0.02
    thumb_side  = abs(ty - wy) < 0.05
    hand_raised = py9 < wy - 0.01

    idx_curl = _angle(lm, 8,  6,  5)
    mid_curl = _angle(lm, 12, 10, 9)
    rng_curl = _angle(lm, 16, 14, 13)
    pnk_curl = _angle(lm, 20, 18, 17)

    spread = _dist(lm, 8, 20) / palm

    _all_curled  = (f[1]==0 and f[2]==0 and f[3]==0 and f[4]==0)
    _all_fist    = _all_curled
    _tips_close  = (im<0.35 and mr<0.35 and rp<0.35)
    _tip_to_palm = (_dist(lm,8,9)/palm<0.55 and _dist(lm,12,9)/palm<0.55)
    _tips_up     = (lm[8].y < lm[5].y and lm[12].y < lm[9].y)

    # FLOWER - all 5 fingers up, very wide spread, tips slightly curved
    _all_up      = (f[0]==1 and f[1]==1 and f[2]==1 and f[3]==1 and f[4]==1)
    _wide_spread = spread > 0.90
    _tips_curved = (idx_curl<160 and mid_curl<160 and rng_curl<160 and pnk_curl<160)
    if _all_up and _wide_spread and _tips_curved:
        return "Flower - Blooming / Beauty", 0.91

    # EATING
    if _all_curled and _tips_close and _tip_to_palm and _tips_up:
        return "Eating Sign - Food / Hungry", 0.91

    # SLEEP
    if _all_fist and abs(lm[9].x - lm[0].x) > 0.06 and not thumb_down:
        return "Sleep Sign - Tired / Rest", 0.90

    # CLAPPING
    if f==[1,1,1,1,1] and 0.55<=spread<0.85 and hand_raised:
        return "Clapping - Congratulations / Bravo", 0.93

    # THUMBS UP HIGH
    if f[0]==1 and f[1]==0 and f[2]==0 and f[3]==0 and f[4]==0 and thumb_up and hand_raised:
        return "Thumbs Up High - Great Job! / Congratulations", 0.94

    # FIST PUMP
    if _all_fist and hand_raised and not thumb_down and not _tips_up:
        return "Fist Pump - Yes! / Celebration", 0.90

    # OK SIGN
    if ti<0.35 and f[2]==1 and f[3]==1 and f[4]==1:
        return "OK Sign - Perfect / Correct", 0.95

    # PINCH
    if ti<0.28 and f[1]==0 and f[2]==0 and f[3]==0 and f[4]==0:
        return "Pinch Sign", 0.92

    # SNAP
    if tm<0.28 and f[0]==1 and f[1]==0 and f[2]==0 and f[3]==0 and f[4]==0:
        return "Snap / Click Sign", 0.88

    # FINGER HEART
    if f[0]==1 and f[1]==1 and f[2]==0 and f[3]==0 and f[4]==0 and ti<0.38:
        return "Finger Heart Sign", 0.90

    # THUMBS UP
    if f==[1,0,0,0,0] and thumb_up:
        return "Thumbs Up - Agree / Like", 0.95

    # THUMBS DOWN
    if f[1]==0 and f[2]==0 and f[3]==0 and f[4]==0 and thumb_down:
        return "Thumbs Down - Disagree", 0.93

    # THINKING
    if f[0]==1 and sum(f[1:])==0 and thumb_side:
        return "Thinking Sign", 0.83

    # OPEN PALM
    if f==[1,1,1,1,1]:
        if spread > 0.80:
            return "Open Palm - Stop / Hello / Wave", 0.95
        elif spread > 0.60:
            return "High Five - Celebrate", 0.91
        elif spread < 0.50 and abs(lm[8].y - lm[20].y) < 0.06:
            return "Namaste - Respect / Prayer", 0.88
        else:
            return "Flat Hand - Five", 0.88

    # FIST
    if f==[0,0,0,0,0]:
        if thumb_down:
            return "Thumbs Down - Disagree", 0.93
        elif lm[9].y > lm[0].y + 0.04:
            return "Fist - Power / Strength", 0.91
        else:
            return "Fist - Closed Hand", 0.92

    # VICTORY
    if f==[0,1,1,0,0]:
        if im > 0.50:
            return "Victory - Peace Sign", 0.94
        elif im < 0.28:
            return "Crossed Fingers - Good Luck", 0.87
        else:
            return "Two Fingers - Scissors", 0.88

    # SINGLE FINGER
    if f==[0,1,0,0,0]:
        dx = lm[8].x - lm[5].x
        dy = lm[8].y - lm[5].y
        if dy < -0.08:
            return "Point Up - Number One / Attention", 0.93
        elif dx > 0.08:
            return "Point Right - That Way", 0.90
        elif dx < -0.08:
            return "Point Left - That Way", 0.90
        elif dy > 0.06:
            return "Point Down - Look Here", 0.89
        else:
            return "Index Point - Indicate", 0.88

    if f==[0,0,1,0,0]: return "Middle Finger Sign", 0.92
    if f==[0,0,0,1,0]: return "Ring Finger Up Sign", 0.85
    if f==[0,0,0,0,1]: return "Pinky Up - Pinky Promise", 0.87

    # THREE FINGERS
    if f==[0,1,1,1,0]: return "Three Fingers - Number 3", 0.92
    if f==[1,1,1,0,0]: return "Three Alt - Thumb Index Middle", 0.88

    # FOUR FINGERS
    if f==[0,1,1,1,1]:
        mid_ring = _dist(lm,12,16)/palm
        idx_mid  = _dist(lm,8,12)/palm
        if mid_ring>0.55 and idx_mid<0.35:
            return "Vulcan Salute - Live Long and Prosper", 0.89
        return "Four Fingers - Number 4", 0.92
    if f==[1,1,1,1,0]: return "Four Alt - No Pinky", 0.87

    # THUMB COMBOS
    if f==[1,1,0,0,0]:
        return "Two - Thumb and Index Open", 0.89 if ti>0.55 else "Gun Sign - Finger Gun", 0.87
    if f==[1,0,0,0,1]: return "Call Me - Phone Sign / Shaka", 0.91
    if f==[1,1,0,0,1]: return "I Love You - ILY Sign", 0.92
    if f==[0,1,0,0,1]: return "Rock On - Metal Sign", 0.91

    # CLAW
    all_bent = (idx_curl<130 and mid_curl<130 and rng_curl<130 and pnk_curl<130)
    if all_bent and f[1]==1 and f[2]==1 and f[3]==1 and f[4]==1:
        return "Claw Hand - Grab Sign", 0.86

    # DAILY LIFE
    if f[0]==1 and f[1]==0 and f[2]==0 and f[3]==0 and f[4]==0 and thumb_side:
        return "Drinking Sign - Thirsty", 0.83
    if f[1]==1 and f[2]==1 and f[3]==0 and f[4]==0 and f[0]==0 and im<0.28:
        return "Writing Sign - Pen / Note", 0.85

    # FALLBACK
    count = sum(f)
    labels = {0:"Closed Fist",1:"One Finger",2:"Two Fingers",
              3:"Three Fingers",4:"Four Fingers",5:"Open Hand"}
    return labels.get(count,"Custom Sign") + " - Unclassified", 0.60


def process_image(image_bytes: bytes) -> dict:
    """
    Analyse a static image for hand signs.
    Accepts raw image bytes (JPEG / PNG / WEBP / BMP).
    Returns annotated image (base64) + list of detected signs.

    ONLY processes hand signs - rejects images with no visible hand.
    """
    mp_hands = mp.solutions.hands
    mp_draw  = mp.solutions.drawing_utils

    # Decode
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return {"error": "Could not decode image. Please upload a valid JPEG, PNG, or WEBP file.",
                "gestures": [], "hand_count": 0, "annotated_image": ""}

    # Resize to max 800px wide (keep aspect ratio)
    h, w = frame.shape[:2]
    if w > 800:
        scale = 800 / w
        frame = cv2.resize(frame, (800, int(h * scale)))
        h, w = frame.shape[:2]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gestures_found = []

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=4,
        min_detection_confidence=0.50,
        model_complexity=1,
    ) as hands:
        results = hands.process(rgb)

        if not results.multi_hand_landmarks:
            # No hand detected - draw clear message
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 70), (10, 10, 30), -1)
            cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
            cv2.putText(frame, "No hand sign detected in this image",
                        (16, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (80, 80, 220), 2, cv2.LINE_AA)
            cv2.putText(frame, "Please upload an image showing a clear hand sign",
                        (16, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 180), 1, cv2.LINE_AA)
        else:
            for idx, hand_lm in enumerate(results.multi_hand_landmarks):
                # Determine hand side
                side = "Right"
                if results.multi_handedness and idx < len(results.multi_handedness):
                    side = results.multi_handedness[idx].classification[0].label

                # Classify the hand sign
                sign_name, conf = classify_gesture(hand_lm.landmark, side)
                gestures_found.append({
                    "hand":       idx + 1,
                    "side":       side,
                    "gesture":    sign_name,
                    "confidence": round(conf * 100, 1),
                })

                # Draw neon landmarks
                conn_style = mp_draw.DrawingSpec(color=(255, 180, 80), thickness=2, circle_radius=0)
                lm_style   = mp_draw.DrawingSpec(color=(255, 220, 34), thickness=-1, circle_radius=5)
                mp_draw.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS, lm_style, conn_style)

                # Bounding box
                xs = [lm.x * w for lm in hand_lm.landmark]
                ys = [lm.y * h for lm in hand_lm.landmark]
                x1 = max(0, int(min(xs)) - 18)
                y1 = max(0, int(min(ys)) - 18)
                x2 = min(w, int(max(xs)) + 18)
                y2 = min(h, int(max(ys)) + 18)

                # Neon box
                ov = frame.copy()
                cv2.rectangle(ov, (x1, y1), (x2, y2), (255, 180, 80), 3)
                cv2.addWeighted(ov, 0.5, frame, 0.5, 0, frame)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 180, 80), 1)

                # Corner brackets
                blen = 16
                for (cx, cy, dx, dy) in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
                    cv2.line(frame, (cx, cy), (cx + dx*blen, cy), (255, 140, 167), 2)
                    cv2.line(frame, (cx, cy), (cx, cy + dy*blen), (255, 140, 167), 2)

                # Label pill above box
                label   = f"#{idx+1} {sign_name}  {conf*100:.0f}%"
                label_y = max(y1 - 12, 22)
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                cv2.rectangle(frame, (x1, label_y - th - 8), (x1 + tw + 10, label_y + 4),
                              (15, 15, 40), -1)
                cv2.putText(frame, label, (x1 + 5, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 220, 255), 2, cv2.LINE_AA)

                # Side label (Left / Right)
                cv2.putText(frame, side, (x1, y2 + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 220), 1, cv2.LINE_AA)

    # Watermark + border
    cv2.putText(frame, "GestureSense AI - Hand Sign Analyser",
                (w // 2 - 160, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 80, 160), 1, cv2.LINE_AA)
    cv2.rectangle(frame, (1, 1), (w - 2, h - 2), (60, 80, 200), 1)

    # Encode to base64 JPEG
    ret, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    b64 = base64.b64encode(buf.tobytes()).decode("utf-8") if ret else ""

    return {
        "gestures":        gestures_found,
        "hand_count":      len(gestures_found),
        "annotated_image": b64,
        "error":           None,
    }


# =============================================================================
# GESTURE SMOOTHER  -  debounce live predictions
# =============================================================================

class GestureSmoother:
    """
    Rolling-window majority vote to prevent flickering between similar signs.
    A sign is only emitted when it appears in >= threshold fraction of recent frames.
    """
    def __init__(self, window: int = 8, threshold: float = 0.60):
        self._window    = window
        self._threshold = threshold
        self._history: deque = deque(maxlen=window)
        self._stable      = "No Hand Detected"
        self._stable_conf = 0.0

    def update(self, gesture: str, confidence: float) -> Tuple[str, float]:
        self._history.append((gesture, confidence))
        if len(self._history) < 3:
            return self._stable, self._stable_conf
        names = [g for g, _ in self._history]
        most_common, count = Counter(names).most_common(1)[0]
        if count / len(self._history) >= self._threshold:
            avg_conf = float(np.mean([c for g, c in self._history if g == most_common]))
            self._stable      = most_common
            self._stable_conf = avg_conf
        return self._stable, self._stable_conf

    def reset(self):
        self._history.clear()
        self._stable      = "No Hand Detected"
        self._stable_conf = 0.0


# =============================================================================
# HUD RENDERER  -  on-frame overlays for live webcam
# =============================================================================

class HUDRenderer:
    """Draws landmarks, bounding box, and info panel onto webcam frames."""

    NEON_BLUE   = (255, 180,  80)   # BGR
    NEON_PURPLE = (255, 140, 167)
    NEON_CYAN   = (255, 220,  34)
    DARK        = ( 10,  10,  30)

    def __init__(self):
        self._mp_draw  = mp.solutions.drawing_utils
        self._mp_hands = mp.solutions.hands
        self._fps_times: deque = deque(maxlen=30)

    def tick(self):
        self._fps_times.append(time.time())

    def get_fps(self) -> float:
        if len(self._fps_times) < 2:
            return 0.0
        elapsed = self._fps_times[-1] - self._fps_times[0]
        return (len(self._fps_times) - 1) / elapsed if elapsed > 0 else 0.0

    def draw_landmarks(self, frame, hand_landmarks):
        conn_s = self._mp_draw.DrawingSpec(color=self.NEON_BLUE, thickness=2, circle_radius=0)
        lm_s   = self._mp_draw.DrawingSpec(color=self.NEON_CYAN, thickness=-1, circle_radius=5)
        self._mp_draw.draw_landmarks(
            frame, hand_landmarks, self._mp_hands.HAND_CONNECTIONS, lm_s, conn_s)

    def draw_bounding_box(self, frame, hand_landmarks):
        h, w = frame.shape[:2]
        xs = [lm.x * w for lm in hand_landmarks.landmark]
        ys = [lm.y * h for lm in hand_landmarks.landmark]
        x1 = max(0, int(min(xs)) - 15)
        y1 = max(0, int(min(ys)) - 15)
        x2 = min(w, int(max(xs)) + 15)
        y2 = min(h, int(max(ys)) + 15)
        ov = frame.copy()
        cv2.rectangle(ov, (x1, y1), (x2, y2), self.NEON_BLUE, 3)
        cv2.addWeighted(ov, 0.5, frame, 0.5, 0, frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), self.NEON_BLUE, 1)
        blen = 14
        for (cx, cy, dx, dy) in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
            cv2.line(frame, (cx, cy), (cx + dx*blen, cy), self.NEON_PURPLE, 2)
            cv2.line(frame, (cx, cy), (cx, cy + dy*blen), self.NEON_PURPLE, 2)

    def draw_hud(self, frame, gesture: str, confidence: float,
                 hand_count: int, _active: bool):
        h, w = frame.shape[:2]
        fps  = self.get_fps()

        # Bottom panel
        ov = frame.copy()
        cv2.rectangle(ov, (0, h - 90), (w, h), self.DARK, -1)
        cv2.addWeighted(ov, 0.78, frame, 0.22, 0, frame)
        cv2.line(frame, (0, h - 90), (w, h - 90), (60, 60, 120), 1)

        # Sign name
        disp = gesture if gesture != "No Hand Detected" else "Show a hand sign..."
        col  = self.NEON_BLUE if gesture != "No Hand Detected" else (70, 70, 110)
        cv2.putText(frame, disp, (16, h - 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.78, col, 2, cv2.LINE_AA)

        # Confidence text
        if confidence > 0:
            cv2.putText(frame, f"Confidence: {confidence*100:.1f}%",
                        (16, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                        (160, 160, 220), 1, cv2.LINE_AA)

        # Confidence bar
        bx, by, bwm, bh = w - 170, h - 68, 150, 10
        cv2.rectangle(frame, (bx, by), (bx + bwm, by + bh), (40, 40, 70), -1)
        if confidence > 0:
            fill = int(bwm * min(confidence, 1.0))
            bc   = ((80, 220, 100) if confidence >= 0.80
                    else (80, 200, 255) if confidence >= 0.50
                    else (80, 80, 220))
            cv2.rectangle(frame, (bx, by), (bx + fill, by + bh), bc, -1)
        cv2.rectangle(frame, (bx, by), (bx + bwm, by + bh), (80, 80, 140), 1)

        # Top overlays
        cv2.putText(frame, f"FPS:{fps:.0f}", (w - 80, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (100, 200, 100), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Hands:{hand_count}", (16, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (160, 160, 220), 1, cv2.LINE_AA)
        cv2.putText(frame, "GestureSense AI", (w // 2 - 70, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (60, 80, 160), 1, cv2.LINE_AA)

        # Neon border
        cv2.rectangle(frame, (1, 1), (w - 2, h - 2), (60, 80, 200), 1)

    def draw_no_hand(self, frame):
        h, w = frame.shape[:2]
        cv2.putText(frame, "Show a hand sign to the camera",
                    (w // 2 - 170, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (60, 60, 110), 1, cv2.LINE_AA)


# =============================================================================
# GESTURE ENGINE  -  live webcam processing
# =============================================================================

class GestureEngine:
    """
    Thread-safe live hand sign + face expression recognition engine.
    A single daemon thread owns the webcam, MediaPipe Hands, and Face Mesh.
    generate_frames() reads from a shared JPEG buffer - no second camera open.
    """

    def __init__(self):
        self._mp_hands        = mp.solutions.hands
        self._mp_face         = mp.solutions.face_mesh
        self._lock            = threading.Lock()
        # Hand state
        self._latest_gesture  = "No Hand Detected"
        self._latest_conf     = 0.0
        self._hand_count      = 0
        # Face state
        self._latest_face_expr = "No Face Detected"
        self._latest_face_conf = 0.0
        self._face_detected    = False
        # Shared
        self._latest_frame    = None
        self._session_start   = None
        self._total_detected  = 0
        self._conf_history: deque = deque(maxlen=50)
        self._running  = False
        self._thread   = None
        self._smoother = GestureSmoother(window=8, threshold=0.60)
        self._hud      = HUDRenderer()

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_latest(self) -> Tuple[str, float]:
        """Return the latest hand gesture."""
        with self._lock:
            return self._latest_gesture, self._latest_conf

    def get_latest_face(self) -> Tuple[str, float, bool]:
        """Return the latest face expression, confidence, and whether face is detected."""
        with self._lock:
            return self._latest_face_expr, self._latest_face_conf, self._face_detected

    def get_stats(self) -> dict:
        with self._lock:
            dur = int(time.time() - self._session_start) if self._session_start else 0
            avg = float(np.mean(list(self._conf_history))) if self._conf_history else 0.0
            return {
                "total_detected":   self._total_detected,
                "hand_count":       self._hand_count,
                "session_seconds":  dur,
                "avg_confidence":   round(avg * 100, 1),
                "fps":              round(self._hud.get_fps(), 1),
                "face_expression":  self._latest_face_expr,
                "face_detected":    self._face_detected,
            }

    def generate_frames(self) -> Generator[bytes, None, None]:
        """MJPEG generator for Flask streaming Response."""
        self._ensure_running()
        while True:
            with self._lock:
                fb = self._latest_frame
            if fb is None:
                time.sleep(0.03)
                continue
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + fb + b"\r\n"
            time.sleep(0.033)

    def stop(self):
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3)

    # ── Private ────────────────────────────────────────────────────────────────

    def _ensure_running(self):
        if not self._running or (self._thread and not self._thread.is_alive()):
            self._running = True
            self._thread  = threading.Thread(
                target=self._capture_loop, daemon=True, name="GestureCapture")
            self._thread.start()

    def _capture_loop(self):
        """Background thread: webcam -> MediaPipe Hands + Face Mesh -> classify -> JPEG."""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS,          30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

        # Initialise both MediaPipe solutions
        hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.70,
            min_tracking_confidence=0.60,
            model_complexity=1,
        )
        face_mesh = self._mp_face.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,          # enables iris landmarks
            min_detection_confidence=0.55,
            min_tracking_confidence=0.50,
        )

        with self._lock:
            self._session_start = time.time()
            self._smoother.reset()

        prev_hand = "No Hand Detected"

        try:
            while self._running:
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.05)
                    continue

                self._hud.tick()
                frame = cv2.flip(frame, 1)

                # Single RGB conversion shared by both detectors
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False

                hand_results = hands.process(rgb)
                face_results = face_mesh.process(rgb)

                rgb.flags.writeable = True

                # ── HAND DETECTION ─────────────────────────────────────────────
                raw_g, raw_c, hc = "No Hand Detected", 0.0, 0

                if hand_results.multi_hand_landmarks:
                    hc = len(hand_results.multi_hand_landmarks)
                    for idx, hlm in enumerate(hand_results.multi_hand_landmarks):
                        side = "Right"
                        if hand_results.multi_handedness and idx < len(hand_results.multi_handedness):
                            side = hand_results.multi_handedness[idx].classification[0].label
                        self._hud.draw_landmarks(frame, hlm)
                        self._hud.draw_bounding_box(frame, hlm)
                        raw_g, raw_c = classify_gesture(hlm.landmark, side)

                g, c = self._smoother.update(raw_g, raw_c)

                # ── FACE EXPRESSION DETECTION ──────────────────────────────────
                face_expr = "No Face Detected"
                face_conf = 0.0
                face_found = False

                if face_results.multi_face_landmarks:
                    face_found = True
                    flm = face_results.multi_face_landmarks[0].landmark
                    face_expr, face_conf = classify_face_expression(flm)
                    self._draw_face_overlay(frame, face_results.multi_face_landmarks[0], face_expr, face_conf)

                # ── HUD ────────────────────────────────────────────────────────
                self._hud.draw_hud(frame, g, c, hc, True)
                if hc == 0:
                    self._hud.draw_no_hand(frame)

                # Draw face expression label on frame
                self._draw_face_label(frame, face_expr, face_conf, face_found)

                # ── Encode JPEG ────────────────────────────────────────────────
                ret, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
                if not ret:
                    continue

                with self._lock:
                    self._latest_gesture    = g
                    self._latest_conf       = c
                    self._latest_frame      = buf.tobytes()
                    self._hand_count        = hc
                    self._latest_face_expr  = face_expr
                    self._latest_face_conf  = face_conf
                    self._face_detected     = face_found
                    if g != "No Hand Detected" and g != prev_hand:
                        self._total_detected += 1
                        if c > 0:
                            self._conf_history.append(c)

                prev_hand = g

        finally:
            cap.release()
            hands.close()
            face_mesh.close()
            self._running = False

    def _draw_face_overlay(self, frame, face_landmarks, expr: str, conf: float):
        """Draw minimal face mesh contours (eyes, mouth, brows) - not all 468 points."""
        h, w = frame.shape[:2]
        mp_draw = mp.solutions.drawing_utils
        mp_face = mp.solutions.face_mesh

        # Draw only the key contours (eyes, lips, face oval)
        contour_spec = mp_draw.DrawingSpec(
            color=(180, 100, 255), thickness=1, circle_radius=0)
        mp_draw.draw_landmarks(
            frame,
            face_landmarks,
            mp_face.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=contour_spec,
        )

    def _draw_face_label(self, frame, expr: str, conf: float, detected: bool):
        # Draw face expression label at top of frame
        h, w = frame.shape[:2]
        if not detected:
            return

        # Background pill
        label = f"Face: {expr}  {conf*100:.0f}%"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
        x = (w - tw) // 2
        y = 50
        cv2.rectangle(frame, (x - 6, y - th - 6), (x + tw + 6, y + 4),
                      (20, 10, 40), -1)
        cv2.rectangle(frame, (x - 6, y - th - 6), (x + tw + 6, y + 4),
                      (180, 100, 255), 1)
        cv2.putText(frame, label, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 150, 255), 1, cv2.LINE_AA)

# =============================================================================
# HAND SIGN CLASSIFIER  -  45+ gestures including Flower
# =============================================================================
