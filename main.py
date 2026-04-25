# main.py (v16 — Face Recognition + Gender Detection + Gesture Control)
#
# Complete system with:
#   - Face detection for person presence (replaces hand-only detection)
#   - Gender classification (Male/Female) using DNN
#   - Dynamic honorific: "sir" for male, "madam" for female
#   - All gesture controls from v15
#   - Room empty logic using face presence
#   - Jarvis voice spectrum via WebSocket
#
# How to run:
#   1. pip install opencv-python mediapipe websockets numpy
#   2. python main.py          (models auto-download on first run)
#   3. Open visualizer.html in Chrome/Edge
#   4. Show face to camera → system detects gender
#   5. Do hand gestures to control devices
#   6. Press ESC to quit

import time
import math
import logging
from collections import deque
from typing import Optional, Tuple

import cv2
import mediapipe as mp

from face_gender import FaceGenderDetector, FaceResult
from voice_ws_bridge import WebVoiceAssistant

# ─────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("SmartRoom")


# ─────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────

# Occupancy
NO_OCCUPANCY_WARN_SEC    = 15
NO_OCCUPANCY_TIMEOUT_SEC = 25

# Unlock
UNLOCK_HOLD_SEC = 0.65

# Cooldowns
CD_UNLOCK  = 0.6
CD_LIGHT   = 0.40
CD_FAN_ON  = 0.45
CD_FAN_OFF = 0.45
CD_BRI_STEP = 0.30       # separate brightness step cooldown
CD_SPD_STEP = 0.30       # separate speed step cooldown

# Auto-lock
AUTO_LOCK_WARN_SEC = 14.0
AUTO_LOCK_SEC      = 18.0

# Stability
POSE_STABLE_FRAMES        = 4
INDEX_TOGGLE_STABLE_FRAMES = 3

THUMB_DOWN_HOLD_SEC = 0.55

# Palm swipe
PALM_SWIPE_WINDOW_SEC = 0.70
PALM_SWIPE_MIN_DX     = 0.17
PALM_SWIPE_MIN_SPEED  = 0.26

# Fan rotation detection
ROT_TRACE_MAXLEN    = 28
ROT_MAX_DURATION    = 1.00
ROT_MIN_PATH_LEN    = 0.14
ROT_MIN_RADIUS      = 0.018
ROT_MAX_RVAR        = 0.009
ROT_MIN_TOTAL_ANGLE = 3.4

# Step control
STEP_PCT = 20.0
STEP_DY  = 0.085
RESET_DY = 0.030

# Smoothing
SMOOTH_ALPHA = 0.38

# Face detection frequency (not every frame — expensive)
FACE_DETECT_INTERVAL_SEC = 0.3

# UI
EVENT_SHOW_SEC = 1.20
FONT       = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.92
FONT_THICK = 2
TEXT_COLOR  = (245, 245, 245)
TEXT_BG    = (18, 18, 18)


# ─────────────────────────────────────────
# STATE
# ─────────────────────────────────────────

LOCKED, UNLOCKED = 0, 1

system_state     = LOCKED
light_on         = False
fan_on           = False
brightness_pct   = 0.0
speed_pct        = 0.0

last_presence_time = time.monotonic()
last_control_time  = time.monotonic()

t_unlock   = 0.0
t_light    = 0.0
t_fan_on   = 0.0
t_fan_off  = 0.0
t_bri_step = 0.0          # separate from speed
t_spd_step = 0.0          # separate from brightness

horns_hold_start      = None
thumb_down_hold_start = None

palm_trace   = deque(maxlen=60)
idx_trace    = deque(maxlen=ROT_TRACE_MAXLEN)
pose_history = deque(maxlen=12)

bri_ref_y = None
bri_armed = True
spd_ref_y = None
spd_armed = True

event_msg   = ""
event_until = 0.0

# Room empty tracking
room_empty_announced = False
room_empty_warned    = False
auto_lock_warned     = False

# Face/gender state
current_face_result = FaceResult()
last_face_detect_time = 0.0
current_honorific     = "sir"      # default until face detected
gender_announced      = False      # track if we announced gender change

mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils


# ─────────────────────────────────────────
# VOICE ENGINE
# ─────────────────────────────────────────

voice = WebVoiceAssistant(
    enabled=True,
    debug=True,
    port=9600,
)


# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────

def now() -> float:
    return time.monotonic()


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def set_event(msg: str) -> None:
    global event_msg, event_until
    event_msg = msg
    event_until = now() + EVENT_SHOW_SEC


def can_fire(last_t: float, cooldown: float) -> bool:
    return (now() - last_t) >= cooldown


def finger_up(lm, tip: int, pip: int, margin: float = 0.014) -> bool:
    return (lm[tip].y + margin) < lm[pip].y


def finger_down(lm, tip: int, pip: int, margin: float = 0.014) -> bool:
    return (lm[tip].y - margin) > lm[pip].y


def pose_is_stable(name: str, frames: int) -> bool:
    if len(pose_history) < frames:
        return False
    return all(p == name for p in list(pose_history)[-frames:])


def palm_center(lm) -> Tuple[float, float]:
    ids = [0, 5, 9, 13, 17]
    x = sum(lm[i].x for i in ids) / len(ids)
    y = sum(lm[i].y for i in ids) / len(ids)
    return x, y


def any_device_on() -> bool:
    return light_on or fan_on or brightness_pct > 0.0 or speed_pct > 0.0


def get_absence_seconds() -> float:
    return now() - last_presence_time


def format_honorific(text: str) -> str:
    """
    Append current honorific (sir/madam) to speech text.
    Handles edge cases: already ends with honorific, punctuation, etc.
    """
    text = text.strip()
    hon = current_honorific

    # Already ends with honorific
    lower = text.lower()
    if lower.endswith(f", {hon}.") or lower.endswith(f", {hon}"):
        if not text.endswith("."):
            return text + "."
        return text

    # Remove trailing punctuation, add honorific
    text = text.rstrip(".,!?")
    return f"{text}, {hon}."


def speak_action(text: str, action_text: str = None) -> None:
    """Speak with dynamic honorific and display action."""
    voice.say(format_honorific(text), action_text=action_text)


def announce(event_text: str, speech_text: str = None) -> None:
    """Set visual event + speak."""
    set_event(event_text)
    final_speech = speech_text if speech_text is not None else event_text
    speak_action(final_speech, action_text=event_text)


# ─────────────────────────────────────────
# GESTURE CLASSIFIER (with named landmarks)
# ─────────────────────────────────────────

# MediaPipe hand landmark indices
INDEX_TIP,  INDEX_PIP  = 8, 6
MIDDLE_TIP, MIDDLE_PIP = 12, 10
RING_TIP,   RING_PIP   = 16, 14
PINKY_TIP,  PINKY_PIP  = 20, 18
THUMB_TIP,  THUMB_IP   = 4, 3


def classify_pose(lm) -> str:
    index_up   = finger_up(lm, INDEX_TIP,  INDEX_PIP,  margin=0.012)
    middle_up  = finger_up(lm, MIDDLE_TIP, MIDDLE_PIP, margin=0.012)
    ring_up    = finger_up(lm, RING_TIP,   RING_PIP,   margin=0.012)
    pinky_up   = finger_up(lm, PINKY_TIP,  PINKY_PIP,  margin=0.012)

    index_down  = not finger_up(lm, INDEX_TIP,  INDEX_PIP,  margin=0.006)
    middle_down = not finger_up(lm, MIDDLE_TIP, MIDDLE_PIP, margin=0.006)
    ring_down   = not finger_up(lm, RING_TIP,   RING_PIP,   margin=0.006)
    pinky_down  = not finger_up(lm, PINKY_TIP,  PINKY_PIP,  margin=0.006)

    thumb_dn = finger_down(lm, THUMB_TIP, THUMB_IP, margin=0.015)

    if thumb_dn and index_down and middle_down and ring_down and pinky_down:
        return "THUMB_DOWN"
    if index_up and pinky_up and middle_down and ring_down:
        return "HORNS"
    if index_up and middle_up and ring_up and pinky_up:
        return "PALM"
    if index_up and middle_down and ring_down and pinky_down:
        return "INDEX_POINT"
    return "OTHER"


# ─────────────────────────────────────────
# MOTION DETECTORS
# ─────────────────────────────────────────

def detect_palm_swipe(trace: deque) -> Optional[str]:
    if len(trace) < 8:
        return None
    t_now = now()
    recent = [(x, t) for x, t in trace if (t_now - t) <= PALM_SWIPE_WINDOW_SEC]
    if len(recent) < 8:
        return None
    x0, t0 = recent[0]
    x1, t1 = recent[-1]
    dx = x1 - x0
    dt = max(1e-6, t1 - t0)
    speed = abs(dx) / dt
    if abs(dx) >= PALM_SWIPE_MIN_DX and speed >= PALM_SWIPE_MIN_SPEED:
        return "RIGHT" if dx > 0 else "LEFT"
    return None


def detect_rotation(trace: deque) -> bool:
    if len(trace) < 14:
        return False
    t_now = now()
    recent = [(x, y, t) for x, y, t in trace if (t_now - t) <= ROT_MAX_DURATION]
    if len(recent) < 14:
        return False

    pts = [(x, y) for x, y, _ in recent]
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    rs = [math.hypot(p[0] - cx, p[1] - cy) for p in pts]
    r_m = sum(rs) / len(rs)
    r_v = sum((r - r_m) ** 2 for r in rs) / len(rs)

    path = sum(
        math.hypot(pts[i][0] - pts[i-1][0], pts[i][1] - pts[i-1][1])
        for i in range(1, len(pts))
    )

    angles = [math.atan2(p[1] - cy, p[0] - cx) for p in pts]
    total = 0.0
    for i in range(1, len(angles)):
        da = angles[i] - angles[i-1]
        while da > math.pi: da -= 2 * math.pi
        while da < -math.pi: da += 2 * math.pi
        total += da

    return (
        path >= ROT_MIN_PATH_LEN
        and r_m >= ROT_MIN_RADIUS
        and r_v <= ROT_MAX_RVAR
        and abs(total) >= ROT_MIN_TOTAL_ANGLE
    )


# ─────────────────────────────────────────
# SENSOR FLOOR MAPPING
# ─────────────────────────────────────────

def min_brightness_from_ldr(ldr: float) -> float:
    return clamp(35.0 - 25.0 * clamp(ldr, 0.0, 1.0), 10.0, 35.0)


def min_speed_from_temp(temp: float) -> float:
    if temp < 28.0:
        return 20.0
    if temp < 36.0:
        return 20.0 + (temp - 28.0) * (40.0 / 8.0)
    return 75.0


# ─────────────────────────────────────────
# STEP CONTROLLER
# ─────────────────────────────────────────

def step_update(
    current_pct: float,
    ref_y: Optional[float],
    armed: bool,
    y_now: float,
    step_pct: float,
) -> Tuple[float, Optional[float], bool, int]:
    if ref_y is None:
        return current_pct, y_now, True, 0
    dy = y_now - ref_y
    if armed:
        if dy <= -STEP_DY:
            return current_pct + step_pct, y_now, False, +1
        if dy >= STEP_DY:
            return current_pct - step_pct, y_now, False, -1
        return current_pct, ref_y, True, 0
    if abs(dy) <= RESET_DY:
        return current_pct, ref_y, True, 0
    return current_pct, ref_y, False, 0


# ─────────────────────────────────────────
# PERSON PRESENCE (face-based)
# ─────────────────────────────────────────

def person_is_present() -> bool:
    """
    Person is present if:
    - Face detected this frame, OR
    - Hand detected this frame (face may be turned away but hand visible)
    """
    return current_face_result.face_detected


def update_face_detection(
    frame, detector: FaceGenderDetector
) -> None:
    """
    Run face detection at controlled interval.
    Updates current_face_result and current_honorific.
    """
    global current_face_result, last_face_detect_time
    global current_honorific, gender_announced

    t = now()
    if (t - last_face_detect_time) < FACE_DETECT_INTERVAL_SEC:
        return

    last_face_detect_time = t
    result = detector.detect(frame)
    current_face_result = result

    # Update honorific based on gender
    if result.face_detected and result.gender != "Unknown":
        new_honorific = result.honorific

        if new_honorific != current_honorific:
            old = current_honorific
            current_honorific = new_honorific
            voice.honorific = new_honorific

            log.info(
                f"Gender: {result.gender} "
                f"({result.gender_confidence:.0%}) → "
                f"honorific changed: {old} → {new_honorific}"
            )

            # Announce gender change only if system is unlocked
            # and only the first time or on genuine change
            if system_state == UNLOCKED and gender_announced:
                # Gender changed mid-session
                log.info(f"Honorific update: {old} → {new_honorific}")

        if not gender_announced and result.gender_confidence > 0.70:
            gender_announced = True
            log.info(
                f"Person identified: {result.gender} → "
                f"using '{current_honorific}'"
            )


# ─────────────────────────────────────────
# ROOM EMPTY LOGIC
# ─────────────────────────────────────────

def apply_room_empty_logic(hand_detected: bool) -> None:
    """
    Two-stage room empty detection using FACE presence.
    Hand detection extends presence (face may be turned away).
    """
    global light_on, fan_on, brightness_pct, speed_pct
    global system_state
    global room_empty_announced, room_empty_warned
    global last_presence_time, gender_announced

    # Update presence based on face OR hand
    if current_face_result.face_detected or hand_detected:
        last_presence_time = now()

    absence = get_absence_seconds()

    is_present = (absence < 2.0)  # present if seen within 2 seconds

    if is_present:
        if room_empty_announced:
            log.info("PERSON RETURNED — Welcome back")
            announce("Welcome Back", "Welcome back. System ready.")
            room_empty_announced = False
            room_empty_warned = False
            gender_announced = False  # re-detect gender
        elif room_empty_warned:
            room_empty_warned = False
        return

    # Stage 1: Warning at 15 seconds
    if (
        absence >= NO_OCCUPANCY_WARN_SEC
        and not room_empty_warned
        and not room_empty_announced
    ):
        if any_device_on() or system_state == UNLOCKED:
            room_empty_warned = True
            remaining = int(NO_OCCUPANCY_TIMEOUT_SEC - absence)
            log.info(f"WARNING: No person for {absence:.0f}s")
            announce(
                f"No Person — OFF in {remaining}s",
                f"No person detected. Devices will turn off in {remaining} seconds.",
            )

    # Stage 2: Auto-OFF at 25 seconds
    if absence >= NO_OCCUPANCY_TIMEOUT_SEC and not room_empty_announced:
        had_devices = any_device_on()
        was_unlocked = system_state == UNLOCKED

        light_on = False
        fan_on = False
        brightness_pct = 0.0
        speed_pct = 0.0
        system_state = LOCKED

        room_empty_announced = True
        room_empty_warned = False

        if had_devices or was_unlocked:
            log.info("AUTO-OFF: Room empty")
            announce(
                "Room Empty — All OFF + Locked",
                "Room is empty. All devices turned off. System locked.",
            )


def apply_auto_lock() -> None:
    """Auto-lock after inactivity (even if person present)."""
    global system_state, auto_lock_warned

    if system_state != UNLOCKED:
        auto_lock_warned = False
        return

    idle = now() - last_control_time

    if idle >= AUTO_LOCK_WARN_SEC and not auto_lock_warned:
        auto_lock_warned = True
        remaining = int(AUTO_LOCK_SEC - idle)
        log.info(f"AUTO-LOCK WARNING: Locking in {remaining}s")
        announce(
            f"Locking in {remaining}s",
            f"System will lock in {remaining} seconds.",
        )

    if idle >= AUTO_LOCK_SEC:
        system_state = LOCKED
        auto_lock_warned = False
        log.info("AUTO-LOCK: System locked")
        announce("Locked", "System locked due to inactivity.")


# ─────────────────────────────────────────
# UI DRAWING
# ─────────────────────────────────────────

def draw_top_center_event(frame) -> None:
    if now() > event_until or not event_msg:
        return
    text = event_msg
    (tw, th), baseline = cv2.getTextSize(text, FONT, FONT_SCALE, FONT_THICK)
    h, w = frame.shape[:2]
    x = (w - tw) // 2
    y = 45
    pad_x, pad_y = 14, 10
    cv2.rectangle(
        frame,
        (max(0, x - pad_x), max(0, y - th - pad_y)),
        (min(w - 1, x + tw + pad_x), min(h - 1, y + baseline + pad_y)),
        TEXT_BG, -1,
    )
    cv2.putText(frame, text, (x, y), FONT, FONT_SCALE, TEXT_COLOR, FONT_THICK, cv2.LINE_AA)


def draw_room_status(frame, hand_detected: bool) -> None:
    h, w = frame.shape[:2]
    absence = get_absence_seconds()
    is_present = (absence < 2.0)

    # ── Person indicator (top-left) ──
    if is_present:
        color = (0, 220, 80)
        if current_face_result.face_detected:
            label = f"PERSON: {current_face_result.gender} [{current_honorific.upper()}]"
        elif hand_detected:
            label = "PERSON DETECTED (hand)"
        else:
            label = "PERSON DETECTED"
        cv2.circle(frame, (25, 85), 8, color, -1)
        cv2.putText(frame, label, (42, 92), FONT, 0.55, color, 1, cv2.LINE_AA)
    else:
        if absence < NO_OCCUPANCY_WARN_SEC:
            color = (0, 180, 255)
            label = f"NO PERSON ({absence:.0f}s)"
        elif absence < NO_OCCUPANCY_TIMEOUT_SEC:
            color = (0, 80, 255)
            remaining = int(NO_OCCUPANCY_TIMEOUT_SEC - absence)
            label = f"NO PERSON — OFF in {remaining}s"
        else:
            color = (0, 0, 220)
            label = "ROOM EMPTY — ALL OFF"
        cv2.circle(frame, (25, 85), 8, color, -1)
        cv2.putText(frame, label, (42, 92), FONT, 0.55, color, 1, cv2.LINE_AA)

        # Pulsing border
        if NO_OCCUPANCY_WARN_SEC <= absence < NO_OCCUPANCY_TIMEOUT_SEC:
            pulse = int(128 + 127 * math.sin(now() * 6))
            cv2.rectangle(frame, (2, 2), (w - 3, h - 3), (0, 0, pulse), 3)

    # ── System state (top-right) ──
    if system_state == UNLOCKED:
        s_color = (0, 220, 80)
        s_label = "UNLOCKED"
    else:
        s_color = (0, 80, 220)
        s_label = "LOCKED"
    (tw, _), _ = cv2.getTextSize(s_label, FONT, 0.6, 2)
    cv2.putText(frame, s_label, (w - tw - 15, 30), FONT, 0.6, s_color, 2, cv2.LINE_AA)

    # ── Honorific indicator (below system state) ──
    hon_label = f"MODE: {current_honorific.upper()}"
    hon_color = (255, 200, 100) if current_honorific == "sir" else (220, 150, 255)
    (tw2, _), _ = cv2.getTextSize(hon_label, FONT, 0.45, 1)
    cv2.putText(frame, hon_label, (w - tw2 - 15, 52), FONT, 0.45, hon_color, 1, cv2.LINE_AA)

    # ── Device status (bottom) ──
    y_pos = h - 18
    items = []
    if light_on:
        items.append((f"LIGHT ON {brightness_pct:.0f}%", (0, 220, 255)))
    else:
        items.append(("LIGHT OFF", (80, 80, 80)))
    if fan_on:
        items.append((f"FAN ON {speed_pct:.0f}%", (0, 255, 200)))
    else:
        items.append(("FAN OFF", (80, 80, 80)))

    x_pos = 15
    for label, col in items:
        cv2.putText(frame, label, (x_pos, y_pos), FONT, 0.50, col, 1, cv2.LINE_AA)
        (tw, _), _ = cv2.getTextSize(label, FONT, 0.50, 1)
        x_pos += tw + 30

    # ── Absence timer bar ──
    if not is_present and absence > 2.0:
        bar_w, bar_h = 150, 8
        bar_x = w - bar_w - 15
        bar_y = h - 15
        progress = min(1.0, absence / NO_OCCUPANCY_TIMEOUT_SEC)
        filled = int(progress * bar_w)
        if progress < 0.6:
            bar_color = (0, 180, 100)
        elif progress < 0.85:
            bar_color = (0, 140, 255)
        else:
            bar_color = (0, 0, 220)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (40, 40, 40), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled, bar_y + bar_h), bar_color, -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), 1)
        cv2.putText(frame, "AUTO-OFF", (bar_x, bar_y - 5), FONT, 0.35, (120, 120, 120), 1, cv2.LINE_AA)


# ─────────────────────────────────────────
# CAMERA
# ─────────────────────────────────────────

def open_camera(index: int = 0, retries: int = 5) -> cv2.VideoCapture:
    for attempt in range(retries):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            log.info(f"Camera opened (index={index}, attempt={attempt + 1})")
            return cap
        cap.release()   # release failed capture to avoid leak
        log.warning(f"Camera open failed (attempt {attempt + 1}/{retries})")
        time.sleep(0.5)
    raise RuntimeError(f"Cannot open camera index {index} after {retries} attempts.")


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

def main() -> None:
    global system_state, last_presence_time, last_control_time
    global t_unlock, t_light, t_fan_on, t_fan_off, t_bri_step, t_spd_step
    global horns_hold_start, thumb_down_hold_start
    global light_on, fan_on, brightness_pct, speed_pct
    global bri_ref_y, bri_armed, spd_ref_y, spd_armed
    global auto_lock_warned, current_honorific, gender_announced

    # Initialize face detector (downloads models on first run)
    log.info("Initializing face + gender detector...")
    face_detector = FaceGenderDetector()

    cap = open_camera(0)

    print("=" * 62)
    print("  SMART ROOM CONTROL v16")
    print("  Face Recognition + Gender Detection + Gesture Control")
    print("=" * 62)
    print("  1. Open visualizer.html in Chrome/Edge")
    print("  2. Show your face → gender detected → sir/madam")
    print("  3. Do hand gestures to control devices")
    print("  4. Press ESC in camera window to quit")
    print("-" * 62)
    print(f"  Room empty warning  at {NO_OCCUPANCY_WARN_SEC}s")
    print(f"  Room empty auto-off at {NO_OCCUPANCY_TIMEOUT_SEC}s")
    print(f"  Auto-lock           at {AUTO_LOCK_SEC}s idle")
    print("=" * 62)

    try:
        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.72,
            min_tracking_confidence=0.74,
        ) as hands:

            while True:
                ok, frame = cap.read()
                if not ok:
                    log.warning("Frame read failed — attempting recovery...")
                    cap.release()
                    time.sleep(1.0)
                    try:
                        cap = open_camera(0)
                    except RuntimeError:
                        log.error("Camera recovery failed. Exiting.")
                        break
                    continue

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # ── Face detection (throttled) ──
                update_face_detection(frame, face_detector)

                # ── Draw face box ──
                face_detector.draw_face_box(frame, current_face_result)

                # ── Hand detection ──
                hand_results = hands.process(rgb)
                hand_detected = bool(hand_results.multi_hand_landmarks)

                # ── Sensor readings (simulated) ──
                ldr_norm = 0.40
                temp_c = 31.0
                min_bri = min_brightness_from_ldr(ldr_norm)
                min_spd = min_speed_from_temp(temp_c)

                t = now()

                if hand_detected:
                    last_presence_time = t

                    hand_lm = hand_results.multi_hand_landmarks[0]
                    mp_draw.draw_landmarks(
                        frame, hand_lm, mp_hands.HAND_CONNECTIONS
                    )
                    lm = hand_lm.landmark

                    pose = classify_pose(lm)
                    pose_history.append(pose)

                    # ── UNLOCK (Horns hold) ──
                    if system_state == LOCKED:
                        if pose == "HORNS":
                            if horns_hold_start is None:
                                horns_hold_start = t
                            elif (
                                (t - horns_hold_start) >= UNLOCK_HOLD_SEC
                                and can_fire(t_unlock, CD_UNLOCK)
                            ):
                                system_state = UNLOCKED
                                t_unlock = t
                                last_control_time = t
                                auto_lock_warned = False
                                horns_hold_start = None
                                thumb_down_hold_start = None
                                palm_trace.clear()
                                idx_trace.clear()
                                pose_history.clear()
                                bri_ref_y, bri_armed = None, True
                                spd_ref_y, spd_armed = None, True
                                log.info("UNLOCKED")

                                # Use detected gender in unlock message
                                if current_face_result.face_detected:
                                    announce(
                                        "Unlocked",
                                        f"System unlocked. Welcome, {current_honorific}.",
                                    )
                                else:
                                    announce("Unlocked", "System unlocked.")
                        else:
                            horns_hold_start = None

                    # ── CONTROLS (unlocked) ──
                    if system_state == UNLOCKED:

                        # Fan OFF — thumb down hold
                        if pose == "THUMB_DOWN" and pose_is_stable(
                            "THUMB_DOWN", POSE_STABLE_FRAMES
                        ):
                            if thumb_down_hold_start is None:
                                thumb_down_hold_start = t
                            elif (
                                (t - thumb_down_hold_start) >= THUMB_DOWN_HOLD_SEC
                                and can_fire(t_fan_off, CD_FAN_OFF)
                            ):
                                fan_on = False
                                speed_pct = 0.0
                                t_fan_off = t
                                last_control_time = t
                                thumb_down_hold_start = None
                                log.info("FAN OFF")
                                announce("Fan: OFF", "Fan off.")
                        else:
                            thumb_down_hold_start = None

                        # Light swipe
                        if pose == "PALM":
                            cx, _ = palm_center(lm)
                            if palm_trace:
                                px, _ = palm_trace[-1]
                                cx = (1.0 - SMOOTH_ALPHA) * px + SMOOTH_ALPHA * cx
                            palm_trace.append((cx, t))

                            if pose_is_stable("PALM", POSE_STABLE_FRAMES) and can_fire(
                                t_light, CD_LIGHT
                            ):
                                direction = detect_palm_swipe(palm_trace)
                                if direction == "RIGHT":
                                    light_on = True
                                    brightness_pct = clamp(
                                        max(brightness_pct, min_bri), min_bri, 100.0
                                    )
                                    t_light = t
                                    last_control_time = t
                                    log.info("LIGHT ON")
                                    announce("Light: ON", "Light on.")
                                    palm_trace.clear()
                                elif direction == "LEFT":
                                    light_on = False
                                    brightness_pct = 0.0
                                    t_light = t
                                    last_control_time = t
                                    log.info("LIGHT OFF")
                                    announce("Light: OFF", "Light off.")
                                    palm_trace.clear()
                        else:
                            palm_trace.clear()

                        # Fan ON — index rotation
                        if pose == "INDEX_POINT":
                            ix, iy = lm[INDEX_TIP].x, lm[INDEX_TIP].y
                            if idx_trace:
                                px, py, _ = idx_trace[-1]
                                ix = (1.0 - SMOOTH_ALPHA) * px + SMOOTH_ALPHA * ix
                                iy = (1.0 - SMOOTH_ALPHA) * py + SMOOTH_ALPHA * iy
                            idx_trace.append((ix, iy, t))

                            if (
                                pose_is_stable(
                                    "INDEX_POINT", INDEX_TOGGLE_STABLE_FRAMES
                                )
                                and can_fire(t_fan_on, CD_FAN_ON)
                                and detect_rotation(idx_trace)
                            ):
                                fan_on = True
                                speed_pct = clamp(
                                    max(speed_pct, min_spd), min_spd, 100.0
                                )
                                t_fan_on = t
                                last_control_time = t
                                log.info("FAN ON")
                                announce("Fan: ON", "Fan on.")
                                idx_trace.clear()
                        else:
                            idx_trace.clear()

                        # Brightness steps (separate cooldown)
                        if (
                            light_on
                            and pose == "PALM"
                            and pose_is_stable("PALM", POSE_STABLE_FRAMES)
                            and can_fire(t_bri_step, CD_BRI_STEP)
                        ):
                            _, py = palm_center(lm)
                            prev = brightness_pct
                            brightness_pct, bri_ref_y, bri_armed, step_dir = (
                                step_update(
                                    brightness_pct, bri_ref_y, bri_armed,
                                    py, STEP_PCT,
                                )
                            )
                            if step_dir != 0:
                                brightness_pct = clamp(brightness_pct, min_bri, 100.0)
                                if brightness_pct != prev:
                                    t_bri_step = t
                                    last_control_time = t
                                    log.info(
                                        f"BRIGHTNESS {brightness_pct:.0f}%"
                                    )
                                    announce(
                                        f"Brightness: {brightness_pct:.0f}%",
                                        f"Brightness {brightness_pct:.0f}.",
                                    )
                        else:
                            bri_ref_y, bri_armed = None, True

                        # Speed steps (separate cooldown)
                        if (
                            fan_on
                            and pose == "INDEX_POINT"
                            and pose_is_stable("INDEX_POINT", POSE_STABLE_FRAMES)
                            and can_fire(t_spd_step, CD_SPD_STEP)
                        ):
                            iy = lm[INDEX_TIP].y
                            prev = speed_pct
                            speed_pct, spd_ref_y, spd_armed, step_dir = (
                                step_update(
                                    speed_pct, spd_ref_y, spd_armed,
                                    iy, STEP_PCT,
                                )
                            )
                            if step_dir != 0:
                                speed_pct = clamp(speed_pct, min_spd, 100.0)
                                if speed_pct != prev:
                                    t_spd_step = t
                                    last_control_time = t
                                    log.info(f"SPEED {speed_pct:.0f}%")
                                    announce(
                                        f"Speed: {speed_pct:.0f}%",
                                        f"Speed {speed_pct:.0f}.",
                                    )
                        else:
                            spd_ref_y, spd_armed = None, True

                else:
                    # No hand
                    pose_history.clear()
                    horns_hold_start = None
                    thumb_down_hold_start = None
                    palm_trace.clear()
                    idx_trace.clear()
                    bri_ref_y, bri_armed = None, True
                    spd_ref_y, spd_armed = None, True

                # ── Room empty + auto-lock ──
                apply_room_empty_logic(hand_detected)
                apply_auto_lock()

                # ── Draw UI ──
                draw_room_status(frame, hand_detected)
                draw_top_center_event(frame)

                cv2.imshow("Smart Room Control v16", frame)
                if (cv2.waitKey(1) & 0xFF) == 27:
                    break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        voice.shutdown()
        log.info("Shutdown complete.")


if __name__ == "__main__":
    main()
