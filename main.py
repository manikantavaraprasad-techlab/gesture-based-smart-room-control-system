"""
Gesture-Based Smart Room Control System

Features:
- Control light and fan using hand gestures
- Voice feedback for actions
- Auto lock system
- Occupancy detection (auto OFF appliances)

Tech: OpenCV, MediaPipe, Pyttsx3
"""

# main.py 
#
# Key:
# - Fast: model_complexity=0, optimized thresholds
# - Accurate: stricter pose rules + hysteresis (stable frames) + per-action cooldowns
# - Appliance-like: no accidental toggles, predictable step behavior, clean UI
# - FIXED: voice confirmations now reliably speak for every action

import time
import math
import threading
import queue
from collections import deque

import cv2
import mediapipe as mp
import pyttsx3


# -----------------------------
# SETTINGS (FAST + ACCURATE)
# -----------------------------
NO_OCCUPANCY_TIMEOUT_SEC = 25

UNLOCK_HOLD_SEC = 0.65

# Per-action cooldowns (feels appliance-like)
CD_UNLOCK = 0.6
CD_LIGHT = 0.40
CD_FAN_ON = 0.45
CD_FAN_OFF = 0.45
CD_STEP = 0.22

AUTO_LOCK_SEC = 12.0

# Stability (hysteresis)
POSE_STABLE_FRAMES = 3
INDEX_TOGGLE_STABLE_FRAMES = 2

THUMB_DOWN_HOLD_SEC = 0.50

# Palm swipe
PALM_SWIPE_WINDOW_SEC = 0.70
PALM_SWIPE_MIN_DX = 0.16
PALM_SWIPE_MIN_SPEED = 0.24

# Fan ON circle detect
ROT_TRACE_MAXLEN = 24
ROT_MAX_DURATION = 0.95
ROT_MIN_PATH_LEN = 0.13
ROT_MIN_RADIUS = 0.017
ROT_MAX_RVAR = 0.010
ROT_MIN_TOTAL_ANGLE = 3.2

# Step control
STEP_PCT = 20.0
STEP_DY = 0.080
RESET_DY = 0.030

# Smoothing
SMOOTH_ALPHA = 0.40

# UI
EVENT_SHOW_SEC = 1.05
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.95
FONT_THICK = 2
TEXT_COLOR = (245, 245, 245)
TEXT_BG = (18, 18, 18)

# Voice
VOICE_ENABLED = True
VOICE_RATE = 172
VOICE_VOLUME = 1.0
VOICE_SIR_MODE = True
VOICE_DEBUG_PRINT = True
VOICE_DUPLICATE_GAP_SEC = 0.25


# -----------------------------
# STATE
# -----------------------------
LOCKED, UNLOCKED = 0, 1
system_state = LOCKED

light_on = False
fan_on = False
brightness_pct = 0.0
speed_pct = 0.0

last_presence_time = time.time()
last_control_time = time.time()

# per-action last time
t_unlock = 0.0
t_light = 0.0
t_fan_on = 0.0
t_fan_off = 0.0
t_step = 0.0

horns_hold_start = None
thumb_down_hold_start = None

palm_trace = deque(maxlen=60)                 # (x, t)
idx_trace = deque(maxlen=ROT_TRACE_MAXLEN)    # (x, y, t)
pose_history = deque(maxlen=12)

bri_ref_y = None
bri_armed = True
spd_ref_y = None
spd_armed = True

event_msg = ""
event_until = 0.0
room_empty_announced = False

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


# -----------------------------
# Voice Engine (FIXED)
# -----------------------------
class VoiceAssistant:
    """
    Reliable pyttsx3 worker:
    - Engine is created INSIDE the worker thread
    - Engine is also used INSIDE the same thread
    This is much more reliable on Windows/SAPI5.
    """
    def __init__(self, enabled=True, rate=172, volume=1.0, debug=False):
        self.enabled = enabled
        self.rate = rate
        self.volume = volume
        self.debug = debug

        self.q = queue.Queue()
        self.stop_flag = threading.Event()
        self.thread = None

        self.last_spoken_text = ""
        self.last_spoken_time = 0.0

        if self.enabled:
            self.thread = threading.Thread(target=self._worker, daemon=True)
            self.thread.start()

    def _worker(self):
        engine = None
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", self.rate)
            engine.setProperty("volume", self.volume)
        except Exception as e:
            print(f"[WARN] Voice engine init failed inside worker: {e}")
            self.enabled = False
            return

        while not self.stop_flag.is_set():
            try:
                text = self.q.get(timeout=0.2)
            except queue.Empty:
                continue

            if text is None:
                break

            try:
                if self.debug:
                    print(f"[VOICE] {text}")
                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                print(f"[WARN] Voice speak failed: {e}")

        try:
            engine.stop()
        except Exception:
            pass

    def say(self, text: str):
        if not self.enabled or not text:
            return

        t = time.time()

        # small duplicate guard
        if text == self.last_spoken_text and (t - self.last_spoken_time) < VOICE_DUPLICATE_GAP_SEC:
            return

        self.last_spoken_text = text
        self.last_spoken_time = t
        self.q.put(text)

    def shutdown(self):
        self.stop_flag.set()
        if self.enabled and self.thread is not None:
            self.q.put(None)
            self.thread.join(timeout=2.0)


voice = VoiceAssistant(
    enabled=VOICE_ENABLED,
    rate=VOICE_RATE,
    volume=VOICE_VOLUME,
    debug=VOICE_DEBUG_PRINT,
)


# -----------------------------
# Helpers
# -----------------------------
def now():
    return time.time()

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def set_event(msg: str):
    global event_msg, event_until
    event_msg = msg
    event_until = now() + EVENT_SHOW_SEC

def can_fire(last_t: float, cooldown: float) -> bool:
    return (now() - last_t) >= cooldown

def finger_up(lm, tip, pip, margin=0.014):
    # tip higher than pip (y smaller)
    return (lm[tip].y + margin) < lm[pip].y

def finger_down(lm, tip, pip, margin=0.014):
    return (lm[tip].y - margin) > lm[pip].y

def pose_is_stable(name: str, frames: int) -> bool:
    if len(pose_history) < frames:
        return False
    recent = list(pose_history)[-frames:]
    return all(p == name for p in recent)

def palm_center(lm):
    ids = [0, 5, 9, 13, 17]
    x = sum(lm[i].x for i in ids) / len(ids)
    y = sum(lm[i].y for i in ids) / len(ids)
    return x, y

def format_sir_text(text: str) -> str:
    text = text.strip()
    if VOICE_SIR_MODE:
        if text.endswith("."):
            return text[:-1] + ", sir."
        if text.endswith(", sir"):
            return text + "."
        if text.lower().endswith("sir."):
            return text
        return text + ", sir."
    return text

def speak_action(text: str):
    voice.say(format_sir_text(text))

def announce(event_text: str, speech_text: str = None):
    """
    One single path for ALL action feedback:
    - update screen
    - speak voice
    """
    set_event(event_text)
    speak_action(speech_text if speech_text is not None else event_text)


# -----------------------------
# Gesture classification (STRICT)
# -----------------------------
def classify_pose(lm):
    """
    Returns: HORNS, PALM, INDEX_POINT, THUMB_DOWN, OTHER
    Strict rules -> higher accuracy.
    """
    index_up = finger_up(lm, 8, 6, margin=0.012)
    middle_up = finger_up(lm, 12, 10, margin=0.012)
    ring_up = finger_up(lm, 16, 14, margin=0.012)
    pinky_up = finger_up(lm, 20, 18, margin=0.012)

    middle_down = not finger_up(lm, 12, 10, margin=0.006)
    ring_down = not finger_up(lm, 16, 14, margin=0.006)
    pinky_down = not finger_up(lm, 20, 18, margin=0.006)
    index_down = not finger_up(lm, 8, 6, margin=0.006)

    # THUMB_DOWN strict:
    # thumb tip much lower than thumb ip
    thumb_down = finger_down(lm, 4, 3, margin=0.015)

    if thumb_down and index_down and middle_down and ring_down and pinky_down:
        return "THUMB_DOWN"

    # HORNS: index + pinky up, middle + ring down
    if index_up and pinky_up and middle_down and ring_down:
        return "HORNS"

    # PALM: all 4 fingers up
    if index_up and middle_up and ring_up and pinky_up:
        return "PALM"

    # INDEX_POINT: index up, others down
    if index_up and middle_down and ring_down and pinky_down:
        return "INDEX_POINT"

    return "OTHER"


# -----------------------------
# Detectors
# -----------------------------
def detect_open_palm_swipe_direction(palm_trace_deque):
    if len(palm_trace_deque) < 8:
        return None

    t_now = now()
    recent = [(x, t) for (x, t) in palm_trace_deque if (t_now - t) <= PALM_SWIPE_WINDOW_SEC]
    if len(recent) < 8:
        return None

    x0, t0 = recent[0]
    x1, t1 = recent[-1]
    dx = x1 - x0
    dt = max(1e-6, (t1 - t0))
    speed = abs(dx) / dt

    if abs(dx) >= PALM_SWIPE_MIN_DX and speed >= PALM_SWIPE_MIN_SPEED:
        return "RIGHT" if dx > 0 else "LEFT"
    return None


def detect_rotation_fast(idx_trace_deque):
    if len(idx_trace_deque) < 14:
        return False

    t_now = now()
    recent = [(x, y, t) for (x, y, t) in idx_trace_deque if (t_now - t) <= ROT_MAX_DURATION]
    if len(recent) < 14:
        return False

    pts = [(x, y) for x, y, _ in recent]
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)

    rs = [math.hypot(p[0] - cx, p[1] - cy) for p in pts]
    r_mean = sum(rs) / len(rs)
    r_var = sum((r - r_mean) ** 2 for r in rs) / len(rs)

    path_len = 0.0
    for i in range(1, len(pts)):
        path_len += math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])

    angles = [math.atan2(p[1] - cy, p[0] - cx) for p in pts]
    total = 0.0
    for i in range(1, len(angles)):
        da = angles[i] - angles[i - 1]
        while da > math.pi:
            da -= 2 * math.pi
        while da < -math.pi:
            da += 2 * math.pi
        total += da

    return (
        path_len >= ROT_MIN_PATH_LEN and
        r_mean >= ROT_MIN_RADIUS and
        r_var <= ROT_MAX_RVAR and
        abs(total) >= ROT_MIN_TOTAL_ANGLE
    )


# -----------------------------
# Floors (simulated for laptop)
# -----------------------------
def min_brightness_from_ldr(ldr_norm: float) -> float:
    return clamp(35.0 - 25.0 * clamp(ldr_norm, 0.0, 1.0), 10.0, 35.0)

def min_speed_from_temp(temp_c: float) -> float:
    if temp_c < 28.0:
        return 20.0
    if temp_c < 36.0:
        return 20.0 + (temp_c - 28.0) * (40.0 / 8.0)
    return 75.0


# -----------------------------
# Step control (appliance-like)
# -----------------------------
def step_update(current_pct: float, ref_y, armed: bool, y_now: float, step_pct: float):
    if ref_y is None:
        return current_pct, y_now, True, 0

    dy = y_now - ref_y
    if armed:
        if dy <= -STEP_DY:
            return current_pct + step_pct, y_now, False, +1
        if dy >= STEP_DY:
            return current_pct - step_pct, y_now, False, -1
        return current_pct, ref_y, armed, 0

    if abs(dy) <= RESET_DY:
        return current_pct, ref_y, True, 0

    return current_pct, ref_y, armed, 0


# -----------------------------
# Automation
# -----------------------------
def apply_auto_off():
    global light_on, fan_on, brightness_pct, speed_pct, room_empty_announced

    empty = (now() - last_presence_time) > NO_OCCUPANCY_TIMEOUT_SEC
    if empty:
        light_on = False
        fan_on = False
        brightness_pct = 0.0
        speed_pct = 0.0
        if not room_empty_announced:
            announce(
                "Room Empty: Light OFF, Fan OFF",
                "Room is empty. Light and fan turned off."
            )
            room_empty_announced = True
    else:
        room_empty_announced = False


def apply_auto_lock():
    global system_state
    if system_state == UNLOCKED and (now() - last_control_time) > AUTO_LOCK_SEC:
        system_state = LOCKED
        announce("Locked", "System locked.")


# -----------------------------
# UI
# -----------------------------
def draw_top_center_event(frame):
    if now() > event_until or not event_msg:
        return

    text = event_msg
    (tw, th), baseline = cv2.getTextSize(text, FONT, FONT_SCALE, FONT_THICK)
    h, w = frame.shape[:2]
    x = (w - tw) // 2
    y = 45

    pad_x, pad_y = 14, 10
    x1 = max(0, x - pad_x)
    y1 = max(0, y - th - pad_y)
    x2 = min(w - 1, x + tw + pad_x)
    y2 = min(h - 1, y + baseline + pad_y)

    cv2.rectangle(frame, (x1, y1), (x2, y2), TEXT_BG, -1)
    cv2.putText(frame, text, (x, y), FONT, FONT_SCALE, TEXT_COLOR, FONT_THICK, cv2.LINE_AA)


# -----------------------------
# Main
# -----------------------------
def main():
    global system_state, last_presence_time, last_control_time
    global t_unlock, t_light, t_fan_on, t_fan_off, t_step
    global horns_hold_start, thumb_down_hold_start
    global light_on, fan_on, brightness_pct, speed_pct
    global bri_ref_y, bri_armed, spd_ref_y, spd_armed

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Camera not found/cannot open. Try VideoCapture(1).")

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
                    break

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)

                # simulated sensors (replace later)
                ldr_norm = 0.40
                temp_c = 31.0
                min_bri = min_brightness_from_ldr(ldr_norm)
                min_spd = min_speed_from_temp(temp_c)

                if results.multi_hand_landmarks:
                    last_presence_time = now()

                    hand_lm = results.multi_hand_landmarks[0]
                    mp_draw.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)
                    lm = hand_lm.landmark

                    pose = classify_pose(lm)
                    pose_history.append(pose)
                    t = now()

                    # -------- Unlock (Horns hold) --------
                    if system_state == LOCKED and pose == "HORNS":
                        if horns_hold_start is None:
                            horns_hold_start = t
                        elif (t - horns_hold_start) >= UNLOCK_HOLD_SEC and can_fire(t_unlock, CD_UNLOCK):
                            system_state = UNLOCKED
                            t_unlock = t
                            last_control_time = t
                            horns_hold_start = None
                            thumb_down_hold_start = None
                            palm_trace.clear()
                            idx_trace.clear()
                            pose_history.clear()
                            bri_ref_y, bri_armed = None, True
                            spd_ref_y, spd_armed = None, True
                            announce("Unlocked", "System unlocked.")
                    else:
                        horns_hold_start = None

                    # -------- Controls (Unlocked only) --------
                    if system_state == UNLOCKED:

                        # Fan OFF (Thumb down hold)
                        if pose == "THUMB_DOWN" and pose_is_stable("THUMB_DOWN", POSE_STABLE_FRAMES):
                            if thumb_down_hold_start is None:
                                thumb_down_hold_start = t
                            elif (t - thumb_down_hold_start) >= THUMB_DOWN_HOLD_SEC and can_fire(t_fan_off, CD_FAN_OFF):
                                fan_on = False
                                speed_pct = 0.0
                                t_fan_off = t
                                last_control_time = t
                                announce("Fan: OFF", "Fan is off.")
                                thumb_down_hold_start = None
                        else:
                            thumb_down_hold_start = None

                        # Light swipe
                        if pose == "PALM":
                            cx, _ = palm_center(lm)
                            if palm_trace:
                                px, _ = palm_trace[-1]
                                cx = (1.0 - SMOOTH_ALPHA) * px + SMOOTH_ALPHA * cx
                            palm_trace.append((cx, t))

                            if pose_is_stable("PALM", POSE_STABLE_FRAMES) and can_fire(t_light, CD_LIGHT):
                                direction = detect_open_palm_swipe_direction(palm_trace)
                                if direction == "RIGHT":
                                    light_on = True
                                    brightness_pct = clamp(max(brightness_pct, min_bri), min_bri, 100.0)
                                    t_light = t
                                    last_control_time = t
                                    announce("Light: ON", "Light is on.")
                                    palm_trace.clear()
                                elif direction == "LEFT":
                                    light_on = False
                                    brightness_pct = 0.0
                                    t_light = t
                                    last_control_time = t
                                    announce("Light: OFF", "Light is off.")
                                    palm_trace.clear()
                        else:
                            palm_trace.clear()

                        # Fan ON (Index rotation)
                        if pose == "INDEX_POINT":
                            ix, iy = lm[8].x, lm[8].y
                            if idx_trace:
                                px, py, _ = idx_trace[-1]
                                ix = (1.0 - SMOOTH_ALPHA) * px + SMOOTH_ALPHA * ix
                                iy = (1.0 - SMOOTH_ALPHA) * py + SMOOTH_ALPHA * iy
                            idx_trace.append((ix, iy, t))

                            if pose_is_stable("INDEX_POINT", INDEX_TOGGLE_STABLE_FRAMES) and can_fire(t_fan_on, CD_FAN_ON):
                                if detect_rotation_fast(idx_trace):
                                    fan_on = True
                                    speed_pct = clamp(max(speed_pct, min_spd), min_spd, 100.0)
                                    t_fan_on = t
                                    last_control_time = t
                                    announce("Fan: ON", "Fan is on.")
                                    idx_trace.clear()
                        else:
                            idx_trace.clear()

                        # Brightness steps (only if light ON)
                        if light_on and pose == "PALM" and pose_is_stable("PALM", POSE_STABLE_FRAMES) and can_fire(t_step, CD_STEP):
                            _, py = palm_center(lm)
                            prev = brightness_pct
                            brightness_pct, bri_ref_y, bri_armed, step_dir = step_update(
                                brightness_pct, bri_ref_y, bri_armed, py, STEP_PCT
                            )
                            if step_dir != 0:
                                brightness_pct = clamp(brightness_pct, min_bri, 100.0)
                                if brightness_pct != prev:
                                    t_step = t
                                    last_control_time = t
                                    announce(
                                        f"Brightness: {brightness_pct:.0f}%",
                                        f"Brightness set to {brightness_pct:.0f} percent."
                                    )
                        else:
                            bri_ref_y, bri_armed = None, True

                        # Speed steps (only if fan ON)
                        if fan_on and pose == "INDEX_POINT" and pose_is_stable("INDEX_POINT", POSE_STABLE_FRAMES) and can_fire(t_step, CD_STEP):
                            prev = speed_pct
                            iy = lm[8].y
                            speed_pct, spd_ref_y, spd_armed, step_dir = step_update(
                                speed_pct, spd_ref_y, spd_armed, iy, STEP_PCT
                            )
                            if step_dir != 0:
                                speed_pct = clamp(speed_pct, min_spd, 100.0)
                                if speed_pct != prev:
                                    t_step = t
                                    last_control_time = t
                                    announce(
                                        f"Speed: {speed_pct:.0f}%",
                                        f"Fan speed set to {speed_pct:.0f} percent."
                                    )
                        else:
                            spd_ref_y, spd_armed = None, True

                else:
                    pose_history.clear()
                    horns_hold_start = None
                    thumb_down_hold_start = None
                    palm_trace.clear()
                    idx_trace.clear()
                    bri_ref_y, bri_armed = None, True
                    spd_ref_y, spd_armed = None, True

                apply_auto_off()
                apply_auto_lock()
                draw_top_center_event(frame)

                cv2.imshow("Smart Room Control (v11 + Reliable Voice)", frame)
                if (cv2.waitKey(1) & 0xFF) == 27:
                    break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        voice.shutdown()


if __name__ == "__main__":
    main()
