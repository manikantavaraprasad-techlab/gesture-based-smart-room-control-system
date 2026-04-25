# face_gender.py
# Face detection + gender classification module
#
# Models auto-downloaded on first run.
# Uses working mirrors with fallback URLs.
# Safe CPU/CUDA backend detection — no crash if CUDA unavailable.

import os
import time
import urllib.request
import urllib.error
import logging
from dataclasses import dataclass
from typing import Optional, Tuple, List

import cv2
import numpy as np

log = logging.getLogger("FaceGender")

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

# ─────────────────────────────────────────
# MODEL DEFINITIONS WITH FALLBACK URLS
# ─────────────────────────────────────────
MODELS = {
    "face_proto": {
        "file": "deploy.prototxt",
        "urls": [
            "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
            "https://raw.githubusercontent.com/Whywolk/face-detection-in-video/main/deploy.prototxt",
        ],
    },
    "face_model": {
        "file": "res10_300x300_ssd_iter_140000.caffemodel",
        "urls": [
            "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
            "https://raw.githubusercontent.com/Whywolk/face-detection-in-video/main/res10_300x300_ssd_iter_140000.caffemodel",
        ],
    },
    "gender_proto": {
        "file": "gender_deploy.prototxt",
        "urls": [
            "https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/master/gender_net_definitions/deploy.prototxt",
            "https://raw.githubusercontent.com/eveningglow/age-and-gender-classifier/master/model/deploy_gender.prototxt",
        ],
    },
    "gender_model": {
        "file": "gender_net.caffemodel",
        "urls": [
            "https://drive.google.com/uc?export=download&id=1W_moLzMlGiELyPxWiYQJ9KFaXroQ_NFQ",
            "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/gender_net.caffemodel",
        ],
    },
}

# Detection thresholds
FACE_CONFIDENCE_THRESH   = 0.70
GENDER_CONFIDENCE_THRESH = 0.60
GENDER_CACHE_SEC         = 0.5
GENDER_STABLE_FRAMES     = 5


# ─────────────────────────────────────────
# RESULT
# ─────────────────────────────────────────

@dataclass
class FaceResult:
    face_detected:     bool  = False
    x1:                int   = 0
    y1:                int   = 0
    x2:                int   = 0
    y2:                int   = 0
    face_confidence:   float = 0.0
    gender:            str   = "Unknown"
    gender_confidence: float = 0.0
    honorific:         str   = "sir"


# ─────────────────────────────────────────
# DOWNLOADER
# ─────────────────────────────────────────

class _ProgressHook:
    def __init__(self, filename: str):
        self.filename  = filename
        self._last_pct = -1

    def __call__(self, block_num: int, block_size: int, total_size: int):
        if total_size <= 0:
            return
        downloaded = block_num * block_size
        pct = min(100, int(downloaded * 100 / total_size))
        if pct != self._last_pct and pct % 10 == 0:
            mb       = downloaded / (1024 * 1024)
            total_mb = total_size  / (1024 * 1024)
            print(f"  {self.filename}: {pct}%  ({mb:.1f} / {total_mb:.1f} MB)")
            self._last_pct = pct


def _download_file(url: str, dest_path: str, filename: str) -> bool:
    try:
        print(f"  Trying: {url[:80]}...")
        hook = _ProgressHook(filename)
        urllib.request.urlretrieve(url, dest_path, reporthook=hook)
        size = os.path.getsize(dest_path)
        if size < 1024:
            log.warning(f"  File too small ({size} B) — likely error page")
            os.remove(dest_path)
            return False
        print(f"  ✓ Saved: {filename} ({size / (1024*1024):.1f} MB)")
        return True
    except Exception as exc:
        log.warning(f"  Failed: {exc}")
        if os.path.exists(dest_path):
            try:
                os.remove(dest_path)
            except OSError:
                pass
        return False


def _ensure_models_downloaded() -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)

    for key, info in MODELS.items():
        filepath = os.path.join(MODEL_DIR, info["file"])

        if os.path.isfile(filepath) and os.path.getsize(filepath) > 1024:
            log.info(
                f"  Model present: {info['file']} "
                f"({os.path.getsize(filepath)/(1024*1024):.1f} MB)"
            )
            continue

        if os.path.isfile(filepath):
            os.remove(filepath)   # remove corrupt/empty file

        print(f"\n[FaceGender] Downloading: {info['file']}")
        success = False

        for i, url in enumerate(info["urls"]):
            print(f"  Attempt {i+1}/{len(info['urls'])}")
            if _download_file(url, filepath, info["file"]):
                success = True
                break
            print(f"  URL {i+1} failed, trying next...")

        if not success:
            manual = {
                "face_proto":   "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
                "face_model":   "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
                "gender_proto": "https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/master/gender_net_definitions/deploy.prototxt",
                "gender_model": "https://drive.google.com/file/d/1W_moLzMlGiELyPxWiYQJ9KFaXroQ_NFQ/view",
            }.get(key, "See documentation")

            print(f"""
╔══════════════════════════════════════════════════════════════╗
║  MANUAL DOWNLOAD REQUIRED                                    ║
║  File : {info['file']:<53}║
║  URL  : {manual[:53]:<53}║
║  Save to:                                                    ║
║  {filepath[:61]:<61}║
╚══════════════════════════════════════════════════════════════╝
""")
            raise RuntimeError(
                f"Cannot download {info['file']}.\n"
                f"Download from: {manual}\n"
                f"Place in: {MODEL_DIR}\\"
            )


# ─────────────────────────────────────────
# SAFE BACKEND PROBE
# ─────────────────────────────────────────

def _probe_cuda_available() -> bool:
    """
    Safely check if OpenCV was compiled with CUDA support.

    cv2.cuda.getCudaEnabledDeviceCount() raises an exception if
    OpenCV has no CUDA module — so we catch everything.

    This is the ONLY safe way to check — setting the backend and
    calling forward() crashes with an assertion error (as seen in logs).
    """
    try:
        count = cv2.cuda.getCudaEnabledDeviceCount()
        if count > 0:
            log.info(f"  CUDA devices available: {count}")
            return True
        log.info("  No CUDA devices found — using CPU")
        return False
    except Exception:
        log.info("  OpenCV built without CUDA — using CPU")
        return False


def _apply_backend(net, use_cuda: bool) -> None:
    """Apply backend to a DNN net. Never raises."""
    if use_cuda:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    else:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


# ─────────────────────────────────────────
# DETECTOR
# ─────────────────────────────────────────

class FaceGenderDetector:
    """
    Detects faces and classifies gender using OpenCV DNN.

    Usage:
        detector = FaceGenderDetector()
        result   = detector.detect(frame)
        if result.face_detected:
            print(f"{result.gender} → {result.honorific}")
    """

    GENDER_LABELS = ["Male", "Female"]
    GENDER_MEAN   = (78.4263377603, 87.7689143744, 114.895847746)

    def __init__(self):
        _ensure_models_downloaded()
        self._use_cuda = _probe_cuda_available()
        self._load_networks()

        # Gender smoothing state
        self._gender_history:   List[str]                    = []
        self._confirmed_gender: str                          = "Unknown"
        self._last_gender_time: float                        = 0.0
        self._last_gender_raw:  Optional[Tuple[str, float]]  = None

        log.info("FaceGenderDetector ready")

    def _load_networks(self) -> None:
        face_proto   = os.path.join(MODEL_DIR, MODELS["face_proto"]["file"])
        face_model   = os.path.join(MODEL_DIR, MODELS["face_model"]["file"])
        gender_proto = os.path.join(MODEL_DIR, MODELS["gender_proto"]["file"])
        gender_model = os.path.join(MODEL_DIR, MODELS["gender_model"]["file"])

        for path in [face_proto, face_model, gender_proto, gender_model]:
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Model file missing: {path}")

        log.info("Loading face detection model...")
        self.face_net = cv2.dnn.readNetFromCaffe(face_proto, face_model)

        log.info("Loading gender classification model...")
        self.gender_net = cv2.dnn.readNetFromCaffe(gender_proto, gender_model)

        # Apply backend — probed safely BEFORE setting it
        backend_name = "CUDA" if self._use_cuda else "CPU"
        log.info(f"  DNN backend: {backend_name}")
        _apply_backend(self.face_net,   self._use_cuda)
        _apply_backend(self.gender_net, self._use_cuda)

    # ── Face detection ────────────────────────────────────────────────

    def _detect_faces(
        self, frame: np.ndarray
    ) -> List[Tuple[int, int, int, int, float]]:
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            frame, 1.0, (300, 300),
            (104.0, 177.0, 123.0),
            swapRB=False, crop=False,
        )
        self.face_net.setInput(blob)
        detections = self.face_net.forward()

        faces = []
        for i in range(detections.shape[2]):
            conf = float(detections[0, 0, i, 2])
            if conf < FACE_CONFIDENCE_THRESH:
                continue

            x1 = max(0, int(detections[0, 0, i, 3] * w))
            y1 = max(0, int(detections[0, 0, i, 4] * h))
            x2 = min(w, int(detections[0, 0, i, 5] * w))
            y2 = min(h, int(detections[0, 0, i, 6] * h))

            fw, fh = x2 - x1, y2 - y1
            if fw < 30 or fh < 30:
                continue
            if fw > w * 0.9 or fh > h * 0.9:
                continue

            faces.append((x1, y1, x2, y2, conf))

        faces.sort(key=lambda f: f[4], reverse=True)
        return faces

    # ── Gender classification ─────────────────────────────────────────

    def _classify_gender(
        self,
        frame: np.ndarray,
        x1: int, y1: int, x2: int, y2: int,
    ) -> Tuple[str, float]:
        h, w = frame.shape[:2]
        pad_x = int((x2 - x1) * 0.15)
        pad_y = int((y2 - y1) * 0.15)
        fx1 = max(0, x1 - pad_x)
        fy1 = max(0, y1 - pad_y)
        fx2 = min(w, x2 + pad_x)
        fy2 = min(h, y2 + pad_y)
        roi = frame[fy1:fy2, fx1:fx2]

        if roi.size == 0:
            return "Unknown", 0.0

        blob = cv2.dnn.blobFromImage(
            roi, 1.0, (227, 227),
            self.GENDER_MEAN,
            swapRB=False, crop=False,
        )
        self.gender_net.setInput(blob)
        preds = self.gender_net.forward()

        idx  = int(np.argmax(preds[0]))
        conf = float(preds[0][idx])

        if conf < GENDER_CONFIDENCE_THRESH:
            return "Unknown", conf

        return self.GENDER_LABELS[idx], conf

    # ── Temporal smoothing ────────────────────────────────────────────

    def _smooth_gender(self, raw: str) -> str:
        """
        Require GENDER_STABLE_FRAMES consecutive same predictions
        before committing to a gender. Prevents frame-to-frame flicker.
        """
        if raw == "Unknown":
            # Unknown frames don't reset history — allow brief occlusion
            return self._confirmed_gender

        self._gender_history.append(raw)

        # Keep history bounded
        max_keep = GENDER_STABLE_FRAMES * 3
        if len(self._gender_history) > max_keep:
            self._gender_history = self._gender_history[-GENDER_STABLE_FRAMES * 2:]

        # Commit only when last N frames all agree
        if len(self._gender_history) >= GENDER_STABLE_FRAMES:
            recent = self._gender_history[-GENDER_STABLE_FRAMES:]
            if all(g == recent[0] for g in recent):
                self._confirmed_gender = recent[0]

        return self._confirmed_gender

    # ── Public API ────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> FaceResult:
        """
        Run face detection + gender classification on one frame.
        Safe to call every frame — gender is internally cached.
        """
        result = FaceResult()
        faces  = self._detect_faces(frame)

        if not faces:
            # Gradually erode history during absence (allows brief look-aways)
            if self._gender_history:
                self._gender_history = self._gender_history[:-1]
            return result

        x1, y1, x2, y2, face_conf = faces[0]
        result.face_detected   = True
        result.x1, result.y1  = x1, y1
        result.x2, result.y2  = x2, y2
        result.face_confidence = face_conf

        # Gender — cached for GENDER_CACHE_SEC to save CPU
        t = time.monotonic()
        if (
            self._last_gender_raw is None
            or (t - self._last_gender_time) >= GENDER_CACHE_SEC
        ):
            raw_gender, raw_conf       = self._classify_gender(frame, x1, y1, x2, y2)
            self._last_gender_raw      = (raw_gender, raw_conf)
            self._last_gender_time     = t
        else:
            raw_gender, raw_conf = self._last_gender_raw

        confirmed = self._smooth_gender(raw_gender)

        result.gender            = confirmed
        result.gender_confidence = raw_conf
        result.honorific         = "madam" if confirmed == "Female" else "sir"

        return result

    # ── Drawing ───────────────────────────────────────────────────────

    def draw_face_box(self, frame: np.ndarray, result: FaceResult) -> None:
        if not result.face_detected:
            return

        if result.gender == "Male":
            box_color  = (255, 160,  50)
            text_color = (255, 200, 100)
        elif result.gender == "Female":
            box_color  = (180, 100, 255)
            text_color = (220, 150, 255)
        else:
            box_color  = (150, 150, 150)
            text_color = (180, 180, 180)

        # Bounding box
        cv2.rectangle(
            frame,
            (result.x1, result.y1),
            (result.x2, result.y2),
            box_color, 2,
        )

        # Corner accents (Jarvis-style)
        clen  = min(20, (result.x2 - result.x1) // 4)
        thick = 3
        corners = [
            ((result.x1, result.y1), (result.x1 + clen, result.y1)),
            ((result.x1, result.y1), (result.x1, result.y1 + clen)),
            ((result.x2, result.y1), (result.x2 - clen, result.y1)),
            ((result.x2, result.y1), (result.x2, result.y1 + clen)),
            ((result.x1, result.y2), (result.x1 + clen, result.y2)),
            ((result.x1, result.y2), (result.x1, result.y2 - clen)),
            ((result.x2, result.y2), (result.x2 - clen, result.y2)),
            ((result.x2, result.y2), (result.x2, result.y2 - clen)),
        ]
        for p1, p2 in corners:
            cv2.line(frame, p1, p2, box_color, thick)

        # Gender label above box
        font  = cv2.FONT_HERSHEY_SIMPLEX
        label = f"{result.gender} ({result.face_confidence:.0%})"
        (tw, th), bl = cv2.getTextSize(label, font, 0.55, 1)
        ly = result.y1 - 8 if (result.y1 - 8 - th) > 0 else result.y2 + th + 8
        cv2.rectangle(
            frame,
            (result.x1, ly - th - 4),
            (result.x1 + tw + 8, ly + bl + 4),
            (18, 18, 18), -1,
        )
        cv2.putText(
            frame, label,
            (result.x1 + 4, ly),
            font, 0.55, text_color, 1, cv2.LINE_AA,
        )

        # Honorific tag below box
        cv2.putText(
            frame,
            f"[{result.honorific.upper()}]",
            (result.x1 + 4, result.y2 + 18),
            font, 0.45, box_color, 1, cv2.LINE_AA,
        )
