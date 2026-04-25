# 🏠 Smart Home — Gesture Control + Face Recognition + Gender-Aware Voice (Jarvis HUD)

A real-time smart room controller with:

- ✋ **Hand gesture control** (MediaPipe): unlock (horns), light on/off (palm swipe), fan on (index-circle), brightness/speed steps (palm/index up/down), fan off (thumb-down hold)
- 🧠 **Face detection + gender classification** (OpenCV DNN): detects presence by face, classifies **Male/Female**, and uses **“sir” / “madam”** dynamically
- 🗣️ **Voice feedback via WebSocket** to a **Jarvis-style HTML visualizer** (local TTS + animated spectrum)
- 🧯 **Room-empty + auto-lock logic**: 15s warning → 25s auto-off + lock; 14s idle warning → 18s auto-lock
- 🎛️ **Realistic controls**: PWM-style brightness (gamma-corrected), fan RPM mapping, cooldowns, hysteresis, sensor noise/drift simulation

---

## 🖥️ Requirements

- **Python** 3.10+ (tested on 3.11/3.12)
- **Camera** (built-in or USB)
- **Windows / macOS / Linux**
- **Modern browser** (Chrome/Edge recommended) for the visualizer

---

## ⚙️ Installation

```bash
# 1) Clone the repo
git clone https://github.com/YOUR_USERNAME/smart-home-gesture.git
cd smart-home-gesture

# 2) Create and activate a virtual environment (recommended)
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# 3) Install dependencies
pip install -r requirements.txt
