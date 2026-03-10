# ⚽ FootTrack — AI Football Retracker

Convert choppy Xbotgo footage into smooth, VEO-style ball-centered video using AI.

---

## How It Works

Xbotgo physically rotates its camera to track the ball — this produces choppy, jerky footage. VEO captures the full field and uses post-production to generate smooth tracking. FootTrack replicates this post-production approach:

```
Xbotgo MP4 input
    │
    ▼
[1] Optical Flow Stabilization  — removes camera shake/jerk artifacts
    │
    ▼
[2] YOLOv8 Ball Detection       — locates the ball on every frame
    │
    ▼
[3] Kalman Filter Prediction    — smoothly fills frames where ball is undetected
    │
    ▼
[4] Virtual Camera Pan          — exponential smoothing keeps ball centered
    │
    ▼
[5] Crop + Resize + Encode      — final H.264 output
    │
    ▼
Smooth VEO-style MP4 output
```

---

## Requirements

- **Python 3.9+**
- **ffmpeg** (must be in PATH)
  - macOS: `brew install ffmpeg`
  - Ubuntu: `sudo apt install ffmpeg`
  - Windows: https://ffmpeg.org/download.html

---

## Quick Start

### macOS / Linux
```bash
chmod +x scripts/run.sh
./scripts/run.sh
```

### Windows
```
Double-click: scripts\run.bat
```

Then open `frontend/index.html` in your browser.

---

## Manual Setup

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # Downloads YOLOv8 model on first run (~6MB)
python app.py
```

---

## Settings Explained

| Setting | Description | Recommended |
|---|---|---|
| **Smoothing** | Higher = camera reacts slower, smoother. Lower = snappier tracking. | 0.90–0.95 |
| **Zoom Factor** | Crop in tighter around the ball. 1.0 = no zoom, 1.5 = 50% closer. | 1.0–1.3 |
| **Output Resolution** | Final video dimensions. | 1920×1080 |
| **Edge Padding** | How much space to keep around the ball. Higher = more context visible. | 30–40% |
| **Stabilization** | Pre-process to remove choppy camera motion. Recommended ON. | ✅ On |

---

## Ball Detection

FootTrack tries YOLOv8 first (best accuracy). If not installed, it falls back to a heuristic detector using:
- White/light color detection (for white balls)
- Hough circle transform
- Contour + circularity filtering

For best results, ensure `ultralytics` is installed (`pip install ultralytics`).

---

## Performance

Processing time depends on video length and your hardware:
- ~30fps source video → ~2–4× realtime on CPU
- GPU (CUDA/MPS) auto-detected by YOLOv8 if available

For long matches (90 min+), expect 20–45 minutes on CPU. Use a GPU machine for faster results.

---

## Project Structure

```
football-retracker/
├── backend/
│   ├── app.py          — Flask API server
│   ├── processor.py    — Core AI processing pipeline
│   └── requirements.txt
├── frontend/
│   └── index.html      — Web UI
└── scripts/
    ├── run.sh          — macOS/Linux launcher
    └── run.bat         — Windows launcher
```

---

## Troubleshooting

**"Cannot open video"** — Check that ffmpeg is installed and your file is not corrupted.

**Ball not centered well** — Try increasing smoothing to 0.95, or lower zoom to 0.9 to capture more field. The heuristic detector works best with white balls on green grass in decent lighting.

**Very slow processing** — Install CUDA for GPU acceleration, or reduce output resolution to 720p.

**YOLOv8 download fails** — Run `pip install ultralytics` manually; it downloads `yolov8n.pt` (~6MB) on first use.
