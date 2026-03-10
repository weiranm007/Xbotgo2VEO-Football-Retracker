"""
VideoProcessor v2 — AI-enhanced football retracker

Key improvements over v1:
  - Claude AI scene analysis: samples keyframes to understand field layout,
    ball size at distance, and camera behavior hints
  - CinematicCamera: proper spring-damper physics instead of simple EMA
  - Dead zone: camera stays still if ball hasn't moved meaningfully
  - Auto-zoom: crops tighter when ball is in far half of pitch (appears small)
  - Ball size estimation: infers distance from apparent ball radius
  - Heavy Gaussian smoothing on detection trajectory before camera update
  - Velocity clamping to prevent sudden lurches
  - Second smoothing pass on the camera path itself
"""

import cv2
import numpy as np
import subprocess
import base64
import json
import os
import tempfile
import multiprocessing
import urllib.request


# ─────────────────────────────────────────────────────────────
# Kalman Filter
# ─────────────────────────────────────────────────────────────
class BallKalman:
    """2D constant-velocity Kalman filter."""

    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        dt = 1.0
        self.kf.transitionMatrix = np.array(
            [[1, 0, dt, 0],
             [0, 1, 0, dt],
             [0, 0, 1,  0],
             [0, 0, 0,  1]], dtype=np.float32
        )
        self.kf.measurementMatrix = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0]], dtype=np.float32
        )
        self.kf.processNoiseCov     = np.eye(4, dtype=np.float32) * 0.05
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.0
        self.kf.errorCovPost        = np.eye(4, dtype=np.float32) * 10
        self.initialized = False

    def update(self, x, y):
        meas = np.array([[np.float32(x)], [np.float32(y)]])
        if not self.initialized:
            self.kf.statePre  = np.array([[x], [y], [0], [0]], dtype=np.float32)
            self.kf.statePost = np.array([[x], [y], [0], [0]], dtype=np.float32)
            self.initialized = True
        self.kf.predict()
        state = self.kf.correct(meas)
        return float(state[0].flat[0]), float(state[1].flat[0])

    def predict(self):
        state = self.kf.predict()
        return float(state[0].flat[0]), float(state[1].flat[0])


# ─────────────────────────────────────────────────────────────
# Ball Detector
# ─────────────────────────────────────────────────────────────
class BallDetector:
    """YOLOv8 with heuristic fallback. Returns (cx, cy, radius, conf)."""

    def __init__(self):
        self.model = None
        self._try_load_yolo()

    def _try_load_yolo(self):
        try:
            from ultralytics import YOLO
            self.model = YOLO("yolov8n.pt")
            print("[BallDetector] YOLOv8 loaded")
        except Exception as e:
            print(f"[BallDetector] YOLOv8 unavailable ({e}), using heuristic")

    def detect(self, frame):
        if self.model:
            return self._yolo_detect(frame)
        return self._heuristic_detect(frame)

    def _yolo_detect(self, frame):
        results = self.model(frame, classes=[32], verbose=False)
        best, best_conf = None, 0.0
        for r in results:
            for box in r.boxes:
                conf = float(box.conf)
                if conf > best_conf:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    radius = ((x2 - x1) + (y2 - y1)) / 4
                    best = (cx, cy, radius, conf)
                    best_conf = conf
        return best if best_conf > 0.25 else None

    def _heuristic_detect(self, frame):
        h, w = frame.shape[:2]
        hsv    = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        grass  = cv2.inRange(hsv, np.array([30, 40, 40]),   np.array([90, 255, 255]))
        white  = cv2.inRange(hsv, np.array([0,  0,  180]),  np.array([180, 50, 255]))
        yellow = cv2.inRange(hsv, np.array([20, 100, 150]), np.array([35, 255, 255]))
        mask   = cv2.bitwise_and(cv2.bitwise_or(white, yellow), cv2.bitwise_not(grass))
        k      = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)

        gray    = cv2.bitwise_and(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), mask)
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
            param1=50, param2=18, minRadius=3, maxRadius=min(w, h) // 15
        )
        if circles is not None:
            c = circles[0][0]
            return float(c[0]), float(c[1]), float(c[2]), 0.4

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates  = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 40 or area > w * h * 0.008:
                continue
            perim = cv2.arcLength(cnt, True)
            if perim == 0:
                continue
            circ = 4 * np.pi * area / perim ** 2
            if circ > 0.35:
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = M["m10"] / M["m00"]
                    cy = M["m01"] / M["m00"]
                    r  = np.sqrt(area / np.pi)
                    candidates.append((circ, cx, cy, r))
        if candidates:
            candidates.sort(reverse=True)
            _, cx, cy, r = candidates[0]
            return cx, cy, r, 0.25
        return None


# ─────────────────────────────────────────────────────────────
# AI Scene Analyser — asks Claude to assess the footage
# ─────────────────────────────────────────────────────────────
class AISceneAnalyser:
    """
    Sends 4 sampled keyframes to Claude and asks for camera hints:
      - Ball radius near vs far
      - Recommended far-zoom multiplier
      - Dead zone size
    Falls back to sensible defaults if API unavailable.
    """

    DEFAULTS = {
        "near_radius_px":    18.0,
        "far_radius_px":      7.0,
        "far_zoom":           1.8,
        "near_zoom_out":      0.72,
        "dead_zone_px":      16.0,
        "pitch_far_y_ratio":  0.4,
    }

    def __init__(self):
        self.hints = dict(self.DEFAULTS)

    def analyse(self, frames, detections, src_w, src_h):
        try:
            self._call_claude(frames, src_w, src_h)
        except Exception as e:
            print(f"[AIScene] Skipped (using defaults): {e}")
        return self.hints

    def _call_claude(self, frames, src_w, src_h):
        indices = [int(len(frames) * r) for r in [0.08, 0.28, 0.55, 0.82]]
        content = []
        for idx in indices:
            idx = min(idx, len(frames) - 1)
            small = cv2.resize(frames[idx], (640, int(640 * src_h / src_w)))
            _, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 72])
            b64 = base64.b64encode(buf).decode()
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}
            })

        content.append({
            "type": "text",
            "text": (
                "These are 4 frames from a football (soccer) match filmed by a rotating AI tracking camera. "
                "Analyse them and return ONLY a valid JSON object with exactly these keys:\n"
                "  near_radius_px  - pixel radius of ball when nearest to camera (integer, e.g. 15)\n"
                "  far_radius_px   - pixel radius of ball when furthest away (integer, e.g. 6)\n"
                "  far_zoom        - zoom-in multiplier (1.3-2.2) when ball is on far side of pitch\n"
                "  near_zoom_out   - zoom-out multiplier (0.55-0.90) for near-side wider view\n"
                "  dead_zone_px    - pixels ball must travel before camera moves (8-30)\n"
                "  pitch_far_y_ratio - fraction 0-1: y below this means far end of pitch\n"
                "Return ONLY the JSON object, no markdown, no explanation."
            )
        })

        payload = json.dumps({
            "model": "claude-sonnet-4-6",
            "max_tokens": 300,
            "messages": [{"role": "user", "content": content}]
        }).encode()

        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=payload,
            headers={"content-type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=25) as resp:
            data = json.loads(resp.read())

        text = "".join(b.get("text", "") for b in data.get("content", []))
        text = text.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        parsed = json.loads(text)

        for key in self.hints:
            if key in parsed:
                val = parsed[key]
                if val is not None:
                    self.hints[key] = float(val)
        print(f"[AIScene] Claude hints: {self.hints}")


# ─────────────────────────────────────────────────────────────
# Ball Trajectory Predictor — lookahead zoom intent
# ─────────────────────────────────────────────────────────────
class BallPredictor:
    """
    Pre-computes a per-frame target_zoom for every frame.

    VEO philosophy (from footage analysis):
      - Zoom range is NARROW: roughly 0.88–1.30.  Never dramatic.
      - Near side: gentle breathe-out so you see ~15% more pitch context
      - Far side:  gentle zoom-in so ball/players aren't tiny specks
      - Long ball: pre-emptively widen by ~12% so the full arc is readable,
                   then slowly settle back in as play resets
      - ALL transitions are slow — the viewer should never consciously notice
        the zoom changing, only feel that the framing always "feels right"
    """

    # ── Zoom envelope (VEO-style narrow range) ────────────────
    NEAR_ZOOM   = 0.90   # slight zoom-out when ball close (more pitch visible)
    MID_ZOOM    = 1.00   # neutral / midfield
    FAR_ZOOM    = 1.28   # modest zoom-in when ball far (not a telescope)
    LONG_ZOOM   = 0.86   # breathe-out during long ball arc
    CLAMP_MIN   = 0.82   # hard floor — never wider than this
    CLAMP_MAX   = 1.35   # hard ceiling — never tighter than this

    def __init__(self, fps=30.0, lookahead_sec=2.0, longball_threshold_ratio=0.28):
        self.lookahead_frames   = int(fps * lookahead_sec)
        self.longball_threshold = longball_threshold_ratio   # fraction of src_w

    def compute(self, detections, src_w, src_h,
                far_r_threshold=7.0, near_r=18.0, **_):
        n = len(detections)
        result = []

        for i in range(n):
            bx, by, br = detections[i]

            # ── 1. Distance-based zoom (radius proxy) ──────
            # Map apparent radius → zoom target in [NEAR_ZOOM, FAR_ZOOM]
            if br >= near_r:
                dist_zoom = self.NEAR_ZOOM
            elif br <= far_r_threshold:
                dist_zoom = self.FAR_ZOOM
            else:
                t = (br - far_r_threshold) / max(near_r - far_r_threshold, 1.0)
                # t=0 → far (FAR_ZOOM), t=1 → near (NEAR_ZOOM)
                dist_zoom = self.FAR_ZOOM + t * (self.NEAR_ZOOM - self.FAR_ZOOM)

            # ── 2. Lookahead: upcoming long ball ───────────
            end = min(n, i + self.lookahead_frames)
            fxs = [detections[j][0] for j in range(i, end)]
            fys = [detections[j][1] for j in range(i, end)]

            is_longball  = False
            longball_mag = 0.0
            if len(fxs) > 1:
                net = float(np.hypot(fxs[-1] - fxs[0], fys[-1] - fys[0]))
                dxs = np.diff(fxs); dys = np.diff(fys)
                path = float(np.sum(np.sqrt(dxs**2 + dys**2)))
                travel = max(net, path * 0.35)
                thr = self.longball_threshold * src_w
                if travel > thr:
                    is_longball  = True
                    longball_mag = min(1.0, (travel - thr) / (thr * 1.5))

            # ── 3. Blend distance-zoom with longball breathe ─
            if is_longball:
                # Blend toward LONG_ZOOM proportionally — but only if it
                # would *widen* the shot (don't zoom in during a long ball)
                longball_target = self.MID_ZOOM + longball_mag * (self.LONG_ZOOM - self.MID_ZOOM)
                target_zoom = min(dist_zoom, longball_target)  # take the wider option
            else:
                target_zoom = dist_zoom

            target_zoom = float(np.clip(target_zoom, self.CLAMP_MIN, self.CLAMP_MAX))
            result.append((target_zoom, is_longball))

        # ── 4. Smooth the target signal itself (avoid flip-flopping) ──
        raw = np.array([r[0] for r in result])
        # Wide Gaussian pass — transitions should take ~1.5s to complete
        hw = 45
        t  = np.arange(-hw, hw + 1, dtype=float)
        k  = np.exp(-0.5 * (t / 20) ** 2); k /= k.sum()
        smoothed = np.convolve(raw, k, mode="same")
        smoothed = np.clip(smoothed, self.CLAMP_MIN, self.CLAMP_MAX)

        return [(float(smoothed[i]), result[i][1]) for i in range(n)]


class CinematicCamera:
    """
    Spring-damper camera — pan and zoom both use physics.
    Zoom spring is deliberately very slow (zoom_spring=0.012) so the viewer
    never consciously notices it moving — they only feel "good framing".
    """

    def __init__(self, src_w, src_h, base_crop_w, base_crop_h,
                 spring=0.04, damping=0.75, dead_zone=16,
                 min_zoom=BallPredictor.CLAMP_MIN, max_zoom=BallPredictor.CLAMP_MAX):
        self.src_w, self.src_h             = src_w, src_h
        self.base_crop_w, self.base_crop_h = base_crop_w, base_crop_h
        self.spring  = spring
        self.damping = damping
        self.cx = float(src_w  / 2)
        self.cy = float(src_h  / 2)
        self.vx = 0.0
        self.vy = 0.0
        self.dead_zone    = dead_zone
        self.min_zoom     = min_zoom
        self.max_zoom     = max_zoom
        self.current_zoom = 1.0
        self.zoom_vel     = 0.0
        # Very slow zoom spring — transitions ~1–2 s, imperceptible to viewer
        self.zoom_spring  = 0.012
        self.zoom_damping = 0.88

    def update(self, tx, ty, target_zoom=1.0):
        # ── Pan ──────────────────────────────────────────
        dist = np.hypot(tx - self.cx, ty - self.cy)
        if dist > self.dead_zone:
            fx = (tx - self.cx) * self.spring
            fy = (ty - self.cy) * self.spring
            self.vx = self.vx * self.damping + fx
            self.vy = self.vy * self.damping + fy
            max_v = max(dist * 0.22, 3.0)
            speed = np.hypot(self.vx, self.vy)
            if speed > max_v:
                self.vx *= max_v / speed
                self.vy *= max_v / speed
            self.cx += self.vx
            self.cy += self.vy
        else:
            self.vx *= 0.80
            self.vy *= 0.80

        # ── Zoom spring (very gentle) ─────────────────────
        target_zoom = float(np.clip(target_zoom, self.min_zoom, self.max_zoom))
        z_force = (target_zoom - self.current_zoom) * self.zoom_spring
        self.zoom_vel     = self.zoom_vel * self.zoom_damping + z_force
        self.current_zoom = float(np.clip(
            self.current_zoom + self.zoom_vel, self.min_zoom, self.max_zoom
        ))

        # ── Crop size ─────────────────────────────────────
        cw = max(64, min(self.src_w,  int(self.base_crop_w / self.current_zoom)))
        ch = max(36, min(self.src_h,  int(self.base_crop_h / self.current_zoom)))

        cam_cx = float(np.clip(self.cx, cw / 2, self.src_w - cw / 2))
        cam_cy = float(np.clip(self.cy, ch / 2, self.src_h - ch / 2))
        return cam_cx, cam_cy, cw, ch


# ─────────────────────────────────────────────────────────────
# Wide-Angle / Static Camera Stitcher
# ─────────────────────────────────────────────────────────────
class WideAngleStitcher:
    """
    Handles two input modes for static/wide-angle footage:

    Mode A — single_static:
        One wide-angle camera covering the full pitch.
        No stitching needed — just pass frames through.
        We skip stabilization (camera is fixed) and apply a tighter
        virtual crop window since we have the full pitch to work with.

    Mode B — dual_phone:
        Two phones placed side-by-side, each capturing half the pitch.
        Steps:
          1. Open both video streams simultaneously
          2. Compute a homography matrix (H) from the first N frames
             by matching SIFT/ORB features in the overlapping centre zone
          3. Warp right frame onto left frame coordinate space
          4. Blend the seam with a simple gradient alpha mask
          5. Return a single wide stitched frame per pair

    The stitcher is initialised once and reused per-frame for efficiency.
    """

    def __init__(self, mode="single_static"):
        assert mode in ("single_static", "dual_phone"), f"Unknown mode: {mode}"
        self.mode = mode
        self.H    = None          # homography (dual_phone only)
        self.canvas_w = None
        self.canvas_h = None
        self.alpha    = None      # blend mask

    # ── Single static: no-op ──────────────────────────────────
    def open_single(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open: {video_path}")
        return cap

    def read_frames_single(self, video_path, max_frames, progress_cb=None):
        cap = self.open_single(video_path)
        frames = []
        while len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            if progress_cb and len(frames) % 100 == 0:
                progress_cb(len(frames), f"Reading frame {len(frames)}")
        cap.release()
        return frames

    # ── Dual phone: stitch ───────────────────────────────────
    def calibrate_homography(self, left_frame, right_frame, n_features=3000):
        """
        Compute homography to map right_frame into left_frame coordinates.
        Uses ORB (no licence issues) with brute-force matching + RANSAC.
        Returns True if homography found, False otherwise.
        """
        gray_l = cv2.cvtColor(left_frame,  cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)

        # Try SIFT first (better quality), fall back to ORB
        try:
            det = cv2.SIFT_create(nfeatures=n_features)
            norm = cv2.NORM_L2
        except AttributeError:
            det = cv2.ORB_create(nfeatures=n_features)
            norm = cv2.NORM_HAMMING

        kp_l, des_l = det.detectAndCompute(gray_l, None)
        kp_r, des_r = det.detectAndCompute(gray_r, None)

        if des_l is None or des_r is None or len(kp_l) < 10 or len(kp_r) < 10:
            return False

        matcher = cv2.BFMatcher(norm, crossCheck=False)
        matches = matcher.knnMatch(des_r, des_l, k=2)

        # Lowe ratio test
        good = []
        for m_list in matches:
            if len(m_list) == 2:
                m, n_m = m_list
                if m.distance < 0.75 * n_m.distance:
                    good.append(m)

        if len(good) < 10:
            print(f"[Stitch] Only {len(good)} good matches — trying without ratio test")
            good = [m_list[0] for m_list in matches if len(m_list) >= 1]

        if len(good) < 8:
            return False

        pts_r = np.float32([kp_r[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        pts_l = np.float32([kp_l[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(pts_r, pts_l, cv2.RANSAC, 5.0)
        if H is None:
            return False

        self.H = H
        print(f"[Stitch] Homography computed from {int(mask.sum())} inliers")

        # Compute canvas size
        h_l, w_l = left_frame.shape[:2]
        h_r, w_r = right_frame.shape[:2]
        corners_r = np.float32([[0,0],[w_r,0],[w_r,h_r],[0,h_r]]).reshape(-1,1,2)
        corners_warped = cv2.perspectiveTransform(corners_r, H)
        all_corners = np.concatenate([
            np.float32([[0,0],[w_l,0],[w_l,h_l],[0,h_l]]).reshape(-1,1,2),
            corners_warped
        ], axis=0)
        x_min = int(np.floor(all_corners[:,:,0].min()))
        y_min = int(np.floor(all_corners[:,:,1].min()))
        x_max = int(np.ceil(all_corners[:,:,0].max()))
        y_max = int(np.ceil(all_corners[:,:,1].max()))

        # Clamp to sane output size (max 4× either input dimension)
        self.canvas_w = min(x_max - x_min, w_l * 4)
        self.canvas_h = min(y_max - y_min, h_l * 3)

        # Translation matrix to shift result into positive coordinates
        T = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float64)
        self.H = T @ H
        self.T_offset = (-x_min, -y_min)
        self.left_shape = (h_l, w_l)

        # Precompute alpha blend mask for seam
        self._build_blend_mask(left_frame, right_frame)
        return True

    def _build_blend_mask(self, left_frame, right_frame):
        """Gradient blend mask in the overlap zone."""
        h_l, w_l = self.left_shape
        # Warp a white right-frame mask
        r_mask = np.ones(right_frame.shape[:2], dtype=np.float32)
        r_warped = cv2.warpPerspective(r_mask, self.H, (self.canvas_w, self.canvas_h))
        # Left mask (just place it)
        tx, ty = self.T_offset
        l_mask = np.zeros((self.canvas_h, self.canvas_w), dtype=np.float32)
        lx1 = max(0, tx); ly1 = max(0, ty)
        lx2 = min(self.canvas_w, tx + w_l); ly2 = min(self.canvas_h, ty + h_l)
        l_mask[ly1:ly2, lx1:lx2] = 1.0

        # In overlap zone: gradient blend
        overlap = (r_warped > 0) & (l_mask > 0)
        # Simple: left gets priority in its region, blend 50/50 in overlap
        self.alpha_l = l_mask.copy()
        self.alpha_r = r_warped.copy()
        # In overlap zone taper left → right
        if overlap.any():
            xs = np.where(overlap)[1]
            x_lo, x_hi = xs.min(), xs.max()
            for x in range(x_lo, x_hi + 1):
                t = (x - x_lo) / max(x_hi - x_lo, 1)
                col_mask = overlap[:, x]
                self.alpha_l[col_mask, x] = 1.0 - t
                self.alpha_r[col_mask, x] = t

    def stitch(self, left_frame, right_frame):
        """Warp right frame and blend with left frame into canvas."""
        if self.H is None:
            # Side-by-side concat as fallback
            return np.concatenate([left_frame, right_frame], axis=1)

        tx, ty = self.T_offset
        h_l, w_l = self.left_shape
        canvas = np.zeros((self.canvas_h, self.canvas_w, 3), dtype=np.uint8)

        # Place left frame
        lx1 = max(0, tx); ly1 = max(0, ty)
        lx2 = min(self.canvas_w, tx + w_l); ly2 = min(self.canvas_h, ty + h_l)
        src_x2 = lx2 - lx1; src_y2 = ly2 - ly1
        canvas[ly1:ly2, lx1:lx2] = left_frame[:src_y2, :src_x2]

        # Warp right frame
        r_warped = cv2.warpPerspective(right_frame, self.H, (self.canvas_w, self.canvas_h))

        # Blend
        al = self.alpha_l[:,:,np.newaxis]
        ar = self.alpha_r[:,:,np.newaxis]
        total = al + ar
        total = np.where(total == 0, 1, total)  # avoid div by zero
        blended = (canvas.astype(np.float32) * al + r_warped.astype(np.float32) * ar) / total
        canvas = np.clip(blended, 0, 255).astype(np.uint8)
        return canvas

    def read_frames_dual(self, left_path, right_path, max_frames,
                         calibration_frames=8, progress_cb=None):
        """
        Read left+right video pair, compute homography from first
        calibration_frames, then stitch all frames.
        """
        cap_l = cv2.VideoCapture(left_path)
        cap_r = cv2.VideoCapture(right_path)
        if not cap_l.isOpened():
            raise RuntimeError(f"Cannot open left video: {left_path}")
        if not cap_r.isOpened():
            raise RuntimeError(f"Cannot open right video: {right_path}")

        frames = []
        calibrated = False
        calib_candidates = []

        while len(frames) < max_frames:
            ret_l, fl = cap_l.read()
            ret_r, fr = cap_r.read()
            if not ret_l or not ret_r:
                break

            if not calibrated:
                calib_candidates.append((fl, fr))
                if len(calib_candidates) >= calibration_frames:
                    # Try each candidate pair until homography succeeds
                    for cf_l, cf_r in calib_candidates:
                        if self.calibrate_homography(cf_l, cf_r):
                            calibrated = True
                            break
                    if not calibrated:
                        print("[Stitch] Homography failed — using side-by-side concat")
                        calibrated = True  # mark done, will concat

            stitched = self.stitch(fl, fr) if calibrated else np.concatenate([fl, fr], axis=1)
            frames.append(stitched)

            if progress_cb and len(frames) % 100 == 0:
                progress_cb(len(frames), f"Stitching frame {len(frames)}")

        cap_l.release()
        cap_r.release()
        return frames


# ─────────────────────────────────────────────────────────────
# Video Stabilizer (optical flow)
# ─────────────────────────────────────────────────────────────
class VideoStabilizer:
    def __init__(self, smoothing_radius=45):
        self.smoothing_radius = smoothing_radius

    def compute_transforms(self, frames, progress_cb=None):
        transforms = []
        prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        for i in range(1, len(frames)):
            curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            pts = cv2.goodFeaturesToTrack(
                prev_gray, maxCorners=300, qualityLevel=0.01, minDistance=20, blockSize=3
            )
            if pts is not None and len(pts) >= 4:
                curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, pts, None)
                idx = np.where(status.ravel() == 1)[0]
                if len(idx) >= 4:
                    m, _ = cv2.estimateAffinePartial2D(pts[idx], curr_pts[idx])
                    if m is not None:
                        transforms.append((m[0, 2], m[1, 2], np.arctan2(m[1, 0], m[0, 0])))
                        prev_gray = curr_gray
                        continue
            transforms.append((0.0, 0.0, 0.0))
            prev_gray = curr_gray
            if progress_cb and i % 60 == 0:
                progress_cb(int(i / len(frames) * 100), f"Motion analysis {i}/{len(frames)}")
        return transforms

    def smooth_transforms(self, transforms):
        r = self.smoothing_radius
        arrs = [np.array([t[i] for t in transforms]) for i in range(3)]
        out = []
        for arr in arrs:
            s = np.zeros_like(arr)
            for i in range(len(arr)):
                a, b = max(0, i - r), min(len(arr), i + r + 1)
                s[i] = np.mean(arr[a:b])
            out.append(s)
        return list(zip(*[o.tolist() for o in out]))


# ─────────────────────────────────────────────────────────────
# Trajectory Smoother
# ─────────────────────────────────────────────────────────────
def smooth_trajectory(detections, kernel_size=31, sigma=8.0):
    xs = np.array([d[0] for d in detections], dtype=np.float64)
    ys = np.array([d[1] for d in detections], dtype=np.float64)
    rs = np.array([d[2] for d in detections], dtype=np.float64)
    half = kernel_size // 2
    t    = np.arange(-half, half + 1)
    k    = np.exp(-0.5 * (t / sigma) ** 2)
    k   /= k.sum()
    return list(zip(
        np.convolve(xs, k, mode="same").tolist(),
        np.convolve(ys, k, mode="same").tolist(),
        np.convolve(rs, k, mode="same").tolist(),
    ))


# ─────────────────────────────────────────────────────────────
# Main Processor
# ─────────────────────────────────────────────────────────────
class VideoProcessor:

    def __init__(self, progress_callback=None):
        self.progress_cb = progress_callback or (lambda p, m: None)

    def _cb(self, pct, msg):
        self.progress_cb(pct, msg)
        print(f"[{pct:3d}%] {msg}")

    def process(self, input_path, output_path, options, input_path_b=None):
        out_w       = options.get("output_width",  1920)
        out_h       = options.get("output_height", 1080)
        smoothing   = options.get("smoothing",     0.92)
        zoom        = options.get("zoom_factor",   1.0)
        do_stab     = options.get("stabilize",     True)
        source_mode = options.get("source_mode",   "xbotgo")  # "xbotgo" | "static" | "dual_phone"

        # Static/wide modes: camera is fixed — no stabilization needed
        if source_mode in ("static", "dual_phone"):
            do_stab = False

        # Map UI smoothing (0.5–0.99) → spring k (0.09 → 0.008)
        spring_k = 0.09 * (1.0 - float(smoothing)) + 0.008

        self._cb(2, "Opening video...")
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open: {input_path}")

        fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
        src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._cb(5, f"Source: {src_w}×{src_h} @ {fps:.1f}fps, mode={source_mode}")

        cap.release()  # release early — we'll re-open via stitcher helpers

        MAX_FRAMES = 4500
        self._cb(8, "Reading frames...")

        if source_mode == "dual_phone":
            if not input_path_b:
                raise RuntimeError("dual_phone mode requires a second video file")
            stitcher = WideAngleStitcher(mode="dual_phone")
            frames = stitcher.read_frames_dual(
                input_path, input_path_b, MAX_FRAMES,
                progress_cb=lambda n, m: self._cb(8 + min(6, n // 100), m)
            )
            # Update src dimensions from stitched output
            if frames:
                src_h, src_w = frames[0].shape[:2]
        elif source_mode == "static":
            stitcher = WideAngleStitcher(mode="single_static")
            frames = stitcher.read_frames_single(
                input_path, MAX_FRAMES,
                progress_cb=lambda n, m: self._cb(8 + min(6, n // 100), m)
            )
        else:
            # xbotgo: normal read
            cap2 = cv2.VideoCapture(input_path)
            frames = []
            while len(frames) < MAX_FRAMES:
                ret, frame = cap2.read()
                if not ret:
                    break
                frames.append(frame)
            cap2.release()

        n = len(frames)
        if n == 0:
            raise RuntimeError("No frames read")
        self._cb(15, f"Read {n} frames ({source_mode} mode)")

        # ── Step 1: Stabilize ──────────────────────────────
        if do_stab and n > 10:
            self._cb(18, "Computing stabilization...")
            stab = VideoStabilizer(smoothing_radius=min(45, n // 8))
            trs  = stab.compute_transforms(
                frames, progress_cb=lambda p, m: self._cb(18 + int(p * 0.10), m)
            )
            strs = stab.smooth_transforms(trs)
            self._cb(29, "Applying stabilization...")
            frames = self._apply_stabilization(frames, trs, strs)

        # ── Step 2: Detect ball ─────────────────────────────
        self._cb(32, "Loading detector...")
        detector = BallDetector()
        kalman   = BallKalman()
        raw = []
        self._cb(35, "Detecting ball...")
        for i, frame in enumerate(frames):
            result = detector.detect(frame)
            if result:
                cx, cy, radius, _ = result
                kx, ky = kalman.update(cx, cy)
                raw.append((kx, ky, float(radius)))
            else:
                if kalman.initialized:
                    kx, ky = kalman.predict()
                    last_r = raw[-1][2] if raw else 10.0
                    raw.append((kx, ky, last_r))
                else:
                    raw.append((float(src_w / 2), float(src_h / 2), 10.0))
            if i % 40 == 0:
                self._cb(35 + int(i / n * 25), f"Detection: {i}/{n}")

        # ── Step 3: AI scene analysis ──────────────────────
        self._cb(62, "AI scene analysis...")
        hints = AISceneAnalyser().analyse(frames, raw, src_w, src_h)
        dead_zone   = int(hints.get("dead_zone_px",   16))
        far_zoom_ai = float(hints.get("far_zoom",     1.7))
        far_r_thr   = float(hints.get("far_radius_px", 7))
        near_r      = float(hints.get("near_radius_px", 18))

        # Static/wide cameras have more extreme radius variation (full pitch visible)
        # Override radius thresholds with tighter values
        if source_mode in ("static", "dual_phone"):
            far_r_thr = float(hints.get("far_radius_px", 4))   # ball is tiny when far
            near_r    = float(hints.get("near_radius_px", 12))  # not as large up close
            dead_zone = max(8, dead_zone - 4)                   # more responsive pan

        # ── Step 4: Smooth trajectory ──────────────────────
        self._cb(65, "Smoothing trajectory...")
        detections = smooth_trajectory(raw, kernel_size=31, sigma=8.0)

        # ── Step 5: Camera path ────────────────────────────
        self._cb(68, "Computing cinematic camera path...")
        ar = out_w / out_h
        if src_w / src_h > ar:
            base_cw = int(src_h * ar)
            base_ch = src_h
        else:
            base_cw = src_w
            base_ch = int(src_w / ar)
        base_cw = max(64, int(base_cw / float(zoom)))
        base_ch = max(36, int(base_ch / float(zoom)))

        # Pre-compute zoom intent for every frame (includes lookahead)
        self._cb(69, "Computing predictive zoom intent...")
        predictor = BallPredictor(fps=fps, lookahead_sec=2.0, longball_threshold_ratio=0.28)
        zoom_intents = predictor.compute(
            detections, src_w, src_h,
            far_r_threshold=far_r_thr,
            near_r=near_r,
        )

        cam = CinematicCamera(
            src_w=src_w, src_h=src_h,
            base_crop_w=base_cw, base_crop_h=base_ch,
            spring=spring_k, damping=0.75,
            dead_zone=dead_zone,
        )

        cam_path = []
        for (bx, by, br), (target_zoom, is_lb) in zip(detections, zoom_intents):
            tx = float(np.clip(bx, base_cw / 2, src_w - base_cw / 2))
            ty = float(np.clip(by, base_ch / 2, src_h - base_ch / 2))
            cam_path.append(cam.update(tx, ty, target_zoom=target_zoom))

        cam_path = self._smooth_cam_path(cam_path, radius=5)

        # ── Step 6: Render ─────────────────────────────────
        self._cb(73, "Rendering...")
        tmp_raw = output_path.replace(".mp4", "_raw.mp4")
        writer  = cv2.VideoWriter(
            tmp_raw, cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h)
        )
        if not writer.isOpened():
            raise RuntimeError("VideoWriter failed to open")

        # Track stabilization border mask — reuse if unchanged
        last_border_mask = None
        last_cw, last_ch = -1, -1

        for i, (frame, (cx, cy, cw, ch)) in enumerate(zip(frames, cam_path)):
            cw = max(2, cw - cw % 2)
            ch = max(2, ch - ch % 2)
            crop    = self._crop(frame, cx, cy, cw, ch, src_w, src_h)
            resized = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)

            # ── Border inpainting ──────────────────────────
            resized = self._fill_borders(resized)

            writer.write(resized)
            if i % 60 == 0:
                self._cb(73 + int(i / n * 20), f"Rendering {i}/{n}")
        writer.release()

        # ── Step 7: Frame interpolation (bump to 60fps) ───
        self._cb(93, "Interpolating frames to 60fps...")
        tmp_interp = output_path.replace(".mp4", "_interp.mp4")
        interp_ok  = self._interpolate_frames(tmp_raw, tmp_interp, fps)
        src_for_encode = tmp_interp if interp_ok else tmp_raw

        # ── Step 8: Final encode ────────────────────────────
        self._cb(97, "Encoding final video...")
        target_fps = 60.0 if interp_ok else fps
        self._reencode(input_path, src_for_encode, output_path, target_fps)
        for f in [tmp_raw, tmp_interp]:
            try:
                os.remove(f)
            except:
                pass
        self._cb(100, "Done!")

    # ── Helpers ────────────────────────────────────────────

    def _smooth_cam_path(self, cam_path, radius=5):
        xs  = np.array([p[0] for p in cam_path])
        ys  = np.array([p[1] for p in cam_path])
        cws = np.array([p[2] for p in cam_path], dtype=float)
        chs = np.array([p[3] for p in cam_path], dtype=float)

        def box(arr, r):
            out = np.zeros_like(arr)
            for i in range(len(arr)):
                s, e = max(0, i - r), min(len(arr), i + r + 1)
                out[i] = np.mean(arr[s:e])
            return out

        return list(zip(
            box(xs, radius).tolist(),
            box(ys, radius).tolist(),
            box(cws, radius).astype(int).tolist(),
            box(chs, radius).astype(int).tolist(),
        ))

    def _apply_stabilization(self, frames, transforms, smooth_t):
        h, w = frames[0].shape[:2]
        result = [frames[0]]
        cum  = [0.0, 0.0, 0.0]
        scum = [0.0, 0.0, 0.0]
        for i, (t, st) in enumerate(zip(transforms, smooth_t)):
            for j in range(3):
                cum[j]  += t[j]
                scum[j] += st[j]
            dx = scum[0] - cum[0]
            dy = scum[1] - cum[1]
            da = scum[2] - cum[2]
            ca, sa = np.cos(da), np.sin(da)
            M = np.array([[ca, -sa, dx], [sa, ca, dy]], dtype=np.float32)
            result.append(cv2.warpAffine(frames[i + 1], M, (w, h)))
        return result

    def _crop(self, frame, cx, cy, cw, ch, src_w, src_h):
        x1 = max(0, min(int(cx - cw / 2), src_w - cw))
        y1 = max(0, min(int(cy - ch / 2), src_h - ch))
        return frame[y1:y1 + ch, x1:x1 + cw]

    def _fill_borders(self, frame, black_thresh=18, inpaint_radius=8):
        """
        Detect black border regions introduced by warp/stabilization and
        fill them with OpenCV Navier-Stokes inpainting so there are no
        visible black edges in the output.

        Strategy:
          - Build a mask of pixels that are nearly-black AND touching an edge
          - Dilate slightly so inpainting has enough context
          - Apply cv2.inpaint with INPAINT_NS
          - Only inpaint if the border region is non-trivial (>0.1% of frame)
        """
        h, w = frame.shape[:2]

        # Pixels darker than threshold in all channels = candidate border
        dark = np.all(frame < black_thresh, axis=2).astype(np.uint8) * 255

        # Keep only connected components that touch the image edge
        # (avoids inpainting dark jerseys / shadows in the middle of the frame)
        border_mask = np.zeros_like(dark)
        # Flood fill from each corner to find edge-connected dark regions
        tmp = dark.copy()
        edge_seeds = []
        for y in range(0, h, max(1, h // 20)):
            edge_seeds += [(y, 0), (y, w - 1)]
        for x in range(0, w, max(1, w // 20)):
            edge_seeds += [(0, x), (h - 1, x)]

        for (y, x) in edge_seeds:
            if dark[y, x] == 255:
                # Flood fill to find this connected dark region
                flood = dark.copy()
                cv2.floodFill(flood, None, (x, y), 128)
                flooded = (flood == 128).astype(np.uint8) * 255
                border_mask = cv2.bitwise_or(border_mask, flooded)

        # Skip if negligible border (< 0.1% of frame area = noise / deep shadow)
        border_px = int(np.count_nonzero(border_mask))
        if border_px < int(h * w * 0.001):
            return frame

        # Dilate mask slightly so inpainting blends more naturally
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        border_mask = cv2.dilate(border_mask, k, iterations=2)
        border_mask = np.clip(border_mask, 0, 255).astype(np.uint8)

        # Navier-Stokes inpainting
        filled = cv2.inpaint(frame, border_mask, inpaint_radius, cv2.INPAINT_NS)
        return filled

    def _interpolate_frames(self, input_path, output_path, source_fps, target_fps=60.0):
        """
        Use ffmpeg minterpolate to bump frame rate to target_fps.
        Splits video into CPU-count chunks for parallelism since minterpolate
        is single-threaded. Falls back gracefully on error.

        Returns True if interpolation succeeded, False otherwise.
        """
        if source_fps >= target_fps * 0.95:
            # Already at or above target — skip
            return False

        # Get video duration
        probe = subprocess.run([
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            input_path
        ], capture_output=True, text=True)
        try:
            duration = float(probe.stdout.strip())
        except Exception:
            duration = 0.0

        if duration <= 0:
            return False

        # Choose chunk count: 1 chunk per available CPU, min 1, max 8
        n_chunks = min(8, max(1, multiprocessing.cpu_count()))

        # For short clips (<30s), single pass is faster
        if duration < 30.0 or n_chunks == 1:
            return self._minterp_single(input_path, output_path, target_fps)

        # Split, interpolate chunks in parallel, concat
        return self._minterp_parallel(input_path, output_path, target_fps,
                                       duration, n_chunks)

    def _minterp_single(self, inp, out, fps):
        """Run minterpolate on a single file."""
        cmd = [
            "ffmpeg", "-y", "-i", inp,
            "-vf", (
                f"minterpolate=fps={fps}:"
                "mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1:scd=fdiff"
            ),
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-an", out
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        return r.returncode == 0

    def _minterp_parallel(self, inp, out, fps, duration, n_chunks):
        """Split → parallel minterpolate → concat."""
        tmpdir = tempfile.mkdtemp()
        chunk_dur = duration / n_chunks

        # Split into chunks
        chunk_paths = []
        for i in range(n_chunks):
            start = i * chunk_dur
            chunk_out = os.path.join(tmpdir, f"chunk_{i:03d}.mp4")
            cmd = [
                "ffmpeg", "-y",
                "-ss", str(start), "-t", str(chunk_dur),
                "-i", inp,
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18",
                "-an", chunk_out
            ]
            subprocess.run(cmd, capture_output=True, timeout=300)
            if os.path.exists(chunk_out):
                chunk_paths.append(chunk_out)

        if not chunk_paths:
            return False

        # Interpolate each chunk (parallel via Python threads)
        interp_paths = []
        import concurrent.futures

        def interp_chunk(chunk_path):
            out_path = chunk_path.replace(".mp4", "_interp.mp4")
            ok = self._minterp_single(chunk_path, out_path, fps)
            return out_path if ok else chunk_path  # fallback to original chunk

        with concurrent.futures.ThreadPoolExecutor(max_workers=n_chunks) as ex:
            futures = {ex.submit(interp_chunk, p): p for p in chunk_paths}
            for fut in concurrent.futures.as_completed(futures):
                result = fut.result()
                interp_paths.append(result)

        # Sort by chunk index (futures may complete out of order)
        interp_paths.sort()

        # Write concat list
        concat_list = os.path.join(tmpdir, "concat.txt")
        with open(concat_list, "w") as f:
            for p in interp_paths:
                f.write(f"file '{p}'\n")

        # Concat
        cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", concat_list,
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-an", out
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

        # Cleanup temp files
        import shutil
        try:
            shutil.rmtree(tmpdir)
        except Exception:
            pass

        return r.returncode == 0

    def _reencode(self, original, raw, output, fps):
        cmd = [
            "ffmpeg", "-y",
            "-i", raw, "-i", original,
            "-map", "0:v:0", "-map", "1:a:0?",
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart", "-r", str(fps),
            output
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            subprocess.run([
                "ffmpeg", "-y", "-i", raw,
                "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                "-movflags", "+faststart", output
            ], check=True, capture_output=True)
