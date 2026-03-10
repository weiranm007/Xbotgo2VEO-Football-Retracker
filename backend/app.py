"""
Football Retracker Backend
Supports three source modes:
  xbotgo     — rotating AI tracking camera (original use case)
  static     — single wide-angle or static camera covering full pitch
  dual_phone — two phones side-by-side, stitched via homography
"""

import os
import uuid
import json
import threading
import traceback
from pathlib import Path
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from processor import VideoProcessor

app = Flask(__name__, static_folder="../frontend", static_url_path="/static")
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024 * 1024  # 4GB max upload

UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

VALID_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".mts", ".m2ts"}

jobs = {}


def _save_upload(file_obj, job_id, suffix=""):
    ext = Path(file_obj.filename).suffix.lower()
    if ext not in VALID_EXTS:
        raise ValueError(f"Unsupported format: {ext}")
    dest = UPLOAD_DIR / f"{job_id}{suffix}{ext}"
    # Stream to disk in 4MB chunks — avoids MemoryError on large videos
    CHUNK = 4 * 1024 * 1024
    with open(str(dest), "wb") as out:
        while True:
            chunk = file_obj.stream.read(CHUNK)
            if not chunk:
                break
            out.write(chunk)
    return str(dest)


# Absolute path to frontend folder (works regardless of working directory)
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"

@app.route("/")
def index():
    return send_file(str(FRONTEND_DIR / "index.html"))

@app.route("/<path:filename>")
def static_files(filename):
    safe = FRONTEND_DIR / filename
    if safe.exists():
        return send_file(str(safe))
    return jsonify({"error": "not found"}), 404

@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "version": "1.0.0"})


@app.route("/api/upload", methods=["POST"])
def upload():
    source_mode = request.form.get("source_mode", "xbotgo")

    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    job_id = str(uuid.uuid4())

    try:
        input_path_a = _save_upload(request.files["video"], job_id, "_a")
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    input_path_b = None
    if source_mode == "wide_angle" and "video_b" in request.files and request.files["video_b"].filename:
        try:
            input_path_b = _save_upload(request.files["video_b"], job_id, "_b")
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

    options = {
        "output_width":  int(request.form.get("output_width",  1920)),
        "output_height": int(request.form.get("output_height", 1080)),
        "smoothing":     float(request.form.get("smoothing",   0.92)),
        "zoom_factor":   float(request.form.get("zoom_factor", 1.0)),
        "stabilize":     request.form.get("stabilize", "true").lower() == "true",
        "padding_ratio": float(request.form.get("padding_ratio", 0.35)),
        "source_mode":   source_mode,
        "interpolate":   request.form.get("interpolate", "false").lower() == "true",
    }

    jobs[job_id] = {"status": "queued", "progress": 0, "message": "Queued", "output_path": None}

    thread = threading.Thread(
        target=run_job,
        args=(job_id, input_path_a, input_path_b, options),
        daemon=True
    )
    thread.start()

    return jsonify({"job_id": job_id})


def run_job(job_id, input_path_a, input_path_b, options):
    try:
        output_path = str(OUTPUT_DIR / f"{job_id}_processed.mp4")

        def progress_cb(pct, msg):
            jobs[job_id]["progress"] = pct
            jobs[job_id]["message"]  = msg
            jobs[job_id]["status"]   = "processing"

        jobs[job_id]["status"] = "processing"
        processor = VideoProcessor(progress_callback=progress_cb)
        processor.process(input_path_a, output_path, options, input_path_b=input_path_b)

        jobs[job_id]["status"]      = "done"
        jobs[job_id]["progress"]    = 100
        jobs[job_id]["message"]     = "Complete"
        jobs[job_id]["output_path"] = output_path

    except Exception as e:
        jobs[job_id]["status"]  = "error"
        jobs[job_id]["message"] = str(e)
        traceback.print_exc()
    finally:
        for p in [input_path_a, input_path_b]:
            if p:
                try:
                    os.remove(p)
                except Exception:
                    pass


@app.route("/api/status/<job_id>")
def status(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)


@app.route("/api/download/<job_id>")
def download(job_id):
    job = jobs.get(job_id)
    if not job or job["status"] != "done":
        return jsonify({"error": "Not ready"}), 404
    return send_file(job["output_path"], as_attachment=True, download_name="retracked.mp4")


@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
