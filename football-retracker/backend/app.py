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
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from processor import VideoProcessor

app = Flask(__name__)
CORS(app)

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
    path = UPLOAD_DIR / f"{job_id}{suffix}{ext}"
    file_obj.save(str(path))
    return str(path)


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
    if source_mode == "dual_phone":
        if "video_b" not in request.files:
            return jsonify({"error": "dual_phone mode requires a second video (video_b)"}), 400
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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
