"""
app.py — PV Inspection ground-station demo
============================================
Local Flask app: point it at a folder of anomaly frames (as produced on the
Pi 5) and it runs RGB+thermal multiclassification and fusion on each one.

Run:
    pip install flask torch torchvision ultralytics tensorflow pillow numpy opencv-python
    python app.py
Then open http://127.0.0.1:5001
"""

import os
import glob
import tempfile
import traceback

from flask import Flask, render_template, request, jsonify, send_file

from core_inference import MulticlassFusionEngine

app = Flask(__name__)

_engine = None  # lazy-loaded on first request so the server starts instantly


def get_engine() -> MulticlassFusionEngine:
    global _engine
    if _engine is None:
        _engine = MulticlassFusionEngine()
    return _engine


def find_frame_dirs(root: str) -> list:
    """Recursively finds every folder containing both rgb.jpg and thermal.jpg,
    at any depth — works for a single frame folder, a flat parent folder of
    frame_XXXXXX/ subfolders, or a nested folder from a drag-and-drop upload."""
    root = os.path.abspath(os.path.expanduser(root))
    matches = []
    for dirpath, _dirnames, filenames in os.walk(root):
        if "rgb.jpg" in filenames and "thermal.jpg" in filenames:
            matches.append(dirpath)
    return sorted(matches)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/process", methods=["POST"])
def process():
    data = request.get_json(force=True)
    folder = (data or {}).get("path", "").strip()

    if not folder:
        return jsonify({"error": "No path provided."}), 400
    if not os.path.exists(folder):
        return jsonify({"error": f"Path does not exist: {folder}"}), 400

    frame_dirs = find_frame_dirs(folder)
    if not frame_dirs:
        return jsonify({"error": "No frame folders with rgb.jpg + thermal.jpg found there."}), 400

    try:
        engine = get_engine()
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Failed to load models: {e}"}), 500

    results = []
    errors = []
    for frame_dir in frame_dirs:
        try:
            results.append(engine.process_frame(frame_dir))
        except Exception as e:
            traceback.print_exc()
            errors.append({"frame": os.path.basename(frame_dir.rstrip("/")), "error": str(e)})

    return jsonify({"results": results, "errors": errors, "count": len(results)})


@app.route("/api/process-upload", methods=["POST"])
def process_upload():
    """Receives files dropped/selected in the browser (with their relative paths)
    and reconstructs the frame folder structure in a temp dir before processing."""
    files = request.files.getlist("files")
    rel_paths = request.form.getlist("paths")

    if not files or len(files) != len(rel_paths):
        return jsonify({"error": "No files received."}), 400

    tmp_root = tempfile.mkdtemp(prefix="pv_upload_")
    for f, rel_path in zip(files, rel_paths):
        rel_path = rel_path.replace("\\", "/").lstrip("/")
        dest = os.path.join(tmp_root, rel_path)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        f.save(dest)

    frame_dirs = find_frame_dirs(tmp_root)
    if not frame_dirs:
        return jsonify({"error": "No frame folders with rgb.jpg + thermal.jpg found in that upload."}), 400

    try:
        engine = get_engine()
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Failed to load models: {e}"}), 500

    results = []
    errors = []
    for frame_dir in frame_dirs:
        try:
            results.append(engine.process_frame(frame_dir))
        except Exception as e:
            traceback.print_exc()
            errors.append({"frame": os.path.basename(frame_dir.rstrip("/")), "error": str(e)})

    return jsonify({"results": results, "errors": errors, "count": len(results)})


@app.route("/api/image")
def image():
    """Serves an image straight off disk so the browser can preview rgb/thermal/annotated frames."""
    path = request.args.get("path", "")
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(path) or os.path.splitext(path)[1].lower() not in (".jpg", ".jpeg", ".png"):
        return "Not found", 404
    return send_file(path)


if __name__ == "__main__":
    print("Starting PV Multiclass + Fusion demo on http://127.0.0.1:5001")
    app.run(host="127.0.0.1", port=5001, debug=False)