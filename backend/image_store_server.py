"""Lightweight Flask server to store generated 2D PNGs and return reachable image URLs."""

import base64
import os
from datetime import datetime
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins="*")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/save_image", methods=["POST"])
def save_image():
    """Decode base64 PNG data, save it locally, and return its static URL."""
    try:
        image_b64 = request.form.get("image_b64")
        if not image_b64:
            return jsonify({"error": "No image data provided"}), 400

        if image_b64.startswith("data:image"):
            image_b64 = image_b64.split(",", 1)[1]

        image_data = base64.b64decode(image_b64)
        file_name = f"sd_{datetime.now().strftime('%Y%m%d%H%M%S_%f')}.png"
        file_path = os.path.join(UPLOAD_FOLDER, file_name)

        with open(file_path, "wb") as f:
            f.write(image_data)

        public_base = os.environ.get("PUBLIC_BASE_URL", request.host_url.rstrip("/"))
        file_url = f"{public_base}/uploads/{file_name}"
        return jsonify({"url": file_url})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    """Serve saved PNG images."""
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


if __name__ == "__main__":
    port = int(os.environ.get("IMAGE_STORE_PORT", "8547"))
    app.run(host="0.0.0.0", port=port, threaded=False)
