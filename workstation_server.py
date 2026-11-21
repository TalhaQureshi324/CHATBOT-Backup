"""
Backend for workstation.html — serves the UI and runs the REAL pipeline:

  POST /api/detect  (multipart image)  -> real WBC detection (01b detector):
        boxes + cropped cell PNGs + purple scores + annotated preview
  POST /api/ask      (JSON)            -> real fine-tuned Qwen2-VL answer
        {question, lang: "english"|"urdu", image: <data-url or base64 PNG>}
  GET  /api/sample                      -> a real sample photo from slides/
  GET  /                                 -> workstation.html

Run:  conda run -n vqa python workstation_server.py
      (or: python workstation_server.py  inside the vqa env)
"""

import base64
import glob
import importlib.util
import io
import json
import os
import random
import tempfile
import time

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HTML_PATH = os.path.join(BASE_DIR, "workstation.html")
SLIDES_DIR = os.path.join(BASE_DIR, "slides")

app = FastAPI()


# ------------------------------------------------------------
# real WBC detector (01b_detect_crop_wbc.py)
# ------------------------------------------------------------

_detector = None


def load_detector():
    global _detector
    if _detector is None:
        spec = importlib.util.spec_from_file_location(
            "wbc_detector",
            os.path.join(BASE_DIR, "01b_detect_crop_wbc.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _detector = mod
    return _detector


def detect_from_array(img_bgr):
    """Run the real detector; return dict for the UI."""

    import argparse

    det = load_detector()
    cfg = argparse.Namespace(**det.DEFAULTS)

    t0 = time.time()
    boxes, _ = det.detect_wbcs(img_bgr, cfg)
    elapsed = time.time() - t0

    H, W = img_bgr.shape[:2]
    pad = cfg.padding

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    S, V = hsv[:, :, 1], hsv[:, :, 2]

    out_cells = []
    annotated = img_bgr.copy()

    for i, (x1, y1, x2, y2) in enumerate(boxes, start=1):
        cx1, cy1 = max(x1 - pad, 0), max(y1 - pad, 0)
        cx2, cy2 = min(x2 + pad, W), min(y2 + pad, H)
        crop = img_bgr[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            continue

        # purple score: fraction of deep-purple pixels in the box
        roi_s = S[y1:y2, x1:x2].astype(int)
        roi_v = V[y1:y2, x1:x2].astype(int)
        score = float(((roi_s >= cfg.core_sat) & (roi_v <= cfg.core_val)).mean())

        ok, buf = cv2.imencode(".png", crop)
        data_url = ("data:image/png;base64," +
                    base64.b64encode(buf.tobytes()).decode())

        out_cells.append({
            "id": i,
            "code": f"WBC #{i:02d}",
            "x": int(x1), "y": int(y1),
            "w": int(x2 - x1), "h": int(y2 - y1),
            "crop": data_url,
            "score": round(score * 100, 1),
        })

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (5, 190, 80), 3)
        cv2.putText(annotated, f"WBC {i}", (x1, max(y1 - 10, 28)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (5, 190, 80), 2)

    ok, buf = cv2.imencode(".png", annotated)
    annotated_url = ("data:image/png;base64," +
                     base64.b64encode(buf.tobytes()).decode())

    return {
        "count": len(out_cells),
        "elapsed": round(elapsed, 2),
        "width": int(W),
        "height": int(H),
        "cells": out_cells,
        "annotated": annotated_url,
    }


# ------------------------------------------------------------
# real VQA (fine-tuned Qwen2-VL via vqa_ui)
# ------------------------------------------------------------

def answer_from_data_url(data_url, question, lang):
    """Save the data-url image to a temp file and ask the fine-tuned model."""

    import vqa_ui  # local module; torch loads lazily inside

    if not data_url or not question:
        return "ERROR: no image or question provided"

    header, _, b64 = data_url.partition(",")
    if not b64:
        return "ERROR: invalid image data"

    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    try:
        tmp.write(base64.b64decode(b64))
        tmp.close()
        return vqa_ui.answer_question(tmp.name, question, lang)
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


# ------------------------------------------------------------
# routes
# ------------------------------------------------------------

@app.get("/")
def index():
    return FileResponse(HTML_PATH)


@app.get("/api/sample")
def sample():
    files = sorted(glob.glob(os.path.join(SLIDES_DIR, "*.jpg"))) + \
            sorted(glob.glob(os.path.join(SLIDES_DIR, "*.png")))
    if not files:
        return JSONResponse({"error": "no samples in slides/"}, status_code=404)
    path = random.choice(files)
    return FileResponse(path, media_type="image/jpeg",
                        headers={"X-Sample-Name": os.path.basename(path)})


@app.post("/api/detect")
async def detect(file: UploadFile = File(...)):
    raw = await file.read()
    arr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse({"error": "could not read image"}, status_code=400)
    return detect_from_array(img)


@app.post("/api/ask")
async def ask(payload: dict = None):
    payload = payload or {}
    question = str(payload.get("question", "")).strip()
    lang = payload.get("lang", "english")
    image = payload.get("image", "")
    if not question or not image:
        return JSONResponse({"error": "question and image required"}, status_code=400)
    answer = answer_from_data_url(image, question, lang)
    return {"answer": answer}


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
