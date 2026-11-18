"""
Detect and crop White Blood Cells (WBCs) from blood smear slide images.

WBCs are identified the way a hematologist sees them:
  1. COLOR  -> nucleus/cytoplasm stains PURPLE (HSV hue mask), RBCs are pink/red
  2. SIZE   -> WBCs are much larger than RBCs and platelets (area + diameter filter)

Usage:
  python 01b_detect_crop_wbc.py slides                 (folder of slide images)
  python 01b_detect_crop_wbc.py slide_001.png          (single image)

Output:
  cropped_wbc/            one PNG per detected WBC
  detection_preview/      slides with boxes drawn (check quality here)
  wbc_detections.json     COCO-style manifest (images + bbox annotations)
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

# ------------------------------------------------------------
# Defaults (tune with CLI flags if your stain/magnification differs)
# ------------------------------------------------------------
DEFAULTS = {
    "hue_low": 105,        # purple range, OpenCV H scale (0-179)
    "hue_high": 165,
    "sat_low": 60,         # purple nuclei are saturated, background is pale
    "val_low": 40,
    "min_area": 800,       # px^2, rejects platelets / debris / specks
    "min_diam": 30,        # px,  equivalent diameter, rejects thin slivers
    "max_aspect": 4.0,     # w/h ratio, rejects elongated artifacts/folds
    "padding": 12,         # px added around the bbox before cropping
    # --- deep-purple core test (kills pale lavender background/RBC ghost areas
    #     that pass the hue check in phone photos of the microscope) ---
    "core_sat": 100,       # a "deep purple" pixel has saturation >= this
    "core_val": 190,       # ... and value (brightness) <= this
    "core_frac": 0.12,     # >=12% of the cell's pixels must be deep purple
    "nms_iou": 0.30,       # drop a box that overlaps a kept box by this IoU
    "min_sharp": 0,        # optional: min Laplacian variance (kills blurry
                           # smudges). 0 = off. On soft phone photos, ~20-30
                           # rejects blur but may also reject soft real cells.
}

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def detect_wbcs(image_bgr, cfg):
    """Return list of (x1, y1, x2, y2) boxes for detected WBCs."""

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    lower = np.array([cfg.hue_low, cfg.sat_low, cfg.val_low], dtype=np.uint8)
    upper = np.array([cfg.hue_high, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower, upper)

    # clean single-pixel noise, then fill holes inside the nucleus
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []

    for cnt in contours:

        area = cv2.contourArea(cnt)
        if area < cfg.min_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        diam = 2 * np.sqrt(area / np.pi)  # equivalent-circle diameter
        if diam < cfg.min_diam:
            continue

        aspect = max(w, h) / max(min(w, h), 1)
        if aspect > cfg.max_aspect:
            continue

        # deep-purple core test: a real WBC nucleus has a solid dark,
        # saturated purple core. Pale lavender background / out-of-focus
        # RBC areas pass the hue check but never pass this.
        comp = np.zeros(mask.shape, np.uint8)
        cv2.drawContours(comp, [cnt], -1, 255, -1)
        comp_px = comp > 0
        deep_px = comp_px & (S >= cfg.core_sat) & (V <= cfg.core_val)
        if deep_px.sum() / max(comp_px.sum(), 1) < cfg.core_frac:
            continue

        # optional sharpness check (Laplacian variance of the bbox region)
        if cfg.min_sharp > 0:
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            region = gray[y:y + h, x:x + w]
            if cv2.Laplacian(region, cv2.CV_64F).var() < cfg.min_sharp:
                continue

        boxes.append((x, y, x + w, y + h))

    # biggest cells first so crop numbering follows visual importance
    boxes.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)

    # drop near-duplicate boxes (one cell split by glare / touching cells)
    kept = []
    for b in boxes:
        if all(iou(b, k) < cfg.nms_iou for k in kept):
            kept.append(b)

    return kept, mask


def iou(a, b):
    """Intersection-over-union of two (x1, y1, x2, y2) boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(ix2 - ix1, 0) * max(iy2 - iy1, 0)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def process_image(image_path, out_crop_dir, out_preview_dir, cfg):
    """Detect WBCs in one slide, save crops + preview. Returns record."""

    image_path = Path(image_path)
    stem = image_path.stem

    img = cv2.imread(str(image_path))
    if img is None:
        print(f"  [SKIP] cannot read: {image_path}")
        return None

    boxes, _ = detect_wbcs(img, cfg)

    H, W = img.shape[:2]
    pad = cfg.padding

    annotations = []
    n_saved = 0

    for i, (x1, y1, x2, y2) in enumerate(boxes, start=1):

        # crop with padding, clamped to image bounds
        cx1, cy1 = max(x1 - pad, 0), max(y1 - pad, 0)
        cx2, cy2 = min(x2 + pad, W), min(y2 + pad, H)

        crop_name = f"{stem}_wbc{i:03d}.png"
        crop = img[cy1:cy2, cx1:cx2]

        if crop.size == 0:
            continue

        cv2.imwrite(str(out_crop_dir / crop_name), crop)
        n_saved += 1

        annotations.append({
            "id": len(annotations) + 1,
            "image_id": None,  # filled by caller
            "bbox_xyxy": [int(x1), int(y1), int(x2), int(y2)],
            "bbox_coco": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
            "area_px": int((x2 - x1) * (y2 - y1)),
            "file_name": crop_name,
        })

    # preview image with boxes
    preview = img.copy()
    for i, (x1, y1, x2, y2) in enumerate(boxes, start=1):
        cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(preview, f"WBC {i}", (x1, max(y1 - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imwrite(str(out_preview_dir / f"{stem}_preview.jpg"), preview)

    print(f"  {image_path.name}: {n_saved} WBC(s) detected")

    return {
        "file_name": image_path.name,
        "width": W,
        "height": H,
        "annotations": annotations,
    }


def main():

    ap = argparse.ArgumentParser(description="Detect and crop WBCs from slides")

    ap.add_argument("input", help="slide image file or folder of images")
    ap.add_argument("--out", default="cropped_wbc", help="cropped cells output folder")
    ap.add_argument("--preview", default="detection_preview", help="preview folder")
    ap.add_argument("--manifest", default="wbc_detections.json", help="output manifest")

    ap.add_argument("--hue-low", type=int, default=DEFAULTS["hue_low"])
    ap.add_argument("--hue-high", type=int, default=DEFAULTS["hue_high"])
    ap.add_argument("--sat-low", type=int, default=DEFAULTS["sat_low"])
    ap.add_argument("--val-low", type=int, default=DEFAULTS["val_low"])
    ap.add_argument("--min-area", type=int, default=DEFAULTS["min_area"],
                    help="min contour area in px^2 (default 800)")
    ap.add_argument("--min-diam", type=int, default=DEFAULTS["min_diam"],
                    help="min equivalent diameter in px (default 30)")
    ap.add_argument("--max-aspect", type=float, default=DEFAULTS["max_aspect"])
    ap.add_argument("--padding", type=int, default=DEFAULTS["padding"])
    ap.add_argument("--core-sat", type=int, default=DEFAULTS["core_sat"],
                    help="saturation that counts as deep purple (default 100)")
    ap.add_argument("--core-val", type=int, default=DEFAULTS["core_val"],
                    help="max brightness that counts as deep purple (default 190)")
    ap.add_argument("--core-frac", type=float, default=DEFAULTS["core_frac"],
                    help="min fraction of deep-purple pixels per cell (default 0.12)")
    ap.add_argument("--nms-iou", type=float, default=DEFAULTS["nms_iou"],
                    help="drop boxes overlapping a kept box above this IoU (default 0.30)")
    ap.add_argument("--min-sharp", type=float, default=DEFAULTS["min_sharp"],
                    help="optional min Laplacian variance; rejects blurry smudges (0 = off)")

    cfg = ap.parse_args()

    in_path = Path(cfg.input)
    if in_path.is_dir():
        slides = sorted(
            p for p in in_path.iterdir()
            if p.suffix.lower() in IMAGE_EXTS
        )
    elif in_path.is_file():
        slides = [in_path]
    else:
        raise FileNotFoundError(f"Input not found: {in_path}")

    if not slides:
        raise FileNotFoundError(f"No images found in: {in_path}")

    out_crop_dir = Path(cfg.out)
    out_preview_dir = Path(cfg.preview)
    out_crop_dir.mkdir(parents=True, exist_ok=True)
    out_preview_dir.mkdir(parents=True, exist_ok=True)

    print(f"Slides to process : {len(slides)}")
    print(f"Cropped output    : {out_crop_dir.resolve()}")
    print(f"Previews          : {out_preview_dir.resolve()}")
    print(f"Purple hue range  : {cfg.hue_low}-{cfg.hue_high}, "
          f"min area {cfg.min_area} px^2, min diam {cfg.min_diam} px")
    print("-" * 60)

    records = []
    for slide in slides:
        rec = process_image(slide, out_crop_dir, out_preview_dir, cfg)
        if rec:
            records.append(rec)

    # COCO-style manifest
    images, annotations = [], []
    for img_id, rec in enumerate(records, start=1):
        rec_img_id = img_id
        images.append({
            "id": rec_img_id,
            "file_name": rec["file_name"],
            "width": rec["width"],
            "height": rec["height"],
        })
        for ann in rec["annotations"]:
            ann["image_id"] = rec_img_id
            annotations.append(ann)

    with open(cfg.manifest, "w", encoding="utf-8") as f:
        json.dump({"images": images, "annotations": annotations},
                  f, indent=2, ensure_ascii=False)

    total = sum(len(r["annotations"]) for r in records)
    print("-" * 60)
    print(f"DONE: {total} WBC(s) cropped from {len(records)} slide(s)")
    print(f"Manifest: {Path(cfg.manifest).resolve()}")


if __name__ == "__main__":
    main()
