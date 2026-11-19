"""
Interactive WBC pipeline tester — full cycle in one page:

  1) upload a FULL microscope frame -> WBCs are detected (purple + size),
     boxes drawn on the frame
  2) every detected WBC is cropped and shown in a gallery
  3) click a crop to select it
  4) ask a question in Urdu or English -> answered about THAT selected WBC

NEW standalone file. None of the existing pipeline scripts are changed —
detection reuses 01b_detect_crop_wbc.detect_wbcs() directly.

Usage:
  python vqa_ui.py            -> open http://127.0.0.1:7860

Terminal one-shot (no browser):
  python vqa_ui.py --ask "What is the morphology of the nucleus?" ^
      --image cropped_wbc/some_cell.png --lang english

NOTE: needs a machine where PyTorch runs (this PC: use the `vqa` conda env).
Torch is imported lazily, so the UI opens even before the model loads.
"""

import argparse
import glob
import os

from PIL import Image

BASE_MODEL = "Qwen/Qwen2-VL-2B-Instruct"

# adapter locations: local PC first, lab server paths as fallback
ADAPTER_CANDIDATES = {
    "english": [
        r"C:\QWEN\models_english",
        "/media/iml/CVML10/Hajra/internvl2_5_finetune/models_english_final_again",
    ],
    "urdu": [
        r"C:\QWEN\models",
        "/media/iml/CVML10/Hajra/internvl2_5_finetune/models",
    ],
}

IMAGE_DIR_CANDIDATES = [
    r"C:\QWEN\cropped_wbc",
    "/media/iml/CVML10/Hajra/QWEN/cropped_wbc",
]

# dataset answers are short sentences; 48 is generous and keeps answers identical
DEFAULT_MAX_NEW_TOKENS = 48


def first_existing(paths):
    for p in paths:
        if os.path.isdir(p):
            return p
    return None


def list_images(folder):
    if not folder or not os.path.isdir(folder):
        return []
    files = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
        files.extend(glob.glob(os.path.join(folder, ext)))
    return sorted(files)


# ------------------------------------------------------------
# model loading (lazy — torch imported only when first needed)
# ------------------------------------------------------------

_STATE = {}


def get_model():
    """Load base model once and attach both LoRA adapters."""

    if "model" in _STATE:
        return _STATE["model"], _STATE["processor"]

    import torch
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    from peft import PeftModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Loading base model {BASE_MODEL} on {device}...")
    processor = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        BASE_MODEL,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )

    for lang, candidates in ADAPTER_CANDIDATES.items():
        path = first_existing(candidates)
        if path is None:
            print(f"[warn] no '{lang}' adapter found, skipping")
            continue
        print(f"Attaching {lang} adapter: {path}")
        if "peft" not in _STATE:
            model = PeftModel.from_pretrained(model, path, adapter_name=lang)
            _STATE["peft"] = True
        else:
            model.load_adapter(path, adapter_name=lang)

    model.eval()
    _STATE["model"] = model
    _STATE["processor"] = processor
    print("Model ready.")
    return model, processor


def answer_question(image_path, question, lang="english",
                    max_new_tokens=DEFAULT_MAX_NEW_TOKENS):
    """Generate an answer for one image + question."""

    if not image_path or not os.path.isfile(image_path):
        return "ERROR: pick or upload an image first"
    if not question or not question.strip():
        return "ERROR: type a question first"

    import torch

    model, processor = get_model()
    model.set_adapter(lang)

    image = Image.open(image_path).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question.strip()},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    inputs = {k: v.to(model.device) if torch.is_tensor(v) else v
              for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             do_sample=False)

    pred = processor.batch_decode(
        out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )[0]
    return pred.strip()


# ------------------------------------------------------------
# WBC detection (reuses 01b_detect_crop_wbc.py exactly as-is)
# ------------------------------------------------------------

DETECTOR_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "01b_detect_crop_wbc.py"
)

_detector = None


def load_detector():
    global _detector
    if _detector is None:
        import importlib.util
        spec = importlib.util.spec_from_file_location("wbc_detector", DETECTOR_PATH)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _detector = mod
    return _detector


def run_detection(frame_path):
    """Detect WBCs on an uploaded microscope frame.

    Returns (annotated_preview_path, gallery_items, status_text,
             first_crop_path, crop_paths, selected_path)
    """

    if not frame_path:
        return None, [], "Upload a microscope frame first.", None, [], None

    import argparse
    import tempfile
    import cv2

    det = load_detector()
    cfg = argparse.Namespace(**det.DEFAULTS)

    img = cv2.imread(frame_path)
    if img is None:
        return None, [], "Could not read that image file.", None, [], None

    boxes, _ = det.detect_wbcs(img, cfg)
    H, W = img.shape[:2]
    pad = cfg.padding

    session_dir = tempfile.mkdtemp(prefix="wbc_ui_")
    preview = img.copy()
    crops = []

    for i, (x1, y1, x2, y2) in enumerate(boxes, start=1):
        cx1, cy1 = max(x1 - pad, 0), max(y1 - pad, 0)
        cx2, cy2 = min(x2 + pad, W), min(y2 + pad, H)
        crop = img[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            continue
        crop_path = os.path.join(session_dir, f"wbc_{i:03d}.png")
        cv2.imwrite(crop_path, crop)
        crops.append(crop_path)

        cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(preview, f"WBC {i}", (x1, max(y1 - 10, 28)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    preview_path = os.path.join(session_dir, "annotated.png")
    cv2.imwrite(preview_path, preview)

    n = len(crops)
    if n:
        status = f"**{n} WBC(s) detected.** Click a crop on the right to select it, then ask your question."
    else:
        status = "No WBCs detected — try a closer/clearer frame (or tune `01b_detect_crop_wbc.py` flags)."

    gallery_items = [(p, f"WBC {i}") for i, p in enumerate(crops, start=1)]
    first = crops[0] if crops else None
    return preview_path, gallery_items, status, first, crops, first


# ------------------------------------------------------------
# browser UI
# ------------------------------------------------------------

def build_ui(image_dir, share, port):

    import gradio as gr

    with gr.Blocks(title="WBC VQA tester") as ui:
        gr.Markdown("## WBC VQA — detect → crop → select → ask")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### 1 · Upload full microscope frame")
                frame_in = gr.Image(type="filepath", label="Microscope frame")
                detect_btn = gr.Button("Detect WBCs", variant="primary")
                annotated = gr.Image(
                    label="Detected WBCs (green boxes)", interactive=False
                )
                status = gr.Markdown()

            with gr.Column():
                gr.Markdown("### 2 · Detected WBC crops — click one to select")
                gallery = gr.Gallery(
                    label="Cropped WBCs", columns=4, height=280,
                    object_fit="contain", interactive=False,
                )
                gr.Markdown("### 3 · Selected WBC")
                selected_img = gr.Image(
                    label="Selected WBC (click a crop above)",
                    interactive=False, height=220,
                )
                qa_upload = gr.Image(
                    type="filepath", height=170,
                    label="...or upload a specific cell image instead",
                )

        with gr.Row():
            with gr.Column():
                lang = gr.Radio(
                    ["english", "urdu"],
                    value="english",
                    label="Model (match your question language)",
                )
            with gr.Column():
                question = gr.Textbox(
                    label="Question",
                    placeholder="What is the morphology of the nucleus?",
                )
                ask_btn = gr.Button("Ask", variant="primary")
                answer = gr.Textbox(label="Answer", lines=3)

        crops_state = gr.State([])
        selected_state = gr.State(None)

        det_outputs = [annotated, gallery, status, selected_img,
                       crops_state, selected_state]

        detect_btn.click(run_detection, [frame_in], det_outputs,
                         api_name="detect")
        frame_in.upload(run_detection, [frame_in], det_outputs)

        def on_gallery_select(evt: gr.SelectData, crop_paths):
            if crop_paths and evt.index is not None and evt.index < len(crop_paths):
                chosen = crop_paths[evt.index]
                return chosen, chosen
            return None, None

        gallery.select(on_gallery_select, [crops_state],
                       [selected_img, selected_state])

        def ask(q, lang_choice, qa_path, sel_path):
            img = qa_path if qa_path else sel_path
            return answer_question(img, q, lang_choice)

        ask_btn.click(ask, [question, lang, qa_upload, selected_state],
                      answer, api_name="ask")
        question.submit(ask, [question, lang, qa_upload, selected_state],
                        answer)

    ui.launch(server_name="0.0.0.0", server_port=port, share=share)


# ------------------------------------------------------------
# main
# ------------------------------------------------------------

def main():

    ap = argparse.ArgumentParser(description="Interactive WBC VQA tester")
    ap.add_argument("--images", default=None,
                    help="folder of cell images to list in the UI")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true",
                    help="create a public gradio link (useful from a server)")
    ap.add_argument("--ask", default=None,
                    help="ask one question from the terminal (no browser)")
    ap.add_argument("--image", default=None, help="image for --ask mode")
    ap.add_argument("--lang", default="english", choices=["english", "urdu"])
    args = ap.parse_args()

    image_dir = args.images or first_existing(IMAGE_DIR_CANDIDATES) or "cropped_wbc"

    if args.ask:
        print("Answer:", answer_question(args.image, args.ask, args.lang))
        return

    print(f"Image folder: {image_dir} ({len(list_images(image_dir))} images)")
    build_ui(image_dir, args.share, args.port)


if __name__ == "__main__":
    main()
