"""
Interactive VQA tester — ask your fine-tuned Qwen2-VL models questions about
detected WBC crops, in Urdu or English, from a browser page.

NEW standalone file. None of the existing pipeline scripts are changed.

Two ways to use:

1) Browser UI (recommended):
   python vqa_ui.py
   -> open the printed URL (http://<machine-ip>:7860) in any browser
   -> pick an image from cropped_wbc (or upload one), type a question,
      choose Urdu or English model, click Ask.

2) One quick terminal question (no browser):
   python vqa_ui.py --ask "What is the morphology of the nucleus?" ^
       --image cropped_wbc/some_cell.png --lang english

NOTE: needs a machine where PyTorch runs (lab server, or this PC only
after the security-policy problem is solved). Torch is imported lazily,
so the UI itself opens even before the model loads.
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

DEFAULT_MAX_NEW_TOKENS = 80


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
# browser UI
# ------------------------------------------------------------

def build_ui(image_dir, share, port):

    import gradio as gr

    def choices():
        return [os.path.basename(p) for p in list_images(image_dir)]

    with gr.Blocks(title="WBC VQA tester") as ui:
        gr.Markdown("## WBC VQA tester — ask your fine-tuned Qwen2-VL")

        with gr.Row():
            with gr.Column():
                dropdown = gr.Dropdown(
                    choices=choices(),
                    label=f"Pick a detected WBC from {image_dir}",
                    type="value",
                )
                upload = gr.Image(
                    type="filepath", label="...or upload any cell image"
                )
                refresh_btn = gr.Button("Refresh image list")
            with gr.Column():
                lang = gr.Radio(
                    ["english", "urdu"],
                    value="english",
                    label="Model (match your question language)",
                )
                question = gr.Textbox(
                    label="Question",
                    placeholder="What is the morphology of the nucleus?",
                )
                ask_btn = gr.Button("Ask", variant="primary")
                answer = gr.Textbox(label="Answer", lines=4)

        def ask(picked, uploaded_path, q, lang_choice):
            img = uploaded_path
            if not img and picked:
                img = os.path.join(image_dir, picked)
            return answer_question(img, q, lang_choice)

        ask_btn.click(ask, [dropdown, upload, question, lang], answer)
        question.submit(ask, [dropdown, upload, question, lang], answer)
        refresh_btn.click(
            lambda: gr.update(choices=choices()), None, dropdown
        )

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
