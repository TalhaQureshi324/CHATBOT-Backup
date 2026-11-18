# WBC VQA System — Qwen2-VL Fine-tuning & Interactive Testing (CHATBOT-Backup)

White Blood Cell (WBC) visual-question-answering system built on
**Qwen2-VL-2B-Instruct** with LoRA fine-tuning, in **English and Urdu**.

Pipeline: detect WBCs in blood-smear images → crop them → fine-tune / evaluate
Qwen2-VL on cell morphology QA → ask questions interactively in a browser UI.

## Repository contents

| Path | What it is |
|---|---|
| `01b_detect_crop_wbc.py` | WBC detection + cropping (purple-stain color + size + deep-purple-core test, NMS) |
| `english_traing.py` | LoRA fine-tuning script (Qwen2-VL-2B, English) |
| `test.py`, `test_batch.py` | Evaluation / prediction scripts (BLEU, ROUGE, BERTScore) |
| `05_evaluate_doc_validated_test_vqa.py` | Doctor-validated test evaluation |
| `07_calculate_scores_only.py` | Re-score saved prediction CSVs |
| `vqa_ui.py` | Interactive Gradio UI — pick/upload a cell image, ask in English or Urdu, get answers from the fine-tuned models |
| `00_config.py` | Paths and training hyper-parameters |
| `models/` | Fine-tuned Urdu LoRA adapter (17 MB) |
| `models_english/` | Fine-tuned English LoRA adapter (17 MB) |
| `English_Train.json`, `English_Test.json`, `Q_A_*.json`, `WBC_leukemia_ATT_VQA.json`, ... | VQA datasets (COCO-style, English + Urdu) |
| `slides/` | Sample microscope photos (phone through eyepiece) |
| `cropped_wbc/` | Detected + cropped WBCs from the sample photos |
| `detection_preview/` | Detection previews with boxes (quality check) |
| `RUN_COMMANDS.md` | Detailed run guide incl. the working Windows setup |
| `translate_nllb.py` | NLLB translation used to build the Urdu dataset |

## Quick start

```bash
pip install torch transformers peft accelerate gradio pillow pandas opencv-python
```

### 1) Detect and crop WBCs

```bash
python 01b_detect_crop_wbc.py slides          # folder of smear images
python 01b_detect_crop_wbc.py slide_001.png   # single image
```

Crops → `cropped_wbc/`, quality previews → `detection_preview/`,
manifest → `wbc_detections.json`. Tuning flags: `--min-area`, `--min-diam`,
`--core-frac`, `--hue-low/high`, `--padding` (see `RUN_COMMANDS.md`).

### 2) Ask questions (interactive UI)

```bash
python vqa_ui.py            # then open http://127.0.0.1:7860
```

Pick a crop (or upload), type a question, choose **english** or **urdu**
(match the question language), click Ask. Runs on CPU or GPU. One-shot mode:

```bash
python vqa_ui.py --ask "What is the morphology of the nucleus?" \
    --image cropped_wbc/some_cell.png --lang english
```

### 3) Fine-tune (LoRA)

Requires the cell image folders referenced by the dataset JSONs
(e.g. `PBC_dataset_normal_DIB/...`, `train/train_resize_images/...` — not
included in this repo due to size). Then:

```bash
python english_traing.py
```

### 4) Evaluate

```bash
python test_batch.py    # predictions CSV + BLEU / ROUGE / BERTScore
```

## Notes

- The base model `Qwen/Qwen2-VL-2B-Instruct` downloads automatically from
  HuggingFace on first use (~4.4 GB); the LoRA adapters are in this repo.
- Training/evaluation scripts were originally run on a Linux GPU server;
  paths in `00_config.py` are set for the local Windows copy — adjust as
  needed.
- Detection accuracy on phone-through-eyepiece photos is limited by image
  quality; mounted-camera captures give the best results.

## Acknowledgements

Built on the WBC/leukemia VQA dataset and fine-tuning work by Hajra et al.
