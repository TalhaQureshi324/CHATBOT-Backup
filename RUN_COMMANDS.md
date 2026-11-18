# How to run the WBC pipeline (this PC — Windows + GTX 1080 Ti)

All commands are run from `C:\QWEN`.

> **Update (Nov 17):** the VQA UI now WORKS on this PC (CPU). The security
> policy blocks newest package builds, so everything runs from a special
> conda env `vqa` with older, policy-passing versions. See section 4.

## 4. Interactive VQA UI (browser — works on this PC)

```bat
cd C:\QWEN
conda run -n vqa python vqa_ui.py
```

then open http://127.0.0.1:7860 in your browser. First question after
starting takes ~2-4 min (model loads onto CPU); after that each answer
takes ~1-4 min (CPU speed). Pick image from cropped_wbc, type question,
choose english or urdu model, Ask.

One-shot terminal question (no browser):

```bat
conda run -n vqa python vqa_ui.py --ask "What is the morphology of the nucleus?" --image cropped_wbc\WIN_20260924_15_54_02_Pro_wbc001.png --lang english
```

**GPU is active** (GTX 1080 Ti, fp16): first question after starting ≈ 7 s
(includes model load), every question after that ≈ 2 s.

The `vqa` env (Python 3.11) has policy-compatible versions pinned:
torch 2.5.1+cu121 (GPU), torchvision 0.20.1+cu121, transformers 4.55.4,
peft 0.21.3, accelerate 1.3.0, huggingface-hub 0.36.0, numpy 2.2.6,
pandas 2.2.3, regex 2025.10.22, gradio 6.17.3. Don't `pip install -U`
inside it — newest builds get blocked by the WDAC policy again.
(transformers must stay on 4.55.x: 5.x requires torch>=2.6, and older
than 4.52 can't load the adapters' `language_model` key layout.)
Base Qwen2-VL-2B weights are already cached in
`C:\Users\assad\.cache\huggingface\hub` (no download needed).
The GPU torch wheel is kept at `C:\QWEN\torch-2.5.1+cu121-...whl`
so the env can be rebuilt without re-downloading.


> **⚠ GPU training/eval is BLOCKED on this PC.** An enterprise WDAC
> (Application Control) security policy blocks unsigned DLLs — `import torch`
> fails on `fbgemm.dll`, and numpy's `numpy.random` pyd is blocked too.
> Only step 1 (WBC detect + crop) runs here; it uses opencv/numpy/pillow
> which passed the policy. See "Notes" at the bottom for options.

## 0. One-time environment setup

```bat
pip install opencv-python numpy pillow
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install transformers peft accelerate pandas tqdm evaluate bert-score rouge-score nltk openpyxl
```

Verify GPU works:

```bat
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 1. Detect + crop WBCs from slide images  (NEW: 01b_detect_crop_wbc.py)

Detects white blood cells by **purple color + large size** (RBCs are pink and
small, platelets are tiny), crops each WBC, saves a preview with boxes.

Put your slide images in a folder (e.g. `slides\`) then:

```bat
python 01b_detect_crop_wbc.py slides
```

- Cropped cells  -> `cropped_wbc\`
- Check quality  -> `detection_preview\`  (green boxes = detections)
- Manifest       -> `wbc_detections.json`

Tuning (if it misses cells or catches junk):

```bat
python 01b_detect_crop_wbc.py slides --min-area 500 --min-diam 25
python 01b_detect_crop_wbc.py slides --hue-low 100 --hue-high 170
python 01b_detect_crop_wbc.py slides --padding 20
```

For phone photos of the microscope (washed-out, purple-fringing): the
detector requires each cell to have a **deep-purple core**
(`--core-frac 0.12` of pixels with saturation >= `--core-sat 100` and
brightness <= `--core-val 190`). If real WBCs get rejected, lower
`--core-frac` to 0.08 or `--core-sat` to 80; if junk comes back, raise them.

Higher magnification images -> raise `--min-area` / `--min-diam`.
Run on a single file: `python 01b_detect_crop_wbc.py slide_001.png`

## 2. Train (LoRA fine-tune Qwen2-VL-2B) — needs cell images locally

Copy the cell image folders from the lab server into `C:\QWEN\` so it looks like:

```
C:\QWEN\PBC_dataset_normal_DIB\...          (images used by English_Train/Test.json)
C:\QWEN\train\train_resize_images\*.png     (images used by the WBC/leukemia VQA jsons)
C:\QWEN\test\test_resize_images\*.png
```

Then:

```bat
python english_traing.py
```

- Reads `English_Train.json` / `English_Test_2.json`, saves to `models_english_final_again\`
- Checkpoints auto-resume if interrupted (just rerun the same command)
- If CUDA out of memory: in `00_config.py` set `"batch_size": 4` and
  `"gradient_accumulation_steps": 24`

## 3. Evaluate (predictions + BLEU / ROUGE / BERTScore)

```bat
python test_batch.py      :: batched inference, urdu leukemia test set
python test.py            :: single-image loop, Q_A_test_large.json
```

Both load the base model from HuggingFace + the local LoRA adapter
(`models_english` / `models`) and write `predictions_*.csv` + metric printout.
First run downloads Qwen2-VL-2B (~4.4 GB) into the HF cache.

## Notes for this machine

- **WDAC policy block (the big one):** Windows Code Integrity event log shows
  `torch\lib\fbgemm.dll` and `numpy\random\_bounded_integers.pyd` "did not
  meet the Enterprise signing level requirement" — an enterprise Application
  Control policy. Every PyTorch DLL is unsigned, so `import torch` cannot work
  until the policy is changed. Options:
  1. University/work-managed laptop -> ask IT to allow it, or just run
     training on the lab server (where the images are anyway; set
     `BASE_DIR` in `00_config.py` back to the `/media/iml/...` path there).
  2. Personally owned + admin -> elevated PowerShell:
     `CiTool --list-policies` to identify it, remove if locally created
     (`CiTool --remove-policy <GUID>`); MDM-deployed policies come back.
  3. Install WSL2 (admin + possibly reboot) and run torch inside Linux.
- `00_config.py` now points to `C:\QWEN` (was the lab server path).
- `bf16` was changed to `fp16` in `english_traing.py`, `test.py`,
  `test_batch.py` — the GTX 1080 Ti does not support bfloat16.
- `05_evaluate_doc_validated_test_vqa.py` still has `D:\Thesis\...` paths —
  only needed for the doctor-validated evaluation, edit its CONFIG block if
  you want that one.
