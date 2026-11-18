import os
import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# =========================
# CONFIG
# =========================

MODEL_NAME = "facebook/nllb-200-distilled-600M"

SRC_LANG = "eng_Latn"
TGT_LANG = "urd_Arab"

INPUT_FILE = "NONVALIDATE_WBC_leukemia_ATT_VQA_TEST.json"
OUTPUT_FILE = "translated_nonvalidated_test_data.json"

BATCH_SIZE = 16
SAVE_EVERY = 100

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", DEVICE)

if DEVICE == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# =========================
# LOAD MODEL
# =========================

print("Loading NLLB model...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    src_lang=SRC_LANG
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME
).to(DEVICE)

model.eval()

forced_bos_token_id = tokenizer.convert_tokens_to_ids(TGT_LANG)


# =========================
# TRANSLATION FUNCTION
# =========================

def translate_batch(texts):

    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            max_length=256,
            num_beams=1,
            do_sample=False
        )

    return tokenizer.batch_decode(
        outputs,
        skip_special_tokens=True
    )


def save_json(data, path):

    temp_path = path + ".tmp"

    with open(temp_path, "w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=4
        )

    os.replace(temp_path, path)


# =========================
# LOAD OR RESUME
# =========================

if os.path.exists(OUTPUT_FILE):

    print("Resuming from existing translated file...")

    with open(OUTPUT_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)

else:

    print("Reading input COCO-VQA file...")

    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)


if "annotations" not in data:
    raise ValueError("The JSON file does not contain an annotations section.")


annotations = data["annotations"]

print("Total annotations:", len(annotations))
print("Starting translation...")


# =========================
# TRANSLATE
# =========================

for start in tqdm(
    range(0, len(annotations), BATCH_SIZE),
    desc="Translation batches"
):

    batch = annotations[start:start + BATCH_SIZE]

    texts = []
    locations = []

    for local_index, annotation in enumerate(batch):

        question = annotation.get("Question")
        answer = annotation.get("Answer")

        if (
            isinstance(question, str)
            and question.strip()
            and not annotation.get("Question_urdu")
        ):
            texts.append(question.strip())
            locations.append(
                (local_index, "Question_urdu")
            )

        if (
            isinstance(answer, str)
            and answer.strip()
            and not annotation.get("Answer_urdu")
        ):
            texts.append(answer.strip())
            locations.append(
                (local_index, "Answer_urdu")
            )

    if texts:

        translations = translate_batch(texts)

        for location, translation in zip(
            locations,
            translations
        ):
            local_index, output_key = location
            batch[local_index][output_key] = translation

    processed = min(
        start + BATCH_SIZE,
        len(annotations)
    )

    if processed % SAVE_EVERY < BATCH_SIZE:
        save_json(data, OUTPUT_FILE)
        print(f"Saved: {processed}/{len(annotations)}")


# =========================
# FINAL SAVE
# =========================

save_json(data, OUTPUT_FILE)

print("\nTranslation completed.")
print("Total annotations:", len(annotations))
print("Saved as:", OUTPUT_FILE)