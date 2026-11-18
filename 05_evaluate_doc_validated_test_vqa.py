import json
import torch
import pandas as pd
from pathlib import Path
from PIL import Image

import evaluate

from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor
)

from peft import PeftModel


# ============================================================
# CONFIG
# ============================================================

BASE_MODEL = "Qwen/Qwen2-VL-2B-Instruct"

ADAPTER_PATH = r"D:\Thesis\Qwen_Finetuning\English_only\models"

TEST_JSON = r"D:\Thesis\Qwen_Finetuning\English_only\data\test_instances.json"

IMAGE_DIR = r"D:\Thesis\Qwen_Finetuning\English_only\test\cropped_cells"

RESULTS_DIR = r"D:\Thesis\Qwen_Finetuning\English_only\results"

OUTPUT_JSON = r"D:\Thesis\Qwen_Finetuning\English_only\results\DOC_VALIDATED_TEST_VQA.json"

OUTPUT_CSV = r"D:\Thesis\Qwen_Finetuning\English_only\results\DOC_VALIDATED_TEST_VQA.csv"

OUTPUT_EXCEL = r"D:\Thesis\Qwen_Finetuning\English_only\results\DOC_VALIDATED_TEST_VQA.xlsx"

MAX_NEW_TOKENS = 80


# ============================================================
# CREATE RESULTS FOLDER
# ============================================================

Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD METRICS
# ============================================================

print("Loading evaluation metrics...")

bleu_metric = evaluate.load("bleu")
rouge_metric = evaluate.load("rouge")
bertscore_metric = evaluate.load("bertscore")


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading processor...")

processor = AutoProcessor.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True
)

print("Loading base model...")

base_model = Qwen2VLForConditionalGeneration.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map={"": 0},
    trust_remote_code=True
)

print("Loading fine-tuned LoRA adapter...")

model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH
)

model.eval()

print("Model ready!")


# ============================================================
# LOAD WHOLE TEST DATA
# ============================================================

with open(TEST_JSON, "r", encoding="utf-8") as f:
    test_data = json.load(f)

samples = test_data

print(f"Total test samples loaded: {len(samples)}")


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize_text(text):
    return str(text).strip().lower()


def generate_answer(image_path, question):

    image = Image.open(image_path).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image
                },
                {
                    "type": "text",
                    "text": question
                }
            ]
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt"
    )

    inputs = {
        k: v.to(model.device)
        if torch.is_tensor(v)
        else v
        for k, v in inputs.items()
    }

    with torch.no_grad():

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False
        )

    generated_ids_trimmed = generated_ids[
        :,
        inputs["input_ids"].shape[1]:
    ]

    prediction = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True
    )[0]

    return prediction.strip()


# ============================================================
# RUN EVALUATION ON WHOLE TEST DATA
# ============================================================

results = []
predictions = []
references = []

correct = 0
error_count = 0

for i, sample in enumerate(samples, start=1):

    image_name = sample["image"]
    question = sample["question"]
    ground_truth = sample["answer"]

    image_path = Path(IMAGE_DIR) / image_name

    print(f"\n[{i}/{len(samples)}]")
    print("Image:", image_name)
    print("Q:", question)

    if not image_path.exists():

        prediction = "ERROR: Image file not found"
        error_count += 1

    else:

        try:
            prediction = generate_answer(
                image_path,
                question
            )

        except Exception as e:
            prediction = f"ERROR: {str(e)}"
            error_count += 1

    exact_match = (
        normalize_text(prediction)
        ==
        normalize_text(ground_truth)
    )

    if exact_match:
        correct += 1

    predictions.append(prediction)
    references.append(ground_truth)

    print("Ground Truth:", ground_truth)
    print("Prediction  :", prediction)
    print("Exact Match :", exact_match)

    results.append({
        "sample_no": i,
        "image": image_name,
        "image_path": str(image_path),
        "question": question,
        "ground_truth_answer": ground_truth,
        "model_prediction": prediction,
        "exact_match": exact_match,
        "doctor_review": "",
        "comments": ""
    })


# ============================================================
# CALCULATE METRICS
# ============================================================

accuracy = (correct / len(results)) * 100
exact_match_score = accuracy

print("\nCalculating BLEU...")
bleu_result = bleu_metric.compute(
    predictions=predictions,
    references=[[ref] for ref in references]
)

print("Calculating ROUGE...")
rouge_result = rouge_metric.compute(
    predictions=predictions,
    references=references
)

print("Calculating BERTScore...")
bertscore_result = bertscore_metric.compute(
    predictions=predictions,
    references=references,
    lang="en"
)

bertscore_precision = (
    sum(bertscore_result["precision"])
    /
    len(bertscore_result["precision"])
)

bertscore_recall = (
    sum(bertscore_result["recall"])
    /
    len(bertscore_result["recall"])
)

bertscore_f1 = (
    sum(bertscore_result["f1"])
    /
    len(bertscore_result["f1"])
)


summary = {
    "evaluation_name": "DOC_VALIDATED_TEST_VQA",
    "total_test_samples": len(results),
    "correct_exact_matches": correct,
    "error_count": error_count,
    "accuracy_percent": accuracy,
    "exact_match_percent": exact_match_score,
    "bleu": bleu_result["bleu"],
    "rouge1": rouge_result["rouge1"],
    "rouge2": rouge_result["rouge2"],
    "rougeL": rouge_result["rougeL"],
    "bertscore_precision": bertscore_precision,
    "bertscore_recall": bertscore_recall,
    "bertscore_f1": bertscore_f1
}


for row in results:
    row.update({
        "overall_accuracy_percent": accuracy,
        "overall_exact_match_percent": exact_match_score,
        "overall_bleu": bleu_result["bleu"],
        "overall_rougeL": rouge_result["rougeL"],
        "overall_bertscore_f1": bertscore_f1
    })


# ============================================================
# SAVE RESULTS
# ============================================================

print("\nSaving JSON...")

with open(
    OUTPUT_JSON,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        {
            "summary_metrics": summary,
            "results": results
        },
        f,
        indent=2,
        ensure_ascii=False
    )


print("Saving CSV...")

df = pd.DataFrame(results)
df.to_csv(
    OUTPUT_CSV,
    index=False,
    encoding="utf-8-sig"
)


print("Saving Excel...")

summary_df = pd.DataFrame([summary])

try:
    with pd.ExcelWriter(
        OUTPUT_EXCEL,
        engine="openpyxl"
    ) as writer:

        summary_df.to_excel(
            writer,
            index=False,
            sheet_name="Summary"
        )

        df.to_excel(
            writer,
            index=False,
            sheet_name="Predictions"
        )

except ModuleNotFoundError:
    print("openpyxl not installed, Excel file not saved.")
    print("Install using: pip install openpyxl")


# ============================================================
# PRINT SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("DOC_VALIDATED_TEST_VQA EVALUATION COMPLETE")
print("=" * 80)

print(f"Total Test Samples  : {len(results)}")
print(f"Correct EM          : {correct}")
print(f"Errors              : {error_count}")
print(f"Accuracy / EM       : {accuracy:.2f}%")
print(f"BLEU                : {bleu_result['bleu']:.4f}")
print(f"ROUGE-1             : {rouge_result['rouge1']:.4f}")
print(f"ROUGE-2             : {rouge_result['rouge2']:.4f}")
print(f"ROUGE-L             : {rouge_result['rougeL']:.4f}")
print(f"BERTScore Precision : {bertscore_precision:.4f}")
print(f"BERTScore Recall    : {bertscore_recall:.4f}")
print(f"BERTScore F1        : {bertscore_f1:.4f}")

print("\nSaved files:")
print(OUTPUT_JSON)
print(OUTPUT_CSV)
print(OUTPUT_EXCEL)

print("=" * 80)