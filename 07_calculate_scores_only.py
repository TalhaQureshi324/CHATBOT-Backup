import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import evaluate


# ============================================================
# CONFIG
# ============================================================

INPUT_CSV = r"D:\Thesis\Qwen_Finetuning\English_only\results\NON_VALIDATED_TEST_VQA.csv"

OUTPUT_CSV = r"D:\Thesis\Qwen_Finetuning\English_only\results\NON_VALIDATED_TEST_VQA_score.csv"


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_CSV)

predictions = df["model_prediction"].astype(str).tolist()
references = df["ground_truth_answer"].astype(str).tolist()


# ============================================================
# NORMALIZATION
# ============================================================

def normalize(text):
    return str(text).strip().lower()


# ============================================================
# EXACT MATCH / ACCURACY
# ============================================================

exact_matches = [
    normalize(pred) == normalize(ref)
    for pred, ref in zip(predictions, references)
]

accuracy = sum(exact_matches) / len(exact_matches) * 100


# ============================================================
# BLEU-1, BLEU-2, BLEU-3, BLEU-4
# ============================================================

smoothie = SmoothingFunction().method1

bleu1_scores = []
bleu2_scores = []
bleu3_scores = []
bleu4_scores = []

for pred, ref in zip(predictions, references):

    pred_tokens = normalize(pred).split()
    ref_tokens = [normalize(ref).split()]

    bleu1_scores.append(
        sentence_bleu(
            ref_tokens,
            pred_tokens,
            weights=(1, 0, 0, 0),
            smoothing_function=smoothie
        )
    )

    bleu2_scores.append(
        sentence_bleu(
            ref_tokens,
            pred_tokens,
            weights=(0.5, 0.5, 0, 0),
            smoothing_function=smoothie
        )
    )

    bleu3_scores.append(
        sentence_bleu(
            ref_tokens,
            pred_tokens,
            weights=(1/3, 1/3, 1/3, 0),
            smoothing_function=smoothie
        )
    )

    bleu4_scores.append(
        sentence_bleu(
            ref_tokens,
            pred_tokens,
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=smoothie
        )
    )


BLEU1 = sum(bleu1_scores) / len(bleu1_scores)
BLEU2 = sum(bleu2_scores) / len(bleu2_scores)
BLEU3 = sum(bleu3_scores) / len(bleu3_scores)
BLEU4 = sum(bleu4_scores) / len(bleu4_scores)


# ============================================================
# ROUGE-L
# ============================================================

print("Calculating ROUGE-L...")

rouge_metric = evaluate.load("rouge")

rouge_result = rouge_metric.compute(
    predictions=predictions,
    references=references
)

ROUGE_L = rouge_result["rougeL"]


# ============================================================
# BERTSCORE
# ============================================================

print("Calculating BERTScore...")

bertscore_metric = evaluate.load("bertscore")

bertscore_result = bertscore_metric.compute(
    predictions=predictions,
    references=references,
    lang="en"
)

BERT_P = sum(bertscore_result["precision"]) / len(bertscore_result["precision"])
BERT_R = sum(bertscore_result["recall"]) / len(bertscore_result["recall"])
BERT_F1 = sum(bertscore_result["f1"]) / len(bertscore_result["f1"])


# ============================================================
# SAVE PER-SAMPLE RESULTS
# ============================================================

df["exact_match"] = exact_matches
df["bleu_1"] = bleu1_scores
df["bleu_2"] = bleu2_scores
df["bleu_3"] = bleu3_scores
df["bleu_4"] = bleu4_scores

df["overall_accuracy_percent"] = accuracy
df["overall_exact_match_percent"] = accuracy
df["overall_bleu_1"] = BLEU1
df["overall_bleu_2"] = BLEU2
df["overall_bleu_3"] = BLEU3
df["overall_bleu_4"] = BLEU4
df["overall_rouge_l"] = ROUGE_L
df["overall_bertscore_precision"] = BERT_P
df["overall_bertscore_recall"] = BERT_R
df["overall_bertscore_f1"] = BERT_F1

df.to_csv(
    OUTPUT_CSV,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# PRINT FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("100 SAMPLE SCORES ONLY")
print("=" * 70)

print(f"Total Samples       : {len(df)}")
print(f"Correct EM          : {sum(exact_matches)}")
print(f"Accuracy / EM       : {accuracy:.2f}%")
print(f"BLEU-1              : {BLEU1:.4f}")
print(f"BLEU-2              : {BLEU2:.4f}")
print(f"BLEU-3              : {BLEU3:.4f}")
print(f"BLEU-4              : {BLEU4:.4f}")
print(f"ROUGE-L             : {ROUGE_L:.4f}")
print(f"BERTScore Precision : {BERT_P:.4f}")
print(f"BERTScore Recall    : {BERT_R:.4f}")
print(f"BERTScore F1        : {BERT_F1:.4f}")

print("\nSaved file:")
print(OUTPUT_CSV)

print("=" * 70)