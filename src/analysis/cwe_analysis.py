# src/analysis/cwe_analysis.py
#
# Phase 3 — CWE-Stratified Error Analysis
# Loads best SecBERT checkpoint, runs inference on the full test set,
# and breaks down accuracy and F1 by CWE weakness category.
#
# Run after: train.py has saved checkpoints/secbert_best/
# Outputs:   results/cwe_analysis.json
#            results/cwe_accuracy.csv
#            results/cwe_worst_accuracy.png
#            results/cwe_accuracy_distribution.png
#            results/cwe_confusion_worst5.png

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset, Value
from sklearn.metrics import accuracy_score, f1_score, classification_report
import warnings
warnings.filterwarnings("ignore")

# ── paths ─────────────────────────────────────────────────────────────────────

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

CKPT_DIR    = PROJECT_ROOT / "checkpoints" / "secbert_best"
SPLIT_DIR   = PROJECT_ROOT / "data"        / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LABEL_ORDER         = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
LABEL2ID            = {l: i for i, l in enumerate(LABEL_ORDER)}
ID2LABEL            = {i: l for i, l in enumerate(LABEL_ORDER)}
MIN_SAMPLES_PER_CWE = 50
TOP_N               = 15

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device : {DEVICE}")

# ── load model ────────────────────────────────────────────────────────────────

print(f"\nLoading checkpoint from {CKPT_DIR}...")
assert (CKPT_DIR / "config.json").exists(), \
    f"Checkpoint not found — run train.py first."

tokenizer = AutoTokenizer.from_pretrained(str(CKPT_DIR))
model     = AutoModelForSequenceClassification.from_pretrained(
    str(CKPT_DIR), num_labels=4, id2label=ID2LABEL,
    label2id=LABEL2ID, ignore_mismatched_sizes=True
)
model.eval()
model.to(DEVICE)
print("  Model loaded.")

# ── load test split ───────────────────────────────────────────────────────────

print("\nLoading test split...")
test_df = pd.read_csv(SPLIT_DIR / "test.csv")
test_df["severity"] = test_df["severity"].astype(str).str.strip().str.upper()
test_df["label"]    = test_df["severity"].map(LABEL2ID)
bad = test_df[test_df["label"].isna()]
if len(bad) > 0:
    print(f"  Dropping {len(bad):,} rows with unmapped severity.")
    test_df = test_df.dropna(subset=["label"]).reset_index(drop=True)
test_df["label"] = test_df["label"].astype(int)
print(f"  Rows        : {len(test_df):,}")
print(f"  Unique CWEs : {test_df['cwe'].nunique():,}")

# ── tokenize ──────────────────────────────────────────────────────────────────

print("\nTokenizing...")
test_ds = Dataset.from_dict({
    "description": test_df["description"].tolist(),
    "label":       test_df["label"].tolist()
})
test_ds = test_ds.map(
    lambda b: tokenizer(b["description"], truncation=True,
                        padding="max_length", max_length=256),
    batched=True, batch_size=512, remove_columns=["description"]
)
test_ds = test_ds.cast_column("label", Value("int64"))

# ── inference ─────────────────────────────────────────────────────────────────

print(f"\nInference on {len(test_df):,} samples...")
all_logits, all_labels = [], []
dataloader = torch.utils.data.DataLoader(
    test_ds.with_format("torch"), batch_size=64, shuffle=False
)
with torch.no_grad():
    for i, batch in enumerate(dataloader):
        out = model(input_ids      = batch["input_ids"].to(DEVICE),
                    attention_mask = batch["attention_mask"].to(DEVICE))
        all_logits.append(out.logits.cpu().numpy())
        all_labels.append(batch["label"].numpy())
        if i % 100 == 0:
            print(f"  Batch {i+1}/{len(dataloader)}", end="\r")

all_logits = np.concatenate(all_logits)
all_labels = np.concatenate(all_labels)
all_preds  = np.argmax(all_logits, axis=-1)
print(f"\n  Done.")

test_df = test_df.reset_index(drop=True)
test_df["pred_label"]    = all_preds
test_df["pred_severity"] = test_df["pred_label"].map(ID2LABEL)
test_df["correct"]       = (test_df["label"] == test_df["pred_label"])

# ── overall metrics ───────────────────────────────────────────────────────────

acc         = float(accuracy_score(all_labels, all_preds))
f1_weighted = float(f1_score(all_labels, all_preds, average="weighted",
                              labels=list(range(4))))
f1_macro    = float(f1_score(all_labels, all_preds, average="macro",
                              labels=list(range(4))))
print(f"\n{'='*60}")
print("  OVERALL TEST PERFORMANCE")
print(f"{'='*60}")
print(f"  Accuracy    : {acc:.4f}")
print(f"  F1-weighted : {f1_weighted:.4f}")
print(f"  F1-macro    : {f1_macro:.4f}")
print()
print(classification_report(
    [ID2LABEL[l] for l in all_labels],
    [ID2LABEL[p] for p in all_preds],
    labels=LABEL_ORDER, digits=4
))

# ── CWE breakdown ─────────────────────────────────────────────────────────────

cwe_stats = (
    test_df.groupby("cwe")
    .agg(count=("correct","count"), accuracy=("correct","mean"))
    .reset_index()
)

cwe_f1 = []
for cwe in cwe_stats["cwe"]:
    sub = test_df[test_df["cwe"] == cwe]
    if len(sub) < MIN_SAMPLES_PER_CWE:
        cwe_f1.append(None)
        continue
    try:
        f = f1_score(sub["label"].values, sub["pred_label"].values,
                     average="macro", labels=list(range(4)), zero_division=0)
        cwe_f1.append(round(float(f), 4))
    except Exception:
        cwe_f1.append(None)

cwe_stats["f1_macro"] = cwe_f1
cwe_filtered = (cwe_stats[cwe_stats["count"] >= MIN_SAMPLES_PER_CWE]
                .copy().sort_values("accuracy").reset_index(drop=True))
worst_cwes = cwe_filtered.head(10)["cwe"].tolist()

print(f"\n{'='*60}")
print("  CWE-STRATIFIED RESULTS")
print(f"{'='*60}")
print(f"  CWEs with >= {MIN_SAMPLES_PER_CWE} samples : {len(cwe_filtered)}")
print(f"\n  {TOP_N} worst:")
print(f"  {'CWE':<15} {'Count':>7} {'Accuracy':>10} {'F1-Macro':>10}")
print(f"  {'-'*44}")
for _, row in cwe_filtered.head(TOP_N).iterrows():
    f1s = f"{row['f1_macro']:.4f}" if row["f1_macro"] else "  —"
    print(f"  {row['cwe']:<15} {row['count']:>7,} {row['accuracy']:>10.4f} {f1s:>10}")

print(f"\n  Error breakdown for worst 10 CWEs:")
for cwe in worst_cwes:
    sub    = test_df[test_df["cwe"] == cwe]
    errors = sub[~sub["correct"]]
    if len(errors) == 0:
        continue
    most_pred = errors["pred_severity"].value_counts().index[0]
    true_sev  = sub["severity"].value_counts().index[0]
    print(f"  {cwe:<15}  true={true_sev:<10}  predicted={most_pred:<10}  "
          f"error={1-sub['correct'].mean():.1%}")

# ── plots ─────────────────────────────────────────────────────────────────────

# worst-N bar chart
fig, ax = plt.subplots(figsize=(9, 6))
wp = cwe_filtered.head(TOP_N).copy()
ax.barh(wp["cwe"][::-1], wp["accuracy"][::-1], color="#58A6FF", edgecolor="none")
ax.axvline(acc, color="#DA3633", linestyle="--", linewidth=1.5,
           label=f"Overall accuracy ({acc:.3f})")
for i, (_, row) in enumerate(wp[::-1].iterrows()):
    ax.text(row["accuracy"] + 0.005, i, f"{row['accuracy']:.3f}",
            va="center", fontsize=9)
ax.set_title(f"Worst {TOP_N} CWEs by test accuracy", fontsize=12)
ax.set_xlabel("Accuracy")
ax.set_xlim(0, 1.05)
ax.legend(fontsize=9)
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "cwe_worst_accuracy.png", dpi=150)
plt.close()
print(f"\n  Saved cwe_worst_accuracy.png")

# accuracy distribution
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(cwe_filtered["accuracy"], bins=25, color="#58A6FF", edgecolor="none", alpha=0.85)
ax.axvline(acc, color="#DA3633", linestyle="--", linewidth=1.5,
           label=f"Overall accuracy ({acc:.3f})")
ax.set_title("Distribution of per-CWE accuracy (test set)", fontsize=12)
ax.set_xlabel("Accuracy")
ax.set_ylabel("Number of CWEs")
ax.legend(fontsize=9)
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "cwe_accuracy_distribution.png", dpi=150)
plt.close()
print(f"  Saved cwe_accuracy_distribution.png")

# confusion heatmaps for worst 5
fig, axes = plt.subplots(1, 5, figsize=(18, 4))
for ax, cwe in zip(axes, worst_cwes[:5]):
    sub = test_df[test_df["cwe"] == cwe]
    cm  = pd.crosstab(sub["severity"], sub["pred_severity"])
    cm  = cm.reindex(index=LABEL_ORDER, columns=LABEL_ORDER, fill_value=0)
    sns.heatmap(cm.div(cm.sum(axis=1), axis=0).fillna(0),
                annot=True, fmt=".2f", cmap="Blues", ax=ax,
                linewidths=0.5, annot_kws={"size":9}, cbar=False)
    ax.set_title(f"{cwe}\n(n={len(sub):,})", fontsize=9)
    ax.set_xlabel("Pred", fontsize=8)
    ax.set_ylabel("True", fontsize=8)
    ax.tick_params(labelsize=7)
plt.suptitle("Confusion matrices — 5 worst CWE categories", fontsize=11)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "cwe_confusion_worst5.png", dpi=150)
plt.close()
print(f"  Saved cwe_confusion_worst5.png")

# ── save ──────────────────────────────────────────────────────────────────────

def to_python(v):
    if isinstance(v, (np.floating, float)): return round(float(v), 4)
    if isinstance(v, (np.integer, int)):    return int(v)
    return v

cwe_results = {
    "overall": {"accuracy": round(acc,4), "f1_weighted": round(f1_weighted,4),
                "f1_macro": round(f1_macro,4)},
    "min_samples_threshold": MIN_SAMPLES_PER_CWE,
    "n_cwe_analysed":        int(len(cwe_filtered)),
    "worst_cwes":            worst_cwes,
    "cwe_stats": [{k: to_python(v) for k,v in row.items()}
                  for row in cwe_filtered.to_dict(orient="records")],
}
with open(RESULTS_DIR / "cwe_analysis.json", "w") as f:
    json.dump(cwe_results, f, indent=2)
cwe_filtered.to_csv(RESULTS_DIR / "cwe_accuracy.csv", index=False)
print(f"  Saved cwe_analysis.json + cwe_accuracy.csv")
print("\nCWE analysis complete.")