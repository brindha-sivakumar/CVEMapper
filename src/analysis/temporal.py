# src/analysis/temporal.py
#
# Phase 3 — Temporal Generalization Test
# Compares model performance on val (2015-2021 holdout) vs test (2022-2024
# blind set). Also breaks down test performance year by year.
#
# Run after: train.py has saved checkpoints/secbert_best/
# Outputs:   results/temporal_results.json
#            results/temporal_val_vs_test.png
#            results/temporal_by_year.png
#            results/temporal_f1_drift.png

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

LABEL_ORDER = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
LABEL2ID    = {l: i for i, l in enumerate(LABEL_ORDER)}
ID2LABEL    = {i: l for i, l in enumerate(LABEL_ORDER)}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device : {DEVICE}")

# ── load model ────────────────────────────────────────────────────────────────

print(f"\nLoading checkpoint...")
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

# ── helpers ───────────────────────────────────────────────────────────────────

def run_inference(df: pd.DataFrame, label: str):
    """Tokenize df and run inference. Returns (preds, labels) as numpy arrays."""
    df = df.copy()
    df["severity"] = df["severity"].astype(str).str.strip().str.upper()
    df["label"]    = df["severity"].map(LABEL2ID)
    df = df.dropna(subset=["label"]).reset_index(drop=True)
    df["label"] = df["label"].astype(int)

    print(f"\n  Tokenizing {label} ({len(df):,} rows)...")
    ds = Dataset.from_dict({
        "description": df["description"].tolist(),
        "label":       df["label"].tolist()
    })
    ds = ds.map(
        lambda b: tokenizer(b["description"], truncation=True,
                            padding="max_length", max_length=256),
        batched=True, batch_size=512, remove_columns=["description"]
    )
    ds = ds.cast_column("label", Value("int64"))

    all_logits, all_labels = [], []
    dl = torch.utils.data.DataLoader(
        ds.with_format("torch"), batch_size=64, shuffle=False
    )
    with torch.no_grad():
        for i, batch in enumerate(dl):
            out = model(input_ids      = batch["input_ids"].to(DEVICE),
                        attention_mask = batch["attention_mask"].to(DEVICE))
            all_logits.append(out.logits.cpu().numpy())
            all_labels.append(batch["label"].numpy())
            if i % 100 == 0:
                print(f"    Batch {i+1}/{len(dl)}", end="\r")

    logits = np.concatenate(all_logits)
    labels = np.concatenate(all_labels)
    preds  = np.argmax(logits, axis=-1)
    print(f"    Done ({len(df):,} samples).     ")
    return preds, labels


def compute_metrics(preds, labels) -> dict:
    f1_pc = f1_score(labels, preds, average=None,
                     labels=list(range(4)), zero_division=0)
    return {
        "accuracy":    round(float(accuracy_score(labels, preds)), 4),
        "f1_weighted": round(float(f1_score(labels, preds, average="weighted",
                                            labels=list(range(4)))), 4),
        "f1_macro":    round(float(f1_score(labels, preds, average="macro",
                                            labels=list(range(4)))), 4),
        "f1_per_class": {
            lbl: round(float(s), 4)
            for lbl, s in zip(LABEL_ORDER, f1_pc)
        }
    }

# ── load splits ───────────────────────────────────────────────────────────────

print("\nLoading splits...")
val_df  = pd.read_csv(SPLIT_DIR / "val.csv")
test_df = pd.read_csv(SPLIT_DIR / "test.csv")
test_df["published"] = pd.to_datetime(test_df["published"])
test_df["year"]      = test_df["published"].dt.year

print(f"  Val  : {len(val_df):,} rows  (2015-2021 holdout)")
print(f"  Test : {len(test_df):,} rows  (2022-2024 blind set)")
print(f"\n  Test year breakdown:")
print(test_df["year"].value_counts().sort_index().to_string())

# ── val vs test comparison ────────────────────────────────────────────────────

val_preds,  val_labels  = run_inference(val_df,  "validation set")
test_preds, test_labels = run_inference(test_df, "test set")

val_metrics  = compute_metrics(val_preds,  val_labels)
test_metrics = compute_metrics(test_preds, test_labels)

print(f"\n{'='*60}")
print("  TEMPORAL GENERALIZATION — VAL vs TEST")
print(f"{'='*60}")
print(f"  {'Metric':<20} {'Val (2015-21)':>15} {'Test (2022-24)':>15} {'Delta':>10}")
print(f"  {'-'*62}")
for metric in ["accuracy", "f1_weighted", "f1_macro"]:
    v     = val_metrics[metric]
    t     = test_metrics[metric]
    delta = t - v
    flag  = "  <-- DRIFT" if abs(delta) > 0.05 else ""
    print(f"  {metric:<20} {v:>15.4f} {t:>15.4f} {delta:>+10.4f}{flag}")

print(f"\n  Per-class F1 drift (test - val):")
print(f"  {'Class':<12} {'Val':>8} {'Test':>8} {'Delta':>8}")
print(f"  {'-'*38}")
for lbl in LABEL_ORDER:
    v = val_metrics["f1_per_class"][lbl]
    t = test_metrics["f1_per_class"][lbl]
    d = t - v
    flag = "  <-- notable" if abs(d) > 0.05 else ""
    print(f"  {lbl:<12} {v:>8.4f} {t:>8.4f} {d:>+8.4f}{flag}")

drift_f1 = test_metrics["f1_weighted"] - val_metrics["f1_weighted"]
if abs(drift_f1) < 0.02:
    print("\n  Interpretation: Minimal temporal drift (<2pp). Model generalises well.")
elif abs(drift_f1) < 0.05:
    print(f"\n  Interpretation: Moderate drift ({drift_f1:+.2%}). Some shift in CVE language.")
else:
    print(f"\n  Interpretation: Significant drift ({drift_f1:+.2%}). Publishable finding.")

# ── year-by-year breakdown ────────────────────────────────────────────────────

print(f"\n{'='*60}")
print("  YEAR-BY-YEAR PERFORMANCE")
print(f"{'='*60}")

year_results = []
for year in sorted(test_df["year"].unique()):
    ydf   = test_df[test_df["year"] == year].copy()
    yp, yl = run_inference(ydf, f"year {year}")
    m      = compute_metrics(yp, yl)
    m["year"]  = int(year)
    m["count"] = int(len(ydf))
    year_results.append(m)
    print(f"  {year}  n={len(ydf):>6,}  "
          f"acc={m['accuracy']:.4f}  f1_w={m['f1_weighted']:.4f}  "
          f"f1_m={m['f1_macro']:.4f}")

# ── plots ─────────────────────────────────────────────────────────────────────

# val vs test bar comparison
metrics_to_plot = ["accuracy", "f1_weighted", "f1_macro"]
metric_labels   = ["Accuracy", "F1-Weighted", "F1-Macro"]

fig, ax = plt.subplots(figsize=(8, 4))
x     = np.arange(len(metrics_to_plot))
width = 0.3
vv = [val_metrics[m]  for m in metrics_to_plot]
tv = [test_metrics[m] for m in metrics_to_plot]

b1 = ax.bar(x - width/2, vv, width, label="Val (2015-21)",
            color="#58A6FF", edgecolor="none")
b2 = ax.bar(x + width/2, tv, width, label="Test (2022-24)",
            color="#DA3633", edgecolor="none", alpha=0.85)
for bar in list(b1) + list(b2):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(metric_labels)
ax.set_ylim(0, 1.05)
ax.set_title("Temporal generalisation — val vs test", fontsize=12)
ax.set_ylabel("Score")
ax.legend(fontsize=9)
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "temporal_val_vs_test.png", dpi=150)
plt.close()
print(f"\n  Saved temporal_val_vs_test.png")

# year-by-year line chart
yr_df = pd.DataFrame(year_results)
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(yr_df["year"], yr_df["f1_weighted"], color="#58A6FF",
        linewidth=2, marker="o", label="F1-Weighted")
ax.plot(yr_df["year"], yr_df["f1_macro"],    color="#3FB950",
        linewidth=2, marker="s", linestyle="--", label="F1-Macro")
ax.plot(yr_df["year"], yr_df["accuracy"],    color="#E3B341",
        linewidth=2, marker="^", linestyle=":", label="Accuracy")
ax.axhline(val_metrics["f1_weighted"], color="#58A6FF", linestyle="--",
           linewidth=1, alpha=0.5, label="Val F1-W baseline")
ax.set_title("Model performance by publication year (test set)", fontsize=12)
ax.set_xlabel("Year")
ax.set_ylabel("Score")
ax.legend(fontsize=9)
ax.set_ylim(0.5, 1.0)
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "temporal_by_year.png", dpi=150)
plt.close()
print(f"  Saved temporal_by_year.png")

# per-class F1 drift bar chart
deltas = [test_metrics["f1_per_class"][l] - val_metrics["f1_per_class"][l]
          for l in LABEL_ORDER]
colors = ["#3FB950" if d >= 0 else "#DA3633" for d in deltas]
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(LABEL_ORDER, deltas, color=colors, edgecolor="none", width=0.5)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_title("Per-class F1 drift (test - val)", fontsize=12)
ax.set_ylabel("Delta F1")
for i, (lbl, d) in enumerate(zip(LABEL_ORDER, deltas)):
    ax.text(i, d + (0.003 if d >= 0 else -0.01),
            f"{d:+.3f}", ha="center", va="bottom", fontsize=10)
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "temporal_f1_drift.png", dpi=150)
plt.close()
print(f"  Saved temporal_f1_drift.png")

# ── save ──────────────────────────────────────────────────────────────────────

temporal_results = {
    "val":  val_metrics,
    "test": test_metrics,
    "drift": {
        m: round(test_metrics[m] - val_metrics[m], 4)
        for m in ["accuracy", "f1_weighted", "f1_macro"]
    },
    "per_class_drift": {
        lbl: round(test_metrics["f1_per_class"][lbl] -
                   val_metrics["f1_per_class"][lbl], 4)
        for lbl in LABEL_ORDER
    },
    "year_by_year": year_results,
}
with open(RESULTS_DIR / "temporal_results.json", "w") as f:
    json.dump(temporal_results, f, indent=2)
print(f"\n  Saved temporal_results.json")
print("\nTemporal analysis complete.")