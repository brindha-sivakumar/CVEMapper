# src/training/hparam_search.py
#
# Week 5 — Three-part hyperparameter experiment:
#   Part A: 6-run grid search (lr x max_len)
#   Part B: bert-base-uncased ablation (domain pre-training benefit)
#   Part C: SecBERT regression head (raw CVSS score prediction)
#
# Fixes applied vs original:
#   1. evaluation_strategy renamed to eval_strategy in newer transformers —
#      detected at runtime using inspect so the same file runs on any version
#   2. All numpy int64/float64 values cast to Python int/float before json.dump
#   3. Typo fj_score -> f1_score in regression evaluation
#   4. Severity normalisation (.str.strip().str.upper()) before label mapping
#   5. Duplicate AutoModelForSequenceClassification import removed
#   6. regression_results.json now saved alongside full_comparison.json

import os
import json
import inspect
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product
import time

import torch
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ── paths ─────────────────────────────────────────────────────────────────────

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

MODEL_NAME  = "jackaduma/SecBERT"
SPLIT_DIR   = PROJECT_ROOT / "data"        / "processed"
CKPT_DIR    = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)

# ── constants ─────────────────────────────────────────────────────────────────

LABEL_ORDER = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
LABEL2ID    = {l: i for i, l in enumerate(LABEL_ORDER)}
ID2LABEL    = {i: l for i, l in enumerate(LABEL_ORDER)}
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device : {DEVICE}")
if DEVICE == "cuda":
    print(f"  GPU  : {torch.cuda.get_device_name(0)}")

# ── detect TrainingArguments API version ──────────────────────────────────────
# evaluation_strategy was renamed to eval_strategy in transformers >= 4.46
# detect at runtime so this file runs on any version without modification

_ta_params  = inspect.signature(TrainingArguments.__init__).parameters
_EVAL_KEY   = "eval_strategy" if "eval_strategy" in _ta_params else "evaluation_strategy"
print(f"TrainingArguments eval key: {_EVAL_KEY}")


# ── load splits ───────────────────────────────────────────────────────────────

print("\nLoading splits...")
train_df = pd.read_csv(SPLIT_DIR / "train.csv")
val_df   = pd.read_csv(SPLIT_DIR / "val.csv")
test_df  = pd.read_csv(SPLIT_DIR / "test.csv")

# normalise severity — strip whitespace and uppercase before mapping
for df in [train_df, val_df, test_df]:
    df["severity"]   = df["severity"].astype(str).str.strip().str.upper()
    df["label"]      = df["severity"].map(LABEL2ID)
    df["score_norm"] = df["score"] / 10.0

# assert no NaN labels
for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
    bad = df[df["label"].isna()]
    if len(bad) > 0:
        print(f"  WARNING: {len(bad):,} rows in {name} have unmapped severity — dropping.")
        df.drop(index=bad.index, inplace=True)
        df.reset_index(drop=True, inplace=True)

print(f"  Train : {len(train_df):,}")
print(f"  Val   : {len(val_df):,}")
print(f"  Test  : {len(test_df):,}")


# ── class weights ─────────────────────────────────────────────────────────────

counts        = train_df["label"].value_counts().sort_index()
weights       = len(train_df) / (len(LABEL_ORDER) * counts)
class_weights = torch.tensor(weights.values, dtype=torch.float32).to(DEVICE)

print("\nClass weights:")
for i, (label, w) in enumerate(zip(LABEL_ORDER, class_weights)):
    print(f"  {label:<10} count={counts[i]:>7,}   weight={w:.4f}")


# ── dataset builder ───────────────────────────────────────────────────────────

def build_dataset(df: pd.DataFrame, tokenizer, max_len: int,
                  regression: bool = False) -> Dataset:
    """Build a HuggingFace Dataset from a DataFrame.

    When regression=False, label column contains integer class indices.
    When regression=True, label column contains normalised CVSS score in [0,1].
    """
    data = {
        "description": df["description"].tolist(),
        "label":       df["score_norm"].tolist() if regression
                       else df["label"].tolist()
    }
    ds = Dataset.from_dict(data)

    def tokenize(batch):
        return tokenizer(
            batch["description"],
            truncation=True,
            padding="max_length",
            max_length=max_len
        )

    return ds.map(tokenize, batched=True, batch_size=512,
                  remove_columns=["description"])


# ── custom trainers ───────────────────────────────────────────────────────────

class WeightedTrainer(Trainer):
    """Trainer with inverse-frequency class weights in CrossEntropyLoss."""
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs.pop("labels")
        outputs = model(**inputs)
        logits  = outputs.logits
        loss    = CrossEntropyLoss(weight=class_weights)(logits, labels.long())
        return (loss, outputs) if return_outputs else loss


class RegressionTrainer(Trainer):
    """Trainer using MSELoss for continuous CVSS score prediction."""
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs.pop("labels").float()
        outputs = model(**inputs)
        logits  = outputs.logits.squeeze(-1)
        loss    = MSELoss()(logits, labels)
        return (loss, outputs) if return_outputs else loss


# ── metric functions ──────────────────────────────────────────────────────────

def clf_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy":    round(float(accuracy_score(labels, preds)), 4),
        "f1_weighted": round(float(f1_score(labels, preds, average="weighted",
                                            labels=list(range(4)))), 4),
        "f1_macro":    round(float(f1_score(labels, preds, average="macro",
                                            labels=list(range(4)))), 4),
    }


def reg_metrics(eval_pred):
    preds, labels = eval_pred
    preds  = preds.squeeze() * 10.0
    labels = labels           * 10.0
    mae    = float(np.mean(np.abs(preds - labels)))
    mse    = float(np.mean((preds - labels) ** 2))

    def bucket(scores):
        out = []
        for s in scores:
            if   s < 4.0: out.append(0)
            elif s < 7.0: out.append(1)
            elif s < 9.0: out.append(2)
            else:         out.append(3)
        return np.array(out)

    f1_ord = float(f1_score(bucket(labels), bucket(preds),
                            average="weighted", labels=list(range(4)),
                            zero_division=0))
    return {
        "mae":    round(mae,    4),
        "mse":    round(mse,    4),
        "f1_ord": round(f1_ord, 4),
    }


# ── TrainingArguments factory ─────────────────────────────────────────────────

def make_args(output_dir, lr, epochs, metric, greater_is_better,
              warmup=200, logging=200, save_limit=1):
    """Build TrainingArguments using the correct eval key for this transformers version."""
    return TrainingArguments(
        output_dir                  = str(output_dir),
        num_train_epochs            = epochs,
        per_device_train_batch_size = 16,
        per_device_eval_batch_size  = 32,
        learning_rate               = lr,
        warmup_steps                = warmup,
        weight_decay                = 0.01,
        **{_EVAL_KEY: "epoch"},
        save_strategy               = "epoch",
        load_best_model_at_end      = True,
        metric_for_best_model       = metric,
        greater_is_better           = greater_is_better,
        logging_steps               = logging,
        fp16                        = (DEVICE == "cuda"),
        dataloader_num_workers      = 0,
        report_to                   = "none",
        save_total_limit            = save_limit,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PART A — HYPERPARAMETER GRID SEARCH
# ══════════════════════════════════════════════════════════════════════════════

LR_VALUES      = [1e-5, 2e-5, 5e-5]
MAX_LEN_VALUES = [128, 256]
GRID           = list(product(LR_VALUES, MAX_LEN_VALUES))

print(f"\n{'='*60}")
print(f"  PART A — Hyperparameter grid search")
print(f"  {len(GRID)} runs: lr x max_len")
print(f"{'='*60}")

tokenizer_cache = {}   # reuse tokenized datasets per max_len
grid_results    = []

for run_idx, (lr, max_len) in enumerate(GRID):
    run_name = f"lr{lr}_len{max_len}"
    print(f"\n[{run_idx+1}/{len(GRID)}] lr={lr}  max_len={max_len}")

    if run_idx == 0:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if max_len not in tokenizer_cache:
        print(f"  Tokenizing at max_len={max_len}...")
        tokenizer_cache[max_len] = {
            "train": build_dataset(train_df, tokenizer, max_len),
            "val":   build_dataset(val_df,   tokenizer, max_len),
        }
    train_ds = tokenizer_cache[max_len]["train"]
    val_ds   = tokenizer_cache[max_len]["val"]

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=4, id2label=ID2LABEL,
        label2id=LABEL2ID, ignore_mismatched_sizes=True
    )

    t0      = time.time()
    trainer = WeightedTrainer(
        model           = model,
        args            = make_args(CKPT_DIR / f"hparam_{run_name}", lr,
                                    epochs=3, metric="f1_weighted",
                                    greater_is_better=True),
        train_dataset   = train_ds,
        eval_dataset    = val_ds,
        compute_metrics = clf_metrics,
        callbacks       = [EarlyStoppingCallback(early_stopping_patience=1)]
    )
    trainer.train()
    elapsed = time.time() - t0

    best = max(
        [e for e in trainer.state.log_history if "eval_f1_weighted" in e],
        key=lambda e: e["eval_f1_weighted"]
    )
    result = {
        "run":         run_name,
        "lr":          float(lr),
        "max_len":     int(max_len),
        "f1_weighted": round(float(best["eval_f1_weighted"]), 4),
        "f1_macro":    round(float(best["eval_f1_macro"]),    4),
        "accuracy":    round(float(best["eval_accuracy"]),    4),
        "epoch":       float(best["epoch"]),
        "time_min":    round(elapsed / 60, 1),
    }
    grid_results.append(result)
    print(f"  f1_weighted={result['f1_weighted']}  "
          f"f1_macro={result['f1_macro']}  "
          f"epoch={result['epoch']}  time={result['time_min']}min")

# summary
print(f"\n{'='*60}")
print("  GRID SEARCH RESULTS")
print(f"{'='*60}")
grid_df  = pd.DataFrame(grid_results).sort_values("f1_weighted", ascending=False)
print(grid_df.to_string(index=False))

best_run     = grid_df.iloc[0]
best_lr      = float(best_run["lr"])
best_max_len = int(best_run["max_len"])
print(f"\n  Best config:  lr={best_lr}  max_len={best_max_len}")
print(f"  Best val F1-weighted : {best_run['f1_weighted']}")

with open(RESULTS_DIR / "hparam_results.json", "w") as f:
    json.dump(grid_results, f, indent=2)
print(f"  Saved hparam_results.json")

# heatmap
pivot = grid_df.pivot(index="lr", columns="max_len", values="f1_weighted")
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(pivot, annot=True, fmt=".4f", cmap="YlGn",
            ax=ax, linewidths=0.5, annot_kws={"size": 12})
ax.set_title("Val F1-weighted — lr x max_len", fontsize=12)
ax.set_xlabel("max_len")
ax.set_ylabel("learning rate")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "hparam_heatmap.png", dpi=150)
plt.close()
print("  Saved hparam_heatmap.png")


# ══════════════════════════════════════════════════════════════════════════════
# PART B — BERT-BASE ABLATION
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print("  PART B — bert-base-uncased ablation")
print(f"{'='*60}")

ABLATION_MODEL = "bert-base-uncased"

abl_tokenizer = AutoTokenizer.from_pretrained(ABLATION_MODEL)
abl_train_ds  = build_dataset(train_df, abl_tokenizer, best_max_len)
abl_val_ds    = build_dataset(val_df,   abl_tokenizer, best_max_len)

abl_model = AutoModelForSequenceClassification.from_pretrained(
    ABLATION_MODEL, num_labels=4, id2label=ID2LABEL,
    label2id=LABEL2ID, ignore_mismatched_sizes=True
)

abl_trainer = WeightedTrainer(
    model           = abl_model,
    args            = make_args(CKPT_DIR / "ablation_bert_base", best_lr,
                                epochs=3, metric="f1_weighted",
                                greater_is_better=True),
    train_dataset   = abl_train_ds,
    eval_dataset    = abl_val_ds,
    compute_metrics = clf_metrics,
    callbacks       = [EarlyStoppingCallback(early_stopping_patience=1)]
)
abl_trainer.train()

abl_best = max(
    [e for e in abl_trainer.state.log_history if "eval_f1_weighted" in e],
    key=lambda e: e["eval_f1_weighted"]
)

abl_f1w = round(float(abl_best["eval_f1_weighted"]), 4)
abl_f1m = round(float(abl_best["eval_f1_macro"]),    4)
abl_acc = round(float(abl_best["eval_accuracy"]),    4)

print(f"\n  bert-base-uncased  f1_weighted={abl_f1w:.4f}  f1_macro={abl_f1m:.4f}")
print(f"  SecBERT (best)     f1_weighted={best_run['f1_weighted']:.4f}  "
      f"f1_macro={best_run['f1_macro']:.4f}")
print(f"  Domain pre-training gain: "
      f"{best_run['f1_weighted'] - abl_f1w:+.4f} F1-weighted")

# all values explicitly cast to Python float/int — no numpy types
ablation_results = {
    "bert_base": {
        "f1_weighted": abl_f1w,
        "f1_macro":    abl_f1m,
        "accuracy":    abl_acc,
    },
    "secbert_best": {
        "lr":          best_lr,
        "max_len":     best_max_len,
        "f1_weighted": float(best_run["f1_weighted"]),
        "f1_macro":    float(best_run["f1_macro"]),
        "accuracy":    float(best_run["accuracy"]),
    },
    "domain_gain_f1_weighted": round(float(best_run["f1_weighted"]) - abl_f1w, 4)
}
with open(RESULTS_DIR / "ablation_results.json", "w") as f:
    json.dump(ablation_results, f, indent=2)
print("  Saved ablation_results.json")


# ══════════════════════════════════════════════════════════════════════════════
# PART C — REGRESSION HEAD
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print("  PART C — SecBERT regression head (raw CVSS score)")
print(f"{'='*60}")

reg_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
reg_train_ds  = build_dataset(train_df, reg_tokenizer, best_max_len, regression=True)
reg_val_ds    = build_dataset(val_df,   reg_tokenizer, best_max_len, regression=True)
reg_test_ds   = build_dataset(test_df,  reg_tokenizer, best_max_len, regression=True)

reg_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=1, ignore_mismatched_sizes=True
)

reg_trainer = RegressionTrainer(
    model           = reg_model,
    args            = make_args(CKPT_DIR / "secbert_regression", best_lr,
                                epochs=5, metric="mae", greater_is_better=False,
                                warmup=500, logging=100, save_limit=2),
    train_dataset   = reg_train_ds,
    eval_dataset    = reg_val_ds,
    compute_metrics = reg_metrics,
    callbacks       = [EarlyStoppingCallback(early_stopping_patience=2)]
)
reg_trainer.train()


def bucket_labels(scores):
    return ["LOW" if s < 4 else "MEDIUM" if s < 7 else "HIGH" if s < 9
            else "CRITICAL" for s in scores]


regression_eval = {}

for split_name, ds, df_ref in [
    ("val",  reg_val_ds,  val_df),
    ("test", reg_test_ds, test_df)
]:
    pred_out    = reg_trainer.predict(ds)
    preds_raw   = pred_out.predictions.squeeze() * 10.0
    labels_raw  = pred_out.label_ids             * 10.0

    mae   = float(np.mean(np.abs(preds_raw - labels_raw)))
    mse   = float(np.mean((preds_raw - labels_raw) ** 2))
    f1_ord = float(f1_score(
        bucket_labels(labels_raw), bucket_labels(preds_raw),
        average="weighted", labels=LABEL_ORDER, zero_division=0
    ))

    print(f"\n  [REGRESSION — {split_name.upper()}]")
    print(f"  MAE (raw score)        : {mae:.4f}")
    print(f"  MSE (raw score)        : {mse:.4f}")
    print(f"  F1-weighted (bucketed) : {f1_ord:.4f}")

    regression_eval[split_name] = {
        "mae":    round(mae,    4),
        "mse":    round(mse,    4),
        "f1_ord": round(f1_ord, 4),
    }

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(labels_raw[:2000], preds_raw[:2000],
               alpha=0.2, s=6, color="#58A6FF")
    ax.plot([0, 10], [0, 10], "r--", linewidth=1.5, label="Perfect prediction")
    for boundary in [4.0, 7.0, 9.0]:
        ax.axvline(boundary, color="gray", linewidth=0.8, linestyle=":")
        ax.axhline(boundary, color="gray", linewidth=0.8, linestyle=":")
    ax.set_title(f"Regression — predicted vs true CVSS score ({split_name})",
                 fontsize=11)
    ax.set_xlabel("True score")
    ax.set_ylabel("Predicted score")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"regression_scatter_{split_name}.png", dpi=150)
    plt.close()
    print(f"  Saved regression_scatter_{split_name}.png")

with open(RESULTS_DIR / "regression_results.json", "w") as f:
    json.dump(regression_eval, f, indent=2)
print("  Saved regression_results.json")


# ══════════════════════════════════════════════════════════════════════════════
# FINAL COMPARISON TABLE
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print("  FULL MODEL COMPARISON — VAL SET")
print(f"{'='*60}")

baseline_path  = RESULTS_DIR / "baseline_results.json"
first_run_path = RESULTS_DIR / "secbert_first_run.json"

rows = []

if baseline_path.exists():
    with open(baseline_path) as f:
        baselines = json.load(f)
    for name, res in baselines.items():
        rows.append({
            "Model":       name,
            "F1-Weighted": float(res["val"]["f1_weighted"]),
            "F1-Macro":    float(res["val"]["f1_macro"]),
            "Accuracy":    float(res["val"]["accuracy"]),
            "MAE":         "—"
        })

if first_run_path.exists():
    with open(first_run_path) as f:
        first = json.load(f)
    rows.append({
        "Model":       "SecBERT first run (lr=2e-5, len=256)",
        "F1-Weighted": float(first["val"]["f1_weighted"]),
        "F1-Macro":    float(first["val"]["f1_macro"]),
        "Accuracy":    float(first["val"]["accuracy"]),
        "MAE":         "—"
    })

rows.append({
    "Model":       f"SecBERT best (lr={best_lr}, len={best_max_len})",
    "F1-Weighted": float(best_run["f1_weighted"]),
    "F1-Macro":    float(best_run["f1_macro"]),
    "Accuracy":    float(best_run["accuracy"]),
    "MAE":         "—"
})
rows.append({
    "Model":       "bert-base-uncased (ablation)",
    "F1-Weighted": ablation_results["bert_base"]["f1_weighted"],
    "F1-Macro":    ablation_results["bert_base"]["f1_macro"],
    "Accuracy":    ablation_results["bert_base"]["accuracy"],
    "MAE":         "—"
})
rows.append({
    "Model":       "SecBERT regression (bucketed, val)",
    "F1-Weighted": regression_eval["val"]["f1_ord"],
    "F1-Macro":    "—",
    "Accuracy":    "—",
    "MAE":         regression_eval["val"]["mae"],
})

comparison_df = pd.DataFrame(rows)
print(comparison_df.to_string(index=False))

with open(RESULTS_DIR / "full_comparison.json", "w") as f:
    json.dump(rows, f, indent=2)
print(f"\n  Saved full_comparison.json")
print("\nWeek 5 complete.")