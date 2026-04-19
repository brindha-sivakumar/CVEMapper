# src/training/train.py
#
# SecBERT fine-tuning for CVSS v3 severity classification.
# Fixes applied vs original:
#   1. evaluation_strategy renamed to eval_strategy in transformers >= 4.46
#      — detected at runtime using inspect
#   2. Severity column normalised (.str.strip().str.upper()) before LABEL2ID
#      mapping — prevents NaN label assertion errors
#   3. full_evaluate uses plain PyTorch DataLoader, NOT trainer.predict()
#      — avoids AcceleratorError / Float label CUDA errors on large test set
#   4. Skip logic at every stage — safe to re-run in a new Colab session
#   5. Results written incrementally — val saved before test begins
#   6. Consistency check warns if checkpoint is missing but results exist
#   7. Selective tokenisation — only tokenises splits that are actually needed

import os
import inspect
import json
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

import torch
from torch.nn import CrossEntropyLoss
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from datasets import Dataset, Value
from sklearn.metrics import (
    accuracy_score, f1_score,
    classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

load_dotenv()

# ── paths ─────────────────────────────────────────────────────────────────────

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

MODEL_NAME   = "jackaduma/SecBERT"
SPLIT_DIR    = PROJECT_ROOT / "data"        / "processed"
CKPT_DIR     = PROJECT_ROOT / "checkpoints" / "secbert_best"
RESULTS_DIR  = PROJECT_ROOT / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)

# ── constants ─────────────────────────────────────────────────────────────────

LABEL_ORDER = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
LABEL2ID    = {l: i for i, l in enumerate(LABEL_ORDER)}
ID2LABEL    = {i: l for i, l in enumerate(LABEL_ORDER)}

MAX_LEN    = 256
BATCH_SIZE = 16
EPOCHS     = 5
LR         = 2e-5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device : {DEVICE}")
if DEVICE == "cuda":
    print(f"  GPU   : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM  : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ── detect correct TrainingArguments eval key ─────────────────────────────────
# evaluation_strategy was renamed to eval_strategy in transformers >= 4.46

_ta_params = inspect.signature(TrainingArguments.__init__).parameters
_EVAL_KEY  = "eval_strategy" if "eval_strategy" in _ta_params else "evaluation_strategy"
print(f"TrainingArguments eval key : {_EVAL_KEY}")

# ── skip flags ────────────────────────────────────────────────────────────────
# Each stage checks for its output file before running.
# In a new Colab session after a completed run, everything is skipped
# and the script finishes in under 2 minutes.
#
# To force re-running any stage, delete the corresponding output file:
#   Training    → delete checkpoints/secbert_best/config.json
#   Curves      → delete results/secbert_training_curves.png
#   Val eval    → delete "val" key from results/secbert_first_run.json
#   Test eval   → delete "test" key from results/secbert_first_run.json

CHECKPOINT_EXISTS = (CKPT_DIR / "config.json").exists()
CURVES_EXIST      = (RESULTS_DIR / "secbert_training_curves.png").exists()
FIRST_RUN_EXISTS  = (RESULTS_DIR / "secbert_first_run.json").exists()

_val_done  = False
_test_done = False
if FIRST_RUN_EXISTS:
    with open(RESULTS_DIR / "secbert_first_run.json") as _f:
        _existing = json.load(_f)
    _val_done  = "val"  in _existing
    _test_done = "test" in _existing

SKIP_TRAINING  = CHECKPOINT_EXISTS
SKIP_CURVES    = CURVES_EXIST
SKIP_VAL_EVAL  = FIRST_RUN_EXISTS and _val_done
SKIP_TEST_EVAL = FIRST_RUN_EXISTS and _test_done

print(f"\nSkip flags:")
print(f"  Training  : {'SKIP' if SKIP_TRAINING  else 'RUN'}"
      f"  ({'checkpoint exists'  if SKIP_TRAINING  else 'no checkpoint'})")
print(f"  Curves    : {'SKIP' if SKIP_CURVES    else 'RUN'}"
      f"  ({'PNG exists'         if SKIP_CURVES    else 'will plot after training'})")
print(f"  Val eval  : {'SKIP' if SKIP_VAL_EVAL  else 'RUN'}"
      f"  ({'results cached'     if SKIP_VAL_EVAL  else 'will evaluate'})")
print(f"  Test eval : {'SKIP' if SKIP_TEST_EVAL else 'RUN'}"
      f"  ({'results cached'     if SKIP_TEST_EVAL else 'will evaluate'})")

# ── consistency check ─────────────────────────────────────────────────────────
# Warn if results exist but checkpoint is missing — the cached results
# came from a model that no longer exists on disk.

if not CHECKPOINT_EXISTS and FIRST_RUN_EXISTS:
    print("\n  WARNING: Results JSON exists but checkpoint is missing.")
    print("  Cached results came from a model that no longer exists on disk.")
    print("  Retraining will produce a new model — cached results will be overwritten.")
    SKIP_VAL_EVAL  = False
    SKIP_TEST_EVAL = False

# ── load splits ───────────────────────────────────────────────────────────────

print("\nLoading splits...")
train_df = pd.read_csv(SPLIT_DIR / "train.csv")
val_df   = pd.read_csv(SPLIT_DIR / "val.csv")
test_df  = pd.read_csv(SPLIT_DIR / "test.csv")

print(f"  Train : {len(train_df):,}")
print(f"  Val   : {len(val_df):,}")
print(f"  Test  : {len(test_df):,}")

# normalise severity — strip whitespace and uppercase before mapping
# prevents NaN labels from inconsistent capitalisation in the CSV
for df in [train_df, val_df, test_df]:
    df["severity"] = df["severity"].astype(str).str.strip().str.upper()
    df["label"]    = df["severity"].map(LABEL2ID)

# drop unmapped rows with a warning rather than hard-asserting
for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
    bad = df[df["label"].isna()]
    if len(bad) > 0:
        print(f"  WARNING: {len(bad):,} rows in {name} have unmapped severity — dropping.")
        df.drop(index=bad.index, inplace=True)
        df.reset_index(drop=True, inplace=True)

assert train_df["label"].notna().all(), "NaN labels remain in train after cleaning"
assert val_df["label"].notna().all(),   "NaN labels remain in val after cleaning"
assert test_df["label"].notna().all(),  "NaN labels remain in test after cleaning"

print(f"\n  After cleaning:")
print(f"  Train : {len(train_df):,}")
print(f"  Val   : {len(val_df):,}")
print(f"  Test  : {len(test_df):,}")

# ── class weights ─────────────────────────────────────────────────────────────

print("\nComputing class weights...")
counts        = train_df["label"].value_counts().sort_index()
weights       = len(train_df) / (len(LABEL_ORDER) * counts)
class_weights = torch.tensor(weights.values, dtype=torch.float32).to(DEVICE)

print("  Class weights (inverse-frequency):")
for i, (label, w) in enumerate(zip(LABEL_ORDER, class_weights)):
    print(f"    {label:<10} count={counts[i]:>7,}   weight={w:.4f}")

# ── tokenizer and datasets ────────────────────────────────────────────────────
# Load tokenizer from checkpoint if it exists (ensures vocab consistency).
# Only tokenise splits that are actually needed this session.

print(f"\nLoading tokenizer...")
if CHECKPOINT_EXISTS:
    tokenizer = AutoTokenizer.from_pretrained(str(CKPT_DIR))
    print(f"  Loaded from checkpoint: {CKPT_DIR}")
else:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print(f"  Loaded from HuggingFace: {MODEL_NAME}")

def tokenize(batch):
    return tokenizer(
        batch["description"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN
    )

def df_to_dataset(df: pd.DataFrame) -> Dataset:
    ds = Dataset.from_dict({
        "description": df["description"].tolist(),
        "label":       df["label"].astype(int).tolist()
    })
    ds = ds.map(tokenize, batched=True, batch_size=512,
                remove_columns=["description"])
    # cast to int64 explicitly — prevents Float label TypeError
    return ds.cast_column("label", Value("int64"))

need_train_ds = not SKIP_TRAINING
need_val_ds   = not SKIP_VAL_EVAL
need_test_ds  = not SKIP_TEST_EVAL

print("Tokenizing splits...")
if need_train_ds:
    print("  Tokenizing train...")
    train_ds = df_to_dataset(train_df)
else:
    train_ds = None
    print("  Train — skipped (training not needed)")

if need_val_ds:
    print("  Tokenizing val...")
    val_ds = df_to_dataset(val_df)
else:
    val_ds = None
    print("  Val   — skipped (val results already cached)")

if need_test_ds:
    print("  Tokenizing test...")
    test_ds = df_to_dataset(test_df)
else:
    test_ds = None
    print("  Test  — skipped (test results already cached)")

if train_ds is not None:
    print(f"  Train tokens shape : {train_ds.shape}")
if val_ds is not None:
    print(f"  Val   tokens shape : {val_ds.shape}")
if test_ds is not None:
    print(f"  Test  tokens shape : {test_ds.shape}")

# ── WeightedTrainer ───────────────────────────────────────────────────────────

class WeightedTrainer(Trainer):
    """
    Trainer subclass with two overrides:

    compute_loss     — applies inverse-frequency class weights to CrossEntropyLoss
    prediction_step  — uses plain forward pass during eval to avoid the
                       Accelerator pad_across_processes CUDA assertion error
                       that occurs on the large (149K+) test set
    """

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs.pop("labels").long()   # force int64
        outputs = model(**inputs)
        logits  = outputs.logits
        loss    = CrossEntropyLoss(weight=class_weights)(logits, labels)
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only,
                        ignore_keys=None):
        """Plain forward pass — bypasses Accelerator padding logic."""
        inputs = self._prepare_inputs(inputs)
        labels = inputs.pop("labels", None)

        with torch.no_grad():
            outputs = model(**inputs)
            logits  = outputs.logits

        loss = None
        if labels is not None:
            loss = CrossEntropyLoss(weight=class_weights)(
                logits, labels.long()
            ).detach()

        return (
            loss,
            logits.detach(),
            labels.detach() if labels is not None else None
        )


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy":    round(float(accuracy_score(labels, preds)), 4),
        "f1_weighted": round(float(f1_score(labels, preds, average="weighted",
                                            labels=list(range(4)))), 4),
        "f1_macro":    round(float(f1_score(labels, preds, average="macro",
                                            labels=list(range(4)))), 4),
    }

# ── load model ────────────────────────────────────────────────────────────────

if CHECKPOINT_EXISTS:
    print(f"\nLoading saved model from {CKPT_DIR}...")
    model = AutoModelForSequenceClassification.from_pretrained(
        str(CKPT_DIR),
        num_labels=len(LABEL_ORDER),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True
    )
else:
    print(f"\nNo checkpoint — loading base model from {MODEL_NAME}...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL_ORDER),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True
    )

total_params     = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Total params     : {total_params:,}")
print(f"  Trainable params : {trainable_params:,}")

# ── TrainingArguments ─────────────────────────────────────────────────────────

args = TrainingArguments(
    output_dir                  = str(CKPT_DIR),
    num_train_epochs            = EPOCHS,
    per_device_train_batch_size = BATCH_SIZE,
    per_device_eval_batch_size  = 32,
    learning_rate               = LR,
    warmup_steps                = 500,
    weight_decay                = 0.01,
    **{_EVAL_KEY: "epoch"},          # eval_strategy or evaluation_strategy
    save_strategy               = "epoch",
    load_best_model_at_end      = True,
    metric_for_best_model       = "f1_weighted",
    greater_is_better           = True,
    logging_steps               = 100,
    fp16                        = (DEVICE == "cuda"),
    dataloader_num_workers      = 0,
    report_to                   = "none",
    save_total_limit            = 2,
)

trainer = WeightedTrainer(
    model           = model,
    args            = args,
    train_dataset   = train_ds,
    eval_dataset    = val_ds,
    compute_metrics = compute_metrics,
    callbacks       = [EarlyStoppingCallback(early_stopping_patience=2)]
)

# ── stage 1: training ─────────────────────────────────────────────────────────

if SKIP_TRAINING:
    print(f"\n[STAGE 1] Training — SKIPPED (checkpoint exists)")
else:
    print(f"\n[STAGE 1] Training — RUNNING")
    print(f"  Epochs     : {EPOCHS}")
    print(f"  Batch size : {BATCH_SIZE}")
    print(f"  LR         : {LR}")
    print(f"  Max length : {MAX_LEN}")
    print(f"  Device     : {DEVICE}")

    train_result = trainer.train()

    print("\n  Training complete.")
    print(f"  Total steps   : {train_result.global_step:,}")
    print(f"  Training loss : {train_result.training_loss:.4f}")
    print(f"  Training time : {train_result.metrics['train_runtime']:.0f}s "
          f"({train_result.metrics['train_runtime']/60:.1f} min)")

    trainer.save_model(str(CKPT_DIR))
    tokenizer.save_pretrained(str(CKPT_DIR))
    print(f"  Checkpoint saved to {CKPT_DIR}")

# ── stage 2: training curves ──────────────────────────────────────────────────

if SKIP_CURVES:
    print(f"\n[STAGE 2] Training curves — SKIPPED (PNG already exists)")
elif SKIP_TRAINING:
    print(f"\n[STAGE 2] Training curves — SKIPPED "
          "(training was skipped; log history unavailable)")
else:
    print(f"\n[STAGE 2] Plotting training curves...")
    log_history = trainer.state.log_history

    train_loss, val_f1, val_acc, epochs_logged = [], [], [], []
    for entry in log_history:
        if "loss" in entry and "eval_loss" not in entry:
            train_loss.append(entry["loss"])
        if "eval_f1_weighted" in entry:
            val_f1.append(entry["eval_f1_weighted"])
            val_acc.append(entry["eval_accuracy"])
            epochs_logged.append(entry["epoch"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(train_loss, color="#58A6FF", linewidth=1.5)
    axes[0].set_title("Training loss")
    axes[0].set_xlabel("Step (per 100)")
    axes[0].set_ylabel("Loss")
    axes[0].spines[["top", "right"]].set_visible(False)

    axes[1].plot(epochs_logged, val_f1, color="#3FB950", linewidth=2,
                 marker="o", label="F1-weighted")
    axes[1].plot(epochs_logged, val_acc, color="#58A6FF", linewidth=2,
                 marker="s", linestyle="--", label="Accuracy")
    axes[1].set_title("Validation metrics per epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].spines[["top", "right"]].set_visible(False)

    plt.suptitle("SecBERT fine-tuning", fontsize=13)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "secbert_training_curves.png", dpi=150)
    plt.close()
    print("  Saved secbert_training_curves.png")

# ── stage 3: evaluation ───────────────────────────────────────────────────────

def full_evaluate(split_name: str, dataset, df_ref: pd.DataFrame) -> dict:
    """
    Run inference using a plain PyTorch DataLoader — does NOT use
    trainer.predict() which routes through the HuggingFace Accelerator
    and triggers CUDA assertion errors on large test sets.
    """
    print(f"\n  Evaluating on {split_name.upper()} ({len(df_ref):,} samples)...")

    model.eval()
    model.to(DEVICE)

    all_logits, all_labels = [], []
    dataloader = torch.utils.data.DataLoader(
        dataset.with_format("torch"),
        batch_size=64,
        shuffle=False
    )

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels         = batch["label"]

            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask)
            all_logits.append(outputs.logits.cpu().numpy())
            all_labels.append(labels.numpy())

            if i % 50 == 0:
                print(f"    Batch {i+1}/{len(dataloader)}", end="\r")

    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    preds       = np.argmax(all_logits, axis=-1)
    pred_labels = [ID2LABEL[p] for p in preds]
    true_labels = [ID2LABEL[l] for l in all_labels]

    acc          = accuracy_score(all_labels, preds)
    f1_weighted  = f1_score(all_labels, preds, average="weighted",
                            labels=list(range(4)))
    f1_macro     = f1_score(all_labels, preds, average="macro",
                            labels=list(range(4)))
    f1_per_class = f1_score(all_labels, preds, average=None,
                            labels=list(range(4)))

    print(f"\n  [{split_name.upper()}]")
    print(f"  {'Accuracy':<20} {acc:.4f}")
    print(f"  {'F1-weighted':<20} {f1_weighted:.4f}")
    print(f"  {'F1-macro':<20} {f1_macro:.4f}")
    print()
    print(classification_report(true_labels, pred_labels,
                                labels=LABEL_ORDER, digits=4))

    # confusion matrix
    cm      = confusion_matrix(true_labels, pred_labels, labels=LABEL_ORDER)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm_norm, annot=True, fmt=".2f",
                xticklabels=LABEL_ORDER, yticklabels=LABEL_ORDER,
                cmap="Blues", ax=ax, linewidths=0.5,
                annot_kws={"size": 11})
    ax.set_title(f"SecBERT — confusion matrix ({split_name})", fontsize=12)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    fname = RESULTS_DIR / f"confusion_secbert_{split_name}.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  Saved {fname.name}")

    return {
        "accuracy":     round(float(acc),         4),
        "f1_weighted":  round(float(f1_weighted),  4),
        "f1_macro":     round(float(f1_macro),     4),
        "f1_per_class": {
            label: round(float(score), 4)
            for label, score in zip(LABEL_ORDER, f1_per_class)
        }
    }


# load any existing results (may already have val from a previous partial run)
secbert_results = {
    "model":      MODEL_NAME,
    "max_len":    MAX_LEN,
    "batch_size": BATCH_SIZE,
    "epochs":     EPOCHS,
    "lr":         LR,
}
if FIRST_RUN_EXISTS:
    with open(RESULTS_DIR / "secbert_first_run.json") as f:
        _cached = json.load(f)
    if "val"  in _cached: secbert_results["val"]  = _cached["val"]
    if "test" in _cached: secbert_results["test"] = _cached["test"]
    print(f"\n[STAGE 3] Loaded existing results from secbert_first_run.json")

# val evaluation
if SKIP_VAL_EVAL:
    print(f"\n[STAGE 3a] Val evaluation — SKIPPED (results cached)")
    val_results = secbert_results["val"]
    print(f"  Cached val F1-weighted : {val_results['f1_weighted']}")
    print(f"  Cached val F1-macro    : {val_results['f1_macro']}")
    print(f"  Cached val Accuracy    : {val_results['accuracy']}")
else:
    print(f"\n[STAGE 3a] Val evaluation — RUNNING")
    val_results = full_evaluate("val", val_ds, val_df)
    secbert_results["val"] = val_results
    # write immediately — so a disconnect before test eval doesn't lose val
    with open(RESULTS_DIR / "secbert_first_run.json", "w") as f:
        json.dump(secbert_results, f, indent=2)
    print("  Val results written to secbert_first_run.json")

# test evaluation
if SKIP_TEST_EVAL:
    print(f"\n[STAGE 3b] Test evaluation — SKIPPED (results cached)")
    test_results = secbert_results["test"]
    print(f"  Cached test F1-weighted : {test_results['f1_weighted']}")
    print(f"  Cached test F1-macro    : {test_results['f1_macro']}")
    print(f"  Cached test Accuracy    : {test_results['accuracy']}")
else:
    print(f"\n[STAGE 3b] Test evaluation — RUNNING")
    test_results = full_evaluate("test", test_ds, test_df)
    secbert_results["test"] = test_results
    with open(RESULTS_DIR / "secbert_first_run.json", "w") as f:
        json.dump(secbert_results, f, indent=2)
    print("  Test results written to secbert_first_run.json")

# ── stage 4: comparison table ─────────────────────────────────────────────────
# Always runs — no heavy compute, just reads JSONs and prints.

print(f"\n[STAGE 4] Model comparison table")

baseline_path = RESULTS_DIR / "baseline_results.json"
if baseline_path.exists():
    print(f"\n{'='*60}")
    print("  MODEL COMPARISON — VAL SET")
    print(f"{'='*60}")
    with open(baseline_path) as f:
        baselines = json.load(f)
    rows = []
    for name, res in baselines.items():
        rows.append({
            "Model":       name,
            "Accuracy":    float(res["val"]["accuracy"]),
            "F1-Weighted": float(res["val"]["f1_weighted"]),
            "F1-Macro":    float(res["val"]["f1_macro"])
        })
    rows.append({
        "Model":       "SecBERT (fine-tuned)",
        "Accuracy":    val_results["accuracy"],
        "F1-Weighted": val_results["f1_weighted"],
        "F1-Macro":    val_results["f1_macro"]
    })
    print(pd.DataFrame(rows).to_string(index=False))
else:
    print("  baseline_results.json not found — run baseline.py first")
    print(f"\n  SecBERT val  — "
          f"F1-weighted={val_results['f1_weighted']}  "
          f"F1-macro={val_results['f1_macro']}")
    print(f"  SecBERT test — "
          f"F1-weighted={test_results['f1_weighted']}  "
          f"F1-macro={test_results['f1_macro']}")

print(f"\n{'='*60}")
print("  DONE")
print(f"{'='*60}")
print(f"  Checkpoint : {CKPT_DIR}")
print(f"  Results    : {RESULTS_DIR / 'secbert_first_run.json'}")
print(f"  Plots      : {RESULTS_DIR}/confusion_secbert_*.png")