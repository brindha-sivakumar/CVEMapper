# src/training/train.py

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import inspect
import torch
from torch.nn import CrossEntropyLoss
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score, f1_score,
    classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

load_dotenv()

# ── config ────────────────────────────────────────────────────────────────────

MODEL_NAME   = "jackaduma/SecBERT"
SPLIT_DIR    = Path("data/processed")
CKPT_DIR     = Path("checkpoints/secbert_best")
RESULTS_DIR  = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_ORDER = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
LABEL2ID    = {l: i for i, l in enumerate(LABEL_ORDER)}
ID2LABEL    = {i: l for i, l in enumerate(LABEL_ORDER)}

SEVERITY_COLORS = {
    "CRITICAL": "#DA3633",
    "HIGH":     "#E3B341",
    "MEDIUM":   "#58A6FF",
    "LOW":      "#3FB950"
}

MAX_LEN    = 256
BATCH_SIZE = 16
EPOCHS     = 5
LR         = 2e-5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device : {DEVICE}")
if DEVICE == "cuda":
    print(f"  GPU   : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM  : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# detect correct eval key for this transformers version
_ta_params = inspect.signature(TrainingArguments.__init__).parameters
_EVAL_KEY   = "eval_strategy" if "eval_strategy" in _ta_params else "evaluation_strategy"
print(f"TrainingArguments eval key : {_EVAL_KEY}")

# ── load splits ───────────────────────────────────────────────────────────────

print("\nLoading splits...")
train_df = pd.read_csv(SPLIT_DIR / "train.csv")
val_df   = pd.read_csv(SPLIT_DIR / "val.csv")
test_df  = pd.read_csv(SPLIT_DIR / "test.csv")

print(f"  Train : {len(train_df):,}")
print(f"  Val   : {len(val_df):,}")
print(f"  Test  : {len(test_df):,}")

# map severity → integer label
for df in [train_df, val_df, test_df]:
    df["label"] = df["severity"].map(LABEL2ID)

# sanity check — no NaN labels
assert train_df["label"].notna().all(), "NaN labels in train"
assert val_df["label"].notna().all(),   "NaN labels in val"


# ── class weights ─────────────────────────────────────────────────────────────

print("\nComputing class weights...")
counts = train_df["label"].value_counts().sort_index()
weights = len(train_df) / (len(LABEL_ORDER) * counts)
class_weights = torch.tensor(weights.values, dtype=torch.float32).to(DEVICE)

print("  Class weights (inverse-frequency):")
for i, (label, w) in enumerate(zip(LABEL_ORDER, class_weights)):
    print(f"    {label:<10} count={counts[i]:>7,}   weight={w:.4f}")


# ── tokenizer ─────────────────────────────────────────────────────────────────

print(f"\nLoading tokenizer from {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

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
        "label":       df["label"].tolist()
    })
    return ds.map(tokenize, batched=True, batch_size=512,
                  remove_columns=["description"])

print("Tokenizing splits (this takes a few minutes)...")
train_ds = df_to_dataset(train_df)
val_ds   = df_to_dataset(val_df)
test_ds  = df_to_dataset(test_df)
print(f"  Train tokens shape : {train_ds.shape}")
print(f"  Val   tokens shape : {val_ds.shape}")


# ── custom trainer with weighted loss ─────────────────────────────────────────

class WeightedTrainer(Trainer):
    """Trainer subclass that applies inverse-frequency class weights to CE loss."""
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits  = outputs.logits
        loss_fn = CrossEntropyLoss(weight=class_weights)
        loss    = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


# ── metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy":    round(accuracy_score(labels, preds), 4),
        "f1_weighted": round(f1_score(labels, preds, average="weighted",
                                      labels=list(range(4))), 4),
        "f1_macro":    round(f1_score(labels, preds, average="macro",
                                      labels=list(range(4))), 4),
    }


# ── model ─────────────────────────────────────────────────────────────────────

print(f"\nLoading model from {MODEL_NAME}...")
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


# ── training arguments ────────────────────────────────────────────────────────

args = TrainingArguments(
    output_dir                  = str(CKPT_DIR),
    num_train_epochs            = EPOCHS,
    per_device_train_batch_size = BATCH_SIZE,
    per_device_eval_batch_size  = 32,
    learning_rate               = LR,
    warmup_steps                = 500,
    weight_decay                = 0.01,
    **{_EVAL_KEY: "epoch"},
    save_strategy               = "epoch",
    load_best_model_at_end      = True,
    metric_for_best_model       = "f1_weighted",
    greater_is_better           = True,
    logging_steps               = 100,
    logging_dir                 = str(RESULTS_DIR / "logs"),
    fp16                        = (DEVICE == "cuda"),
    dataloader_num_workers      = 0,
    report_to                   = "none",
    save_total_limit            = 2,
)


# ── train ─────────────────────────────────────────────────────────────────────

print("\nStarting training...")
print(f"  Epochs     : {EPOCHS}")
print(f"  Batch size : {BATCH_SIZE}")
print(f"  LR         : {LR}")
print(f"  Max length : {MAX_LEN}")
print(f"  Device     : {DEVICE}")
print()

trainer = WeightedTrainer(
    model           = model,
    args            = args,
    train_dataset   = train_ds,
    eval_dataset    = val_ds,
    compute_metrics = compute_metrics,
    callbacks       = [EarlyStoppingCallback(early_stopping_patience=2)]
)

train_result = trainer.train()

print("\nTraining complete.")
print(f"  Total steps     : {train_result.global_step:,}")
print(f"  Training loss   : {train_result.training_loss:.4f}")
print(f"  Training time   : {train_result.metrics['train_runtime']:.0f}s "
      f"({train_result.metrics['train_runtime']/60:.1f} min)")


# ── training curve ────────────────────────────────────────────────────────────

print("\nPlotting training curves...")
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

axes[1].plot(epochs_logged, val_f1,  color="#3FB950", linewidth=2,
             marker="o", label="F1-weighted")
axes[1].plot(epochs_logged, val_acc, color="#58A6FF", linewidth=2,
             marker="s", linestyle="--", label="Accuracy")
axes[1].set_title("Validation metrics per epoch")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Score")
axes[1].legend()
axes[1].spines[["top", "right"]].set_visible(False)

plt.suptitle("SecBERT fine-tuning — first run", fontsize=13)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "secbert_training_curves.png", dpi=150)
plt.close()
print("  Saved secbert_training_curves.png")


# ── evaluate on val and test ──────────────────────────────────────────────────

print("\nEvaluating best checkpoint...")

def full_evaluate(split_name: str, dataset, df_ref: pd.DataFrame) -> dict:
    pred_out = trainer.predict(dataset)
    preds    = np.argmax(pred_out.predictions, axis=-1)
    labels   = pred_out.label_ids
    pred_labels = [ID2LABEL[p] for p in preds]
    true_labels = [ID2LABEL[l] for l in labels]

    acc         = accuracy_score(labels, preds)
    f1_weighted = f1_score(labels, preds, average="weighted", labels=list(range(4)))
    f1_macro    = f1_score(labels, preds, average="macro",    labels=list(range(4)))
    f1_per_class = f1_score(labels, preds, average=None,      labels=list(range(4)))

    print(f"\n  [{split_name.upper()}]")
    print(f"  {'Accuracy':<20} {acc:.4f}")
    print(f"  {'F1-weighted':<20} {f1_weighted:.4f}")
    print(f"  {'F1-macro':<20} {f1_macro:.4f}")
    print()
    print(classification_report(
        true_labels, pred_labels,
        labels=LABEL_ORDER, digits=4
    ))

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
        "accuracy":     round(acc, 4),
        "f1_weighted":  round(f1_weighted, 4),
        "f1_macro":     round(f1_macro, 4),
        "f1_per_class": {
            label: round(score, 4)
            for label, score in zip(LABEL_ORDER, f1_per_class)
        }
    }

val_results  = full_evaluate("val",  val_ds,  val_df)
test_results = full_evaluate("test", test_ds, test_df)


# ── comparison vs baselines ───────────────────────────────────────────────────

print("\n" + "="*60)
print("  MODEL COMPARISON — VAL SET")
print("="*60)

baseline_path = RESULTS_DIR / "baseline_results.json"
if baseline_path.exists():
    with open(baseline_path) as f:
        baselines = json.load(f)
    rows = []
    for name, res in baselines.items():
        rows.append({
            "Model": name,
            "Accuracy":    res["val"]["accuracy"],
            "F1-Weighted": res["val"]["f1_weighted"],
            "F1-Macro":    res["val"]["f1_macro"]
        })
    rows.append({
        "Model":       "SecBERT (fine-tuned)",
        "Accuracy":    val_results["accuracy"],
        "F1-Weighted": val_results["f1_weighted"],
        "F1-Macro":    val_results["f1_macro"]
    })
    comparison_df = pd.DataFrame(rows)
    print(comparison_df.to_string(index=False))


# ── save results ──────────────────────────────────────────────────────────────

secbert_results = {
    "model":      MODEL_NAME,
    "max_len":    MAX_LEN,
    "batch_size": BATCH_SIZE,
    "epochs":     EPOCHS,
    "lr":         LR,
    "val":        val_results,
    "test":       test_results,
}
with open(RESULTS_DIR / "secbert_first_run.json", "w") as f:
    json.dump(secbert_results, f, indent=2)
print(f"\n  Saved secbert_first_run.json")

# save model and tokenizer
trainer.save_model(str(CKPT_DIR))
tokenizer.save_pretrained(str(CKPT_DIR))
print(f"  Saved best checkpoint → {CKPT_DIR}")
print("\nFirst run complete.")