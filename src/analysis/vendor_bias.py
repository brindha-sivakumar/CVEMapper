# src/analysis/vendor_bias.py
#
# Phase 3 — Vendor Bias Study
# Evaluates whether model performance varies across CVEs from Microsoft,
# Linux, and Apache by using regex pattern matching on description text.
#
# Run after: train.py has saved checkpoints/secbert_best/
# Outputs:   results/vendor_results.json
#            results/vendor_comparison.png
#            results/vendor_f1_heatmap.png
#            results/vendor_severity_dist.png

import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset, Value
from sklearn.metrics import accuracy_score, f1_score
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

SEVERITY_COLORS = {
    "CRITICAL": "#DA3633", "HIGH": "#E3B341",
    "MEDIUM":   "#58A6FF", "LOW":  "#3FB950",
}

VENDOR_PATTERNS = {
    "Microsoft": [
        r"\bmicrosoft\b", r"\bwindows\b", r"\bazure\b", r"\boffice\b",
        r"\bexchange server\b", r"\bactive directory\b", r"\b\.net\b",
        r"\bvisual studio\b", r"\bpowershell\b", r"\binternet explorer\b",
        r"\bedge browser\b", r"\bsharepoint\b",
    ],
    "Linux": [
        r"\blinux kernel\b", r"\bin the linux\b", r"\bkernel\b.*\blinux\b",
        r"\blinux\b.*\bkernel\b", r"\bnetfilter\b", r"\bebpf\b", r"\bkvm\b",
    ],
    "Apache": [
        r"\bapache\b", r"\bhttpd\b", r"\btomcat\b", r"\bstruts\b",
        r"\blog4j\b", r"\bsolr\b",
    ],
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device : {DEVICE}")

# ── load model ────────────────────────────────────────────────────────────────

print(f"\nLoading checkpoint...")
assert (CKPT_DIR / "config.json").exists(), \
    "Checkpoint not found — run train.py first."

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
test_df = test_df.dropna(subset=["label"]).reset_index(drop=True)
test_df["label"] = test_df["label"].astype(int)
print(f"  Rows : {len(test_df):,}")

# ── vendor detection ──────────────────────────────────────────────────────────

def detect_vendor(desc: str) -> str:
    desc_lower = str(desc).lower()
    for vendor, patterns in VENDOR_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, desc_lower):
                return vendor
    return "Other"

test_df["vendor"] = test_df["description"].apply(detect_vendor)

print("\nVendor distribution in test set:")
vc = test_df["vendor"].value_counts()
print(vc.to_string())
print(f"\n  Coverage: {(test_df['vendor'] != 'Other').mean():.1%} matched a vendor")

# ── inference helper ──────────────────────────────────────────────────────────

def run_inference(df: pd.DataFrame) -> np.ndarray:
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
    all_logits = []
    dl = torch.utils.data.DataLoader(
        ds.with_format("torch"), batch_size=64, shuffle=False
    )
    with torch.no_grad():
        for batch in dl:
            out = model(input_ids      = batch["input_ids"].to(DEVICE),
                        attention_mask = batch["attention_mask"].to(DEVICE))
            all_logits.append(out.logits.cpu().numpy())
    return np.argmax(np.concatenate(all_logits), axis=-1)


def compute_metrics(df: pd.DataFrame, preds: np.ndarray) -> dict:
    labels = df["label"].values
    f1_pc  = f1_score(labels, preds, average=None,
                      labels=list(range(4)), zero_division=0)
    return {
        "n":           int(len(df)),
        "accuracy":    round(float(accuracy_score(labels, preds)), 4),
        "f1_weighted": round(float(f1_score(labels, preds, average="weighted",
                                            labels=list(range(4)))), 4),
        "f1_macro":    round(float(f1_score(labels, preds, average="macro",
                                            labels=list(range(4)))), 4),
        "f1_per_class": {lbl: round(float(s), 4)
                         for lbl, s in zip(LABEL_ORDER, f1_pc)},
        "severity_dist": df["severity"].value_counts(normalize=True).round(3).to_dict(),
    }

# ── per-vendor inference ──────────────────────────────────────────────────────

VENDORS = ["Microsoft", "Linux", "Apache", "Other"]
vendor_results = {}

print(f"\n{'='*60}")
print("  VENDOR-STRATIFIED PERFORMANCE")
print(f"{'='*60}")

for vendor in VENDORS:
    sub = test_df[test_df["vendor"] == vendor].copy()
    if len(sub) < 100:
        print(f"\n  {vendor}: too few samples ({len(sub)}) — skipping")
        continue
    print(f"\n  {vendor} ({len(sub):,} CVEs)...")
    preds = run_inference(sub)
    m     = compute_metrics(sub, preds)
    vendor_results[vendor] = m
    print(f"    Accuracy    : {m['accuracy']:.4f}")
    print(f"    F1-Weighted : {m['f1_weighted']:.4f}")
    print(f"    F1-Macro    : {m['f1_macro']:.4f}")

# overall baseline
print(f"\n  Overall ({len(test_df):,} CVEs)...")
all_preds = run_inference(test_df)
overall   = compute_metrics(test_df, all_preds)
vendor_results["Overall"] = overall

# comparison table
print(f"\n{'='*60}")
print("  COMPARISON TABLE")
print(f"{'='*60}")
print(f"  {'Vendor':<12} {'N':>8} {'Accuracy':>10} {'F1-Weighted':>12} {'F1-Macro':>10}")
print(f"  {'-'*54}")
for v in VENDORS + ["Overall"]:
    if v not in vendor_results:
        continue
    m = vendor_results[v]
    print(f"  {v:<12} {m['n']:>8,} {m['accuracy']:>10.4f} "
          f"{m['f1_weighted']:>12.4f} {m['f1_macro']:>10.4f}")

# ── plots ─────────────────────────────────────────────────────────────────────

available = [v for v in VENDORS if v in vendor_results]
overall_f1 = overall["f1_weighted"]

# grouped bar chart
fig, ax = plt.subplots(figsize=(9, 4))
x, w = np.arange(len(available)), 0.25
f1w = [vendor_results[v]["f1_weighted"] for v in available]
f1m = [vendor_results[v]["f1_macro"]    for v in available]
acc = [vendor_results[v]["accuracy"]    for v in available]
ax.bar(x-w, f1w, w, label="F1-Weighted", color="#58A6FF", edgecolor="none")
ax.bar(x,   f1m, w, label="F1-Macro",    color="#3FB950", edgecolor="none")
ax.bar(x+w, acc, w, label="Accuracy",    color="#E3B341", edgecolor="none")
ax.axhline(overall_f1, color="#DA3633", linestyle="--", linewidth=1.2,
           label=f"Overall F1-W ({overall_f1:.3f})")
ax.set_xticks(x)
ax.set_xticklabels(available)
ax.set_ylim(0.5, 1.0)
ax.set_title("Model performance by vendor", fontsize=12)
ax.set_ylabel("Score")
ax.legend(fontsize=9)
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "vendor_comparison.png", dpi=150)
plt.close()
print(f"\n  Saved vendor_comparison.png")

# per-class F1 heatmap
named = [v for v in available if v != "Other"]
f1_matrix = pd.DataFrame(
    {v: vendor_results[v]["f1_per_class"] for v in named},
    index=LABEL_ORDER
).T

fig, ax = plt.subplots(figsize=(7, len(named)*1.2+1))
sns.heatmap(f1_matrix, annot=True, fmt=".3f", cmap="YlGn",
            ax=ax, linewidths=0.5, annot_kws={"size":11}, vmin=0.3, vmax=1.0)
ax.set_title("Per-class F1 by vendor", fontsize=12)
ax.set_xlabel("Severity class")
ax.set_ylabel("Vendor")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "vendor_f1_heatmap.png", dpi=150)
plt.close()
print(f"  Saved vendor_f1_heatmap.png")

# severity distribution
fig, axes = plt.subplots(1, len(named), figsize=(4*len(named), 4), sharey=True)
if len(named) == 1:
    axes = [axes]
for ax, vendor in zip(axes, named):
    dist = test_df[test_df["vendor"] == vendor]["severity"].value_counts()
    dist = dist.reindex(LABEL_ORDER, fill_value=0)
    ax.bar(dist.index, dist.values,
           color=[SEVERITY_COLORS[s] for s in dist.index],
           edgecolor="none", width=0.6)
    ax.set_title(vendor, fontsize=11)
    ax.set_ylabel("Count")
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.spines[["top","right"]].set_visible(False)
plt.suptitle("Severity distribution by vendor (test set)", fontsize=12)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "vendor_severity_dist.png", dpi=150)
plt.close()
print(f"  Saved vendor_severity_dist.png")

# ── save ──────────────────────────────────────────────────────────────────────

with open(RESULTS_DIR / "vendor_results.json", "w") as f:
    json.dump(vendor_results, f, indent=2)
print(f"  Saved vendor_results.json")
print("\nVendor bias analysis complete.")