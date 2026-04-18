# src/analysis/attribution.py
#
# Phase 3 — Token Attribution Analysis
# Uses transformers-interpret (integrated gradients) to identify which
# words in CVE descriptions drive severity predictions.
# Runs on CPU for numerical stability.
#
# Install: pip install transformers-interpret
# Run after: train.py has saved checkpoints/secbert_best/
# Outputs:   results/attribution_heatmaps/*.html   (60 files)
#            results/attribution_top_tokens.png
#            results/attribution_score_dist.png
#            results/attribution_results.json
#            results/attribution_samples.csv

import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import warnings
warnings.filterwarnings("ignore")

# ── paths ─────────────────────────────────────────────────────────────────────

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

CKPT_DIR    = PROJECT_ROOT / "checkpoints" / "secbert_best"
SPLIT_DIR   = PROJECT_ROOT / "data"        / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
ATTR_DIR    = RESULTS_DIR  / "attribution_heatmaps"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
ATTR_DIR.mkdir(parents=True, exist_ok=True)

LABEL_ORDER          = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
LABEL2ID             = {l: i for i, l in enumerate(LABEL_ORDER)}
ID2LABEL             = {i: l for i, l in enumerate(LABEL_ORDER)}
N_SAMPLES_PER_CLASS  = 15
TOP_K                = 10
RANDOM_SEED          = 42

SEVERITY_COLORS = {
    "CRITICAL": "#DA3633", "HIGH": "#E3B341",
    "MEDIUM":   "#58A6FF", "LOW":  "#3FB950",
}

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ── check dependency ──────────────────────────────────────────────────────────

try:
    from transformers_interpret import SequenceClassificationExplainer
except ImportError:
    raise ImportError(
        "transformers-interpret not installed.\n"
        "Run:  pip install transformers-interpret"
    )

# attribution runs on CPU — GPU can produce NaN integrated gradients
print("Running attribution on CPU for numerical stability.")

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
# explicitly keep on CPU
model.to("cpu")
print("  Model loaded on CPU.")

explainer = SequenceClassificationExplainer(model, tokenizer)

# ── sample test set ───────────────────────────────────────────────────────────

print("\nLoading test split...")
test_df = pd.read_csv(SPLIT_DIR / "test.csv")
test_df["severity"] = test_df["severity"].astype(str).str.strip().str.upper()
test_df = test_df[test_df["severity"].isin(LABEL_ORDER)].reset_index(drop=True)
print(f"  Rows : {len(test_df):,}")

samples = {}
for sev in LABEL_ORDER:
    pool = test_df[test_df["severity"] == sev].sample(
        frac=1, random_state=RANDOM_SEED
    )
    samples[sev] = pool.head(N_SAMPLES_PER_CLASS)
    print(f"  Sampled {len(samples[sev])} {sev} examples")

# ── attribution loop ──────────────────────────────────────────────────────────

print(f"\nRunning attribution ({N_SAMPLES_PER_CLASS * len(LABEL_ORDER)} samples)...")
print("  Expect ~2–5 min total on CPU.")

token_importance = defaultdict(lambda: defaultdict(float))
token_counts     = defaultdict(lambda: defaultdict(int))
rows             = []

for sev in LABEL_ORDER:
    print(f"\n  {sev}...")
    for i, (_, row) in enumerate(samples[sev].iterrows()):
        desc = str(row["description"])
        try:
            word_attr = explainer(desc, class_name=sev)
        except Exception as e:
            print(f"    Sample {i+1}: failed — {e}")
            continue

        pred_class = explainer.predicted_class_name

        # save HTML heatmap
        html_path = ATTR_DIR / f"{sev.lower()}_sample_{i+1:02d}.html"
        try:
            explainer.visualize(html_path)
        except Exception:
            pass

        # aggregate token importance
        if word_attr:
            sorted_attr = sorted(word_attr, key=lambda x: abs(x[1]), reverse=True)
            top_tokens  = [(tok, float(score))
                           for tok, score in sorted_attr[:TOP_K]
                           if tok not in ["[CLS]","[SEP]","[PAD]"]]
            for tok, score in top_tokens:
                clean = tok.lower().strip("##")
                if len(clean) > 2:
                    token_importance[sev][clean] += abs(score)
                    token_counts[sev][clean]     += 1
            rows.append({
                "severity":    sev,
                "predicted":   pred_class,
                "correct":     (pred_class == sev),
                "description": desc[:200],
                "top_tokens":  str(top_tokens[:5]),
                "top_1_token": top_tokens[0][0] if top_tokens else "",
                "top_1_score": top_tokens[0][1] if top_tokens else 0.0,
            })

        if (i+1) % 5 == 0:
            print(f"    {i+1}/{N_SAMPLES_PER_CLASS}", end="\r")
    print(f"    {N_SAMPLES_PER_CLASS}/{N_SAMPLES_PER_CLASS}")

print(f"\n  HTML heatmaps saved to {ATTR_DIR.name}/")

# ── aggregate top tokens ──────────────────────────────────────────────────────

print(f"\n{'='*60}")
print("  TOP ATTRIBUTION TOKENS PER SEVERITY CLASS")
print(f"{'='*60}")

top_tokens_per_class = {}
for sev in LABEL_ORDER:
    mean_imp = {
        tok: token_importance[sev][tok] / token_counts[sev][tok]
        for tok in token_importance[sev]
        if token_counts[sev][tok] >= 2
    }
    top = sorted(mean_imp.items(), key=lambda x: x[1], reverse=True)[:TOP_K]
    top_tokens_per_class[sev] = top
    print(f"\n  {sev}:")
    for rank, (tok, score) in enumerate(top, 1):
        print(f"    {rank:>2}. {tok:<25} mean |attr| = {score:.4f}")

# ── plots ─────────────────────────────────────────────────────────────────────

# bar charts per class
fig, axes = plt.subplots(1, 4, figsize=(18, 5))
for ax, sev in zip(axes, LABEL_ORDER):
    tokens = top_tokens_per_class.get(sev, [])
    if not tokens:
        ax.set_title(f"{sev}\n(no data)")
        continue
    toks   = [t[0] for t in tokens[:8]]
    scores = [t[1] for t in tokens[:8]]
    ax.barh(toks[::-1], scores[::-1],
            color=SEVERITY_COLORS[sev], edgecolor="none")
    ax.set_title(f"{sev}\ntop attribution tokens", fontsize=10)
    ax.set_xlabel("Mean |attribution|")
    ax.spines[["top","right"]].set_visible(False)
plt.suptitle("Top influential tokens per severity class", fontsize=12)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "attribution_top_tokens.png", dpi=150)
plt.close()
print(f"\n  Saved attribution_top_tokens.png")

# score distribution
attr_df = pd.DataFrame(rows)
if not attr_df.empty:
    fig, ax = plt.subplots(figsize=(8, 4))
    for sev in LABEL_ORDER:
        sub = attr_df[attr_df["severity"] == sev]["top_1_score"].dropna()
        if len(sub) > 0:
            ax.hist(sub, bins=15, alpha=0.55,
                    color=SEVERITY_COLORS[sev], label=sev)
    ax.set_title("Distribution of top-1 attribution score by severity", fontsize=11)
    ax.set_xlabel("Attribution score (|integrated gradient|)")
    ax.set_ylabel("Count")
    ax.legend(fontsize=9)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "attribution_score_dist.png", dpi=150)
    plt.close()
    print(f"  Saved attribution_score_dist.png")

# ── save ──────────────────────────────────────────────────────────────────────

attribution_results = {
    "n_samples_per_class": N_SAMPLES_PER_CLASS,
    "top_k_tokens":        TOP_K,
    "sample_count":        len(rows),
    "correct_predictions": int(attr_df["correct"].sum()) if not attr_df.empty else 0,
    "top_tokens_per_class": {
        sev: [(tok, round(score, 4)) for tok, score in tokens]
        for sev, tokens in top_tokens_per_class.items()
    },
}
with open(RESULTS_DIR / "attribution_results.json", "w") as f:
    json.dump(attribution_results, f, indent=2)

if not attr_df.empty:
    attr_df.to_csv(RESULTS_DIR / "attribution_samples.csv", index=False)
    print(f"  Saved attribution_samples.csv")

print(f"  Saved attribution_results.json")
n_html = len(list(ATTR_DIR.glob("*.html")))
print(f"  {n_html} HTML heatmaps in {ATTR_DIR.name}/")
print("\nAttribution analysis complete.")