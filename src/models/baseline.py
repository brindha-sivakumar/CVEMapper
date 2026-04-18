import pandas as pd
import numpy as np
import json
import time
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score,
    classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

SPLIT_DIR  = Path("../../data/processed")
RESULTS_DIR = Path("../../results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LABEL_ORDER = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
SEVERITY_COLORS = {
    "CRITICAL": "#DA3633",
    "HIGH":     "#E3B341",
    "MEDIUM":   "#58A6FF",
    "LOW":      "#3FB950"
}

print("Loading splits")
train = pd.read_csv(SPLIT_DIR / "train.csv")
val   = pd.read_csv(SPLIT_DIR / "val.csv")
test  = pd.read_csv(SPLIT_DIR / "test.csv")

print(f"  Train : {len(train):,}")
print(f"  Val   : {len(val):,}")
print(f"  Test  : {len(test):,}")

X_train, y_train = train["description"], train["severity"]
X_val,   y_val   = val["description"],   val["severity"]
X_test,  y_test  = test["description"],  test["severity"]


print("\nFitting TF-IDF vectorizer")
tfidf = TfidfVectorizer(
    max_features=50_000,
    ngram_range=(1, 2),       # unigrams + bigrams
    sublinear_tf=True,        # log(1+tf) dampening
    min_df=3,                 # ignore very rare terms
    strip_accents="unicode",
    analyzer="word"
)
t0 = time.time()
X_train_tfidf = tfidf.fit_transform(X_train)
X_val_tfidf   = tfidf.transform(X_val)
X_test_tfidf  = tfidf.transform(X_test)
print(f"  Vocab size : {len(tfidf.vocabulary_):,} features")
print(f"  Matrix     : {X_train_tfidf.shape}  (sparse)")
print(f"  Time       : {time.time()-t0:.1f}s")


def evaluate(name: str, model, X_val, y_val, X_test, y_test) -> dict:
    results = {}
    for split_name, X, y_true in [("val", X_val, y_val), ("test", X_test, y_test)]:
        t0   = time.time()
        preds = model.predict(X)
        inf_time = time.time() - t0

        acc        = accuracy_score(y_true, preds)
        f1_weighted = f1_score(y_true, preds, average="weighted", labels=LABEL_ORDER)
        f1_macro    = f1_score(y_true, preds, average="macro",    labels=LABEL_ORDER)
        f1_per_class = f1_score(y_true, preds, average=None,      labels=LABEL_ORDER)

        results[split_name] = {
            "accuracy":    round(acc, 4),
            "f1_weighted": round(f1_weighted, 4),
            "f1_macro":    round(f1_macro, 4),
            "f1_per_class": {
                label: round(score, 4)
                for label, score in zip(LABEL_ORDER, f1_per_class)
            },
            "inference_time_s": round(inf_time, 3)
        }

        print(f"\n  [{name}] {split_name.upper()} RESULTS")
        print(f"  {'Accuracy':<20} {acc:.4f}")
        print(f"  {'F1-weighted':<20} {f1_weighted:.4f}")
        print(f"  {'F1-macro':<20} {f1_macro:.4f}")
        print(f"  {'Inference time':<20} {inf_time:.2f}s ({len(y_true)/inf_time:,.0f} samples/sec)")
        print()
        print(classification_report(y_true, preds, labels=LABEL_ORDER, digits=4))

    return results


def plot_confusion(name: str, model, X, y_true, split: str):
    preds = model.predict(X)
    cm    = confusion_matrix(y_true, preds, labels=LABEL_ORDER)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f",
        xticklabels=LABEL_ORDER, yticklabels=LABEL_ORDER,
        cmap="Blues", ax=ax, linewidths=0.5,
        annot_kws={"size": 11}
    )
    ax.set_title(f"{name} — confusion matrix ({split})", fontsize=12)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    fname = RESULTS_DIR / f"confusion_{name.lower().replace(' ','_')}_{split}.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  Saved {fname.name}")


print("\n" + "="*60)
print("  MODEL 1: TF-IDF + Logistic Regression")
print("="*60)

t0 = time.time()
lr = LogisticRegression(
    C=5.0,
    max_iter=1000,
    class_weight="balanced",   # handles LOW underrepresentation
    solver="saga",             # fast for large sparse matrices
    n_jobs=-1,
    random_state=42
)
lr.fit(X_train_tfidf, y_train)
print(f"  Training time: {time.time()-t0:.1f}s")

lr_results = evaluate("LogReg", lr, X_val_tfidf, y_val, X_test_tfidf, y_test)
plot_confusion("LogReg", lr, X_val_tfidf, y_val, "val")
plot_confusion("LogReg", lr, X_test_tfidf, y_test, "test")

# top discriminative terms per class
print("\n  Top 15 TF-IDF terms per class (LogReg coefficients):")
feature_names = tfidf.get_feature_names_out()
for i, label in enumerate(lr.classes_):
    top_idx = np.argsort(lr.coef_[i])[-15:][::-1]
    top_terms = ", ".join(feature_names[top_idx])
    print(f"  {label:<10} {top_terms}")

print("\n" + "="*60)
print("  MODEL 2: TF-IDF + Random Forest")
print("="*60)

t0 = time.time()
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    class_weight="balanced_subsample",
    n_jobs=-1,
    random_state=42
)
rf.fit(X_train_tfidf, y_train)
print(f"  Training time: {time.time()-t0:.1f}s")

rf_results = evaluate("RandomForest", rf, X_val_tfidf, y_val, X_test_tfidf, y_test)
plot_confusion("RandomForest", rf, X_val_tfidf, y_val, "val")
plot_confusion("RandomForest", rf, X_test_tfidf, y_test, "test")


print("\n" + "="*60)
print("  BASELINE SUMMARY — VAL SET")
print("="*60)
print(f"  {'Model':<25} {'Accuracy':>10} {'F1-Weighted':>12} {'F1-Macro':>10}")
print(f"  {'-'*57}")
for name, res in [("TF-IDF + LogReg", lr_results), ("TF-IDF + RandomForest", rf_results)]:
    v = res["val"]
    print(f"  {name:<25} {v['accuracy']:>10.4f} {v['f1_weighted']:>12.4f} {v['f1_macro']:>10.4f}")

print("\n" + "="*60)
print("  BASELINE SUMMARY — TEST SET")
print("="*60)
print(f"  {'Model':<25} {'Accuracy':>10} {'F1-Weighted':>12} {'F1-Macro':>10}")
print(f"  {'-'*57}")
for name, res in [("TF-IDF + LogReg", lr_results), ("TF-IDF + RandomForest", rf_results)]:
    t = res["test"]
    print(f"  {name:<25} {t['accuracy']:>10.4f} {t['f1_weighted']:>12.4f} {t['f1_macro']:>10.4f}")


fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
models = [("TF-IDF + LogReg", lr_results), ("TF-IDF + Random Forest", rf_results)]

for ax, (name, res) in zip(axes, models):
    f1s = [res["val"]["f1_per_class"][lbl] for lbl in LABEL_ORDER]
    bars = ax.bar(
        LABEL_ORDER, f1s,
        color=[SEVERITY_COLORS[l] for l in LABEL_ORDER],
        width=0.55, edgecolor="none"
    )
    for bar, f1 in zip(bars, f1s):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{f1:.3f}", ha="center", va="bottom", fontsize=10
        )
    ax.set_title(name, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1 score")
    ax.spines[["top", "right"]].set_visible(False)

plt.suptitle("Per-class F1 on validation set", fontsize=13)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "baseline_f1_per_class.png", dpi=150)
plt.close()
print(f"\n  Saved baseline_f1_per_class.png")


baseline_results = {
    "TF-IDF + LogReg":        lr_results,
    "TF-IDF + RandomForest":  rf_results,
}
with open(RESULTS_DIR / "baseline_results.json", "w") as f:
    json.dump(baseline_results, f, indent=2)

print(f"  Saved baseline_results.json")
print("\nBaseline training complete.")