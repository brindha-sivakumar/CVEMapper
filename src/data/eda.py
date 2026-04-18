
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
import json

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "raw_cves.parquet"
OUT_DIR   = PROJECT_ROOT / "data" / "eda"
SPLIT_DIR = PROJECT_ROOT / "data" / "processed"

OUT_DIR.mkdir(parents=True, exist_ok=True)
SPLIT_DIR.mkdir(parents=True, exist_ok=True)

SEVERITY_COLORS = {
    "CRITICAL": "#DA3633",
    "HIGH":     "#E3B341",
    "MEDIUM":   "#58A6FF",
    "LOW":      "#3FB950"
}

print("Loading parquet")
df = pd.read_parquet(DATA_PATH)
df["published"] = pd.to_datetime(df["published"])
df["year"]      = df["published"].dt.year
df["token_len"] = df["description"].str.split().str.len()

print(f"  Loaded {len(df):,} rows")
print(f"  Columns: {list(df.columns)}")
print(f"  Date range: {df['published'].min().date()} → {df['published'].max().date()}")

print("\nSeverity distribution")

sev_counts = df["severity"].value_counts().reindex(["CRITICAL","HIGH","MEDIUM","LOW"])
sev_pct    = (sev_counts / len(df) * 100).round(1)

print(sev_counts.to_string())
print()
for sev, pct in sev_pct.items():
    print(f"  {sev:<10} {pct}%")

fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(
    sev_counts.index,
    sev_counts.values,
    color=[SEVERITY_COLORS[s] for s in sev_counts.index],
    width=0.55, edgecolor="none"
)
for bar, pct in zip(bars, sev_pct.values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 500,
        f"{pct}%", ha="center", va="bottom", fontsize=11
    )
ax.set_title("CVE severity distribution", fontsize=13)
ax.set_ylabel("Number of CVEs")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig(OUT_DIR / "01_severity_distribution.png", dpi=150)
plt.close()
print(f"  Saved 01_severity_distribution.png")


print("\nDescription length distribution")

print(df["token_len"].describe().round(1).to_string())

fig, ax = plt.subplots(figsize=(8, 4))
for sev, color in SEVERITY_COLORS.items():
    subset = df[df["severity"] == sev]["token_len"]
    ax.hist(subset, bins=60, alpha=0.55, color=color, label=sev, range=(0, 250))
ax.axvline(256, color="black", linestyle="--", linewidth=1, label="max_len=256 cutoff")
ax.set_title("Description length by severity (words)", fontsize=13)
ax.set_xlabel("Word count")
ax.set_ylabel("Number of CVEs")
ax.legend(fontsize=9)
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig(OUT_DIR / "02_description_length.png", dpi=150)
plt.close()
print(f"  Saved 02_description_length.png")
print(f"  CVEs exceeding 256 tokens: {(df['token_len'] > 256).sum():,} ({(df['token_len'] > 256).mean()*100:.1f}%)")


print("\nCVEs per year")

year_sev = (
    df.groupby(["year", "severity"])
    .size()
    .unstack(fill_value=0)
    .reindex(columns=["CRITICAL","HIGH","MEDIUM","LOW"])
)
print(year_sev.to_string())

fig, ax = plt.subplots(figsize=(10, 4))
bottom = pd.Series(0, index=year_sev.index)
for sev, color in SEVERITY_COLORS.items():
    ax.bar(year_sev.index, year_sev[sev], bottom=bottom,
           color=color, label=sev, width=0.7, edgecolor="none")
    bottom += year_sev[sev]
ax.axvline(2021.5, color="black", linestyle="--", linewidth=1.2, label="Train / Test split")
ax.set_title("CVEs published per year by severity", fontsize=13)
ax.set_ylabel("Number of CVEs")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.legend(fontsize=9, loc="upper left")
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig(OUT_DIR / "03_cves_per_year.png", dpi=150)
plt.close()
print(f"  Saved 03_cves_per_year.png")


print("\nCWE coverage")

cwe_counts = df[df["cwe"] != "UNKNOWN"]["cwe"].value_counts()
top20       = cwe_counts.head(20)
top20_pct   = (top20.sum() / len(df) * 100).round(1)
print(f"  Unique CWEs: {df['cwe'].nunique():,}")
print(f"  Top 20 CWEs cover {top20_pct}% of corpus")
print(top20.to_string())

fig, ax = plt.subplots(figsize=(8, 7))
ax.barh(top20.index[::-1], top20.values[::-1],
        color="#58A6FF", edgecolor="none")
ax.set_title("Top 20 CWE categories", fontsize=13)
ax.set_xlabel("Number of CVEs")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig(OUT_DIR / "04_cwe_coverage.png", dpi=150)
plt.close()
print(f"  Saved 04_cwe_coverage.png")


print("\nCVSS score distribution")

print(df["score"].describe().round(2).to_string())

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(df["score"], bins=40, color="#58A6FF", edgecolor="none", alpha=0.85)
for boundary, label in [(4.0,"LOW|MED"),(7.0,"MED|HIGH"),(9.0,"HIGH|CRIT")]:
    ax.axvline(boundary, color="#DA3633", linestyle="--", linewidth=1)
    ax.text(boundary + 0.05, ax.get_ylim()[1] * 0.92,
            label, fontsize=8, color="#DA3633")
ax.set_title("CVSS v3 base score distribution", fontsize=13)
ax.set_xlabel("Base score")
ax.set_ylabel("Number of CVEs")
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig(OUT_DIR / "05_score_distribution.png", dpi=150)
plt.close()
print(f"  Saved 05_score_distribution.png")


print("\n── Temporal split ───────────────────────────────────────────")

train_pool = df[df["year"] <= 2021].copy()
test       = df[df["year"] >= 2022].copy()

val        = train_pool.sample(frac=0.15, random_state=42)
train      = train_pool.drop(val.index)

print(f"  Train : {len(train):>7,} rows  ({len(train)/len(df)*100:.1f}%)  years ≤ 2021")
print(f"  Val   : {len(val):>7,} rows  ({len(val)/len(df)*100:.1f}%)  random 15% of train pool")
print(f"  Test  : {len(test):>7,} rows  ({len(test)/len(df)*100:.1f}%)  years ≥ 2022")
print()

for name, split in [("Train", train), ("Val", val), ("Test", test)]:
    dist = split["severity"].value_counts(normalize=True).reindex(
        ["CRITICAL","HIGH","MEDIUM","LOW"]
    ).mul(100).round(1)
    print(f"  {name} severity %:  " + "  ".join(f"{s}={v}%" for s, v in dist.items()))

# verify no leakage
assert train.index.intersection(val.index).empty,   "LEAKAGE: train/val overlap"
assert train.index.intersection(test.index).empty,  "LEAKAGE: train/test overlap"
assert val.index.intersection(test.index).empty,    "LEAKAGE: val/test overlap"
print("\n  Leakage check passed.")

train.drop(columns=["year","token_len"]).to_csv(SPLIT_DIR / "train.csv", index=False)
val.drop(columns=["year","token_len"]).to_csv(SPLIT_DIR / "val.csv",   index=False)
test.drop(columns=["year","token_len"]).to_csv(SPLIT_DIR / "test.csv",  index=False)

# save split summary
summary = {
    "total":  len(df),
    "train":  len(train),
    "val":    len(val),
    "test":   len(test),
    "train_years": f"≤ 2021",
    "test_years":  f"≥ 2022",
    "severity_distribution": {
        split: df[df.index.isin(split_df.index)]["severity"]
               .value_counts().to_dict()
        for split, split_df in [("train", train), ("val", val), ("test", test)]
    }
}
with open(SPLIT_DIR / "split_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n  Saved train.csv  ({len(train):,} rows)")
print(f"  Saved val.csv    ({len(val):,} rows)")
print(f"  Saved test.csv   ({len(test):,} rows)")
print(f"  Saved split_summary.json")
print("\nEDA and split complete.")