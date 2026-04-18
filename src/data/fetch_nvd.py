import requests
import time
import json
import pandas as pd
from pathlib import Path
import os

def preview(label: str, data, n: int = 3):
    """Print a labelled sample of data at any stage."""
    print(f"\n{'='*60}")
    print(f"  PREVIEW: {label}")
    print(f"{'='*60}")
    if isinstance(data, pd.DataFrame):
        print(f"  Shape: {data.shape}")
        print(f"  Columns: {list(data.columns)}")
        print()
        print(data.head(n).to_string(max_colwidth=80))
    elif isinstance(data, list):
        print(f"  Total items: {len(data):,}")
        print()
        for i, item in enumerate(data[:n]):
            print(f"  --- Entry {i+1} ---")
            print(json.dumps(item, indent=2, default=str)[:600])  # cap at 600 chars
            print()
    print(f"{'='*60}\n")

def fetch_all_cves(api_key: str) -> list:
    url = "https://services.nvd.nist.gov/rest/json/cves/2.0"
    headers = {"apiKey": api_key}
    all_cves, idx, total = [], 0, None

    print("Starting NVD fetch...")
    while total is None or idx < total:
        for attempt in range(5):
            try:
                r = requests.get(url,params={"startIndex": idx, "resultsPerPage": 2000},headers=headers)
                if r.status_code == 429:
                    wait = 30 * (attempt + 1)  # 30s, 60s, 90s, 120s, 150s
                    print(f"\n  429 received at index {idx}. Waiting {wait}s before retry {attempt + 1}/5...")
                    time.sleep(wait)
                    continue
                r.raise_for_status()
                break
            except requests.exceptions.Timeout:
                print(f"\n  Timeout at index {idx}. Retrying ({attempt + 1}/5)...")
                time.sleep(10)
        else:
            print(f"\n  Failed after 5 attempts at index {idx}. Saving progress...")
            break
        data = r.json()
        total = data["totalResults"]
        batch = data["vulnerabilities"]
        all_cves.extend(batch)
        if idx == 0:
            preview("RAW API RESPONSE — first batch (before any parsing)", batch, n=2)
        idx += 2000
        print(f"  Fetched {min(idx, total):,} / {total:,}", end="\r")
        time.sleep(1.2)

    print(f"\nDone. Total CVEs fetched: {len(all_cves):,}")
    return all_cves


def parse_to_dataframe(all_cves: list) -> pd.DataFrame:
    preview("STAGE 2A — raw CVE item structure (before field extraction)", all_cves, n=3)
    records = []
    skipped = 0
    for item in all_cves:
        cve = item["cve"]
        desc = next(
            (d["value"] for d in cve.get("descriptions", []) if d["lang"] == "en"),
            None
        )
        metrics = cve.get("metrics", {})
        cvss = metrics.get("cvssMetricV31", metrics.get("cvssMetricV30", []))
        if not cvss or not desc:
            skipped += 1
            continue
        records.append({
            "id": cve["id"],
            "description": desc,
            "score": cvss[0]["cvssData"]["baseScore"],
            "severity": cvss[0]["cvssData"]["baseSeverity"],
            "cwe": cve.get("weaknesses", [{}])[0].get("description", [{}])[0].get("value", "UNKNOWN"),
            "published": cve["published"][:10]
        })
    print(f"  Skipped {skipped:,} CVEs (missing CVSS v3 or English description)")

    preview("STAGE 2B — extracted record dicts (before DataFrame)", records, n=5)
    df = pd.DataFrame(records)
    preview("STAGE 2C — final DataFrame (after parsing)", df, n=5)

    print(f"\n  Severity distribution:")
    print(df["severity"].value_counts().to_string())
    print(f"\n  Date range: {df['published'].min()} → {df['published'].max()}")
    print(f"  Score range: {df['score'].min()} → {df['score'].max()}")
    print(f"  Unique CWEs: {df['cwe'].nunique():,}")
    print(f"Parsed {len(df):,} CVEs with valid CVSS v3 scores.")
    print("\n=== DATAFRAME SHAPE ===")
    print(df.shape)

    print("\n=== COLUMN TYPES ===")
    print(df.dtypes)

    print("\n=== FIRST 5 ROWS ===")
    print(df.head(5).to_string(max_colwidth=80))

    print("\n=== SEVERITY DISTRIBUTION ===")
    print(df["severity"].value_counts())

    print("\n=== DATE RANGE ===")
    print(df["published"].min(), "→", df["published"].max())

    print("\n=== SAMPLE — one row fully expanded ===")
    row = df.iloc[0].to_dict()
    for k, v in row.items():
        print(f"  {k:15} : {v}")

    print("\n=== SAMPLE — 3 CRITICAL entries ===")
    print(df[df["severity"] == "CRITICAL"].sample(3).to_string(max_colwidth=120))

    print("\n=== SAMPLE — 3 LOW entries ===")
    print(df[df["severity"] == "LOW"].sample(3).to_string(max_colwidth=120))

    return df


if __name__ == "__main__":

    API_KEY = os.getenv("NVD_API_KEY")
    # fetch
    all_cves = fetch_all_cves(api_key=API_KEY)

    # parse
    df = parse_to_dataframe(all_cves)

    # save
    out_path = Path("data/raw/raw_cves.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Saved to {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")
    print(df["severity"].value_counts())