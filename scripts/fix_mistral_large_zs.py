"""Fix Mistral Large 3 zero-shot parse failures and recompute metrics.

The original _normalize() did exact string matching against top-level category names,
but the model returned subcategory format (e.g., "Aircraft Equipment Problem: Less Severe").
Items with colons silently failed, causing 43% parse failures (3,458/8,044).

This script:
1. Reads the existing raw outputs CSV (with llm_raw_output column)
2. Re-parses all outputs with a fixed _normalize() that strips ": subcategory" suffixes
3. Saves corrected metrics, raw outputs, and summary files
4. Also prints metrics for ONLY the 4,586 originally-parsed reports (for comparison)

No GPU or API calls needed — pure local CSV processing.
"""

import json
import os
import re

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

RESULTS_DIR = "results"


# ---------------------------------------------------------------------------
# Fixed parsing helpers
# ---------------------------------------------------------------------------

def _normalize(items: list, cat_lower: dict[str, str]) -> list[str]:
    """Map parsed items to exact category names via case-insensitive match.

    Fixed version: strips ": subcategory" suffixes before matching, so
    "Aircraft Equipment Problem: Less Severe" maps to "Aircraft Equipment Problem".
    """
    result = []
    seen = set()
    for item in items:
        if not isinstance(item, str):
            continue
        key = item.strip().lower()
        # Exact match first
        if key in cat_lower:
            cat = cat_lower[key]
            if cat not in seen:
                result.append(cat)
                seen.add(cat)
            continue
        # Strip subcategory suffix (e.g., "Aircraft Equipment Problem: Less Severe")
        if ":" in key:
            prefix = key.split(":")[0].strip()
            if prefix in cat_lower:
                cat = cat_lower[prefix]
                if cat not in seen:
                    result.append(cat)
                    seen.add(cat)
    return result


def parse_llm_output(raw: str, categories: list[str]) -> list[str]:
    """Parse LLM output into a list of valid category names.

    Three-tier strategy:
      1. Direct JSON parse (with code fence stripping)
      2. Regex extraction of JSON array from surrounding text
      3. Fuzzy substring matching of category names
    """
    cat_lower = {c.lower(): c for c in categories}

    # Strip markdown code fences if present
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    # Tier 1: direct JSON parse
    try:
        parsed = json.loads(cleaned.strip())
        if isinstance(parsed, list):
            result = _normalize(parsed, cat_lower)
            if result:
                return result
    except (json.JSONDecodeError, TypeError):
        pass

    # Tier 2: regex — find first [...] block
    m = re.search(r"\[.*?\]", raw, re.DOTALL)
    if m:
        try:
            parsed = json.loads(m.group())
            if isinstance(parsed, list):
                result = _normalize(parsed, cat_lower)
                if result:
                    return result
        except (json.JSONDecodeError, TypeError):
            pass

    # Tier 3: fuzzy substring matching
    matched = []
    raw_lower = raw.lower()
    for cat in categories:
        if cat.lower() in raw_lower:
            matched.append(cat)
    return matched


# ---------------------------------------------------------------------------
# Metrics (same as original)
# ---------------------------------------------------------------------------

def compute_metrics(y_true, y_pred, categories):
    """Compute per-category and aggregate metrics matching classic ML format."""
    rows = []
    for i, cat in enumerate(categories):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        p = precision_score(yt, yp, zero_division=0)
        r = recall_score(yt, yp, zero_division=0)
        f1 = f1_score(yt, yp, zero_division=0)

        if len(set(yt)) < 2 or len(set(yp)) < 2:
            auc = 0.5
        else:
            try:
                auc = roc_auc_score(yt, yp)
            except ValueError:
                auc = 0.5

        rows.append({
            "Category": cat,
            "Precision": p,
            "Recall": r,
            "F1": f1,
            "ROC-AUC": auc,
        })

    macro = {
        "Category": "MACRO",
        "Precision": np.mean([r["Precision"] for r in rows]),
        "Recall": np.mean([r["Recall"] for r in rows]),
        "F1": np.mean([r["F1"] for r in rows]),
        "ROC-AUC": np.mean([r["ROC-AUC"] for r in rows]),
    }

    y_true_flat = y_true.ravel()
    y_pred_flat = y_pred.ravel()
    micro = {
        "Category": "MICRO",
        "Precision": precision_score(y_true_flat, y_pred_flat, zero_division=0),
        "Recall": recall_score(y_true_flat, y_pred_flat, zero_division=0),
        "F1": f1_score(y_true_flat, y_pred_flat, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true_flat, y_pred_flat)
            if len(set(y_true_flat)) > 1 and len(set(y_pred_flat)) > 1
            else 0.5,
    }

    rows.append(macro)
    rows.append(micro)
    return rows


def format_summary(metrics_rows, n_test, parse_failures):
    """Format a human-readable summary."""
    fail_rate = parse_failures / n_test * 100
    lines = [
        "Zero-Shot LLM: Mistral Large 3 (Batch API, taxonomy-enriched prompt)",
        "=" * 70,
        f"Test set: {n_test} reports | Model: mistral-large-latest",
        "Batch API: temperature=0.0, max_tokens=256",
        "Zero-shot: taxonomy-enriched prompt with NASA ASRS subcategories, no examples",
        f"Parse failures: {parse_failures}/{n_test} ({fail_rate:.1f}%)",
        "",
        f"{'Category':<35} {'Precision':>10} {'Recall':>10} {'F1':>10} {'ROC-AUC':>10}",
        "-" * 75,
    ]
    for row in metrics_rows:
        cat = row["Category"]
        if cat in ("MACRO", "MICRO"):
            continue
        lines.append(
            f"{cat:<35} {row['Precision']:>10.4f} {row['Recall']:>10.4f} "
            f"{row['F1']:>10.4f} {row['ROC-AUC']:>10.4f}"
        )
    lines.append("-" * 75)
    for label in ("MACRO", "MICRO"):
        row = next(r for r in metrics_rows if r["Category"] == label)
        lines.append(
            f"{label:<35} {row['Precision']:>10.4f} {row['Recall']:>10.4f} "
            f"{row['F1']:>10.4f} {row['ROC-AUC']:>10.4f}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Load existing raw outputs
    raw_path = os.path.join(RESULTS_DIR, "mistral_large_zs_raw_outputs.csv")
    print(f"Loading {raw_path}...")
    raw_df = pd.read_csv(raw_path)
    n_test = len(raw_df)
    print(f"  {n_test} rows")

    # Load test set for ground truth
    test_df = pd.read_csv("data/test_set.csv")
    categories = [c for c in test_df.columns if c not in ("ACN", "Narrative")]
    print(f"  {len(categories)} categories: {categories[:3]}...")

    y_true = test_df[categories].values

    # Count old parse failures for comparison
    old_failures = sum(1 for x in raw_df["parsed_labels"] if x == "[]")
    print(f"\nOLD parse failures: {old_failures}/{n_test} ({old_failures/n_test*100:.1f}%)")

    # Re-parse all raw outputs with fixed _normalize
    print("\nRe-parsing with fixed _normalize (colon-prefix stripping)...")
    all_raw = raw_df["llm_raw_output"].fillna("").tolist()
    parsed_labels = []
    parse_failures = 0

    for raw in all_raw:
        result = parse_llm_output(raw, categories)
        if not result:
            parse_failures += 1
        parsed_labels.append(result)

    fail_rate = parse_failures / n_test * 100
    print(f"NEW parse failures: {parse_failures}/{n_test} ({fail_rate:.1f}%)")
    print(f"  Fixed: {old_failures - parse_failures} reports")

    # Build prediction matrix
    y_pred = np.zeros((n_test, len(categories)), dtype=int)
    for i, labels in enumerate(parsed_labels):
        for cat in labels:
            if cat in categories:
                y_pred[i, categories.index(cat)] = 1

    # Compute full metrics
    print("\n" + "=" * 70)
    print("CORRECTED METRICS (all 8,044 reports)")
    print("=" * 70)
    metrics_rows = compute_metrics(y_true, y_pred, categories)

    for row in metrics_rows:
        print(f"  {row['Category']:<35} P={row['Precision']:.4f}  R={row['Recall']:.4f}  "
              f"F1={row['F1']:.4f}  AUC={row['ROC-AUC']:.4f}")

    # Also compute metrics for only the originally-parsed subset
    old_parsed = raw_df["parsed_labels"].tolist()
    old_mask = [x != "[]" for x in old_parsed]
    n_old_parsed = sum(old_mask)
    print(f"\n{'=' * 70}")
    print(f"SUBSET METRICS (only {n_old_parsed} originally-parsed reports)")
    print("=" * 70)

    old_indices = [i for i, m in enumerate(old_mask) if m]
    y_true_sub = y_true[old_indices]
    y_pred_sub = y_pred[old_indices]
    subset_rows = compute_metrics(y_true_sub, y_pred_sub, categories)

    for row in subset_rows:
        if row["Category"] in ("MACRO", "MICRO"):
            print(f"  {row['Category']:<35} P={row['Precision']:.4f}  R={row['Recall']:.4f}  "
                  f"F1={row['F1']:.4f}  AUC={row['ROC-AUC']:.4f}")

    # Save corrected metrics CSV
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = os.path.join(RESULTS_DIR, "mistral_large_zs_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nSaved {metrics_path}")

    # Save corrected raw outputs CSV
    raw_df["parsed_labels"] = [json.dumps(l) for l in parsed_labels]
    for cat in categories:
        raw_df[f"pred_{cat}"] = y_pred[:, categories.index(cat)]
    raw_df.to_csv(raw_path, index=False)
    print(f"Saved {raw_path}")

    # Save corrected summary
    summary = format_summary(metrics_rows, n_test, parse_failures)
    summary_path = os.path.join(RESULTS_DIR, "mistral_large_zs_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"Saved {summary_path}")

    # Print comparison with few-shot and classic ML
    print("\n" + "=" * 95)
    print(f"{'Category':<35} {'Classic ML':>12} {'ML3 FS':>12} {'ML3 ZS':>12} {'ZS-FS':>10}")
    print("-" * 95)

    classic_path = os.path.join(RESULTS_DIR, "classic_ml_text_metrics.csv")
    fs_path = os.path.join(RESULTS_DIR, "mistral_large_metrics.csv")
    classic_f1 = {}
    fs_f1 = {}
    if os.path.exists(classic_path):
        classic = pd.read_csv(classic_path)
        classic_f1 = dict(zip(classic["Category"], classic["F1"]))
    if os.path.exists(fs_path):
        fs = pd.read_csv(fs_path)
        fs_f1 = dict(zip(fs["Category"], fs["F1"]))

    for row in metrics_rows:
        cat = row["Category"]
        zf1 = row["F1"]
        cf1 = classic_f1.get(cat, float("nan"))
        ff1 = fs_f1.get(cat, float("nan"))
        delta = zf1 - ff1 if not (ff1 != ff1) else float("nan")
        print(f"{cat:<35} {cf1:>12.4f} {ff1:>12.4f} {zf1:>12.4f} {delta:>+10.4f}")
    print("=" * 95)

    print("\nDone! All corrected files saved.")


if __name__ == "__main__":
    main()
