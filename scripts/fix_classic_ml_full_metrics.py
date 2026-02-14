"""
One-time fix: recompute MACRO/MICRO rows in classic_ml_full_metrics.csv.

The original script used .ravel() before calling average="macro"/"micro",
which flattened the 2D (8044x13) arrays into 1D and made sklearn treat it
as a single binary classification problem — inflating metrics.

Per-category rows are correct. This script:
1. Reads per-category P/R/F1/AUC from the CSV
2. Derives TP/FP/FN per category using support from test_set.csv
3. Computes correct MACRO (mean of per-category metrics)
4. Computes correct MICRO (from global TP/FP/FN sums)
5. Overwrites the CSV and regenerates the summary .txt
"""

import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np


def main():
    metrics_path = "results/classic_ml_full_metrics.csv"
    test_path = "data/test_set.csv"
    summary_path = "results/classic_ml_full_summary.txt"

    # Load existing metrics (per-category rows are correct)
    df = pd.read_csv(metrics_path)
    cats = df[~df["Category"].isin(["MACRO", "MICRO"])].copy()

    print(f"Found {len(cats)} per-category rows")
    print(f"Old MACRO F1: {df[df['Category'] == 'MACRO']['F1'].values[0]:.4f}")
    print(f"Old MICRO F1: {df[df['Category'] == 'MICRO']['F1'].values[0]:.4f}")

    # Load test set to get per-category support counts
    test_df = pd.read_csv(test_path)
    categories = cats["Category"].tolist()

    # Derive TP, FP, FN per category from precision, recall, and support
    supports = {}
    tps = {}
    fps = {}
    fns = {}

    for _, row in cats.iterrows():
        cat = row["Category"]
        p = row["Precision"]
        r = row["Recall"]
        support = int(test_df[cat].sum())  # number of positive samples
        supports[cat] = support

        tp = r * support
        fp = (tp / p - tp) if p > 0 else 0
        fn = support - tp

        tps[cat] = tp
        fps[cat] = fp
        fns[cat] = fn

    # MACRO: simple average of per-category metrics
    macro_p = cats["Precision"].mean()
    macro_r = cats["Recall"].mean()
    macro_f1 = cats["F1"].mean()
    macro_auc = cats["ROC-AUC"].mean()

    # MICRO: from global TP/FP/FN sums
    total_tp = sum(tps.values())
    total_fp = sum(fps.values())
    total_fn = sum(fns.values())

    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0
    # For MICRO-AUC: use macro-average of per-category AUCs (no raw probabilities saved)
    micro_auc = macro_auc

    print(f"\nCorrected MACRO: P={macro_p:.4f} R={macro_r:.4f} F1={macro_f1:.4f} AUC={macro_auc:.4f}")
    print(f"Corrected MICRO: P={micro_p:.4f} R={micro_r:.4f} F1={micro_f1:.4f} AUC={micro_auc:.4f}")

    # Verification
    print(f"\nVerification: MACRO F1 = mean of per-category F1s = {cats['F1'].mean():.6f}")
    print(f"Global TP={total_tp:.1f}, FP={total_fp:.1f}, FN={total_fn:.1f}")

    # Build corrected rows
    macro_row = pd.DataFrame([{
        "Category": "MACRO",
        "Precision": macro_p,
        "Recall": macro_r,
        "F1": macro_f1,
        "ROC-AUC": macro_auc,
    }])
    micro_row = pd.DataFrame([{
        "Category": "MICRO",
        "Precision": micro_p,
        "Recall": micro_r,
        "F1": micro_f1,
        "ROC-AUC": micro_auc,
    }])

    corrected = pd.concat([cats, macro_row, micro_row], ignore_index=True)
    corrected.to_csv(metrics_path, index=False)
    print(f"\nOverwrote {metrics_path}")

    # Regenerate summary using format_summary from the source script
    metrics_rows = corrected.to_dict("records")
    n_train = 164_139
    n_test = 8_044

    from scripts.modal_classic_ml_full import format_summary
    summary = format_summary(metrics_rows, n_train, n_test)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"Regenerated {summary_path}")

    print(f"\nDone! MACRO-F1: {macro_f1:.4f}, MICRO-F1: {micro_f1:.4f}")


if __name__ == "__main__":
    main()
