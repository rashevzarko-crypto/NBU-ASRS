"""
Classic ML (TF-IDF + XGBoost) on the full 172K dataset — Modal GPU version.

Same hyperparams as the 32K baseline, but trained on ~164K reports
(all of asrs_multilabel.csv minus the frozen 8,044 test set).
Isolates the effect of training data scaling.

Usage:
    python -m modal run scripts/modal_classic_ml_full.py
"""

import os
import time
import json
import modal

app = modal.App("asrs-classic-ml-full")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("pandas", "numpy", "scikit-learn", "xgboost")
    .add_local_file("data/asrs_multilabel.csv", remote_path="/mnt/data/asrs_multilabel.csv")
    .add_local_file("data/test_set.csv", remote_path="/mnt/data/test_set.csv")
)


@app.function(image=image, cpu=32, memory=65536, timeout=10800)
def train_and_predict() -> dict:
    """Load data, TF-IDF, train 13 XGBoost classifiers on GPU, predict on test set."""
    import pandas as pd
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from xgboost import XGBClassifier

    t_start = time.time()

    # --- Load data ---
    full_df = pd.read_csv("/mnt/data/asrs_multilabel.csv")
    test_df = pd.read_csv("/mnt/data/test_set.csv")

    categories = [c for c in test_df.columns if c not in ("ACN", "Narrative")]
    text_col = "Narrative"

    test_acns = set(test_df["ACN"].values)
    train_df = full_df[~full_df["ACN"].isin(test_acns)].reset_index(drop=True)

    n_train = len(train_df)
    n_test = len(test_df)
    print(f"Training set: {n_train} reports (full dataset minus test)")
    print(f"Test set: {n_test} reports (frozen)")
    print(f"Categories: {len(categories)}")

    # --- TF-IDF (exact baseline params) ---
    print("\nFitting TF-IDF...")
    t_tfidf = time.time()
    tfidf = TfidfVectorizer(
        max_features=50_000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        dtype=np.float32,
    )
    X_train = tfidf.fit_transform(train_df[text_col].fillna(""))
    X_test = tfidf.transform(test_df[text_col].fillna(""))
    print(f"TF-IDF done in {time.time() - t_tfidf:.1f}s — "
          f"train: {X_train.shape}, test: {X_test.shape}")

    # --- Train 13 XGBoost classifiers ---
    y_pred = np.zeros((n_test, len(categories)), dtype=int)
    y_proba = np.zeros((n_test, len(categories)), dtype=float)
    timings = {}

    print(f"\nTraining {len(categories)} XGBoost classifiers (hist, 32 CPU cores)...")
    for i, col in enumerate(categories):
        t_clf = time.time()
        y_tr = train_df[col].values

        n_pos = int(y_tr.sum())
        n_neg = n_train - n_pos
        spw = n_neg / n_pos if n_pos > 0 else 1.0

        clf = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=spw,
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
            tree_method="hist",
            nthread=32,
        )
        clf.fit(X_train, y_tr)

        y_pred[:, i] = clf.predict(X_test)
        y_proba[:, i] = clf.predict_proba(X_test)[:, 1]

        elapsed = time.time() - t_clf
        timings[col] = elapsed
        print(f"  [{i+1:2d}/13] {col:<35s} "
              f"pos={n_pos:>6d} ({n_pos/n_train*100:5.1f}%) "
              f"spw={spw:6.2f}  {elapsed:6.1f}s")

    total_train = time.time() - t_start
    print(f"\nTotal time: {total_train:.1f}s ({total_train/60:.1f} min)")

    return {
        "y_pred": y_pred.tolist(),
        "y_proba": y_proba.tolist(),
        "categories": categories,
        "n_train": n_train,
        "timings": timings,
        "total_seconds": total_train,
    }


# ---------------------------------------------------------------------------
# Local helpers
# ---------------------------------------------------------------------------

def compute_metrics(y_true, y_pred, y_proba, categories):
    """Compute per-category and aggregate metrics — matches baseline exactly."""
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
    import numpy as np

    rows = []

    # Per-category
    for i, cat in enumerate(categories):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        ypr = y_proba[:, i]

        p = precision_score(yt, yp, zero_division=0)
        r = recall_score(yt, yp, zero_division=0)
        f1 = f1_score(yt, yp, zero_division=0)
        auc = roc_auc_score(yt, ypr) if len(set(yt)) > 1 else 0.5

        rows.append({"Category": cat, "Precision": p, "Recall": r, "F1": f1, "ROC-AUC": auc})

    # Macro / Micro on 2D arrays (n_samples × n_labels)
    macro = {
        "Category": "MACRO",
        "Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, y_proba, average="macro"),
    }

    micro = {
        "Category": "MICRO",
        "Precision": precision_score(y_true, y_pred, average="micro", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="micro", zero_division=0),
        "F1": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, y_proba, average="micro"),
    }

    rows.append(macro)
    rows.append(micro)
    return rows


def format_summary(metrics_rows, n_train, n_test):
    """Format a human-readable summary."""
    lines = [
        "Classic ML (Full Dataset): TF-IDF + XGBoost",
        "=" * 75,
        f"Train set: {n_train:,} reports (full 172K minus test) | Test set: {n_test:,} reports",
        "TF-IDF: max_features=50000, ngram_range=(1,2), sublinear_tf=True",
        "XGBoost: n_estimators=300, max_depth=6, lr=0.1, tree_method=hist",
        "scale_pos_weight=auto per category",
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


def print_comparison(full_rows):
    """Print comparison: baseline (32K) vs full (164K)."""
    import pandas as pd

    baseline_path = "results/classic_ml_text_metrics.csv"
    if not os.path.exists(baseline_path):
        print("Baseline metrics not found, skipping comparison.")
        return

    baseline = pd.read_csv(baseline_path)
    baseline_f1 = dict(zip(baseline["Category"], baseline["F1"]))
    baseline_auc = dict(zip(baseline["Category"], baseline["ROC-AUC"]))

    print("\n" + "=" * 95)
    print(f"{'Category':<35} {'Base F1':>9} {'Full F1':>9} {'Delta':>8} "
          f"{'Base AUC':>10} {'Full AUC':>10}")
    print("-" * 95)
    for row in full_rows:
        cat = row["Category"]
        f_full = row["F1"]
        a_full = row["ROC-AUC"]
        f_base = baseline_f1.get(cat, float("nan"))
        a_base = baseline_auc.get(cat, float("nan"))
        delta = f_full - f_base if f_base == f_base else float("nan")
        print(f"{cat:<35} {f_base:>9.4f} {f_full:>9.4f} {delta:>+8.4f} "
              f"{a_base:>10.4f} {a_full:>10.4f}")
    print("=" * 95)


# ---------------------------------------------------------------------------
# Main entrypoint (runs locally)
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main():
    """Run classic ML on full dataset via Modal GPU, compute metrics locally."""
    import pandas as pd
    import numpy as np

    print("Launching XGBoost training on Modal A100...")
    t0 = time.time()
    result = train_and_predict.remote()
    wall_time = time.time() - t0

    # Unpack
    y_pred = np.array(result["y_pred"], dtype=int)
    y_proba = np.array(result["y_proba"], dtype=float)
    categories = result["categories"]
    n_train = result["n_train"]
    total_gpu_seconds = result["total_seconds"]

    # Load test set for ground truth
    test_df = pd.read_csv("data/test_set.csv")
    n_test = len(test_df)
    y_true = test_df[categories].values

    print(f"\nRemote completed: {n_train:,} train, {n_test:,} test")
    print(f"GPU time: {total_gpu_seconds:.1f}s ({total_gpu_seconds/60:.1f} min)")
    print(f"Wall-clock: {wall_time:.1f}s ({wall_time/60:.1f} min)")
    # Modal CPU pricing: 32 cores ~ $1.28/hr (32 × $0.04/core/hr)
    print(f"Estimated cost: ${wall_time / 3600 * 1.28:.2f} (32-core CPU @ ~$1.28/hr)")

    # Compute metrics
    print("\nComputing metrics...")
    metrics_rows = compute_metrics(y_true, y_pred, y_proba, categories)

    # Save metrics CSV
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = "results/classic_ml_full_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved {metrics_path}")

    # Save summary
    summary = format_summary(metrics_rows, n_train, n_test)
    summary_path = "results/classic_ml_full_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"Saved {summary_path}")

    # Print results
    print(f"\n{summary}")

    # Print comparison
    print_comparison(metrics_rows)

    # Per-classifier timing
    print("\nPer-classifier GPU timing:")
    for cat, t in result["timings"].items():
        print(f"  {cat:<35s} {t:6.1f}s")
