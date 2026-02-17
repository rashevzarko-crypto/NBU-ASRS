"""
Classic ML Phase 3 — Final evaluation with baseline XGBoost params.

Retrains baseline XGBoost (n_estimators=300, max_depth=6, lr=0.1) on full
train set with original TF-IDF (50K features, bigrams, sublinear_tf=True).
Evaluates on parent (13-label) and subcategory (48-label) test sets.
Saves all results to Modal Volume for retrieval.

Usage:
    python -m modal run --detach scripts/modal_classic_ml_phase3.py
    python -m modal run scripts/modal_classic_ml_phase3.py::download_results

Expected runtime: ~45-60 min on 32 CPUs (~$1.00-1.30)
"""

import json
import os
import time
import modal

app = modal.App("asrs-classic-ml-phase3")

vol = modal.Volume.from_name("asrs-phase3-results", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("pandas", "numpy", "scikit-learn", "xgboost")
    .add_local_file("data/train_set.csv", remote_path="/mnt/data/train_set.csv")
    .add_local_file("data/test_set.csv", remote_path="/mnt/data/test_set.csv")
    .add_local_file(
        "data/subcategory_train_set.csv",
        remote_path="/mnt/data/subcategory_train_set.csv",
    )
    .add_local_file(
        "data/subcategory_test_set.csv",
        remote_path="/mnt/data/subcategory_test_set.csv",
    )
)

# Original baseline params (confirmed near-optimal by 16-combo search)
TFIDF_PARAMS = {
    "max_features": 50000,
    "ngram_range": (1, 2),
    "sublinear_tf": True,
    "min_df": 1,
}
XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.1,
    "tree_method": "hist",
}


@app.function(image=image, cpu=32, memory=65536, timeout=14400,
              volumes={"/vol": vol})
def run_phase3():
    """Retrain baseline XGBoost on full train, evaluate on parent + subcategory."""
    import numpy as np
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import (
        f1_score, precision_score, recall_score, roc_auc_score,
    )
    from xgboost import XGBClassifier
    import sys
    sys.stdout.reconfigure(line_buffering=True)

    def compute_metrics(y_true, y_pred, y_proba, categories):
        rows = []
        for i, cat in enumerate(categories):
            yt, yp, ypr = y_true[:, i], y_pred[:, i], y_proba[:, i]
            rows.append({
                "Category": cat,
                "Precision": precision_score(yt, yp, zero_division=0),
                "Recall": recall_score(yt, yp, zero_division=0),
                "F1": f1_score(yt, yp, zero_division=0),
                "ROC-AUC": roc_auc_score(yt, ypr) if len(set(yt)) > 1 else 0.5,
            })
        rows.append({
            "Category": "MACRO",
            "Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "ROC-AUC": roc_auc_score(y_true, y_proba, average="macro"),
        })
        rows.append({
            "Category": "MICRO",
            "Precision": precision_score(y_true, y_pred, average="micro", zero_division=0),
            "Recall": recall_score(y_true, y_pred, average="micro", zero_division=0),
            "F1": f1_score(y_true, y_pred, average="micro", zero_division=0),
            "ROC-AUC": roc_auc_score(y_true, y_proba, average="micro"),
        })
        return rows

    def train_and_eval(X_train, y_train, X_test, y_test, categories, label):
        n_train, n_test = X_train.shape[0], X_test.shape[0]
        n_labels = len(categories)
        y_pred = np.zeros((n_test, n_labels), dtype=int)
        y_proba = np.zeros((n_test, n_labels), dtype=float)

        t0 = time.time()
        for i, cat in enumerate(categories):
            yt = y_train[:, i]
            n_pos = int(yt.sum())
            n_neg = n_train - n_pos
            spw = n_neg / n_pos if n_pos > 0 else 1.0

            clf = XGBClassifier(
                scale_pos_weight=spw, eval_metric="logloss",
                random_state=42, verbosity=0, nthread=32,
                **XGB_PARAMS,
            )
            clf.fit(X_train, yt)
            y_pred[:, i] = clf.predict(X_test)
            y_proba[:, i] = clf.predict_proba(X_test)[:, 1]

            elapsed = time.time() - t0
            print(f"  [{i+1:2d}/{n_labels}] {cat:<55s} "
                  f"pos={n_pos:>6d} ({n_pos/n_train*100:5.1f}%) {elapsed:6.1f}s",
                  flush=True)

        total = time.time() - t0
        print(f"  {label} training: {total:.0f}s ({total/60:.1f} min)", flush=True)

        metrics = compute_metrics(y_test, y_pred, y_proba, categories)
        return metrics, y_pred, y_proba

    t_start = time.time()

    # ===================================================================
    # Parent (13-label)
    # ===================================================================
    print("=" * 70, flush=True)
    print("PARENT (13-label) — Baseline XGBoost on full train set", flush=True)
    print("=" * 70, flush=True)

    train_df = pd.read_csv("/mnt/data/train_set.csv")
    test_df = pd.read_csv("/mnt/data/test_set.csv")
    parent_cats = [c for c in train_df.columns if c not in ("ACN", "Narrative")]

    print(f"Data: {len(train_df)} train, {len(test_df)} test, {len(parent_cats)} labels",
          flush=True)
    print(f"TF-IDF: {TFIDF_PARAMS}", flush=True)
    print(f"XGBoost: {XGB_PARAMS}", flush=True)

    tfidf = TfidfVectorizer(dtype=np.float32, **TFIDF_PARAMS)
    X_train = tfidf.fit_transform(train_df["Narrative"].fillna(""))
    X_test = tfidf.transform(test_df["Narrative"].fillna(""))
    y_train = train_df[parent_cats].values
    y_test = test_df[parent_cats].values

    print(f"TF-IDF matrix: {X_train.shape}", flush=True)

    parent_metrics, parent_pred, parent_proba = train_and_eval(
        X_train, y_train, X_test, y_test, parent_cats, "Parent"
    )

    pm = next(r for r in parent_metrics if r["Category"] == "MACRO")
    pmi = next(r for r in parent_metrics if r["Category"] == "MICRO")
    print(f"\n>>> Parent: Macro-F1={pm['F1']:.4f}, Micro-F1={pmi['F1']:.4f}, "
          f"Macro-AUC={pm['ROC-AUC']:.4f} <<<", flush=True)

    # ===================================================================
    # Subcategory (48-label)
    # ===================================================================
    print(f"\n{'=' * 70}", flush=True)
    print("SUBCATEGORY (48-label) — Baseline XGBoost on full train set", flush=True)
    print("=" * 70, flush=True)

    sub_train_df = pd.read_csv("/mnt/data/subcategory_train_set.csv")
    sub_test_df = pd.read_csv("/mnt/data/subcategory_test_set.csv")
    sub_cats = [c for c in sub_train_df.columns if c not in ("ACN", "Narrative")]

    print(f"Data: {len(sub_train_df)} train, {len(sub_test_df)} test, {len(sub_cats)} labels",
          flush=True)

    sub_tfidf = TfidfVectorizer(dtype=np.float32, **TFIDF_PARAMS)
    X_sub_train = sub_tfidf.fit_transform(sub_train_df["Narrative"].fillna(""))
    X_sub_test = sub_tfidf.transform(sub_test_df["Narrative"].fillna(""))
    y_sub_train = sub_train_df[sub_cats].values
    y_sub_test = sub_test_df[sub_cats].values

    print(f"TF-IDF matrix: {X_sub_train.shape}", flush=True)

    sub_metrics, sub_pred, sub_proba = train_and_eval(
        X_sub_train, y_sub_train, X_sub_test, y_sub_test, sub_cats, "Subcategory"
    )

    sm = next(r for r in sub_metrics if r["Category"] == "MACRO")
    smi = next(r for r in sub_metrics if r["Category"] == "MICRO")
    print(f"\n>>> Subcategory: Macro-F1={sm['F1']:.4f}, Micro-F1={smi['F1']:.4f}, "
          f"Macro-AUC={sm['ROC-AUC']:.4f} <<<", flush=True)

    total_time = time.time() - t_start
    print(f"\nTotal time: {total_time:.0f}s ({total_time/60:.1f} min)", flush=True)

    # ===================================================================
    # Save results to Volume
    # ===================================================================
    print("\nSaving results to Volume...", flush=True)

    # Parent metrics CSV
    pd.DataFrame(parent_metrics).to_csv(
        "/vol/classic_ml_tuned_parent_metrics.csv", index=False)

    # Subcategory metrics CSV
    pd.DataFrame(sub_metrics).to_csv(
        "/vol/classic_ml_tuned_subcategory_metrics.csv", index=False)

    # Parent summary
    parent_summary = _format_summary(
        parent_metrics, len(train_df), len(test_df), parent_cats,
        "13 Parent Labels", 35, 77, 0.691, 0.746,
    )
    with open("/vol/classic_ml_tuned_parent_summary.txt", "w") as f:
        f.write(parent_summary)

    # Subcategory summary
    sub_summary = _format_summary(
        sub_metrics, len(sub_train_df), len(sub_test_df), sub_cats,
        "48 Subcategory Labels", 55, 97, 0.510, 0.600,
    )
    with open("/vol/classic_ml_tuned_subcategory_summary.txt", "w") as f:
        f.write(sub_summary)

    # Full result JSON
    result = {
        "tfidf_params": {k: str(v) for k, v in TFIDF_PARAMS.items()},
        "xgb_params": {k: str(v) for k, v in XGB_PARAMS.items()},
        "parent_metrics": parent_metrics,
        "parent_n_train": len(train_df),
        "parent_n_test": len(test_df),
        "parent_cats": parent_cats,
        "sub_metrics": sub_metrics,
        "sub_n_train": len(sub_train_df),
        "sub_n_test": len(sub_test_df),
        "sub_cats": sub_cats,
        "total_time_s": total_time,
    }
    with open("/vol/phase3_result.json", "w") as f:
        json.dump(result, f, indent=2)

    vol.commit()
    print("All results saved to volume 'asrs-phase3-results'", flush=True)
    print(f"\n{parent_summary}", flush=True)
    print(f"\n{sub_summary}", flush=True)


def _format_summary(metrics_rows, n_train, n_test, categories,
                    label_desc, col_width, line_width,
                    baseline_macro, baseline_micro):
    macro = next(r for r in metrics_rows if r["Category"] == "MACRO")
    micro = next(r for r in metrics_rows if r["Category"] == "MICRO")
    tfidf_str = "50K features, bigrams (1,2), sublinear_tf=True"
    xgb_str = "n_estimators=300, max_depth=6, lr=0.1, tree_method=hist"

    lines = [
        f"Classic ML Tuned: TF-IDF + XGBoost ({label_desc})",
        "=" * line_width,
        f"Train set: {n_train:,} reports | Test set: {n_test:,} reports",
        f"TF-IDF: {tfidf_str}",
        f"XGBoost: {xgb_str} + per-label scale_pos_weight",
        "",
        f"{'Category':<{col_width}} {'Precision':>10} {'Recall':>10} {'F1':>10} {'ROC-AUC':>10}",
        "-" * line_width,
    ]
    for row in metrics_rows:
        if row["Category"] in ("MACRO", "MICRO"):
            continue
        lines.append(
            f"{row['Category']:<{col_width}} {row['Precision']:>10.4f} "
            f"{row['Recall']:>10.4f} {row['F1']:>10.4f} {row['ROC-AUC']:>10.4f}"
        )
    lines.append("-" * line_width)
    for label in ("MACRO", "MICRO"):
        row = next(r for r in metrics_rows if r["Category"] == label)
        lines.append(
            f"{label:<{col_width}} {row['Precision']:>10.4f} "
            f"{row['Recall']:>10.4f} {row['F1']:>10.4f} {row['ROC-AUC']:>10.4f}"
        )
    lines.append("")
    lines.append(f"Comparison to original baseline:")
    lines.append("=" * line_width)
    lines.append(f"  Baseline Macro-F1: {baseline_macro:.4f}  |  "
                 f"Tuned: {macro['F1']:.4f}  |  Delta: {macro['F1'] - baseline_macro:+.4f}")
    lines.append(f"  Baseline Micro-F1: {baseline_micro:.4f}  |  "
                 f"Tuned: {micro['F1']:.4f}  |  Delta: {micro['F1'] - baseline_micro:+.4f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Download results from Volume
# ---------------------------------------------------------------------------

@app.local_entrypoint(name="download_results")
def download_results():
    """Download results from Volume after detached run completes.

    Usage: python -m modal run scripts/modal_classic_ml_phase3.py::download_results
    """
    result_json = read_volume_file.remote("/vol/phase3_result.json")
    if not result_json:
        print("No results found in volume yet. Job may still be running.")
        return

    result = json.loads(result_json)
    print(f"Found results (total time: {result['total_time_s']/60:.1f} min)")

    files = [
        ("classic_ml_tuned_parent_metrics.csv", "results/classic_ml_tuned_parent_metrics.csv"),
        ("classic_ml_tuned_subcategory_metrics.csv", "results/classic_ml_tuned_subcategory_metrics.csv"),
        ("classic_ml_tuned_parent_summary.txt", "results/classic_ml_tuned_parent_summary.txt"),
        ("classic_ml_tuned_subcategory_summary.txt", "results/classic_ml_tuned_subcategory_summary.txt"),
    ]
    for vol_name, local_path in files:
        content = read_volume_file.remote(f"/vol/{vol_name}")
        if content:
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            with open(local_path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"Saved {local_path}")

    # Also save the full result JSON
    with open("results/classic_ml_tuned_result.json", "w", encoding="utf-8") as f:
        f.write(result_json)
    print("Saved results/classic_ml_tuned_result.json")

    # Print summaries
    for summary_file in ["classic_ml_tuned_parent_summary.txt",
                         "classic_ml_tuned_subcategory_summary.txt"]:
        content = read_volume_file.remote(f"/vol/{summary_file}")
        if content:
            print(f"\n{content}")


@app.local_entrypoint()
def main():
    """Run Phase 3 and save results locally (non-detach mode)."""
    run_phase3.remote()
    print("\nDone. Now downloading results from volume...")
    download_results()


@app.function(image=modal.Image.debian_slim(python_version="3.11"),
              volumes={"/vol": vol})
def read_volume_file(path: str) -> str:
    """Read a text file from the volume."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""
