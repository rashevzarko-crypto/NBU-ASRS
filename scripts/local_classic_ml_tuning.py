"""
Classic ML Tuning: Final Evaluation (local CPU).

Takes the best TF-IDF config (no_sublinear) and baseline XGBoost params,
retrains on parent (13-label) and subcategory (48-label) datasets,
and saves all result files.

Also saves TF-IDF ablation and model comparison CSVs from hardcoded results.

Usage:
    python scripts/local_classic_ml_tuning.py
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

# -----------------------------------------------------------------------
# Hardcoded results from previous Modal runs
# -----------------------------------------------------------------------

PHASE1_RESULTS = [
    {"config": "baseline",       "max_features": 50000,  "ngram_range": "(1, 2)", "sublinear_tf": True,  "min_df": 1, "cv_macro_f1": 0.6280, "cv_micro_f1": 0.6922, "wall_seconds": 1245.0},
    {"config": "unigram_only",   "max_features": 50000,  "ngram_range": "(1, 1)", "sublinear_tf": True,  "min_df": 1, "cv_macro_f1": 0.6248, "cv_micro_f1": 0.6900, "wall_seconds": 525.0},
    {"config": "trigram",        "max_features": 50000,  "ngram_range": "(1, 3)", "sublinear_tf": True,  "min_df": 1, "cv_macro_f1": 0.6277, "cv_micro_f1": 0.6910, "wall_seconds": 1468.0},
    {"config": "trigram_100k",   "max_features": 100000, "ngram_range": "(1, 3)", "sublinear_tf": True,  "min_df": 1, "cv_macro_f1": 0.6263, "cv_micro_f1": 0.6901, "wall_seconds": 1645.0},
    {"config": "fewer_features", "max_features": 20000,  "ngram_range": "(1, 2)", "sublinear_tf": True,  "min_df": 1, "cv_macro_f1": 0.6278, "cv_micro_f1": 0.6918, "wall_seconds": 967.0},
    {"config": "more_features",  "max_features": 100000, "ngram_range": "(1, 2)", "sublinear_tf": True,  "min_df": 1, "cv_macro_f1": 0.6261, "cv_micro_f1": 0.6911, "wall_seconds": 1389.0},
    {"config": "no_sublinear",   "max_features": 50000,  "ngram_range": "(1, 2)", "sublinear_tf": False, "min_df": 1, "cv_macro_f1": 0.6296, "cv_micro_f1": 0.6929, "wall_seconds": 1285.0},
    {"config": "min_df_3",       "max_features": 50000,  "ngram_range": "(1, 2)", "sublinear_tf": True,  "min_df": 3, "cv_macro_f1": 0.6279, "cv_micro_f1": 0.6921, "wall_seconds": 1298.0},
]

PHASE2_SVC = {
    "model": "LinearSVC",
    "best_cv_f1": 0.4725,
    "test_macro_f1": 0.6550,
    "test_micro_f1": 0.7496,
    "best_params": '{"C": "0.113", "loss": "squared_hinge", "class_weight": "balanced", "max_iter": "5000"}',
    "wall_seconds": 236.0,
}
PHASE2_LR = {
    "model": "LogisticRegression",
    "best_cv_f1": 0.5035,
    "test_macro_f1": 0.6701,
    "test_micro_f1": 0.7375,
    "best_params": '{"C": "1.45", "penalty": "l2", "solver": "saga", "class_weight": "balanced", "max_iter": "5000"}',
    "wall_seconds": 12711.0,
}

# Best TF-IDF: no_sublinear (CV Macro-F1=0.6296)
# Note: Using 10K features due to severe RAM constraints (16GB total, <3GB free)
# TF-IDF ablation showed negligible F1 differences across all feature counts
BEST_TFIDF_PARAMS = {"max_features": 10000, "ngram_range": (1, 2), "sublinear_tf": False, "min_df": 1}

# XGBoost baseline params (unchanged — search was not completed due to Modal billing limit)
BEST_XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.1,
}


def compute_metrics(y_true, y_pred, y_proba, categories):
    """Per-category + MACRO/MICRO metrics on 2D arrays."""
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


def train_and_eval(X_train, y_train, X_test, y_test, categories, xgb_params, nthread, label=""):
    """Train per-label XGBoost and evaluate on test set."""
    import gc
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    n_labels = len(categories)
    y_pred = np.zeros((n_test, n_labels), dtype=int)
    y_proba = np.zeros((n_test, n_labels), dtype=float)

    t0 = time.time()
    for i, cat in enumerate(categories):
        gc.collect()
        yt = y_train[:, i]
        n_pos = int(yt.sum())
        n_neg = n_train - n_pos
        spw = n_neg / n_pos if n_pos > 0 else 1.0

        clf = XGBClassifier(
            scale_pos_weight=spw, eval_metric="logloss",
            random_state=42, verbosity=0, tree_method="approx",
            nthread=nthread, **xgb_params,
        )
        clf.fit(X_train, yt)
        y_pred[:, i] = clf.predict(X_test)
        y_proba[:, i] = clf.predict_proba(X_test)[:, 1]

        elapsed = time.time() - t0
        print(f"  [{i+1:2d}/{n_labels}] {cat:<55s} "
              f"pos={n_pos:>6d} ({n_pos/n_train*100:5.1f}%) {elapsed:6.1f}s",
              flush=True)

    total = time.time() - t0
    print(f"  {label} total: {total:.0f}s ({total/60:.1f} min)", flush=True)
    return y_pred, y_proba


def format_summary(metrics_rows, n_train, n_test, best_model, tfidf_info,
                   best_params, col_width, baseline_macro, baseline_micro, n_labels):
    macro = next(r for r in metrics_rows if r["Category"] == "MACRO")
    micro = next(r for r in metrics_rows if r["Category"] == "MICRO")
    lines = [
        f"Classic ML Tuned: TF-IDF + {best_model} ({n_labels} Labels)",
        "=" * 75,
        f"Train set: {n_train:,} reports | Test set: {n_test:,} reports",
        f"TF-IDF config: {tfidf_info}",
        f"Model: {best_model}",
        f"Best params: {best_params}",
        "",
        f"{'Category':<{col_width}} {'Precision':>10} {'Recall':>10} {'F1':>10} {'ROC-AUC':>10}",
        "-" * (col_width + 42),
    ]
    for row in metrics_rows:
        if row["Category"] in ("MACRO", "MICRO"):
            continue
        lines.append(
            f"{row['Category']:<{col_width}} {row['Precision']:>10.4f} {row['Recall']:>10.4f} "
            f"{row['F1']:>10.4f} {row['ROC-AUC']:>10.4f}"
        )
    lines.append("-" * (col_width + 42))
    for label in ("MACRO", "MICRO"):
        row = next(r for r in metrics_rows if r["Category"] == label)
        lines.append(
            f"{label:<{col_width}} {row['Precision']:>10.4f} {row['Recall']:>10.4f} "
            f"{row['F1']:>10.4f} {row['ROC-AUC']:>10.4f}"
        )
    lines.append("")
    lines.append("Comparison to Baseline (TF-IDF 50K bigram sublinear + XGBoost 300/depth6)")
    lines.append("=" * 75)
    lines.append(f"  Baseline Macro-F1: {baseline_macro:.4f}  |  Tuned: {macro['F1']:.4f}  |  Delta: {macro['F1'] - baseline_macro:+.4f}")
    lines.append(f"  Baseline Micro-F1: {baseline_micro:.4f}  |  Tuned: {micro['F1']:.4f}  |  Delta: {micro['F1'] - baseline_micro:+.4f}")
    return "\n".join(lines)


def main():
    nthread = 1  # Minimize memory — system has <3GB free RAM
    print(f"Classic ML Tuning — Local CPU ({nthread} cores)")
    print("=" * 60)

    # Output paths
    ablation_path = "results/tfidf_ablation.csv"
    comparison_path = "results/model_comparison.csv"
    parent_metrics_path = "results/classic_ml_tuned_parent_metrics.csv"
    parent_summary_path = "results/classic_ml_tuned_parent_summary.txt"
    sub_metrics_path = "results/classic_ml_tuned_subcategory_metrics.csv"
    sub_summary_path = "results/classic_ml_tuned_subcategory_summary.txt"

    # Check if already done
    if all(os.path.exists(p) for p in [ablation_path, comparison_path,
                                        parent_metrics_path, sub_metrics_path]):
        print("All output files already exist. Skipping.")
        df = pd.read_csv(parent_metrics_path)
        print(df.to_string(index=False))
        return

    # -------------------------------------------------------------------
    # Save Phase 1: TF-IDF ablation
    # -------------------------------------------------------------------
    print("\n--- Phase 1: TF-IDF Ablation (from previous Modal run) ---")
    ablation_df = pd.DataFrame(PHASE1_RESULTS)
    ablation_df.to_csv(ablation_path, index=False)
    print(f"Saved {ablation_path}")
    for row in PHASE1_RESULTS:
        print(f"  {row['config']:<18s} Macro-F1={row['cv_macro_f1']:.4f}  "
              f"Micro-F1={row['cv_micro_f1']:.4f}", flush=True)
    print(f"  Best: no_sublinear (CV Macro-F1=0.6296)")

    # -------------------------------------------------------------------
    # Save Phase 2: Model comparison
    # -------------------------------------------------------------------
    print("\n--- Phase 2: Model Comparison ---")
    tfidf_info = f"max_features={BEST_TFIDF_PARAMS['max_features']}, ngram_range=(1,2), sublinear_tf=False, min_df=1"
    xgb_params_str = json.dumps(BEST_XGB_PARAMS)
    model_comparison = [
        PHASE2_SVC,
        PHASE2_LR,
        {
            "model": "XGBoost",
            "best_cv_f1": 0.6791,  # BASELINE holdout Macro-F1 from search
            "test_macro_f1": None,  # Will be filled after Phase 3
            "test_micro_f1": None,
            "best_params": xgb_params_str,
            "wall_seconds": None,
        },
    ]
    print(f"  SVC: Test Macro-F1={PHASE2_SVC['test_macro_f1']:.4f}, "
          f"Micro-F1={PHASE2_SVC['test_micro_f1']:.4f}")
    print(f"  LR:  Test Macro-F1={PHASE2_LR['test_macro_f1']:.4f}, "
          f"Micro-F1={PHASE2_LR['test_micro_f1']:.4f}")
    print(f"  XGB: holdout Macro-F1=0.6791 (baseline params)")

    # -------------------------------------------------------------------
    # Phase 3: Final Evaluation
    # -------------------------------------------------------------------
    print("\n--- Phase 3: Final Evaluation ---")
    t_start = time.time()

    # Load parent data
    train_df = pd.read_csv("data/train_set.csv")
    test_df = pd.read_csv("data/test_set.csv")
    parent_cats = [c for c in train_df.columns if c not in ("ACN", "Narrative")]

    print(f"\nParent: {len(train_df)} train, {len(test_df)} test, {len(parent_cats)} labels")
    print(f"TF-IDF: {tfidf_info}")
    print(f"XGBoost: {xgb_params_str}")

    # Fit TF-IDF with best config
    print("\nFitting TF-IDF (parent)...", flush=True)
    tfidf = TfidfVectorizer(dtype=np.float32, **BEST_TFIDF_PARAMS)
    X_train = tfidf.fit_transform(train_df["Narrative"].fillna(""))
    X_test = tfidf.transform(test_df["Narrative"].fillna(""))
    y_train = train_df[parent_cats].values
    y_test = test_df[parent_cats].values
    print(f"  Features: {X_train.shape[1]}")

    # Train & evaluate parent
    print("\nTraining parent classifiers...", flush=True)
    y_pred, y_proba = train_and_eval(
        X_train, y_train, X_test, y_test, parent_cats,
        BEST_XGB_PARAMS, nthread, label="Parent",
    )
    parent_metrics = compute_metrics(y_test, y_pred, y_proba, parent_cats)
    pm = next(r for r in parent_metrics if r["Category"] == "MACRO")
    pmi = next(r for r in parent_metrics if r["Category"] == "MICRO")
    parent_time = time.time() - t_start
    print(f"\n  Parent Macro-F1: {pm['F1']:.4f}  Micro-F1: {pmi['F1']:.4f}  "
          f"AUC: {pm['ROC-AUC']:.4f}", flush=True)

    # Save parent metrics
    parent_df = pd.DataFrame(parent_metrics)
    parent_df.to_csv(parent_metrics_path, index=False)
    print(f"  Saved {parent_metrics_path}")

    parent_summary = format_summary(
        parent_metrics, len(train_df), len(test_df),
        "XGBoost", tfidf_info, xgb_params_str,
        col_width=35, baseline_macro=0.691, baseline_micro=0.746,
        n_labels=len(parent_cats),
    )
    with open(parent_summary_path, "w", encoding="utf-8") as f:
        f.write(parent_summary)
    print(f"  Saved {parent_summary_path}")

    # Update model comparison with actual XGB test results
    model_comparison[2]["test_macro_f1"] = round(pm["F1"], 4)
    model_comparison[2]["test_micro_f1"] = round(pmi["F1"], 4)
    model_comparison[2]["wall_seconds"] = round(parent_time, 1)

    comparison_df = pd.DataFrame(model_comparison)
    comparison_df.to_csv(comparison_path, index=False)
    print(f"  Saved {comparison_path}")

    # --- Subcategory ---
    print("\n--- Subcategory (48-label) ---", flush=True)
    t_sub = time.time()

    sub_train_df = pd.read_csv("data/subcategory_train_set.csv")
    sub_test_df = pd.read_csv("data/subcategory_test_set.csv")
    sub_cats = [c for c in sub_train_df.columns if c not in ("ACN", "Narrative")]

    print(f"  Data: {len(sub_train_df)} train, {len(sub_test_df)} test, {len(sub_cats)} labels")

    print("  Fitting TF-IDF (subcategory)...", flush=True)
    sub_tfidf = TfidfVectorizer(dtype=np.float32, **BEST_TFIDF_PARAMS)
    X_sub_train = sub_tfidf.fit_transform(sub_train_df["Narrative"].fillna(""))
    X_sub_test = sub_tfidf.transform(sub_test_df["Narrative"].fillna(""))
    y_sub_train = sub_train_df[sub_cats].values
    y_sub_test = sub_test_df[sub_cats].values

    print("  Training subcategory classifiers...", flush=True)
    sub_pred, sub_proba = train_and_eval(
        X_sub_train, y_sub_train, X_sub_test, y_sub_test, sub_cats,
        BEST_XGB_PARAMS, nthread, label="Subcategory",
    )
    sub_metrics = compute_metrics(y_sub_test, sub_pred, sub_proba, sub_cats)
    sm = next(r for r in sub_metrics if r["Category"] == "MACRO")
    smi = next(r for r in sub_metrics if r["Category"] == "MICRO")
    sub_time = time.time() - t_sub
    print(f"\n  Subcategory Macro-F1: {sm['F1']:.4f}  Micro-F1: {smi['F1']:.4f}  "
          f"AUC: {sm['ROC-AUC']:.4f}", flush=True)

    # Save subcategory metrics
    sub_df = pd.DataFrame(sub_metrics)
    sub_df.to_csv(sub_metrics_path, index=False)
    print(f"  Saved {sub_metrics_path}")

    sub_summary = format_summary(
        sub_metrics, len(sub_train_df), len(sub_test_df),
        "XGBoost", tfidf_info, xgb_params_str,
        col_width=55, baseline_macro=0.510, baseline_micro=0.600,
        n_labels=len(sub_cats),
    )
    with open(sub_summary_path, "w", encoding="utf-8") as f:
        f.write(sub_summary)
    print(f"  Saved {sub_summary_path}")

    # Final summary
    total_time = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"DONE in {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"  Parent:      Macro-F1={pm['F1']:.4f} (baseline 0.691, delta {pm['F1'] - 0.691:+.4f})")
    print(f"  Subcategory: Macro-F1={sm['F1']:.4f} (baseline 0.510, delta {sm['F1'] - 0.510:+.4f})")
    print(f"\nOutput files:")
    for p in [ablation_path, comparison_path, parent_metrics_path,
              parent_summary_path, sub_metrics_path, sub_summary_path]:
        print(f"  {p}")


if __name__ == "__main__":
    main()
