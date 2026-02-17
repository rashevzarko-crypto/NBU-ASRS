"""
Classic ML Hyperparameter Tuning v3 — Fast XGB search + final evaluation.

Phase 1: TF-IDF ablation — hardcoded from previous run (8 configs, 3-fold CV)
Phase 2: Model comparison — SVC/LR hardcoded from v2, XGB sequential search
Phase 3: Final evaluation on parent (13-label) and subcategory (48-label) test sets

v3 optimizations vs v2:
- SVC/LR results hardcoded (saves 3.7 hours)
- XGB: sequential (nthread=32), no lr=0.01 in grid (avoids 40+ min/combo)
- Uses 20K TF-IDF features for XGB search speed, 50K for final eval
- 15 combos instead of 30

Usage:
    python -m modal run --detach scripts/modal_classic_ml_tuning.py
    python -m modal run scripts/modal_classic_ml_tuning.py::download_results

Expected runtime: ~3-4 hours on 32 CPUs (~$4-5)
"""

import json
import os
import time
import modal

app = modal.App("asrs-classic-ml-tuning-v3")

vol = modal.Volume.from_name("asrs-tuning-results", create_if_missing=True)

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

# Phase 1 results from previous run (8 TF-IDF configs, 3-fold CV with XGB n_est=50)
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

# Phase 2 SVC/LR results from v2 run (hardcoded)
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

BEST_TFIDF_PARAMS = {"max_features": 50000, "ngram_range": (1, 2), "sublinear_tf": False, "min_df": 1}


@app.function(image=image, cpu=32, memory=65536, timeout=28800,
              volumes={"/vol": vol})
def run_all_phases() -> dict:
    """Run XGB search + Phase 3 final evaluation remotely."""
    import numpy as np
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import (
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.model_selection import ParameterSampler, train_test_split
    from xgboost import XGBClassifier

    # Force unbuffered stdout so Modal log streaming works on Windows
    import sys
    sys.stdout.reconfigure(line_buffering=True)

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

    def train_xgb_per_label(X_tr, y_tr, X_eval, categories, xgb_params, nthread=32, label=""):
        """Train per-label XGBoost, return predictions + probabilities."""
        n_test = X_eval.shape[0]
        n_labels = len(categories)
        n_train = X_tr.shape[0]
        y_pred = np.zeros((n_test, n_labels), dtype=int)
        y_proba = np.zeros((n_test, n_labels), dtype=float)
        t0 = time.time()
        for i, cat in enumerate(categories):
            yt = y_tr[:, i]
            n_pos = int(yt.sum())
            n_neg = n_train - n_pos
            spw = n_neg / n_pos if n_pos > 0 else 1.0
            clf = XGBClassifier(
                scale_pos_weight=spw, eval_metric="logloss",
                random_state=42, verbosity=0, tree_method="hist",
                nthread=nthread, **xgb_params,
            )
            clf.fit(X_tr, yt)
            y_pred[:, i] = clf.predict(X_eval)
            y_proba[:, i] = clf.predict_proba(X_eval)[:, 1]
            elapsed = time.time() - t0
            print(f"    [{i+1:2d}/{n_labels}] {cat:<55s} "
                  f"pos={n_pos:>6d} ({n_pos/n_train*100:5.1f}%) {elapsed:6.1f}s",
                  flush=True)
        total = time.time() - t0
        print(f"    {label} training: {total:.0f}s ({total/60:.1f} min)")
        return y_pred, y_proba

    t_start = time.time()

    # -----------------------------------------------------------------------
    # Load parent data
    # -----------------------------------------------------------------------
    train_df = pd.read_csv("/mnt/data/train_set.csv")
    test_df = pd.read_csv("/mnt/data/test_set.csv")
    parent_cats = [c for c in train_df.columns if c not in ("ACN", "Narrative")]
    n_labels = len(parent_cats)

    X_train_text = train_df["Narrative"].fillna("")
    X_test_text = test_df["Narrative"].fillna("")
    y_train = train_df[parent_cats].values
    y_test = test_df[parent_cats].values

    print(f"Parent data: {len(train_df)} train, {len(test_df)} test, {n_labels} labels")

    # ===================================================================
    # PHASE 1: TF-IDF Ablation — HARDCODED
    # ===================================================================
    print("\n" + "=" * 70)
    print("PHASE 1: TF-IDF Ablation (hardcoded from previous run)")
    print("=" * 70)
    for row in PHASE1_RESULTS:
        print(f"  {row['config']:<18s} Macro-F1={row['cv_macro_f1']:.4f}  "
              f"Micro-F1={row['cv_micro_f1']:.4f}  ({row['wall_seconds']:.0f}s)")
    print(f"\n>>> Best TF-IDF: no_sublinear (CV Macro-F1=0.6296)")

    # ===================================================================
    # PHASE 2: Model Comparison
    # ===================================================================
    print("\n" + "=" * 70)
    print("PHASE 2: Model Comparison")
    print("=" * 70)

    # SVC/LR: hardcoded
    print(f"\n  SVC (hardcoded): Test Macro-F1={PHASE2_SVC['test_macro_f1']:.4f}, "
          f"Micro-F1={PHASE2_SVC['test_micro_f1']:.4f}")
    print(f"  LR  (hardcoded): Test Macro-F1={PHASE2_LR['test_macro_f1']:.4f}, "
          f"Micro-F1={PHASE2_LR['test_micro_f1']:.4f}")

    # --- XGB Search: sequential, nthread=32, 20K features for speed ---
    print("\n--- XGBoost: sequential param search (15 combos, early stopping) ---")
    print("  Using 20K TF-IDF features for search speed, nthread=32")

    # Fit 20K TF-IDF for search
    search_tfidf = TfidfVectorizer(
        max_features=20000, ngram_range=(1, 2), sublinear_tf=False,
        min_df=1, dtype=np.float32,
    )
    X_train_search = search_tfidf.fit_transform(X_train_text)

    # Holdout split
    X_htrain, X_hval, y_htrain, y_hval = train_test_split(
        X_train_search, y_train, test_size=0.2, random_state=42,
    )
    X_es_tr, X_es_val, y_es_tr, y_es_val = train_test_split(
        X_htrain, y_htrain, test_size=0.1, random_state=42,
    )
    print(f"  Split: {X_es_tr.shape[0]} train, {X_es_val.shape[0]} early-stop eval, "
          f"{X_hval.shape[0]} holdout (features: {X_es_tr.shape[1]})")

    # Grid: no lr=0.01 (too slow), no depth=3 (too shallow)
    xgb_param_dist = {
        "max_depth": [4, 6, 8, 10],
        "learning_rate": [0.05, 0.1, 0.15, 0.2],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "min_child_weight": [1, 3, 5],
    }
    xgb_combos = list(ParameterSampler(xgb_param_dist, n_iter=15, random_state=42))

    # Also include the baseline combo for comparison
    baseline_combo = {
        "max_depth": 6, "learning_rate": 0.1, "subsample": 1.0,
        "colsample_bytree": 1.0, "min_child_weight": 1,
    }
    xgb_combos.insert(0, baseline_combo)
    n_combos = len(xgb_combos)  # 16 total (baseline + 15 random)

    xgb_search_results = []
    best_f1 = -1
    best_combo = None
    best_avg_rounds = None
    best_combo_idx = -1
    t_search = time.time()

    for ci, combo in enumerate(xgb_combos):
        t0 = time.time()
        y_pred_h = np.zeros((X_hval.shape[0], n_labels), dtype=int)
        rounds_list = []

        for i in range(n_labels):
            yt = y_es_tr[:, i]
            n_pos = int(yt.sum())
            n_neg = len(yt) - n_pos
            spw = n_neg / n_pos if n_pos > 0 else 1.0

            clf = XGBClassifier(
                n_estimators=500,
                scale_pos_weight=spw, eval_metric="logloss",
                random_state=42, verbosity=0, tree_method="hist",
                nthread=32, early_stopping_rounds=30,
                **combo,
            )
            clf.fit(X_es_tr, yt, eval_set=[(X_es_val, y_es_val[:, i])], verbose=False)
            rounds_list.append(clf.best_iteration + 1)
            y_pred_h[:, i] = clf.predict(X_hval)

        macro_f1 = f1_score(y_hval, y_pred_h, average="macro", zero_division=0)
        micro_f1 = f1_score(y_hval, y_pred_h, average="micro", zero_division=0)
        elapsed = time.time() - t0
        avg_rounds = np.mean(rounds_list)

        label = "BASELINE" if ci == 0 else f"combo {ci}"
        xgb_search_results.append({
            "combo_idx": ci,
            "label": label,
            "macro_f1": macro_f1,
            "micro_f1": micro_f1,
            "avg_rounds": avg_rounds,
            "wall_seconds": elapsed,
            **combo,
        })

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_combo = dict(combo)
            best_avg_rounds = avg_rounds
            best_combo_idx = ci

        print(f"  [{ci+1:2d}/{n_combos}] {label:<10s} Macro-F1={macro_f1:.4f} "
              f"Micro-F1={micro_f1:.4f} avg_rounds={avg_rounds:.0f} "
              f"depth={combo['max_depth']} lr={combo['learning_rate']} "
              f"sub={combo['subsample']} col={combo['colsample_bytree']} "
              f"mcw={combo['min_child_weight']}  ({elapsed:.0f}s)"
              + ("  <<<" if macro_f1 >= best_f1 else ""),
              flush=True)

    search_wall = time.time() - t_search
    print(f"\n  Search time: {search_wall:.0f}s ({search_wall/60:.1f} min)")
    best_label = "BASELINE" if best_combo_idx == 0 else f"combo {best_combo_idx}"
    print(f"  >>> Best: {best_label} Macro-F1={best_f1:.4f} (avg {best_avg_rounds:.0f} rounds)")
    print(f"  >>> Params: {best_combo}")

    # Set n_estimators from avg_rounds with headroom
    best_n_est = max(100, int(best_avg_rounds * 1.2))
    best_xgb_params = dict(best_combo)
    best_xgb_params["n_estimators"] = best_n_est
    print(f"  Using n_estimators={best_n_est}")

    # Retrain best XGB on FULL train (50K features) → test set
    print("\n  Retraining best XGB on full train (50K features, nthread=32)...")
    full_tfidf = TfidfVectorizer(dtype=np.float32, **BEST_TFIDF_PARAMS)
    X_train_full = full_tfidf.fit_transform(X_train_text)
    X_test_full = full_tfidf.transform(X_test_text)

    xgb_test_pred, xgb_test_proba = train_xgb_per_label(
        X_train_full, y_train, X_test_full, parent_cats,
        best_xgb_params, nthread=32, label="XGB-parent",
    )
    xgb_test_macro = f1_score(y_test, xgb_test_pred, average="macro", zero_division=0)
    xgb_test_micro = f1_score(y_test, xgb_test_pred, average="micro", zero_division=0)
    print(f"  XGB Test: Macro-F1={xgb_test_macro:.4f}, Micro-F1={xgb_test_micro:.4f}")

    model_comparison_rows = [
        PHASE2_SVC,
        PHASE2_LR,
        {
            "model": "XGBoost",
            "best_cv_f1": round(best_f1, 4),
            "test_macro_f1": round(xgb_test_macro, 4),
            "test_micro_f1": round(xgb_test_micro, 4),
            "best_params": json.dumps({k: str(v) for k, v in best_xgb_params.items()}),
            "wall_seconds": round(search_wall, 1),
        },
    ]

    # ===================================================================
    # PHASE 3: Final Evaluation (parent + subcategory)
    # ===================================================================
    print("\n" + "=" * 70)
    print("PHASE 3: Final Evaluation (XGBoost)")
    print("=" * 70)

    # Parent metrics (already computed above)
    parent_metrics = compute_metrics(y_test, xgb_test_pred, xgb_test_proba, parent_cats)
    pm = next(r for r in parent_metrics if r["Category"] == "MACRO")
    print(f"  Parent: Macro-F1={pm['F1']:.4f}, Macro-AUC={pm['ROC-AUC']:.4f}")

    # --- Subcategory (48-label) ---
    print("\n--- Subcategory (48-label) final eval ---")
    sub_train_df = pd.read_csv("/mnt/data/subcategory_train_set.csv")
    sub_test_df = pd.read_csv("/mnt/data/subcategory_test_set.csv")
    sub_cats = [c for c in sub_train_df.columns if c not in ("ACN", "Narrative")]
    n_sub = len(sub_cats)

    sub_tfidf = TfidfVectorizer(dtype=np.float32, **BEST_TFIDF_PARAMS)
    X_sub_train = sub_tfidf.fit_transform(sub_train_df["Narrative"].fillna(""))
    X_sub_test = sub_tfidf.transform(sub_test_df["Narrative"].fillna(""))
    y_sub_train = sub_train_df[sub_cats].values
    y_sub_test = sub_test_df[sub_cats].values

    print(f"  Data: {len(sub_train_df)} train, {len(sub_test_df)} test, {n_sub} labels")

    sub_pred, sub_proba = train_xgb_per_label(
        X_sub_train, y_sub_train, X_sub_test, sub_cats,
        best_xgb_params, nthread=32, label="XGB-subcategory",
    )
    sub_metrics = compute_metrics(y_sub_test, sub_pred, sub_proba, sub_cats)

    sm = next(r for r in sub_metrics if r["Category"] == "MACRO")
    print(f"  Subcategory: Macro-F1={sm['F1']:.4f}, Macro-AUC={sm['ROC-AUC']:.4f}")

    total_time = time.time() - t_start
    print(f"\nTotal remote time: {total_time:.0f}s ({total_time/60:.1f} min)")

    # --- Save all results to Modal Volume for --detach mode ---
    print("\nSaving results to Modal Volume...")
    pd.DataFrame(PHASE1_RESULTS).to_csv("/vol/tfidf_ablation.csv", index=False)
    pd.DataFrame(xgb_search_results).to_csv("/vol/xgb_search_results.csv", index=False)
    pd.DataFrame(model_comparison_rows).to_csv("/vol/model_comparison.csv", index=False)
    pd.DataFrame(parent_metrics).to_csv("/vol/classic_ml_tuned_parent_metrics.csv", index=False)
    pd.DataFrame(sub_metrics).to_csv("/vol/classic_ml_tuned_subcategory_metrics.csv", index=False)

    result_dict = {
        "ablation_rows": PHASE1_RESULTS,
        "xgb_search_results": xgb_search_results,
        "model_comparison_rows": model_comparison_rows,
        "best_tfidf_name": "no_sublinear",
        "best_tfidf_params": {k: str(v) for k, v in BEST_TFIDF_PARAMS.items()},
        "best_model_name": "XGBoost",
        "best_params": {k: str(v) for k, v in best_xgb_params.items()},
        "parent_metrics": parent_metrics,
        "parent_cats": parent_cats,
        "parent_n_train": len(train_df),
        "parent_n_test": len(test_df),
        "sub_metrics": sub_metrics,
        "sub_cats": sub_cats,
        "sub_n_train": len(sub_train_df),
        "sub_n_test": len(sub_test_df),
        "total_time_s": total_time,
    }
    with open("/vol/tuning_result.json", "w") as f:
        json.dump(result_dict, f, indent=2)

    vol.commit()
    print("Results saved to volume 'asrs-tuning-results'")

    return {
        "ablation_rows": PHASE1_RESULTS,
        "xgb_search_results": xgb_search_results,
        "model_comparison_rows": model_comparison_rows,
        "best_tfidf_name": "no_sublinear",
        "best_tfidf_params": {k: str(v) for k, v in BEST_TFIDF_PARAMS.items()},
        "best_model_name": "XGBoost",
        "best_params": {k: str(v) for k, v in best_xgb_params.items()},
        "parent_metrics": parent_metrics,
        "parent_cats": parent_cats,
        "parent_n_train": len(train_df),
        "parent_n_test": len(test_df),
        "sub_metrics": sub_metrics,
        "sub_cats": sub_cats,
        "sub_n_train": len(sub_train_df),
        "sub_n_test": len(sub_test_df),
    }


# ---------------------------------------------------------------------------
# Local helpers
# ---------------------------------------------------------------------------

def format_parent_summary(metrics_rows, n_train, n_test, best_model, best_tfidf,
                          best_params, baseline_macro=0.691, baseline_micro=0.746):
    macro = next(r for r in metrics_rows if r["Category"] == "MACRO")
    micro = next(r for r in metrics_rows if r["Category"] == "MICRO")
    lines = [
        f"Classic ML Tuned: TF-IDF + {best_model} (13 Parent Labels)",
        "=" * 75,
        f"Train set: {n_train:,} reports | Test set: {n_test:,} reports",
        f"TF-IDF config: {best_tfidf}",
        f"Model: {best_model}",
        f"Best params: {best_params}",
        "",
        f"{'Category':<35} {'Precision':>10} {'Recall':>10} {'F1':>10} {'ROC-AUC':>10}",
        "-" * 77,
    ]
    for row in metrics_rows:
        if row["Category"] in ("MACRO", "MICRO"):
            continue
        lines.append(
            f"{row['Category']:<35} {row['Precision']:>10.4f} {row['Recall']:>10.4f} "
            f"{row['F1']:>10.4f} {row['ROC-AUC']:>10.4f}"
        )
    lines.append("-" * 77)
    for label in ("MACRO", "MICRO"):
        row = next(r for r in metrics_rows if r["Category"] == label)
        lines.append(
            f"{label:<35} {row['Precision']:>10.4f} {row['Recall']:>10.4f} "
            f"{row['F1']:>10.4f} {row['ROC-AUC']:>10.4f}"
        )
    lines.append("")
    lines.append("Comparison to Baseline (TF-IDF 50K bigram sublinear + XGBoost 300/depth6)")
    lines.append("=" * 75)
    lines.append(f"  Baseline Macro-F1: {baseline_macro:.4f}  |  Tuned: {macro['F1']:.4f}  |  Delta: {macro['F1'] - baseline_macro:+.4f}")
    lines.append(f"  Baseline Micro-F1: {baseline_micro:.4f}  |  Tuned: {micro['F1']:.4f}  |  Delta: {micro['F1'] - baseline_micro:+.4f}")
    return "\n".join(lines)


def format_sub_summary(metrics_rows, n_train, n_test, best_model, best_tfidf,
                       best_params, baseline_macro=0.510, baseline_micro=0.600):
    macro = next(r for r in metrics_rows if r["Category"] == "MACRO")
    micro = next(r for r in metrics_rows if r["Category"] == "MICRO")
    lines = [
        f"Classic ML Tuned: TF-IDF + {best_model} (48 Subcategory Labels)",
        "=" * 75,
        f"Train set: {n_train:,} reports | Test set: {n_test:,} reports",
        f"TF-IDF config: {best_tfidf}",
        f"Model: {best_model}",
        f"Best params: {best_params}",
        "",
        f"{'Category':<55} {'Precision':>10} {'Recall':>10} {'F1':>10} {'ROC-AUC':>10}",
        "-" * 97,
    ]
    for row in metrics_rows:
        if row["Category"] in ("MACRO", "MICRO"):
            continue
        lines.append(
            f"{row['Category']:<55} {row['Precision']:>10.4f} {row['Recall']:>10.4f} "
            f"{row['F1']:>10.4f} {row['ROC-AUC']:>10.4f}"
        )
    lines.append("-" * 97)
    for label in ("MACRO", "MICRO"):
        row = next(r for r in metrics_rows if r["Category"] == label)
        lines.append(
            f"{label:<55} {row['Precision']:>10.4f} {row['Recall']:>10.4f} "
            f"{row['F1']:>10.4f} {row['ROC-AUC']:>10.4f}"
        )
    lines.append("")
    lines.append("Comparison to Baseline (TF-IDF 50K bigram sublinear + XGBoost 300/depth6)")
    lines.append("=" * 75)
    lines.append(f"  Baseline Macro-F1: {baseline_macro:.4f}  |  Tuned: {macro['F1']:.4f}  |  Delta: {macro['F1'] - baseline_macro:+.4f}")
    lines.append(f"  Baseline Micro-F1: {baseline_micro:.4f}  |  Tuned: {micro['F1']:.4f}  |  Delta: {micro['F1'] - baseline_micro:+.4f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main entrypoint (runs locally)
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main():
    import pandas as pd

    ablation_path = "results/tfidf_ablation.csv"
    comparison_path = "results/model_comparison.csv"
    xgb_search_path = "results/xgb_search_results.csv"
    parent_metrics_path = "results/classic_ml_tuned_parent_metrics.csv"
    parent_summary_path = "results/classic_ml_tuned_parent_summary.txt"
    sub_metrics_path = "results/classic_ml_tuned_subcategory_metrics.csv"
    sub_summary_path = "results/classic_ml_tuned_subcategory_summary.txt"

    all_exist = all(os.path.exists(p) for p in [
        ablation_path, comparison_path, parent_metrics_path, sub_metrics_path,
    ])
    if all_exist:
        print("All output files already exist. Skipping.")
        df = pd.read_csv(parent_metrics_path)
        print(df.to_string(index=False))
        return

    print("Launching Classic ML tuning v3 on Modal (32 CPU cores)...")
    print("Phase 1: hardcoded | Phase 2: SVC/LR hardcoded + XGB search | Phase 3: final eval")
    print("Expected runtime: ~3-4 hours")
    t0 = time.time()
    result = run_all_phases.remote()
    wall_time = time.time() - t0
    print(f"\nTotal wall-clock: {wall_time:.0f}s ({wall_time / 60:.1f} min)")
    print(f"Estimated cost: ${wall_time / 3600 * 1.28:.2f} (32-core CPU @ ~$1.28/hr)")

    # Save TF-IDF ablation
    ablation_df = pd.DataFrame(result["ablation_rows"])
    ablation_df.to_csv(ablation_path, index=False)
    print(f"\nSaved {ablation_path}")
    print(ablation_df.to_string(index=False))

    # Save XGB search results
    xgb_search_df = pd.DataFrame(result["xgb_search_results"])
    xgb_search_df.to_csv(xgb_search_path, index=False)
    print(f"\nSaved {xgb_search_path}")
    print(xgb_search_df[["label", "macro_f1", "micro_f1", "max_depth",
                          "learning_rate", "avg_rounds", "wall_seconds"]].to_string(index=False))

    # Save model comparison
    comparison_df = pd.DataFrame(result["model_comparison_rows"])
    comparison_df.to_csv(comparison_path, index=False)
    print(f"\nSaved {comparison_path}")
    print(comparison_df.to_string(index=False))

    # Save parent metrics + summary
    parent_df = pd.DataFrame(result["parent_metrics"])
    parent_df.to_csv(parent_metrics_path, index=False)
    print(f"\nSaved {parent_metrics_path}")

    parent_summary = format_parent_summary(
        result["parent_metrics"],
        result["parent_n_train"], result["parent_n_test"],
        result["best_model_name"], result["best_tfidf_params"],
        result["best_params"],
    )
    with open(parent_summary_path, "w", encoding="utf-8") as f:
        f.write(parent_summary)
    print(f"Saved {parent_summary_path}")
    print(f"\n{parent_summary}")

    # Save subcategory metrics + summary
    sub_df = pd.DataFrame(result["sub_metrics"])
    sub_df.to_csv(sub_metrics_path, index=False)
    print(f"\nSaved {sub_metrics_path}")

    sub_summary = format_sub_summary(
        result["sub_metrics"],
        result["sub_n_train"], result["sub_n_test"],
        result["best_model_name"], result["best_tfidf_params"],
        result["best_params"],
    )
    with open(sub_summary_path, "w", encoding="utf-8") as f:
        f.write(sub_summary)
    print(f"Saved {sub_summary_path}")
    print(f"\n{sub_summary}")

    # Final summary
    p_macro = next(r for r in result["parent_metrics"] if r["Category"] == "MACRO")
    s_macro = next(r for r in result["sub_metrics"] if r["Category"] == "MACRO")
    print(f"\n{'=' * 70}")
    print(f"FINAL: Best model = XGBoost")
    print(f"  Params: {result['best_params']}")
    print(f"  Parent:      Macro-F1={p_macro['F1']:.4f} (baseline 0.691, delta {p_macro['F1'] - 0.691:+.4f})")
    print(f"  Subcategory: Macro-F1={s_macro['F1']:.4f} (baseline 0.510, delta {s_macro['F1'] - 0.510:+.4f})")


# ---------------------------------------------------------------------------
# Download results from Modal Volume (after --detach run completes)
# ---------------------------------------------------------------------------

@app.local_entrypoint(name="download_results")
def download_results():
    """Download results from Modal Volume after a detached run completes.

    Usage: python -m modal run scripts/modal_classic_ml_tuning.py::download_results
    """
    import pandas as pd

    # Check if volume has results
    vol_ref = modal.Volume.from_name("asrs-tuning-results")

    # Read the result JSON from volume via a remote helper
    result_json = read_volume_file.remote("/vol/tuning_result.json")
    if not result_json:
        print("No results found in volume yet. The job may still be running.")
        print("Check: https://modal.com/apps")
        return

    result = json.loads(result_json)
    print(f"Found completed results (total time: {result.get('total_time_s', 0)/60:.1f} min)")

    # Download all CSV files
    files_to_download = [
        ("tfidf_ablation.csv", "results/tfidf_ablation.csv"),
        ("xgb_search_results.csv", "results/xgb_search_results.csv"),
        ("model_comparison.csv", "results/model_comparison.csv"),
        ("classic_ml_tuned_parent_metrics.csv", "results/classic_ml_tuned_parent_metrics.csv"),
        ("classic_ml_tuned_subcategory_metrics.csv", "results/classic_ml_tuned_subcategory_metrics.csv"),
    ]
    for vol_name, local_path in files_to_download:
        content = read_volume_file.remote(f"/vol/{vol_name}")
        if content:
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            with open(local_path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"Saved {local_path}")

    # Generate and save summaries locally
    parent_summary = format_parent_summary(
        result["parent_metrics"],
        result["parent_n_train"], result["parent_n_test"],
        result["best_model_name"], result["best_tfidf_params"],
        result["best_params"],
    )
    with open("results/classic_ml_tuned_parent_summary.txt", "w", encoding="utf-8") as f:
        f.write(parent_summary)
    print(f"Saved results/classic_ml_tuned_parent_summary.txt")
    print(f"\n{parent_summary}")

    sub_summary = format_sub_summary(
        result["sub_metrics"],
        result["sub_n_train"], result["sub_n_test"],
        result["best_model_name"], result["best_tfidf_params"],
        result["best_params"],
    )
    with open("results/classic_ml_tuned_subcategory_summary.txt", "w", encoding="utf-8") as f:
        f.write(sub_summary)
    print(f"Saved results/classic_ml_tuned_subcategory_summary.txt")
    print(f"\n{sub_summary}")

    p_macro = next(r for r in result["parent_metrics"] if r["Category"] == "MACRO")
    s_macro = next(r for r in result["sub_metrics"] if r["Category"] == "MACRO")
    print(f"\n{'=' * 70}")
    print(f"FINAL: Best model = XGBoost")
    print(f"  Params: {result['best_params']}")
    print(f"  Parent:      Macro-F1={p_macro['F1']:.4f} (baseline 0.691, delta {p_macro['F1'] - 0.691:+.4f})")
    print(f"  Subcategory: Macro-F1={s_macro['F1']:.4f} (baseline 0.510, delta {s_macro['F1'] - 0.510:+.4f})")


@app.function(image=modal.Image.debian_slim(python_version="3.11"),
              volumes={"/vol": vol})
def read_volume_file(path: str) -> str:
    """Read a text file from the volume."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""
