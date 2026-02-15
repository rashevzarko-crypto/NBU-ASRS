"""
Classic ML (TF-IDF + XGBoost) on the 48-label subcategory dataset — Modal CPU version.

Same hyperparams as the 13-label baseline, but on finer-grained subcategory labels.
Compares with parent-level Classic ML Macro-F1 0.691 / Micro-F1 0.746.

Usage:
    python -m modal run scripts/modal_classic_ml_subcategory.py
"""

import os
import time
import modal

app = modal.App("asrs-classic-ml-subcategory")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("pandas", "numpy", "scikit-learn", "xgboost")
    .add_local_file("data/subcategory_train_set.csv", remote_path="/mnt/data/subcategory_train_set.csv")
    .add_local_file("data/subcategory_test_set.csv", remote_path="/mnt/data/subcategory_test_set.csv")
)


@app.function(image=image, cpu=32, memory=65536, timeout=10800)
def train_and_predict() -> dict:
    """Load data, TF-IDF, train 48 XGBoost classifiers, predict on test set."""
    import pandas as pd
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from xgboost import XGBClassifier

    t_start = time.time()

    # --- Load data ---
    train_df = pd.read_csv("/mnt/data/subcategory_train_set.csv")
    test_df = pd.read_csv("/mnt/data/subcategory_test_set.csv")

    categories = [c for c in train_df.columns if c not in ("ACN", "Narrative")]
    n_train = len(train_df)
    n_test = len(test_df)
    print(f"Training set: {n_train} reports")
    print(f"Test set: {n_test} reports")
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
    X_train = tfidf.fit_transform(train_df["Narrative"].fillna(""))
    X_test = tfidf.transform(test_df["Narrative"].fillna(""))
    print(f"TF-IDF done in {time.time() - t_tfidf:.1f}s — "
          f"train: {X_train.shape}, test: {X_test.shape}")

    # --- Train 48 XGBoost classifiers ---
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
        print(f"  [{i+1:2d}/{len(categories)}] {col:<55s} "
              f"pos={n_pos:>6d} ({n_pos/n_train*100:5.1f}%) "
              f"spw={spw:6.2f}  {elapsed:6.1f}s")

    total_train = time.time() - t_start
    print(f"\nTotal time: {total_train:.1f}s ({total_train/60:.1f} min)")

    return {
        "y_pred": y_pred.tolist(),
        "y_proba": y_proba.tolist(),
        "categories": categories,
        "n_train": n_train,
        "n_test": n_test,
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

    # Macro / Micro on 2D arrays (n_samples x n_labels)
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


def format_summary(metrics_rows, n_train, n_test, parent_comparison):
    """Format a human-readable summary with parent-level comparison."""
    lines = [
        "Classic ML Baseline: TF-IDF + XGBoost (48 Subcategory Labels)",
        "=" * 65,
        f"Train set: {n_train:,} reports | Test set: {n_test:,} reports",
        "TF-IDF: max_features=50000, ngram_range=(1,2), sublinear_tf=True",
        "XGBoost: n_estimators=300, max_depth=6, lr=0.1, scale_pos_weight=auto",
        "",
        f"{'Category':<55} {'Precision':>10} {'Recall':>10} {'F1':>10} {'ROC-AUC':>10}",
        "-" * 97,
    ]
    for row in metrics_rows:
        cat = row["Category"]
        if cat in ("MACRO", "MICRO"):
            continue
        lines.append(
            f"{cat:<55} {row['Precision']:>10.4f} {row['Recall']:>10.4f} "
            f"{row['F1']:>10.4f} {row['ROC-AUC']:>10.4f}"
        )
    lines.append("-" * 97)
    for label in ("MACRO", "MICRO"):
        row = next(r for r in metrics_rows if r["Category"] == label)
        lines.append(
            f"{label:<55} {row['Precision']:>10.4f} {row['Recall']:>10.4f} "
            f"{row['F1']:>10.4f} {row['ROC-AUC']:>10.4f}"
        )

    # Parent-group comparison
    lines.append("")
    lines.append("")
    lines.append("Parent-Level Comparison (13-label vs 48-label)")
    lines.append("=" * 65)
    lines.append(parent_comparison)

    return "\n".join(lines)


def make_parent_comparison(metrics_rows, categories):
    """Build parent-level comparison table."""
    import pandas as pd
    import numpy as np

    def get_parent(col):
        if ":" in col:
            return col.split(":")[0].strip()
        return col

    # Load parent-level baseline
    baseline_path = "results/classic_ml_text_metrics.csv"
    if not os.path.exists(baseline_path):
        return "Parent-level baseline not found, skipping comparison."

    parent_metrics = pd.read_csv(baseline_path)
    parent_f1 = parent_metrics[~parent_metrics["Category"].isin(["MACRO", "MICRO"])].set_index("Category")["F1"]

    parent_macro_f1 = parent_metrics.loc[parent_metrics["Category"] == "MACRO", "F1"].values[0]
    parent_micro_f1 = parent_metrics.loc[parent_metrics["Category"] == "MICRO", "F1"].values[0]

    sub_macro_f1 = next(r["F1"] for r in metrics_rows if r["Category"] == "MACRO")
    sub_micro_f1 = next(r["F1"] for r in metrics_rows if r["Category"] == "MICRO")

    # Per-category subcategory F1
    sub_f1 = {r["Category"]: r["F1"] for r in metrics_rows if r["Category"] not in ("MACRO", "MICRO")}
    parent_map = {col: get_parent(col) for col in categories}

    # Group subcategory F1 by parent
    from collections import defaultdict
    parent_groups = defaultdict(list)
    for col in categories:
        parent_groups[parent_map[col]].append(sub_f1[col])

    lines = []
    lines.append(f"{'Parent Category':<30} {'Parent F1':>10} {'Avg Sub F1':>11} {'Delta':>8} {'#Subs':>6}")
    lines.append("-" * 67)
    for parent in sorted(parent_groups.keys()):
        pf1 = parent_f1.get(parent, float("nan"))
        avg_sf1 = np.mean(parent_groups[parent])
        delta = avg_sf1 - pf1 if not np.isnan(pf1) else float("nan")
        n_subs = len(parent_groups[parent])
        lines.append(f"{parent:<30} {pf1:>10.4f} {avg_sf1:>11.4f} {delta:>+8.4f} {n_subs:>6}")
    lines.append("-" * 67)
    lines.append(f"{'Macro-F1':<30} {parent_macro_f1:>10.4f} {sub_macro_f1:>11.4f} {sub_macro_f1 - parent_macro_f1:>+8.4f}")
    lines.append(f"{'Micro-F1':<30} {parent_micro_f1:>10.4f} {sub_micro_f1:>11.4f} {sub_micro_f1 - parent_micro_f1:>+8.4f}")

    return "\n".join(lines)


def save_barchart(metrics_rows, output_path):
    """Save horizontal bar chart of F1 scores."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cat_data = [(r["Category"], r["F1"]) for r in metrics_rows if r["Category"] not in ("MACRO", "MICRO")]
    cat_data.sort(key=lambda x: x[1])
    cats = [c for c, _ in cat_data]
    f1s = [f for _, f in cat_data]

    macro_f1 = next(r["F1"] for r in metrics_rows if r["Category"] == "MACRO")

    fig, ax = plt.subplots(figsize=(12, 14))
    bars = ax.barh(cats, f1s, color="steelblue", edgecolor="white")
    ax.axvline(macro_f1, color="red", linestyle="--", linewidth=1.5, label=f"Macro-F1 = {macro_f1:.3f}")

    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{width:.3f}", va="center", fontsize=8)

    ax.set_xlabel("F1 Score")
    ax.set_title("TF-IDF + XGBoost: Per-Subcategory F1 Scores (48 Labels)")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1.0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved {output_path}")


# ---------------------------------------------------------------------------
# Main entrypoint (runs locally)
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main():
    """Run classic ML on subcategory dataset via Modal CPU, compute metrics locally."""
    import pandas as pd
    import numpy as np

    metrics_path = "results/classic_ml_subcategory_metrics.csv"
    pred_path = "results/classic_ml_subcategory_predictions.csv"
    summary_path = "results/classic_ml_subcategory_summary.txt"
    chart_path = "results/classic_ml_subcategory_f1_barchart.png"

    if os.path.exists(metrics_path) and os.path.exists(pred_path):
        print(f"Checkpoint: {metrics_path} and {pred_path} already exist.")
        metrics_df = pd.read_csv(metrics_path)
        print(metrics_df.to_string(index=False))
        return

    print("Launching XGBoost training on Modal (32 CPU cores)...")
    t0 = time.time()
    result = train_and_predict.remote()
    wall_time = time.time() - t0

    # Unpack
    y_pred = np.array(result["y_pred"], dtype=int)
    y_proba = np.array(result["y_proba"], dtype=float)
    categories = result["categories"]
    n_train = result["n_train"]
    n_test = result["n_test"]
    total_seconds = result["total_seconds"]

    # Load test set for ground truth
    test_df = pd.read_csv("data/subcategory_test_set.csv")
    y_true = test_df[categories].values

    print(f"\nRemote completed: {n_train:,} train, {n_test:,} test, {len(categories)} labels")
    print(f"Remote time: {total_seconds:.1f}s ({total_seconds/60:.1f} min)")
    print(f"Wall-clock: {wall_time:.1f}s ({wall_time/60:.1f} min)")
    print(f"Estimated cost: ${wall_time / 3600 * 1.28:.2f} (32-core CPU @ ~$1.28/hr)")

    # Compute metrics
    print("\nComputing metrics...")
    metrics_rows = compute_metrics(y_true, y_pred, y_proba, categories)

    # Save metrics CSV
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved {metrics_path}")

    # Save predictions CSV
    pred_data = {"ACN": test_df["ACN"].values}
    for i, col in enumerate(categories):
        pred_data[f"{col}_true"] = y_true[:, i]
        pred_data[f"{col}_pred"] = y_pred[:, i]
        pred_data[f"{col}_proba"] = y_proba[:, i]
    predictions_df = pd.DataFrame(pred_data)
    predictions_df.to_csv(pred_path, index=False)
    print(f"Saved {pred_path}")

    # Parent comparison
    parent_comparison = make_parent_comparison(metrics_rows, categories)
    print(f"\n{parent_comparison}")

    # Save summary
    summary = format_summary(metrics_rows, n_train, n_test, parent_comparison)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"\nSaved {summary_path}")

    # Save bar chart
    save_barchart(metrics_rows, chart_path)

    # Print full results
    print(f"\n{summary}")

    # Per-classifier timing
    print("\nPer-classifier timing:")
    for cat, t in result["timings"].items():
        print(f"  {cat:<55s} {t:6.1f}s")
