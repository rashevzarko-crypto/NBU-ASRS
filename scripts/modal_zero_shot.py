"""Zero-shot classification of ASRS reports using Qwen3-8B on Modal.

Sends test set narratives to vLLM on a Modal L4 GPU, parses LLM JSON outputs
into binary labels, computes per-category and aggregate metrics, and saves
results in the canonical format matching classic_ml_text_metrics.csv.
"""

import modal
import json
import re
import os
import time

MODEL_ID = "Qwen/Qwen3-8B"
GPU = "L4"
BATCH_SIZE = 64
CHECKPOINT_EVERY = 10  # save checkpoint every N batches

app = modal.App("asrs-zero-shot")

vllm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm", "torch", "transformers", "huggingface_hub")
)


@app.cls(
    image=vllm_image,
    gpu=GPU,
    timeout=3600,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
class ZeroShotClassifier:
    @modal.enter()
    def load_model(self):
        from vllm import LLM, SamplingParams

        self.llm = LLM(
            model=MODEL_ID,
            max_model_len=8192,
            dtype="auto",
            gpu_memory_utilization=0.90,
        )
        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=256,
        )

    def build_messages(self, narrative: str, categories: list[str]) -> list[dict]:
        """Build chat messages for zero-shot classification."""
        cat_list = "\n".join(f"- {c}" for c in categories)
        system_msg = (
            "You are an aviation safety analyst classifying ASRS incident reports. "
            "For each report, identify ALL applicable anomaly categories from the "
            "list below. A report can belong to multiple categories. "
            "Return ONLY a JSON array of matching category names, nothing else.\n\n"
            f"Categories:\n{cat_list}"
        )
        narrative = narrative[:2000]
        return [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": f"Classify this ASRS report into applicable anomaly categories:\n\n{narrative}"},
        ]

    @modal.method()
    def classify_batch(self, narratives: list[str], categories: list[str]) -> list[str]:
        """Classify a batch of narratives, returning raw LLM text outputs."""
        conversations = [self.build_messages(n, categories) for n in narratives]
        outputs = self.llm.chat(
            conversations, self.sampling_params,
            chat_template_kwargs={"enable_thinking": False},
        )
        return [o.outputs[0].text for o in outputs]


# ---------------------------------------------------------------------------
# Local helpers (run on the caller's machine, not on Modal)
# ---------------------------------------------------------------------------

def parse_llm_output(raw: str, categories: list[str]) -> list[str]:
    """Parse LLM output into a list of valid category names.

    Three-tier strategy:
      1. Direct JSON parse
      2. Regex extraction of JSON array from surrounding text
      3. Fuzzy substring matching of category names
    """
    cat_lower = {c.lower(): c for c in categories}

    # Tier 1: direct JSON parse
    try:
        parsed = json.loads(raw.strip())
        if isinstance(parsed, list):
            return _normalize(parsed, cat_lower)
    except (json.JSONDecodeError, TypeError):
        pass

    # Tier 2: regex — find first [...] block
    m = re.search(r"\[.*?\]", raw, re.DOTALL)
    if m:
        try:
            parsed = json.loads(m.group())
            if isinstance(parsed, list):
                return _normalize(parsed, cat_lower)
        except (json.JSONDecodeError, TypeError):
            pass

    # Tier 3: fuzzy substring matching
    matched = []
    raw_lower = raw.lower()
    for cat in categories:
        if cat.lower() in raw_lower:
            matched.append(cat)
    return matched


def _normalize(items: list, cat_lower: dict[str, str]) -> list[str]:
    """Map parsed items to exact category names via case-insensitive match."""
    result = []
    for item in items:
        if not isinstance(item, str):
            continue
        key = item.strip().lower()
        if key in cat_lower:
            result.append(cat_lower[key])
    return result


def compute_metrics(y_true, y_pred, categories):
    """Compute per-category and aggregate metrics matching classic ML format.

    Returns a list of dicts with keys: Category, Precision, Recall, F1, ROC-AUC
    """
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
    import numpy as np

    rows = []
    for i, cat in enumerate(categories):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        p = precision_score(yt, yp, zero_division=0)
        r = recall_score(yt, yp, zero_division=0)
        f1 = f1_score(yt, yp, zero_division=0)

        # ROC-AUC with binary predictions; handle constant predictions
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

    # Macro averages
    macro = {
        "Category": "MACRO",
        "Precision": np.mean([r["Precision"] for r in rows]),
        "Recall": np.mean([r["Recall"] for r in rows]),
        "F1": np.mean([r["F1"] for r in rows]),
        "ROC-AUC": np.mean([r["ROC-AUC"] for r in rows]),
    }

    # Micro averages
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


def format_summary(metrics_rows, categories, n_train, n_test):
    """Format a human-readable summary matching classic_ml_summary.txt."""
    lines = [
        "Zero-Shot LLM: Qwen3-8B (vLLM)",
        "=" * 55,
        f"Test set: {n_test} reports | Model: {MODEL_ID}",
        "vLLM: dtype=auto, max_model_len=8192, temperature=0.0, max_tokens=256",
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


def print_comparison(zero_shot_rows, classic_ml_path):
    """Print side-by-side F1 comparison between classic ML and zero-shot."""
    import pandas as pd

    classic = pd.read_csv(classic_ml_path)
    classic_f1 = dict(zip(classic["Category"], classic["F1"]))

    print("\n" + "=" * 70)
    print(f"{'Category':<35} {'Classic ML F1':>14} {'Zero-Shot F1':>14} {'Delta':>8}")
    print("-" * 70)
    for row in zero_shot_rows:
        cat = row["Category"]
        zf1 = row["F1"]
        cf1 = classic_f1.get(cat, float("nan"))
        delta = zf1 - cf1
        print(f"{cat:<35} {cf1:>14.4f} {zf1:>14.4f} {delta:>+8.4f}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Main entrypoint (runs locally, calls Modal for GPU inference)
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(test_csv: str = "data/test_set.csv", batch_size: int = BATCH_SIZE):
    """Run zero-shot classification: inference on Modal, metrics locally."""
    import pandas as pd
    import numpy as np

    # --- Load data ---
    df = pd.read_csv(test_csv)
    categories = [c for c in df.columns if c not in ("ACN", "Narrative")]
    narratives = df["Narrative"].tolist()
    acns = df["ACN"].tolist()
    n_test = len(df)
    print(f"Loaded {n_test} test reports, {len(categories)} categories")

    # --- Check for checkpoint ---
    checkpoint_path = "results/zero_shot_checkpoint.csv"
    completed = {}
    if os.path.exists(checkpoint_path):
        cp = pd.read_csv(checkpoint_path)
        completed = dict(zip(cp["ACN"].astype(str), cp["llm_raw_output"]))
        print(f"Resuming from checkpoint: {len(completed)} reports already done")

    # --- Inference ---
    classifier = ZeroShotClassifier()
    all_raw = []
    skipped = 0
    t0 = time.time()
    batch_count = 0

    for i in range(0, len(narratives), batch_size):
        batch_acns = acns[i : i + batch_size]
        batch_narratives = narratives[i : i + batch_size]

        # Check which ones we already have
        to_run_idx = []
        for j, acn in enumerate(batch_acns):
            if str(acn) not in completed:
                to_run_idx.append(j)

        if not to_run_idx:
            # All in this batch already done
            batch_results = [completed[str(a)] for a in batch_acns]
            all_raw.extend(batch_results)
            skipped += len(batch_acns)
        elif len(to_run_idx) == len(batch_acns):
            # None done — run full batch
            results = classifier.classify_batch.remote(batch_narratives, categories)
            all_raw.extend(results)
            for j, acn in enumerate(batch_acns):
                completed[str(acn)] = results[j]
        else:
            # Partial — run only the missing ones
            sub_narratives = [batch_narratives[j] for j in to_run_idx]
            results = classifier.classify_batch.remote(sub_narratives, categories)
            res_iter = iter(results)
            for j, acn in enumerate(batch_acns):
                if j in to_run_idx:
                    r = next(res_iter)
                    completed[str(acn)] = r
                    all_raw.append(r)
                else:
                    all_raw.append(completed[str(acn)])

        batch_count += 1
        done = len(all_raw)
        print(f"  Batch {batch_count}: {done}/{n_test} done")

        # Checkpoint every N batches
        if batch_count % CHECKPOINT_EVERY == 0:
            cp_df = pd.DataFrame({
                "ACN": list(completed.keys()),
                "llm_raw_output": list(completed.values()),
            })
            cp_df.to_csv(checkpoint_path, index=False)
            print(f"  [checkpoint saved: {len(completed)} reports]")

    elapsed = time.time() - t0
    if skipped:
        print(f"Skipped {skipped} reports from checkpoint")
    print(f"Total wall-clock time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Estimated Modal cost: ${elapsed / 3600 * 0.80:.2f} (L4 @ $0.80/hr)")

    # --- Parse outputs ---
    print("\nParsing LLM outputs...")
    parsed_labels = []
    parse_failures = 0
    tier_counts = {"json": 0, "regex": 0, "fuzzy": 0, "empty": 0}

    for raw in all_raw:
        # Track which tier succeeded
        result = None

        # Tier 1
        try:
            p = json.loads(raw.strip())
            if isinstance(p, list):
                cat_lower = {c.lower(): c for c in categories}
                result = _normalize(p, cat_lower)
                if result:
                    tier_counts["json"] += 1
        except (json.JSONDecodeError, TypeError):
            pass

        # Tier 2
        if result is None:
            m = re.search(r"\[.*?\]", raw, re.DOTALL)
            if m:
                try:
                    p = json.loads(m.group())
                    if isinstance(p, list):
                        cat_lower = {c.lower(): c for c in categories}
                        result = _normalize(p, cat_lower)
                        if result:
                            tier_counts["regex"] += 1
                except (json.JSONDecodeError, TypeError):
                    pass

        # Tier 3
        if result is None:
            raw_lower = raw.lower()
            matched = [c for c in categories if c.lower() in raw_lower]
            if matched:
                result = matched
                tier_counts["fuzzy"] += 1

        # Fallback
        if result is None:
            result = []
            tier_counts["empty"] += 1
            parse_failures += 1

        parsed_labels.append(result)

    fail_rate = parse_failures / n_test * 100
    print(f"Parse results: {tier_counts}")
    print(f"Parse failures (empty output): {parse_failures}/{n_test} ({fail_rate:.1f}%)")
    if fail_rate > 10:
        print("WARNING: Parse failure rate >10% — consider prompt tuning")

    # --- Build prediction matrix ---
    y_pred = np.zeros((n_test, len(categories)), dtype=int)
    for i, labels in enumerate(parsed_labels):
        for cat in labels:
            if cat in categories:
                y_pred[i, categories.index(cat)] = 1

    y_true = df[categories].values

    # --- Compute metrics ---
    print("\nComputing metrics...")
    metrics_rows = compute_metrics(y_true, y_pred, categories)

    # --- Save metrics CSV ---
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = "results/zero_shot_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved {metrics_path}")

    # --- Save raw outputs CSV ---
    raw_df = df.copy()
    raw_df["llm_raw_output"] = all_raw
    raw_df["parsed_labels"] = [json.dumps(l) for l in parsed_labels]
    for cat in categories:
        raw_df[f"pred_{cat}"] = y_pred[:, categories.index(cat)]
    raw_path = "results/zero_shot_raw_outputs.csv"
    raw_df.to_csv(raw_path, index=False)
    print(f"Saved {raw_path}")

    # --- Save summary ---
    summary = format_summary(metrics_rows, categories, 31850, n_test)
    summary_path = "results/zero_shot_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"Saved {summary_path}")

    # --- Print results ---
    print(f"\n{summary}")

    # --- Print comparison ---
    classic_path = "results/classic_ml_text_metrics.csv"
    if os.path.exists(classic_path):
        print_comparison(metrics_rows, classic_path)

    # --- Cleanup checkpoint ---
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print("\nCheckpoint file removed (run complete)")
