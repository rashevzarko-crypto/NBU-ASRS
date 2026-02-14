"""Few-shot classification of ASRS reports using Qwen3-8B on Modal
with taxonomy-enriched system prompt and THINKING MODE enabled.

Same as modal_few_shot_taxonomy.py but with enable_thinking=True so the model
produces <think>...</think> chain-of-thought reasoning before its JSON answer.
"""

import modal
import json
import re
import os
import time
import statistics

MODEL_ID = "Qwen/Qwen3-8B"
GPU = "A100"
BATCH_SIZE = 32  # A100 has much more KV cache headroom
CHECKPOINT_EVERY = 10  # save checkpoint every N batches

app = modal.App("asrs-few-shot-taxonomy-thinking")

vllm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm", "torch", "transformers", "huggingface_hub")
)

# ---------------------------------------------------------------------------
# Taxonomy-enriched system prompt (from Mistral Large experiment)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert aviation safety analyst trained in NASA ASRS report classification.

Classify the following aviation safety report into one or more anomaly categories.
Output ONLY a JSON array of matching category names, nothing else.

Categories (with official NASA ASRS taxonomy subcategories):

1. Aircraft Equipment Problem: Aircraft system malfunction or failure.
   Subcategories: Critical, Less Severe.

2. Airspace Violation: Unauthorized entry or operation in controlled/restricted airspace.

3. ATC Issue: Problems involving air traffic control services, instructions, or communications.

4. Conflict: Loss of separation or near collision.
   Subcategories: NMAC, Airborne Conflict, Ground Conflict (Critical), Ground Conflict (Less Severe).

5. Deviation - Altitude: Departure from assigned altitude.
   Subcategories: Excursion from Assigned Altitude, Crossing Restriction Not Met, Undershoot, Overshoot.

6. Deviation - Procedural: Departure from established procedures, clearances, regulations, or policies.
   Subcategories: Clearance, FAR, Published Material/Policy, Landing without Clearance, Maintenance, MEL/CDL, Weight and Balance.
   Note: This is the broadest category (~65% of reports).

7. Deviation - Speed: Departure from assigned or appropriate speed.

8. Deviation - Track/Heading: Departure from assigned or intended track or heading.

9. Flight Deck/Cabin Event: Events in the flight deck or cabin.
   Subcategories: Illness/Injury, Passenger Electronic Device, Passenger Misconduct, Smoke/Fire/Fumes/Odor.

10. Ground Event/Encounter: Events occurring ON the ground involving equipment or objects.
    Subcategories: Aircraft, FOD, Fuel Issue, Gear Up Landing, Ground Strike, Person/Animal/Bird, Ground Equipment Issue, Jet Blast, Loss of Aircraft Control, Object.

11. Ground Excursion: Aircraft LEAVING the intended surface.
    Subcategories: Ramp, Runway, Taxiway.

12. Ground Incursion: Unauthorized ENTRY onto a surface.
    Subcategories: Ramp, Runway, Taxiway, Vehicle.

13. Inflight Event/Encounter: Events occurring IN THE AIR.
    Subcategories: CFTT/CFIT, VFR in IMC, Fuel Issue, Laser, Loss of Aircraft Control, Wake Vortex Encounter, Weather/Turbulence, Object, Unstabilized Approach, Bird/Animal.

IMPORTANT distinctions:
- Ground Excursion = aircraft LEAVING the intended surface vs Ground Incursion = unauthorized ENTRY onto a surface
- Ground Event/Encounter = events ON the ground vs Inflight Event/Encounter = events IN THE AIR
- Deviation - Procedural is very broad — when in doubt about procedural compliance, include it

A report can belong to multiple categories. Only select categories clearly supported by the narrative. Be precise — avoid over-predicting."""


def select_few_shot_examples(train_csv: str, categories: list[str], n_per_cat: int = 3):
    """Select n diverse examples per category from the training set.

    Strategy: for each category, prefer examples with fewer total labels
    (clearer signal) and shorter narratives (save context budget).
    Tracks used ACNs to avoid duplicates across categories.
    """
    import pandas as pd

    df = pd.read_csv(train_csv)
    examples = []
    used_acns = set()

    # Compute label count per row
    df["_label_count"] = df[categories].sum(axis=1)
    df["_nlen"] = df["Narrative"].str.len()

    for cat in categories:
        cat_df = df[df[cat] == 1].copy()
        # Sort: fewer labels first, then shorter narratives
        cat_df = cat_df.sort_values(["_label_count", "_nlen"])
        cat_df = cat_df[~cat_df["ACN"].isin(used_acns)]

        selected = cat_df.head(n_per_cat)
        for _, row in selected.iterrows():
            labels = [c for c in categories if row[c] == 1]
            examples.append({
                "narrative": row["Narrative"][:600],  # truncate for context budget
                "labels": labels,
            })
            used_acns.add(row["ACN"])

    # Print example stats
    avg_labels = sum(len(ex["labels"]) for ex in examples) / len(examples) if examples else 0
    print(f"Selected {len(examples)} few-shot examples from {len(categories)} categories")
    print(f"  Avg labels per example: {avg_labels:.1f}")
    print(f"  Unique ACNs used: {len(used_acns)}")

    return examples


# ---------------------------------------------------------------------------
# Thinking-mode helpers
# ---------------------------------------------------------------------------

def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from model output."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def extract_thinking_lengths(raw_outputs: list[str]) -> dict:
    """Compute statistics on thinking block lengths across all outputs."""
    lengths = []
    count_with_thinking = 0
    for raw in raw_outputs:
        blocks = re.findall(r"<think>(.*?)</think>", raw, flags=re.DOTALL)
        if blocks:
            count_with_thinking += 1
            for block in blocks:
                lengths.append(len(block))
    stats = {
        "total_outputs": len(raw_outputs),
        "outputs_with_thinking": count_with_thinking,
        "pct_with_thinking": count_with_thinking / len(raw_outputs) * 100 if raw_outputs else 0,
    }
    if lengths:
        stats["avg_chars"] = statistics.mean(lengths)
        stats["median_chars"] = statistics.median(lengths)
        stats["max_chars"] = max(lengths)
        stats["min_chars"] = min(lengths)
        stats["total_blocks"] = len(lengths)
    else:
        stats["avg_chars"] = 0
        stats["median_chars"] = 0
        stats["max_chars"] = 0
        stats["min_chars"] = 0
        stats["total_blocks"] = 0
    return stats


@app.cls(
    image=vllm_image,
    gpu=GPU,
    timeout=3600,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
class FewShotClassifier:
    @modal.enter()
    def load_model(self):
        from vllm import LLM, SamplingParams

        self.llm = LLM(
            model=MODEL_ID,
            max_model_len=32768,  # room for thinking output on A100 80GB
            dtype="auto",
            gpu_memory_utilization=0.90,
        )
        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=4096,  # thinking blocks can be long
        )

    def build_messages(self, narrative: str, categories: list[str],
                       examples: list[dict]) -> list[dict]:
        """Build chat messages with few-shot examples for classification."""
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for ex in examples:
            messages.append({"role": "user", "content": f"Classify this ASRS report:\n\n{ex['narrative']}"})
            messages.append({"role": "assistant", "content": json.dumps(ex["labels"])})

        narrative = narrative[:1500]  # shorter than zero-shot (2000) to fit context
        messages.append({"role": "user", "content": f"Classify this ASRS report:\n\n{narrative}"})
        return messages

    @modal.method()
    def classify_batch(self, narratives: list[str], categories: list[str],
                       examples: list[dict]) -> list[str]:
        """Classify a batch of narratives, returning raw LLM text outputs."""
        conversations = [self.build_messages(n, categories, examples) for n in narratives]
        outputs = self.llm.chat(
            conversations, self.sampling_params,
            chat_template_kwargs={"enable_thinking": True},
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


def format_summary(metrics_rows, categories, n_test):
    """Format a human-readable summary matching classic_ml_summary.txt."""
    lines = [
        "Few-Shot LLM: Qwen3-8B (vLLM, taxonomy-enriched prompt, THINKING MODE)",
        "=" * 60,
        f"Test set: {n_test} reports | Model: {MODEL_ID}",
        "vLLM: dtype=auto, max_model_len=32768, temperature=0.0, max_tokens=4096",
        "Few-shot: 3 examples/category, 600 char truncation, sorted by label_count+length",
        "Prompt: taxonomy-enriched with NASA ASRS subcategories and discriminative hints",
        "Thinking: enable_thinking=True (chain-of-thought reasoning before JSON answer)",
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


def print_comparison(thinking_rows):
    """Print comparison: basic few-shot vs taxonomy few-shot vs taxonomy+thinking."""
    import pandas as pd

    classic_path = "results/classic_ml_text_metrics.csv"
    basic_path = "results/few_shot_metrics.csv"
    taxonomy_path = "results/few_shot_taxonomy_metrics.csv"

    classic_f1 = {}
    basic_f1 = {}
    taxonomy_f1 = {}

    if os.path.exists(classic_path):
        classic = pd.read_csv(classic_path)
        classic_f1 = dict(zip(classic["Category"], classic["F1"]))

    if os.path.exists(basic_path):
        basic = pd.read_csv(basic_path)
        basic_f1 = dict(zip(basic["Category"], basic["F1"]))

    if os.path.exists(taxonomy_path):
        taxonomy = pd.read_csv(taxonomy_path)
        taxonomy_f1 = dict(zip(taxonomy["Category"], taxonomy["F1"]))

    print("\n" + "=" * 100)
    print(f"{'Category':<35} {'Classic ML':>12} {'FS Basic':>12} {'FS Taxonomy':>13} {'FS Tax+Think':>14} {'Delta':>9}")
    print("-" * 100)
    for row in thinking_rows:
        cat = row["Category"]
        tf1 = row["F1"]
        cf1 = classic_f1.get(cat, float("nan"))
        bf1 = basic_f1.get(cat, float("nan"))
        txf1 = taxonomy_f1.get(cat, float("nan"))
        delta = tf1 - txf1 if not (txf1 != txf1) else float("nan")
        print(f"{cat:<35} {cf1:>12.4f} {bf1:>12.4f} {txf1:>13.4f} {tf1:>14.4f} {delta:>+9.4f}")
    print("=" * 100)


# ---------------------------------------------------------------------------
# Main entrypoint (runs locally, calls Modal for GPU inference)
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    test_csv: str = "data/test_set.csv",
    train_csv: str = "data/train_set.csv",
    batch_size: int = BATCH_SIZE,
):
    """Run few-shot classification with thinking mode: inference on Modal, metrics locally."""
    import pandas as pd
    import numpy as np

    # --- Load data ---
    df = pd.read_csv(test_csv)
    categories = [c for c in df.columns if c not in ("ACN", "Narrative")]
    narratives = df["Narrative"].tolist()
    acns = df["ACN"].tolist()
    n_test = len(df)
    print(f"Loaded {n_test} test reports, {len(categories)} categories")

    # --- Select few-shot examples ---
    examples = select_few_shot_examples(train_csv, categories, n_per_cat=3)

    # --- Check for checkpoint ---
    checkpoint_path = "results/few_shot_taxonomy_thinking_checkpoint.csv"
    completed = {}
    if os.path.exists(checkpoint_path):
        cp = pd.read_csv(checkpoint_path)
        completed = dict(zip(cp["ACN"].astype(str), cp["llm_raw_output"]))
        print(f"Resuming from checkpoint: {len(completed)} reports already done")

    # --- Inference ---
    classifier = FewShotClassifier()
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
            results = classifier.classify_batch.remote(batch_narratives, categories, examples)
            all_raw.extend(results)
            for j, acn in enumerate(batch_acns):
                completed[str(acn)] = results[j]
        else:
            # Partial — run only the missing ones
            sub_narratives = [batch_narratives[j] for j in to_run_idx]
            results = classifier.classify_batch.remote(sub_narratives, categories, examples)
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
    print(f"Estimated Modal cost: ${elapsed / 3600 * 2.78:.2f} (A100 @ $2.78/hr)")

    # --- Thinking mode stats ---
    print("\n--- Thinking Mode Statistics ---")
    thinking_stats = extract_thinking_lengths(all_raw)
    print(f"Outputs with <think> blocks: {thinking_stats['outputs_with_thinking']}/{thinking_stats['total_outputs']} ({thinking_stats['pct_with_thinking']:.1f}%)")
    print(f"Total thinking blocks: {thinking_stats['total_blocks']}")
    if thinking_stats['total_blocks'] > 0:
        print(f"Thinking block length (chars): avg={thinking_stats['avg_chars']:.0f}, median={thinking_stats['median_chars']:.0f}, max={thinking_stats['max_chars']}, min={thinking_stats['min_chars']}")

    # --- Parse outputs (strip thinking blocks first) ---
    print("\nParsing LLM outputs (stripping <think> blocks first)...")
    parsed_labels = []
    parse_failures = 0
    tier_counts = {"json": 0, "regex": 0, "fuzzy": 0, "empty": 0}

    for raw in all_raw:
        # Strip thinking blocks before parsing
        cleaned = strip_thinking(raw)
        result = None

        # Tier 1: direct JSON
        try:
            p = json.loads(cleaned.strip())
            if isinstance(p, list):
                cat_lower = {c.lower(): c for c in categories}
                result = _normalize(p, cat_lower)
                if result:
                    tier_counts["json"] += 1
        except (json.JSONDecodeError, TypeError):
            pass

        # Tier 2: regex extract
        if result is None:
            m = re.search(r"\[.*?\]", cleaned, re.DOTALL)
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

        # Tier 3: fuzzy substring
        if result is None:
            cleaned_lower = cleaned.lower()
            matched = [c for c in categories if c.lower() in cleaned_lower]
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
    metrics_path = "results/few_shot_taxonomy_thinking_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved {metrics_path}")

    # --- Save raw outputs CSV ---
    raw_df = df.copy()
    raw_df["llm_raw_output"] = all_raw
    raw_df["parsed_labels"] = [json.dumps(l) for l in parsed_labels]
    for cat in categories:
        raw_df[f"pred_{cat}"] = y_pred[:, categories.index(cat)]
    raw_path = "results/few_shot_taxonomy_thinking_raw_outputs.csv"
    raw_df.to_csv(raw_path, index=False)
    print(f"Saved {raw_path}")

    # --- Save summary ---
    summary = format_summary(metrics_rows, categories, n_test)
    summary_path = "results/few_shot_taxonomy_thinking_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"Saved {summary_path}")

    # --- Print results ---
    print(f"\n{summary}")

    # --- Print comparison ---
    print_comparison(metrics_rows)

    # --- Cleanup checkpoint ---
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print("\nCheckpoint file removed (run complete)")
