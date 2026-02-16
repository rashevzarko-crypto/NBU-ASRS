"""Zero-shot classification of ASRS reports using Qwen3-8B on Modal
with taxonomy-enriched prompt for 48-label subcategory dataset.

Same vLLM/Modal infrastructure as modal_zero_shot_taxonomy.py,
but adapted for 48 subcategory labels using the prompt and normalization
from mistral_large_subcategory.py.
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
MAX_TOKENS = 512

app = modal.App("asrs-zero-shot-subcategory")

vllm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm", "torch", "transformers", "huggingface_hub")
)

# ---------------------------------------------------------------------------
# 48-label subcategory taxonomy prompt (identical to mistral_large_subcategory.py)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert aviation safety analyst trained in NASA ASRS report classification.

Classify the following aviation safety report into one or more anomaly subcategories.
Output ONLY a JSON array of matching subcategory names, nothing else.

Subcategories (use these exact names in your output):

1. ATC Issue — Problems involving air traffic control services, instructions, or communications.
2. Aircraft Equipment Problem: Critical — Critical aircraft system malfunction or failure affecting safety of flight.
3. Aircraft Equipment Problem: Less Severe — Non-critical aircraft system malfunction or failure.
4. Airspace Violation — Unauthorized entry or operation in controlled/restricted airspace.
5. Conflict: Airborne Conflict — Loss of separation or near collision between aircraft in flight.
6. Conflict: Ground Conflict — Loss of separation or near collision on the ground.
7. Conflict: NMAC — Near mid-air collision.
8. Deviation - Altitude: Crossing Restriction Not Met — Failure to meet an altitude crossing restriction.
9. Deviation - Altitude: Excursion From Assigned Altitude — Departure from assigned or expected altitude.
10. Deviation - Altitude: Overshoot — Overshooting an assigned altitude during climb or descent.
11. Deviation - Altitude: Undershoot — Undershooting an assigned altitude during climb or descent.
12. Deviation - Procedural: Clearance — Deviation from ATC clearance or instructions.
13. Deviation - Procedural: FAR — Violation of Federal Aviation Regulations.
14. Deviation - Procedural: Hazardous Material Violation — Improper handling or transport of hazardous materials.
15. Deviation - Procedural: Landing Without Clearance — Landing at an airport without proper ATC clearance.
16. Deviation - Procedural: MEL/CDL — Issues with Minimum Equipment List or Configuration Deviation List compliance.
17. Deviation - Procedural: Maintenance — Maintenance procedure error or deviation.
18. Deviation - Procedural: Other/Unknown — Procedural deviation not fitting other subcategories.
19. Deviation - Procedural: Published Material/Policy — Deviation from published procedures, charts, or company policy.
20. Deviation - Procedural: Unauthorized Flight Operations (UAS) — Unauthorized drone/UAS operations.
21. Deviation - Procedural: Weight And Balance — Weight and balance calculation or loading error.
22. Deviation - Speed — Departure from assigned or appropriate speed.
23. Deviation - Track/Heading — Departure from assigned or intended track or heading.
24. Flight Deck/Cabin Event: Illness/Injury — Crew or passenger illness or injury during flight.
25. Flight Deck/Cabin Event: Other/Unknown — Flight deck or cabin event not fitting other subcategories.
26. Flight Deck/Cabin Event: Passenger Misconduct — Disruptive or unruly passenger behavior.
27. Flight Deck/Cabin Event: Smoke/Fire/Fumes/Odor — Smoke, fire, fumes, or unusual odor in the cabin or flight deck.
28. Ground Event/Encounter: Gear Up Landing — Landing with gear not extended.
29. Ground Event/Encounter: Ground Equipment Issue — Problems with ground support equipment.
30. Ground Event/Encounter: Ground Strike - Aircraft — Aircraft contacting ground or obstacle on the ground.
31. Ground Event/Encounter: Loss Of Aircraft Control — Loss of control of aircraft while on the ground.
32. Ground Event/Encounter: Object — Encounter with object on the ground (FOD, debris, etc.).
33. Ground Event/Encounter: Other/Unknown — Ground event not fitting other subcategories.
34. Ground Event/Encounter: Vehicle — Encounter or conflict with ground vehicle.
35. Ground Event/Encounter: Weather/Turbulence — Weather-related event while on the ground.
36. Ground Excursion: Runway — Aircraft departing the runway surface.
37. Ground Excursion: Taxiway — Aircraft departing the taxiway surface.
38. Ground Incursion: Runway — Unauthorized entry onto a runway.
39. Ground Incursion: Taxiway — Unauthorized entry onto a taxiway.
40. Inflight Event/Encounter: Bird/Animal — Bird or animal strike during flight.
41. Inflight Event/Encounter: CFTT/CFIT — Controlled flight toward or into terrain.
42. Inflight Event/Encounter: Fuel Issue — Fuel-related problem during flight.
43. Inflight Event/Encounter: Loss Of Aircraft Control — Loss of aircraft control during flight.
44. Inflight Event/Encounter: Other/Unknown — Inflight event not fitting other subcategories.
45. Inflight Event/Encounter: Unstabilized Approach — Approach not meeting stabilized approach criteria.
46. Inflight Event/Encounter: VFR In IMC — VFR flight encountering instrument meteorological conditions.
47. Inflight Event/Encounter: Wake Vortex Encounter — Encounter with wake turbulence from another aircraft.
48. Inflight Event/Encounter: Weather/Turbulence — Weather or turbulence encounter during flight.

IMPORTANT distinctions:
- Ground Excursion = aircraft LEAVING the intended surface vs Ground Incursion = unauthorized ENTRY onto a surface
- Ground Event/Encounter = events ON the ground vs Inflight Event/Encounter = events IN THE AIR
- Conflict: NMAC = near mid-air collision (most severe) vs Conflict: Airborne Conflict = general loss of separation in flight
- Use the most specific subcategory that applies. A report can belong to multiple subcategories.
- Only select subcategories clearly supported by the narrative. Be precise — avoid over-predicting."""


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
            max_tokens=MAX_TOKENS,
        )

    def build_messages(self, narrative: str) -> list[dict]:
        """Build chat messages for zero-shot classification."""
        narrative = narrative[:2000]
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Classify this ASRS report:\n\n{narrative}"},
        ]

    @modal.method()
    def classify_batch(self, narratives: list[str]) -> list[str]:
        """Classify a batch of narratives, returning raw LLM text outputs."""
        conversations = [self.build_messages(n) for n in narratives]
        outputs = self.llm.chat(
            conversations, self.sampling_params,
            chat_template_kwargs={"enable_thinking": False},
        )
        return [o.outputs[0].text for o in outputs]


# ---------------------------------------------------------------------------
# Local helpers (run on the caller's machine, not on Modal)
# ---------------------------------------------------------------------------

def _normalize(items: list, cat_lower: dict[str, str]) -> list[str]:
    """Map parsed items to exact subcategory names via case-insensitive match.

    Unlike the parent-level script, this does NOT fall back to parent-only names.
    Handles:
      - Exact match: "Aircraft Equipment Problem: Critical" -> match
      - Number prefix: "1. ATC Issue" -> "ATC Issue" -> match
      - Description suffix: "ATC Issue — Problems with ATC" -> "ATC Issue" -> match
    """
    result = []
    seen = set()
    for item in items:
        if not isinstance(item, str):
            continue
        key = item.strip()
        # Strip leading number prefix ("1. ", "12. ", etc.)
        key = re.sub(r'^\d+\.\s*', '', key)
        # Strip trailing description after em dash " — "
        if ' \u2014 ' in key:
            key = key.split(' \u2014 ')[0].strip()
        # Strip trailing description after double dash " -- "
        if ' -- ' in key:
            key = key.split(' -- ')[0].strip()

        lower_key = key.lower()
        if lower_key in cat_lower:
            cat = cat_lower[lower_key]
            if cat not in seen:
                result.append(cat)
                seen.add(cat)
    return result


def parse_llm_output(raw: str, categories: list[str]) -> list[str]:
    """Parse LLM output into a list of valid subcategory names.

    Three-tier strategy:
      1. Direct JSON parse (with code fence stripping)
      2. Regex extraction of JSON array from surrounding text
      3. Fuzzy substring matching of category names (longest first)
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

    # Tier 3: fuzzy substring matching (longest labels first to avoid false matches)
    matched = []
    raw_lower = raw.lower()
    sorted_cats = sorted(categories, key=len, reverse=True)
    for cat in sorted_cats:
        if cat.lower() in raw_lower:
            matched.append(cat)
    return matched


def compute_metrics(y_true, y_pred, categories):
    """Compute per-category and aggregate metrics matching classic ML format."""
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
    import numpy as np

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


def format_summary(metrics_rows, n_test, parent_comparison):
    """Format a human-readable summary with parent-group comparison."""
    lines = [
        "Zero-Shot LLM: Qwen3-8B — Subcategory (48-label) Classification",
        "=" * 97,
        f"Test set: {n_test} reports | Model: {MODEL_ID}",
        f"vLLM: dtype=auto, max_model_len=8192, temperature=0.0, max_tokens={MAX_TOKENS}",
        "Prompt: taxonomy-enriched with 48 NASA ASRS subcategories, thinking disabled",
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

    # Headline metrics
    macro_row = next(r for r in metrics_rows if r["Category"] == "MACRO")
    micro_row = next(r for r in metrics_rows if r["Category"] == "MICRO")
    lines.append("")
    lines.append(f">>> Macro-F1: {macro_row['F1']:.4f}  |  Micro-F1: {micro_row['F1']:.4f} <<<")

    # Parent-group comparison
    lines.append("")
    lines.append("")
    lines.append("Parent-Level Comparison (avg subcategory F1 by parent group)")
    lines.append("=" * 67)
    lines.append(parent_comparison)

    return "\n".join(lines)


def make_parent_comparison(metrics_rows, categories):
    """Build parent-level comparison table."""
    from collections import defaultdict
    import numpy as np
    import pandas as pd

    def get_parent(col):
        if ":" in col:
            return col.split(":")[0].strip()
        return col

    baseline_path = "results/classic_ml_text_metrics.csv"
    if not os.path.exists(baseline_path):
        return "Parent-level baseline not found, skipping comparison."

    parent_metrics = pd.read_csv(baseline_path)
    parent_f1 = parent_metrics[
        ~parent_metrics["Category"].isin(["MACRO", "MICRO"])
    ].set_index("Category")["F1"]

    parent_macro_f1 = parent_metrics.loc[
        parent_metrics["Category"] == "MACRO", "F1"
    ].values[0]
    parent_micro_f1 = parent_metrics.loc[
        parent_metrics["Category"] == "MICRO", "F1"
    ].values[0]

    sub_macro_f1 = next(r["F1"] for r in metrics_rows if r["Category"] == "MACRO")
    sub_micro_f1 = next(r["F1"] for r in metrics_rows if r["Category"] == "MICRO")

    sub_f1 = {
        r["Category"]: r["F1"]
        for r in metrics_rows
        if r["Category"] not in ("MACRO", "MICRO")
    }
    parent_map = {col: get_parent(col) for col in categories}

    parent_groups = defaultdict(list)
    for col in categories:
        parent_groups[parent_map[col]].append(sub_f1[col])

    lines = []
    lines.append(
        f"{'Parent Category':<30} {'Parent F1':>10} {'Avg Sub F1':>11} "
        f"{'Delta':>8} {'#Subs':>6}"
    )
    lines.append("-" * 67)
    for parent in sorted(parent_groups.keys()):
        pf1 = parent_f1.get(parent, float("nan"))
        avg_sf1 = np.mean(parent_groups[parent])
        delta = avg_sf1 - pf1 if not np.isnan(pf1) else float("nan")
        n_subs = len(parent_groups[parent])
        lines.append(
            f"{parent:<30} {pf1:>10.4f} {avg_sf1:>11.4f} "
            f"{delta:>+8.4f} {n_subs:>6}"
        )
    lines.append("-" * 67)
    lines.append(
        f"{'Macro-F1':<30} {parent_macro_f1:>10.4f} {sub_macro_f1:>11.4f} "
        f"{sub_macro_f1 - parent_macro_f1:>+8.4f}"
    )
    lines.append(
        f"{'Micro-F1':<30} {parent_micro_f1:>10.4f} {sub_micro_f1:>11.4f} "
        f"{sub_micro_f1 - parent_micro_f1:>+8.4f}"
    )

    return "\n".join(lines)


def print_comparison(qwen_rows):
    """Print comparison: Classic ML subcategory vs Mistral Large subcategory vs Qwen3."""
    import pandas as pd

    classic_path = "results/classic_ml_subcategory_metrics.csv"
    mistral_path = "results/mistral_large_subcategory_metrics.csv"

    classic_f1 = {}
    mistral_f1 = {}

    if os.path.exists(classic_path):
        classic = pd.read_csv(classic_path)
        classic_f1 = dict(zip(classic["Category"], classic["F1"]))

    if os.path.exists(mistral_path):
        mistral = pd.read_csv(mistral_path)
        mistral_f1 = dict(zip(mistral["Category"], mistral["F1"]))

    print("\n" + "=" * 105)
    if mistral_f1:
        print(
            f"{'Category':<55} {'Classic ML':>12} {'ML3 ZS':>12} "
            f"{'Qwen3 ZS':>12} {'Q-ML':>10}"
        )
    else:
        print(
            f"{'Category':<55} {'Classic ML':>12} {'Qwen3 ZS':>12} {'Delta':>10}"
        )
    print("-" * 105)
    for row in qwen_rows:
        cat = row["Category"]
        qf1 = row["F1"]
        cf1 = classic_f1.get(cat, float("nan"))
        mf1 = mistral_f1.get(cat, float("nan"))
        if mistral_f1:
            delta = qf1 - mf1 if not (mf1 != mf1) else float("nan")
            print(
                f"{cat:<55} {cf1:>12.4f} {mf1:>12.4f} "
                f"{qf1:>12.4f} {delta:>+10.4f}"
            )
        else:
            delta = qf1 - cf1 if not (cf1 != cf1) else float("nan")
            print(f"{cat:<55} {cf1:>12.4f} {qf1:>12.4f} {delta:>+10.4f}")
    print("=" * 105)

    # Headline
    macro_row = next(r for r in qwen_rows if r["Category"] == "MACRO")
    micro_row = next(r for r in qwen_rows if r["Category"] == "MICRO")
    print(
        f"\n>>> Qwen3-8B Subcategory ZS: "
        f"Macro-F1 = {macro_row['F1']:.4f}, Micro-F1 = {micro_row['F1']:.4f} <<<"
    )


# ---------------------------------------------------------------------------
# Main entrypoint (runs locally, calls Modal for GPU inference)
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main():
    """Run zero-shot subcategory classification: inference on Modal, metrics locally."""
    import pandas as pd
    import numpy as np

    test_csv = "data/subcategory_test_set.csv"
    batch_size = BATCH_SIZE

    # --- Load data ---
    df = pd.read_csv(test_csv)
    categories = [c for c in df.columns if c not in ("ACN", "Narrative")]
    narratives = df["Narrative"].tolist()
    acns = df["ACN"].tolist()
    n_test = len(df)
    print(f"Loaded {n_test} test reports, {len(categories)} categories")

    # --- Check for checkpoint ---
    checkpoint_path = "results/qwen_zero_shot_subcategory_checkpoint.csv"
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
            batch_results = [completed[str(a)] for a in batch_acns]
            all_raw.extend(batch_results)
            skipped += len(batch_acns)
        elif len(to_run_idx) == len(batch_acns):
            results = classifier.classify_batch.remote(batch_narratives)
            all_raw.extend(results)
            for j, acn in enumerate(batch_acns):
                completed[str(acn)] = results[j]
        else:
            sub_narratives = [batch_narratives[j] for j in to_run_idx]
            results = classifier.classify_batch.remote(sub_narratives)
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

    cat_lower = {c.lower(): c for c in categories}

    for raw in all_raw:
        result = None

        # Strip code fences
        cleaned = raw.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

        # Tier 1: direct JSON
        try:
            p = json.loads(cleaned.strip())
            if isinstance(p, list):
                result = _normalize(p, cat_lower)
                if result is not None:
                    tier_counts["json"] += 1
        except (json.JSONDecodeError, TypeError):
            pass

        # Tier 2: regex
        if result is None:
            m = re.search(r"\[.*?\]", raw, re.DOTALL)
            if m:
                try:
                    p = json.loads(m.group())
                    if isinstance(p, list):
                        result = _normalize(p, cat_lower)
                        if result is not None:
                            tier_counts["regex"] += 1
                except (json.JSONDecodeError, TypeError):
                    pass

        # Tier 3: fuzzy substring (longest first)
        if result is None:
            raw_lower = raw.lower()
            sorted_cats = sorted(categories, key=len, reverse=True)
            matched = [c for c in sorted_cats if c.lower() in raw_lower]
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
    metrics_path = "results/qwen_zero_shot_subcategory_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved {metrics_path}")

    # --- Save raw outputs CSV ---
    raw_df = df.copy()
    raw_df["llm_raw_output"] = all_raw
    raw_df["parsed_labels"] = [json.dumps(l) for l in parsed_labels]
    for cat in categories:
        raw_df[f"pred_{cat}"] = y_pred[:, categories.index(cat)]
    raw_path = "results/qwen_zero_shot_subcategory_raw_outputs.csv"
    raw_df.to_csv(raw_path, index=False)
    print(f"Saved {raw_path}")

    # --- Save summary ---
    parent_comparison = make_parent_comparison(metrics_rows, categories)
    summary = format_summary(metrics_rows, n_test, parent_comparison)
    summary_path = "results/qwen_zero_shot_subcategory_summary.txt"
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
