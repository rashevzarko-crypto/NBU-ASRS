"""GLM-5 few-shot classification of ASRS reports with taxonomy-enriched prompt and thinking.

Uses Modal's free research API (OpenAI-compatible endpoint) with GLM-5-FP8.
Thinking mode is on by default — no special API config needed.
Supports both parent-level (13-label) and subcategory (48-label) classification.

Usage:
    python scripts/glm5_zero_shot.py parent        # 13-label
    python scripts/glm5_zero_shot.py subcategory   # 48-label
    python scripts/glm5_zero_shot.py both          # sequential
"""

import json
import os
import re
import sys
import time
import statistics

import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

API_BASE_URL = "https://api.us-west-2.modal.direct/v1"
API_KEY = os.environ.get("GLM5_API_KEY", "")
MODEL = "zai-org/GLM-5-FP8"
TEMPERATURE = 0.0
MAX_TOKENS = 4096  # generous for thinking output
TIMEOUT = 120  # seconds per request
DELAY = 0.5  # seconds between requests

N_EXAMPLES_PER_CAT = 2  # 2 examples per category

RESULTS_DIR = "results"

# Retry config
MAX_RETRIES = 3
BACKOFF_BASE = 5  # 5s, 15s, 45s
RATE_LIMIT_WAIT = 30

# ---------------------------------------------------------------------------
# Parent-level (13-label) taxonomy-enriched system prompt
# ---------------------------------------------------------------------------

PARENT_SYSTEM_PROMPT = """\
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

# ---------------------------------------------------------------------------
# Subcategory (48-label) taxonomy system prompt
# ---------------------------------------------------------------------------

SUBCATEGORY_SYSTEM_PROMPT = """\
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


# ---------------------------------------------------------------------------
# Few-shot example selection
# ---------------------------------------------------------------------------

def select_few_shot_examples(train_csv: str, categories: list[str],
                             n_per_cat: int = N_EXAMPLES_PER_CAT) -> list[dict]:
    """Select n diverse examples per category from the training set.

    Strategy: for each category, prefer examples with fewer total labels
    (clearer signal) and shorter narratives (save context budget).
    Tracks used ACNs to avoid duplicates across categories.
    """
    df = pd.read_csv(train_csv)
    examples = []
    used_acns = set()

    df["_label_count"] = df[categories].sum(axis=1)
    df["_nlen"] = df["Narrative"].str.len()

    for cat in categories:
        cat_df = df[df[cat] == 1].copy()
        cat_df = cat_df.sort_values(["_label_count", "_nlen"])
        cat_df = cat_df[~cat_df["ACN"].isin(used_acns)]

        selected = cat_df.head(n_per_cat)
        for _, row in selected.iterrows():
            labels = [c for c in categories if row[c] == 1]
            examples.append({
                "narrative": row["Narrative"][:600],
                "labels": labels,
            })
            used_acns.add(row["ACN"])

    avg_labels = sum(len(ex["labels"]) for ex in examples) / len(examples) if examples else 0
    print(f"Selected {len(examples)} few-shot examples from {len(categories)} categories")
    print(f"  Avg labels per example: {avg_labels:.1f}")
    print(f"  Unique ACNs used: {len(used_acns)}")
    return examples


def build_messages(narrative: str, examples: list[dict],
                   system_prompt: str) -> list[dict]:
    """Build chat messages with system prompt, few-shot examples, and test narrative."""
    messages = [{"role": "system", "content": system_prompt}]

    for ex in examples:
        messages.append({"role": "user", "content": f"Classify this ASRS report:\n\n{ex['narrative']}"})
        messages.append({"role": "assistant", "content": json.dumps(ex["labels"])})

    narrative = narrative[:1500]
    messages.append({"role": "user", "content": f"Classify this ASRS report:\n\n{narrative}"})
    return messages


# ---------------------------------------------------------------------------
# Thinking-mode helpers
# ---------------------------------------------------------------------------

def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from model output."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def extract_thinking_stats(raw_outputs: list[str]) -> dict:
    """Compute statistics on <think> block lengths (for models that inline thinking)."""
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


def extract_thinking_stats_from_reasoning(reasoning_outputs: list[str]) -> dict:
    """Compute statistics on GLM-5's reasoning_content field lengths."""
    lengths = []
    count_with = 0
    for rc in reasoning_outputs:
        if rc and rc.strip():
            count_with += 1
            lengths.append(len(rc))
    stats = {
        "total_outputs": len(reasoning_outputs),
        "outputs_with_thinking": count_with,
        "pct_with_thinking": count_with / len(reasoning_outputs) * 100 if reasoning_outputs else 0,
    }
    if lengths:
        stats["avg_chars"] = statistics.mean(lengths)
        stats["median_chars"] = statistics.median(lengths)
        stats["max_chars"] = max(lengths)
        stats["min_chars"] = min(lengths)
    else:
        stats["avg_chars"] = 0
        stats["median_chars"] = 0
        stats["max_chars"] = 0
        stats["min_chars"] = 0
    return stats


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def _normalize_parent(items: list, cat_lower: dict[str, str]) -> list[str]:
    """Map parsed items to exact parent category names via case-insensitive match.

    Handles subcategory format (e.g., "Aircraft Equipment Problem: Less Severe")
    by stripping the ": subcategory" suffix before matching.
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
        # Strip subcategory suffix
        if ":" in key:
            prefix = key.split(":")[0].strip()
            if prefix in cat_lower:
                cat = cat_lower[prefix]
                if cat not in seen:
                    result.append(cat)
                    seen.add(cat)
    return result


def _normalize_subcategory(items: list, cat_lower: dict[str, str]) -> list[str]:
    """Map parsed items to exact subcategory names via case-insensitive match.

    Handles:
      - Exact match: "Aircraft Equipment Problem: Critical" -> match
      - Number prefix: "1. ATC Issue" -> "ATC Issue" -> match
      - Description suffix: "ATC Issue — Problems with ATC" -> "ATC Issue" -> match
    No parent-only fallback.
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


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def parse_llm_output(raw: str, categories: list[str],
                     is_subcategory: bool = False) -> tuple[list[str], str]:
    """Parse LLM output into a list of valid category names.

    Returns (labels, tier) where tier is 'json', 'regex', 'fuzzy', or 'empty'.

    Three-tier strategy:
      1. Direct JSON parse (with code fence + thinking block stripping)
      2. Regex extraction of JSON array from surrounding text
      3. Fuzzy substring matching of category names (longest first for subcategory)
    """
    cat_lower = {c.lower(): c for c in categories}
    normalize = _normalize_subcategory if is_subcategory else _normalize_parent

    # Strip thinking blocks and code fences
    cleaned = strip_thinking(raw)
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    # Tier 1: direct JSON parse
    try:
        parsed = json.loads(cleaned.strip())
        if isinstance(parsed, list):
            result = normalize(parsed, cat_lower)
            if result:
                return result, "json"
    except (json.JSONDecodeError, TypeError):
        pass

    # Tier 2: regex — find first [...] block
    m = re.search(r"\[.*?\]", cleaned, re.DOTALL)
    if m:
        try:
            parsed = json.loads(m.group())
            if isinstance(parsed, list):
                result = normalize(parsed, cat_lower)
                if result:
                    return result, "regex"
        except (json.JSONDecodeError, TypeError):
            pass

    # Tier 3: fuzzy substring matching (longest labels first for subcategory)
    cleaned_lower = cleaned.lower()
    if is_subcategory:
        sorted_cats = sorted(categories, key=len, reverse=True)
    else:
        sorted_cats = categories
    matched = [c for c in sorted_cats if c.lower() in cleaned_lower]
    if matched:
        return matched, "fuzzy"

    return [], "empty"


# ---------------------------------------------------------------------------
# Metrics
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


# ---------------------------------------------------------------------------
# API call with retry
# ---------------------------------------------------------------------------

def call_api(client, messages: list[dict]) -> tuple[str, str]:
    """Call GLM-5 API with retry logic.

    Returns (content, reasoning_content). GLM-5 puts the JSON answer in
    `content` and chain-of-thought thinking in `reasoning_content`.
    """
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            msg = response.choices[0].message
            content = msg.content or ""
            reasoning = getattr(msg, "reasoning_content", None) or ""
            # Defensive: if content is empty but reasoning has JSON, use it
            if not content.strip() and reasoning.strip():
                content = reasoning
            return content, reasoning
        except Exception as e:
            err_str = str(e).lower()
            is_rate_limit = ("429" in str(e) or "rate" in err_str
                             or "too many" in err_str)
            wait = RATE_LIMIT_WAIT if is_rate_limit else BACKOFF_BASE * (3 ** attempt)
            if attempt < MAX_RETRIES - 1:
                print(f"    Retry {attempt + 1}/{MAX_RETRIES} "
                      f"({'rate limit' if is_rate_limit else 'error'}): "
                      f"{str(e)[:120]}. Waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"    All retries exhausted: {str(e)[:120]}")
                return "", ""


# ---------------------------------------------------------------------------
# Run experiment
# ---------------------------------------------------------------------------

def run_experiment(mode: str):
    """Run parent or subcategory experiment."""
    from openai import OpenAI

    assert mode in ("parent", "subcategory")
    is_sub = mode == "subcategory"

    # Config per mode
    if is_sub:
        test_csv = "data/subcategory_test_set.csv"
        train_csv = "data/subcategory_train_set.csv"
        system_prompt = SUBCATEGORY_SYSTEM_PROMPT
        prefix = "glm5_subcategory"
    else:
        test_csv = "data/test_set.csv"
        train_csv = "data/train_set.csv"
        system_prompt = PARENT_SYSTEM_PROMPT
        prefix = "glm5_parent"

    checkpoint_path = os.path.join(RESULTS_DIR, f"{prefix}_checkpoint.json")
    metrics_path = os.path.join(RESULTS_DIR, f"{prefix}_metrics.csv")
    raw_path = os.path.join(RESULTS_DIR, f"{prefix}_raw_outputs.csv")
    summary_path = os.path.join(RESULTS_DIR, f"{prefix}_summary.txt")

    print(f"\n{'=' * 70}")
    print(f"GLM-5 {'Subcategory (48-label)' if is_sub else 'Parent (13-label)'} Experiment")
    print(f"{'=' * 70}")

    # Load test data
    df = pd.read_csv(test_csv)
    categories = [c for c in df.columns if c not in ("ACN", "Narrative")]
    narratives = df["Narrative"].tolist()
    acns = df["ACN"].tolist()
    n_test = len(df)
    print(f"Loaded {n_test} test reports, {len(categories)} categories")

    # Select few-shot examples
    examples = select_few_shot_examples(train_csv, categories)

    # Token budget check for subcategory (rough estimate)
    if is_sub:
        est_tokens = len(examples) * 2 * 200 + 3000  # ~200 tokens per message pair + system
        print(f"  Estimated few-shot token budget: ~{est_tokens:,} tokens")
        if est_tokens > 150_000:
            print("  WARNING: Token budget exceeds 150K, reducing to 1 example/category")
            examples = select_few_shot_examples(train_csv, categories, n_per_cat=1)

    # Load checkpoint
    # checkpoint stores {"acn": ..., "raw_output": ..., "reasoning": ...}
    completed = {}      # acn -> content (JSON answer)
    reasoning_map = {}  # acn -> reasoning_content (thinking)
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                checkpoint_data = json.load(f)
            for item in checkpoint_data:
                completed[str(item["acn"])] = item["raw_output"]
                reasoning_map[str(item["acn"])] = item.get("reasoning", "")
            print(f"Resumed from checkpoint: {len(completed)}/{n_test} completed")
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"WARNING: Corrupt checkpoint file ({e}), starting fresh")
            completed = {}
            reasoning_map = {}

    # Initialize API client
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY,
        timeout=TIMEOUT,
    )

    # --- Inference ---
    t_start = time.time()
    failed_acns = []
    new_requests = 0  # count only new (non-checkpoint) requests

    for i, (acn, narrative) in enumerate(zip(acns, narratives)):
        acn_key = str(acn)

        # Skip if already completed
        if acn_key in completed:
            continue

        messages = build_messages(narrative, examples, system_prompt)
        content, reasoning = call_api(client, messages)

        if not content:
            failed_acns.append(acn_key)

        completed[acn_key] = content
        reasoning_map[acn_key] = reasoning
        new_requests += 1

        # Save checkpoint after every successful response
        checkpoint_data = [
            {"acn": k, "raw_output": v, "reasoning": reasoning_map.get(k, "")}
            for k, v in completed.items()
        ]
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(checkpoint_data, f, ensure_ascii=False)

        # Progress report every 100 new requests
        if new_requests % 100 == 0 or len(completed) == n_test:
            done = len(completed)
            elapsed = time.time() - t_start
            pct = done / n_test * 100
            if new_requests > 0 and elapsed > 0:
                rate = new_requests / elapsed
                remaining = (n_test - done) / rate if rate > 0 else 0
                elapsed_m = elapsed / 60
                remaining_m = remaining / 60
                print(f"  [{mode}] {done}/{n_test} ({pct:.1f}%) — "
                      f"elapsed: {elapsed_m:.1f}m, est remaining: {remaining_m:.0f}m")
            else:
                print(f"  [{mode}] {done}/{n_test} ({pct:.1f}%)")

        # Delay between requests
        time.sleep(DELAY)

    total_time = time.time() - t_start
    print(f"\nInference done: {len(completed)}/{n_test} in {total_time / 60:.1f} min")
    if failed_acns:
        print(f"Failed ACNs ({len(failed_acns)}): {failed_acns[:20]}"
              + ("..." if len(failed_acns) > 20 else ""))

    # --- Collect outputs in test set order ---
    all_raw = [completed.get(str(acn), "") for acn in acns]
    all_reasoning = [reasoning_map.get(str(acn), "") for acn in acns]

    # --- Thinking mode stats (from reasoning_content field) ---
    print("\n--- Thinking Mode Statistics ---")
    thinking_stats = extract_thinking_stats_from_reasoning(all_reasoning)
    print(f"Outputs with reasoning: "
          f"{thinking_stats['outputs_with_thinking']}/{thinking_stats['total_outputs']} "
          f"({thinking_stats['pct_with_thinking']:.1f}%)")
    if thinking_stats['outputs_with_thinking'] > 0:
        print(f"Reasoning length (chars): "
              f"avg={thinking_stats['avg_chars']:.0f}, "
              f"median={thinking_stats['median_chars']:.0f}, "
              f"max={thinking_stats['max_chars']}, "
              f"min={thinking_stats['min_chars']}")

    # --- Parse outputs ---
    print("\nParsing LLM outputs...")
    parsed_labels = []
    tier_counts = {"json": 0, "regex": 0, "fuzzy": 0, "empty": 0}

    for raw in all_raw:
        labels, tier = parse_llm_output(raw, categories, is_subcategory=is_sub)
        parsed_labels.append(labels)
        tier_counts[tier] += 1

    parse_failures = tier_counts["empty"]
    fail_rate = parse_failures / n_test * 100
    print(f"Parse tiers: {tier_counts}")
    print(f"Parse failures (empty): {parse_failures}/{n_test} ({fail_rate:.1f}%)")

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
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved {metrics_path}")

    # --- Save raw outputs CSV ---
    raw_df = df.copy()
    raw_df["llm_raw_output"] = all_raw
    raw_df["parsed_labels"] = [json.dumps(l) for l in parsed_labels]
    for cat in categories:
        raw_df[f"pred_{cat}"] = y_pred[:, categories.index(cat)]
    raw_df.to_csv(raw_path, index=False)
    print(f"Saved {raw_path}")

    # --- Save summary ---
    summary = format_summary(metrics_rows, n_test, mode, thinking_stats,
                             tier_counts, total_time, len(failed_acns))
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"Saved {summary_path}")

    # --- Print summary ---
    print(f"\n{summary}")

    # --- Print comparison ---
    print_comparison(metrics_rows, mode)

    # --- Cleanup checkpoint ---
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print("\nCheckpoint file removed (run complete)")


# ---------------------------------------------------------------------------
# Summary formatting
# ---------------------------------------------------------------------------

def format_summary(metrics_rows, n_test, mode, thinking_stats,
                   tier_counts, wall_time, n_failed):
    """Format a human-readable summary."""
    is_sub = mode == "subcategory"
    col_width = 55 if is_sub else 35
    line_width = col_width + 42

    label_desc = "Subcategory (48-label)" if is_sub else "Parent (13-label)"
    lines = [
        f"GLM-5 Few-Shot + Thinking: {label_desc} Classification",
        "=" * line_width,
        f"Test set: {n_test} reports | Model: {MODEL}",
        f"API: {API_BASE_URL}, temperature={TEMPERATURE}, max_tokens={MAX_TOKENS}",
        f"Few-shot: {N_EXAMPLES_PER_CAT} examples/category, 600 char truncation",
        f"Thinking: on by default (GLM-5 native)",
        f"Wall time: {wall_time / 60:.1f} min | Failed requests: {n_failed}",
        f"Parse tiers: {tier_counts}",
        f"Thinking stats: {thinking_stats['outputs_with_thinking']}/{thinking_stats['total_outputs']} "
        f"with <think> blocks ({thinking_stats['pct_with_thinking']:.1f}%)",
        "",
        f"{'Category':<{col_width}} {'Precision':>10} {'Recall':>10} {'F1':>10} {'ROC-AUC':>10}",
        "-" * line_width,
    ]
    for row in metrics_rows:
        cat = row["Category"]
        if cat in ("MACRO", "MICRO"):
            continue
        lines.append(
            f"{cat:<{col_width}} {row['Precision']:>10.4f} {row['Recall']:>10.4f} "
            f"{row['F1']:>10.4f} {row['ROC-AUC']:>10.4f}"
        )
    lines.append("-" * line_width)
    for label in ("MACRO", "MICRO"):
        row = next(r for r in metrics_rows if r["Category"] == label)
        lines.append(
            f"{label:<{col_width}} {row['Precision']:>10.4f} {row['Recall']:>10.4f} "
            f"{row['F1']:>10.4f} {row['ROC-AUC']:>10.4f}"
        )

    # Headline metrics
    macro_row = next(r for r in metrics_rows if r["Category"] == "MACRO")
    micro_row = next(r for r in metrics_rows if r["Category"] == "MICRO")
    lines.append("")
    lines.append(f">>> Macro-F1: {macro_row['F1']:.4f}  |  Micro-F1: {micro_row['F1']:.4f} <<<")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Comparison printing
# ---------------------------------------------------------------------------

def print_comparison(glm_rows, mode):
    """Print comparison with existing baselines."""
    is_sub = mode == "subcategory"

    if is_sub:
        classic_path = os.path.join(RESULTS_DIR, "classic_ml_subcategory_metrics.csv")
        mistral_path = os.path.join(RESULTS_DIR, "mistral_large_subcategory_metrics.csv")
        col_width = 55
    else:
        classic_path = os.path.join(RESULTS_DIR, "classic_ml_text_metrics.csv")
        mistral_path = os.path.join(RESULTS_DIR, "mistral_large_zs_metrics.csv")
        col_width = 35

    classic_f1 = {}
    mistral_f1 = {}

    if os.path.exists(classic_path):
        classic = pd.read_csv(classic_path)
        classic_f1 = dict(zip(classic["Category"], classic["F1"]))

    if os.path.exists(mistral_path):
        mistral = pd.read_csv(mistral_path)
        mistral_f1 = dict(zip(mistral["Category"], mistral["F1"]))

    hdr_width = col_width + 50
    print("\n" + "=" * hdr_width)
    print(f"{'Category':<{col_width}} {'Classic ML':>12} {'Mistral L':>12} {'GLM-5':>12} {'vs ML3':>10}")
    print("-" * hdr_width)
    for row in glm_rows:
        cat = row["Category"]
        gf1 = row["F1"]
        cf1 = classic_f1.get(cat, float("nan"))
        mf1 = mistral_f1.get(cat, float("nan"))
        delta = gf1 - mf1 if not (mf1 != mf1) else float("nan")
        print(f"{cat:<{col_width}} {cf1:>12.4f} {mf1:>12.4f} {gf1:>12.4f} {delta:>+10.4f}")
    print("=" * hdr_width)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/glm5_zero_shot.py [parent|subcategory|both]")
        sys.exit(1)

    mode = sys.argv[1].lower()

    if mode not in ("parent", "subcategory", "both"):
        print(f"Unknown mode: {mode}")
        print("Usage: python scripts/glm5_zero_shot.py [parent|subcategory|both]")
        sys.exit(1)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    if mode == "both":
        run_experiment("parent")
        run_experiment("subcategory")
    else:
        run_experiment(mode)


if __name__ == "__main__":
    main()
