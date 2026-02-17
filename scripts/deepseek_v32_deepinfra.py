"""DeepSeek V3.2 zero-shot classification of ASRS reports via DeepInfra API.

Async with aiohttp for high concurrency.
Supports parent (13-label) and subcategory (48-label) classification.
Optional --reasoning flag enables thinking mode.

Usage:
    python scripts/deepseek_v32_deepinfra.py --mode parent
    python scripts/deepseek_v32_deepinfra.py --mode subcategory
    python scripts/deepseek_v32_deepinfra.py --mode both
    python scripts/deepseek_v32_deepinfra.py --mode both --reasoning
"""

import argparse
import asyncio
import json
import os
import re
import time
from difflib import SequenceMatcher

import aiohttp
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

API_URL = "https://api.deepinfra.com/v1/openai/chat/completions"
API_KEY = os.environ.get("DEEPINFRA_API_KEY", "")
MODEL = "deepseek-ai/DeepSeek-V3.2"
TEMPERATURE = 0.0
MAX_TOKENS_DEFAULT = 500
MAX_TOKENS_REASONING = 4096

CONCURRENCY = 50
MAX_RETRIES = 3
BACKOFF_BASE = 2  # 2s, 6s, 18s
CHECKPOINT_EVERY = 100

RESULTS_DIR = "results"

# DeepInfra pricing for DeepSeek V3.2
COST_INPUT_PER_M = 0.26       # $/M uncached input tokens
COST_INPUT_CACHED_PER_M = 0.13  # $/M cached input tokens
COST_OUTPUT_PER_M = 0.38      # $/M output tokens

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
# Thinking mode helpers
# ---------------------------------------------------------------------------

def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from text."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def _normalize_parent(items: list, cat_lower: dict[str, str]) -> list[str]:
    """Map parsed items to exact parent category names."""
    result = []
    seen = set()
    for item in items:
        if not isinstance(item, str):
            continue
        key = item.strip().lower()
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
    """Map parsed items to exact subcategory names."""
    result = []
    seen = set()
    for item in items:
        if not isinstance(item, str):
            continue
        key = item.strip()
        key = re.sub(r'^\d+\.\s*', '', key)
        if ' \u2014 ' in key:
            key = key.split(' \u2014 ')[0].strip()
        if ' -- ' in key:
            key = key.split(' -- ')[0].strip()
        lower_key = key.lower()
        if lower_key in cat_lower:
            cat = cat_lower[lower_key]
            if cat not in seen:
                result.append(cat)
                seen.add(cat)
    return result


def _fuzzy_match(item: str, categories: list[str],
                 threshold: float = 0.8) -> str | None:
    """Find closest category match using difflib, if above threshold."""
    best_ratio = 0.0
    best_cat = None
    item_lower = item.strip().lower()
    for cat in categories:
        ratio = SequenceMatcher(None, item_lower, cat.lower()).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_cat = cat
    if best_ratio >= threshold:
        return best_cat
    return None


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_llm_output(raw: str, categories: list[str],
                     is_subcategory: bool = False) -> tuple[list[str], str]:
    """Parse LLM output into valid category names.

    Returns (labels, tier): 'json', 'regex', 'fuzzy', or 'empty'.
    """
    cat_lower = {c.lower(): c for c in categories}
    normalize = _normalize_subcategory if is_subcategory else _normalize_parent

    # Strip code fences
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    # Tier 1: direct JSON parse
    try:
        parsed = json.loads(cleaned.strip())
        if isinstance(parsed, list):
            result = normalize(parsed, cat_lower)
            if result:
                return result, "json"
            # Try fuzzy matching on unmatched items
            fuzzy_result = []
            for item in parsed:
                if isinstance(item, str):
                    fm = _fuzzy_match(item, categories)
                    if fm and fm not in fuzzy_result:
                        fuzzy_result.append(fm)
            if fuzzy_result:
                return fuzzy_result, "json_fuzzy"
    except (json.JSONDecodeError, TypeError):
        pass

    # Tier 2: regex extract [...] block
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
    """Compute per-category and aggregate metrics."""
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
            "Category": cat, "Precision": p, "Recall": r,
            "F1": f1, "ROC-AUC": auc,
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
# Async API calls
# ---------------------------------------------------------------------------

async def call_api(session: aiohttp.ClientSession, semaphore: asyncio.Semaphore,
                   narrative: str, system_prompt: str, idx: int,
                   pbar: tqdm, reasoning: bool = False,
                   max_tokens: int = MAX_TOKENS_DEFAULT) -> tuple[int, str, dict]:
    """Call DeepInfra API for one report. Returns (idx, raw_output, usage_dict)."""
    narrative = narrative[:1500]
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Classify this ASRS report:\n\n{narrative}"},
        ],
        "temperature": TEMPERATURE,
        "max_tokens": max_tokens,
    }
    if reasoning:
        payload["reasoning_effort"] = "high"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    for attempt in range(MAX_RETRIES):
        async with semaphore:
            try:
                async with session.post(API_URL, json=payload, headers=headers,
                                        timeout=aiohttp.ClientTimeout(total=120)) as resp:
                    if resp.status == 429:
                        wait = BACKOFF_BASE * (3 ** attempt) + 10
                        await asyncio.sleep(wait)
                        continue
                    if resp.status >= 500:
                        wait = BACKOFF_BASE * (3 ** attempt)
                        await asyncio.sleep(wait)
                        continue

                    data = await resp.json()

                    if resp.status != 200:
                        if attempt < MAX_RETRIES - 1:
                            await asyncio.sleep(BACKOFF_BASE * (3 ** attempt))
                            continue
                        pbar.update(1)
                        return idx, "", {}

                    msg = data["choices"][0]["message"]
                    content = msg.get("content", "") or ""
                    # Strip <think> blocks if present in content
                    if "<think>" in content:
                        content = strip_thinking(content)
                    # If content empty but reasoning_content exists, model
                    # may have put answer there (defensive fallback)
                    if not content.strip() and msg.get("reasoning_content"):
                        content = msg["reasoning_content"]
                    usage = data.get("usage", {})
                    pbar.update(1)
                    return idx, content, usage

            except (aiohttp.ClientError, asyncio.TimeoutError, Exception) as e:
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(BACKOFF_BASE * (3 ** attempt))
                    continue
                pbar.update(1)
                return idx, "", {}

    pbar.update(1)
    return idx, "", {}


async def run_all_requests(narratives: list[str], system_prompt: str,
                           completed: dict[str, str], acns: list,
                           checkpoint_path: str, mode: str,
                           reasoning: bool = False,
                           max_tokens: int = MAX_TOKENS_DEFAULT) -> tuple[dict, dict]:
    """Run all API requests with concurrency and checkpointing."""
    semaphore = asyncio.Semaphore(CONCURRENCY)
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0,
                   "cached_tokens": 0}

    # Find remaining work
    remaining = []
    for i, (acn, narr) in enumerate(zip(acns, narratives)):
        if str(acn) not in completed:
            remaining.append((i, acn, narr))

    if not remaining:
        print(f"All {len(completed)} reports already completed from checkpoint")
        return completed, total_usage

    print(f"Remaining: {len(remaining)} requests with {CONCURRENCY} concurrency")

    pbar = tqdm(total=len(remaining), desc=f"DeepSeek V3.2 {mode}", unit="req")

    connector = aiohttp.TCPConnector(limit=CONCURRENCY + 10)
    async with aiohttp.ClientSession(connector=connector) as session:
        chunk_size = CHECKPOINT_EVERY
        for chunk_start in range(0, len(remaining), chunk_size):
            chunk = remaining[chunk_start:chunk_start + chunk_size]

            tasks = [
                call_api(session, semaphore, narr, system_prompt, i, pbar,
                         reasoning=reasoning, max_tokens=max_tokens)
                for i, acn, narr in chunk
            ]
            results = await asyncio.gather(*tasks)

            for i, raw_output, usage in results:
                acn_key = str(acns[i])
                completed[acn_key] = raw_output
                total_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
                total_usage["completion_tokens"] += usage.get("completion_tokens", 0)
                # Track cached tokens from DeepInfra response
                ptd = usage.get("prompt_tokens_details", {}) or {}
                total_usage["cached_tokens"] += ptd.get("cached_tokens", 0) or 0

            # Save checkpoint
            checkpoint_data = [
                {"acn": k, "raw_output": v} for k, v in completed.items()
            ]
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(checkpoint_data, f, ensure_ascii=False)

            done = len(completed)
            if (chunk_start + chunk_size) % 500 < chunk_size:
                elapsed = pbar.format_dict.get("elapsed", 0)
                rate = pbar.format_dict.get("rate", 0) or 0
                cached = total_usage["cached_tokens"]
                prompt = total_usage["prompt_tokens"]
                cache_pct = cached / prompt * 100 if prompt > 0 else 0
                print(f"\n  [{mode}] {done}/{len(acns)} done, "
                      f"{rate:.1f} req/s, elapsed {elapsed:.0f}s, "
                      f"cache hit {cache_pct:.0f}%")

    pbar.close()
    return completed, total_usage


# ---------------------------------------------------------------------------
# Run experiment
# ---------------------------------------------------------------------------

def run_experiment(mode: str, reasoning: bool = False):
    """Run parent or subcategory experiment."""
    assert mode in ("parent", "subcategory")
    is_sub = mode == "subcategory"
    max_tokens = MAX_TOKENS_REASONING if reasoning else MAX_TOKENS_DEFAULT

    if is_sub:
        test_csv = "data/subcategory_test_set.csv"
        system_prompt = SUBCATEGORY_SYSTEM_PROMPT
        prefix = "deepseek_v32_thinking_subcategory" if reasoning else "deepseek_v32_subcategory"
    else:
        test_csv = "data/test_set.csv"
        system_prompt = PARENT_SYSTEM_PROMPT
        prefix = "deepseek_v32_thinking_parent" if reasoning else "deepseek_v32_parent"

    checkpoint_path = os.path.join(RESULTS_DIR, f"{prefix}_checkpoint.json")
    metrics_path = os.path.join(RESULTS_DIR, f"{prefix}_metrics.csv")
    raw_path = os.path.join(RESULTS_DIR, f"{prefix}_raw_outputs.csv")
    summary_path = os.path.join(RESULTS_DIR, f"{prefix}_summary.txt")

    reasoning_tag = " + Thinking" if reasoning else ""
    print(f"\n{'=' * 70}")
    print(f"DeepSeek V3.2 {'Subcategory (48-label)' if is_sub else 'Parent (13-label)'} Zero-Shot{reasoning_tag}")
    print(f"{'=' * 70}")

    # Load test data
    df = pd.read_csv(test_csv)
    categories = [c for c in df.columns if c not in ("ACN", "Narrative")]
    narratives = df["Narrative"].tolist()
    acns = df["ACN"].tolist()
    n_test = len(df)
    print(f"Loaded {n_test} test reports, {len(categories)} categories")

    # Load checkpoint
    completed = {}
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                checkpoint_data = json.load(f)
            for item in checkpoint_data:
                completed[str(item["acn"])] = item["raw_output"]
            print(f"Resumed from checkpoint: {len(completed)}/{n_test} completed")
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"WARNING: Corrupt checkpoint ({e}), starting fresh")
            completed = {}

    # --- Inference ---
    t_start = time.time()
    completed, total_usage = asyncio.run(
        run_all_requests(narratives, system_prompt, completed, acns,
                         checkpoint_path, mode, reasoning=reasoning,
                         max_tokens=max_tokens)
    )
    total_time = time.time() - t_start

    # --- Collect raw outputs in test set order ---
    all_raw = [completed.get(str(acn), "") for acn in acns]
    failed_count = sum(1 for r in all_raw if not r.strip())

    print(f"\nInference: {len(completed)}/{n_test} in {total_time / 60:.1f} min")
    print(f"Failed/empty: {failed_count}")
    if total_time > 0:
        print(f"Throughput: {len(completed) / total_time:.1f} req/s")

    # --- Cost estimate ---
    prompt_tok = total_usage["prompt_tokens"]
    comp_tok = total_usage["completion_tokens"]
    cached_tok = total_usage["cached_tokens"]
    uncached_tok = prompt_tok - cached_tok

    cost_cached = cached_tok / 1e6 * COST_INPUT_CACHED_PER_M
    cost_uncached = uncached_tok / 1e6 * COST_INPUT_PER_M
    cost_out = comp_tok / 1e6 * COST_OUTPUT_PER_M
    total_cost = cost_cached + cost_uncached + cost_out

    print(f"\nToken usage: {prompt_tok:,} prompt ({cached_tok:,} cached) + {comp_tok:,} completion")
    print(f"Cost breakdown:")
    print(f"  Input (uncached): {uncached_tok:,} tokens x ${COST_INPUT_PER_M}/M = ${cost_uncached:.3f}")
    print(f"  Input (cached):   {cached_tok:,} tokens x ${COST_INPUT_CACHED_PER_M}/M = ${cost_cached:.3f}")
    print(f"  Output:           {comp_tok:,} tokens x ${COST_OUTPUT_PER_M}/M = ${cost_out:.3f}")
    print(f"  Total: ${total_cost:.3f}")

    # --- Parse outputs ---
    print("\nParsing LLM outputs...")
    parsed_labels = []
    tier_counts = {"json": 0, "json_fuzzy": 0, "regex": 0, "fuzzy": 0, "empty": 0}

    for raw in all_raw:
        labels, tier = parse_llm_output(raw, categories, is_subcategory=is_sub)
        parsed_labels.append(labels)
        tier_counts[tier] += 1

    parse_failures = tier_counts["empty"]
    fail_rate = parse_failures / n_test * 100
    print(f"Parse tiers: {tier_counts}")
    print(f"Parse failures: {parse_failures}/{n_test} ({fail_rate:.1f}%)")

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
    macro_row = next(r for r in metrics_rows if r["Category"] == "MACRO")
    micro_row = next(r for r in metrics_rows if r["Category"] == "MICRO")

    col_width = 55 if is_sub else 35
    line_width = col_width + 42
    label_desc = "Subcategory (48-label)" if is_sub else "Parent (13-label)"

    thinking_label = " + Thinking" if reasoning else ""
    lines = [
        f"DeepSeek V3.2 Zero-Shot{thinking_label}: {label_desc} Classification",
        "=" * line_width,
        f"Test set: {n_test} reports | Model: {MODEL}",
        f"API: DeepInfra, temperature={TEMPERATURE}, max_tokens={max_tokens}, reasoning={reasoning}",
        f"Zero-shot: taxonomy-enriched prompt, {CONCURRENCY} concurrent",
        f"Wall time: {total_time / 60:.1f} min | Failed: {failed_count} | Cost: ${total_cost:.3f}",
        f"Tokens: {prompt_tok:,} prompt ({cached_tok:,} cached) + {comp_tok:,} completion",
        f"Parse tiers: {tier_counts}",
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
    lines.append("")
    lines.append(f">>> Macro-F1: {macro_row['F1']:.4f}  |  Micro-F1: {micro_row['F1']:.4f} <<<")

    summary = "\n".join(lines)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"Saved {summary_path}")

    # --- Save summary JSON ---
    summary_json = {
        "model": MODEL,
        "mode": mode,
        "reasoning": reasoning,
        "n_test": n_test,
        "n_categories": len(categories),
        "macro_f1": macro_row["F1"],
        "micro_f1": micro_row["F1"],
        "macro_auc": macro_row["ROC-AUC"],
        "wall_time_min": total_time / 60,
        "prompt_tokens": prompt_tok,
        "completion_tokens": comp_tok,
        "cached_tokens": cached_tok,
        "estimated_cost_usd": total_cost,
        "failed_requests": failed_count,
        "parse_tiers": tier_counts,
    }
    json_path = os.path.join(RESULTS_DIR, f"{prefix}_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2)
    print(f"Saved {json_path}")

    # --- Print summary ---
    print(f"\n{summary}")

    # --- Print comparison ---
    print_comparison(metrics_rows, mode)

    # --- Cleanup checkpoint ---
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print("\nCheckpoint file removed (run complete)")


def print_comparison(ds_rows, mode):
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
    print(f"{'Category':<{col_width}} {'Classic ML':>12} {'Mistral L':>12} {'DS V3.2':>12} {'vs ML3':>10}")
    print("-" * hdr_width)
    for row in ds_rows:
        cat = row["Category"]
        df1 = row["F1"]
        cf1 = classic_f1.get(cat, float("nan"))
        mf1 = mistral_f1.get(cat, float("nan"))
        delta = df1 - mf1 if not (mf1 != mf1) else float("nan")
        print(f"{cat:<{col_width}} {cf1:>12.4f} {mf1:>12.4f} {df1:>12.4f} {delta:>+10.4f}")
    print("=" * hdr_width)

    macro_row = next(r for r in ds_rows if r["Category"] == "MACRO")
    micro_row = next(r for r in ds_rows if r["Category"] == "MICRO")
    print(f"\n>>> DeepSeek V3.2 {mode}: Macro-F1 = {macro_row['F1']:.4f}, "
          f"Micro-F1 = {micro_row['F1']:.4f} <<<")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="DeepSeek V3.2 zero-shot via DeepInfra")
    parser.add_argument("--mode", choices=["parent", "subcategory", "both"],
                        required=True, help="Classification level")
    parser.add_argument("--reasoning", action="store_true", default=False,
                        help="Enable thinking/reasoning mode (max_tokens=2048)")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    if args.mode == "both":
        run_experiment("parent", reasoning=args.reasoning)
        run_experiment("subcategory", reasoning=args.reasoning)
    else:
        run_experiment(args.mode, reasoning=args.reasoning)


if __name__ == "__main__":
    main()
