"""Mistral Large 3 zero-shot classification on 48-label subcategory dataset.

Supports two modes:
  - Batch API: file-based batch (cheaper, but may queue)
  - Real-time API: individual requests with rate limiting and checkpointing

Usage:
    python scripts/mistral_large_subcategory.py submit     # Build & submit batch job
    python scripts/mistral_large_subcategory.py poll        # Check job status
    python scripts/mistral_large_subcategory.py results     # Download batch results & compute metrics
    python scripts/mistral_large_subcategory.py realtime    # Real-time API with checkpointing (~25 min)
    python scripts/mistral_large_subcategory.py             # Auto: submit if no job, else poll/results
"""

import json
import os
import re
import sys
import time

import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

API_KEY = "8uo0c1gFEzkVS8j1eEVzO1Jb6yHXEp1B"
MODEL = "mistral-large-latest"
TEMPERATURE = 0.0
MAX_TOKENS = 512

RESULTS_DIR = "results"
JOB_ID_PATH = os.path.join(RESULTS_DIR, "mistral_large_subcategory_job_id.txt")
CHECKPOINT_PATH = os.path.join(RESULTS_DIR, "mistral_large_subcategory_checkpoint.json")

TEST_CSV = "data/subcategory_test_set.csv"

# Real-time API rate limiting (Scale plan: 6 req/s, 2M input tokens/min)
REQUESTS_PER_SECOND = 5  # stay under 6 RPS limit
CHECKPOINT_EVERY = 100   # save progress every N requests

# ---------------------------------------------------------------------------
# Subcategory taxonomy system prompt (48 labels)
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


def build_messages(narrative: str) -> list[dict]:
    """Build chat messages with system prompt and test narrative (no examples)."""
    narrative = narrative[:1500]
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Classify this ASRS report:\n\n{narrative}"},
    ]


# ---------------------------------------------------------------------------
# Batch API operations
# ---------------------------------------------------------------------------

def submit_batch():
    """Build JSONL file, upload it, and submit to Mistral Batch API."""
    from mistralai import Mistral

    client = Mistral(api_key=API_KEY)

    # Load data
    df = pd.read_csv(TEST_CSV)
    categories = [c for c in df.columns if c not in ("ACN", "Narrative")]
    narratives = df["Narrative"].tolist()
    acns = df["ACN"].tolist()
    n_test = len(df)
    print(f"Loaded {n_test} test reports, {len(categories)} categories")

    # Build JSONL file
    jsonl_path = os.path.join(RESULTS_DIR, "mistral_large_subcategory_batch.jsonl")
    print(f"\nBuilding {n_test} batch requests to {jsonl_path}...")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for acn, narrative in zip(acns, narratives):
            messages = build_messages(narrative)
            line = json.dumps({
                "custom_id": str(acn),
                "body": {
                    "temperature": TEMPERATURE,
                    "max_tokens": MAX_TOKENS,
                    "messages": messages,
                },
            }, ensure_ascii=False)
            f.write(line + "\n")

    file_size_mb = os.path.getsize(jsonl_path) / (1024 * 1024)
    print(f"Written {jsonl_path} ({file_size_mb:.1f} MB)")

    # Upload file
    print("Uploading JSONL file...")
    with open(jsonl_path, "rb") as fh:
        batch_data = client.files.upload(
            file={
                "file_name": "asrs_subcategory_batch.jsonl",
                "content": fh,
            },
            purpose="batch",
        )
    print(f"Uploaded file: {batch_data.id}")

    # Create batch job
    print(f"Creating batch job with model {MODEL}...")
    job = client.batch.jobs.create(
        input_files=[batch_data.id],
        model=MODEL,
        endpoint="/v1/chat/completions",
        metadata={"experiment": "asrs-mistral-large-subcategory"},
    )

    print(f"Batch job created: {job.id}")
    print(f"  Status: {job.status}")
    print(f"  Total requests: {job.total_requests}")

    # Save job ID for later polling
    with open(JOB_ID_PATH, "w") as f:
        f.write(job.id)
    print(f"Job ID saved to {JOB_ID_PATH}")

    # Clean up local JSONL
    os.remove(jsonl_path)
    print(f"Removed local {jsonl_path}")

    return job.id


def poll_job(job_id: str = None):
    """Check batch job status."""
    from mistralai import Mistral

    client = Mistral(api_key=API_KEY)

    if job_id is None:
        if not os.path.exists(JOB_ID_PATH):
            print("No job ID found. Run 'submit' first.")
            return None
        with open(JOB_ID_PATH) as f:
            job_id = f.read().strip()

    job = client.batch.jobs.get(job_id=job_id)
    print(f"Job {job.id}")
    print(f"  Status: {job.status}")
    print(f"  Total requests: {job.total_requests}")
    print(f"  Succeeded: {job.succeeded_requests}")
    print(f"  Failed: {job.failed_requests}")
    if hasattr(job, "created_at") and job.created_at:
        print(f"  Created: {job.created_at}")
    if hasattr(job, "completed_at") and job.completed_at:
        print(f"  Completed: {job.completed_at}")
    if hasattr(job, "output_file") and job.output_file:
        print(f"  Output file: {job.output_file}")

    return job


def run_realtime():
    """Send requests via real-time API with rate limiting and checkpointing."""
    from mistralai import Mistral

    client = Mistral(api_key=API_KEY)

    # Load test data
    df = pd.read_csv(TEST_CSV)
    categories = [c for c in df.columns if c not in ("ACN", "Narrative")]
    narratives = df["Narrative"].tolist()
    acns = df["ACN"].tolist()
    n_test = len(df)
    print(f"Loaded {n_test} test reports, {len(categories)} categories")

    # Load checkpoint if exists
    results_map = {}
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
            results_map = json.load(f)
        print(f"Resumed from checkpoint: {len(results_map)}/{n_test} completed")

    # Find remaining requests
    remaining = [(acn, narr) for acn, narr in zip(acns, narratives)
                 if str(acn) not in results_map]
    print(f"Remaining: {len(remaining)} requests at ~{REQUESTS_PER_SECOND} req/s")
    if remaining:
        eta_sec = len(remaining) / REQUESTS_PER_SECOND
        print(f"Estimated time: {eta_sec / 60:.1f} min")

    t_start = time.time()
    errors = 0

    for i, (acn, narrative) in enumerate(remaining):
        t_req = time.time()

        messages = build_messages(narrative)
        try:
            response = client.chat.complete(
                model=MODEL,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            content = response.choices[0].message.content
        except Exception as e:
            content = ""
            errors += 1
            if errors <= 5:
                print(f"  ERROR on ACN {acn}: {e}")
            if "429" in str(e) or "rate" in str(e).lower():
                print("  Rate limited — sleeping 10s...")
                time.sleep(10)

        results_map[str(acn)] = content

        # Checkpoint
        if (i + 1) % CHECKPOINT_EVERY == 0:
            with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
                json.dump(results_map, f, ensure_ascii=False)
            done = len(results_map)
            elapsed = time.time() - t_start
            rps = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(remaining) - i - 1) / rps if rps > 0 else 0
            print(f"  [{done}/{n_test}] {rps:.1f} req/s, "
                  f"elapsed {elapsed / 60:.1f} min, ETA {eta / 60:.1f} min")

        # Rate limiting
        req_time = time.time() - t_req
        min_interval = 1.0 / REQUESTS_PER_SECOND
        if req_time < min_interval:
            time.sleep(min_interval - req_time)

    # Final save
    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        json.dump(results_map, f, ensure_ascii=False)

    total_time = time.time() - t_start
    print(f"\nDone! {len(results_map)} results in {total_time / 60:.1f} min")
    if errors:
        print(f"  Errors: {errors}")

    # Process results
    all_raw = [results_map.get(str(acn), "") for acn in acns]
    process_and_save(df, categories, all_raw)

    # Clean up checkpoint
    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)
        print("Checkpoint file removed (results processed)")


def download_and_process(job_id: str = None):
    """Download batch results, parse outputs, compute metrics."""
    from mistralai import Mistral

    client = Mistral(api_key=API_KEY)

    if job_id is None:
        if not os.path.exists(JOB_ID_PATH):
            print("No job ID found. Run 'submit' first.")
            return
        with open(JOB_ID_PATH) as f:
            job_id = f.read().strip()

    # Check job status
    job = client.batch.jobs.get(job_id=job_id)
    if job.status != "SUCCESS":
        print(f"Job status is {job.status}, not SUCCESS. Cannot process results.")
        if job.status in ("QUEUED", "RUNNING"):
            print("Job is still running. Run 'poll' to check progress.")
        return

    print(f"Job {job.id} completed successfully")
    print(f"  Succeeded: {job.succeeded_requests}")
    print(f"  Failed: {job.failed_requests}")

    # Download results
    print("\nDownloading results...")
    output_stream = client.files.download(file_id=job.output_file)
    raw_content = output_stream.read()

    # Handle bytes vs string
    if isinstance(raw_content, bytes):
        raw_content = raw_content.decode("utf-8")

    results_lines = raw_content.strip().split("\n")
    print(f"Downloaded {len(results_lines)} result lines")

    # Parse batch results into {custom_id: content} map
    results_map = {}
    for line in results_lines:
        item = json.loads(line)
        custom_id = item["custom_id"]
        if item.get("error"):
            results_map[custom_id] = ""
        else:
            content = item["response"]["body"]["choices"][0]["message"]["content"]
            results_map[custom_id] = content

    print(f"Parsed {len(results_map)} results")

    # Load test data and map results
    df = pd.read_csv(TEST_CSV)
    categories = [c for c in df.columns if c not in ("ACN", "Narrative")]
    acns = df["ACN"].tolist()

    all_raw = []
    missing = 0
    for acn in acns:
        key = str(acn)
        if key in results_map:
            all_raw.append(results_map[key])
        else:
            all_raw.append("")
            missing += 1

    if missing > 0:
        print(f"WARNING: {missing} reports missing from results")

    process_and_save(df, categories, all_raw)

    # Cleanup job ID file
    if os.path.exists(JOB_ID_PATH):
        os.remove(JOB_ID_PATH)
        print("\nJob ID file removed (results processed)")


def process_and_save(df, categories, all_raw):
    """Parse LLM outputs, compute metrics, save all output files."""
    n_test = len(df)

    # Parse outputs
    print("\nParsing LLM outputs...")
    parsed_labels = []
    parse_failures = 0

    for raw in all_raw:
        result = parse_llm_output(raw, categories)
        if not result:
            parse_failures += 1
        parsed_labels.append(result)

    fail_rate = parse_failures / n_test * 100
    print(f"Parse failures (empty output): {parse_failures}/{n_test} ({fail_rate:.1f}%)")

    # Build prediction matrix
    y_pred = np.zeros((n_test, len(categories)), dtype=int)
    for i, labels in enumerate(parsed_labels):
        for cat in labels:
            if cat in categories:
                y_pred[i, categories.index(cat)] = 1

    y_true = df[categories].values

    # Compute metrics
    print("\nComputing metrics...")
    metrics_rows = compute_metrics(y_true, y_pred, categories)

    # Save metrics CSV
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = os.path.join(RESULTS_DIR, "mistral_large_subcategory_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved {metrics_path}")

    # Save raw outputs CSV
    raw_df = df.copy()
    raw_df["llm_raw_output"] = all_raw
    raw_df["parsed_labels"] = [json.dumps(l) for l in parsed_labels]
    for cat in categories:
        raw_df[f"pred_{cat}"] = y_pred[:, categories.index(cat)]
    raw_path = os.path.join(RESULTS_DIR, "mistral_large_subcategory_raw_outputs.csv")
    raw_df.to_csv(raw_path, index=False)
    print(f"Saved {raw_path}")

    # Save summary
    parent_comparison = make_parent_comparison(metrics_rows, categories)
    summary = format_summary(metrics_rows, n_test, parent_comparison)
    summary_path = os.path.join(RESULTS_DIR, "mistral_large_subcategory_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"Saved {summary_path}")

    # Print results
    print(f"\n{summary}")

    # Print comparison with classic ML subcategory baseline
    print_comparison(metrics_rows)


# ---------------------------------------------------------------------------
# Parsing helpers (adapted for 48 subcategory labels)
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


def format_summary(metrics_rows, n_test, parent_comparison):
    """Format a human-readable summary with parent-group comparison."""
    lines = [
        "Zero-Shot LLM: Mistral Large 3 — Subcategory (48-label) Classification",
        "=" * 97,
        f"Test set: {n_test} reports | Model: {MODEL}",
        f"Batch API: temperature={TEMPERATURE}, max_tokens={MAX_TOKENS}",
        "Zero-shot: taxonomy-enriched prompt with 48 NASA ASRS subcategories, no examples",
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
    """Build parent-level comparison table (same logic as modal_classic_ml_subcategory.py)."""
    from collections import defaultdict

    def get_parent(col):
        if ":" in col:
            return col.split(":")[0].strip()
        return col

    # Load parent-level Classic ML baseline
    baseline_path = os.path.join(RESULTS_DIR, "classic_ml_text_metrics.csv")
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

    # Per-subcategory F1
    sub_f1 = {
        r["Category"]: r["F1"]
        for r in metrics_rows
        if r["Category"] not in ("MACRO", "MICRO")
    }
    parent_map = {col: get_parent(col) for col in categories}

    # Group subcategory F1 by parent
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


def print_comparison(subcategory_rows):
    """Print comparison: Classic ML subcategory vs Mistral Large subcategory."""
    classic_path = os.path.join(RESULTS_DIR, "classic_ml_subcategory_metrics.csv")

    classic_f1 = {}
    if os.path.exists(classic_path):
        classic = pd.read_csv(classic_path)
        classic_f1 = dict(zip(classic["Category"], classic["F1"]))

    print("\n" + "=" * 97)
    print(
        f"{'Category':<55} {'Classic ML':>12} {'ML3 ZS':>12} {'Delta':>10}"
    )
    print("-" * 97)
    for row in subcategory_rows:
        cat = row["Category"]
        zf1 = row["F1"]
        cf1 = classic_f1.get(cat, float("nan"))
        delta = zf1 - cf1 if not (cf1 != cf1) else float("nan")
        print(f"{cat:<55} {cf1:>12.4f} {zf1:>12.4f} {delta:>+10.4f}")
    print("=" * 97)

    # Print headline
    macro_row = next(r for r in subcategory_rows if r["Category"] == "MACRO")
    micro_row = next(r for r in subcategory_rows if r["Category"] == "MICRO")
    print(
        f"\n>>> Mistral Large 3 Subcategory ZS: "
        f"Macro-F1 = {macro_row['F1']:.4f}, Micro-F1 = {micro_row['F1']:.4f} <<<"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    action = sys.argv[1] if len(sys.argv) > 1 else "auto"

    if action == "submit":
        submit_batch()

    elif action == "poll":
        poll_job()

    elif action == "results":
        download_and_process()

    elif action == "realtime":
        run_realtime()

    elif action == "auto":
        # Auto mode: submit if no job, poll if running, process if done
        if os.path.exists(JOB_ID_PATH):
            job = poll_job()
            if job and job.status == "SUCCESS":
                print("\nJob complete! Processing results...\n")
                download_and_process()
            elif job and job.status in ("FAILED", "TIMEOUT_EXCEEDED", "CANCELLED"):
                print(f"\nJob {job.status}. Removing job ID file.")
                os.remove(JOB_ID_PATH)
        else:
            submit_batch()

    else:
        print(f"Unknown action: {action}")
        print(
            "Usage: python scripts/mistral_large_subcategory.py "
            "[submit|poll|results|realtime|auto]"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
