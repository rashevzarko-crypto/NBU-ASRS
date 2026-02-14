"""Mistral Large 3 zero-shot classification of ASRS reports with taxonomy-enriched prompt.

Uses Mistral Batch API — builds all 8,044 requests as file-based batch, submits once,
polls for completion, downloads results. No rate limit issues.

Usage:
    python scripts/mistral_large_zero_shot.py submit    # Build & submit batch job
    python scripts/mistral_large_zero_shot.py poll       # Check job status
    python scripts/mistral_large_zero_shot.py results    # Download results & compute metrics
    python scripts/mistral_large_zero_shot.py            # Auto: submit if no job, else poll/results
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
MAX_TOKENS = 256

RESULTS_DIR = "results"
JOB_ID_PATH = os.path.join(RESULTS_DIR, "mistral_large_zs_job_id.txt")

# ---------------------------------------------------------------------------
# Taxonomy-enriched system prompt (same as few-shot experiment)
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

def submit_batch(test_csv: str):
    """Build JSONL file, upload it, and submit to Mistral Batch API."""
    from mistralai import Mistral

    client = Mistral(api_key=API_KEY)

    # Load data
    df = pd.read_csv(test_csv)
    categories = [c for c in df.columns if c not in ("ACN", "Narrative")]
    narratives = df["Narrative"].tolist()
    acns = df["ACN"].tolist()
    n_test = len(df)
    print(f"Loaded {n_test} test reports, {len(categories)} categories")

    # Build JSONL file
    jsonl_path = os.path.join(RESULTS_DIR, "mistral_large_zs_batch.jsonl")
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
                "file_name": "asrs_zs_batch.jsonl",
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
        metadata={"experiment": "asrs-mistral-large-zero-shot"},
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

    # Load test data
    df = pd.read_csv("data/test_set.csv")
    categories = [c for c in df.columns if c not in ("ACN", "Narrative")]
    acns = df["ACN"].tolist()
    n_test = len(df)

    # Map results to test set order
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
    metrics_path = os.path.join(RESULTS_DIR, "mistral_large_zs_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved {metrics_path}")

    # Save raw outputs CSV
    raw_df = df.copy()
    raw_df["llm_raw_output"] = all_raw
    raw_df["parsed_labels"] = [json.dumps(l) for l in parsed_labels]
    for cat in categories:
        raw_df[f"pred_{cat}"] = y_pred[:, categories.index(cat)]
    raw_path = os.path.join(RESULTS_DIR, "mistral_large_zs_raw_outputs.csv")
    raw_df.to_csv(raw_path, index=False)
    print(f"Saved {raw_path}")

    # Save summary
    summary = format_summary(metrics_rows, n_test)
    summary_path = os.path.join(RESULTS_DIR, "mistral_large_zs_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"Saved {summary_path}")

    # Print results
    print(f"\n{summary}")

    # Print comparison
    print_comparison(metrics_rows)

    # Cleanup job ID file
    if os.path.exists(JOB_ID_PATH):
        os.remove(JOB_ID_PATH)
        print("\nJob ID file removed (results processed)")


# ---------------------------------------------------------------------------
# Parsing helpers (same 3-tier approach with code fence stripping)
# ---------------------------------------------------------------------------

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


def parse_llm_output(raw: str, categories: list[str]) -> list[str]:
    """Parse LLM output into a list of valid category names.

    Three-tier strategy:
      1. Direct JSON parse (with code fence stripping)
      2. Regex extraction of JSON array from surrounding text
      3. Fuzzy substring matching of category names
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

    # Tier 3: fuzzy substring matching
    matched = []
    raw_lower = raw.lower()
    for cat in categories:
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


def format_summary(metrics_rows, n_test):
    """Format a human-readable summary."""
    lines = [
        "Zero-Shot LLM: Mistral Large 3 (Batch API, taxonomy-enriched prompt)",
        "=" * 70,
        f"Test set: {n_test} reports | Model: {MODEL}",
        f"Batch API: temperature={TEMPERATURE}, max_tokens={MAX_TOKENS}",
        "Zero-shot: taxonomy-enriched prompt with NASA ASRS subcategories, no examples",
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


def print_comparison(zs_rows):
    """Print comparison: Classic ML vs Mistral Large few-shot vs Mistral Large zero-shot."""
    classic_path = os.path.join(RESULTS_DIR, "classic_ml_text_metrics.csv")
    fs_path = os.path.join(RESULTS_DIR, "mistral_large_metrics.csv")

    classic_f1 = {}
    fs_f1 = {}

    if os.path.exists(classic_path):
        classic = pd.read_csv(classic_path)
        classic_f1 = dict(zip(classic["Category"], classic["F1"]))

    if os.path.exists(fs_path):
        fs = pd.read_csv(fs_path)
        fs_f1 = dict(zip(fs["Category"], fs["F1"]))

    print("\n" + "=" * 95)
    print(f"{'Category':<35} {'Classic ML':>12} {'ML3 FS':>12} {'ML3 ZS':>12} {'ZS-FS':>10}")
    print("-" * 95)
    for row in zs_rows:
        cat = row["Category"]
        zf1 = row["F1"]
        cf1 = classic_f1.get(cat, float("nan"))
        ff1 = fs_f1.get(cat, float("nan"))
        delta = zf1 - ff1 if not (ff1 != ff1) else float("nan")
        print(f"{cat:<35} {cf1:>12.4f} {ff1:>12.4f} {zf1:>12.4f} {delta:>+10.4f}")
    print("=" * 95)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    action = sys.argv[1] if len(sys.argv) > 1 else "auto"

    if action == "submit":
        submit_batch("data/test_set.csv")

    elif action == "poll":
        poll_job()

    elif action == "results":
        download_and_process()

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
            submit_batch("data/test_set.csv")

    else:
        print(f"Unknown action: {action}")
        print("Usage: python scripts/mistral_large_zero_shot.py [submit|poll|results|auto]")
        sys.exit(1)


if __name__ == "__main__":
    main()
