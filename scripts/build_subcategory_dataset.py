"""
Build subcategory-level multi-label dataset from raw ASRS CSVs.

Replicates the dedup pipeline from notebook 01, then parses subcategory-level
anomaly labels, applies a 500-mention threshold, and creates train/test splits.

Output files:
  data/asrs_subcategory_multilabel.csv  (~172K rows, ACN + Narrative + 48 binary cols)
  data/subcategory_train_set.csv        (~31,850 rows)
  data/subcategory_test_set.csv         (~8,044 rows)
  results/subcategory_label_summary.txt (label counts, merge decisions, distributions)
"""
import pandas as pd
import numpy as np
import glob
import os
import time
from collections import Counter

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "raw data")
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

FULL_PATH = os.path.join(DATA_DIR, "asrs_subcategory_multilabel.csv")
TRAIN_PATH = os.path.join(DATA_DIR, "subcategory_train_set.csv")
TEST_PATH = os.path.join(DATA_DIR, "subcategory_test_set.csv")
SUMMARY_PATH = os.path.join(RESULTS_DIR, "subcategory_label_summary.txt")

# ---------------------------------------------------------------------------
# Subcategory prefix mapping: (parent_name, raw_string_prefix)
# Each raw anomaly string starts with one of these prefixes, followed by the
# subcategory name. The prefix is stripped to extract the subcategory.
# ---------------------------------------------------------------------------
SUBCATEGORY_PREFIXES = [
    ("Aircraft Equipment Problem", "Aircraft Equipment Problem "),
    ("Airspace Violation",         "Airspace Violation "),
    ("ATC Issue",                  "ATC Issue "),
    ("Conflict",                   "Conflict "),
    ("Deviation - Altitude",       "Deviation - Altitude "),
    ("Deviation - Procedural",     "Deviation / Discrepancy - Procedural "),
    ("Deviation - Speed",          "Deviation - Speed "),
    ("Deviation - Track/Heading",  "Deviation - Track / Heading "),
    ("Flight Deck/Cabin Event",    "Flight Deck / Cabin / Aircraft Event "),
    ("Ground Event/Encounter",     "Ground Event / Encounter "),
    ("Ground Excursion",           "Ground Excursion "),
    ("Ground Incursion",           "Ground Incursion "),
    ("Inflight Event/Encounter",   "Inflight Event / Encounter "),
]

# Parents that have only "All Types" — keep as parent-level columns
ALL_TYPES_PARENTS = {
    "Airspace Violation",
    "ATC Issue",
    "Deviation - Speed",
    "Deviation - Track/Heading",
}

THRESHOLD = 500  # minimum mentions to survive as its own column


def merge_anomalies(series):
    """Combine anomaly strings from duplicate ACN rows, keeping unique values."""
    all_vals = []
    for val in series.dropna():
        for part in str(val).split(";"):
            part = part.strip()
            if part and part not in all_vals:
                all_vals.append(part)
    return "; ".join(all_vals) if all_vals else np.nan


def first_non_null(series):
    """Return the first non-null value in a series."""
    vals = series.dropna()
    return vals.iloc[0] if len(vals) > 0 else np.nan


def parse_subcategory(raw_str):
    """Parse a single raw anomaly string into (parent, subcategory) or None."""
    for parent, prefix in SUBCATEGORY_PREFIXES:
        if raw_str.startswith(prefix):
            subcat = raw_str[len(prefix):].strip()
            return (parent, subcat)
    return None


def combine_narratives(row):
    """Combine Narrative and Narrative.1 with a space separator."""
    parts = []
    if pd.notna(row["Narrative"]):
        parts.append(str(row["Narrative"]).strip())
    if pd.notna(row["Narrative.1"]):
        parts.append(str(row["Narrative.1"]).strip())
    return " ".join(parts) if parts else np.nan


# ===================================================================
# STEP 1: Load & Dedup (replicate notebook 01 exactly)
# ===================================================================
def step1_load_and_dedup():
    """Load 61 CSVs, concatenate, dedup by ACN."""
    print("=" * 70)
    print("STEP 1: Load & Dedup")
    print("=" * 70)

    files = sorted(glob.glob(os.path.join(RAW_DIR, "*.csv")))
    print(f"Found {len(files)} CSV files")

    t0 = time.time()
    dfs = [pd.read_csv(f, header=1, low_memory=False) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df.columns = df.columns.str.strip()
    print(f"Total rows: {len(df):,} ({time.time() - t0:.1f}s)")

    print(f"Unique ACNs: {df['ACN'].nunique():,}")

    t0 = time.time()
    dedup = df.groupby("ACN", sort=False).agg({
        "Anomaly": merge_anomalies,
        "Narrative": first_non_null,
        "Narrative.1": first_non_null,
        "Synopsis": first_non_null,
    }).reset_index()
    print(f"After dedup: {len(dedup):,} rows ({time.time() - t0:.1f}s)")

    return dedup


# ===================================================================
# STEP 2: Parse anomaly strings into (parent, subcategory) pairs
# ===================================================================
def step2_parse_subcategories(dedup):
    """Parse all anomaly strings into (parent, subcategory) pairs and count."""
    print("\n" + "=" * 70)
    print("STEP 2: Parse Subcategories")
    print("=" * 70)

    # Count (parent, subcategory) mentions across all reports
    subcat_counts = Counter()
    for val in dedup["Anomaly"].dropna():
        seen = set()
        for part in str(val).split(";"):
            part = part.strip()
            if not part:
                continue
            parsed = parse_subcategory(part)
            if parsed and parsed not in seen:
                subcat_counts[parsed] += 1
                seen.add(parsed)

    print(f"Total unique (parent, subcategory) pairs: {len(subcat_counts)}")

    # Group by parent
    by_parent = {}
    for (parent, subcat), count in subcat_counts.items():
        by_parent.setdefault(parent, []).append((subcat, count))

    for parent in by_parent:
        by_parent[parent].sort(key=lambda x: -x[1])

    print("\nPer-parent breakdown:")
    for parent, prefix in SUBCATEGORY_PREFIXES:
        if parent not in by_parent:
            continue
        subcats = by_parent[parent]
        total = sum(c for _, c in subcats)
        print(f"\n  {parent} ({len(subcats)} subcategories, {total:,} mentions)")
        for subcat, count in subcats:
            print(f"    {count:>7,}  {subcat}")

    return subcat_counts, by_parent


# ===================================================================
# STEP 3: Apply 500-mention threshold
# ===================================================================
def step3_apply_threshold(by_parent):
    """Apply threshold, merge rare subcategories into Other/Unknown."""
    print("\n" + "=" * 70)
    print(f"STEP 3: Apply {THRESHOLD}-Mention Threshold")
    print("=" * 70)

    # Build the final column set with merge decisions
    # Format: { column_name: set_of_subcategory_names_it_covers }
    final_columns = {}
    merge_log = []  # human-readable log of merge decisions

    for parent, _ in SUBCATEGORY_PREFIXES:
        if parent in ALL_TYPES_PARENTS:
            # Single "All Types" -> keep parent name as column
            final_columns[parent] = {s for s, _ in by_parent.get(parent, [])}
            merge_log.append(f"  {parent}: parent-level column (All Types)")
            continue

        subcats = by_parent.get(parent, [])
        above = [(s, c) for s, c in subcats if c >= THRESHOLD]
        below = [(s, c) for s, c in subcats if c < THRESHOLD]

        # Check if parent already has an Other/Unknown bucket
        has_other = any(s == "Other / Unknown" for s, _ in subcats)
        other_count = sum(c for s, c in subcats if s == "Other / Unknown")

        if not below:
            # All subcategories above threshold
            for subcat, count in above:
                # Normalize subcategory name
                col_subcat = subcat.replace(" / ", "/")
                col_name = f"{parent}: {col_subcat}"
                final_columns[col_name] = {subcat}
            merge_log.append(f"  {parent}: all {len(above)} subcategories above threshold")
        else:
            # Some below threshold -> merge into Other/Unknown
            merged_names = []
            merged_total = 0

            for subcat, count in below:
                if subcat == "Other / Unknown":
                    continue  # will be handled separately
                merged_names.append(subcat)
                merged_total += count

            # Add existing Other/Unknown count
            merged_total += other_count
            merged_subcats = set(merged_names)
            if has_other:
                merged_subcats.add("Other / Unknown")

            # Keep above-threshold subcategories (excluding Other/Unknown which gets merged)
            for subcat, count in above:
                if subcat == "Other / Unknown":
                    # This will be part of the merged bucket
                    continue
                col_subcat = subcat.replace(" / ", "/")
                col_name = f"{parent}: {col_subcat}"
                final_columns[col_name] = {subcat}

            # Handle the merged Other/Unknown bucket
            if merged_total >= THRESHOLD:
                col_name = f"{parent}: Other/Unknown"
                final_columns[col_name] = merged_subcats
                absorbed = ", ".join(f"{s} ({c})" for s, c in below if s != "Other / Unknown")
                merge_log.append(
                    f"  {parent}: {len(above) - (1 if has_other else 0)} kept + "
                    f"Other/Unknown ({merged_total:,} merged from: {absorbed or 'none'})"
                )
            else:
                # Merged bucket still below threshold -> drop entirely
                dropped_names = ", ".join(f"{s} ({c})" for s, c in below)
                merge_log.append(
                    f"  {parent}: {len(above)} kept, "
                    f"dropped Other/Unknown ({merged_total:,} < {THRESHOLD}, from: {dropped_names})"
                )

    # Sort columns alphabetically
    sorted_columns = sorted(final_columns.keys())

    print(f"Surviving columns: {len(sorted_columns)}")
    print("\nMerge decisions:")
    for line in merge_log:
        print(line)

    print(f"\nFinal column list ({len(sorted_columns)}):")
    for col in sorted_columns:
        covered = final_columns[col]
        if len(covered) > 1:
            print(f"  {col}  <- merges: {', '.join(sorted(covered))}")
        else:
            print(f"  {col}")

    return final_columns, sorted_columns, merge_log


# ===================================================================
# STEP 4: Build binary multi-label matrix
# ===================================================================
def step4_build_label_matrix(dedup, final_columns, sorted_columns):
    """Build binary matrix: one column per surviving (sub)category."""
    print("\n" + "=" * 70)
    print("STEP 4: Build Binary Multi-Label Matrix")
    print("=" * 70)

    # Build reverse lookup: (parent, subcategory) -> column_name
    subcat_to_col = {}
    for col_name, covered_subcats in final_columns.items():
        # Determine parent from column name
        if ":" in col_name:
            parent = col_name.split(":")[0].strip()
        else:
            parent = col_name
        for subcat in covered_subcats:
            subcat_to_col[(parent, subcat)] = col_name

    t0 = time.time()
    # Initialize matrix
    n_rows = len(dedup)
    n_cols = len(sorted_columns)
    col_idx = {col: i for i, col in enumerate(sorted_columns)}
    matrix = np.zeros((n_rows, n_cols), dtype=np.int8)

    for row_i, anomaly_str in enumerate(dedup["Anomaly"].values):
        if pd.isna(anomaly_str):
            continue
        seen = set()
        for part in str(anomaly_str).split(";"):
            part = part.strip()
            if not part:
                continue
            parsed = parse_subcategory(part)
            if parsed is None:
                continue
            col_name = subcat_to_col.get(parsed)
            if col_name and col_name not in seen:
                matrix[row_i, col_idx[col_name]] = 1
                seen.add(col_name)

    label_df = pd.DataFrame(matrix, columns=sorted_columns)
    label_df["n_labels"] = matrix.sum(axis=1)

    elapsed = time.time() - t0
    print(f"Matrix built: {n_rows:,} x {n_cols} ({elapsed:.1f}s)")
    print(f"Reports with >= 1 label: {(label_df['n_labels'] > 0).sum():,}")
    print(f"Reports with 0 labels: {(label_df['n_labels'] == 0).sum():,}")

    return label_df


# ===================================================================
# STEP 5: Combine narratives & save full dataset
# ===================================================================
def step5_save_full_dataset(dedup, label_df, sorted_columns):
    """Combine narratives, filter, and save."""
    print("\n" + "=" * 70)
    print("STEP 5: Combine Narratives & Save")
    print("=" * 70)

    dedup = dedup.copy()
    dedup["Narrative_combined"] = dedup.apply(combine_narratives, axis=1)

    # Merge ACN, Narrative_combined with label matrix
    output = pd.DataFrame({
        "ACN": dedup["ACN"].values,
        "Narrative": dedup["Narrative_combined"].values,
    })
    for col in sorted_columns:
        output[col] = label_df[col].values
    output["n_labels"] = label_df["n_labels"].values

    # Filter: keep reports with n_labels > 0 AND Narrative not null
    before = len(output)
    output = output[(output["n_labels"] > 0) & (output["Narrative"].notna())].copy()
    output = output.drop(columns=["n_labels"]).reset_index(drop=True)
    after = len(output)

    print(f"Before filter: {before:,}")
    print(f"After filter (n_labels > 0 & Narrative not null): {after:,}")
    print(f"Dropped: {before - after:,}")

    output.to_csv(FULL_PATH, index=False)
    print(f"Saved {FULL_PATH}")
    print(f"Columns: ACN + Narrative + {len(sorted_columns)} label columns = {len(output.columns)} total")

    return output


# ===================================================================
# STEP 6: Two-stage stratified train/test split
# ===================================================================
def step6_stratified_split(output, sorted_columns):
    """Two-stage split: 172K -> ~40K sample -> ~32K train / ~8K test."""
    print("\n" + "=" * 70)
    print("STEP 6: Stratified Train/Test Split")
    print("=" * 70)

    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

    y = output[sorted_columns].values
    X_idx = np.arange(len(output))

    # Stage 1: full -> ~40K sample
    t0 = time.time()
    splitter_40k = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=40_000, random_state=42
    )
    _, sample_idx = next(splitter_40k.split(X_idx, y))
    sample_df = output.iloc[sample_idx].reset_index(drop=True)
    print(f"Stage 1: {len(output):,} -> {len(sample_df):,} sample ({time.time() - t0:.1f}s)")

    # Stage 2: ~40K -> ~32K train / ~8K test
    t0 = time.time()
    y_sample = sample_df[sorted_columns].values
    X_sample_idx = np.arange(len(sample_df))
    splitter_split = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=8_000, random_state=42
    )
    train_idx, test_idx = next(splitter_split.split(X_sample_idx, y_sample))
    train_df = sample_df.iloc[train_idx].reset_index(drop=True)
    test_df = sample_df.iloc[test_idx].reset_index(drop=True)
    print(f"Stage 2: {len(sample_df):,} -> train {len(train_df):,} / test {len(test_df):,} ({time.time() - t0:.1f}s)")

    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)
    print(f"Saved {TRAIN_PATH}")
    print(f"Saved {TEST_PATH}")

    return train_df, test_df


# ===================================================================
# STEP 7: Save summary
# ===================================================================
def step7_save_summary(output, train_df, test_df, sorted_columns, final_columns, merge_log, subcat_counts, by_parent):
    """Save comprehensive summary file."""
    print("\n" + "=" * 70)
    print("STEP 7: Save Summary")
    print("=" * 70)

    lines = []
    lines.append("ASRS Subcategory-Level Multi-Label Dataset Summary")
    lines.append("=" * 60)
    lines.append("")

    # Dataset sizes
    lines.append(f"Full dataset: {len(output):,} reports")
    lines.append(f"Train set: {len(train_df):,} reports")
    lines.append(f"Test set: {len(test_df):,} reports")
    lines.append(f"Label columns: {len(sorted_columns)}")
    lines.append(f"Threshold: {THRESHOLD} mentions minimum")
    lines.append("")

    # Label count distribution
    for split_name, df in [("Full", output), ("Train", train_df), ("Test", test_df)]:
        lc = df[sorted_columns].sum(axis=1)
        lines.append(f"{split_name} label count: mean={lc.mean():.2f}, median={lc.median():.1f}, "
                      f"min={lc.min()}, max={lc.max()}")
    lines.append("")

    # Zero-label check
    zero_full = (output[sorted_columns].sum(axis=1) == 0).sum()
    zero_train = (train_df[sorted_columns].sum(axis=1) == 0).sum()
    zero_test = (test_df[sorted_columns].sum(axis=1) == 0).sum()
    lines.append(f"Zero-label reports: full={zero_full}, train={zero_train}, test={zero_test}")
    lines.append("")

    # Merge decisions
    lines.append("=" * 60)
    lines.append("MERGE DECISIONS (threshold = 500)")
    lines.append("=" * 60)
    for line in merge_log:
        lines.append(line)
    lines.append("")

    # Per-column distribution comparison
    lines.append("=" * 60)
    lines.append("PER-COLUMN DISTRIBUTION")
    lines.append("=" * 60)
    lines.append(f"{'Column':<50} {'Full':>8} {'Train':>8} {'Test':>8} {'Train%':>8} {'Test%':>8}")
    lines.append("-" * 82)

    for col in sorted_columns:
        c_full = output[col].sum()
        c_train = train_df[col].sum()
        c_test = test_df[col].sum()
        p_train = c_train / len(train_df) * 100
        p_test = c_test / len(test_df) * 100
        lines.append(f"{col:<50} {c_full:>8,} {c_train:>8,} {c_test:>8,} {p_train:>7.2f}% {p_test:>7.2f}%")

    lines.append("")

    # Imbalance ratio
    col_sums = output[sorted_columns].sum()
    max_col = col_sums.idxmax()
    min_col = col_sums.idxmin()
    ratio = col_sums.max() / col_sums.min()
    lines.append(f"Imbalance ratio: {ratio:.1f}x")
    lines.append(f"  Most common:  {max_col} ({col_sums.max():,})")
    lines.append(f"  Least common: {min_col} ({col_sums.min():,})")
    lines.append("")

    # Raw subcategory counts for reference
    lines.append("=" * 60)
    lines.append("RAW SUBCATEGORY COUNTS (before threshold)")
    lines.append("=" * 60)
    for parent, _ in SUBCATEGORY_PREFIXES:
        subcats = by_parent.get(parent, [])
        lines.append(f"\n  {parent}:")
        for subcat, count in subcats:
            marker = "  [OK]" if count >= THRESHOLD else f"  [DROPPED] (<{THRESHOLD})"
            lines.append(f"    {count:>7,}  {subcat}{marker}")

    summary_text = "\n".join(lines)
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write(summary_text)
    print(f"Saved {SUMMARY_PATH}")
    print(f"\n{summary_text}")


# ===================================================================
# Main
# ===================================================================
def main():
    total_t0 = time.time()

    # Check if all outputs already exist
    if (os.path.exists(FULL_PATH) and os.path.exists(TRAIN_PATH)
            and os.path.exists(TEST_PATH) and os.path.exists(SUMMARY_PATH)):
        print("All output files already exist. Loading for verification...")
        output = pd.read_csv(FULL_PATH)
        train_df = pd.read_csv(TRAIN_PATH)
        test_df = pd.read_csv(TEST_PATH)
        sorted_columns = [c for c in output.columns if c not in ("ACN", "Narrative")]
        print(f"Full: {len(output):,}  Train: {len(train_df):,}  Test: {len(test_df):,}")
        print(f"Label columns: {len(sorted_columns)}")
        with open(SUMMARY_PATH, encoding="utf-8") as f:
            print(f.read())
        return

    # Run pipeline
    dedup = step1_load_and_dedup()
    subcat_counts, by_parent = step2_parse_subcategories(dedup)
    final_columns, sorted_columns, merge_log = step3_apply_threshold(by_parent)
    label_df = step4_build_label_matrix(dedup, final_columns, sorted_columns)
    output = step5_save_full_dataset(dedup, label_df, sorted_columns)
    train_df, test_df = step6_stratified_split(output, sorted_columns)
    step7_save_summary(output, train_df, test_df, sorted_columns,
                       final_columns, merge_log, subcat_counts, by_parent)

    total_elapsed = time.time() - total_t0
    print(f"\nTotal pipeline time: {total_elapsed:.1f}s")


if __name__ == "__main__":
    main()
