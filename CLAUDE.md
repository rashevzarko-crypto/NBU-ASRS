# NBU-ASRS: Aviation Safety Report Classification

## Session Workflow — ALWAYS FOLLOW
1. **Start of session:** Read `STATUS.md` before doing any work. Check what's complete, what's in progress, what's next.
2. **End of session / after completing a task:** Update `STATUS.md` with current progress, new results, files created, compute costs.
3. **Before starting any experiment:** Verify train/test split files exist and row counts match STATUS.md.

## Project Context
Master's thesis at New Bulgarian University: "Multimodal Deep Learning for Runway Incursion Detection." Revised to expand from binary to multi-label classification across 13 ASRS anomaly categories.

## Core Experiment: Four-Way Model Comparison
Classify ASRS reports across 13 anomaly categories (multi-label), comparing:
1. Zero-shot LLM (prompt-based, no examples)
2. Few-shot LLM (examples in prompt)
3. Fine-tuned LLM (QLoRA on base model)
4. Classic ML (TF-IDF + XGBoost on text)

All four models evaluated on the **same frozen test set**.

## Infrastructure
- **VS Code + Jupyter (local, no GPU):** Data processing, classic ML, visualization
- **Modal (cloud GPU, L4 only):** All LLM work — zero-shot, few-shot, fine-tuning, inference
- **Single notebook per phase** with checkpoint logic (if output exists, skip step)

## Repo Structure
- `CLAUDE.md` — static instructions (this file, DO NOT put dynamic data here)
- `STATUS.md` — dynamic state, progress, results, configs (UPDATE AFTER EVERY TASK)
- `notebooks/` — experiment notebooks
- `scripts/` — Modal GPU scripts
- `data/` — processed datasets (NOT raw data)
- `results/` — metrics, plots, summaries
- `raw data/` — source CSVs (gitignored, ~670MB)

## Key Principles
- Technical work first, thesis writing after all results
- Save intermediate outputs to disk, check existence before rerunning
- Track compute costs and training times (log in STATUS.md)
- All experiments use the same frozen test set
- Never hardcode category names — read from CSV headers
- GPU budget: L4 only, ~$50 total for Modal

## Output Format Contract
All experiment metrics CSVs must follow this exact structure:
- **Columns:** Category, Precision, Recall, F1, ROC-AUC
- **Rows:** 13 category rows (alphabetical) + MACRO + MICRO (15 rows total)
- **File naming:** `results/{experiment}_metrics.csv` (e.g., `zero_shot_metrics.csv`, `few_shot_metrics.csv`, `finetune_metrics.csv`)
- **Companion files:** `results/{experiment}_raw_outputs.csv` (per-report predictions), `results/{experiment}_summary.txt` (human-readable table matching `classic_ml_summary.txt` format)
- **Reference file:** `results/classic_ml_text_metrics.csv` is the canonical format — match it exactly

## What NOT to Do
- Don't modify train/test CSVs once created — they are frozen
- Don't run GPU workloads locally — use Modal
- Don't hardcode the 13 category names — always read dynamically
- Don't start thesis writing until all four experiments are complete
- Don't use A100/H100 GPUs — L4 is sufficient and within budget
- Don't put dynamic data (sample sizes, results, progress) in this file — use STATUS.md
