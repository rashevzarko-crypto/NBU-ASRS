# NBU-ASRS Project Status

Last updated: 2026-02-18 (Removed bug story slide from defense presentations — PPTX + HTML, 12 slides each)

> **Correction (2026-02-17):** Mistral Large 3 (`mistral-large-latest`) is a 675B MoE (41B active) model, NOT ~123B. The 123B figure was Mistral Large 2 (mid-2024, dense). Updated all references across FINAL_EXPERIMENT_SUMMARY.md, thesis_context.md, generate_thesis_context.py, and 03_final_visualizations.ipynb. Regenerated Fig 4 (cost vs performance — corrected point sizes) and Fig 5 (replaced "scale effect" line plot with "Dense vs MoE Architecture" grouped scatter). Also corrected license from "Proprietary" to "Apache 2.0".
>
> **Model switch #1:** Changed from meta-llama/Llama-3.1-8B-Instruct to mistralai/Ministral-3-8B-Instruct-2512 on 2026-02-13 (Llama gate approval delay).
>
> **Model switch #2:** Changed from Ministral-3-8B to Qwen/Qwen3-8B on 2026-02-13. Reason: Ministral 3 8B is stored as a multimodal `Mistral3ForConditionalGeneration` with FP8 quantization, preventing proper QLoRA (4-bit NF4) training. Fine-tuning produced no improvement over zero-shot (Macro-F1: 0.489 vs 0.491). Qwen3-8B is a pure text-only CausalLM, Apache 2.0, no gate, supporting standard QLoRA workflow. Ministral results archived in `results/ministral/`.

## Progress Tracker

| Phase | Status | Output Files |
|-------|--------|-------------|
| Data exploration | ✅ Complete | `data/asrs_multilabel.csv`, `results/data_exploration_summary.txt`, `results/co_occurrence_heatmap.png` |
| Stratified sampling | ✅ Complete | `data/train_set.csv` (31,850), `data/test_set.csv` (8,044) |
| Classic ML baseline | ✅ Complete | `results/classic_ml_text_metrics.csv`, `results/classic_ml_f1_barchart.png` |
| Structured features extraction | ✅ Complete | `data/structured_features.csv` (39,894 rows) |
| Modal scripts scaffolding | ✅ Complete | `scripts/modal_zero_shot.py`, `scripts/modal_few_shot.py`, `scripts/modal_finetune.py` |
| Zero-shot LLM (Ministral) | ✅ Complete (archived) | `results/ministral/zero_shot_*.csv/.txt` |
| Few-shot LLM (Ministral) | ✅ Complete (archived) | `results/ministral/few_shot_*.csv/.txt` |
| Fine-tuned LLM (Ministral) | ✅ Complete (archived) | `results/ministral/finetune_*.csv/.txt` |
| Zero-shot LLM (Qwen3) | ✅ Complete | `results/zero_shot_metrics.csv`, `results/zero_shot_raw_outputs.csv`, `results/zero_shot_summary.txt` |
| Few-shot LLM (Qwen3) | ✅ Complete | `results/few_shot_metrics.csv`, `results/few_shot_raw_outputs.csv`, `results/few_shot_summary.txt` |
| QLoRA fine-tuning (Qwen3) | ✅ Complete | Adapter on Modal volume `asrs-finetune-vol` |
| Fine-tuned LLM inference (Qwen3) | ✅ Complete | `results/finetune_metrics.csv`, `results/finetune_raw_outputs.csv`, `results/finetune_summary.txt` |
| Few-shot LLM (Mistral Large 3) | ✅ Complete | `results/mistral_large_metrics.csv`, `results/mistral_large_raw_outputs.csv`, `results/mistral_large_summary.txt` |
| Classic ML full dataset (164K) | ✅ Complete | `results/classic_ml_full_metrics.csv`, `results/classic_ml_full_summary.txt` |
| Zero-shot taxonomy (Qwen3) | ✅ Complete | `results/zero_shot_taxonomy_metrics.csv`, `results/zero_shot_taxonomy_raw_outputs.csv`, `results/zero_shot_taxonomy_summary.txt` |
| Few-shot taxonomy (Qwen3) | ✅ Complete | `results/few_shot_taxonomy_metrics.csv`, `results/few_shot_taxonomy_raw_outputs.csv`, `results/few_shot_taxonomy_summary.txt` |
| Zero-shot (Mistral Large 3) | ✅ Complete | `results/mistral_large_zs_metrics.csv`, `results/mistral_large_zs_raw_outputs.csv`, `results/mistral_large_zs_summary.txt` |
| Few-shot taxonomy + thinking (Qwen3) | ✅ Complete | `results/few_shot_taxonomy_thinking_metrics.csv`, `results/few_shot_taxonomy_thinking_raw_outputs.csv`, `results/few_shot_taxonomy_thinking_summary.txt` |
| Subcategory dataset build | ✅ Complete | `data/asrs_subcategory_multilabel.csv` (172,173), `data/subcategory_train_set.csv` (32,089), `data/subcategory_test_set.csv` (8,017), `results/subcategory_label_summary.txt` |
| Classic ML subcategory (48 labels) | ✅ Complete | `results/classic_ml_subcategory_metrics.csv`, `results/classic_ml_subcategory_predictions.csv`, `results/classic_ml_subcategory_summary.txt`, `results/classic_ml_subcategory_f1_barchart.png` |
| Zero-shot subcategory (Mistral Large 3) | ✅ Complete | `results/mistral_large_subcategory_metrics.csv`, `results/mistral_large_subcategory_raw_outputs.csv`, `results/mistral_large_subcategory_summary.txt` |
| Zero-shot subcategory (Qwen3-8B) | ✅ Complete | `results/qwen_zero_shot_subcategory_metrics.csv`, `results/qwen_zero_shot_subcategory_raw_outputs.csv`, `results/qwen_zero_shot_subcategory_summary.txt` |
| Zero-shot parent (DeepSeek V3.2) | ✅ Complete | `results/deepseek_v32_parent_metrics.csv`, `results/deepseek_v32_parent_raw_outputs.csv`, `results/deepseek_v32_parent_summary.txt` |
| Zero-shot subcategory (DeepSeek V3.2) | ✅ Complete | `results/deepseek_v32_subcategory_metrics.csv`, `results/deepseek_v32_subcategory_raw_outputs.csv`, `results/deepseek_v32_subcategory_summary.txt` |
| Zero-shot + thinking parent (DeepSeek V3.2) | ✅ Complete | `results/deepseek_v32_thinking_parent_metrics.csv`, `results/deepseek_v32_thinking_parent_raw_outputs.csv`, `results/deepseek_v32_thinking_parent_summary.txt` |
| Zero-shot + thinking subcategory (DeepSeek V3.2) | ✅ Complete | `results/deepseek_v32_thinking_subcategory_metrics.csv`, `results/deepseek_v32_thinking_subcategory_raw_outputs.csv`, `results/deepseek_v32_thinking_subcategory_summary.txt` |
| Classic ML hyperparameter tuning | ✅ Complete | `results/tfidf_ablation.csv`, `results/model_comparison.csv`, `results/classic_ml_tuning_summary.txt` |
| Classic ML tuned final eval | ✅ Complete | `results/classic_ml_tuned_parent_metrics.csv`, `results/classic_ml_tuned_subcategory_metrics.csv`, `results/classic_ml_tuned_parent_summary.txt`, `results/classic_ml_tuned_subcategory_summary.txt`, `results/classic_ml_tuned_result.json` |
| Final comparison & visualization | ✅ Complete | `notebooks/03_final_visualizations.ipynb`, `results/fig_grand_comparison.png`, `results/fig_approach_summary.png`, `results/fig_category_heatmap.png`, `results/fig_cost_vs_performance.png`, `results/fig_scale_effect.png`, `results/fig_sub_grand_comparison.png`, `results/fig_parent_vs_sub.png` |
| Thesis writing (.docx generator) | ✅ Complete | `scripts/generate_thesis_docx.js`, `thesis.docx` (117K+ chars, 30 tables, 9 figures, ~70+ pages) |
| Defense presentation (PPTX + HTML) | ✅ Complete | `scripts/generate_defense_pptx.js`, `scripts/generate_defense_html.js`, `defense_presentation.pptx` (12 slides), `defense_presentation.html` (2.0MB, 12 slides, self-contained) |

## Dataset Parameters

- **Source:** NASA ASRS database, 61 CSV files, 282,371 raw rows
- **Unique reports:** 172,183 (after ACN dedup)
- **Anomaly categories:** 13 top-level ASRS categories (multi-label)
- **Sample size:** 39,894 reports (stratified from 172K)
- **Train set:** 31,850 reports
- **Test set:** 8,044 reports (FROZEN — same for all experiments)
- **Stratification:** MultilabelStratifiedShuffleSplit, random_state=42
- **Imbalance ratio:** 30.3x (Deviation-Procedural: 112K vs Ground Excursion: 3.7K in full dataset)
- **Label distribution:** median 2 labels per report; 22% single-label, 78% multi-label

## Category Distribution (Full 172K Dataset)

| Category | Count | % of reports |
|----------|-------|-------------|
| Deviation - Procedural | 112,606 | 65.4% |
| Aircraft Equipment Problem | 49,305 | 28.6% |
| Conflict | 46,285 | 26.9% |
| Inflight Event/Encounter | 38,658 | 22.5% |
| ATC Issue | 29,422 | 17.1% |
| Deviation - Altitude | 28,369 | 16.5% |
| Deviation - Track/Heading | 20,268 | 11.8% |
| Ground Event/Encounter | 14,234 | 8.3% |
| Ground Incursion | 12,601 | 7.3% |
| Flight Deck/Cabin Event | 12,291 | 7.1% |
| Airspace Violation | 6,834 | 4.0% |
| Deviation - Speed | 5,000 | 2.9% |
| Ground Excursion | 3,718 | 2.2% |

## Model Configurations

### Classic ML (TF-IDF + XGBoost)
- TF-IDF: max_features=50000, ngram_range=(1,2), sublinear_tf=True
- XGBoost: 13 independent binary classifiers, n_estimators=300, max_depth=6, lr=0.1, tree_method=hist, scale_pos_weight per category
- Runs locally, no GPU

### Zero-shot LLM
- Model: Qwen/Qwen3-8B (Apache 2.0, text-only CausalLM)
- GPU: Modal L4 (24GB)
- vLLM: dtype=auto, max_model_len=8192, gpu_memory_utilization=0.90
- chat_template_kwargs: enable_thinking=False
- Prompt: system role + category list + narrative → JSON list output
- Inference on 8,044 test set

### Few-shot LLM
- Same model: Qwen/Qwen3-8B (Apache 2.0)
- GPU: Modal L4 (24GB)
- vLLM: dtype=auto, max_model_len=16384, gpu_memory_utilization=0.90
- chat_template_kwargs: enable_thinking=False
- 3 examples per category (39 total), selected from train set
- Example selection: sort by label_count ascending then narrative length ascending (prefer single-label, shorter)
- Narrative truncation: examples 600 chars, test narratives 1500 chars
- Batch size: 16 (down from 64 in zero-shot due to longer prompts)

### Fine-tuned LLM (QLoRA)
- Base: Qwen/Qwen3-8B
- Training: 4-bit NF4 quantization via BitsAndBytesConfig (proper QLoRA)
- LoRA: r=16, alpha=16, target_modules=[q_proj, v_proj], dropout=0.05
- Training: 31,850 samples, 2 epochs, batch_size=4, grad_accum=4, lr=2e-5
- Scheduler: cosine with warmup_ratio=0.05, optim=paged_adamw_8bit, bf16=True
- max_length=1024, narrative truncation=1500 chars
- Training GPU: Modal A100 (80GB), timeout=14400s (4h)
- Inference GPU: Modal L4 (24GB), vLLM with LoRA adapter
- chat_template_kwargs: enable_thinking=False
- Inference on 8,044 test set, batch_size=64

### Zero-shot LLM (Mistral Large 3)
- Model: mistral-large-latest (Mistral Large 3, proprietary)
- API: Mistral Batch API (free tier), no GPU needed
- temperature=0.0, max_tokens=256
- Taxonomy-enriched system prompt with NASA ASRS subcategories and discriminative hints
- Batch processing: ~5 min for 8,044 reports, 2 parse failures (0.0%)
- Note: original run had 43% parse failures due to _normalize() bug (subcategory colon format); fixed via fix_mistral_large_zs.py

### Few-shot taxonomy + thinking (Qwen3-8B)
- Model: Qwen/Qwen3-8B (Apache 2.0, text-only CausalLM)
- GPU: Modal A100 (80GB) — L4 too slow due to high output token count from thinking
- vLLM: dtype=auto, max_model_len=32768, gpu_memory_utilization=0.90
- chat_template_kwargs: enable_thinking=True (chain-of-thought reasoning)
- temperature=0.0, max_tokens=4096 (thinking blocks can be long)
- Taxonomy-enriched system prompt with NASA ASRS subcategories and discriminative hints
- 3 examples per category (39 total), same selection as few-shot taxonomy
- Batch size: 32
- Thinking stats: 99.6% outputs had `<think>` blocks, avg 2986 chars, median 2513, max 15945
- Parse results: 7990 json, 4 regex, 31 fuzzy, 0 empty (0% failures)
- strip_thinking() regex removes `<think>...</think>` before JSON parsing

### Few-shot LLM (Mistral Large 3)
- Model: mistral-large-latest (Mistral Large 3, proprietary)
- API: Mistral Batch API (free tier), no GPU needed
- temperature=0.0, max_tokens=256
- Taxonomy-enriched system prompt with NASA ASRS subcategories and discriminative hints
- 2 examples per category (26 total), selected from train set
- Example selection: sort by label_count ascending then narrative length ascending (prefer single-label, shorter)
- Narrative truncation: examples 600 chars, test narratives 1500 chars
- Batch processing: ~4 min for 8,044 reports, 0 failures, 2 parse failures (0.0%)

### Zero-shot LLM (DeepSeek V3.2)
- Model: deepseek-ai/DeepSeek-V3.2 (671B MoE, non-reasoning)
- API: DeepInfra OpenAI-compatible, 50 concurrent requests via aiohttp
- temperature=0.0, max_tokens=500
- Taxonomy-enriched system prompt (same as Mistral Large 3)
- Cost: $0.26/M input (uncached), $0.13/M (cached), $0.38/M output
- Prefix caching: ~62% parent, ~79% subcategory
- Parse failures: 3/8044 parent (0.0%), 3/8017 subcategory (0.0%)

### Zero-shot + Thinking LLM (DeepSeek V3.2)
- Model: deepseek-ai/DeepSeek-V3.2 (671B MoE, reasoning mode)
- API: DeepInfra OpenAI-compatible, 50 concurrent requests via aiohttp
- temperature=0.0, max_tokens=4096, reasoning=True
- Taxonomy-enriched system prompt (same as non-thinking)
- Reasoning tokens via `reasoning_content` field (clean JSON in `content`)
- Script: `scripts/deepseek_v32_deepinfra.py` (same script, `--thinking` flag)

## Infrastructure

- **Local (VS Code + Jupyter):** Data processing, classic ML, visualization
- **Modal (cloud GPU):** All LLM experiments (L4 @ ~$0.80/hr)
- **Budget:** $50 total Modal, estimated ~$27 for all experiments
- **GitHub:** github.com/rashevzarko-crypto/NBU-ASRS

## Results

### All Models Comparison

| Model | Prompt | Macro-F1 | Micro-F1 | Macro-AUC |
|-------|--------|----------|----------|-----------|
| Classic ML 32K | — | 0.691 | 0.746 | 0.932 |
| Classic ML 164K | — | 0.678 | 0.739 | 0.942 |
| Mistral Large 3 zero-shot | taxonomy | 0.658 | 0.712 | 0.793 |
| Mistral Large 3 few-shot | taxonomy | 0.640 | 0.686 | 0.793 |
| DeepSeek V3.2 zero-shot + thinking | taxonomy | 0.681 | 0.723 | 0.810 |
| DeepSeek V3.2 zero-shot | taxonomy | 0.623 | 0.693 | 0.746 |
| Ministral 8B few-shot | basic | 0.540 | 0.536 | 0.746 |
| Qwen3-8B few-shot + thinking | taxonomy | 0.533 | 0.556 | 0.705 |
| Qwen3-8B few-shot | taxonomy | 0.526 | 0.544 | 0.706 |
| Qwen3-8B fine-tuned (QLoRA) | basic | 0.510 | 0.632 | 0.700 |
| Qwen3-8B zero-shot | taxonomy | 0.499 | 0.605 | 0.701 |
| Ministral 8B zero-shot | basic | 0.491 | 0.543 | 0.744 |
| Ministral 8B fine-tuned (LoRA/FP8) | basic | 0.489 | 0.542 | 0.744 |
| Qwen3-8B zero-shot | basic | 0.459 | 0.473 | 0.727 |
| Qwen3-8B few-shot | basic | 0.453 | 0.468 | 0.704 |

### Subcategory (48-label) All Models

| Model | Macro-F1 | Micro-F1 | Macro-AUC |
|-------|----------|----------|-----------|
| Classic ML (XGBoost) | 0.510 | 0.600 | 0.934 |
| Mistral Large 3 ZS | 0.449 | 0.494 | 0.744 |
| DeepSeek V3.2 ZS | 0.422 | 0.456 | 0.708 |
| DeepSeek V3.2 ZS + thinking | 0.419 | 0.466 | 0.690 |
| Qwen3-8B ZS | 0.235 | 0.304 | 0.629 |

Parent-group comparison (Classic ML):
- Biggest drops: Ground Event/Encounter (-0.325, 8 subs), Deviation-Procedural (-0.298, 10 subs), Aircraft Equipment Problem (-0.243, 2 subs)
- Slight gains: ATC Issue (+0.011, 1 sub), Airspace Violation (+0.022, 1 sub), Deviation-Track/Heading (+0.008, 1 sub)
- Best subcategories: Hazardous Material Violation (F1=0.824), Smoke/Fire/Fumes/Odor (F1=0.815), Wake Vortex Encounter (F1=0.813)
- Worst subcategories: Weather/Turbulence (Ground) (F1=0.000), Ground Equipment Issue (F1=0.118), Vehicle (F1=0.164)

Mistral Large 3 subcategory highlights:
- Best: Passenger Misconduct (F1=0.876), Smoke/Fire (F1=0.825), Haz Mat (F1=0.791)
- Worst: Undershoot (F1=0.007), Other/Unknown procedural (F1=0.019), Ground Equipment (F1=0.033)
- Beats Classic ML on 11/48 subcategories (notable: Landing Without Clearance +0.194, Gear Up Landing +0.196, UAS +0.177)
- Runtime: 119 min real-time API, 7 network errors, 0.1% parse failures

Qwen3-8B subcategory highlights:
- Best: Smoke/Fire (F1=0.673), Passenger Misconduct (F1=0.596), Clearance (F1=0.584)
- Worst: CFTT/CFIT (F1=0.005), Undershoot (F1=0.015), Ground Equipment (F1=0.029)
- Dramatic drop from parent-level (Macro-F1: 0.499 → 0.235, Micro-F1: 0.605 → 0.304)
- Runtime: ~30 min on Modal L4, vLLM engine crash after inference (results saved)

### Classic ML Hyperparameter Tuning Results

**TF-IDF Ablation (8 configs, 3-fold CV):**
- All configs within 0.005 Macro-F1 (range: 0.6248 - 0.6296)
- Best: no_sublinear (sublinear_tf=False), CV Macro-F1 = 0.6296
- Baseline (sublinear_tf=True), CV Macro-F1 = 0.6280, delta = +0.0016
- Conclusion: TF-IDF parameters have negligible impact

**Model Comparison (RandomizedSearchCV, 3-fold CV):**

| Model | CV Macro-F1 | Test Macro-F1 | Test Micro-F1 |
|-------|------------|---------------|---------------|
| XGBoost | 0.679 | 0.691 | 0.746 |
| LogisticRegression | 0.504 | 0.670 | 0.738 |
| LinearSVC | 0.473 | 0.655 | 0.750 |

- XGBoost: baseline params (300 trees, depth 6, lr 0.1) are near-optimal
- LinearSVC: highest Micro-F1 (0.750) but weaker Macro-F1 (0.655) — biases toward frequent labels
- LogisticRegression: competitive but 3.5h for RSCV due to SAGA solver on sparse 50K features
- Final results: existing baseline (Macro-F1 0.691, Micro-F1 0.746) confirmed as effectively tuned
- Note: Modal billing limit reached during XGB hyperparameter search; baseline params confirmed optimal from 7/16 holdout combos

**Phase 3 Final Evaluation (baseline XGBoost retrained + evaluated on both tasks):**

| Task | Macro-F1 | Micro-F1 | Macro-AUC | vs Baseline |
|------|----------|----------|-----------|-------------|
| Parent (13-label) | 0.6928 | 0.7454 | 0.9321 | +0.002 / -0.001 |
| Subcategory (48-label) | 0.5099 | 0.5998 | 0.9339 | -0.0001 / -0.0002 |

- Retrained with baseline params (300 trees, depth 6, lr 0.1, hist, scale_pos_weight) on full train set
- Per-label scale_pos_weight for class imbalance
- Confirms baseline is effectively optimal — tuning provides negligible improvement
- Script: `scripts/modal_classic_ml_phase3.py` (with Modal Volume for result persistence)

### DeepSeek V3.2 Thinking Analysis

**Parent (13-label):** Macro-F1 0.681 vs 0.623 non-thinking (+0.058). Significant improvement — thinking adds real value at 671B scale.
- Biggest gains: Airspace Violation (+0.126 F1), ATC Issue (+0.160 F1), Ground Excursion (+0.013 F1)
- Cost: $6.73 (vs $1.39 non-thinking, 4.8x more expensive), 291 min (vs 6.5 min, 45x slower)
- 0.6% empty parse failures (46/8044), 14 failed requests

**Subcategory (48-label):** Macro-F1 0.419 vs 0.422 non-thinking (-0.003). Thinking HURT performance.
- 21.6% parse failures (1729/8017 empty) — thinking generates too-long outputs for 48-label task
- Cost: $5.24 (vs $1.92 non-thinking, 2.7x), 545 min (vs 7.5 min, 73x slower)
- Conclusion: thinking mode useful at 671B scale for parent categories, but fails on complex 48-label subcategory task

### Per-Category Results (DeepSeek V3.2 Zero-Shot + Thinking, Parent)

| Category | Precision | Recall | F1 | ROC-AUC |
|----------|-----------|--------|-----|---------|
| Aircraft Equipment Problem | 0.861 | 0.783 | 0.820 | 0.866 |
| Airspace Violation | 0.601 | 0.572 | 0.586 | 0.778 |
| ATC Issue | 0.441 | 0.685 | 0.536 | 0.753 |
| Conflict | 0.833 | 0.840 | 0.837 | 0.889 |
| Deviation - Altitude | 0.739 | 0.815 | 0.775 | 0.879 |
| Deviation - Procedural | 0.797 | 0.721 | 0.757 | 0.688 |
| Deviation - Speed | 0.595 | 0.592 | 0.594 | 0.790 |
| Deviation - Track/Heading | 0.721 | 0.661 | 0.690 | 0.814 |
| Flight Deck/Cabin Event | 0.745 | 0.700 | 0.722 | 0.841 |
| Ground Event/Encounter | 0.487 | 0.631 | 0.549 | 0.785 |
| Ground Excursion | 0.557 | 0.740 | 0.635 | 0.864 |
| Ground Incursion | 0.764 | 0.624 | 0.687 | 0.804 |
| Inflight Event/Encounter | 0.684 | 0.640 | 0.661 | 0.777 |

### Per-Category Results (DeepSeek V3.2 Zero-Shot, Parent)

| Category | Precision | Recall | F1 | ROC-AUC |
|----------|-----------|--------|-----|---------|
| Aircraft Equipment Problem | 0.874 | 0.688 | 0.770 | 0.824 |
| Airspace Violation | 0.663 | 0.352 | 0.460 | 0.672 |
| ATC Issue | 0.483 | 0.308 | 0.376 | 0.620 |
| Conflict | 0.876 | 0.698 | 0.777 | 0.831 |
| Deviation - Altitude | 0.700 | 0.813 | 0.752 | 0.872 |
| Deviation - Procedural | 0.689 | 0.866 | 0.768 | 0.567 |
| Deviation - Speed | 0.785 | 0.361 | 0.494 | 0.679 |
| Deviation - Track/Heading | 0.777 | 0.560 | 0.651 | 0.770 |
| Flight Deck/Cabin Event | 0.788 | 0.585 | 0.671 | 0.786 |
| Ground Event/Encounter | 0.467 | 0.573 | 0.515 | 0.757 |
| Ground Excursion | 0.599 | 0.647 | 0.622 | 0.819 |
| Ground Incursion | 0.795 | 0.549 | 0.649 | 0.769 |
| Inflight Event/Encounter | 0.610 | 0.568 | 0.588 | 0.732 |

### Ministral 3 8B (archived — see `results/ministral/`)

| Model | Macro-F1 | Micro-F1 | Macro-AUC | Notes |
|-------|----------|----------|-----------|-------|
| Classic ML (TF-IDF + XGBoost) | 0.691 | 0.746 | 0.932 | 13 binary classifiers, text only |
| Zero-shot LLM (Ministral) | 0.491 | 0.543 | 0.744 | FP8, zero-shot, 0% parse failures |
| Few-shot LLM (Ministral) | 0.540 | 0.536 | 0.746 | FP8, 3 examples/cat, 0% parse failures |
| Fine-tuned LLM (Ministral) | 0.489 | 0.542 | 0.744 | LoRA on FP8 (not true QLoRA), 0% parse failures |

### Per-Category Results (Mistral Large 3 Zero-Shot, corrected)

| Category | Precision | Recall | F1 | ROC-AUC |
|----------|-----------|--------|-----|---------|
| Aircraft Equipment Problem | 0.819 | 0.812 | 0.816 | 0.870 |
| Airspace Violation | 0.616 | 0.443 | 0.516 | 0.716 |
| ATC Issue | 0.410 | 0.754 | 0.531 | 0.766 |
| Conflict | 0.862 | 0.639 | 0.734 | 0.801 |
| Deviation - Altitude | 0.750 | 0.768 | 0.759 | 0.859 |
| Deviation - Procedural | 0.709 | 0.900 | 0.793 | 0.603 |
| Deviation - Speed | 0.649 | 0.579 | 0.612 | 0.785 |
| Deviation - Track/Heading | 0.707 | 0.656 | 0.680 | 0.810 |
| Flight Deck/Cabin Event | 0.579 | 0.766 | 0.660 | 0.862 |
| Ground Event/Encounter | 0.496 | 0.608 | 0.546 | 0.776 |
| Ground Excursion | 0.498 | 0.786 | 0.610 | 0.884 |
| Ground Incursion | 0.676 | 0.661 | 0.668 | 0.818 |
| Inflight Event/Encounter | 0.604 | 0.654 | 0.628 | 0.765 |

### Per-Category Results (Mistral Large 3 Few-Shot)

| Category | Precision | Recall | F1 | ROC-AUC |
|----------|-----------|--------|-----|---------|
| Aircraft Equipment Problem | 0.918 | 0.640 | 0.754 | 0.809 |
| Airspace Violation | 0.693 | 0.390 | 0.499 | 0.691 |
| ATC Issue | 0.461 | 0.619 | 0.528 | 0.735 |
| Conflict | 0.881 | 0.636 | 0.739 | 0.803 |
| Deviation - Altitude | 0.693 | 0.818 | 0.750 | 0.873 |
| Deviation - Procedural | 0.719 | 0.822 | 0.767 | 0.610 |
| Deviation - Speed | 0.597 | 0.592 | 0.595 | 0.790 |
| Deviation - Track/Heading | 0.638 | 0.744 | 0.687 | 0.844 |
| Flight Deck/Cabin Event | 0.614 | 0.796 | 0.693 | 0.879 |
| Ground Event/Encounter | 0.434 | 0.649 | 0.520 | 0.786 |
| Ground Excursion | 0.422 | 0.798 | 0.552 | 0.887 |
| Ground Incursion | 0.596 | 0.738 | 0.660 | 0.849 |
| Inflight Event/Encounter | 0.453 | 0.784 | 0.574 | 0.756 |

### Per-Category Results (Qwen3-8B Few-Shot Taxonomy + Thinking)

| Category | Precision | Recall | F1 | ROC-AUC |
|----------|-----------|--------|-----|---------|
| Aircraft Equipment Problem | 0.924 | 0.529 | 0.673 | 0.756 |
| Airspace Violation | 0.578 | 0.421 | 0.487 | 0.704 |
| ATC Issue | 0.402 | 0.415 | 0.409 | 0.644 |
| Conflict | 0.878 | 0.646 | 0.744 | 0.806 |
| Deviation - Altitude | 0.771 | 0.520 | 0.622 | 0.745 |
| Deviation - Procedural | 0.858 | 0.360 | 0.507 | 0.624 |
| Deviation - Speed | 0.730 | 0.313 | 0.438 | 0.655 |
| Deviation - Track/Heading | 0.703 | 0.206 | 0.318 | 0.597 |
| Flight Deck/Cabin Event | 0.808 | 0.581 | 0.676 | 0.785 |
| Ground Event/Encounter | 0.522 | 0.428 | 0.471 | 0.697 |
| Ground Excursion | 0.536 | 0.555 | 0.546 | 0.772 |
| Ground Incursion | 0.735 | 0.388 | 0.508 | 0.689 |
| Inflight Event/Encounter | 0.692 | 0.422 | 0.524 | 0.684 |

### Per-Category Results (Fine-tuned Qwen3-8B)

| Category | Precision | Recall | F1 | ROC-AUC |
|----------|-----------|--------|-----|---------|
| Aircraft Equipment Problem | 0.799 | 0.767 | 0.783 | 0.845 |
| Airspace Violation | 0.188 | 0.088 | 0.120 | 0.536 |
| ATC Issue | 0.500 | 0.312 | 0.384 | 0.624 |
| Conflict | 0.839 | 0.636 | 0.724 | 0.796 |
| Deviation - Altitude | 0.608 | 0.796 | 0.689 | 0.847 |
| Deviation - Procedural | 0.691 | 0.817 | 0.749 | 0.566 |
| Deviation - Speed | 0.461 | 0.532 | 0.494 | 0.757 |
| Deviation - Track/Heading | 0.402 | 0.619 | 0.487 | 0.748 |
| Flight Deck/Cabin Event | 0.395 | 0.330 | 0.359 | 0.646 |
| Ground Event/Encounter | 0.542 | 0.496 | 0.518 | 0.729 |
| Ground Excursion | 0.536 | 0.301 | 0.385 | 0.647 |
| Ground Incursion | 0.520 | 0.463 | 0.490 | 0.715 |
| Inflight Event/Encounter | 0.440 | 0.453 | 0.446 | 0.643 |

### Per-Category Results (Classic ML Full 164K)

| Category | Precision | Recall | F1 | ROC-AUC |
|----------|-----------|--------|-----|---------|
| Aircraft Equipment Problem | 0.806 | 0.849 | 0.827 | 0.951 |
| Airspace Violation | 0.380 | 0.808 | 0.517 | 0.953 |
| ATC Issue | 0.566 | 0.809 | 0.666 | 0.928 |
| Conflict | 0.761 | 0.866 | 0.810 | 0.951 |
| Deviation - Altitude | 0.638 | 0.869 | 0.736 | 0.955 |
| Deviation - Procedural | 0.831 | 0.762 | 0.795 | 0.809 |
| Deviation - Speed | 0.375 | 0.807 | 0.512 | 0.956 |
| Deviation - Track/Heading | 0.532 | 0.823 | 0.646 | 0.940 |
| Flight Deck/Cabin Event | 0.612 | 0.862 | 0.716 | 0.971 |
| Ground Event/Encounter | 0.443 | 0.813 | 0.574 | 0.941 |
| Ground Excursion | 0.428 | 0.838 | 0.566 | 0.981 |
| Ground Incursion | 0.567 | 0.906 | 0.698 | 0.979 |
| Inflight Event/Encounter | 0.699 | 0.803 | 0.747 | 0.930 |

## Per-Category Results (Classic ML 32K Baseline)

| Category | F1 | AUC | Notes |
|----------|-----|-----|-------|
| Aircraft Equipment Problem | 0.816 | 0.944 | Strong |
| Conflict | 0.801 | 0.943 | Strong |
| Deviation - Procedural | 0.795 | 0.794 | High F1, low AUC (ubiquitous category) |
| Flight Deck/Cabin Event | 0.738 | 0.963 | |
| Inflight Event/Encounter | 0.734 | 0.920 | |
| Ground Incursion | 0.729 | 0.976 | |
| Deviation - Altitude | 0.729 | 0.949 | |
| ATC Issue | 0.672 | 0.916 | |
| Deviation - Track/Heading | 0.655 | 0.928 | |
| Ground Event/Encounter | 0.592 | 0.923 | Weak — low precision |
| Deviation - Speed | 0.577 | 0.949 | Weak — rare category |
| Ground Excursion | 0.572 | 0.973 | Weak — rarest category |
| Airspace Violation | 0.568 | 0.938 | Weak — low precision |

## Compute Log

| Experiment | GPU | Duration | Cost | Date |
|-----------|-----|----------|------|------|
| Classic ML (XGBoost) | CPU (local) | ~55 min | $0 | 2025-02-12 |
| Zero-shot LLM (Ministral) | L4 (Modal) | ~18.5 min | ~$0.25 | 2026-02-13 |
| Zero-shot LLM (Qwen3) | L4 (Modal) | ~26.4 min | ~$0.35 | 2026-02-13 |
| Few-shot LLM (Ministral) | L4 (Modal) | ~30.5 min | ~$0.41 | 2026-02-13 |
| Few-shot LLM (Qwen3) | L4 (Modal) | ~34.2 min | ~$0.46 | 2026-02-13 |
| Fine-tuned LLM training (Ministral) | A100 (Modal) | ~3h48min (228 min) | ~$10.66 | 2026-02-13 |
| Fine-tuned LLM inference (Ministral) | L4 (Modal) | ~21.7 min | ~$0.29 | 2026-02-13 |
| QLoRA training (Qwen3) | A100 (Modal) | ~3h47min (227.9 min) | ~$10.56 | 2026-02-13 |
| Fine-tuned LLM inference (Qwen3) | L4 (Modal) | ~20 min | ~$0.27 | 2026-02-14 |
| Few-shot LLM (Mistral Large 3) | API (Batch) | ~4 min | $0 (free tier) | 2026-02-14 |
| Classic ML full (164K XGBoost) | 32-core CPU (Modal) | ~30 min | ~$0.64 | 2026-02-14 |
| Zero-shot taxonomy (Qwen3) | L4 (Modal) | ~24.4 min | ~$0.33 | 2026-02-14 |
| Few-shot taxonomy (Qwen3) | L4 (Modal) | ~33.6 min | ~$0.45 | 2026-02-14 |
| Zero-shot LLM (Mistral Large 3) | API (Batch) | ~5 min | $0 (free tier) | 2026-02-14 |
| Few-shot taxonomy + thinking (Qwen3) | A100 (Modal) | ~144 min | ~$6.67 | 2026-02-14 |
| Classic ML subcategory (48 XGBoost) | 32-core CPU (Modal) | ~142 min | ~$3.03 | 2026-02-15 |
| Zero-shot subcategory (Mistral Large 3) | API (Real-time) | ~119 min | paid plan | 2026-02-16 |
| Zero-shot subcategory (Qwen3-8B) | L4 (Modal) | ~30 min | ~$0.40 | 2026-02-16 |
| Zero-shot parent (DeepSeek V3.2) | API (DeepInfra) | ~6.5 min | ~$1.39 | 2026-02-16 |
| Zero-shot subcategory (DeepSeek V3.2) | API (DeepInfra) | ~7.5 min | ~$1.92 | 2026-02-16 |
| Zero-shot + thinking parent (DeepSeek V3.2) | API (DeepInfra) | ~291 min | ~$6.73 | 2026-02-17 |
| Zero-shot + thinking subcategory (DeepSeek V3.2) | API (DeepInfra) | ~545 min | ~$5.24 | 2026-02-17 |
| Classic ML tuning Phase 3 | 32-core CPU (Modal) | ~154 min | ~$3.30 | 2026-02-17 |

**Total Modal spend:** ~$38.03 (Ministral: ~$11.61 + Qwen3: ~$19.45 + Classic ML full: ~$0.64 + Classic ML subcategory: ~$3.03 + Classic ML tuning Phase 3: ~$3.30)
**Total Mistral API spend:** paid plan (real-time API for subcategory experiment)
**Total DeepInfra spend:** ~$15.28 (DeepSeek V3.2: parent ~$1.39 + subcategory ~$1.92 + thinking parent ~$6.73 + thinking subcategory ~$5.24)
