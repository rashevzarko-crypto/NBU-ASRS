# Thesis Context: NBU-ASRS Multi-Label Aviation Safety Report Classification

> **Purpose:** This file is self-contained. It provides ALL data, results, prompts, configs, and sample reports needed to write the thesis. No other files are needed.
>
> **Last generated:** 2026-02-17

---

## Section 1: STATUS.md (Verbatim Copy)

# NBU-ASRS Project Status

Last updated: 2026-02-17 (Classic ML tuning Phase 3 complete — baseline confirmed optimal)

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
| Final comparison & visualization | ❌ Not started | |
| Thesis writing | ❌ Not started | |

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


---

## Section 2: Dataset Overview

### Source
- **Database:** NASA Aviation Safety Reporting System (ASRS)
- **Format:** 61 CSV files, 282,371 raw rows
- **Unique reports:** 172,183 (after deduplication by ACN - Accession Number)
- **Reports with second narrative:** 15,720 (9.1%)
- **Text field used:** "Report Narrative" (concatenation of Narrative 1 + Narrative 2 where available)

### Anomaly Labels
- **Unique raw anomaly strings in source:** 8,272
- **Mapped to top-level categories:** 64 unique strings mapped to 13 categories
- **Unmapped strings:** 8,208 (mostly subcategories, duplicates, or noise)
- **Coverage:** 172,183 (100%) reports have at least 1 mapped category; 0 have no category
- **Task type:** Multi-label binary classification (each report can have 1-9 labels)

### Sampling
- **Method:** MultilabelStratifiedShuffleSplit (from iterstrat library)
- **Parameters:** test_size=40000 (approximate), random_state=42
- **Actual sizes:** 39,894 total (31,850 train / 8,044 test)
- **Note:** MultilabelStratifiedShuffleSplit is approximate, not exact (requested 40K, got 39,894)
- **Frozen test set:** Same 8,044 reports used for ALL experiments

### Multi-Label Statistics
- **Labels per report:** mean=2.21, median=2, min=1, max=9 (train); mean=2.20, median=2, min=1, max=7 (test)
- **Single-label reports:** ~22% of dataset
- **Multi-label reports:** ~78% of dataset
- **Imbalance ratio (max/min):** 30.3x (Deviation-Procedural: 112,606 vs Ground Excursion: 3,718)

### Label Count Distribution (Full 172K Dataset)

| Labels | Count |
|--------|-------|
| 1 label | 38,403 |
| 2 labels | 77,602 |
| 3 labels | 41,494 |
| 4 labels | 12,224 |
| 5 labels | 2,178 |
| 6 labels | 260 |
| 7 labels | 21 |
| 9 labels | 1 |

---

## Section 3: Category Distribution Table

### Full Dataset (172,183 reports)

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

### Train/Test Split Distribution

| Category | Train | Train% | Test | Test% |
|----------|-------|--------|------|-------|
| ATC Issue | 5,464 | 17.2% | 1,371 | 17.0% |
| Aircraft Equipment Problem | 9,157 | 28.8% | 2,297 | 28.6% |
| Airspace Violation | 1,270 | 4.0% | 318 | 4.0% |
| Conflict | 8,597 | 27.0% | 2,156 | 26.8% |
| Deviation - Altitude | 5,268 | 16.5% | 1,322 | 16.4% |
| Deviation - Procedural | 20,914 | 65.7% | 5,246 | 65.2% |
| Deviation - Speed | 929 | 2.9% | 233 | 2.9% |
| Deviation - Track/Heading | 3,764 | 11.8% | 944 | 11.7% |
| Flight Deck/Cabin Event | 2,282 | 7.2% | 573 | 7.1% |
| Ground Event/Encounter | 2,644 | 8.3% | 663 | 8.2% |
| Ground Excursion | 691 | 2.2% | 173 | 2.2% |
| Ground Incursion | 2,340 | 7.3% | 587 | 7.3% |
| Inflight Event/Encounter | 7,180 | 22.5% | 1,801 | 22.4% |

Category percentages are nearly identical between train and test, confirming successful stratified sampling.

---

## Section 4: Text Statistics

### Word Count Statistics

| Split | Reports | Mean | Median | Min | Max | Std |
|-------|---------|------|--------|-----|-----|-----|
| Train | 31,850 | 264.9 | 212 | 2 | 3,657 | 212.3 |
| Test | 8,044 | 266.9 | 212 | 2 | 2,477 | 214.6 |
| Combined | 39,894 | 265.3 | 212 | 2 | 3,657 | 212.7 |

### Character Count Statistics

| Split | Mean | Median | Min | Max |
|-------|------|--------|-----|-----|
| Train | 1,438.2 | 1,135 | 10 | 20,579 |
| Test | 1,449.2 | 1,137 | 13 | 14,712 |
| Combined | 1,440.4 | 1,136 | 10 | 20,579 |

### Estimated Token Count (words x 1.3)

| Split | Mean | Median |
|-------|------|--------|
| Train | 344 | 276 |
| Test | 347 | 276 |
| Combined | 345 | 276 |

- **Null narratives:** 0 in both train and test
- **Empty narratives:** 0 in both train and test

### Top 20 Most Frequent Words (lowercased, excluding English stopwords)

| Rank | Word | Count |
|------|------|-------|
| 1 | aircraft | 52,957 |
| 2 | ft | 46,091 |
| 3 | acft | 43,716 |
| 4 | us | 43,528 |
| 5 | rwy | 42,498 |
| 6 | time | 30,200 |
| 7 | runway | 27,068 |
| 8 | flight | 24,787 |
| 9 | apch | 23,307 |
| 10 | flt | 23,054 |
| 11 | atc | 23,050 |
| 12 | turn | 22,655 |
| 13 | approach | 21,348 |
| 14 | told | 21,265 |
| 15 | asked | 18,584 |
| 16 | said | 18,443 |
| 17 | fl | 18,162 |
| 18 | zzz | 17,296 |
| 19 | called | 16,324 |
| 20 | twr | 15,573 |

Note: "zzz" is the ASRS anonymization placeholder for airport/location names. "acft" = aircraft, "rwy" = runway, "apch" = approach, "flt" = flight, "twr" = tower, "fl" = flight level. Aviation abbreviations are heavily represented.

### Top 10 Most Common Label Combinations

| Rank | Count | % | Categories |
|------|-------|---|------------|
| 1 | 2,495 | 6.3% | Aircraft Equipment Problem, Deviation - Procedural |
| 2 | 2,245 | 5.6% | Deviation - Altitude, Deviation - Procedural |
| 3 | 2,233 | 5.6% | Aircraft Equipment Problem |
| 4 | 2,096 | 5.3% | Conflict |
| 5 | 1,656 | 4.2% | Conflict, Deviation - Procedural |
| 6 | 1,517 | 3.8% | Deviation - Procedural, Deviation - Track/Heading |
| 7 | 1,488 | 3.7% | Inflight Event/Encounter |
| 8 | 1,480 | 3.7% | ATC Issue, Conflict, Deviation - Procedural |
| 9 | 1,332 | 3.3% | Deviation - Procedural, Ground Incursion |
| 10 | 1,220 | 3.1% | Deviation - Procedural |

Note: Deviation - Procedural appears in 8 of the top 10 combinations, reflecting its 65.4% prevalence. The most common single-label reports are Aircraft Equipment Problem and Conflict.

---

## Section 5: Sample Reports from Test Set

### Sample 1: Single-label (Aircraft Equipment Problem) -- 183 words
**ACN:** 1333851
**Labels:** Aircraft Equipment Problem
**Criteria:** Single-label, medium length

Part 91 flight. Aircraft just released for flight after a heavy c-check. During first leg of the ferry; during ILS approach to landing; right main landing gear did not extend during normal extension. Using QRH procedures for manual extension; all main gear extended to down and locked.As a side note; this airport requires the landing gear to be extended while out over the water well in advance of when the crew would normally extend the gear. As a flight test crew; we were very familiar with the procedure and time required to manually extend the gear. Using the QRH manual extension system; we had 3 down and locked and all checklist completed well in advance of the required stable call. Uneventful landing. After landing; exited onto the high-speed taxiway which ended directly at our parking gate. Maintenance troubleshot and conducted the required repairs. We continued the flight on the next day as scheduled. No [priority handling wasn't requested]; but notification was made to Tower of possible taxiing problems. Due to parking spot; taxiing was not an issue. [Report narrative contained no additional information.]

### Sample 2: Single-label (Conflict) -- 177 words
**ACN:** 581639
**Labels:** Conflict
**Criteria:** Single-label, medium length

RETURNING FROM THE FRENCH VALLEY ARPT TO TOA USING FLT FOLLOWING FROM SOCAL APCH. JUST AS I WAS RELEASED TO BEGIN DSCNT BELOW 3000 FT (A PREVIOUSLY ASSIGNED RESTR REQUESTED BY SOCAL); SOCAL PROVIDED A TA; A CHEROKEE 3 MI AT 9-10 O'CLOCK; 1900 FT DSNDING. A LITTLE LATER; A SECOND ADVISORY CAME. THIS TIME; SOCAL RPTED THAT THE PLANE HAD ME IN SIGHT. I REPLIED TO BOTH RPTS WITH 'LOOKING; NOT IN SIGHT.' THEN SOCAL SAID 'OK; LET'S DECLARE A LITTLE EMER HERE; GIVE ME MAX CLB (OR SOMETHING CLOSE) FOR SPACING.' I COMPLIED; AND ALMOST IMMEDIATELY SAW THE TFC WELL CLR TO MY L AND LOWER; WHICH I RPTED. SOCAL THEN SAID PROCEED PLT'S DISCRETION; SQUAWK VFR; CHANGE TO THE TWR. THIS IS BEING RPTED; BECAUSE THE CTLR 'CALLED A LITTLE EMER.' HE OBVIOUSLY THOUGHT IT PROPER; SO THERE MIGHT HAVE BEEN THE POSSIBILITY FOR AN UNDESIRED CONSEQUENCE. I DON'T BELIEVE EITHER PLANE WAS IN ANY ACTUAL DANGER; BUT WAS HAPPY TO HAVE THE ASSISTANCE. THE SYS WORKED; AND WE ARE ALL GRATEFUL FOR THAT.

### Sample 3: Two-label (Ground Event/Encounter + Ground Incursion) -- 150 words
**ACN:** 1117236
**Labels:** Ground Event/Encounter, Ground Incursion
**Criteria:** Two-label

Truck towing a piece of farm implement drove onto the taxiway and past the runway hold bars without stopping and or looking and at the edge of the runway turned into the field to unload the piece of equipment. Vehicle had no lights; flags; or VHF radio with numerous aircraft in the traffic pattern. No effort to remove F.O.D. from taxiway/runway from the dirt on vehicle tires. The piece of farm machinery; a plow; is now next to Runway 16/34; unmarked; not NOTAMed; unlighted. Yolo County Airport is unfenced; non-gated; [and] poor to no markings identifying either taxiways or active runways. In its 68 year existence it has NEVER had a fence or gate to prevent people from walking; driving; or bicycling on the taxiways and runway. Yolo County refuses to address fencing issues. In the reporters opinion it is only a matter of time before an event takes place.

### Sample 4: Two-label (Deviation - Procedural + Ground Event/Encounter) -- 278 words
**ACN:** 1934989
**Labels:** Deviation - Procedural, Ground Event/Encounter
**Criteria:** Two-label

I was working as left wingwalker for a flight in [gate] XXX. Once we got to push out spot; I wait to the signal of the push driver to remove the locking pin of the towbar. I remove it then I pull the ladder but the ladder but latch did not drop. I tried to shake the towbar; that didn't work. I decided to remove the pin of the pushtruck; that didn't work either. Then I decided to use the hydraulic but that didn't work too. The push driver was moving reverse; the latch finally dropped but unfortunately the plane was still moving; passing under the towbar breaking the handle. The towbar got stuck under the nose gear. Because of the towbar connected to the plane did not drop after been removed the lock pin and pulling the ladder. Maintenance in the towbar; pushback tractor grease where towbar connected needs to be release. I was pushing back the aircraft when the wing walker was trying to disconnect the tow bar; so the tow bar got stuck and we couldn't get it loose. I notified the Captain; that he was going to call Maintenance. In the process the tow bar got loose and I notified the Captain to set his brakes several occasions. It was too late and the tow bar got under the nose gear of the aircraft. The tow bar was stuck under the nose gear of the aircraft. 1st; tow bar should be greased on a regular basis adding the job into their pm checks. 2nd; Captain of the aircraft should stop the aircraft right away when I tell him to stop. Aircraft Maintenance show up.

### Sample 5: Three-label (Aircraft Equipment Problem + Deviation - Procedural + Deviation - Track/Heading) -- 174 words
**ACN:** 693160
**Labels:** Aircraft Equipment Problem, Deviation - Procedural, Deviation - Track/Heading
**Criteria:** 3-label

DURING PREFLT WE PLANNED AN E SIDE DEP. ON TAXI OUT WE WERE ASSIGNED A W SIDE RWY AT DFW. I RELOADED THE RWY AND DEP. DOING SO I MUST HAVE MISS LOADED THE TRANSITION. I CLOSED THE DISCONTINUITY IN THE FLT PLAN AND DIDNT NOTICE THAT IT WENT FROM GRABE INTXN TO EOS. THE PLAN SHOULD HAVE HAD THE OKM TRANS. WE GOT TO CRUISE ALT AND WERE LOOKING AT THE NEXT WAYPOINT AFTER GRABE AND IT DIDN'T COINCIDE WITH OUR PLAN. APPROX THE SAME TIME CTR CALLED AND CLRED US DIRECT TO OKM. I REALIZED HOW I HAD CAUSED THE NAV MISS LOAD. CTR SAID HE SAW US DRIFT SLIGHTLY R AND THAT THERE WAS NO PROBLEM. REMAINDER OF THE FLT WAS UNEVENTFUL. CAUSE OF PROBLEM WAS MISS LOAD OF FMS; NOT RECHKING LOAD; PASSING WAYPOINT BEFORE CHKING NEXT WAYPOINT. FO WAS ALSO BUSY DURING TAXI OUT AND WE SHOULD HAVE SLOWED DOWN AND TAKEN TIME TO RE-VERIFY THE DEP PAST THE RWY AND FIRST TURNING FIXES. WON'T LET THIS HAPPEN AGAIN.

### Sample 6: Three-label (ATC Issue + Conflict + Deviation - Procedural) -- 327 words
**ACN:** 1652671
**Labels:** ATC Issue, Conflict, Deviation - Procedural
**Criteria:** 3-label, long

Denver TRACON has opted to implement new procedures to fix overshoots on the 16 Runway finals. At any given time one of the 16 finals is called the 'high' runway and is told to conduct ILS approaches while the other is the 'low' runway and is told to conduct visual approaches. However here at Denver we also have RNAV ZULU Approaches which is an instrument approach that arcs to the runway from the downwind. It was decided that I was going to go from being the 'low' runway to being the 'high' runway. Unfortunately this was never coordinated with me I had to figure it out myself based on how I was being fed. As I began to transition the volume picked up and I was focused on meeting the requirements put forth by management that super-exceed the 7110.65 for separation. As I began to run the final and get all of the aircraft onto the ILS I noticed an RNAV ZULU approach within 3 miles of the aircraft I had cleared for the ILS. I issued the visual approach clearance to the aircraft but at that point it was already a loss of separation. There are 3 clear options to resolve this from happening again or to another controller. The first is to remove the procedures and return to working traffic as we had previously. The FAA has taken the responsibility of the pilots and the airlines and taken it upon themselves to fix a problem with procedures that are ineffective and damaging to the NAS. The second is to explain to the Tower and the airlines that for safety we will now be landing on Runway 17L and 17R which has more space between finals. Finally the last is to once again refine the procedures; create new STARS that will allow for the smooth operation of these procedures; and properly train the workforce rather than rushing out the procedure so local management can collect data.

### Sample 7: Short report (58 words)
**ACN:** 432290
**Labels:** Deviation - Procedural, Deviation - Track/Heading
**Criteria:** Short (<100 words)

DURING A XFER OF ACFT CTL ON AUTOPLT; ACFT WAS IN AN UNNOTICED 10 DEG L BANK TURN. TURN WAS NOT NOTICED UNTIL APPROX 60 DEGS OFF DESIRED HDG (NAVING ON AIRWAYS; NOT ON VECTORS). IMMEDIATE TURN TO DESIRED COURSE WAS INITIATED. ESTIMATED DEV FROM COURSE CTRLINE WAS 10 NM. NO CONFLICT EXISTED. COCKPIT DISTR WAS PROBABLE CAUSE.

### Sample 8: Short report (79 words)
**ACN:** 2004094
**Labels:** Conflict, Deviation - Procedural
**Criteria:** Short (<100 words)

Landing on Runway XX at ZZZ; Aircraft Y took off on the crossing Runway YY. Looking at the recording we made a 10 mile; 5 mile; and short final call. While the Aircraft Y did make a take-off call; they took off while we were short final and we missed the aircraft by 500 ft. horizontally. Used the term did not see you on the scope and based upon the conversation they were not listening to the CTAF frequency.

### Sample 9: Long report (405 words)
**ACN:** 342063
**Labels:** Deviation - Procedural, Ground Incursion
**Criteria:** Long (>300 words)

I FLEW A SOLO XCOUNTRY FROM DVT IN PHOENIX TO PALM SPRINGS REGIONAL IN CALIFORNIA VFR. THERE WERE NO NOTAMS FOR THE RTE ACCORDING TO FSS. ON MY RETURN FROM PALM SPRINGS I RECEIVED CLRNC FROM PALM SPRINGS CLRNC DELIVERY AND TAXIED TO THE RUN-UP AREA BEFORE RWY 31L. WITH MY RUN-UP COMPLETE I TAXIED TO THE HOLD SHORT LINE AND TRIED UNSUCCESSFULLY TO CONTACT THE TWR FOR A CLRNC TO DEPART. FAILING TO ESTABLISH ANY RECEPTION OR XMISSION OF COM I TRIED GND CTL WITH NO CONTACT EITHER. I THEN NOTICED SIGNS SAYING THAT THE AREA WAS IN A RADIO XMISSION BLIND SPOT AND IF UNABLE TO MAKE RADIO CONTACT TO TAXI TOWARDS THE RWY. THIS OF COURSE I ALREADY HAD DONE; WITH NO RESULT. BY THIS TIME AN ACFT WAS RIGHT BEHIND ME ON THE TXWY. AFTER 2 MINS OF TRYING TO CONTACT ATC WHILST WAITING FOR LIGHT SIGNALS I DECIDED I WAS SUPPOSED TO TKOF; THINKING ATC HAD ANTICIPATED SUCH A SCENARIO. THIS INCIDENT RESULTED IN ME BREAKING A WELL KNOWN FAR DUE TO POOR ATC TO GND COM FACILITIES; LACK OF ANY FOREWARNING OF SUCH AN INCIDENT THROUGH ATIS; NOTAMS; ETC; AND SIGNS POSITIONED IN SUCH A MANNER TO MAKE THEIR COMPREHENSION ALMOST IMPOSSIBLE WITH PERFECT VISION. MY REF IS TO THE LOWER; LESS CLR; SMALLER WORDING BELOW THE AMBIGUOUS LARGER WORDING. I HAVE SINCE CONTACTED PALM SPRINGS TWR AND EXPLAINED AND APOLOGIZED AND FOUND OUT I WAS ONE OF THREE PLTS ON THAT DAY ALONE TO DO THIS. PLEASE TAKE ACTION TO REMEDY FUTURE SITS. CALLBACK CONVERSATION WITH RPTR REVEALED THE FOLLOWING INFO: RPTR STATES THAT HE HAS SPOKEN WITH LCL FSDO REPRESENTATIVE WHO INDICATED HE WILL GO TO PSP AND CHK OUT THE SIGNAGE. RPTR SAYS THERE IS A VERY WIDE RUN-UP AREA AT RWY 31L AND THEN 2 LARGE SIGNS WHEN ONE TURNS TO TAXI TO HOLD SHORT LINE. THE TOP LINE OF THE SIGN HAS LARGE LETTERING TO INDICATE THE LACK OF RADIO XMISSION IN THAT AREA. THE BOTTOM LINES ARE IN VERY SMALL PRINT AND DIFFICULT TO READ; EVEN THOUGH RPTR HAS PERFECT EYESIGHT. HIS SUGGESTION IS TO HAVE SAME SIZE LETTERING FOR BOTH PARTS OF THE SIGN. SINCE HE WAS 1 O

### Sample 10: Long report (332 words)
**ACN:** 1883021
**Labels:** Deviation - Procedural, Ground Event/Encounter
**Criteria:** Long (>300 words)

The airport rotating beacon at Louisa airport (LKU) is very weak and difficult to see in clear weather. The condition has been reported. [Airport personnel] had the lighting system inspected and said the rotating beacon is within specifications.While on several VFR flights into Louisa over the past year or so; the airport rotating beacon is extremely hard to pick up visually. I know where it is located on the airport; and on a most recent approach at dusk it was difficult for myself and another pilot flying with me to see the beacon when we were within 5 miles of the airport setting up for entry on the downwind leg for arrival into Louisa. The beacon lighting appears dim; or the beam an incorrect angle.On this particular flight about a week ago; I was 30 miles South of Louisa setting up for our arrival at near dusk time. A small airport to the east (Bumpass/Lake Anna Airport 7W4) beacon was easily seen at that distance; but unable to see anything around the Louisa airport except for the neighboring cell phone tower lights. When we were about 10 miles south of the airport we turned on the airport runway lighting system to confirm airport location; with no beacon in sight. The Bumpass 7W4 beacon easily seen.On another flight coming into Louisa from the South East it was the same thing. Almost like flying toward a dark hole; the Bumpass/Lake Anna Airport 7W4 beacon was shining brightly; but nothing distinguishable from the Louisa airport. We turned on the airport runway lighting system to confirm airport location for the night landing. I am reporting this as an airport safety issue for all VFR pilots flying at nighttime or dusk; and possibly it should be NOTAM'd since it is so difficult to be visually acquired; and can only be seen for short distances. The beacon is also located on the water tower; which is not very high; and surrounded with tree growth. Not a good location.

### Sample 11: Rare category (Ground Excursion) -- 234 words
**ACN:** 580113
**Labels:** Deviation - Procedural, Ground Event/Encounter, Ground Excursion
**Criteria:** Rare category (Ground Excursion, 2.2% prevalence)

PREFLT AND START-UP WERE COMPLETED AS NORMAL. AS TAXI WAS STARTED; I TESTED THE L AND R RUDDER PEDALS; THEN THE BRAKES. THE STUDENT THEN DID THE SAME. ONCE REACHING THE INTXN OF TXWY I AND TXWY G; THE STUDENT RECEIVED A TAXI CLRNC TO RWY 28R VIA TXWY H AND TXWY A. WE PROCEEDED TO TAXI AND WHEN THE STUDENT MADE THE R-HAND TURN FROM TXWY G AND TXWY H; I REALIZED OUR TAXI SPD WAS HIGH. I TOLD THE STUDENT TO REDUCE PWR AND SLOW DOWN. THE STUDENT DID AS INSTRUCTED AND WE CONTINUED E ON TXWY H. UPON REACHING THE INTXN OF TXWY H AND TXWY A; THE STUDENT APPLIED L RUDDER AND THE ACFT CONTINUED STRAIGHT AHEAD. I REACTED BY PULLING THE REMAINING PWR AND APPLYING HVY BRAKING. THE ACFT WAS UNABLE TO BE STOPPED ON THE TXWY AND DID NOT COME TO A COMPLETE STOP UNTIL IN THE DIRT AT THE E END OF THE TXWY. I FEEL THAT THE INCIDENT COULD HAVE BEEN AVOIDED HAD I; THE INSTRUCTOR; HAD THE STUDENT REDUCE PWR TO IDLE BEFORE THE TURN; AND ALSO SLOWED THE TAXI SPD. THE ACFT MAY NOT HAVE REACTED TO THE CTL INPUT AS QUICKLY AS EXPECTED; BUT AFTER TESTING THE CTLS ONCE BACK ON THE TXWY; I WAS UNABLE TO FIND ANY FAULT DUE TO ACFT SYS MALFUNCTION. THIS INCIDENT OCCURRED AT MYF; SAN DIEGO; CA.

### Sample 12: Rare category (Airspace Violation) -- 230 words
**ACN:** 695364
**Labels:** Airspace Violation, Deviation - Procedural
**Criteria:** Rare category (Airspace Violation, 4.0% prevalence)

A YOUNG MAN RAN AWAY FROM A PVT FACILITY IN THE DESERT. I HAD AN AIRPLANE AND ONE OF THE DIRECTORS NEEDED A WAY TO SEARCH FOR THE YOUNG MAN. I CALLED FSS ON MY CELL AND GOT MY HOME FSS AT DAYTON. THERE WERE NO RESTRS TO FLT NOTED. ON MY WAY TO THE PLACE OF SEARCH; I FLEW NEAR A BUNCH OF BUILDINGS. I THEN REMEMBERED A TFR BEING IN THAT AREA IN THE LAST SEVERAL YRS. I DO NOT RECALL MY EXACT ALT BECAUSE I WAS CLBING TO GO OVER A RIDGE SOME DISTANCE AWAY. I FLEW S IMMEDIATELY AS SOON AS MY MIND REALIZED THAT I MIGHT BE NEAR A TFR. THEN WE WENT OVER THE RIDGE AND COMPLETED THE SEARCH (THE YOUNG MAN WAS FOUND). I CALLED CEDAR CITY FSS AND THEY TOLD ME IT WAS NOW A PERMANENT FLT RESTR; 8000 FT AND BELOW; SIMPLY MARKED 'ORDNANCE DEPOT' ON THE CHARTS. SURE ENOUGH THERE IT WAS ON MY CHART; BUT NO INDICATION OF A RESTR. I RETURNED ABOVE 8000 FT. I THINK THERE ARE 3 PROBS. 1) I DO NOT CHK PRINTED NOTAMS OFTEN; AND I SUPPOSE IT WAS THERE. 2) WE NEED TO FIGURE OUT HOW TO CONTACT THE LCL FSS WITH A CELL PHONE; OR MAKE THE LCL NOTAMS AVAILABLE EVERYWHERE. 3) FINALLY; PERMANENT FLT RESTRS SHOULD BE MARKED ON THE CHARTS.

---

## Section 6: All Experimental Results (Per-Category Tables)

### Experiment 1: Classic ML 32K (TF-IDF + XGBoost, 31,850 train)

| Category | Precision | Recall | F1 | ROC-AUC |
|----------|-----------|--------|------|---------|
| Aircraft Equipment Problem | 0.8134 | 0.8180 | 0.8157 | 0.9438 |
| Airspace Violation | 0.4804 | 0.6950 | 0.5681 | 0.9377 |
| ATC Issue | 0.6053 | 0.7549 | 0.6719 | 0.9156 |
| Conflict | 0.7691 | 0.8358 | 0.8011 | 0.9433 |
| Deviation - Altitude | 0.6636 | 0.8086 | 0.7289 | 0.9494 |
| Deviation - Procedural | 0.8117 | 0.7791 | 0.7951 | 0.7940 |
| Deviation - Speed | 0.5508 | 0.6052 | 0.5767 | 0.9490 |
| Deviation - Track/Heading | 0.5920 | 0.7331 | 0.6550 | 0.9275 |
| Flight Deck/Cabin Event | 0.7013 | 0.7784 | 0.7378 | 0.9633 |
| Ground Event/Encounter | 0.5032 | 0.7195 | 0.5922 | 0.9232 |
| Ground Excursion | 0.5723 | 0.5723 | 0.5723 | 0.9734 |
| Ground Incursion | 0.6457 | 0.8382 | 0.7294 | 0.9758 |
| Inflight Event/Encounter | 0.7045 | 0.7651 | 0.7336 | 0.9202 |
| **MACRO** | **0.6472** | **0.7464** | **0.6906** | **0.9320** |
| **MICRO** | **0.7134** | **0.7814** | **0.7459** | **0.9500** |

Config: TF-IDF max_features=50000, ngram_range=(1,2), sublinear_tf=True. XGBoost n_estimators=300, max_depth=6, lr=0.1, scale_pos_weight=auto, tree_method=hist. 13 independent binary classifiers.

### Experiment 2: Classic ML 164K (TF-IDF + XGBoost, 164,139 train)

| Category | Precision | Recall | F1 | ROC-AUC |
|----------|-----------|--------|------|---------|
| Aircraft Equipment Problem | 0.8062 | 0.8494 | 0.8272 | 0.9510 |
| Airspace Violation | 0.3802 | 0.8082 | 0.5171 | 0.9533 |
| ATC Issue | 0.5664 | 0.8089 | 0.6663 | 0.9284 |
| Conflict | 0.7606 | 0.8664 | 0.8101 | 0.9507 |
| Deviation - Altitude | 0.6376 | 0.8691 | 0.7356 | 0.9554 |
| Deviation - Procedural | 0.8307 | 0.7623 | 0.7950 | 0.8092 |
| Deviation - Speed | 0.3752 | 0.8069 | 0.5123 | 0.9561 |
| Deviation - Track/Heading | 0.5318 | 0.8231 | 0.6462 | 0.9403 |
| Flight Deck/Cabin Event | 0.6121 | 0.8621 | 0.7159 | 0.9710 |
| Ground Event/Encounter | 0.4433 | 0.8130 | 0.5737 | 0.9414 |
| Ground Excursion | 0.4277 | 0.8382 | 0.5664 | 0.9814 |
| Ground Incursion | 0.5672 | 0.9063 | 0.6977 | 0.9787 |
| Inflight Event/Encounter | 0.6986 | 0.8029 | 0.7471 | 0.9305 |
| **MACRO** | **0.5875** | **0.8321** | **0.6777** | **0.9421** |
| **MICRO** | **0.6736** | **0.8173** | **0.7385** | **0.9421** |

Config: Same as 32K but trained on 164,139 reports (full dataset minus test set). Higher recall across all categories, but lower precision due to class imbalance amplification.

### Experiment 3: Classic ML Tuned (TF-IDF + XGBoost, 31,850 train, Phase 3)

| Category | Precision | Recall | F1 | ROC-AUC |
|----------|-----------|--------|------|---------|
| Aircraft Equipment Problem | 0.8131 | 0.8237 | 0.8183 | 0.9440 |
| Airspace Violation | 0.5000 | 0.7075 | 0.5859 | 0.9361 |
| ATC Issue | 0.6077 | 0.7491 | 0.6710 | 0.9158 |
| Conflict | 0.7683 | 0.8353 | 0.8004 | 0.9436 |
| Deviation - Altitude | 0.6650 | 0.8064 | 0.7289 | 0.9478 |
| Deviation - Procedural | 0.8081 | 0.7787 | 0.7931 | 0.7919 |
| Deviation - Speed | 0.5391 | 0.5923 | 0.5644 | 0.9491 |
| Deviation - Track/Heading | 0.5810 | 0.7331 | 0.6482 | 0.9286 |
| Flight Deck/Cabin Event | 0.7009 | 0.7853 | 0.7407 | 0.9643 |
| Ground Event/Encounter | 0.5010 | 0.7255 | 0.5927 | 0.9259 |
| Ground Excursion | 0.5892 | 0.6301 | 0.6089 | 0.9723 |
| Ground Incursion | 0.6385 | 0.8245 | 0.7197 | 0.9760 |
| Inflight Event/Encounter | 0.7049 | 0.7651 | 0.7338 | 0.9216 |
| **MACRO** | **0.6474** | **0.7505** | **0.6928** | **0.9321** |
| **MICRO** | **0.7122** | **0.7819** | **0.7454** | **0.9499** |

Config: Same TF-IDF and XGBoost params as 32K baseline. Retrained via modal_classic_ml_phase3.py with per-label scale_pos_weight. Delta vs baseline: Macro-F1 +0.002, Micro-F1 -0.001. Confirms baseline is near-optimal.

### Experiment 4: Qwen3-8B Zero-Shot (basic prompt)

| Category | Precision | Recall | F1 | ROC-AUC |
|----------|-----------|--------|------|---------|
| Aircraft Equipment Problem | 0.8141 | 0.6426 | 0.7182 | 0.7920 |
| Airspace Violation | 0.4575 | 0.2201 | 0.2972 | 0.6047 |
| ATC Issue | 0.3275 | 0.7513 | 0.4562 | 0.7172 |
| Conflict | 0.6159 | 0.7815 | 0.6889 | 0.8015 |
| Deviation - Altitude | 0.5632 | 0.8328 | 0.6720 | 0.8529 |
| Deviation - Procedural | 0.7268 | 0.2327 | 0.3526 | 0.5344 |
| Deviation - Speed | 0.4639 | 0.6609 | 0.5451 | 0.8191 |
| Deviation - Track/Heading | 0.2762 | 0.8040 | 0.4112 | 0.7619 |
| Flight Deck/Cabin Event | 0.1347 | 0.8028 | 0.2307 | 0.7036 |
| Ground Event/Encounter | 0.2064 | 0.6410 | 0.3123 | 0.7098 |
| Ground Excursion | 0.2864 | 0.6821 | 0.4034 | 0.8224 |
| Ground Incursion | 0.4415 | 0.3918 | 0.4152 | 0.6764 |
| Inflight Event/Encounter | 0.3559 | 0.6657 | 0.4638 | 0.6591 |
| **MACRO** | **0.4361** | **0.6238** | **0.4590** | **0.7273** |
| **MICRO** | **0.4079** | **0.5614** | **0.4725** | **0.6978** |

Config: Qwen3-8B, vLLM, max_model_len=8192, temperature=0.0, max_tokens=256, enable_thinking=False. Basic system prompt.

### Experiment 5: Qwen3-8B Few-Shot (basic prompt, 3 examples/category)

| Category | Precision | Recall | F1 | ROC-AUC |
|----------|-----------|--------|------|---------|
| Aircraft Equipment Problem | 0.9657 | 0.3674 | 0.5323 | 0.6811 |
| Airspace Violation | 0.5059 | 0.4057 | 0.4503 | 0.6947 |
| ATC Issue | 0.3495 | 0.6718 | 0.4598 | 0.7075 |
| Conflict | 0.8632 | 0.6322 | 0.7299 | 0.7978 |
| Deviation - Altitude | 0.6098 | 0.7080 | 0.6552 | 0.8095 |
| Deviation - Procedural | 0.8122 | 0.1138 | 0.1996 | 0.5322 |
| Deviation - Speed | 0.4549 | 0.5408 | 0.4941 | 0.7607 |
| Deviation - Track/Heading | 0.3421 | 0.7733 | 0.4743 | 0.7878 |
| Flight Deck/Cabin Event | 0.4079 | 0.6143 | 0.4903 | 0.7730 |
| Ground Event/Encounter | 0.3448 | 0.0603 | 0.1027 | 0.5250 |
| Ground Excursion | 0.2305 | 0.4104 | 0.2952 | 0.6901 |
| Ground Incursion | 0.4971 | 0.4327 | 0.4627 | 0.6991 |
| Inflight Event/Encounter | 0.5883 | 0.4919 | 0.5358 | 0.6963 |
| **MACRO** | **0.5363** | **0.4787** | **0.4525** | **0.7042** |
| **MICRO** | **0.5439** | **0.4099** | **0.4675** | **0.6700** |

Config: Same model, max_model_len=16384, batch_size=16. 39 examples (3 per category), narrative truncated to 600 chars (examples) / 1500 chars (test).

### Experiment 6: Qwen3-8B Fine-Tuned (QLoRA 4-bit NF4)

| Category | Precision | Recall | F1 | ROC-AUC |
|----------|-----------|--------|------|---------|
| Aircraft Equipment Problem | 0.7990 | 0.7667 | 0.7825 | 0.8448 |
| Airspace Violation | 0.1879 | 0.0881 | 0.1199 | 0.5362 |
| ATC Issue | 0.5000 | 0.3115 | 0.3838 | 0.6237 |
| Conflict | 0.8390 | 0.6359 | 0.7235 | 0.7956 |
| Deviation - Altitude | 0.6077 | 0.7958 | 0.6892 | 0.8474 |
| Deviation - Procedural | 0.6908 | 0.8170 | 0.7486 | 0.5658 |
| Deviation - Speed | 0.4610 | 0.5322 | 0.4940 | 0.7568 |
| Deviation - Track/Heading | 0.4017 | 0.6186 | 0.4871 | 0.7481 |
| Flight Deck/Cabin Event | 0.3946 | 0.3298 | 0.3593 | 0.6455 |
| Ground Event/Encounter | 0.5420 | 0.4962 | 0.5181 | 0.7293 |
| Ground Excursion | 0.5361 | 0.3006 | 0.3852 | 0.6474 |
| Ground Incursion | 0.5201 | 0.4634 | 0.4901 | 0.7149 |
| Inflight Event/Encounter | 0.4397 | 0.4531 | 0.4463 | 0.6432 |
| **MACRO** | **0.5323** | **0.5084** | **0.5098** | **0.6999** |
| **MICRO** | **0.6252** | **0.6385** | **0.6318** | **0.7803** |

Config: QLoRA 4-bit NF4, r=16, alpha=16, target=[q_proj, v_proj], dropout=0.05. Training: 2 epochs, batch=4, grad_accum=4, lr=2e-5, cosine scheduler, paged_adamw_8bit, bf16. 3,982 steps, final loss 1.691, token accuracy 66.8%. Training time: 3h47m on A100.

### Experiment 7: Qwen3-8B Zero-Shot Taxonomy

| Category | Precision | Recall | F1 | ROC-AUC |
|----------|-----------|--------|------|---------|
| Aircraft Equipment Problem | 0.8762 | 0.4558 | 0.5997 | 0.7150 |
| Airspace Violation | 0.6698 | 0.2233 | 0.3349 | 0.6094 |
| ATC Issue | 0.3295 | 0.5419 | 0.4098 | 0.6577 |
| Conflict | 0.6768 | 0.7032 | 0.6897 | 0.7901 |
| Deviation - Altitude | 0.7261 | 0.4433 | 0.5505 | 0.7052 |
| Deviation - Procedural | 0.6870 | 0.8757 | 0.7700 | 0.5638 |
| Deviation - Speed | 0.6063 | 0.3305 | 0.4278 | 0.6620 |
| Deviation - Track/Heading | 0.6966 | 0.1727 | 0.2767 | 0.5813 |
| Flight Deck/Cabin Event | 0.4601 | 0.6545 | 0.5403 | 0.7978 |
| Ground Event/Encounter | 0.2948 | 0.6471 | 0.4051 | 0.7540 |
| Ground Excursion | 0.2944 | 0.6416 | 0.4036 | 0.8039 |
| Ground Incursion | 0.5048 | 0.5349 | 0.5194 | 0.7468 |
| Inflight Event/Encounter | 0.4936 | 0.6441 | 0.5589 | 0.7267 |
| **MACRO** | **0.5628** | **0.5283** | **0.4990** | **0.7011** |
| **MICRO** | **0.5805** | **0.6325** | **0.6054** | **0.7698** |

Config: Same as basic zero-shot but with taxonomy-enriched system prompt (NASA ASRS subcategories + discriminative hints).

### Experiment 8: Qwen3-8B Few-Shot Taxonomy

| Category | Precision | Recall | F1 | ROC-AUC |
|----------|-----------|--------|------|---------|
| Aircraft Equipment Problem | 0.9873 | 0.2699 | 0.4239 | 0.6343 |
| Airspace Violation | 0.6170 | 0.3648 | 0.4585 | 0.6777 |
| ATC Issue | 0.3760 | 0.6725 | 0.4823 | 0.7216 |
| Conflict | 0.8251 | 0.6498 | 0.7270 | 0.7997 |
| Deviation - Altitude | 0.7368 | 0.5825 | 0.6506 | 0.7708 |
| Deviation - Procedural | 0.7653 | 0.3791 | 0.5071 | 0.5806 |
| Deviation - Speed | 0.6327 | 0.3991 | 0.4895 | 0.6961 |
| Deviation - Track/Heading | 0.5476 | 0.4513 | 0.4948 | 0.7008 |
| Flight Deck/Cabin Event | 0.7489 | 0.6143 | 0.6750 | 0.7993 |
| Ground Event/Encounter | 0.4728 | 0.3937 | 0.4296 | 0.6771 |
| Ground Excursion | 0.4015 | 0.3064 | 0.3475 | 0.6482 |
| Ground Incursion | 0.6637 | 0.5145 | 0.5797 | 0.7470 |
| Inflight Event/Encounter | 0.5634 | 0.5697 | 0.5665 | 0.7212 |
| **MACRO** | **0.6414** | **0.4744** | **0.5255** | **0.7057** |
| **MICRO** | **0.6426** | **0.4711** | **0.5436** | **0.7089** |

Config: Same as basic few-shot but with taxonomy-enriched system prompt. 39 examples (3 per category).

### Experiment 9: Qwen3-8B Few-Shot Taxonomy + Thinking Mode

| Category | Precision | Recall | F1 | ROC-AUC |
|----------|-----------|--------|------|---------|
| Aircraft Equipment Problem | 0.9240 | 0.5294 | 0.6731 | 0.7560 |
| Airspace Violation | 0.5776 | 0.4214 | 0.4873 | 0.7043 |
| ATC Issue | 0.4024 | 0.4150 | 0.4086 | 0.6442 |
| Conflict | 0.8777 | 0.6456 | 0.7440 | 0.8063 |
| Deviation - Altitude | 0.7713 | 0.5204 | 0.6215 | 0.7450 |
| Deviation - Procedural | 0.8575 | 0.3601 | 0.5072 | 0.6239 |
| Deviation - Speed | 0.7300 | 0.3133 | 0.4384 | 0.6549 |
| Deviation - Track/Heading | 0.7029 | 0.2055 | 0.3180 | 0.5970 |
| Flight Deck/Cabin Event | 0.8083 | 0.5812 | 0.6761 | 0.7853 |
| Ground Event/Encounter | 0.5221 | 0.4284 | 0.4706 | 0.6966 |
| Ground Excursion | 0.5363 | 0.5549 | 0.5455 | 0.7722 |
| Ground Incursion | 0.7355 | 0.3884 | 0.5084 | 0.6887 |
| Inflight Event/Encounter | 0.6922 | 0.4220 | 0.5243 | 0.6839 |
| **MACRO** | **0.7029** | **0.4450** | **0.5325** | **0.7045** |
| **MICRO** | **0.7438** | **0.4442** | **0.5563** | **0.7065** |

Config: enable_thinking=True, max_tokens=4096, max_model_len=32768, A100 GPU. 99.6% outputs had <think> blocks, avg 2,986 chars thinking per response. Runtime: 144 min on A100 (~$6.67). Marginal gain over non-thinking: Macro-F1 +0.007, Micro-F1 +0.013.

### Experiment 10: Mistral Large 3 Zero-Shot (taxonomy prompt, Batch API)

| Category | Precision | Recall | F1 | ROC-AUC |
|----------|-----------|--------|------|---------|
| Aircraft Equipment Problem | 0.8191 | 0.8119 | 0.8155 | 0.8701 |
| Airspace Violation | 0.6157 | 0.4434 | 0.5155 | 0.7160 |
| ATC Issue | 0.4100 | 0.7542 | 0.5312 | 0.7656 |
| Conflict | 0.8623 | 0.6391 | 0.7342 | 0.8009 |
| Deviation - Altitude | 0.7496 | 0.7678 | 0.7586 | 0.8587 |
| Deviation - Procedural | 0.7085 | 0.8997 | 0.7927 | 0.6028 |
| Deviation - Speed | 0.6490 | 0.5794 | 0.6122 | 0.7850 |
| Deviation - Track/Heading | 0.7066 | 0.6557 | 0.6802 | 0.8098 |
| Flight Deck/Cabin Event | 0.5792 | 0.7661 | 0.6597 | 0.8617 |
| Ground Event/Encounter | 0.4963 | 0.6078 | 0.5464 | 0.7762 |
| Ground Excursion | 0.4982 | 0.7861 | 0.6099 | 0.8844 |
| Ground Incursion | 0.6760 | 0.6610 | 0.6684 | 0.8180 |
| Inflight Event/Encounter | 0.6041 | 0.6541 | 0.6281 | 0.7652 |
| **MACRO** | **0.6442** | **0.6943** | **0.6579** | **0.7934** |
| **MICRO** | **0.6694** | **0.7606** | **0.7121** | **0.8421** |

Config: mistral-large-latest, Mistral Batch API, temperature=0.0, max_tokens=256. Parse failures: 2/8,044 (0.0%). Runtime: ~5 min. Cost: $0 (free tier).

### Experiment 11: Mistral Large 3 Few-Shot (taxonomy prompt, Batch API)

| Category | Precision | Recall | F1 | ROC-AUC |
|----------|-----------|--------|------|---------|
| Aircraft Equipment Problem | 0.9182 | 0.6400 | 0.7542 | 0.8086 |
| Airspace Violation | 0.6927 | 0.3899 | 0.4990 | 0.6914 |
| ATC Issue | 0.4606 | 0.6185 | 0.5280 | 0.7349 |
| Conflict | 0.8812 | 0.6364 | 0.7390 | 0.8025 |
| Deviation - Altitude | 0.6929 | 0.8177 | 0.7502 | 0.8732 |
| Deviation - Procedural | 0.7189 | 0.8218 | 0.7669 | 0.6096 |
| Deviation - Speed | 0.5974 | 0.5923 | 0.5948 | 0.7902 |
| Deviation - Track/Heading | 0.6382 | 0.7436 | 0.6869 | 0.8438 |
| Flight Deck/Cabin Event | 0.6137 | 0.7958 | 0.6930 | 0.8787 |
| Ground Event/Encounter | 0.4335 | 0.6486 | 0.5196 | 0.7862 |
| Ground Excursion | 0.4220 | 0.7977 | 0.5520 | 0.8868 |
| Ground Incursion | 0.5964 | 0.7376 | 0.6596 | 0.8492 |
| Inflight Event/Encounter | 0.4534 | 0.7835 | 0.5744 | 0.7555 |
| **MACRO** | **0.6245** | **0.6941** | **0.6398** | **0.7931** |
| **MICRO** | **0.6468** | **0.7303** | **0.6860** | **0.8246** |

Config: mistral-large-latest, Mistral Batch API, temperature=0.0, max_tokens=256. 26 examples (2 per category). Runtime: ~4 min. Cost: $0 (free tier).

### Experiment 12: Ministral 8B Zero-Shot (basic prompt, FP8)

| Category | Precision | Recall | F1 | ROC-AUC |
|----------|-----------|--------|------|---------|
| Aircraft Equipment Problem | 0.7868 | 0.7601 | 0.7733 | 0.8389 |
| Airspace Violation | 0.5691 | 0.2201 | 0.3175 | 0.6066 |
| ATC Issue | 0.3325 | 0.8942 | 0.4848 | 0.7627 |
| Conflict | 0.7681 | 0.7375 | 0.7525 | 0.8280 |
| Deviation - Altitude | 0.5312 | 0.8623 | 0.6574 | 0.8563 |
| Deviation - Procedural | 0.7517 | 0.5084 | 0.6065 | 0.5968 |
| Deviation - Speed | 0.4417 | 0.6824 | 0.5363 | 0.8283 |
| Deviation - Track/Heading | 0.5654 | 0.5127 | 0.5378 | 0.7302 |
| Flight Deck/Cabin Event | 0.1357 | 0.9058 | 0.2360 | 0.7316 |
| Ground Event/Encounter | 0.2763 | 0.6968 | 0.3957 | 0.7664 |
| Ground Excursion | 0.2209 | 0.8439 | 0.3501 | 0.8893 |
| Ground Incursion | 0.4809 | 0.2572 | 0.3352 | 0.6177 |
| Inflight Event/Encounter | 0.5572 | 0.3054 | 0.3945 | 0.6177 |
| **MACRO** | **0.4937** | **0.6298** | **0.4906** | **0.7439** |
| **MICRO** | **0.4856** | **0.6169** | **0.5434** | **0.7420** |

Config: Ministral-3-8B-Instruct-2512 (FP8 multimodal wrapper), vLLM on Modal L4.

### Experiment 13: Ministral 8B Few-Shot (basic prompt, FP8)

| Category | Precision | Recall | F1 | ROC-AUC |
|----------|-----------|--------|------|---------|
| Aircraft Equipment Problem | 0.9464 | 0.5459 | 0.6924 | 0.7668 |
| Airspace Violation | 0.6699 | 0.4340 | 0.5267 | 0.7126 |
| ATC Issue | 0.3568 | 0.7141 | 0.4758 | 0.7248 |
| Conflict | 0.8540 | 0.6215 | 0.7195 | 0.7913 |
| Deviation - Altitude | 0.6008 | 0.7035 | 0.6481 | 0.8058 |
| Deviation - Procedural | 0.7979 | 0.2627 | 0.3952 | 0.5690 |
| Deviation - Speed | 0.4940 | 0.5322 | 0.5124 | 0.7580 |
| Deviation - Track/Heading | 0.6733 | 0.5000 | 0.5739 | 0.7339 |
| Flight Deck/Cabin Event | 0.3125 | 0.7574 | 0.4424 | 0.8148 |
| Ground Event/Encounter | 0.4402 | 0.4495 | 0.4448 | 0.6991 |
| Ground Excursion | 0.3580 | 0.7283 | 0.4800 | 0.8498 |
| Ground Incursion | 0.5921 | 0.6405 | 0.6154 | 0.8029 |
| Inflight Event/Encounter | 0.5438 | 0.4481 | 0.4913 | 0.6698 |
| **MACRO** | **0.5877** | **0.5644** | **0.5398** | **0.7460** |
| **MICRO** | **0.5926** | **0.4895** | **0.5361** | **0.7105** |

Config: Same model, 39 examples (3 per category), batch_size=16.

### Experiment 14: Ministral 8B Fine-Tuned (LoRA on FP8)

| Category | Precision | Recall | F1 | ROC-AUC |
|----------|-----------|--------|------|---------|
| Aircraft Equipment Problem | 0.7873 | 0.7462 | 0.7662 | 0.8328 |
| Airspace Violation | 0.5635 | 0.2233 | 0.3198 | 0.6081 |
| ATC Issue | 0.3313 | 0.8818 | 0.4817 | 0.7581 |
| Conflict | 0.7717 | 0.7338 | 0.7523 | 0.8271 |
| Deviation - Altitude | 0.5204 | 0.8676 | 0.6506 | 0.8552 |
| Deviation - Procedural | 0.7567 | 0.4975 | 0.6003 | 0.5988 |
| Deviation - Speed | 0.4343 | 0.6953 | 0.5347 | 0.8341 |
| Deviation - Track/Heading | 0.5567 | 0.5201 | 0.5378 | 0.7325 |
| Flight Deck/Cabin Event | 0.1401 | 0.9040 | 0.2426 | 0.7393 |
| Ground Event/Encounter | 0.2765 | 0.7014 | 0.3966 | 0.7682 |
| Ground Excursion | 0.2097 | 0.8208 | 0.3341 | 0.8764 |
| Ground Incursion | 0.4775 | 0.2709 | 0.3457 | 0.6238 |
| Inflight Event/Encounter | 0.5519 | 0.3098 | 0.3969 | 0.6186 |
| **MACRO** | **0.4906** | **0.6287** | **0.4892** | **0.7441** |
| **MICRO** | **0.4853** | **0.6123** | **0.5415** | **0.7401** |

Note: Ministral fine-tuning was LoRA on FP8 (not true QLoRA on 4-bit NF4). This produced essentially no improvement over zero-shot. The model became a "yes-machine" with high recall but very low precision.

### Experiment 15: DeepSeek V3.2 Zero-Shot (taxonomy prompt, DeepInfra API)

| Category | Precision | Recall | F1 | ROC-AUC |
|----------|-----------|--------|------|---------|
| Aircraft Equipment Problem | 0.8735 | 0.6883 | 0.7699 | 0.8242 |
| Airspace Violation | 0.6627 | 0.3522 | 0.4600 | 0.6724 |
| ATC Issue | 0.4828 | 0.3078 | 0.3759 | 0.6200 |
| Conflict | 0.8755 | 0.6981 | 0.7768 | 0.8309 |
| Deviation - Altitude | 0.6999 | 0.8132 | 0.7523 | 0.8723 |
| Deviation - Procedural | 0.6892 | 0.8664 | 0.7677 | 0.5669 |
| Deviation - Speed | 0.7850 | 0.3605 | 0.4941 | 0.6788 |
| Deviation - Track/Heading | 0.7768 | 0.5604 | 0.6511 | 0.7695 |
| Flight Deck/Cabin Event | 0.7882 | 0.5846 | 0.6713 | 0.7863 |
| Ground Event/Encounter | 0.4668 | 0.5732 | 0.5146 | 0.7572 |
| Ground Excursion | 0.5989 | 0.6474 | 0.6222 | 0.8189 |
| Ground Incursion | 0.7951 | 0.5486 | 0.6492 | 0.7687 |
| Inflight Event/Encounter | 0.6097 | 0.5680 | 0.5881 | 0.7316 |
| **MACRO** | **0.7003** | **0.5822** | **0.6225** | **0.7460** |
| **MICRO** | **0.7074** | **0.6800** | **0.6934** | **0.8114** |

Config: deepseek-ai/DeepSeek-V3.2 (671B MoE), DeepInfra API, temperature=0.0, max_tokens=500. 50 concurrent requests via aiohttp. Prefix caching: ~62%. Parse failures: 3/8044 (0.0%). Runtime: ~6.5 min. Cost: ~$1.39.

### Experiment 16: DeepSeek V3.2 Zero-Shot + Thinking (taxonomy prompt, DeepInfra API)

| Category | Precision | Recall | F1 | ROC-AUC |
|----------|-----------|--------|------|---------|
| Aircraft Equipment Problem | 0.8612 | 0.7832 | 0.8203 | 0.8664 |
| Airspace Violation | 0.6007 | 0.5723 | 0.5862 | 0.7783 |
| ATC Issue | 0.4408 | 0.6849 | 0.5364 | 0.7532 |
| Conflict | 0.8330 | 0.8400 | 0.8365 | 0.8892 |
| Deviation - Altitude | 0.7392 | 0.8147 | 0.7751 | 0.8791 |
| Deviation - Procedural | 0.7968 | 0.7205 | 0.7568 | 0.6880 |
| Deviation - Speed | 0.5948 | 0.5923 | 0.5935 | 0.7901 |
| Deviation - Track/Heading | 0.7206 | 0.6610 | 0.6895 | 0.8135 |
| Flight Deck/Cabin Event | 0.7454 | 0.6998 | 0.7219 | 0.8407 |
| Ground Event/Encounter | 0.4866 | 0.6305 | 0.5493 | 0.7854 |
| Ground Excursion | 0.5565 | 0.7399 | 0.6352 | 0.8635 |
| Ground Incursion | 0.7641 | 0.6235 | 0.6867 | 0.8042 |
| Inflight Event/Encounter | 0.6835 | 0.6402 | 0.6611 | 0.7773 |
| **MACRO** | **0.6787** | **0.6925** | **0.6807** | **0.8099** |
| **MICRO** | **0.7205** | **0.7247** | **0.7226** | **0.8337** |

Config: Same as non-thinking but reasoning=True, max_tokens=4096. Reasoning via reasoning_content field (clean JSON in content). Prefix caching: ~63%. Parse failures: 46/8044 empty (0.6%). Runtime: ~291 min. Cost: ~$6.73. Thinking adds +0.058 Macro-F1 at 671B scale.

### Experiment 17: Classic ML Subcategory (TF-IDF + XGBoost, 48 labels, 32,089 train)

| Category | Precision | Recall | F1 | ROC-AUC |
|----------|-----------|--------|------|---------|
| ATC Issue | 0.6159 | 0.7660 | 0.6828 | 0.9285 |
| Aircraft Equipment Problem: Critical | 0.6122 | 0.8026 | 0.6946 | 0.9317 |
| Aircraft Equipment Problem: Less Severe | 0.3686 | 0.5801 | 0.4508 | 0.8224 |
| Airspace Violation | 0.5000 | 0.7192 | 0.5899 | 0.9374 |
| Conflict: Airborne Conflict | 0.5433 | 0.7515 | 0.6306 | 0.9251 |
| Conflict: Ground Conflict | 0.4544 | 0.7145 | 0.5555 | 0.9270 |
| Conflict: NMAC | 0.5806 | 0.7840 | 0.6671 | 0.9604 |
| Deviation - Altitude: Crossing Restriction Not Met | 0.6417 | 0.6763 | 0.6586 | 0.9788 |
| Deviation - Altitude: Excursion From Assigned Altitude | 0.3892 | 0.6597 | 0.4895 | 0.9083 |
| Deviation - Altitude: Overshoot | 0.5007 | 0.6914 | 0.5808 | 0.9397 |
| Deviation - Altitude: Undershoot | 0.4531 | 0.4328 | 0.4427 | 0.8973 |
| Deviation - Procedural: Clearance | 0.6612 | 0.7530 | 0.7041 | 0.8835 |
| Deviation - Procedural: FAR | 0.4395 | 0.6442 | 0.5225 | 0.8145 |
| Deviation - Procedural: Hazardous Material Violation | 0.8974 | 0.7609 | 0.8235 | 0.9888 |
| Deviation - Procedural: Landing Without Clearance | 0.5882 | 0.1724 | 0.2667 | 0.9640 |
| Deviation - Procedural: MEL/CDL | 0.4533 | 0.4789 | 0.4658 | 0.9759 |
| Deviation - Procedural: Maintenance | 0.5375 | 0.6938 | 0.6058 | 0.9601 |
| Deviation - Procedural: Other/Unknown | 0.1567 | 0.1789 | 0.1671 | 0.7244 |
| Deviation - Procedural: Published Material/Policy | 0.6235 | 0.6423 | 0.6328 | 0.7600 |
| Deviation - Procedural: Unauthorized Flight Operations (UAS) | 0.4375 | 0.4828 | 0.4590 | 0.9634 |
| Deviation - Procedural: Weight And Balance | 0.4615 | 0.2449 | 0.3200 | 0.9859 |
| Deviation - Speed | 0.4980 | 0.5431 | 0.5196 | 0.9497 |
| Deviation - Track/Heading | 0.6056 | 0.7327 | 0.6631 | 0.9296 |
| Flight Deck/Cabin Event: Illness/Injury | 0.6415 | 0.6018 | 0.6210 | 0.9755 |
| Flight Deck/Cabin Event: Other/Unknown | 0.2805 | 0.3780 | 0.3221 | 0.8943 |
| Flight Deck/Cabin Event: Passenger Misconduct | 0.8000 | 0.7792 | 0.7895 | 0.9850 |
| Flight Deck/Cabin Event: Smoke/Fire/Fumes/Odor | 0.7778 | 0.8561 | 0.8151 | 0.9920 |
| Ground Event/Encounter: Gear Up Landing | 0.2692 | 0.2188 | 0.2414 | 0.9701 |
| Ground Event/Encounter: Ground Equipment Issue | 0.3333 | 0.0714 | 0.1176 | 0.8996 |
| Ground Event/Encounter: Ground Strike - Aircraft | 0.5577 | 0.4833 | 0.5179 | 0.9823 |
| Ground Event/Encounter: Loss Of Aircraft Control | 0.4658 | 0.5743 | 0.5144 | 0.9609 |
| Ground Event/Encounter: Object | 0.7500 | 0.1957 | 0.3103 | 0.9778 |
| Ground Event/Encounter: Other/Unknown | 0.2082 | 0.4064 | 0.2753 | 0.8561 |
| Ground Event/Encounter: Vehicle | 0.2500 | 0.1224 | 0.1644 | 0.9526 |
| Ground Event/Encounter: Weather/Turbulence | 0.0000 | 0.0000 | 0.0000 | 0.9680 |
| Ground Excursion: Runway | 0.5865 | 0.5778 | 0.5821 | 0.9865 |
| Ground Excursion: Taxiway | 0.6154 | 0.2353 | 0.3404 | 0.9873 |
| Ground Incursion: Runway | 0.6180 | 0.8240 | 0.7063 | 0.9767 |
| Ground Incursion: Taxiway | 0.4000 | 0.3585 | 0.3781 | 0.9660 |
| Inflight Event/Encounter: Bird/Animal | 0.4808 | 0.7812 | 0.5952 | 0.9513 |
| Inflight Event/Encounter: CFTT/CFIT | 0.6575 | 0.6704 | 0.6639 | 0.9577 |
| Inflight Event/Encounter: Fuel Issue | 0.6786 | 0.8559 | 0.7570 | 0.9891 |
| Inflight Event/Encounter: Loss Of Aircraft Control | 0.3820 | 0.5065 | 0.4355 | 0.9177 |
| Inflight Event/Encounter: Other/Unknown | 0.2277 | 0.2782 | 0.2505 | 0.7877 |
| Inflight Event/Encounter: Unstabilized Approach | 0.4201 | 0.4176 | 0.4189 | 0.9345 |
| Inflight Event/Encounter: VFR In IMC | 0.6173 | 0.5208 | 0.5650 | 0.9788 |
| Inflight Event/Encounter: Wake Vortex Encounter | 0.8018 | 0.8241 | 0.8128 | 0.9895 |
| Inflight Event/Encounter: Weather/Turbulence | 0.6254 | 0.7766 | 0.6928 | 0.9418 |
| **MACRO** | **0.5097** | **0.5463** | **0.5100** | **0.9341** |
| **MICRO** | **0.5405** | **0.6731** | **0.5995** | **0.9556** |

Config: Same TF-IDF + XGBoost params as parent baseline. 48 independent binary classifiers. Runtime: 142 min on Modal 32-core CPU (~$3.03).

### Experiment 18: Classic ML Tuned Subcategory (TF-IDF + XGBoost, 48 labels, 32,089 train)

| Category | Precision | Recall | F1 | ROC-AUC |
|----------|-----------|--------|------|---------|
| ATC Issue | 0.6210 | 0.7740 | 0.6891 | 0.9272 |
| Aircraft Equipment Problem: Critical | 0.6235 | 0.8018 | 0.7015 | 0.9320 |
| Aircraft Equipment Problem: Less Severe | 0.3755 | 0.5957 | 0.4606 | 0.8217 |
| Airspace Violation | 0.4989 | 0.7035 | 0.5838 | 0.9350 |
| Conflict: Airborne Conflict | 0.5467 | 0.7596 | 0.6358 | 0.9269 |
| Conflict: Ground Conflict | 0.4560 | 0.7178 | 0.5577 | 0.9255 |
| Conflict: NMAC | 0.5729 | 0.7619 | 0.6540 | 0.9615 |
| Deviation - Altitude: Crossing Restriction Not Met | 0.6482 | 0.6805 | 0.6640 | 0.9801 |
| Deviation - Altitude: Excursion From Assigned Altitude | 0.3852 | 0.6548 | 0.4851 | 0.9062 |
| Deviation - Altitude: Overshoot | 0.5000 | 0.6836 | 0.5776 | 0.9388 |
| Deviation - Altitude: Undershoot | 0.4560 | 0.4254 | 0.4402 | 0.8928 |
| Deviation - Procedural: Clearance | 0.6639 | 0.7555 | 0.7068 | 0.8831 |
| Deviation - Procedural: FAR | 0.4314 | 0.6382 | 0.5148 | 0.8111 |
| Deviation - Procedural: Hazardous Material Violation | 0.8571 | 0.7826 | 0.8182 | 0.9882 |
| Deviation - Procedural: Landing Without Clearance | 0.7143 | 0.1724 | 0.2778 | 0.9619 |
| Deviation - Procedural: MEL/CDL | 0.4458 | 0.5211 | 0.4805 | 0.9803 |
| Deviation - Procedural: Maintenance | 0.5324 | 0.7016 | 0.6054 | 0.9600 |
| Deviation - Procedural: Other/Unknown | 0.1734 | 0.2263 | 0.1963 | 0.7221 |
| Deviation - Procedural: Published Material/Policy | 0.6236 | 0.6460 | 0.6346 | 0.7612 |
| Deviation - Procedural: Unauthorized Flight Operations (UAS) | 0.4412 | 0.5172 | 0.4762 | 0.9750 |
| Deviation - Procedural: Weight And Balance | 0.4000 | 0.2449 | 0.3038 | 0.9844 |
| Deviation - Speed | 0.5233 | 0.5819 | 0.5510 | 0.9507 |
| Deviation - Track/Heading | 0.5959 | 0.7178 | 0.6512 | 0.9301 |
| Flight Deck/Cabin Event: Illness/Injury | 0.6436 | 0.5752 | 0.6075 | 0.9748 |
| Flight Deck/Cabin Event: Other/Unknown | 0.2467 | 0.3415 | 0.2864 | 0.8896 |
| Flight Deck/Cabin Event: Passenger Misconduct | 0.7945 | 0.7532 | 0.7733 | 0.9851 |
| Flight Deck/Cabin Event: Smoke/Fire/Fumes/Odor | 0.7888 | 0.8597 | 0.8227 | 0.9927 |
| Ground Event/Encounter: Gear Up Landing | 0.3077 | 0.2500 | 0.2759 | 0.9694 |
| Ground Event/Encounter: Ground Equipment Issue | 0.4167 | 0.1190 | 0.1852 | 0.9072 |
| Ground Event/Encounter: Ground Strike - Aircraft | 0.5686 | 0.4833 | 0.5225 | 0.9791 |
| Ground Event/Encounter: Loss Of Aircraft Control | 0.4560 | 0.5823 | 0.5115 | 0.9614 |
| Ground Event/Encounter: Object | 0.6923 | 0.1957 | 0.3051 | 0.9801 |
| Ground Event/Encounter: Other/Unknown | 0.2087 | 0.4183 | 0.2785 | 0.8545 |
| Ground Event/Encounter: Vehicle | 0.1500 | 0.0612 | 0.0870 | 0.9572 |
| Ground Event/Encounter: Weather/Turbulence | 0.5000 | 0.0400 | 0.0741 | 0.9629 |
| Ground Excursion: Runway | 0.5878 | 0.5704 | 0.5789 | 0.9835 |
| Ground Excursion: Taxiway | 0.5556 | 0.1471 | 0.2326 | 0.9883 |
| Ground Incursion: Runway | 0.6302 | 0.8219 | 0.7134 | 0.9755 |
| Ground Incursion: Taxiway | 0.3600 | 0.3396 | 0.3495 | 0.9624 |
| Inflight Event/Encounter: Bird/Animal | 0.4808 | 0.7812 | 0.5952 | 0.9517 |
| Inflight Event/Encounter: CFTT/CFIT | 0.6559 | 0.6816 | 0.6685 | 0.9571 |
| Inflight Event/Encounter: Fuel Issue | 0.6786 | 0.8559 | 0.7570 | 0.9917 |
| Inflight Event/Encounter: Loss Of Aircraft Control | 0.3697 | 0.5032 | 0.4262 | 0.9188 |
| Inflight Event/Encounter: Other/Unknown | 0.2310 | 0.2823 | 0.2541 | 0.7805 |
| Inflight Event/Encounter: Unstabilized Approach | 0.4540 | 0.4647 | 0.4593 | 0.9348 |
| Inflight Event/Encounter: VFR In IMC | 0.6049 | 0.5104 | 0.5537 | 0.9762 |
| Inflight Event/Encounter: Wake Vortex Encounter | 0.8037 | 0.7963 | 0.8000 | 0.9901 |
| Inflight Event/Encounter: Weather/Turbulence | 0.6211 | 0.7827 | 0.6926 | 0.9446 |
| **MACRO** | **0.5186** | **0.5475** | **0.5099** | **0.9339** |
| **MICRO** | **0.5399** | **0.6746** | **0.5998** | **0.9553** |

Config: Same params as baseline, retrained via Phase 3. Delta vs subcategory baseline: Macro-F1 -0.0001, Micro-F1 +0.0003. Near-identical.

### Experiment 19: Mistral Large 3 Zero-Shot Subcategory (48 labels, taxonomy prompt)

| Category | Precision | Recall | F1 | ROC-AUC |
|----------|-----------|--------|------|---------|
| ATC Issue | 0.3989 | 0.8173 | 0.5361 | 0.7825 |
| Aircraft Equipment Problem: Critical | 0.6393 | 0.7186 | 0.6766 | 0.8193 |
| Aircraft Equipment Problem: Less Severe | 0.3729 | 0.3247 | 0.3472 | 0.6249 |
| Airspace Violation | 0.6446 | 0.4921 | 0.5581 | 0.7405 |
| Conflict: Airborne Conflict | 0.5376 | 0.7840 | 0.6378 | 0.8447 |
| Conflict: Ground Conflict | 0.7741 | 0.3409 | 0.4734 | 0.6664 |
| Conflict: NMAC | 0.5281 | 0.7517 | 0.6204 | 0.8493 |
| Deviation - Altitude: Crossing Restriction Not Met | 0.6293 | 0.6763 | 0.6520 | 0.8320 |
| Deviation - Altitude: Excursion From Assigned Altitude | 0.3290 | 0.6468 | 0.4361 | 0.7681 |
| Deviation - Altitude: Overshoot | 0.5169 | 0.2988 | 0.3787 | 0.6399 |
| Deviation - Altitude: Undershoot | 0.0060 | 0.0075 | 0.0066 | 0.4932 |
| Deviation - Procedural: Clearance | 0.5584 | 0.5076 | 0.5318 | 0.6703 |
| Deviation - Procedural: FAR | 0.7530 | 0.0827 | 0.1490 | 0.5382 |
| Deviation - Procedural: Hazardous Material Violation | 0.8000 | 0.7826 | 0.7912 | 0.8907 |
| Deviation - Procedural: Landing Without Clearance | 0.4727 | 0.4483 | 0.4602 | 0.7223 |
| Deviation - Procedural: MEL/CDL | 0.2783 | 0.9014 | 0.4252 | 0.9403 |
| Deviation - Procedural: Maintenance | 0.4564 | 0.6899 | 0.5494 | 0.8313 |
| Deviation - Procedural: Other/Unknown | 0.0246 | 0.0158 | 0.0192 | 0.5003 |
| Deviation - Procedural: Published Material/Policy | 0.6301 | 0.2883 | 0.3956 | 0.5872 |
| Deviation - Procedural: Unauthorized Flight Operations (UAS) | 0.5676 | 0.7241 | 0.6364 | 0.8611 |
| Deviation - Procedural: Weight And Balance | 0.1928 | 0.3265 | 0.2424 | 0.6591 |
| Deviation - Speed | 0.6058 | 0.5431 | 0.5727 | 0.7663 |
| Deviation - Track/Heading | 0.6729 | 0.6550 | 0.6638 | 0.8064 |
| Flight Deck/Cabin Event: Illness/Injury | 0.4464 | 0.6637 | 0.5338 | 0.8260 |
| Flight Deck/Cabin Event: Other/Unknown | 0.3187 | 0.1768 | 0.2275 | 0.5845 |
| Flight Deck/Cabin Event: Passenger Misconduct | 0.8816 | 0.8701 | 0.8758 | 0.9345 |
| Flight Deck/Cabin Event: Smoke/Fire/Fumes/Odor | 0.7814 | 0.8741 | 0.8251 | 0.9327 |
| Ground Event/Encounter: Gear Up Landing | 0.2989 | 0.8125 | 0.4370 | 0.9024 |
| Ground Event/Encounter: Ground Equipment Issue | 0.0556 | 0.0238 | 0.0333 | 0.5108 |
| Ground Event/Encounter: Ground Strike - Aircraft | 0.1604 | 0.5000 | 0.2429 | 0.7401 |
| Ground Event/Encounter: Loss Of Aircraft Control | 0.5507 | 0.5020 | 0.5252 | 0.7444 |
| Ground Event/Encounter: Object | 0.1100 | 0.4783 | 0.1789 | 0.7280 |
| Ground Event/Encounter: Other/Unknown | 0.2215 | 0.2869 | 0.2500 | 0.6271 |
| Ground Event/Encounter: Vehicle | 0.1875 | 0.6735 | 0.2933 | 0.8278 |
| Ground Event/Encounter: Weather/Turbulence | 0.0714 | 0.1600 | 0.0988 | 0.5767 |
| Ground Excursion: Runway | 0.6296 | 0.7556 | 0.6869 | 0.8740 |
| Ground Excursion: Taxiway | 0.5455 | 0.3529 | 0.4286 | 0.6758 |
| Ground Incursion: Runway | 0.6421 | 0.7764 | 0.7029 | 0.8743 |
| Ground Incursion: Taxiway | 0.4390 | 0.1698 | 0.2449 | 0.5835 |
| Inflight Event/Encounter: Bird/Animal | 0.3733 | 0.8750 | 0.5234 | 0.9346 |
| Inflight Event/Encounter: CFTT/CFIT | 0.7627 | 0.2514 | 0.3782 | 0.6239 |
| Inflight Event/Encounter: Fuel Issue | 0.3722 | 0.6036 | 0.4605 | 0.7947 |
| Inflight Event/Encounter: Loss Of Aircraft Control | 0.4415 | 0.2677 | 0.3333 | 0.6271 |
| Inflight Event/Encounter: Other/Unknown | 0.1432 | 0.2137 | 0.1715 | 0.5865 |
| Inflight Event/Encounter: Unstabilized Approach | 0.3454 | 0.5059 | 0.4105 | 0.7426 |
| Inflight Event/Encounter: VFR In IMC | 0.4127 | 0.8125 | 0.5474 | 0.8992 |
| Inflight Event/Encounter: Wake Vortex Encounter | 0.6494 | 0.9259 | 0.7634 | 0.9595 |
| Inflight Event/Encounter: Weather/Turbulence | 0.7770 | 0.5189 | 0.6223 | 0.7510 |
| **MACRO** | **0.4585** | **0.5182** | **0.4491** | **0.7437** |
| **MICRO** | **0.4999** | **0.4878** | **0.4938** | **0.7298** |

Config: mistral-large-latest, real-time API (batch API was stuck). 48-label taxonomy-enriched prompt. Runtime: ~119 min, 7 network errors, 0.1% parse failures. Cost: paid plan.

### Experiment 20: DeepSeek V3.2 Zero-Shot Subcategory (48 labels, taxonomy prompt)

| Category | Precision | Recall | F1 | ROC-AUC |
|----------|-----------|--------|------|---------|
| ATC Issue | 0.5380 | 0.4050 | 0.4621 | 0.6669 |
| Aircraft Equipment Problem: Critical | 0.6441 | 0.6846 | 0.6637 | 0.8049 |
| Aircraft Equipment Problem: Less Severe | 0.4707 | 0.2161 | 0.2962 | 0.5914 |
| Airspace Violation | 0.6875 | 0.3123 | 0.4295 | 0.6532 |
| Conflict: Airborne Conflict | 0.6204 | 0.5669 | 0.5925 | 0.7591 |
| Conflict: Ground Conflict | 0.7966 | 0.3768 | 0.5116 | 0.6844 |
| Conflict: NMAC | 0.6031 | 0.7211 | 0.6569 | 0.8418 |
| Deviation - Altitude: Crossing Restriction Not Met | 0.6933 | 0.6473 | 0.6695 | 0.8192 |
| Deviation - Altitude: Excursion From Assigned Altitude | 0.3233 | 0.6774 | 0.4377 | 0.7793 |
| Deviation - Altitude: Overshoot | 0.7857 | 0.0645 | 0.1191 | 0.5316 |
| Deviation - Altitude: Undershoot | 0.0263 | 0.0149 | 0.0190 | 0.5028 |
| Deviation - Procedural: Clearance | 0.4883 | 0.4975 | 0.4929 | 0.6403 |
| Deviation - Procedural: FAR | 0.7196 | 0.1019 | 0.1784 | 0.5463 |
| Deviation - Procedural: Hazardous Material Violation | 0.8250 | 0.7174 | 0.7674 | 0.8583 |
| Deviation - Procedural: Landing Without Clearance | 0.3898 | 0.3966 | 0.3932 | 0.6960 |
| Deviation - Procedural: MEL/CDL | 0.3471 | 0.8310 | 0.4896 | 0.9085 |
| Deviation - Procedural: Maintenance | 0.6147 | 0.5504 | 0.5808 | 0.7695 |
| Deviation - Procedural: Other/Unknown | 0.0462 | 0.1000 | 0.0632 | 0.5250 |
| Deviation - Procedural: Published Material/Policy | 0.6495 | 0.1953 | 0.3003 | 0.5622 |
| Deviation - Procedural: Unauthorized Flight Operations (UAS) | 0.4318 | 0.6552 | 0.5205 | 0.8260 |
| Deviation - Procedural: Weight And Balance | 0.2424 | 0.1633 | 0.1951 | 0.5801 |
| Deviation - Speed | 0.7619 | 0.4138 | 0.5363 | 0.7050 |
| Deviation - Track/Heading | 0.7946 | 0.4696 | 0.5904 | 0.7268 |
| Flight Deck/Cabin Event: Illness/Injury | 0.4512 | 0.6549 | 0.5343 | 0.8217 |
| Flight Deck/Cabin Event: Other/Unknown | 0.1786 | 0.0305 | 0.0521 | 0.5138 |
| Flight Deck/Cabin Event: Passenger Misconduct | 0.9492 | 0.7273 | 0.8235 | 0.8634 |
| Flight Deck/Cabin Event: Smoke/Fire/Fumes/Odor | 0.8479 | 0.8022 | 0.8244 | 0.8985 |
| Ground Event/Encounter: Gear Up Landing | 0.3200 | 0.7500 | 0.4486 | 0.8718 |
| Ground Event/Encounter: Ground Equipment Issue | 0.1053 | 0.1429 | 0.1212 | 0.5682 |
| Ground Event/Encounter: Ground Strike - Aircraft | 0.1678 | 0.4000 | 0.2365 | 0.6925 |
| Ground Event/Encounter: Loss Of Aircraft Control | 0.6754 | 0.3092 | 0.4242 | 0.6522 |
| Ground Event/Encounter: Object | 0.1646 | 0.2826 | 0.2080 | 0.6372 |
| Ground Event/Encounter: Other/Unknown | 0.2880 | 0.2191 | 0.2489 | 0.6008 |
| Ground Event/Encounter: Vehicle | 0.1942 | 0.5510 | 0.2872 | 0.7685 |
| Ground Event/Encounter: Weather/Turbulence | 0.0667 | 0.0800 | 0.0727 | 0.5382 |
| Ground Excursion: Runway | 0.6383 | 0.6667 | 0.6522 | 0.8301 |
| Ground Excursion: Taxiway | 0.4783 | 0.3235 | 0.3860 | 0.6610 |
| Ground Incursion: Runway | 0.7759 | 0.6812 | 0.7255 | 0.8343 |
| Ground Incursion: Taxiway | 0.5152 | 0.1604 | 0.2446 | 0.5792 |
| Inflight Event/Encounter: Bird/Animal | 0.4375 | 0.8750 | 0.5833 | 0.9352 |
| Inflight Event/Encounter: CFTT/CFIT | 0.7710 | 0.2821 | 0.4131 | 0.6391 |
| Inflight Event/Encounter: Fuel Issue | 0.3455 | 0.3423 | 0.3439 | 0.6666 |
| Inflight Event/Encounter: Loss Of Aircraft Control | 0.4607 | 0.1323 | 0.2055 | 0.5630 |
| Inflight Event/Encounter: Other/Unknown | 0.1422 | 0.1210 | 0.1307 | 0.5488 |
| Inflight Event/Encounter: Unstabilized Approach | 0.3070 | 0.5706 | 0.3992 | 0.7713 |
| Inflight Event/Encounter: VFR In IMC | 0.5573 | 0.7604 | 0.6432 | 0.8765 |
| Inflight Event/Encounter: Wake Vortex Encounter | 0.5506 | 0.9074 | 0.6853 | 0.9486 |
| Inflight Event/Encounter: Weather/Turbulence | 0.7552 | 0.4896 | 0.5941 | 0.7358 |
| **MACRO** | **0.4926** | **0.4384** | **0.4220** | **0.7082** |
| **MICRO** | **0.5361** | **0.3960** | **0.4555** | **0.6881** |

Config: deepseek-ai/DeepSeek-V3.2, DeepInfra API, temperature=0.0, max_tokens=500. Prefix caching: ~79%. Parse failures: 3/8017 (0.0%). Runtime: ~7.5 min. Cost: ~$1.92.

### Experiment 21: DeepSeek V3.2 Zero-Shot + Thinking Subcategory (48 labels, taxonomy prompt)

| Category | Precision | Recall | F1 | ROC-AUC |
|----------|-----------|--------|------|---------|
| ATC Issue | 0.5174 | 0.7645 | 0.6171 | 0.8092 |
| Aircraft Equipment Problem: Critical | 0.6796 | 0.5983 | 0.6364 | 0.7713 |
| Aircraft Equipment Problem: Less Severe | 0.5035 | 0.2989 | 0.3751 | 0.6292 |
| Airspace Violation | 0.7045 | 0.5489 | 0.6170 | 0.7697 |
| Conflict: Airborne Conflict | 0.6240 | 0.5639 | 0.5924 | 0.7581 |
| Conflict: Ground Conflict | 0.7921 | 0.5595 | 0.6558 | 0.7737 |
| Conflict: NMAC | 0.6266 | 0.6735 | 0.6492 | 0.8209 |
| Deviation - Altitude: Crossing Restriction Not Met | 0.7043 | 0.6722 | 0.6879 | 0.8317 |
| Deviation - Altitude: Excursion From Assigned Altitude | 0.4784 | 0.5371 | 0.5061 | 0.7440 |
| Deviation - Altitude: Overshoot | 0.7320 | 0.2188 | 0.3368 | 0.6066 |
| Deviation - Altitude: Undershoot | 0.0211 | 0.0224 | 0.0217 | 0.5024 |
| Deviation - Procedural: Clearance | 0.7198 | 0.3990 | 0.5134 | 0.6672 |
| Deviation - Procedural: FAR | 0.7172 | 0.0688 | 0.1255 | 0.5312 |
| Deviation - Procedural: Hazardous Material Violation | 0.8182 | 0.7826 | 0.8000 | 0.8908 |
| Deviation - Procedural: Landing Without Clearance | 0.5750 | 0.3966 | 0.4694 | 0.6972 |
| Deviation - Procedural: MEL/CDL | 0.3846 | 0.7042 | 0.4975 | 0.8471 |
| Deviation - Procedural: Maintenance | 0.5327 | 0.6628 | 0.5907 | 0.8217 |
| Deviation - Procedural: Other/Unknown | 0.0556 | 0.0421 | 0.0479 | 0.5124 |
| Deviation - Procedural: Published Material/Policy | 0.7243 | 0.1832 | 0.2924 | 0.5681 |
| Deviation - Procedural: Unauthorized Flight Operations (UAS) | 0.4681 | 0.7586 | 0.5789 | 0.8777 |
| Deviation - Procedural: Weight And Balance | 0.2615 | 0.3469 | 0.2982 | 0.6705 |
| Deviation - Speed | 0.7414 | 0.3707 | 0.4943 | 0.6834 |
| Deviation - Track/Heading | 0.7227 | 0.3802 | 0.4983 | 0.6804 |
| Flight Deck/Cabin Event: Illness/Injury | 0.5211 | 0.6549 | 0.5804 | 0.8231 |
| Flight Deck/Cabin Event: Other/Unknown | 0.1930 | 0.0671 | 0.0995 | 0.5306 |
| Flight Deck/Cabin Event: Passenger Misconduct | 0.9394 | 0.8052 | 0.8671 | 0.9023 |
| Flight Deck/Cabin Event: Smoke/Fire/Fumes/Odor | 0.7863 | 0.6619 | 0.7188 | 0.8277 |
| Ground Event/Encounter: Gear Up Landing | 0.4667 | 0.4375 | 0.4516 | 0.7177 |
| Ground Event/Encounter: Ground Equipment Issue | 0.1216 | 0.2143 | 0.1552 | 0.6031 |
| Ground Event/Encounter: Ground Strike - Aircraft | 0.1770 | 0.3333 | 0.2312 | 0.6608 |
| Ground Event/Encounter: Loss Of Aircraft Control | 0.6127 | 0.3494 | 0.4450 | 0.6712 |
| Ground Event/Encounter: Object | 0.1515 | 0.2174 | 0.1786 | 0.6052 |
| Ground Event/Encounter: Other/Unknown | 0.1915 | 0.0359 | 0.0604 | 0.5155 |
| Ground Event/Encounter: Vehicle | 0.2676 | 0.3878 | 0.3167 | 0.6906 |
| Ground Event/Encounter: Weather/Turbulence | 0.0556 | 0.1200 | 0.0759 | 0.5568 |
| Ground Excursion: Runway | 0.7045 | 0.4593 | 0.5561 | 0.7280 |
| Ground Excursion: Taxiway | 0.5556 | 0.2941 | 0.3846 | 0.6466 |
| Ground Incursion: Runway | 0.8328 | 0.5259 | 0.6447 | 0.7596 |
| Ground Incursion: Taxiway | 0.6190 | 0.1226 | 0.2047 | 0.5608 |
| Inflight Event/Encounter: Bird/Animal | 0.5000 | 0.4062 | 0.4483 | 0.7023 |
| Inflight Event/Encounter: CFTT/CFIT | 0.7143 | 0.1257 | 0.2138 | 0.5617 |
| Inflight Event/Encounter: Fuel Issue | 0.4203 | 0.5225 | 0.4659 | 0.7562 |
| Inflight Event/Encounter: Loss Of Aircraft Control | 0.5632 | 0.1581 | 0.2469 | 0.5766 |
| Inflight Event/Encounter: Other/Unknown | 0.2500 | 0.0363 | 0.0634 | 0.5164 |
| Inflight Event/Encounter: Unstabilized Approach | 0.3218 | 0.3824 | 0.3495 | 0.6824 |
| Inflight Event/Encounter: VFR In IMC | 0.6452 | 0.2083 | 0.3150 | 0.6035 |
| Inflight Event/Encounter: Wake Vortex Encounter | 0.7857 | 0.6111 | 0.6875 | 0.8044 |
| Inflight Event/Encounter: Weather/Turbulence | 0.7054 | 0.3040 | 0.4249 | 0.6448 |
| **MACRO** | **0.5251** | **0.3957** | **0.4185** | **0.6898** |
| **MICRO** | **0.5989** | **0.3815** | **0.4661** | **0.6834** |

Config: Same as non-thinking but reasoning=True, max_tokens=4096. 21.6% parse failures (1729/8017 empty) -- reasoning tokens exhaust max_tokens. Runtime: ~545 min. Cost: ~$5.24. Thinking HURTS subcategory: -0.003 Macro-F1.

### Experiment 22: Qwen3-8B Zero-Shot Subcategory (48 labels, taxonomy prompt)

| Category | Precision | Recall | F1 | ROC-AUC |
|----------|-----------|--------|------|---------|
| ATC Issue | 0.4428 | 0.1335 | 0.2052 | 0.5496 |
| Aircraft Equipment Problem: Critical | 0.6245 | 0.4402 | 0.5164 | 0.6940 |
| Aircraft Equipment Problem: Less Severe | 0.4396 | 0.0414 | 0.0756 | 0.5171 |
| Airspace Violation | 0.6667 | 0.0442 | 0.0828 | 0.5216 |
| Conflict: Airborne Conflict | 0.3725 | 0.7170 | 0.4903 | 0.7738 |
| Conflict: Ground Conflict | 0.9444 | 0.0277 | 0.0539 | 0.5138 |
| Conflict: NMAC | 0.4286 | 0.1633 | 0.2365 | 0.5730 |
| Deviation - Altitude: Crossing Restriction Not Met | 0.4815 | 0.2158 | 0.2980 | 0.6043 |
| Deviation - Altitude: Excursion From Assigned Altitude | 0.3620 | 0.0952 | 0.1507 | 0.5406 |
| Deviation - Altitude: Overshoot | 0.3448 | 0.1953 | 0.2494 | 0.5850 |
| Deviation - Altitude: Undershoot | 0.0147 | 0.0149 | 0.0148 | 0.4990 |
| Deviation - Procedural: Clearance | 0.4883 | 0.7267 | 0.5841 | 0.7049 |
| Deviation - Procedural: FAR | 0.2943 | 0.1878 | 0.2293 | 0.5416 |
| Deviation - Procedural: Hazardous Material Violation | 0.4400 | 0.7174 | 0.5455 | 0.8561 |
| Deviation - Procedural: Landing Without Clearance | 0.0553 | 0.4828 | 0.0993 | 0.7114 |
| Deviation - Procedural: MEL/CDL | 0.2389 | 0.8310 | 0.3711 | 0.9037 |
| Deviation - Procedural: Maintenance | 0.4000 | 0.2248 | 0.2878 | 0.6068 |
| Deviation - Procedural: Other/Unknown | 0.0261 | 0.2526 | 0.0472 | 0.5117 |
| Deviation - Procedural: Published Material/Policy | 0.5920 | 0.1785 | 0.2744 | 0.5478 |
| Deviation - Procedural: Unauthorized Flight Operations (UAS) | 0.1792 | 0.6552 | 0.2815 | 0.8221 |
| Deviation - Procedural: Weight And Balance | 0.0654 | 0.2041 | 0.0990 | 0.5931 |
| Deviation - Speed | 0.5000 | 0.1034 | 0.1714 | 0.5502 |
| Deviation - Track/Heading | 0.6136 | 0.0288 | 0.0549 | 0.5132 |
| Flight Deck/Cabin Event: Illness/Injury | 0.4103 | 0.5664 | 0.4758 | 0.7774 |
| Flight Deck/Cabin Event: Other/Unknown | 0.0788 | 0.0976 | 0.0872 | 0.5369 |
| Flight Deck/Cabin Event: Passenger Misconduct | 0.5248 | 0.6883 | 0.5955 | 0.8411 |
| Flight Deck/Cabin Event: Smoke/Fire/Fumes/Odor | 0.7331 | 0.6223 | 0.6732 | 0.8071 |
| Ground Event/Encounter: Gear Up Landing | 0.1057 | 0.4062 | 0.1677 | 0.6962 |
| Ground Event/Encounter: Ground Equipment Issue | 0.0357 | 0.0238 | 0.0286 | 0.5102 |
| Ground Event/Encounter: Ground Strike - Aircraft | 0.0503 | 0.1667 | 0.0772 | 0.5715 |
| Ground Event/Encounter: Loss Of Aircraft Control | 0.2876 | 0.2610 | 0.2737 | 0.6202 |
| Ground Event/Encounter: Object | 0.0468 | 0.7391 | 0.0880 | 0.8261 |
| Ground Event/Encounter: Other/Unknown | 0.0963 | 0.1235 | 0.1082 | 0.5430 |
| Ground Event/Encounter: Vehicle | 0.1263 | 0.2449 | 0.1667 | 0.6172 |
| Ground Event/Encounter: Weather/Turbulence | 0.0164 | 0.0400 | 0.0233 | 0.5162 |
| Ground Excursion: Runway | 0.5125 | 0.3037 | 0.3814 | 0.6494 |
| Ground Excursion: Taxiway | 0.2000 | 0.2941 | 0.2381 | 0.6446 |
| Ground Incursion: Runway | 0.4306 | 0.1284 | 0.1978 | 0.5587 |
| Ground Incursion: Taxiway | 0.1892 | 0.1321 | 0.1556 | 0.5622 |
| Inflight Event/Encounter: Bird/Animal | 0.2347 | 0.7188 | 0.3538 | 0.8547 |
| Inflight Event/Encounter: CFTT/CFIT | 0.0769 | 0.0028 | 0.0054 | 0.5006 |
| Inflight Event/Encounter: Fuel Issue | 0.1481 | 0.1441 | 0.1461 | 0.5663 |
| Inflight Event/Encounter: Loss Of Aircraft Control | 0.1900 | 0.1839 | 0.1869 | 0.5762 |
| Inflight Event/Encounter: Other/Unknown | 0.0824 | 0.0605 | 0.0698 | 0.5195 |
| Inflight Event/Encounter: Unstabilized Approach | 0.2402 | 0.3235 | 0.2757 | 0.6507 |
| Inflight Event/Encounter: VFR In IMC | 0.2708 | 0.1354 | 0.1806 | 0.5655 |
| Inflight Event/Encounter: Wake Vortex Encounter | 0.4437 | 0.6574 | 0.5299 | 0.8231 |
| Inflight Event/Encounter: Weather/Turbulence | 0.4098 | 0.3468 | 0.3757 | 0.6450 |
| **MACRO** | **0.3116** | **0.2936** | **0.2350** | **0.6294** |
| **MICRO** | **0.3340** | **0.2789** | **0.3040** | **0.6234** |

Config: Qwen3-8B, vLLM on Modal L4, taxonomy-enriched prompt. Runtime: ~30 min (~$0.40). Dramatic drop from parent: Macro-F1 0.499->0.235, Micro-F1 0.605->0.304.

---

## Section 7: Prompt Templates (Verbatim from Scripts)

### Basic System Prompt (used in zero-shot, few-shot, fine-tuning)

```
You are an aviation safety analyst classifying ASRS incident reports. For each report, identify ALL applicable anomaly categories from the list below. A report can belong to multiple categories. Return ONLY a JSON array of matching category names, nothing else.

Categories:
- Aircraft Equipment Problem
- Airspace Violation
- ATC Issue
- Conflict
- Deviation - Altitude
- Deviation - Procedural
- Deviation - Speed
- Deviation - Track/Heading
- Flight Deck/Cabin Event
- Ground Event/Encounter
- Ground Excursion
- Ground Incursion
- Inflight Event/Encounter
```

### Taxonomy-Enriched System Prompt (used in taxonomy experiments + Mistral Large)

```
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
- Deviation - Procedural is very broad -- when in doubt about procedural compliance, include it

A report can belong to multiple categories. Only select categories clearly supported by the narrative. Be precise -- avoid over-predicting.
```

### User Message Template (all experiments)

```
Classify this ASRS report into applicable anomaly categories:

{narrative}
```

(For zero-shot basic, the user message was: "Classify this ASRS report into applicable anomaly categories:\n\n{narrative}". For all other experiments including few-shot and fine-tuning, the user message was: "Classify this ASRS report:\n\n{narrative}")

### Few-Shot Example Format

Each few-shot example is presented as a user/assistant turn pair. The selection strategy (`select_few_shot_examples()`) picks `n_per_cat` examples per category from the training set, preferring examples with fewer total labels (clearer signal) and shorter narratives (to save context budget). An ACN tracking set prevents the same report from being selected for multiple categories.

- **Qwen3-8B few-shot (basic):** 3 examples/category = 39 total, narrative truncated to 600 chars
- **Qwen3-8B few-shot (taxonomy):** 3 examples/category = 39 total, narrative truncated to 600 chars
- **Mistral Large 3 few-shot:** 2 examples/category = 26 total, narrative truncated to 600 chars

Format of each example in the message history:
```
User: Classify this ASRS report:

{example_narrative_truncated_to_600_chars}
Assistant: ["Category A", "Category B"]
```

The assistant response is always a JSON array of category names.

### Fine-Tuning Instruction Format

For QLoRA fine-tuning, each training example is a complete chat conversation with system + user + assistant messages, tokenized via `tokenizer.apply_chat_template()` with `enable_thinking=False`:

```python
def format_example(row):
    active_cats = [c for c in categories if row.get(c, 0) == 1]
    narrative = str(row["Narrative"])[:1500]
    msgs = [
        {"role": "system", "content": system_msg},  # basic system prompt
        {"role": "user", "content": f"Classify this ASRS report:\n\n{narrative}"},
        {"role": "assistant", "content": json.dumps(active_cats)},
    ]
    return tokenizer.apply_chat_template(msgs, tokenize=False, enable_thinking=False)
```

Key parameters:
- Narrative truncated to 1,500 characters (vs 600 for few-shot examples, 2,000 for zero-shot inference)
- `enable_thinking=False` prevents Qwen3's thinking mode tokens from appearing in training data
- Assistant response is a plain JSON array of the ground-truth category names

### Output Parsing Logic (Three-Tier Strategy)

All LLM experiments use the same three-tier parsing strategy to convert raw model output into category predictions:

```python
def parse_llm_output(raw: str, categories: list[str]) -> list[str]:
    cat_lower = {c.lower(): c for c in categories}

    # Tier 1: Direct JSON parse
    try:
        parsed = json.loads(raw.strip())
        if isinstance(parsed, list):
            return _normalize(parsed, cat_lower)
    except (json.JSONDecodeError, TypeError):
        pass

    # Tier 2: Regex -- find first [...] block
    m = re.search(r"\[.*?\]", raw, re.DOTALL)
    if m:
        try:
            parsed = json.loads(m.group())
            if isinstance(parsed, list):
                return _normalize(parsed, cat_lower)
        except (json.JSONDecodeError, TypeError):
            pass

    # Tier 3: Fuzzy substring matching
    matched = []
    raw_lower = raw.lower()
    for cat in categories:
        if cat.lower() in raw_lower:
            matched.append(cat)
    return matched
```

The `_normalize()` function maps parsed items to exact category names via case-insensitive matching. For Mistral Large experiments, it also strips subcategory suffixes (e.g., "Aircraft Equipment Problem: Less Severe" -> "Aircraft Equipment Problem") and handles markdown code fence stripping (```` ```json ... ``` ````).

### Thinking Mode Specifics

For the thinking mode experiment (`modal_few_shot_taxonomy_thinking.py`):
- `enable_thinking=True` in vLLM `chat_template_kwargs`
- `max_tokens=4096` (vs 256 for non-thinking) to accommodate reasoning tokens
- `max_model_len=32768` (vs 16384) for KV cache headroom
- A100 GPU required (L4 cannot fit the increased KV cache)
- Thinking blocks are stripped before JSON parsing:

```python
def strip_thinking(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
```

### Mistral Large Batch API Specifics

Mistral Large experiments use file-based Batch API instead of real-time API:
- Real-time API rate limits (60 RPM, 50K TPM) are too restrictive for 8,044 reports
- Batch API bypasses all rate limits, processes 8K requests in ~4-5 minutes
- File upload via `client.files.upload()`, batch job via `client.batch.jobs.create()`
- Results downloaded as JSONL, parsed via `custom_id` matching to test set ACNs
- Mistral Large wraps JSON in markdown code fences -- stripped via regex before parsing

### DeepSeek V3.2 API Specifics

DeepSeek V3.2 experiments use the DeepInfra OpenAI-compatible API:
- Endpoint: `https://api.deepinfra.com/v1/openai/chat/completions`
- Model: `deepseek-ai/DeepSeek-V3.2`
- temperature=0.0, max_tokens=500 (non-thinking) / 4096 (thinking)
- 50 concurrent async requests via aiohttp
- Taxonomy-enriched system prompt (same as Mistral Large 3)
- Thinking mode: `reasoning_effort="high"` in request payload
- Reasoning tokens returned via `reasoning_content` field in `choices[0].message` (NOT `<think>` blocks)
- Content field has clean JSON -- no stripping needed
- Prefix caching: identical system prompts get ~62-82% cache hits across requests
- Checkpoint logic: results saved as JSON every 100 requests, resumable
- Script: `scripts/deepseek_v32_deepinfra.py` (`--reasoning` flag for thinking mode)

### DeepSeek V3.2 API Specifics

DeepSeek V3.2 experiments use the DeepInfra OpenAI-compatible API:
- Endpoint: `https://api.deepinfra.com/v1/openai/chat/completions`
- Model: `deepseek-ai/DeepSeek-V3.2`
- temperature=0.0, max_tokens=500 (non-thinking) / 4096 (thinking)
- 50 concurrent async requests via aiohttp
- Taxonomy-enriched system prompt (same as Mistral Large 3)
- Thinking mode: `reasoning_effort="high"` in request payload
- Reasoning tokens returned via `reasoning_content` field in `choices[0].message` (NOT `<think>` blocks)
- Content field has clean JSON -- no stripping needed
- Prefix caching: identical system prompts get ~62-82% cache hits across requests
- Checkpoint logic: results saved as JSON every 100 requests, resumable
- Script: `scripts/deepseek_v32_deepinfra.py` (`--reasoning` flag for thinking mode)


---

## Section 8: Model Configuration Details

### Model Family Overview

| Model | Type | Parameters | License | Quantization | Hosting |
|-------|------|-----------|---------|--------------|---------|
| XGBoost | Gradient boosted trees | N/A | Apache 2.0 | N/A | Local CPU |
| Qwen/Qwen3-8B | CausalLM (text-only) | 8B | Apache 2.0 | 4-bit NF4 (QLoRA) / FP16 (vLLM) | Modal (L4/A100) |
| mistral-large-latest | Chat model (MoE) | 675B MoE (41B active) | Apache 2.0 | N/A (API) | Mistral Batch API |
| Ministral-3-8B-Instruct-2512 | Multimodal (Mistral3ForConditionalGeneration) | 8B | Apache 2.0 | FP8 | Modal (L4/A100) |
| deepseek-ai/DeepSeek-V3.2 | MoE (non-reasoning / reasoning) | 671B MoE | Proprietary | N/A (API) | DeepInfra API |
| deepseek-ai/DeepSeek-V3.2 | MoE (non-reasoning / reasoning) | 671B MoE | Proprietary | N/A (API) | DeepInfra API |

### Classic ML (XGBoost) Configuration

```
TF-IDF Vectorizer:
  max_features: 50,000
  ngram_range: (1, 2)
  sublinear_tf: True
  dtype: float32

XGBoost Classifier (13 independent binary classifiers):
  n_estimators: 300
  max_depth: 6
  learning_rate: 0.1
  scale_pos_weight: auto (computed per category)
  tree_method: hist
  eval_metric: logloss
  use_label_encoder: False
```

### Qwen3-8B vLLM Inference Configuration

| Parameter | Zero-Shot Basic | Few-Shot Basic | Fine-Tuned | Taxonomy ZS/FS | Thinking |
|-----------|----------------|---------------|------------|----------------|----------|
| max_model_len | 8192 | 16384 | 4096 | 8192/16384 | 32768 |
| gpu_memory_utilization | 0.90 | 0.90 | 0.90 | 0.90 | 0.90 |
| dtype | auto | auto | auto | auto | auto |
| temperature | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| max_tokens | 256 | 256 | 256 | 256 | 4096 |
| batch_size | 64 | 16 | 64 | 64/16 | 32 |
| GPU | L4 | L4 | L4 | L4 | A100 |
| enable_thinking | False | False | False | False | True |
| enable_lora | -- | -- | True | -- | -- |
| max_lora_rank | -- | -- | 16 | -- | -- |
| narrative_truncation | 2000 chars | 1500 chars | 1500 chars | 2000/1500 chars | 1500 chars |

### Qwen3-8B QLoRA Fine-Tuning Configuration

```
Quantization (BitsAndBytesConfig):
  load_in_4bit: True
  bnb_4bit_quant_type: nf4
  bnb_4bit_compute_dtype: bfloat16
  bnb_4bit_use_double_quant: True

LoRA (LoraConfig):
  r: 16
  lora_alpha: 16
  target_modules: [q_proj, v_proj]
  lora_dropout: 0.05
  bias: none
  task_type: CAUSAL_LM

Training (SFTConfig via TRL):
  num_train_epochs: 2
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4
  effective_batch_size: 16
  learning_rate: 2e-5
  warmup_ratio: 0.05
  lr_scheduler_type: cosine
  optim: paged_adamw_8bit
  bf16: True
  max_length: 1024
  save_strategy: epoch

Training Results:
  total_steps: 3,982
  final_loss: 1.691
  token_accuracy: 66.8%
  training_time: 3h47m on A100
```

### Mistral Large 3 Batch API Configuration

```
Model: mistral-large-latest
Endpoint: /v1/chat/completions
Temperature: 0.0
Max Tokens: 256
Batch Method: File-based JSONL upload
Rate Limits: Bypassed (batch mode)
Cost: $0 (free tier, <4M tokens/month)
```

### Ministral 8B Configuration (Archived)

```
Model: mistralai/Ministral-3-8B-Instruct-2512
Architecture: Mistral3ForConditionalGeneration (multimodal wrapper)
Stored Quantization: FP8 (prevents proper 4-bit NF4 QLoRA)
Fine-tuning: LoRA on FP8 (not true QLoRA)
  - r=16, alpha=16, target=[q_proj, v_proj], dropout=0.05
  - Same training hyperparameters as Qwen3
Result: Fine-tuning produced no improvement (Macro-F1: 0.489 vs 0.491 zero-shot)
Reason for archival: FP8 multimodal wrapper prevented proper QLoRA; switched to Qwen3-8B
```

### DeepSeek V3.2 Configuration

```
Model: deepseek-ai/DeepSeek-V3.2
Architecture: Mixture of Experts (MoE), 671B total parameters
API: DeepInfra (OpenAI-compatible endpoint)
Temperature: 0.0
Max Tokens: 500 (non-thinking) / 4096 (thinking)
Concurrency: 50 async requests via aiohttp
Thinking Mode: reasoning_effort="high" in payload
Reasoning Output: reasoning_content field (separate from content)
Checkpoint: results_map JSON saved every 100 requests
```

### DeepInfra Pricing (DeepSeek V3.2)

| Token Type | Price per M tokens |
|------------|-------------------|
| Input (uncached) | $0.26 |
| Input (cached) | $0.13 |
| Output | $0.38 |

### DeepSeek V3.2 Configuration

```
Model: deepseek-ai/DeepSeek-V3.2
Architecture: Mixture of Experts (MoE), 671B total parameters
API: DeepInfra (OpenAI-compatible endpoint)
Temperature: 0.0
Max Tokens: 500 (non-thinking) / 4096 (thinking)
Concurrency: 50 async requests via aiohttp
Thinking Mode: reasoning_effort="high" in payload
Reasoning Output: reasoning_content field (separate from content)
Checkpoint: results_map JSON saved every 100 requests
```

### DeepInfra Pricing (DeepSeek V3.2)

| Token Type | Price per M tokens |
|------------|-------------------|
| Input (uncached) | $0.26 |
| Input (cached) | $0.13 |
| Output | $0.38 |


---

## Section 9: Compute & Cost Log

### Detailed Compute Table

| # | Experiment | GPU | Duration | Cost | Date |
|---|-----------|-----|----------|------|------|
| 1 | Classic ML (XGBoost, 32K) | CPU (local) | ~55 min | $0 | 2025-02-12 |
| 2 | Zero-shot LLM (Ministral) | L4 (Modal) | ~18.5 min | ~$0.25 | 2026-02-13 |
| 3 | Zero-shot LLM (Qwen3) | L4 (Modal) | ~26.4 min | ~$0.35 | 2026-02-13 |
| 4 | Few-shot LLM (Ministral) | L4 (Modal) | ~30.5 min | ~$0.41 | 2026-02-13 |
| 5 | Few-shot LLM (Qwen3) | L4 (Modal) | ~34.2 min | ~$0.46 | 2026-02-13 |
| 6 | Fine-tuned LLM training (Ministral) | A100 (Modal) | ~3h48min | ~$10.66 | 2026-02-13 |
| 7 | Fine-tuned LLM inference (Ministral) | L4 (Modal) | ~21.7 min | ~$0.29 | 2026-02-13 |
| 8 | QLoRA training (Qwen3) | A100 (Modal) | ~3h47min | ~$10.56 | 2026-02-13 |
| 9 | Fine-tuned LLM inference (Qwen3) | L4 (Modal) | ~20 min | ~$0.27 | 2026-02-14 |
| 10 | Few-shot LLM (Mistral Large 3) | API (Batch) | ~4 min | $0 (free tier) | 2026-02-14 |
| 11 | Classic ML full (164K XGBoost) | 32-core CPU (Modal) | ~30 min | ~$0.64 | 2026-02-14 |
| 12 | Zero-shot taxonomy (Qwen3) | L4 (Modal) | ~24.4 min | ~$0.33 | 2026-02-14 |
| 13 | Few-shot taxonomy (Qwen3) | L4 (Modal) | ~33.6 min | ~$0.45 | 2026-02-14 |
| 14 | Zero-shot LLM (Mistral Large 3) | API (Batch) | ~5 min | $0 (free tier) | 2026-02-14 |
| 15 | Few-shot taxonomy + thinking (Qwen3) | A100 (Modal) | ~144 min | ~$6.67 | 2026-02-14 |
| 16 | Classic ML subcategory (48 XGBoost) | 32-core CPU (Modal) | ~142 min | ~$3.03 | 2026-02-15 |
| 17 | Zero-shot subcategory (Mistral Large 3) | API (Real-time) | ~119 min | paid plan | 2026-02-16 |
| 18 | Zero-shot subcategory (Qwen3-8B) | L4 (Modal) | ~30 min | ~$0.40 | 2026-02-16 |
| 19 | Zero-shot parent (DeepSeek V3.2) | API (DeepInfra) | ~6.5 min | ~$1.39 | 2026-02-16 |
| 20 | Zero-shot subcategory (DeepSeek V3.2) | API (DeepInfra) | ~7.5 min | ~$1.92 | 2026-02-16 |
| 21 | Zero-shot + thinking parent (DeepSeek V3.2) | API (DeepInfra) | ~291 min | ~$6.73 | 2026-02-17 |
| 22 | Zero-shot + thinking subcategory (DeepSeek V3.2) | API (DeepInfra) | ~545 min | ~$5.24 | 2026-02-17 |
| 23 | Classic ML tuning Phase 3 | 32-core CPU (Modal) | ~154 min | ~$3.30 | 2026-02-17 |

### Cost Summary

| Category | Cost |
|----------|------|
| Ministral 8B (Modal) | ~$11.61 |
| Qwen3-8B (Modal) | ~$19.45 |
| Classic ML (Modal CPU) | ~$6.97 |
| Mistral Large 3 (API) | paid plan |
| DeepSeek V3.2 (DeepInfra) | ~$15.28 |
| Classic ML 32K (local) | $0 |
| **Total** | **~$54.30** |

### GPU Pricing Reference (Modal)

| GPU | Price/hr |
|-----|----------|
| NVIDIA L4 (24 GB) | $0.80 |
| NVIDIA A100 (80 GB) | $2.78 |
| 32-core CPU | ~$1.28 |

---

## Section 10: Generated Figures Inventory

| File | Description |
|------|-------------|
| `results/co_occurrence_heatmap.png` | 13x13 heatmap showing pairwise co-occurrence counts between all anomaly categories. Generated in Notebook 01 (data exploration). Shows that Deviation-Procedural co-occurs heavily with all other categories. |
| `results/classic_ml_f1_barchart.png` | Bar chart comparing per-category F1 scores for the Classic ML 32K baseline. Generated in Notebook 02. Shows Conflict and Aircraft Equipment Problem as the best-performing categories. |

| `results/classic_ml_subcategory_f1_barchart.png` | Bar chart comparing per-subcategory F1 scores for Classic ML subcategory baseline (48 labels). Generated in Notebook 05. |

| `results/classic_ml_subcategory_f1_barchart.png` | Bar chart comparing per-subcategory F1 scores for Classic ML subcategory baseline (48 labels). Generated in Notebook 05. |

Note: Additional plots (comparison charts, confusion matrices, etc.) may be generated during thesis writing but are not yet in the results directory. The thesis writer should create additional visualizations from the per-category metrics tables above.

---

## Section 11: Co-occurrence & Label Patterns

### Co-occurrence Matrix (13x13, raw counts from full 172K dataset)

|  | AircEquip | AirViol | ATC | Conflict | DevAlt | DevProc | DevSpd | DevTrk | FlightDk | GndEvt | GndExc | GndInc | Inflight |
|--|----------|---------|-----|----------|--------|---------|--------|--------|----------|--------|--------|--------|----------|
| AircEquip | 49305 | 648 | 1507 | 2282 | 5336 | 25857 | 1181 | 3957 | 6354 | 4750 | 1192 | 620 | 9726 |
| AirViol | 648 | 6834 | 1795 | 1337 | 361 | 6196 | 43 | 870 | 62 | 32 | 3 | 132 | 1014 |
| ATC | 1507 | 1795 | 29422 | 16890 | 3938 | 20935 | 618 | 3433 | 180 | 734 | 65 | 1920 | 5074 |
| Conflict | 2282 | 1337 | 16890 | 46285 | 5093 | 28288 | 440 | 3750 | 222 | 2469 | 306 | 4424 | 3895 |
| DevAlt | 5336 | 361 | 3938 | 5093 | 28369 | 24300 | 1227 | 2716 | 317 | 119 | 2 | 45 | 7365 |
| DevProc | 25857 | 6196 | 20935 | 28288 | 24300 | 112606 | 3863 | 18009 | 5875 | 7534 | 1328 | 11066 | 22122 |
| DevSpd | 1181 | 43 | 618 | 440 | 1227 | 3863 | 5000 | 443 | 229 | 193 | 68 | 24 | 2185 |
| DevTrk | 3957 | 870 | 3433 | 3750 | 2716 | 18009 | 443 | 20268 | 230 | 164 | 48 | 330 | 4485 |
| FlightDk | 6354 | 62 | 180 | 222 | 317 | 5875 | 229 | 230 | 12291 | 636 | 40 | 79 | 1325 |
| GndEvt | 4750 | 32 | 734 | 2469 | 119 | 7534 | 193 | 164 | 636 | 14234 | 2670 | 811 | 2219 |
| GndExc | 1192 | 3 | 65 | 306 | 2 | 1328 | 68 | 48 | 40 | 2670 | 3718 | 122 | 1038 |
| GndInc | 620 | 132 | 1920 | 4424 | 45 | 11066 | 24 | 330 | 79 | 811 | 122 | 12601 | 429 |
| Inflight | 9726 | 1014 | 5074 | 3895 | 7365 | 22122 | 2185 | 4485 | 1325 | 2219 | 1038 | 429 | 38658 |

### Key Co-occurrence Patterns

**Strongest co-occurrences (besides Deviation-Procedural):**
- ATC Issue + Conflict: 16,890 (57.4% of ATC Issue reports also have Conflict)
- Aircraft Equipment Problem + Inflight Event/Encounter: 9,726 (19.7% of AircEquip)
- Deviation-Altitude + Inflight Event/Encounter: 7,365 (25.9% of DevAlt)
- Flight Deck/Cabin Event + Aircraft Equipment Problem: 6,354 (51.7% of FlightDk)
- Ground Event/Encounter + Ground Excursion: 2,670 (71.8% of Ground Excursion)

**Near-zero co-occurrences (semantically exclusive):**
- Ground Excursion + Airspace Violation: 3
- Ground Excursion + Deviation-Altitude: 2
- Ground Incursion + Deviation-Speed: 24
- Ground Excursion + Ground Incursion: 122

### Top 10 Most Common Label Combinations (from full 172K dataset)

| Rank | Count | % | Categories |
|------|-------|---|------------|
| 1 | 10,448 | 6.1% | Aircraft Equipment Problem (single) |
| 2 | 10,218 | 5.9% | Aircraft Equipment Problem + Deviation - Procedural |
| 3 | 9,783 | 5.7% | Deviation - Altitude + Deviation - Procedural |
| 4 | 8,983 | 5.2% | Conflict (single) |
| 5 | 7,129 | 4.1% | Conflict + Deviation - Procedural |
| 6 | 6,478 | 3.8% | Deviation - Procedural + Deviation - Track/Heading |
| 7 | 6,367 | 3.7% | Inflight Event/Encounter (single) |
| 8 | 6,261 | 3.6% | ATC Issue + Conflict + Deviation - Procedural |
| 9 | 5,677 | 3.3% | Deviation - Procedural + Ground Incursion |
| 10 | 5,513 | 3.2% | Deviation - Procedural + Inflight Event/Encounter |

Deviation-Procedural appears in 8 of top 10 combinations (reflecting its 65.4% prevalence). The most common single-label reports are Aircraft Equipment Problem (6.1%) and Conflict (5.2%).

---

## Section 12: Error Analysis Hints

### F1 Cross-Comparison Across All 16 Parent Experiments

| Category | CML | CMLf | CMLt | Q-ZS | Q-FS | Q-FT | Q-ZSt | Q-FSt | Q-FStk | ML3-ZS | ML3-FS | Min-ZS | Min-FS | Min-FT | DS-ZS | DS-ZStk |
|----------|-----|------|------|------|------|------|-------|-------|--------|--------|--------|--------|--------|--------|-------|---------|
| Aircraft Equipment Problem | 0.816 | 0.827 | 0.818 | 0.718 | 0.532 | 0.783 | 0.600 | 0.424 | 0.673 | 0.816 | 0.754 | 0.773 | 0.692 | 0.766 | 0.770 | 0.820 |
| Airspace Violation | 0.568 | 0.517 | 0.586 | 0.297 | 0.450 | 0.120 | 0.335 | 0.459 | 0.487 | 0.516 | 0.499 | 0.318 | 0.527 | 0.320 | 0.460 | 0.586 |
| ATC Issue | 0.672 | 0.666 | 0.671 | 0.456 | 0.460 | 0.384 | 0.410 | 0.482 | 0.409 | 0.531 | 0.528 | 0.485 | 0.476 | 0.482 | 0.376 | 0.536 |
| Conflict | 0.801 | 0.810 | 0.800 | 0.689 | 0.730 | 0.724 | 0.690 | 0.727 | 0.744 | 0.734 | 0.739 | 0.753 | 0.720 | 0.752 | 0.777 | 0.837 |
| Deviation - Altitude | 0.729 | 0.736 | 0.729 | 0.672 | 0.655 | 0.689 | 0.551 | 0.651 | 0.622 | 0.759 | 0.750 | 0.657 | 0.648 | 0.651 | 0.752 | 0.775 |
| Deviation - Procedural | 0.795 | 0.795 | 0.793 | 0.353 | 0.200 | 0.749 | 0.770 | 0.507 | 0.507 | 0.793 | 0.767 | 0.607 | 0.395 | 0.600 | 0.768 | 0.757 |
| Deviation - Speed | 0.577 | 0.512 | 0.564 | 0.545 | 0.494 | 0.494 | 0.428 | 0.490 | 0.438 | 0.612 | 0.595 | 0.536 | 0.512 | 0.535 | 0.494 | 0.594 |
| Deviation - Track/Heading | 0.655 | 0.646 | 0.648 | 0.411 | 0.474 | 0.487 | 0.277 | 0.495 | 0.318 | 0.680 | 0.687 | 0.538 | 0.574 | 0.538 | 0.651 | 0.690 |
| Flight Deck/Cabin Event | 0.738 | 0.716 | 0.741 | 0.231 | 0.490 | 0.359 | 0.540 | 0.675 | 0.676 | 0.660 | 0.693 | 0.236 | 0.442 | 0.243 | 0.671 | 0.722 |
| Ground Event/Encounter | 0.592 | 0.574 | 0.593 | 0.312 | 0.103 | 0.518 | 0.405 | 0.430 | 0.471 | 0.546 | 0.520 | 0.396 | 0.445 | 0.397 | 0.515 | 0.549 |
| Ground Excursion | 0.572 | 0.566 | 0.609 | 0.403 | 0.295 | 0.385 | 0.404 | 0.348 | 0.546 | 0.610 | 0.552 | 0.350 | 0.480 | 0.334 | 0.622 | 0.635 |
| Ground Incursion | 0.729 | 0.698 | 0.720 | 0.415 | 0.463 | 0.490 | 0.519 | 0.580 | 0.508 | 0.668 | 0.660 | 0.335 | 0.615 | 0.346 | 0.649 | 0.687 |
| Inflight Event/Encounter | 0.734 | 0.747 | 0.734 | 0.464 | 0.536 | 0.446 | 0.559 | 0.567 | 0.524 | 0.628 | 0.574 | 0.395 | 0.491 | 0.397 | 0.588 | 0.661 |
| **MACRO** | **0.691** | **0.678** | **0.693** | **0.459** | **0.453** | **0.510** | **0.499** | **0.526** | **0.533** | **0.658** | **0.640** | **0.491** | **0.540** | **0.489** | **0.623** | **0.681** |
| **MICRO** | **0.746** | **0.739** | **0.745** | **0.473** | **0.468** | **0.632** | **0.605** | **0.544** | **0.556** | **0.712** | **0.686** | **0.543** | **0.536** | **0.542** | **0.693** | **0.723** |

Legend: CML=Classic ML 32K, CMLf=Classic ML 164K, CMLt=Classic ML Tuned, Q=Qwen3-8B, ML3=Mistral Large 3, Min=Ministral 8B, DS=DeepSeek V3.2, ZS=zero-shot, FS=few-shot, FT=fine-tuned, t=taxonomy, tk=thinking

### Hard vs Easy Categories (across all 16 parent experiments)

| Category | Min F1 | Max F1 | Range | Below 0.55 | Above 0.70 | Difficulty |
|----------|--------|--------|-------|-----------|-----------|------------|
| Conflict | 0.689 | 0.837 | 0.148 | 0/16 | 14/16 | EASY |
| Deviation - Altitude | 0.551 | 0.775 | 0.225 | 0/16 | 7/16 | EASY |
| Aircraft Equipment Problem | 0.424 | 0.827 | 0.403 | 2/16 | 11/16 | MODERATE |
| Deviation - Procedural | 0.200 | 0.795 | 0.595 | 5/16 | 9/16 | VARIABLE |
| Flight Deck/Cabin Event | 0.231 | 0.741 | 0.510 | 7/16 | 4/16 | HARD |
| Ground Incursion | 0.335 | 0.729 | 0.394 | 7/16 | 2/16 | HARD |
| Inflight Event/Encounter | 0.395 | 0.747 | 0.353 | 7/16 | 3/16 | HARD |
| Deviation - Track/Heading | 0.277 | 0.690 | 0.413 | 7/16 | 0/16 | HARD |
| Ground Excursion | 0.295 | 0.635 | 0.340 | 9/16 | 0/16 | VERY HARD |
| Deviation - Speed | 0.428 | 0.612 | 0.184 | 11/16 | 0/16 | VERY HARD |
| ATC Issue | 0.376 | 0.672 | 0.296 | 13/16 | 0/16 | VERY HARD |
| Ground Event/Encounter | 0.103 | 0.593 | 0.490 | 13/16 | 0/16 | VERY HARD |
| Airspace Violation | 0.120 | 0.586 | 0.466 | 13/16 | 0/16 | VERY HARD |

### Key Error Analysis Observations

**Consistently easy categories (F1 > 0.70 in most experiments):**
- **Conflict** (min F1 = 0.689, range = 0.148): Most consistent across all 16 models. Clear lexical signals (separation, TCAS, conflict, near-miss). Even zero-shot LLMs achieve F1 > 0.68. DeepSeek V3.2 + thinking achieves the highest F1 (0.837).
- **Aircraft Equipment Problem** (above 0.70 in 11/16): Strong lexical cues (malfunction, failure, MEL, maintenance). Only struggles with taxonomy few-shot Qwen3 (0.424) where high precision killed recall. DeepSeek V3.2 + thinking matches Classic ML at 0.820.

**Consistently hard categories (F1 < 0.55 in most experiments):**
- **Airspace Violation** (13/16 below 0.55): Rare (4.0% prevalence), short narratives, easily confused with ATC Issue and Deviation-Procedural. Fine-tuned Qwen3 collapsed to F1=0.120. Best: Classic ML Tuned and DeepSeek+thinking (both 0.586).
- **Ground Event/Encounter** (13/16 below 0.55): Semantically overlaps with Ground Excursion and Ground Incursion. Qwen3 few-shot basic collapsed to F1=0.103.
- **ATC Issue** (13/16 below 0.55): Frequently co-occurs with Conflict (57% overlap), making it hard to distinguish as a separate category. Classic ML (0.672) is best by a wide margin.
- **Deviation - Speed** (11/16 below 0.55): Very rare (2.9% prevalence), short narratives, often co-occurs with Deviation-Altitude. Best: Mistral Large ZS (0.612).

**Most variable category:**
- **Deviation - Procedural** (range = 0.595, min = 0.200, max = 0.795): The broadest category (65.4% prevalence) has wildly different F1 depending on model type. Classic ML and Mistral Large achieve F1 ~0.79, but Qwen3 few-shot basic collapses to F1 = 0.200. The taxonomy prompt dramatically helps (0.353 -> 0.770 zero-shot), suggesting small models need explicit category boundary definitions for broad categories.

**Key insights -- Model size vs prompting strategy:**
- Classic ML (XGBoost) dominates all LLM approaches on Macro-F1 (0.693 vs best LLM 0.681)
- DeepSeek V3.2 + thinking (671B MoE, 0.681) is the best LLM, within 0.012 of XGBoost, but costs $6.73 and takes ~5 hours
- Mistral Large 3 (675B MoE/41B active, 0.658) is the best cost-efficient LLM ($0, 5 min via free batch API)
- 8B models (Qwen3, Ministral) peak at ~0.54 Macro-F1 -- far below XGBoost
- Taxonomy-enriched prompts provide the largest gains for 8B models: Qwen3 ZS Micro-F1 jumps from 0.473 to 0.605 (+0.132)
- Fine-tuning helps Qwen3 on Micro-F1 (0.632) but not Macro-F1 (0.510), improving common categories at the expense of rare ones
- Thinking mode scales with model size: +0.058 F1 at 671B, +0.007 at 8B
- Thinking mode HURTS on 48-label subcategory task (-0.003 F1, 21.6% parse failures from token exhaustion)

### FINAL_EXPERIMENT_SUMMARY Tables

The following tables from `results/FINAL_EXPERIMENT_SUMMARY.md` provide cross-cutting analysis:

#### Table 3: TF-IDF Ablation (3-Fold CV, XGBoost Baseline Params)

| Config | max_features | ngram_range | sublinear_tf | min_df | CV Macro-F1 | CV Micro-F1 | Wall (s) |
|--------|-------------|-------------|--------------|--------|-------------|-------------|----------|
| no_sublinear | 50,000 | (1,2) | False | 1 | **0.6296** | **0.6929** | 1285 |
| baseline | 50,000 | (1,2) | True | 1 | 0.6280 | 0.6922 | 1245 |
| min_df_3 | 50,000 | (1,2) | True | 3 | 0.6279 | 0.6921 | 1298 |
| fewer_features | 20,000 | (1,2) | True | 1 | 0.6278 | 0.6918 | 967 |
| trigram | 50,000 | (1,3) | True | 1 | 0.6277 | 0.6910 | 1468 |
| trigram_100k | 100,000 | (1,3) | True | 1 | 0.6263 | 0.6901 | 1645 |
| more_features | 100,000 | (1,2) | True | 1 | 0.6261 | 0.6911 | 1389 |
| unigram_only | 50,000 | (1,1) | True | 1 | 0.6248 | 0.6900 | 525 |

**Conclusion:** Total range = 0.0048 Macro-F1. TF-IDF hyperparameters have negligible impact.

#### Table 4: Classic ML Model Comparison (RandomizedSearchCV, 3-Fold CV)

| Model | Best Params | CV Macro-F1 | Test Macro-F1 | Test Micro-F1 | RSCV Time |
|-------|------------|-------------|---------------|---------------|-----------|
| **XGBoost** | n_est=300, depth=6, lr=0.1, hist, scale_pos_weight | **0.6791** | **0.6906** | 0.7459 | 23 min |
| LogisticRegression | C=1.45, L2, SAGA, balanced | 0.5035 | 0.6701 | 0.7375 | 212 min |
| LinearSVC | C=0.113, squared_hinge, balanced | 0.4725 | 0.6550 | **0.7496** | 4 min |

#### Table 5: Prompt Engineering Impact (Qwen3-8B, Parent Task)

| Prompt Strategy | Approach | Macro-F1 | Micro-F1 | Delta Macro-F1 |
|-----------------|----------|----------|----------|----------------|
| basic | Zero-shot | 0.459 | 0.473 | -- |
| taxonomy | Zero-shot | 0.499 | 0.605 | +0.040 |
| basic | Few-shot | 0.453 | 0.468 | -- |
| taxonomy | Few-shot | 0.526 | 0.544 | +0.073 |
| taxonomy + thinking | Few-shot | 0.533 | 0.556 | +0.080 |

#### Table 6: Thinking Mode Impact Across Model Scales

| Model | Scale | Task | Without Thinking | With Thinking | Delta F1 | Cost Ratio |
|-------|-------|------|-----------------|---------------|----------|------------|
| Qwen3-8B | 8B | Parent (13) | 0.526 | 0.533 | +0.007 | 4.3x ($0.45 vs $6.67) |
| DeepSeek V3.2 | 671B | Parent (13) | 0.623 | 0.681 | +0.058 | 4.8x ($1.39 vs $6.73) |
| DeepSeek V3.2 | 671B | Subcat (48) | 0.422 | 0.419 | -0.003 | 2.7x ($1.92 vs $5.24) |

#### Table 7: Fine-Tuning Impact (8B Models)

| Model | Zero-Shot F1 | Fine-Tuned F1 | Delta | Training Cost | Training Time |
|-------|-------------|---------------|-------|---------------|---------------|
| Qwen3-8B (QLoRA 4-bit NF4) | 0.459 | 0.510 | +0.051 | ~$10.83 | 3h48m (A100) |
| Ministral 8B (LoRA on FP8) | 0.491 | 0.489 | -0.002 | ~$10.95 | 3h48m (A100) |

#### Table 8: Per-Category F1 Comparison (Top 5 Models, Parent Task)

| Category | XGBoost 32K | DeepSeek+Think | Mistral Large ZS | DeepSeek ZS | Qwen3 FS Tax |
|----------|-------------|----------------|-------------------|-------------|--------------|
| Aircraft Equipment Problem | 0.816 | 0.820 | 0.816 | 0.770 | 0.424 |
| Airspace Violation | 0.568 | 0.586 | 0.516 | 0.460 | 0.459 |
| ATC Issue | 0.672 | 0.536 | 0.531 | 0.376 | 0.482 |
| Conflict | 0.801 | 0.837 | 0.734 | 0.777 | 0.727 |
| Deviation - Altitude | 0.729 | 0.775 | 0.759 | 0.752 | 0.651 |
| Deviation - Procedural | 0.795 | 0.757 | 0.793 | 0.768 | 0.507 |
| Deviation - Speed | 0.577 | 0.594 | 0.612 | 0.494 | 0.490 |
| Deviation - Track/Heading | 0.655 | 0.690 | 0.680 | 0.651 | 0.495 |
| Flight Deck/Cabin Event | 0.738 | 0.722 | 0.660 | 0.671 | 0.675 |
| Ground Event/Encounter | 0.592 | 0.549 | 0.546 | 0.515 | 0.430 |
| Ground Excursion | 0.572 | 0.635 | 0.610 | 0.622 | 0.348 |
| Ground Incursion | 0.729 | 0.687 | 0.668 | 0.649 | 0.580 |
| Inflight Event/Encounter | 0.734 | 0.661 | 0.628 | 0.588 | 0.567 |
| **MACRO** | **0.691** | **0.681** | **0.658** | **0.623** | **0.526** |

XGBoost wins 7/13 categories. DeepSeek+Thinking wins 6/13 (Conflict, Altitude, Speed, Track/Heading, Airspace Violation, Ground Excursion).

### Key Findings (from FINAL_EXPERIMENT_SUMMARY.md)

1. **Classic ML (TF-IDF + XGBoost) is the overall winner** for this multi-label classification task, with Macro-F1 0.693 and the highest AUC (0.932). It requires no GPU, minimal tuning, and runs in minutes.

2. **LLMs close the gap but don't surpass Classic ML.** The best LLM (DeepSeek V3.2 671B + thinking, 0.681 F1) comes within 0.012 of XGBoost but costs $6.73 and takes 5 hours.

3. **Model scale matters more than technique.** MoE 670B+ models (0.62–0.68 Macro-F1) dramatically outperform dense 8B models (0.45–0.54). Both Mistral Large 3 (675B MoE/41B active) and DeepSeek V3.2 (671B MoE/37B active) are sparse MoE architectures. Within a size class, prompt engineering and fine-tuning provide only modest improvements. Mistral Large 3 outperforms DeepSeek V3.2 on zero-shot (0.658 vs 0.623), but DeepSeek with thinking mode (0.681) closes the gap to XGBoost.

4. **Taxonomy-enriched prompts are the most cost-effective LLM improvement** (+0.04-0.08 F1 for free, just better prompts). Thinking mode and fine-tuning are expensive with diminishing returns.

5. **Subcategory (48-label) classification is hard for everyone.** All models drop 0.18-0.46 F1 going from 13 to 48 labels. Classic ML maintains the lead with AUC 0.934 -- its ranking quality is preserved even when hard classification thresholds suffer.

6. **XGBoost hyperparameters are robust.** Extensive tuning (TF-IDF ablation + 3-model comparison + RandomizedSearchCV) confirmed the baseline config as near-optimal (delta < 0.005 F1).

*End of thesis_context.md*
