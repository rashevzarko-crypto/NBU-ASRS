# NBU-ASRS Project Status

Last updated: 2026-02-14 (all four Qwen3-8B experiments complete)

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

## Infrastructure

- **Local (VS Code + Jupyter):** Data processing, classic ML, visualization
- **Modal (cloud GPU):** All LLM experiments (L4 @ ~$0.80/hr)
- **Budget:** $50 total Modal, estimated ~$27 for all experiments
- **GitHub:** github.com/rashevzarko-crypto/NBU-ASRS

## Results

### Ministral 3 8B (archived — see `results/ministral/`)

| Model | Macro-F1 | Micro-F1 | Macro-AUC | Notes |
|-------|----------|----------|-----------|-------|
| Classic ML (TF-IDF + XGBoost) | 0.691 | 0.746 | 0.932 | 13 binary classifiers, text only |
| Zero-shot LLM (Ministral) | 0.491 | 0.543 | 0.744 | FP8, zero-shot, 0% parse failures |
| Few-shot LLM (Ministral) | 0.540 | 0.536 | 0.746 | FP8, 3 examples/cat, 0% parse failures |
| Fine-tuned LLM (Ministral) | 0.489 | 0.542 | 0.744 | LoRA on FP8 (not true QLoRA), 0% parse failures |

### Qwen3-8B (complete)

| Model | Macro-F1 | Micro-F1 | Macro-AUC | Notes |
|-------|----------|----------|-----------|-------|
| Classic ML (TF-IDF + XGBoost) | 0.691 | 0.746 | 0.932 | Same baseline |
| Zero-shot LLM (Qwen3) | 0.459 | 0.473 | 0.727 | 26.4 min on L4, ~$0.35, 0% parse failures |
| Few-shot LLM (Qwen3) | 0.453 | 0.468 | 0.704 | 34.2 min on L4, ~$0.46, 0% parse failures |
| Fine-tuned LLM (Qwen3) | 0.510 | 0.632 | 0.700 | QLoRA 4-bit NF4, 3h47m train A100 + ~20min inference L4 |

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

## Per-Category Results (Classic ML)

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

**Total Modal spend:** ~$23.21 (Ministral: ~$11.61 + Qwen3: ~$11.60)
