# NBU-ASRS Project Status

Last updated: 2025-02-13

## Progress Tracker

| Phase | Status | Output Files |
|-------|--------|-------------|
| Data exploration | ✅ Complete | `data/asrs_multilabel.csv`, `results/data_exploration_summary.txt`, `results/co_occurrence_heatmap.png` |
| Stratified sampling | ✅ Complete | `data/train_set.csv` (31,850), `data/test_set.csv` (8,044) |
| Classic ML baseline | ✅ Complete | `results/classic_ml_text_metrics.csv`, `results/classic_ml_f1_barchart.png` |
| Structured features extraction | ✅ Complete | `data/structured_features.csv` (39,894 rows) |
| Modal scripts scaffolding | ✅ Complete | `scripts/modal_zero_shot.py`, `scripts/modal_few_shot.py`, `scripts/modal_finetune.py` |
| Zero-shot LLM | ❌ Not started | |
| Few-shot LLM | ❌ Not started | |
| QLoRA fine-tuning | ❌ Not started | |
| Fine-tuned LLM inference | ❌ Not started | |
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
- Model: meta-llama/Llama-3.1-8B-Instruct (4-bit quantized via bitsandbytes)
- GPU: Modal L4 (24GB)
- Prompt: system role + category list + narrative → JSON list output
- Inference only on 8,044 test set

### Few-shot LLM
- Same model and GPU as zero-shot
- 3 examples per category in prompt (selected from train set)
- max_model_len ~8192 for longer context

### Fine-tuned LLM (QLoRA)
- Base: meta-llama/Llama-3.1-8B-Instruct
- QLoRA: r=16, alpha=16, target_modules=[q_proj, v_proj], dropout=0.05
- Training: 31,850 samples, 3 epochs, batch_size=4, lr=2e-5
- GPU: Modal L4 (24GB)
- Inference on 8,044 test set, same GPU

## Infrastructure

- **Local (VS Code + Jupyter):** Data processing, classic ML, visualization
- **Modal (cloud GPU):** All LLM experiments (L4 @ ~$0.80/hr)
- **Budget:** $50 total Modal, estimated ~$27 for all experiments
- **GitHub:** github.com/rashevzarko-crypto/NBU-ASRS

## Results

| Model | Macro-F1 | Micro-F1 | Macro-AUC | Notes |
|-------|----------|----------|-----------|-------|
| Classic ML (TF-IDF + XGBoost) | 0.691 | 0.746 | 0.932 | 13 binary classifiers, text only |
| Zero-shot LLM | — | — | — | |
| Few-shot LLM | — | — | — | |
| Fine-tuned LLM (QLoRA) | — | — | — | |

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
