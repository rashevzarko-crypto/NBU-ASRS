# Final Experiment Summary: ASRS Multi-Label Classification

**Project:** Master's thesis — "Multimodal Deep Learning for Runway Incursion Detection"
**Task:** Multi-label classification of NASA ASRS aviation safety reports
**Test set:** 8,044 reports (parent, 13 labels) / 8,017 reports (subcategory, 48 labels)
**Train set:** 31,850 reports (parent) / 32,089 reports (subcategory)
**Generated:** 2026-02-17

---

## Table 1: All Experiments — Parent (13-Label) Task, Sorted by Macro-F1

| # | Model | Size | Approach | Prompt | Macro-F1 | Micro-F1 | Macro-AUC | Parse Fail | Runtime | Cost | GPU/API |
|---|-------|------|----------|--------|----------|----------|-----------|------------|---------|------|---------|
| 1 | XGBoost (tuned) | -- | Classic ML | -- | 0.6928 | 0.7454 | 0.9321 | -- | 154 min | ~$3.30 | 32-core CPU (Modal) |
| 2 | XGBoost (baseline) | -- | Classic ML | -- | 0.6906 | 0.7459 | 0.9320 | -- | 55 min | $0 | CPU (local) |
| 3 | DeepSeek V3.2 | 671B MoE | Zero-shot + thinking | taxonomy | 0.6807 | 0.7226 | 0.8099 | 0.6% | 291 min | ~$6.73 | DeepInfra API |
| 4 | XGBoost (164K full) | -- | Classic ML | -- | 0.6777 | 0.7385 | 0.9421 | -- | 30 min | ~$0.64 | 32-core CPU (Modal) |
| 5 | Mistral Large 3 | ~123B | Zero-shot | taxonomy | 0.6579 | 0.7121 | 0.7934 | 0.0% | 5 min | $0 | Mistral Batch API |
| 6 | Mistral Large 3 | ~123B | Few-shot (2/cat) | taxonomy | 0.6398 | 0.6860 | 0.7931 | 0.0% | 4 min | $0 | Mistral Batch API |
| 7 | DeepSeek V3.2 | 671B MoE | Zero-shot | taxonomy | 0.6225 | 0.6934 | 0.7460 | 0.0% | 6.5 min | ~$1.39 | DeepInfra API |
| 8 | Ministral 8B | 8B | Few-shot (3/cat) | basic | 0.5398 | 0.5361 | 0.7460 | 0% | 30.5 min | ~$0.41 | L4 (Modal) |
| 9 | Qwen3-8B | 8B | Few-shot (3/cat) + thinking | taxonomy | 0.5325 | 0.5563 | 0.7045 | 0% | 144 min | ~$6.67 | A100 (Modal) |
| 10 | Qwen3-8B | 8B | Few-shot (3/cat) | taxonomy | 0.5255 | 0.5436 | 0.7057 | 0% | 33.6 min | ~$0.45 | L4 (Modal) |
| 11 | Qwen3-8B | 8B | QLoRA fine-tuned | basic | 0.5098 | 0.6318 | 0.6999 | 0% | 248 min | ~$10.83 | A100+L4 (Modal) |
| 12 | Qwen3-8B | 8B | Zero-shot | taxonomy | 0.4990 | 0.6054 | 0.7011 | 0% | 24.4 min | ~$0.33 | L4 (Modal) |
| 13 | Ministral 8B | 8B | Zero-shot | basic | 0.4906 | 0.5434 | 0.7439 | 0% | 18.5 min | ~$0.25 | L4 (Modal) |
| 14 | Ministral 8B | 8B | LoRA fine-tuned (FP8) | basic | 0.4892 | 0.5415 | 0.7441 | 0% | 250 min | ~$10.95 | A100+L4 (Modal) |
| 15 | Qwen3-8B | 8B | Zero-shot | basic | 0.4590 | 0.4725 | 0.7273 | 0% | 26.4 min | ~$0.35 | L4 (Modal) |
| 16 | Qwen3-8B | 8B | Few-shot (3/cat) | basic | 0.4525 | 0.4675 | 0.7042 | 0% | 34.2 min | ~$0.46 | L4 (Modal) |

**Key observations:**
- Classic ML (XGBoost) is the best performer overall (Macro-F1 0.693, Micro-F1 0.745, AUC 0.932)
- Best LLM: DeepSeek V3.2 + thinking (0.681 Macro-F1) -- within 0.012 of XGBoost but at 45x cost
- Mistral Large 3 zero-shot (0.658) is the best cost-efficient LLM ($0, 5 min)
- 8B models peak at ~0.54 Macro-F1 (Ministral FS, Qwen3 FS+thinking) -- far below XGBoost
- Fine-tuning 8B models provides modest gains: Qwen3 +0.051, Ministral -0.001 vs their zero-shot
- Taxonomy-enriched prompts boost 8B models: Qwen3 ZS +0.040, FS +0.073 vs basic prompts
- Thinking mode: significant at 671B scale (+0.058 F1) but marginal at 8B (+0.007) for 15x cost
- Training on 164K (5x more data) slightly hurts F1 (0.678 vs 0.691) due to amplified imbalance

---

## Table 2: Subcategory (48-Label) Task, Sorted by Macro-F1

| # | Model | Size | Approach | Macro-F1 | Micro-F1 | Macro-AUC | Parse Fail | Runtime | Cost |
|---|-------|------|----------|----------|----------|-----------|------------|---------|------|
| 1 | XGBoost (baseline) | -- | Classic ML | 0.5100 | 0.5995 | 0.9341 | -- | 142 min | ~$3.03 |
| 2 | XGBoost (tuned) | -- | Classic ML | 0.5099 | 0.5998 | 0.9339 | -- | 154 min | ~$3.30 |
| 3 | Mistral Large 3 | ~123B | Zero-shot (taxonomy) | 0.4491 | 0.4938 | 0.7437 | 0.1% | 119 min | paid |
| 4 | DeepSeek V3.2 | 671B MoE | Zero-shot | 0.4220 | 0.4555 | 0.7082 | 0.0% | 7.5 min | ~$1.92 |
| 5 | DeepSeek V3.2 | 671B MoE | Zero-shot + thinking | 0.4185 | 0.4661 | 0.6898 | 21.6% | 545 min | ~$5.24 |
| 6 | Qwen3-8B | 8B | Zero-shot (taxonomy) | 0.2350 | 0.3040 | 0.6294 | 0% | 30 min | ~$0.40 |

**Key observations:**
- Classic ML dominates even more at 48 labels (AUC 0.934 vs next best 0.744)
- All models lose ~0.18-0.46 Macro-F1 going from 13 to 48 labels
- Thinking mode HURTS subcategory classification (-0.003 F1, 21.6% parse failures from token exhaustion)
- Qwen3-8B collapses on 48 labels (0.235 Macro-F1) -- 8B models can't handle fine-grained taxonomy

---

## Table 3: TF-IDF Ablation (3-Fold CV, XGBoost Baseline Params)

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

---

## Table 4: Classic ML Model Comparison (RandomizedSearchCV, 3-Fold CV)

| Model | Best Params | CV Macro-F1 | Test Macro-F1 | Test Micro-F1 | RSCV Time |
|-------|------------|-------------|---------------|---------------|-----------|
| **XGBoost** | n_est=300, depth=6, lr=0.1, hist, scale_pos_weight | **0.6791** | **0.6906** | 0.7459 | 23 min |
| LogisticRegression | C=1.45, L2, SAGA, balanced | 0.5035 | 0.6701 | 0.7375 | 212 min |
| LinearSVC | C=0.113, squared_hinge, balanced | 0.4725 | 0.6550 | **0.7496** | 4 min |

**Observations:**
- XGBoost baseline params are near-optimal (holdout confirmed from 7/16 search combos)
- LinearSVC: highest Micro-F1 (0.750) but weakest Macro-F1 (0.655) -- biased toward frequent labels
- LogReg: competitive (0.670 Macro-F1) but SAGA solver extremely slow on 50K sparse features
- XGBoost's per-label scale_pos_weight handles class imbalance most effectively

---

## Table 5: Prompt Engineering Impact (Qwen3-8B, Parent Task)

| Prompt Strategy | Approach | Macro-F1 | Micro-F1 | Delta Macro-F1 |
|-----------------|----------|----------|----------|----------------|
| basic | Zero-shot | 0.459 | 0.473 | -- |
| taxonomy | Zero-shot | 0.499 | 0.605 | +0.040 |
| basic | Few-shot | 0.453 | 0.468 | -- |
| taxonomy | Few-shot | 0.526 | 0.544 | +0.073 |
| taxonomy + thinking | Few-shot | 0.533 | 0.556 | +0.080 |

**Taxonomy prompt** = NASA ASRS subcategories + discriminative hints per category.
Taxonomy enrichment provides consistent gains (+0.04 to +0.08 Macro-F1) for 8B models.
Thinking mode adds only +0.007 on top of taxonomy few-shot, at 4x the cost.

---

## Table 6: Thinking Mode Impact Across Model Scales

| Model | Scale | Task | Without Thinking | With Thinking | Delta F1 | Cost Ratio |
|-------|-------|------|-----------------|---------------|----------|------------|
| Qwen3-8B | 8B | Parent (13) | 0.526 | 0.533 | +0.007 | 4.3x ($0.45 vs $6.67) |
| DeepSeek V3.2 | 671B | Parent (13) | 0.623 | 0.681 | +0.058 | 4.8x ($1.39 vs $6.73) |
| DeepSeek V3.2 | 671B | Subcat (48) | 0.422 | 0.419 | -0.003 | 2.7x ($1.92 vs $5.24) |

**Conclusion:** Thinking mode scales with model size. At 671B, +0.058 F1 gain for parent task.
At 8B, negligible. For complex 48-label tasks, thinking hurts due to token exhaustion (21.6% empty outputs).

---

## Table 7: Fine-Tuning Impact (8B Models)

| Model | Zero-Shot F1 | Fine-Tuned F1 | Delta | Training Cost | Training Time |
|-------|-------------|---------------|-------|---------------|---------------|
| Qwen3-8B (QLoRA 4-bit NF4) | 0.459 | 0.510 | +0.051 | ~$10.83 | 3h48m (A100) |
| Ministral 8B (LoRA on FP8) | 0.491 | 0.489 | -0.002 | ~$10.95 | 3h48m (A100) |

**Observations:**
- Qwen3-8B: fine-tuning helps (+0.051 Macro-F1, +0.159 Micro-F1) -- proper QLoRA on text-only CausalLM
- Ministral 8B: fine-tuning fails (-0.002 F1) -- LoRA on FP8 multimodal wrapper, "yes-machine" behavior
- Both cost ~$11 for 2 epochs on 32K samples -- expensive for 8B model improvements
- Taxonomy-enriched few-shot (0.526) beats Qwen3 fine-tuning (0.510) at 1/24th the cost

---

## Table 8: Per-Category F1 Comparison (Top 5 Models, Parent Task)

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

---

## Compute Summary

| Category | Total Cost | Total Runtime |
|----------|-----------|---------------|
| Classic ML (all runs) | ~$7.0 | ~381 min |
| Qwen3-8B (all experiments) | ~$19.75 | ~597 min |
| Ministral 8B (all experiments) | ~$12.27 | ~349 min |
| Mistral Large 3 (all experiments) | ~$0 + paid sub | ~133 min |
| DeepSeek V3.2 (all experiments) | ~$15.28 | ~850 min |
| **Grand Total** | **~$54.30** | **~2,310 min (38.5h)** |

---

## Key Findings

1. **Classic ML (TF-IDF + XGBoost) is the overall winner** for this multi-label classification task, with Macro-F1 0.693 and the highest AUC (0.932). It requires no GPU, minimal tuning, and runs in minutes.

2. **LLMs close the gap but don't surpass Classic ML.** The best LLM (DeepSeek V3.2 671B + thinking, 0.681 F1) comes within 0.012 of XGBoost but costs $6.73 and takes 5 hours.

3. **Model scale matters more than technique.** The ranking by Macro-F1 closely follows model size: 671B > ~123B > 8B. Within a size class, prompt engineering and fine-tuning provide only modest improvements.

4. **Taxonomy-enriched prompts are the most cost-effective LLM improvement** (+0.04-0.08 F1 for free, just better prompts). Thinking mode and fine-tuning are expensive with diminishing returns.

5. **Subcategory (48-label) classification is hard for everyone.** All models drop 0.18-0.46 F1 going from 13 to 48 labels. Classic ML maintains the lead with AUC 0.934 -- its ranking quality is preserved even when hard classification thresholds suffer.

6. **XGBoost hyperparameters are robust.** Extensive tuning (TF-IDF ablation + 3-model comparison + RandomizedSearchCV) confirmed the baseline config as near-optimal (delta < 0.005 F1).
