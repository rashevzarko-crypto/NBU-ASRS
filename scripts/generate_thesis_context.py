#!/usr/bin/env python3
"""Generate thesis_context.md with all 22 experiments from source data."""
import csv, re, os

R = "results"

def read(p):
    with open(p, encoding="utf-8") as f:
        return f.read()

def csv_table(p):
    hdr = ["| Category | Precision | Recall | F1 | ROC-AUC |",
           "|----------|-----------|--------|------|---------|"]
    with open(p, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            c = r["Category"]
            v = [f"{float(r[k]):.4f}" for k in ("Precision","Recall","F1","ROC-AUC")]
            if c in ("MACRO","MICRO"):
                hdr.append(f"| **{c}** | **{v[0]}** | **{v[1]}** | **{v[2]}** | **{v[3]}** |")
            else:
                hdr.append(f"| {c} | {v[0]} | {v[1]} | {v[2]} | {v[3]} |")
    return "\n".join(hdr)

def get_sections(text):
    d = {}
    ms = [(m.start(), int(m.group(1))) for m in re.finditer(r'^## Section (\d+):', text, re.MULTILINE)]
    for i,(s,n) in enumerate(ms):
        e = ms[i+1][0] if i+1 < len(ms) else len(text)
        c = text[s:e].rstrip()
        while c.endswith("---"): c = c[:-3].rstrip()
        d[n] = c
    return d

EXPS = [
    (1, "Classic ML 32K (TF-IDF + XGBoost, 31,850 train)", f"{R}/classic_ml_text_metrics.csv",
     "Config: TF-IDF max_features=50000, ngram_range=(1,2), sublinear_tf=True. XGBoost n_estimators=300, max_depth=6, lr=0.1, scale_pos_weight=auto, tree_method=hist. 13 independent binary classifiers."),
    (2, "Classic ML 164K (TF-IDF + XGBoost, 164,139 train)", f"{R}/classic_ml_full_metrics.csv",
     "Config: Same as 32K but trained on 164,139 reports (full dataset minus test set). Higher recall across all categories, but lower precision due to class imbalance amplification."),
    (3, "Classic ML Tuned (TF-IDF + XGBoost, 31,850 train, Phase 3)", f"{R}/classic_ml_tuned_parent_metrics.csv",
     "Config: Same TF-IDF and XGBoost params as 32K baseline. Retrained via modal_classic_ml_phase3.py with per-label scale_pos_weight. Delta vs baseline: Macro-F1 +0.002, Micro-F1 -0.001. Confirms baseline is near-optimal."),
    (4, "Qwen3-8B Zero-Shot (basic prompt)", f"{R}/zero_shot_metrics.csv",
     "Config: Qwen3-8B, vLLM, max_model_len=8192, temperature=0.0, max_tokens=256, enable_thinking=False. Basic system prompt."),
    (5, "Qwen3-8B Few-Shot (basic prompt, 3 examples/category)", f"{R}/few_shot_metrics.csv",
     "Config: Same model, max_model_len=16384, batch_size=16. 39 examples (3 per category), narrative truncated to 600 chars (examples) / 1500 chars (test)."),
    (6, "Qwen3-8B Fine-Tuned (QLoRA 4-bit NF4)", f"{R}/finetune_metrics.csv",
     "Config: QLoRA 4-bit NF4, r=16, alpha=16, target=[q_proj, v_proj], dropout=0.05. Training: 2 epochs, batch=4, grad_accum=4, lr=2e-5, cosine scheduler, paged_adamw_8bit, bf16. 3,982 steps, final loss 1.691, token accuracy 66.8%. Training time: 3h47m on A100."),
    (7, "Qwen3-8B Zero-Shot Taxonomy", f"{R}/zero_shot_taxonomy_metrics.csv",
     "Config: Same as basic zero-shot but with taxonomy-enriched system prompt (NASA ASRS subcategories + discriminative hints)."),
    (8, "Qwen3-8B Few-Shot Taxonomy", f"{R}/few_shot_taxonomy_metrics.csv",
     "Config: Same as basic few-shot but with taxonomy-enriched system prompt. 39 examples (3 per category)."),
    (9, "Qwen3-8B Few-Shot Taxonomy + Thinking Mode", f"{R}/few_shot_taxonomy_thinking_metrics.csv",
     "Config: enable_thinking=True, max_tokens=4096, max_model_len=32768, A100 GPU. 99.6% outputs had <think> blocks, avg 2,986 chars thinking per response. Runtime: 144 min on A100 (~$6.67). Marginal gain over non-thinking: Macro-F1 +0.007, Micro-F1 +0.013."),
    (10, "Mistral Large 3 Zero-Shot (taxonomy prompt, Batch API)", f"{R}/mistral_large_zs_metrics.csv",
     "Config: mistral-large-latest, Mistral Batch API, temperature=0.0, max_tokens=256. Parse failures: 2/8,044 (0.0%). Runtime: ~5 min. Cost: $0 (free tier)."),
    (11, "Mistral Large 3 Few-Shot (taxonomy prompt, Batch API)", f"{R}/mistral_large_metrics.csv",
     "Config: mistral-large-latest, Mistral Batch API, temperature=0.0, max_tokens=256. 26 examples (2 per category). Runtime: ~4 min. Cost: $0 (free tier)."),
    (12, "Ministral 8B Zero-Shot (basic prompt, FP8)", f"{R}/ministral/zero_shot_metrics.csv",
     "Config: Ministral-3-8B-Instruct-2512 (FP8 multimodal wrapper), vLLM on Modal L4."),
    (13, "Ministral 8B Few-Shot (basic prompt, FP8)", f"{R}/ministral/few_shot_metrics.csv",
     "Config: Same model, 39 examples (3 per category), batch_size=16."),
    (14, "Ministral 8B Fine-Tuned (LoRA on FP8)", f"{R}/ministral/finetune_metrics.csv",
     "Note: Ministral fine-tuning was LoRA on FP8 (not true QLoRA on 4-bit NF4). This produced essentially no improvement over zero-shot. The model became a \"yes-machine\" with high recall but very low precision."),
    (15, "DeepSeek V3.2 Zero-Shot (taxonomy prompt, DeepInfra API)", f"{R}/deepseek_v32_parent_metrics.csv",
     "Config: deepseek-ai/DeepSeek-V3.2 (671B MoE), DeepInfra API, temperature=0.0, max_tokens=500. 50 concurrent requests via aiohttp. Prefix caching: ~62%. Parse failures: 3/8044 (0.0%). Runtime: ~6.5 min. Cost: ~$1.39."),
    (16, "DeepSeek V3.2 Zero-Shot + Thinking (taxonomy prompt, DeepInfra API)", f"{R}/deepseek_v32_thinking_parent_metrics.csv",
     "Config: Same as non-thinking but reasoning=True, max_tokens=4096. Reasoning via reasoning_content field (clean JSON in content). Prefix caching: ~63%. Parse failures: 46/8044 empty (0.6%). Runtime: ~291 min. Cost: ~$6.73. Thinking adds +0.058 Macro-F1 at 671B scale."),
    (17, "Classic ML Subcategory (TF-IDF + XGBoost, 48 labels, 32,089 train)", f"{R}/classic_ml_subcategory_metrics.csv",
     "Config: Same TF-IDF + XGBoost params as parent baseline. 48 independent binary classifiers. Runtime: 142 min on Modal 32-core CPU (~$3.03)."),
    (18, "Classic ML Tuned Subcategory (TF-IDF + XGBoost, 48 labels, 32,089 train)", f"{R}/classic_ml_tuned_subcategory_metrics.csv",
     "Config: Same params as baseline, retrained via Phase 3. Delta vs subcategory baseline: Macro-F1 -0.0001, Micro-F1 +0.0003. Near-identical."),
    (19, "Mistral Large 3 Zero-Shot Subcategory (48 labels, taxonomy prompt)", f"{R}/mistral_large_subcategory_metrics.csv",
     "Config: mistral-large-latest, real-time API (batch API was stuck). 48-label taxonomy-enriched prompt. Runtime: ~119 min, 7 network errors, 0.1% parse failures. Cost: paid plan."),
    (20, "DeepSeek V3.2 Zero-Shot Subcategory (48 labels, taxonomy prompt)", f"{R}/deepseek_v32_subcategory_metrics.csv",
     "Config: deepseek-ai/DeepSeek-V3.2, DeepInfra API, temperature=0.0, max_tokens=500. Prefix caching: ~79%. Parse failures: 3/8017 (0.0%). Runtime: ~7.5 min. Cost: ~$1.92."),
    (21, "DeepSeek V3.2 Zero-Shot + Thinking Subcategory (48 labels, taxonomy prompt)", f"{R}/deepseek_v32_thinking_subcategory_metrics.csv",
     "Config: Same as non-thinking but reasoning=True, max_tokens=4096. 21.6% parse failures (1729/8017 empty) -- reasoning tokens exhaust max_tokens. Runtime: ~545 min. Cost: ~$5.24. Thinking HURTS subcategory: -0.003 Macro-F1."),
    (22, "Qwen3-8B Zero-Shot Subcategory (48 labels, taxonomy prompt)", f"{R}/qwen_zero_shot_subcategory_metrics.csv",
     "Config: Qwen3-8B, vLLM on Modal L4, taxonomy-enriched prompt. Runtime: ~30 min (~$0.40). Dramatic drop from parent: Macro-F1 0.499->0.235, Micro-F1 0.605->0.304."),
]

SECTION_9 = """## Section 9: Compute & Cost Log

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
| 32-core CPU | ~$1.28 |"""

SECTION_12 = r"""## Section 12: Error Analysis Hints

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

6. **XGBoost hyperparameters are robust.** Extensive tuning (TF-IDF ablation + 3-model comparison + RandomizedSearchCV) confirmed the baseline config as near-optimal (delta < 0.005 F1)."""

DEEPSEEK_S7 = """

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
"""

DEEPSEEK_S8 = """

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
"""

def main():
    existing = read("thesis_context.md")
    sec = get_sections(existing)
    out = []

    # Header
    out.append("# Thesis Context: NBU-ASRS Multi-Label Aviation Safety Report Classification\n\n"
               "> **Purpose:** This file is self-contained. It provides ALL data, results, prompts, configs, "
               "and sample reports needed to write the thesis. No other files are needed.\n"
               ">\n> **Last generated:** 2026-02-17\n\n---\n\n")

    # Section 1: STATUS.md
    out.append("## Section 1: STATUS.md (Verbatim Copy)\n\n")
    out.append(read("STATUS.md"))
    out.append("\n\n---\n\n")

    # Sections 2-5: unchanged
    for n in (2, 3, 4, 5):
        out.append(sec[n])
        out.append("\n\n---\n\n")

    # Section 6: All 22 experiments
    out.append("## Section 6: All Experimental Results (Per-Category Tables)\n\n")
    for num, title, csv_path, config in EXPS:
        out.append(f"### Experiment {num}: {title}\n\n")
        out.append(csv_table(csv_path))
        out.append(f"\n\n{config}\n\n")
    out.append("---\n\n")

    # Section 7: Prompt templates + DeepSeek
    out.append(sec[7])
    out.append(DEEPSEEK_S7)
    out.append("\n\n---\n\n")

    # Section 8: Model configs + DeepSeek
    s8 = sec[8]
    s8 = s8.replace(
        "| Ministral-3-8B-Instruct-2512 | Multimodal (Mistral3ForConditionalGeneration) | 8B | Apache 2.0 | FP8 | Modal (L4/A100) |",
        "| Ministral-3-8B-Instruct-2512 | Multimodal (Mistral3ForConditionalGeneration) | 8B | Apache 2.0 | FP8 | Modal (L4/A100) |\n"
        "| deepseek-ai/DeepSeek-V3.2 | MoE (non-reasoning / reasoning) | 671B MoE | Proprietary | N/A (API) | DeepInfra API |"
    )
    out.append(s8)
    out.append(DEEPSEEK_S8)
    out.append("\n\n---\n\n")

    # Section 9: Complete rewrite
    out.append(SECTION_9)
    out.append("\n\n---\n\n")

    # Section 10: Figures + subcategory barchart
    s10 = sec[10]
    s10 = s10.replace(
        "Note: Additional plots",
        "| `results/classic_ml_subcategory_f1_barchart.png` | Bar chart comparing per-subcategory F1 scores for Classic ML subcategory baseline (48 labels). Generated in Notebook 05. |\n\n"
        "Note: Additional plots"
    )
    out.append(s10)
    out.append("\n\n---\n\n")

    # Section 11: unchanged
    out.append(sec[11])
    out.append("\n\n---\n\n")

    # Section 12: Complete rewrite
    out.append(SECTION_12)
    out.append("\n\n*End of thesis_context.md*\n")

    result = "".join(out)
    with open("thesis_context.md", "w", encoding="utf-8") as f:
        f.write(result)

    # Verification
    lines = result.splitlines()
    print(f"Written {len(result):,} chars, {len(lines):,} lines")
    assert "2026-02-17" in result[:500], "Missing updated date in header!"
    # Note: 2026-02-14 legitimately appears in STATUS.md compute log as historical run dates
    header_line = lines[0]
    assert "2026-02-14" not in header_line, "Stale 2026-02-14 date in header line!"
    assert "DeepSeek" in result, "Missing DeepSeek references!"
    for i in range(1, 23):
        assert f"Experiment {i}:" in result, f"Missing Experiment {i}!"
    assert "~$54.30" in result, "Missing updated cost total!"
    print("All 22 experiments verified. All checks passed.")

if __name__ == "__main__":
    main()
