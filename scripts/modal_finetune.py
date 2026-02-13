"""QLoRA fine-tuning of Qwen3-8B for ASRS report classification on Modal.

Phase 1: Fine-tune base model with QLoRA on 31,850 training examples (A100).
Phase 2: Run inference on 8,044 test set with vLLM + LoRA adapter (L4).

Parses LLM JSON outputs into binary labels, computes per-category and aggregate
metrics, and saves results in the canonical format matching classic_ml_text_metrics.csv.
"""

import modal
import json
import re
import os
import time
import io

MODEL_ID = "Qwen/Qwen3-8B"
BATCH_SIZE = 64
CHECKPOINT_EVERY = 10

app = modal.App("asrs-finetune")

train_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch", "transformers", "peft", "trl",
        "datasets", "accelerate", "huggingface_hub", "pandas", "bitsandbytes",
    )
)

vllm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm", "torch", "transformers", "huggingface_hub", "peft")
)

vol = modal.Volume.from_name("asrs-finetune-vol", create_if_missing=True)
VOLUME_PATH = "/vol"
ADAPTER_PATH = f"{VOLUME_PATH}/finetune-adapter"
MERGED_PATH = f"{VOLUME_PATH}/merged-model"


def _build_system_message(categories: list[str]) -> str:
    """Build the exact system message used across all experiments."""
    cat_list = "\n".join(f"- {c}" for c in categories)
    return (
        "You are an aviation safety analyst classifying ASRS incident reports. "
        "For each report, identify ALL applicable anomaly categories from the "
        "list below. A report can belong to multiple categories. "
        "Return ONLY a JSON array of matching category names, nothing else.\n\n"
        f"Categories:\n{cat_list}"
    )


# ---------------------------------------------------------------------------
# Phase 1: Training (A100)
# ---------------------------------------------------------------------------

@app.function(
    image=train_image,
    gpu="A100",
    timeout=14400,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={VOLUME_PATH: vol},
)
def train(train_bytes: bytes, categories: list[str]):
    """Fine-tune Qwen3-8B with QLoRA on ASRS training data."""
    import pandas as pd
    import torch
    from transformers import AutoTokenizer
    from peft import LoraConfig, get_peft_model
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset

    # Check if adapter already exists (resume logic)
    vol.reload()
    if os.path.exists(f"{ADAPTER_PATH}/adapter_config.json"):
        print(f"Adapter already exists at {ADAPTER_PATH}, skipping training.")
        return

    # Load training data from bytes
    df = pd.read_csv(io.BytesIO(train_bytes))
    print(f"Training on {len(df)} reports with {len(categories)} categories")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Build training examples
    system_msg = _build_system_message(categories)

    def format_example(row):
        active_cats = [c for c in categories if row.get(c, 0) == 1]
        narrative = str(row["Narrative"])[:1500]
        msgs = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": f"Classify this ASRS report:\n\n{narrative}"},
            {"role": "assistant", "content": json.dumps(active_cats)},
        ]
        return tokenizer.apply_chat_template(msgs, tokenize=False, enable_thinking=False)

    print("Formatting training examples...")
    texts = [format_example(row) for _, row in df.iterrows()]
    dataset = Dataset.from_dict({"text": texts})
    print(f"Dataset ready: {len(dataset)} examples")

    # Sequence length stats
    tokenized_lens = [len(tokenizer.encode(t)) for t in texts[:500]]
    import numpy as np
    print(f"Token length stats (first 500): "
          f"mean={np.mean(tokenized_lens):.0f}, "
          f"median={np.median(tokenized_lens):.0f}, "
          f"p95={np.percentile(tokenized_lens, 95):.0f}, "
          f"max={np.max(tokenized_lens)}")

    # Load model with 4-bit NF4 quantization (proper QLoRA).
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    print("Loading base model with 4-bit NF4 quantization (QLoRA)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    print(f"Model type: {type(model).__name__}")
    model.config.use_cache = False  # Required for gradient checkpointing
    model.enable_input_require_grads()  # Required for LoRA training

    # Apply QLoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Training config
    training_args = SFTConfig(
        output_dir=f"{VOLUME_PATH}/finetune-checkpoints",
        num_train_epochs=2,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        bf16=True,
        logging_steps=50,
        save_strategy="epoch",
        max_length=1024,
    )

    # Train
    print("Starting training...")
    t0 = time.time()
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    train_time = time.time() - t0
    print(f"Training complete in {train_time:.0f}s ({train_time/60:.1f} min)")

    # Save adapter
    os.makedirs(ADAPTER_PATH, exist_ok=True)
    model.save_pretrained(ADAPTER_PATH)
    tokenizer.save_pretrained(ADAPTER_PATH)
    vol.commit()
    print(f"Adapter saved to {ADAPTER_PATH}")


# ---------------------------------------------------------------------------
# Optional: Merge adapter into base model (fallback if vLLM LoRA fails)
# ---------------------------------------------------------------------------

@app.function(
    image=train_image,
    gpu="A100",
    timeout=3600,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={VOLUME_PATH: vol},
)
def merge_adapter():
    """Merge LoRA adapter into base model for direct vLLM loading."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    vol.reload()
    if os.path.exists(f"{MERGED_PATH}/config.json"):
        print(f"Merged model already exists at {MERGED_PATH}, skipping.")
        return

    print("Loading base model in fp16 for merging...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    print(f"Loading adapter from {ADAPTER_PATH}...")
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)

    print("Merging and unloading...")
    model = model.merge_and_unload()

    os.makedirs(MERGED_PATH, exist_ok=True)
    model.save_pretrained(MERGED_PATH)
    tokenizer.save_pretrained(MERGED_PATH)
    vol.commit()
    print(f"Merged model saved to {MERGED_PATH}")


# ---------------------------------------------------------------------------
# Phase 2: Inference (L4) — Option A: vLLM with LoRA
# ---------------------------------------------------------------------------

@app.cls(
    image=vllm_image,
    gpu="L4",
    timeout=3600,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={VOLUME_PATH: vol},
)
class FineTunedClassifier:
    """Inference with fine-tuned model (base + LoRA adapter via vLLM)."""

    @modal.enter()
    def load_model(self):
        from vllm import LLM, SamplingParams

        vol.reload()

        # Option A: vLLM with LoRA adapter
        self.use_lora = True
        try:
            self.llm = LLM(
                model=MODEL_ID,
                max_model_len=4096,
                dtype="auto",
                gpu_memory_utilization=0.90,
                enable_lora=True,
                max_lora_rank=16,
            )
            from vllm.lora.request import LoRARequest
            self.lora_request = LoRARequest("asrs_adapter", 1, ADAPTER_PATH)
            print("Loaded model with LoRA adapter (Option A)")
        except Exception as e:
            print(f"LoRA loading failed: {e}")
            print("Falling back to merged model (Option B)")
            self.use_lora = False
            self.llm = LLM(
                model=MERGED_PATH,
                max_model_len=4096,
                dtype="auto",
                gpu_memory_utilization=0.90,
            )
            self.lora_request = None
            print("Loaded merged model (Option B)")

        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=256,
        )

    def build_messages(self, narrative: str, categories: list[str]) -> list[dict]:
        system_msg = _build_system_message(categories)
        narrative = narrative[:1500]
        return [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": f"Classify this ASRS report:\n\n{narrative}"},
        ]

    @modal.method()
    def classify_batch(self, narratives: list[str], categories: list[str]) -> list[str]:
        conversations = [self.build_messages(n, categories) for n in narratives]
        if self.use_lora:
            outputs = self.llm.chat(
                conversations, self.sampling_params,
                lora_request=self.lora_request,
                chat_template_kwargs={"enable_thinking": False},
            )
        else:
            outputs = self.llm.chat(
                conversations, self.sampling_params,
                chat_template_kwargs={"enable_thinking": False},
            )
        return [o.outputs[0].text for o in outputs]


# ---------------------------------------------------------------------------
# Lightweight probe: check if adapter exists on volume
# ---------------------------------------------------------------------------

@app.function(
    image=modal.Image.debian_slim(python_version="3.11"),
    volumes={VOLUME_PATH: vol},
)
def check_adapter_exists() -> bool:
    vol.reload()
    return os.path.exists(f"{ADAPTER_PATH}/adapter_config.json")


# ---------------------------------------------------------------------------
# Local helpers (run on the caller's machine, not on Modal)
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
      1. Direct JSON parse
      2. Regex extraction of JSON array from surrounding text
      3. Fuzzy substring matching of category names
    """
    cat_lower = {c.lower(): c for c in categories}

    # Tier 1: direct JSON parse
    try:
        parsed = json.loads(raw.strip())
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


def compute_metrics(y_true, y_pred, categories):
    """Compute per-category and aggregate metrics matching classic ML format."""
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
    import numpy as np

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
        "Fine-Tuned LLM (QLoRA): Qwen3-8B",
        "=" * 55,
        f"Test set: {n_test} reports | Model: {MODEL_ID}",
        "QLoRA: 4-bit NF4, r=16, alpha=16, target=[q_proj, v_proj], dropout=0.05",
        "Training: 2 epochs, batch=4, grad_accum=4, lr=2e-5, cosine, bf16, paged_adamw_8bit",
        "Inference: vLLM, dtype=auto, max_model_len=4096, temperature=0.0",
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


def print_comparison(finetune_rows):
    """Print four-way F1 comparison."""
    import pandas as pd

    classic_f1, zero_f1, few_f1 = {}, {}, {}

    classic_path = "results/classic_ml_text_metrics.csv"
    zero_path = "results/zero_shot_metrics.csv"
    few_path = "results/few_shot_metrics.csv"

    if os.path.exists(classic_path):
        c = pd.read_csv(classic_path)
        classic_f1 = dict(zip(c["Category"], c["F1"]))
    if os.path.exists(zero_path):
        z = pd.read_csv(zero_path)
        zero_f1 = dict(zip(z["Category"], z["F1"]))
    if os.path.exists(few_path):
        f = pd.read_csv(few_path)
        few_f1 = dict(zip(f["Category"], f["F1"]))

    print("\n" + "=" * 100)
    print(f"{'Category':<35} {'Classic ML':>11} {'Zero-Shot':>11} {'Few-Shot':>11} {'Fine-Tuned':>11} {'Delta(ML)':>10}")
    print("-" * 100)
    for row in finetune_rows:
        cat = row["Category"]
        ft = row["F1"]
        cm = classic_f1.get(cat, float("nan"))
        zs = zero_f1.get(cat, float("nan"))
        fs = few_f1.get(cat, float("nan"))
        delta = ft - cm if cm == cm else float("nan")
        print(f"{cat:<35} {cm:>11.4f} {zs:>11.4f} {fs:>11.4f} {ft:>11.4f} {delta:>+10.4f}")
    print("=" * 100)


# ---------------------------------------------------------------------------
# Main entrypoint (runs locally, calls Modal for GPU work)
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    test_csv: str = "data/test_set.csv",
    train_csv: str = "data/train_set.csv",
    batch_size: int = BATCH_SIZE,
):
    """Run QLoRA fine-tuning + inference: training and inference on Modal, metrics locally."""
    import pandas as pd
    import numpy as np

    # --- Load data ---
    df = pd.read_csv(test_csv)
    categories = [c for c in df.columns if c not in ("ACN", "Narrative")]
    narratives = df["Narrative"].tolist()
    acns = df["ACN"].tolist()
    n_test = len(df)
    print(f"Loaded {n_test} test reports, {len(categories)} categories")

    # --- Phase 1: Training ---
    adapter_exists = check_adapter_exists.remote()
    if adapter_exists:
        print("Adapter already exists on volume, skipping training.")
    else:
        print("Starting QLoRA fine-tuning on A100...")
        train_df = pd.read_csv(train_csv)
        train_bytes = train_df.to_csv(index=False).encode("utf-8")
        print(f"Sending {len(train_bytes) / 1024 / 1024:.1f} MB of training data to Modal")

        t_train_start = time.time()
        train.remote(train_bytes, categories)
        t_train_elapsed = time.time() - t_train_start
        print(f"Training complete in {t_train_elapsed:.0f}s ({t_train_elapsed/60:.1f} min)")
        print(f"Estimated training cost: ${t_train_elapsed / 3600 * 2.78:.2f} (A100 @ $2.78/hr)")

    # --- Phase 2: Inference ---
    checkpoint_path = "results/finetune_checkpoint.csv"
    completed = {}
    if os.path.exists(checkpoint_path):
        cp = pd.read_csv(checkpoint_path)
        completed = dict(zip(cp["ACN"].astype(str), cp["llm_raw_output"]))
        print(f"Resuming from checkpoint: {len(completed)} reports already done")

    classifier = FineTunedClassifier()
    all_raw = []
    skipped = 0
    t0 = time.time()
    batch_count = 0

    for i in range(0, len(narratives), batch_size):
        batch_acns = acns[i : i + batch_size]
        batch_narratives = narratives[i : i + batch_size]

        to_run_idx = []
        for j, acn in enumerate(batch_acns):
            if str(acn) not in completed:
                to_run_idx.append(j)

        if not to_run_idx:
            batch_results = [completed[str(a)] for a in batch_acns]
            all_raw.extend(batch_results)
            skipped += len(batch_acns)
        elif len(to_run_idx) == len(batch_acns):
            results = classifier.classify_batch.remote(batch_narratives, categories)
            all_raw.extend(results)
            for j, acn in enumerate(batch_acns):
                completed[str(acn)] = results[j]
        else:
            sub_narratives = [batch_narratives[j] for j in to_run_idx]
            results = classifier.classify_batch.remote(sub_narratives, categories)
            res_iter = iter(results)
            for j, acn in enumerate(batch_acns):
                if j in to_run_idx:
                    r = next(res_iter)
                    completed[str(acn)] = r
                    all_raw.append(r)
                else:
                    all_raw.append(completed[str(acn)])

        batch_count += 1
        done = len(all_raw)
        print(f"  Batch {batch_count}: {done}/{n_test} done")

        if batch_count % CHECKPOINT_EVERY == 0:
            cp_df = pd.DataFrame({
                "ACN": list(completed.keys()),
                "llm_raw_output": list(completed.values()),
            })
            cp_df.to_csv(checkpoint_path, index=False)
            print(f"  [checkpoint saved: {len(completed)} reports]")

    elapsed = time.time() - t0
    if skipped:
        print(f"Skipped {skipped} reports from checkpoint")
    print(f"Inference wall-clock time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Estimated inference cost: ${elapsed / 3600 * 0.80:.2f} (L4 @ $0.80/hr)")

    # --- Parse outputs ---
    print("\nParsing LLM outputs...")
    parsed_labels = []
    parse_failures = 0
    tier_counts = {"json": 0, "regex": 0, "fuzzy": 0, "empty": 0}

    for raw in all_raw:
        result = None

        # Tier 1: direct JSON
        try:
            p = json.loads(raw.strip())
            if isinstance(p, list):
                cat_lower = {c.lower(): c for c in categories}
                result = _normalize(p, cat_lower)
                if result:
                    tier_counts["json"] += 1
        except (json.JSONDecodeError, TypeError):
            pass

        # Tier 2: regex extract
        if result is None:
            m = re.search(r"\[.*?\]", raw, re.DOTALL)
            if m:
                try:
                    p = json.loads(m.group())
                    if isinstance(p, list):
                        cat_lower = {c.lower(): c for c in categories}
                        result = _normalize(p, cat_lower)
                        if result:
                            tier_counts["regex"] += 1
                except (json.JSONDecodeError, TypeError):
                    pass

        # Tier 3: fuzzy substring
        if result is None:
            raw_lower = raw.lower()
            matched = [c for c in categories if c.lower() in raw_lower]
            if matched:
                result = matched
                tier_counts["fuzzy"] += 1

        # Fallback
        if result is None:
            result = []
            tier_counts["empty"] += 1
            parse_failures += 1

        parsed_labels.append(result)

    fail_rate = parse_failures / n_test * 100
    print(f"Parse results: {tier_counts}")
    print(f"Parse failures (empty output): {parse_failures}/{n_test} ({fail_rate:.1f}%)")
    if fail_rate > 10:
        print("WARNING: Parse failure rate >10% — investigate prompt format")

    # --- Build prediction matrix ---
    y_pred = np.zeros((n_test, len(categories)), dtype=int)
    for i, labels in enumerate(parsed_labels):
        for cat in labels:
            if cat in categories:
                y_pred[i, categories.index(cat)] = 1

    y_true = df[categories].values

    # --- Compute metrics ---
    print("\nComputing metrics...")
    metrics_rows = compute_metrics(y_true, y_pred, categories)

    # --- Save metrics CSV ---
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = "results/finetune_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved {metrics_path}")

    # --- Save raw outputs CSV ---
    raw_df = df.copy()
    raw_df["llm_raw_output"] = all_raw
    raw_df["parsed_labels"] = [json.dumps(l) for l in parsed_labels]
    for cat in categories:
        raw_df[f"pred_{cat}"] = y_pred[:, categories.index(cat)]
    raw_path = "results/finetune_raw_outputs.csv"
    raw_df.to_csv(raw_path, index=False)
    print(f"Saved {raw_path}")

    # --- Save summary ---
    summary = format_summary(metrics_rows, n_test)
    summary_path = "results/finetune_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"Saved {summary_path}")

    # --- Print results ---
    print(f"\n{summary}")

    # --- Print four-way comparison ---
    print_comparison(metrics_rows)

    # --- Cleanup checkpoint ---
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print("\nCheckpoint file removed (run complete)")
