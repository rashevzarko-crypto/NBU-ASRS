"""QLoRA fine-tuning of Llama 3.1 8B for ASRS report classification on Modal."""
import modal
import json

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
GPU = "L4"  # 24GB VRAM; QLoRA: 4-bit base ~6GB + adapters fits in 24GB

app = modal.App("asrs-finetune")

train_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch", "transformers", "peft", "trl", "bitsandbytes",
        "datasets", "accelerate", "huggingface_hub", "pandas",
    )
)

vllm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm", "torch", "transformers", "huggingface_hub", "peft")
)

# Modal volume to persist adapter weights between training and inference
vol = modal.Volume.from_name("asrs-finetune-vol", create_if_missing=True)
VOLUME_PATH = "/vol"


def format_training_example(row: dict, categories: list[str]) -> str:
    """Format a single training example as instruction/input/output.

    Output format for SFTTrainer: a single text string with the full
    chat template applied.
    """
    active_cats = [c for c in categories if row.get(c, 0) == 1]
    cat_list = "\n".join(f"- {c}" for c in categories)

    system_msg = (
        "You are an aviation safety analyst. Given an ASRS incident report, "
        "classify it into one or more of the following anomaly categories. "
        "Return ONLY a JSON list of matching category names.\n\n"
        f"Categories:\n{cat_list}"
    )
    narrative = str(row["Narrative"])[:2000]
    answer = json.dumps(active_cats)

    # Llama 3.1 chat template
    text = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{system_msg}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"Classify this ASRS report:\n\n{narrative}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{answer}<|eot_id|>"
    )
    return text


@app.function(
    image=train_image,
    gpu=GPU,
    timeout=7200,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={VOLUME_PATH: vol},
)
def train(train_csv: str):
    """Fine-tune Llama 3.1 8B with QLoRA on ASRS training data."""
    import pandas as pd
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset
    import torch

    # TODO: Load training data
    df = pd.read_csv(train_csv)
    categories = [c for c in df.columns if c not in ("ACN", "Narrative")]
    print(f"Training on {len(df)} reports with {len(categories)} categories")

    # TODO: Format training examples
    texts = [format_training_example(row, categories) for _, row in df.iterrows()]
    dataset = Dataset.from_dict({"text": texts})

    # TODO: Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # TODO: Load model with 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    # TODO: Apply LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # TODO: Configure training
    training_args = SFTConfig(
        output_dir=f"{VOLUME_PATH}/checkpoints",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=50,
        save_strategy="epoch",
        max_seq_length=2048,
        dataset_text_field="text",
    )

    # TODO: Train
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )
    trainer.train()

    # TODO: Save adapter weights
    adapter_path = f"{VOLUME_PATH}/adapter"
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    vol.commit()
    print(f"Adapter saved to {adapter_path}")


@app.cls(
    image=vllm_image,
    gpu=GPU,
    timeout=3600,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={VOLUME_PATH: vol},
)
class FineTunedClassifier:
    """Inference with fine-tuned model (base + LoRA adapter)."""

    @modal.enter()
    def load_model(self):
        from vllm import LLM, SamplingParams
        # TODO: vLLM supports loading LoRA adapters natively
        # Need to check if adapter exists and load accordingly
        adapter_path = f"{VOLUME_PATH}/adapter"
        self.llm = LLM(
            model=MODEL_ID,
            max_model_len=4096,
            dtype="half",
            trust_remote_code=True,
            enable_lora=True,
            max_lora_rank=16,
        )
        from vllm.lora.request import LoRARequest
        self.lora_request = LoRARequest("asrs_adapter", 1, adapter_path)
        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=256,
        )

    def build_prompt(self, narrative: str, categories: list[str]) -> str:
        cat_list = "\n".join(f"- {c}" for c in categories)
        system_msg = (
            "You are an aviation safety analyst. Given an ASRS incident report, "
            "classify it into one or more of the following anomaly categories. "
            "Return ONLY a JSON list of matching category names.\n\n"
            f"Categories:\n{cat_list}"
        )
        narrative = narrative[:2000]
        prompt = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{system_msg}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"Classify this ASRS report:\n\n{narrative}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        return prompt

    @modal.method()
    def classify_batch(self, narratives: list[str], categories: list[str]) -> list[str]:
        prompts = [self.build_prompt(n, categories) for n in narratives]
        outputs = self.llm.generate(
            prompts, self.sampling_params, lora_request=self.lora_request
        )
        return [o.outputs[0].text for o in outputs]


@app.local_entrypoint()
def main(
    test_csv: str = "data/test_set.csv",
    train_csv: str = "data/train_set.csv",
    do_train: bool = True,
    batch_size: int = 64,
):
    import pandas as pd
    import time

    if do_train:
        print("Starting fine-tuning...")
        train.remote(train_csv)
        print("Fine-tuning complete.")

    # Inference
    df = pd.read_csv(test_csv)
    categories = [c for c in df.columns if c not in ("ACN", "Narrative")]
    narratives = df["Narrative"].tolist()
    print(f"Classifying {len(narratives)} reports")

    classifier = FineTunedClassifier()
    all_results = []
    t0 = time.time()

    for i in range(0, len(narratives), batch_size):
        batch = narratives[i:i + batch_size]
        results = classifier.classify_batch.remote(batch, categories)
        all_results.extend(results)
        print(f"  Batch {i//batch_size + 1}: {len(all_results)}/{len(narratives)} done")

    elapsed = time.time() - t0
    print(f"Total inference time: {elapsed:.1f}s")

    df["llm_raw_output"] = all_results
    out_path = "results/finetune_raw_outputs.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {out_path}")
