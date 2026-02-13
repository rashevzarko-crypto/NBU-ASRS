"""Zero-shot classification of ASRS reports using Llama 3.1 8B-Instruct on Modal."""
import modal
import json
import csv
import io

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
GPU = "L4"  # 24GB VRAM, ~6GB for 4-bit 8B model

app = modal.App("asrs-zero-shot")

vllm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm", "torch", "transformers", "huggingface_hub")
)


@app.cls(
    image=vllm_image,
    gpu=GPU,
    timeout=3600,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
class ZeroShotClassifier:
    @modal.enter()
    def load_model(self):
        from vllm import LLM, SamplingParams
        self.llm = LLM(
            model=MODEL_ID,
            max_model_len=4096,
            dtype="half",
            trust_remote_code=True,
        )
        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=256,
        )

    def build_prompt(self, narrative: str, categories: list[str]) -> str:
        """Build Llama 3.1 chat-format prompt for zero-shot classification."""
        cat_list = "\n".join(f"- {c}" for c in categories)
        system_msg = (
            "You are an aviation safety analyst. Given an ASRS incident report, "
            "classify it into one or more of the following anomaly categories. "
            "Return ONLY a JSON list of matching category names.\n\n"
            f"Categories:\n{cat_list}"
        )
        # Truncate narrative to 2000 chars
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
        """Classify a batch of narratives."""
        prompts = [self.build_prompt(n, categories) for n in narratives]
        outputs = self.llm.generate(prompts, self.sampling_params)
        return [o.outputs[0].text for o in outputs]


@app.local_entrypoint()
def main(test_csv: str = "data/test_set.csv", batch_size: int = 64):
    """Upload test data, run zero-shot classification, save results."""
    import pandas as pd
    import time

    df = pd.read_csv(test_csv)
    categories = [c for c in df.columns if c not in ("ACN", "Narrative")]
    narratives = df["Narrative"].tolist()
    print(f"Classifying {len(narratives)} reports with {len(categories)} categories")

    classifier = ZeroShotClassifier()
    all_results = []
    t0 = time.time()

    for i in range(0, len(narratives), batch_size):
        batch = narratives[i:i + batch_size]
        results = classifier.classify_batch.remote(batch, categories)
        all_results.extend(results)
        print(f"  Batch {i//batch_size + 1}: {len(all_results)}/{len(narratives)} done")

    elapsed = time.time() - t0
    print(f"Total inference time: {elapsed:.1f}s")

    # Save raw LLM outputs
    df["llm_raw_output"] = all_results
    out_path = "results/zero_shot_raw_outputs.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {out_path}")
