"""Few-shot classification of ASRS reports using Llama 3.1 8B-Instruct on Modal."""
import modal
import json

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
GPU = "L4"  # 24GB VRAM

app = modal.App("asrs-few-shot")

vllm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm", "torch", "transformers", "huggingface_hub")
)


def build_few_shot_examples(train_csv: str, categories: list[str], n_per_cat: int = 3):
    """Select n diverse examples per category from the training set.

    Strategy: for each category, pick examples where that category is the
    primary (or only) label, to give the model clear signal.
    Prefer shorter narratives to save context.
    """
    import pandas as pd
    df = pd.read_csv(train_csv)
    examples = []
    used_acns = set()

    for cat in categories:
        # Reports with this category = 1, sorted by narrative length
        cat_df = df[df[cat] == 1].copy()
        cat_df["_nlen"] = cat_df["Narrative"].str.len()
        cat_df = cat_df.sort_values("_nlen")
        cat_df = cat_df[~cat_df["ACN"].isin(used_acns)]

        selected = cat_df.head(n_per_cat)
        for _, row in selected.iterrows():
            labels = [c for c in categories if row[c] == 1]
            examples.append({
                "narrative": row["Narrative"][:800],  # truncate for context budget
                "labels": labels,
            })
            used_acns.add(row["ACN"])

    # NOTE: with 13 categories * 3 examples = 39 examples.
    # Need to monitor total prompt length vs max_model_len.
    return examples


@app.cls(
    image=vllm_image,
    gpu=GPU,
    timeout=3600,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
class FewShotClassifier:
    @modal.enter()
    def load_model(self):
        from vllm import LLM, SamplingParams
        self.llm = LLM(
            model=MODEL_ID,
            max_model_len=8192,  # longer context for few-shot examples
            dtype="half",
            trust_remote_code=True,
        )
        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=256,
        )

    def build_prompt(self, narrative: str, categories: list[str],
                     examples: list[dict]) -> str:
        """Build Llama 3.1 chat-format prompt with few-shot examples."""
        cat_list = "\n".join(f"- {c}" for c in categories)
        system_msg = (
            "You are an aviation safety analyst. Given an ASRS incident report, "
            "classify it into one or more of the following anomaly categories. "
            "Return ONLY a JSON list of matching category names.\n\n"
            f"Categories:\n{cat_list}"
        )

        # Build few-shot turns
        example_turns = ""
        for ex in examples:
            example_turns += (
                f"<|start_header_id|>user<|end_header_id|>\n\n"
                f"Classify this ASRS report:\n\n{ex["narrative"]}<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>\n\n"
                f"{json.dumps(ex["labels"])}<|eot_id|>"
            )

        narrative = narrative[:2000]
        prompt = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{system_msg}<|eot_id|>"
            f"{example_turns}"
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"Classify this ASRS report:\n\n{narrative}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        return prompt

    @modal.method()
    def classify_batch(self, narratives: list[str], categories: list[str],
                       examples: list[dict]) -> list[str]:
        prompts = [self.build_prompt(n, categories, examples) for n in narratives]
        outputs = self.llm.generate(prompts, self.sampling_params)
        return [o.outputs[0].text for o in outputs]


@app.local_entrypoint()
def main(
    test_csv: str = "data/test_set.csv",
    train_csv: str = "data/train_set.csv",
    batch_size: int = 32,
):
    import pandas as pd
    import time

    df = pd.read_csv(test_csv)
    categories = [c for c in df.columns if c not in ("ACN", "Narrative")]
    narratives = df["Narrative"].tolist()

    examples = build_few_shot_examples(train_csv, categories, n_per_cat=3)
    print(f"Built {len(examples)} few-shot examples")
    print(f"Classifying {len(narratives)} reports")

    classifier = FewShotClassifier()
    all_results = []
    t0 = time.time()

    for i in range(0, len(narratives), batch_size):
        batch = narratives[i:i + batch_size]
        results = classifier.classify_batch.remote(batch, categories, examples)
        all_results.extend(results)
        print(f"  Batch {i//batch_size + 1}: {len(all_results)}/{len(narratives)} done")

    elapsed = time.time() - t0
    print(f"Total inference time: {elapsed:.1f}s")

    df["llm_raw_output"] = all_results
    out_path = "results/few_shot_raw_outputs.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {out_path}")
