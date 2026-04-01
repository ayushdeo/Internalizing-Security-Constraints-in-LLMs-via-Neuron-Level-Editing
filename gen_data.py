#!/usr/bin/env python3
"""
Generate code outputs for a prompt dataset using small code models on a low-VRAM GPU.

Models:
- Qwen/Qwen2.5-Coder-1.5B-Instruct
- bigcode/starcoder2-3b

Input:
- data/prompts/all_prompts.json

Outputs:
- data/outputs/qwen_qwen2_5_coder_1_5b_instruct.jsonl
- data/outputs/bigcode_starcoder2_3b.jsonl

Requirements:
    pip install --upgrade torch transformers accelerate bitsandbytes sentencepiece
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


# ----------------------------
# Data structures
# ----------------------------

@dataclass
class PromptItem:
    id: int
    bucket: str
    category: str
    prompt: str
    expected_risk: str
    source: str


# ----------------------------
# Helpers
# ----------------------------

def show_gpu_info() -> None:
    if not torch.cuda.is_available():
        print("CUDA is not available.")
        return

    props = torch.cuda.get_device_properties(0)
    total_gb = props.total_memory / (1024 ** 3)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {total_gb:.2f} GB")
    print(f"CUDA: {torch.version.cuda}")
    print("-" * 70)


def free_mem() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def load_prompts(path: Path) -> List[PromptItem]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    prompts: List[PromptItem] = []
    for row in data:
        prompts.append(
            PromptItem(
                id=int(row["id"]),
                bucket=str(row["bucket"]),
                category=str(row["category"]),
                prompt=str(row["prompt"]),
                expected_risk=str(row.get("expected_risk", "")),
                source=str(row.get("source", "")),
            )
        )
    return prompts


def safe_model_slug(model_name: str) -> str:
    return model_name.replace("/", "_").replace("-", "_")


def make_quant_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )


def build_chat_prompt(model_name: str, user_prompt: str, tokenizer) -> str:
    """
    Use a chat template for instruct models when supported.
    """
    if "Qwen" in model_name and hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": "You are a helpful coding assistant. Return only code unless asked otherwise."},
            {"role": "user", "content": user_prompt},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # StarCoder2 works fine with plain prompt text.
    return user_prompt


def extract_code(text: str) -> str:
    """
    Try to extract code from model output:
    - remove leading/trailing whitespace
    - if markdown fenced code exists, return first fenced block
    - otherwise return raw text
    """
    text = text.strip()

    # Remove common chat markers if present
    text = re.sub(r"^(system|user|assistant)\s*", "", text, flags=re.IGNORECASE)

    # Extract first fenced code block if present
    fence = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        return fence.group(1).strip()

    return text


def load_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=make_quant_config(),
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    return tokenizer, model


@torch.inference_mode()
def generate_one(model, tokenizer, prompt: str, max_new_tokens: int = 128) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    input_len = inputs["input_ids"].shape[1]

    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        repetition_penalty=1.05,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    # Decode only newly generated tokens, not the prompt.
    new_tokens = out[0][input_len:]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return extract_code(raw)


def save_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_model_on_prompts(
    model_name: str,
    prompts: List[PromptItem],
    out_dir: Path,
    max_new_tokens: int,
) -> Path:
    print("=" * 70)
    print(f"Loading: {model_name}")
    free_mem()

    t0 = time.time()
    tokenizer, model = load_model(model_name)
    load_time = time.time() - t0
    print(f"Loaded in {load_time:.2f} sec")

    results: List[Dict[str, Any]] = []

    try:
        for i, item in enumerate(prompts, start=1):
            prompt_text = build_chat_prompt(model_name, item.prompt, tokenizer)

            try:
                t1 = time.time()
                output = generate_one(model, tokenizer, prompt_text, max_new_tokens=max_new_tokens)
                gen_time = time.time() - t1

                results.append(
                    {
                        "id": item.id,
                        "bucket": item.bucket,
                        "category": item.category,
                        "prompt": item.prompt,
                        "model": model_name,
                        "output": output,
                        "expected_risk": item.expected_risk,
                        "source": item.source,
                        "gen_time_sec": round(gen_time, 4),
                        "success": True,
                    }
                )

                if i % 25 == 0 or i == len(prompts):
                    print(f"  completed {i}/{len(prompts)}")

            except Exception as e:
                results.append(
                    {
                        "id": item.id,
                        "bucket": item.bucket,
                        "category": item.category,
                        "prompt": item.prompt,
                        "model": model_name,
                        "output": "",
                        "expected_risk": item.expected_risk,
                        "source": item.source,
                        "gen_time_sec": None,
                        "success": False,
                        "error": str(e),
                    }
                )
                print(f"  failed id={item.id}: {e}")

    finally:
        del model
        del tokenizer
        free_mem()

    out_file = out_dir / f"{safe_model_slug(model_name)}.jsonl"
    save_jsonl(out_file, results)

    print(f"Saved: {out_file}")
    print("=" * 70)
    return out_file


def main():
    parser = argparse.ArgumentParser(description="Generate model outputs for a prompt dataset.")
    parser.add_argument(
        "--prompts",
        type=Path,
        default=Path("data/prompts/all_prompts.json"),
        help="Path to the prompt JSON file.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("data/outputs"),
        help="Directory to write model outputs.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum new tokens to generate per prompt.",
    )
    args = parser.parse_args()

    show_gpu_info()
    prompts = load_prompts(args.prompts)
    print(f"Loaded {len(prompts)} prompts from {args.prompts}")

    models = [
        "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "bigcode/starcoder2-3b",
    ]

    for model_name in models:
        run_model_on_prompts(
            model_name=model_name,
            prompts=prompts,
            out_dir=args.outdir,
            max_new_tokens=args.max_new_tokens,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()