#!/usr/bin/env python3
"""
Test smaller code models on a low-VRAM GPU (e.g. RTX 3050 4GB) using 4-bit quantization.

Models tested:
- Qwen/Qwen2.5-Coder-1.5B-Instruct
- bigcode/starcoder2-3b

Requirements:
    pip install --upgrade torch transformers accelerate bitsandbytes sentencepiece
"""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


@dataclass
class Result:
    model_name: str
    success: bool
    load_time_sec: float
    gen_time_sec: float
    output: str
    error: Optional[str] = None


def show_gpu_info() -> None:
    if not torch.cuda.is_available():
        print("CUDA is not available. This script expects a GPU.")
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


def make_quant_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )


def build_prompt(model_name: str, user_prompt: str, tokenizer) -> str:
    # Qwen instruct models work best with a chat template.
    if "Qwen" in model_name and hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": "You are a helpful coding assistant."},
            {"role": "user", "content": user_prompt},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # StarCoder2 works fine with a plain prompt.
    return user_prompt


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
def generate(tokenizer, model, prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    out = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        repetition_penalty=1.05,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    return tokenizer.decode(out[0], skip_special_tokens=True)


def test_one(model_name: str, user_prompt: str) -> Result:
    print(f"\nLoading model: {model_name}")
    free_mem()

    t0 = time.time()
    try:
        tokenizer, model = load_model(model_name)
    except Exception as e:
        return Result(
            model_name=model_name,
            success=False,
            load_time_sec=time.time() - t0,
            gen_time_sec=0.0,
            output="",
            error=f"Load failed: {e}",
        )

    load_time = time.time() - t0
    print(f"Loaded in {load_time:.2f} sec")

    try:
        prompt = build_prompt(model_name, user_prompt, tokenizer)
        t1 = time.time()
        output = generate(tokenizer, model, prompt)
        gen_time = time.time() - t1
    except Exception as e:
        return Result(
            model_name=model_name,
            success=False,
            load_time_sec=load_time,
            gen_time_sec=0.0,
            output="",
            error=f"Generation failed: {e}",
        )
    finally:
        del model
        del tokenizer
        free_mem()

    return Result(
        model_name=model_name,
        success=True,
        load_time_sec=load_time,
        gen_time_sec=gen_time,
        output=output,
        error=None,
    )


def main() -> None:
    show_gpu_info()

    prompt = (
        "Write a Python function that securely hashes a password using modern best practices. "
        "Return only the code."
    )

    models = [
        "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "bigcode/starcoder2-3b",
    ]

    results = []
    for model_name in models:
        print("=" * 70)
        result = test_one(model_name, prompt)
        results.append(result)

        print(f"Model: {result.model_name}")
        print(f"Success: {result.success}")
        print(f"Load time: {result.load_time_sec:.2f} sec")
        if result.success:
            print(f"Generation time: {result.gen_time_sec:.2f} sec")
            print("Output:")
            print(result.output[:4000])
        else:
            print(f"Error: {result.error}")
        print("=" * 70)

    print("\nSummary")
    for r in results:
        status = "OK" if r.success else "FAIL"
        print(f"{status} | {r.model_name} | load={r.load_time_sec:.2f}s | gen={r.gen_time_sec:.2f}s")
        if r.error:
            print(f"    {r.error}")


if __name__ == "__main__":
    main()