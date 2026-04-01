# Mitigating Security Hallucinations in Code LLMs

## Overview

This project investigates **security hallucinations in code-generating LLMs** — cases where models produce:

- insecure code (e.g., MD5 hashing)
- deprecated or incorrect APIs
- hallucinated or misleading dependencies

The goal is to **measure, analyze, and eventually mitigate** these failures.

---

## Progress So Far

### 1. Dataset Construction

Built a **500-prompt benchmark dataset**:

- Benign prompts (HumanEval + MBPP)
- Security-aware prompts
- Adversarial prompts
- Vulnerability-based prompts

This dataset covers:

- normal coding tasks
- security-specific scenarios
- adversarial / prompt-injection cases

---

### 2. Model Setup

- Configured **CUDA + PyTorch**
- Running locally on **RTX 3050 (4GB VRAM)**
- Using **4-bit quantization** for efficiency

Models used:

- `Qwen2.5-Coder-1.5B-Instruct`
- `StarCoder2-3B`

---

### 3. Output Generation

- Ran both models on all 500 prompts
- Stored outputs in structured JSON format

---

### 4. Labeling Pipeline

Implemented a **rule-based labeling system** (no LLM bias) to detect:

- Insecure patterns (MD5, eval, exec, etc.)
- Package hallucinations
- API conflicts
- Data exposure (hardcoded secrets)
- Language drift (non-Python outputs)
- Python correctness

---

## Initial Results

### Qwen (1.5B)

- 451 / 500 Python-like outputs
- 35 insecure patterns
- 18 data exposure issues
- 2 package hallucinations
- 1 language drift

### StarCoder2 (3B)

- ~236 Python-like outputs
- Higher hallucination and drift
- More unstable / off-task outputs
- Fewer insecure patterns but worse alignment

---

## Key Insight

Different models fail in different ways:

- **Qwen**
  - Generates mostly correct Python
  - But often produces **insecure code**

- **StarCoder2**
  - Frequently produces **hallucinated or irrelevant outputs**
  - Struggles with instruction following

This confirms that code LLMs exhibit both:

- **semantic hallucinations**
- **security hallucinations**

---

## Current Status
