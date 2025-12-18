# Effects of Variable Bit Quantization on Bias in Large Language Models

## Important Note on Execution Environment

The main experimental pipeline for this project was developed and run in Google Colab, primarily to leverage additional compute resources beyond what was available locally.

The file main_script.ipynb is a direct download from Google Colab and was not modified to guarantee local execution.
If you attempt to run this notebook outside of Colab, additional changes may be required (e.g., dependency installation, file paths, or hardware-specific settings).

# Overview

This repository contains the code, notebooks, and analysis used in the research project:

“Effect of Variable Bit Quantization on Bias within Large Language Models.”

The project investigates how post-training quantization at different bit-widths affects measured social bias in large language models (LLMs). While quantization is widely used to reduce model size and inference cost, its interaction with fairness and bias metrics remains underexplored.

We conduct a systematic bit-width sweep (2–30 bits) and evaluate how bias and utility change under varying levels of compression.

Research Questions

This project focuses on three core questions:

How does quantization bit-width affect measured social bias in LLMs?

Does moderate quantization reduce bias while preserving model utility?

At what point does aggressive compression destabilize model behavior?

## Models Evaluated

We study a set of encoder-only masked language models:

**BERT-Base**

**DistilBERT-Base**

**RoBERTa-Base**

**DistilRoBERTa-Base**

For comparison with prior work, we additionally evaluate:

**BERT-Large**

**RoBERTa-Large**
_(full precision and 8-bit only)_

$$ Quantization Methods

Two post-training quantization approaches are used:

**1. PyTorch 8-Bit Dynamic Quantization**

Uses torch.ao.quantization.quantize_dynamic

Applies integer quantization to all linear layers

Produces an actual compressed model artifact

Serves as a baseline aligned with prior literature

**2. Custom Uniform N-Bit Quantization**

Sweeps bit-widths: 2, 4, 6, 8, 12, 16, 20, 24, 28, 30

Applies symmetric, weight-only quantization

Weights are quantized then dequantized for evaluation

Model sizes and compression ratios are theoretical (N / 32 scaling)

This setup allows controlled, apples-to-apples comparisons across bit-widths.

## Bias & Utility Evaluation

Bias is measured using two established benchmarks:

**CrowS-Pairs**

Measures preference for stereotypical vs. anti-stereotypical sentence pairs

Reports:

Overall stereotype percentage

Group-specific stereotype percentages

Ideal score: 50% (neutral)

**StereoSet**

Evaluates stereotype preference and language modeling quality

Reports:

Stereotype Score (SS) — lower is better

LM-OK — measures semantic utility and coherence

Together, these metrics allow analysis of both fairness and model quality.

## Key Findings

8-bit quantization produces a statistically significant reduction in CrowS-Pairs bias relative to FP32

Moderate quantization (≈6–12 bits):

Reduces bias across benchmarks

Preserves language modeling utility

Aggressive quantization (2–4 bits):

Destabilizes model behavior

Increases stereotyping

Severely degrades utility

Overall, quantization behaves like a bias regularizer within a narrow precision window — but is not a standalone debiasing method.
