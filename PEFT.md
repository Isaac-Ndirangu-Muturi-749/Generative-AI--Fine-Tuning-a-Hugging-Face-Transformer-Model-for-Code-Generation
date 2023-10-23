# Progressive Embedding Fine-Tuning (PEFT)

This document provides details on Progressive Embedding Fine-Tuning (PEFT) and quantization configuration used for fine-tuning the model for code generation.

## Table of Contents
- [PEFT Configuration](#peft-configuration)
- [Quantization Configuration](#quantization-configuration)
- [Target Modules](#target-modules)

## PEFT Configuration

Progressive Embedding Fine-Tuning (PEFT) is a technique that improves model training. In the context of code generation, PEFT can be used to enhance the model's understanding of code-related tasks. Here is the configuration used:

- `Lora Alpha`: Set to 16
- `Lora Dropout`: Set to 0.5
- `Bias`: None
- `Task Type`: CAUSAL_LM
- `Target Modules`: q_proj, v_proj, k_proj

## Quantization Configuration

Quantization is the process of reducing model memory and improving inference efficiency. For this project, a 4-bit quantization with float16 compute dtype is used:

- `Quantization Type`: 4-bit
- `Compute Dtype`: float16
- `Double Quant`: Enabled

## Target Modules

Target modules are specific components of the model that are fine-tuned using PEFT. In this code generation task, the following modules are targeted:

- `q_proj`
- `v_proj`
- `k_proj`

These modules play a crucial role in enhancing the model's understanding of code-related prompts.

By configuring PEFT and quantization settings, the fine-tuned model becomes optimized for code generation tasks.

For training details, please refer to [TRAINING.md](TRAINING.md).
