# Qwen 2.5 Math + Medical CoT Enhanced Model with RAG & Tool Use

This repository hosts a fine-tuned version of the [Qwen 2.5 3B Instruct](https://huggingface.co/Qwen) model that has undergone specialized training and enhancements for improved reasoning and real-world application. The model is optimized for mathematical reasoning and medical chain-of-thought tasks, and is equipped with Retrieval-Augmented Generation (RAG) and tool-calling capabilities.

## Overview

This project builds upon the base Qwen 2.5 3B Instruct model and enhances it in two major training stages:

### 1. Reinforcement Learning on Math Dataset
- **Approach**: Applied Reinforcement Learning with Generalized Policy Optimization (GRPO).
- **Dataset**: [Grade School Math 8K] (https://huggingface.co/datasets/openai/gsm8k)
- **Goal**: Improve the model’s problem-solving ability and mathematical reasoning.

### 2. Instruction Finetuning with Medical Chain-of-Thought (CoT)
- **Dataset**: Chain-of-thought annotated medical dataset distilled from DeepSeek-R1 [FreedomIntelligence/medical-o1-reasoning-SFT] (https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT).
- **Objective**: Enable structured reasoning and interpretability for complex medical QA scenarios.

## 🔧 Features

- ✅ **Enhanced Math Reasoning** – Optimized through RL with GRPO for accuracy and step-by-step logic.
- ✅ **Medical Domain Chain-of-Thought** – Finetuned to provide transparent and multi-step explanations in the medical domain.
- ✅ **RAG (Retrieval-Augmented Generation)** – Integrates external knowledge sources to ground responses with factual information.
- ✅ **Tool-Calling** – Model can interface with built-in tools and functionalities.

