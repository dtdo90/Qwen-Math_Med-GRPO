# Qwen 2.5 Math and Medical CoT Model with RAG & Tool Use

This repository hosts a fine-tuned version of the [Qwen 2.5 3B Instruct](https://huggingface.co/Qwen) model that has undergone specialized training and enhancements for improved reasoning and real-world application. The model is optimized for **mathematical reasoning** and **medical chain-of-thought** tasks, and is equipped with **Retrieval-Augmented Generation (RAG)** and **tool-calling** capabilities.

## Training

This project builds upon the Qwen 2.5 3B Instruct model and enhances it in two major training stages:

### 1. Reinforcement Learning on Math Dataset
- **Approach**: Applied Reinforcement Learning with Group Relative Policy Optimization (GRPO).
- **Dataset**: [Grade School Math 8K] (https://huggingface.co/datasets/openai/gsm8k)
- **Goal**: Improve the model’s problem-solving ability and mathematical reasoning.

### 2. Instruction Finetuning with Medical Chain-of-Thought (CoT)
- **Dataset**: Chain-of-thought annotated medical dataset distilled from DeepSeek-R1 [FreedomIntelligence/medical-o1-reasoning-SFT] (https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT).
- **Objective**: Enable structured reasoning and interpretability for complex medical QA scenarios.


## Inference
The model is equipped with Retrieval-Augmented Generation (RAG) and tool-calling capabilities.
### 1. Retrieval augmented generation (RAG) and 
- **Prompt routing** – Automatically determine the type of knowledge  based on input question. 
- **RAG** - Integrate external knowledge sources to ground responses with factual information.
### 2. Agentic capability
- **Tool-Calling** – Model can interface with built-in tools and functionalities.

