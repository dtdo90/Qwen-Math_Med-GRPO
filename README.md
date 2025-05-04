# Qwen 2.5 Math and Medical CoT Model with RAG & Tool Use

This repository hosts a fine-tuned version of the [Qwen 2.5 3B Instruct](https://huggingface.co/Qwen) model that has undergone specialized training and enhancements for improved reasoning and real-world application. The model is optimized for **mathematical reasoning** and **medical chain-of-thought** tasks, and is equipped with **Retrieval-Augmented Generation (RAG)** and **tool-calling** capabilities.

## Training

This project builds upon the Qwen 2.5 3B Instruct model and enhances it in two major training stages:

### 1. Reinforcement Learning on Math Dataset
- **Approach**: Applied Reinforcement Learning with Group Relative Policy Optimization (GRPO).
- **Dataset**: [Grade School Math 8K] (https://huggingface.co/datasets/openai/gsm8k)
- **Goal**: Improve the model's problem-solving ability and mathematical reasoning.
-  **Code:** See `01_02_qwen_GRPO_CoT.ipynb` (Part 1).

### 2. Instruction Finetuning with Medical Chain-of-Thought (CoT)
- **Dataset**: Chain-of-thought annotated medical dataset distilled from DeepSeek-R1 [FreedomIntelligence/medical-o1-reasoning-SFT] (https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT).
- **Objective**: Enable structured reasoning and interpretability for complex medical QA scenarios.
-  **Code:** See `01_02_qwen_GRPO_CoT.ipynb` (Part 2).



## Inference
The model is equipped with Retrieval-Augmented Generation (RAG) and tool-calling capabilities.

### 1. Prompt Routing
For a Chain-of-Thought (CoT) model with specialized knowledge domains, it's crucial to identify the appropriate knowledge domain for each input question. This process, known as prompt routing, ensures the model applies the correct expertise and reasoning framework.

We implement two prompt routing approaches:
1. **Embedding-Based Routing**: Uses semantic similarity to match questions with appropriate knowledge domains
2. **LLM-Based Routing**: Leverages the model's own understanding to determine the most relevant domain
➡️ **Code:** `03_inference_prompt_routing.ipynb`


### 2. Retrieval-Augmented Generation (RAG)
To enhance response accuracy and factual grounding, the model integrates external knowledge through two RAG implementations:

1. **Local Document RAG**: Retrieves and incorporates information from pre-processed local documents
    *   ➡️ **Code:** `04_rag_local_medical/` (run `data.py` first, then `inference_local.py`).
2. **Web RAG**: Crawl web for relevant information and fetch them into the model as context
    *   ➡️ **Code:** `05_rag_web_medical/` (run `data_scrap.py` first, then `inference.py`).

### 3. Agentic Capabilities
The model can interact with both built-in tools and external APIs to extend its functionality
1. **Basic Agent:** Demonstrates simple tool calling with the model.
    *   ➡️ **Code:** `06_agent_basic.py`
2. **Advanced CoT Agent:** Combines prompt routing, tool-use decision-making, and CoT formatting with the fine-tuned model.
    *   ➡️ **Code:** `07_agent_cot.py`
