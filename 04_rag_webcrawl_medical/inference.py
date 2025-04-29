import unsloth, torch, os
from unsloth import FastLanguageModel
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer
from langchain_community.vectorstores import FAISS
from transformers import TextStreamer
import numpy as np
import gc
from pathlib import Path

def load_model_tokenizer_ft(base_model_id="Qwen/Qwen2.5-3B-Instruct", adapter_path="grpo_lora"):
    """Load the finetuned model with proper error handling and validation"""
    # Go up one directory from cwd, then into grpo_lora
    adapter_path = os.path.join(os.path.dirname(os.getcwd()), adapter_path)
    # Load the base model with memory optimization
    base_model, _ = FastLanguageModel.from_pretrained(
        model_name=base_model_id,
        max_seq_length=4200,  # accommodate up to 2 documents of 1000 tokens each
        load_in_4bit=True,
        device_map="auto"  
    )

    # Apply LoRA weights
    model = FastLanguageModel.get_peft_model(
        base_model,
        r=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=32,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    # Load adapter 
    model.load_adapter(adapter_path, adapter_name="default")
    # put model in evaluation mode
    model.eval()

    # Load tokenizer 
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)

    return model, tokenizer

def cleanup_resources(model=None):
    """Clean up GPU memory and other resources"""
    if model is not None:
        del model
    torch.cuda.empty_cache()
    gc.collect()

# Use SentenceTransformer as a LangChain-compatible embedding function
embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")

prompt_names=["math", "medical", "others"]
prompt_template_math=(
    "You are an expert in solving math problems. "
    "You first think through the reasoning process step-by-step in your mind and then provide the user with the answer."
)

prompt_template_medical=(
    "You are a medical expert with advanced knowledge in clinical reasoning, diagonstics, and treatment planning. "
    "You first think through the reasoning process step-by-step in your mind and then provide the user with the answer."
)

prompt_template_others = (
    f"You are great at answering all questions not from the following themes: {prompt_names[:-1]}. "
    "You first think through the reasoning process step-by-step in your mind and then provide the user with the answer."
)

prompt_templates = [
    prompt_template_math,
    prompt_template_medical,
    prompt_template_others,
]

prompt_embeddings = embedding.embed_documents(prompt_templates)
prompt_embeddings = np.asarray(prompt_embeddings)

def select_prompt(question):
    """Select prompt based on question's content"""
    
    try:
        question_embedding = embedding.embed_documents([question])
        question_embedding = np.asarray(question_embedding)
        scores = question_embedding@prompt_embeddings.T
        selected_prompt_idx = scores.squeeze(0).argmax()
        
        if selected_prompt_idx == 0:
            return prompt_templates[selected_prompt_idx], 0.1
        elif selected_prompt_idx == 1:
            return prompt_templates[selected_prompt_idx], 0.4
        else:
            return prompt_templates[selected_prompt_idx], 0.7
    except Exception as e:
        print(f"Error in prompt selection: {str(e)}")
        return prompt_templates[-1], 0.7  # Fallback to default prompt

def inference(model, tokenizer, question):
    """Run inference with proper error handling and resource management"""
    system_prompt, temperature = select_prompt(question)

    # Validate vector store path
    vector_store_path = os.path.join(os.getcwd(), "cancer_vectorstore")
    
    # Initialize retriever with error handling
    try:
        # Load FAISS vector store
        vectorstore = FAISS.load_local(
            vector_store_path,
            embeddings=embedding,
            allow_dangerous_deserialization=True  # Safe since we created the vector store
        )
        
        # Get documents with similarity scores
        docs_with_scores = vectorstore.similarity_search_with_score(question, k=2)
        
        # Filter documents based on distance threshold
        MAX_DISTANCE = 1  # Maximum acceptable L2 distance
        relevant_docs = []
        
        print("\nDocument distances (lower is better):")
        for doc, score in docs_with_scores:
            if score <= MAX_DISTANCE:
                relevant_docs.append(doc)
                print(f"Distance: {score:.4f} - Document selected")
            else:
                print(f"Distance: {score:.4f} - Document too far")
        
        if not relevant_docs:
            print("Warning: No documents found within distance threshold. Proceeding without context.")
            context = ""
        else:
            context = "\n\n".join([d.page_content for d in relevant_docs])
            print(f"\nUsing {len(relevant_docs)} relevant documents for context")

    except Exception as e:
        print(f"Error retrieving documents: {str(e)}")
        relevant_docs = []
        context = ""

    # Prepare messages with validation
    template = (
        "Use the following pieces of context to answer the question at the end. "
        "If the context is None or empty, simply answer the question based on your knowledge. "
        "If you don't know the answer, just say that you don't know, don't make up an answer.\n"
        "Show your reasoning in <think> </think> tags. "
        "And return the final answer in <answer> </answer> tags. "
        "Stop generating after <answer> </answer> tags\n"
        "{context}\n"
        "Question: {question}"
    )
    
    user_prompt = template.format(context=context, question=question)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": "Let me solve this step by step.\n<think>"}
    ]

    # Generate response with error handling
    try:
        inputs = tokenizer.apply_chat_template(messages, tokenize=False, continue_final_message=True)
        model_inputs = tokenizer([inputs], return_tensors="pt").to(model.device)
        
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=1024,
            temperature=temperature,
            streamer=TextStreamer(tokenizer, skip_special_tokens=True),
            pad_token_id=tokenizer.pad_token_id
        )
        
        # Process response with validation
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]            
        return response

    except Exception as e:
        print(f"Error during generation: {str(e)}")
        return f"Error: {str(e)}"

if __name__ == "__main__":
    try:
        model, tokenizer = load_model_tokenizer_ft()
        print("Type your question (or exit to quit)")
        
        while True:
            try:
                question = input(">> ").strip()
                if not question:
                    continue
                if question.lower() == "exit":
                    break
                    
                response = inference(model, tokenizer, question)
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {str(e)}")
                continue
                
    finally: # free up memory after exiting the loop
        cleanup_resources(model)
