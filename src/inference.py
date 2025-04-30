from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

def load_legal_simplifier(base_model="meta-llama/Meta-Llama-3-8B", adapter_path="./models/qlora_adapter"):
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # Load QLoRA adapter
    model = PeftModel.from_pretrained(model, adapter_path)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    
    # Create pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto"
    )
    
    return pipe

def simplify_clause(pipe, legal_clause):
    prompt = f"""Simplify this legal clause to plain English:
    
Complex: {legal_clause}

Simple:"""
    
    result = pipe(
        prompt,
        max_new_tokens=256,
        temperature=0.3,
        top_p=0.9,
        do_sample=True
    )
    
    return result[0]['generated_text'].split("Simple:")[-1].strip()