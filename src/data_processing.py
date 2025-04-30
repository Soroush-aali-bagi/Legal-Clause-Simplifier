import json
from datasets import load_dataset
from transformers import AutoTokenizer

def load_legal_dataset(file_path):
    """Load and process legal documents"""
    with open(file_path) as f:
        contracts = json.load(f)
    
    # Process into clause-simplification pairs
    pairs = []
    for contract in contracts:
        for clause in contract['clauses']:
            pairs.append({
                'complex': clause['text'],
                'simple': clause.get('explanation', '')
            })
    return pairs

def tokenize_dataset(pairs, model_name="meta-llama/Meta-Llama-3-8B"):
    """Tokenize the dataset for training"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Format for instruction fine-tuning
    formatted_texts = [
        f"Simplify this legal clause to plain English:\n\nComplex: {pair['complex']}\n\nSimple: {pair['simple']}"
        for pair in pairs
    ]
    
    return tokenizer(formatted_texts, truncation=True, padding="max_length", max_length=512)