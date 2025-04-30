#!/usr/bin/env python3
# legal_simplifier_training.py
# QLoRA Fine-tuning of Llama 3 for Legal Clause Simplification

import torch
from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from src.data_processing import load_legal_dataset, tokenize_dataset
from src.training import setup_qlora
from src.evaluation import evaluate_simplification
from src.inference import load_legal_simplifier
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting Legal Clause Simplifier Training")
    
    # 1. Data Preparation
    logger.info("Loading and processing data...")
    try:
        data_pairs = load_legal_dataset("../data/raw_contracts/sample.json")
        tokenized_data = tokenize_dataset(data_pairs)
        
        dataset = Dataset.from_dict({
            "input_ids": tokenized_data["input_ids"],
            "attention_mask": tokenized_data["attention_mask"]
        })
        
        dataset = dataset.train_test_split(test_size=0.1)
        train_dataset = dataset["train"]
        eval_dataset = dataset["test"]
        logger.info(f"Data loaded successfully. Train samples: {len(train_dataset)}, Eval samples: {len(eval_dataset)}")
    except Exception as e:
        logger.error(f"Data loading failed: {str(e)}")
        raise

    # 2. Model Setup
    logger.info("Setting up QLoRA model...")
    try:
        model, lora_config = setup_qlora()
        logger.info(f"Model loaded successfully on device: {model.device}")
    except Exception as e:
        logger.error(f"Model setup failed: {str(e)}")
        raise

    # 3. Training
    logger.info("Starting training process...")
    try:
        training_args = TrainingArguments(
            output_dir="../models/qlora_adapter",
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            optim="paged_adamw_32bit",
            save_steps=500,
            logging_steps=10,
            learning_rate=2e-4,
            num_train_epochs=3,
            fp16=False,
            bf16=torch.cuda.get_device_capability()[0] >= 8,
            report_to="wandb",
            evaluation_strategy="steps",
            eval_steps=100,
            load_best_model_at_end=True
        )

        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            peft_config=lora_config,
            args=training_args,
            max_seq_length=512
        )

        logger.info("Training started...")
        trainer.train()
        trainer.save_model("../models/qlora_adapter")
        logger.info("Training completed and model saved successfully.")
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

    # 4. Evaluation
    logger.info("Running evaluation...")
    try:
        pipe = load_legal_simplifier()
        
        test_clauses = [
            "The Indemnifying Party shall hold harmless and indemnify the Indemnified Party...",
            "Notwithstanding anything to the contrary herein..."
        ]

        for clause in test_clauses:
            simplified = simplify_clause(pipe, clause)
            logger.info(f"\nOriginal: {clause}\nSimplified: {simplified}\n")
            
            # Evaluate against reference if available
            if any(pair['complex'] == clause for pair in data_pairs):
                reference = next(pair for pair in data_pairs if pair['complex'] == clause)
                metrics = evaluate_simplification(simplified, reference)
                logger.info(f"Evaluation Metrics: {metrics}")
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise

def simplify_clause(pipe, legal_clause):
    """Helper function to simplify a single clause"""
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

if __name__ == "__main__":
    main()