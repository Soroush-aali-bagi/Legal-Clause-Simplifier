from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer
import torch

def setup_qlora(model_name="meta-llama/Meta-Llama-3-8B"):
    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )
    
    # Load model with QLoRA config
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # LoRA config
    lora_config = LoraConfig(
        r=64,               # Rank
        lora_alpha=16,      # Alpha scaling
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    return model, lora_config

def train_legal_simplifier(train_dataset, eval_dataset=None):
    model, lora_config = setup_qlora()
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./models/qlora_adapter",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        optim="paged_adamw_32bit",
        save_steps=500,
        logging_steps=10,
        learning_rate=2e-4,
        fp16=False,
        bf16=True,
        max_grad_norm=0.3,
        num_train_epochs=3,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        evaluation_strategy="steps" if eval_dataset else "no",
        eval_steps=100 if eval_dataset else None,
        report_to="wandb"
    )
    
    # Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=lora_config,
        dataset_text_field="text",
        max_seq_length=512,
        tokenizer=tokenizer,
        args=training_args,
        packing=True
    )
    
    # Train
    trainer.train()
    trainer.save_model("./models/qlora_adapter")