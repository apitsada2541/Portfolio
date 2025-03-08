#!pip install transformers torch peft datasets

import pandas as pd
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
import torch
import os
from peft import LoraConfig, get_peft_model
from datasets import Dataset

# Load data from Excel file 
df = pd.read_csv("500_review.csv")

# Define model checkpoint
checkpoint = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenizer.pad_token = tokenizer.eos_token  # Ensure pad token is set

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    checkpoint, 
    torch_dtype=torch.float16,  # Use float16 for efficiency
    device_map="auto"  # Auto-select GPU/CPU
)

# Define LoRA config
lora_config = LoraConfig(
    r=8,  # Rank of LoRA update matrices
    lora_alpha=32,  # Scaling factor
    lora_dropout=0.1,  # Dropout probability
    target_modules=["q_proj", "v_proj"],  # Fine-tune attention layers only
    bias="none",
    task_type="CAUSAL_LM"
)

# Wrap model with LoRA adapter
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

dataset = Dataset.from_pandas(df[['reviewText', 'rating']])
dataset = dataset.rename_column('reviewText', 'text')
dataset = dataset.rename_column('rating', 'label')


def prepare_and_tokenize(example):
    review = example["text"]
    rating = str(example["label"])
    
    template = f'''<|im_start|>system
You are a helpful AI assistant named SmolLM, trained by Hugging Face.<|im_end|>
<|im_start|>user
Review: {review}

Sentiment: Please provide a sentiment rating from 1 to 5, where:
1 = Very Negative, 2 = Negative, 3 = Neutral, 4 = Positive, 5 = Very Positive.

Rating: <|im_end|>
<|im_start|>assistant
{rating}<|im_end|>'''
    
    # Tokenize the inputs
    inputs = tokenizer(template, 
                       truncation=True, 
                       padding="max_length", 
                       max_length=512, 
                       return_tensors="pt")
    
    # Set labels to be the same as input_ids for causal language modeling
    inputs["labels"] = inputs["input_ids"].clone()
    
    # Mask the padding tokens in the labels
    inputs["labels"][inputs["input_ids"] == tokenizer.pad_token_id] = -100
    
    return {key: value.squeeze() for key, value in inputs.items()}

# Apply the function to your dataset
tokenized_datasets = dataset.map(prepare_and_tokenize, remove_columns=dataset.column_names)

training_args = TrainingArguments(
    output_dir="./finetuned_model",  
    per_device_train_batch_size=4,  
    num_train_epochs=3,  
    learning_rate=2e-4,  
    fp16=False,  
    save_strategy="epoch",  
    logging_dir="./logs",
    logging_steps = 1,
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets
)

# Start fine-tuning with LoRA
trainer.train()

model.save_pretrained("Pearl_finetuned_smolLM")
tokenizer.save_pretrained("Pearl_finetuned_smolLM")

model.push_to_hub("pearl41/Pearl_finetuned_smolLM")
tokenizer.push_to_hub("pearl41/Pearl_finetuned_smolLM")

