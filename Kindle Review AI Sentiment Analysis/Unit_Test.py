from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
import torch
import pandas as pd
from datasets import Dataset

# Load the model and tokenizer (LoRA applied)
checkpoint = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint)

# Apply LoRA to the model
lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.1)
model = get_peft_model(model, lora_config)

# Sample small dataset (e.g., 5 rows)
data = {
    "reviewText": [
        "The product is great, really liked it.",
        "Worst purchase I have made.",
        "Not bad, could be better.",
        "I love this product, highly recommend.",
        "Terrible, not worth the price."
    ],
    "rating": [5, 1, 3, 5, 1]  # Example ratings
}
df = pd.DataFrame(data)

# Convert to HuggingFace dataset
dataset = Dataset.from_pandas(df)

# Tokenize function
def tokenize_function(examples):
    return tokenizer(examples['reviewText'], padding="max_length", truncation=True)

# Tokenize the dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Set training arguments for a quick test
training_args = TrainingArguments(
    output_dir="./finetuned_model",  
    per_device_train_batch_size=2,  
    num_train_epochs=1,  # Quick test, 1 epoch
    learning_rate=2e-4,  
    fp16=True,  
    save_strategy="epoch",  
    logging_dir="./logs"
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets
)

# Train for a short period (1 epoch, small batch size)
trainer.train()

print("Fine-tuning with LoRA completed for the unit test!")
