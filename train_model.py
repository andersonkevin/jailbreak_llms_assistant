import pandas as pd
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
import logging
from config import PROMPTS_DIR, JAILBREAK_PROMPTS_FILE, REGULAR_PROMPTS_FILE, MODEL_DIR
from huggingface_hub import login

logging.basicConfig(level=logging.INFO)

# Ensure authentication with Hugging Face Hub
try:
    login()
    logging.info("Successfully authenticated with Hugging Face Hub.")
except Exception as e:
    logging.error(f"Authentication failed: {e}")
    raise

def load_data():
    try:
        jailbreak_prompts = pd.read_csv(f'{PROMPTS_DIR}/{JAILBREAK_PROMPTS_FILE}')
        regular_prompts = pd.read_csv(f'{PROMPTS_DIR}/{REGULAR_PROMPTS_FILE}')
        combined_data = pd.concat([jailbreak_prompts[['prompt']], regular_prompts[['prompt']]], ignore_index=True)
        logging.info("Data loaded successfully.")
        return combined_data
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def train_model():
    try:
        combined_data = load_data()
        logging.info("Loading tokenizer and model.")
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        
        # Set padding token
        tokenizer.pad_token = tokenizer.eos_token

        logging.info("Tokenizing the dataset.")
        def tokenize_function(examples):
            return tokenizer(examples['prompt'], truncation=True, padding='max_length', max_length=512)
        
        dataset = Dataset.from_pandas(combined_data)
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        model = GPT2LMHeadModel.from_pretrained('gpt2')

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )

        training_args = TrainingArguments(
            output_dir=MODEL_DIR,
            num_train_epochs=3,
            per_device_train_batch_size=2,
            save_steps=10_000,
            save_total_limit=2,
        )

        logging.info("Initializing Trainer.")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
        )

        logging.info("Starting training.")
        trainer.train()
        logging.info("Model training completed successfully.")
        
        logging.info("Saving the model.")
        trainer.save_model(MODEL_DIR)
    except Exception as e:
        logging.error(f"Error during training: {e}")
        raise

if __name__ == "__main__":
    train_model()
