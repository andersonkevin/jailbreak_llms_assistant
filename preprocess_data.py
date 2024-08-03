import pandas as pd
import logging
from config import PROMPTS_DIR, JAILBREAK_PROMPTS_FILE, REGULAR_PROMPTS_FILE, FORBIDDEN_QUESTIONS_FILE

logging.basicConfig(level=logging.INFO)

def load_and_clean_data():
    try:
        jailbreak_prompts = pd.read_csv(f'{PROMPTS_DIR}/{JAILBREAK_PROMPTS_FILE}')
        regular_prompts = pd.read_csv(f'{PROMPTS_DIR}/{REGULAR_PROMPTS_FILE}')
        forbidden_questions = pd.read_csv(f'{PROMPTS_DIR}/{FORBIDDEN_QUESTIONS_FILE}')

        # Drop missing values
        jailbreak_prompts.dropna(inplace=True)
        regular_prompts.dropna(inplace=True)
        forbidden_questions.dropna(inplace=True)

        logging.info("Data loaded and cleaned successfully")
        return jailbreak_prompts, regular_prompts, forbidden_questions
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

if __name__ == "__main__":
    jp, rp, fq = load_and_clean_data()
    print("Data loaded and cleaned successfully")
