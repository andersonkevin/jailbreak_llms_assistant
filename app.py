from flask import Flask, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import pandas as pd
import logging
from config import PROMPTS_DIR, JAILBREAK_PROMPTS_FILE, REGULAR_PROMPTS_FILE, MODEL_DIR

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)

# Load the model and tokenizer
try:
    model = GPT2LMHeadModel.from_pretrained(MODEL_DIR)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt-2')
    logging.info("Model and tokenizer loaded successfully")
except Exception as e:
    logging.error(f"Error loading model/tokenizer: {e}")
    raise

# Load datasets
def load_prompts(file_name):
    try:
        return pd.read_csv(f'{PROMPTS_DIR}/{file_name}')
    except Exception as e:
        logging.error(f"Error loading prompts: {e}")
        raise

jailbreak_prompts = load_prompts(JAILBREAK_PROMPTS_FILE)
regular_prompts = load_prompts(REGULAR_PROMPTS_FILE)

@app.route('/query', methods=['POST'])
def query():
    try:
        data = request.get_json()
        prompt = data.get('prompt')
        inputs = tokenizer(prompt, return_tensors='pt')
        outputs = model.generate(**inputs)
        response = tokenizer.decode(outputs[0])
        return jsonify({'response': response})
    except Exception as e:
        logging.error(f"Error during query processing: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/get_prompts', methods=['GET'])
def get_prompts():
    try:
        prompt_type = request.args.get('type', default='regular')
        scenario = request.args.get('scenario')

        if prompt_type == 'jailbreak':
            results = jailbreak_prompts
        else:
            results = regular_prompts

        if scenario:
            results = results[results['content_policy_name'] == scenario]

        return results.sample(5).to_json(orient='records')
    except Exception as e:
        logging.error(f"Error getting prompts: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
