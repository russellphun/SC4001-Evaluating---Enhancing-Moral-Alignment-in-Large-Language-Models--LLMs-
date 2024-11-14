import time
import logging
import os
import random
import pickle
import torch
import pandas as pd
import openai
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType
from dotenv import load_dotenv
from sage import sage  # Ensure you have the sage module installed and properly configured
from util import QuestionBank  # Import the QuestionBank class from util.py

# Set up logging to a file
logging.basicConfig(
    filename='model_training.log',
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Load environment variables from .env file
load_dotenv()

# Set the OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Set the device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
logging.info(f"Using device: {device}")

# Set random seeds for reproducibility
random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Load the pre-trained GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token  # Set the pad token
model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
model.config.pad_token_id = tokenizer.eos_token_id  # Ensure pad token is set

# Function to get aligned response from OpenAI's GPT-3.5-turbo
def get_aligned_response(question, model_name="gpt-3.5-turbo", temperature=0.8):
    prompt = f"Answer the following question in one paragraph, be concise.\nQuestion: {question}"
    for i in range(5):  # 5 attempts with exponential backoff
        try:
            response = openai.ChatCompletion.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
            )
            aligned_response = response['choices'][0]['message']['content'].strip()
            logging.info(f"Aligned response for question '{question}': {aligned_response}")
            return aligned_response
        except openai.error.OpenAIError as e:
            logging.error(f"Error fetching aligned response: {e}")
            if i == 4:  # On the last attempt, raise the exception
                raise e
            else:
                time.sleep((2 ** i) + 1)  # Exponential backoff

# Function to generate response from the target GPT-2 model
def get_target_response(question, model, tokenizer, max_length=100):
    # Encode the question and move to device
    input_ids = tokenizer.encode(question, return_tensors='pt').to(device)
    # Generate response
    output_ids = model.generate(
        input_ids=input_ids,
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )
    # Decode the generated response, skip the input question
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    response = generated_text[len(question):].strip()
    logging.info(f"Target model response for question '{question}': {response}")
    return response

# Function to evaluate the model using the SAGE benchmark
def evaluate_model(model, tokenizer, questions, is_target=False):
    if is_target:
        def get_response(question):
            return get_target_response(question, model, tokenizer)
    else:
        def get_response(question):
            return get_aligned_response(question)
    # Now evaluate using SAGE
    results = sage.score(questions, get_response, use_rots=True)
    return results

# Function to create training data with a decaying threshold
def create_training_data(questions, target_model, tokenizer, initial_threshold, min_threshold, decay_rate):
    training_data = []
    feedback_loop = []
    threshold = initial_threshold
    for idx, question in enumerate(questions):
        target_response = get_target_response(question, target_model, tokenizer)
        aligned_response = get_aligned_response(question)

        target_score_df = sage.score([question], lambda q: target_response, use_rots=True)
        aligned_score_df = sage.score([question], lambda q: aligned_response, use_rots=True)

        target_score = target_score_df.iloc[0]['score']
        aligned_score = aligned_score_df.iloc[0]['score']

        logging.info(f"Question: {question}")
        logging.info(f"Target model score: {target_score}")
        logging.info(f"Aligned model score: {aligned_score}")
        logging.info(f"Current threshold: {threshold}")

        if aligned_score > threshold:
            training_data.append((question, aligned_response))
            logging.info("Aligned response added to training data.")
        elif target_score > aligned_score:
            feedback_loop.append((question, target_response, aligned_response))
            logging.info("Target model response better than aligned model response; added to feedback loop.")

        # Decay the threshold for the next iteration
        threshold = max(min_threshold, threshold * decay_rate)

    return training_data, feedback_loop

# Custom dataset for fine-tuning
class CustomDataset(Dataset):
    def __init__(self, tokenizer, data, max_length=512):
        self.tokenizer = tokenizer
        self.data = data
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question, response = self.data[idx]
        prompt = f"Question: {question}\nAnswer:"
        # We will train the model to generate the response given the prompt
        input_text = prompt + response
        encoding = self.tokenizer(
            input_text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        # Labels for language modeling
        labels = input_ids.clone()
        # Set labels for the prompt part to -100 so that loss is not computed on them
        prompt_len = len(self.tokenizer.encode(prompt, add_special_tokens=False))
        labels[:prompt_len] = -100
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

# Data collator function
def data_collator(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch]).to(device)
    attention_mask = torch.stack([item['attention_mask'] for item in batch]).to(device)
    labels = torch.stack([item['labels'] for item in batch]).to(device)
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }

def main():
    # Load the list of questions using the QuestionBank class
    question_bank = QuestionBank('data/mcc.csv')

    # Initial parameters
    initial_threshold = 0.9
    min_threshold = 0.7
    decay_rate = 0.95
    max_iterations = 5  # Number of training iterations
    num_questions_per_iteration = 50  # Number of questions to sample each iteration
    previous_avg_score = None
    best_avg_score = -float('inf')
    best_model_state = None
    patience = 2
    patience_counter = 0

    # Load the pre-trained GPT-2 tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token  # Set the pad token
    model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    model.config.pad_token_id = tokenizer.eos_token_id  # Ensure pad token is set

    # Evaluate the baseline GPT-2 model
    baseline_questions = question_bank.get_question(10)
    print("Evaluating the baseline GPT-2 model...")
    logging.info("Starting evaluation of the baseline GPT-2 model.")
    baseline_results = evaluate_model(model, tokenizer, baseline_questions, is_target=True)
    print("Baseline Evaluation Results:")
    for idx, row in baseline_results.iterrows():
        question = row['question']
        score = row['score']
        print(f"Question: {question}")
        print(f"Score: {score}\n")
        logging.info(f"Baseline score for question '{question}': {score}")

    for iteration in range(max_iterations):
        print(f"Starting training iteration {iteration+1}/{max_iterations}")
        logging.info(f"Starting training iteration {iteration+1}/{max_iterations}")

        # Sample questions for this iteration
        questions = question_bank.get_question(num_questions_per_iteration)

        # Create training data
        training_data, feedback_loop = create_training_data(
            questions, model, tokenizer, initial_threshold, min_threshold, decay_rate
        )

        # Check if training data is sufficient
        if not training_data:
            print("No training data was collected. Please adjust the threshold or provide more questions.")
            logging.warning("No training data collected. Consider adjusting the threshold or providing more questions.")
            continue
        else:
            print(f"Collected {len(training_data)} training samples.")
            logging.info(f"Collected {len(training_data)} training samples.")

            # Create the dataset
            dataset = CustomDataset(tokenizer, training_data, max_length=512)

            # LoRA configuration
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=8,
                lora_alpha=32,
                lora_dropout=0.1
            )

            # Reset the model to initial state and prepare for LoRA fine-tuning
            model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
            model.config.pad_token_id = tokenizer.eos_token_id
            model = get_peft_model(model, peft_config).to(device)
            model.print_trainable_parameters()  # Log the trainable parameters

            # Training arguments
            training_args = TrainingArguments(
                output_dir=f'lora_finetuned_gpt2_iteration_{iteration+1}',
                per_device_train_batch_size=1,
                num_train_epochs=1,
                learning_rate=5e-5,
                logging_steps=10,
                save_steps=500,
                save_total_limit=2,
                weight_decay=0.01,
                prediction_loss_only=True,
                report_to='none',
                fp16=True if torch.cuda.is_available() else False,
            )

            # Define the trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=dataset,
                data_collator=data_collator,
            )

            # Start training
            print("Fine-tuning the GPT-2 model with LoRA...")
            logging.info("Starting fine-tuning with LoRA.")
            trainer.train()

            # Evaluate the fine-tuned model
            print("Evaluating the fine-tuned GPT-2 model...")
            logging.info("Evaluating the fine-tuned GPT-2 model.")
            finetuned_results = evaluate_model(model, tokenizer, baseline_questions, is_target=True)
            print("Fine-tuned Evaluation Results:")
            for idx, row in finetuned_results.iterrows():
                question = row['question']
                score = row['score']
                print(f"Question: {question}")
                print(f"Score: {score}\n")
                logging.info(f"Fine-tuned model score for question '{question}': {score}")

            # Compare the results
            print("Comparing the baseline and fine-tuned model results...")
            logging.info("Comparing baseline and fine-tuned model results.")
            improvements = []
            for idx in range(len(baseline_questions)):
                question = baseline_questions[idx]
                baseline_score = baseline_results.iloc[idx]['score']
                finetuned_score = finetuned_results.iloc[idx]['score']
                improvement = finetuned_score - baseline_score
                improvements.append(improvement)
                print(f"Question: {question}")
                print(f"Baseline Score: {baseline_score}")
                print(f"Fine-tuned Score: {finetuned_score}")
                print(f"Score Improvement: {improvement}\n")
                logging.info(f"Question: {question}")
                logging.info(f"Baseline Score: {baseline_score}")
                logging.info(f"Fine-tuned Score: {finetuned_score}")
                logging.info(f"Score Improvement: {improvement}\n")

            # Calculate average improvement
            avg_improvement = sum(improvements) / len(improvements)
            print(f"Average Score Improvement: {avg_improvement}")
            logging.info(f"Average Score Improvement: {avg_improvement}")

            # Save the model if it's the best so far
            if avg_improvement > best_avg_score:
                best_avg_score = avg_improvement
                best_model_state = model.state_dict()
                # Save the best model
                model.save_pretrained('best_lora_finetuned_gpt2')
                tokenizer.save_pretrained('best_lora_finetuned_gpt2')
                print("New best model saved.")
                logging.info("New best model saved.")

            # Check if score has stabilized (change less than a threshold)
            if previous_avg_score is not None:
                score_change = abs(avg_improvement - previous_avg_score)
                if score_change < 0.01:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print("Score has stabilized. Stopping training.")
                        logging.info("Score has stabilized. Stopping training.")
                        break
                else:
                    patience_counter = 0
            previous_avg_score = avg_improvement

    print("Training complete.")
    logging.info("Training complete.")

    # Load the best model for final evaluation
    print("Loading the best model for final evaluation...")
    logging.info("Loading the best model for final evaluation.")
    model = GPT2LMHeadModel.from_pretrained('best_lora_finetuned_gpt2').to(device)
    tokenizer = GPT2Tokenizer.from_pretrained('best_lora_finetuned_gpt2')

    # Final evaluation
    print("Evaluating the best fine-tuned GPT-2 model...")
    logging.info("Evaluating the best fine-tuned GPT-2 model.")
    final_results = evaluate_model(model, tokenizer, baseline_questions, is_target=True)
    print("Final Evaluation Results:")
    for idx, row in final_results.iterrows():
        question = row['question']
        score = row['score']
        print(f"Question: {question}")
        print(f"Score: {score}\n")
        logging.info(f"Final model score for question '{question}': {score}")

if __name__ == "__main__":
    main()
