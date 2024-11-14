import torch
import logging
from transformers import GPT2Tokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from sage import sage

from util import QuestionBank  # Ensure you have the sage module installed and properly configured

# Set up logging to a file
logging.basicConfig(
    filename='part2_training.log',
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

NUM_QN = 10

# Load the pre-trained GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

# Load the GPT-2 model with a value head for PPO
model = AutoModelForCausalLMWithValueHead.from_pretrained('gpt2')

# Define the PPO configuration
ppo_config = PPOConfig(
    model_name='gpt2',
    learning_rate=1e-5,
    log_with='tensorboard',
    batch_size=32,
    forward_batch_size=1,
    remove_unused_columns=False,  # Important for causal LM
)

# Initialize the PPO trainer
ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=model,
    tokenizer=tokenizer,
)

# Function to build the prompt
def build_prompt(question):
    return f"Question: {question}\nAnswer:"

# Function to generate response from the model
def generate_response(model, tokenizer, question, max_length=100):
    prompt = build_prompt(question)
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(model.device)
    output = model.generate(
        input_ids=input_ids,
        max_length=max_length,
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=2,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    answer = response[len(prompt):].strip()
    return prompt, answer

# Function to compute reward using SAGE
def compute_reward(question, response):
    result = sage.score([question], [response], use_rots=True)
    score = result[0]['score']
    return score


question_bank = QuestionBank('data/mcc.csv')

import numpy as np
import torch

# Define stability criteria
def check_stability(reward_history, threshold=0.01, patience=3):
    # If reward history has fewer than patience cycles, return False
    if len(reward_history) < patience:
        return False
    # Check if the change in average reward over the last `patience` cycles is below the threshold
    recent_rewards = reward_history[-patience:]
    avg_change = np.abs(np.diff(recent_rewards)).mean()
    return avg_change < threshold

# Track rewards across cycles to check for stability
reward_history = []

# Training loop with stability criterion
print("Starting RLHF training with PPO until stability...")
cycle = 0
max_cycles = 20  # Set a maximum number of cycles to avoid infinite loops

while cycle < max_cycles:
    cycle += 1
    print(f"Cycle {cycle}: Training Phase")
    logging.info(f"Cycle {cycle}: Training Phase")

    questions = question_bank.get_question(NUM_QN)

    # Run training for one epoch
    for question in questions:
        prompt = build_prompt(question)
        query_tensor = tokenizer.encode(prompt, return_tensors='pt').to(model.device)

        # Generate response
        response_tensor = ppo_trainer.model.generate(
            input_ids=query_tensor,
            max_length=200,
            pad_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=2,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

        # Extract and log the answer
        response_text = tokenizer.decode(response_tensor[0], skip_special_tokens=True)
        answer = response_text[len(prompt):].strip()
        reward = compute_reward(question, answer)
        reward_tensor = torch.tensor([reward]).to(model.device)

        logging.info(f"Question: {question}")
        logging.info(f"Response: {answer}")
        logging.info(f"Reward: {reward}")

        # Run PPO update
        ppo_trainer.step(query_tensor, response_tensor, reward_tensor)

    # Evaluate model and check stability
    print(f"Cycle {cycle}: Evaluation Phase")
    logging.info(f"Cycle {cycle}: Evaluation Phase")
    total_score = 0

    for question in questions:
        prompt, answer = generate_response(model, tokenizer, question)
        score = compute_reward(question, answer)
        total_score += score
        logging.info(f"Question: {question}")
        logging.info(f"Response: {answer}")
        logging.info(f"Score: {score}")

    # Calculate and log average reward
    avg_reward = total_score / len(questions)
    reward_history.append(avg_reward)
    print(f"Cycle {cycle} Average Reward: {avg_reward}")
    logging.info(f"Cycle {cycle} Average Reward: {avg_reward}")

    # Check stability and break loop if stable
    if check_stability(reward_history):
        print(f"Model stabilized with average reward: {avg_reward}")
        logging.info(f"Model stabilized with average reward: {avg_reward}")
        break

# Save the fine-tuned model after achieving stability
print("Saving the fine-tuned model...")
model.save_pretrained('gpt2_rlhf_ppo')
tokenizer.save_pretrained('gpt2_rlhf_ppo')
logging.info("Model saved successfully.")
