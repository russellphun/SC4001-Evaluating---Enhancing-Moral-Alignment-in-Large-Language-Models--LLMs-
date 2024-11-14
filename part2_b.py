import torch
from transformers import GPT2Tokenizer, AutoModelForCausalLM
import datetime

# Load the fine-tuned model and tokenizer
model_path = "gpt2_rlhf_ppo"  # Path where your model is saved
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Set the device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define a pre-prompt to set the context or tone for the model
pre_prompt = """The following is a friendly and helpful assistant who answers questions thoughtfully and provides insightful responses. The assistant is well-versed in ethical discussions and can handle morally challenging questions with care and wisdom."""

# Prepare a text file to save the conversation history with a timestamped filename
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
conversation_log_filename = f"data/conversation_{timestamp}.txt"

# Initialize conversation history
conversation_history = pre_prompt + "\n\n"


# Function to generate a response
def generate_response(model, tokenizer, prompt, max_length=100):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # Generate response
    output = model.generate(
        input_ids=input_ids,
        max_length=max_length,
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=2,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )

    # Decode and return response
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response[len(prompt) :].strip()


# Main loop for interactive terminal input
def main():
    print("Chat with the fine-tuned model! Type 'exit' to quit.\n")
    print("Model is ready. Type your question below.\n")

    while True:
        # Get user input
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Exiting chat.")
            break

        # Prepare the prompt with pre-prompt and the latest question only
        prompt = f"{pre_prompt}\n\nUser: {user_input}\nMorally correct answer:"

        # Generate response
        response = generate_response(model, tokenizer, prompt)

        # Log the conversation in the text file
        with open(conversation_log_filename, "a") as log_file:
            log_file.write(f"User: {user_input}\n")
            log_file.write(f"Model: {response}\n\n")

        # Print the model's response
        print(f"Model: {response}\n")


if __name__ == "__main__":
    main()
