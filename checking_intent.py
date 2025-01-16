from transformers import AutoModelForCausalLM, AutoTokenizer

from dotenv import load_dotenv
import os

# Load variables from .env file
load_dotenv()

# Access the variables
API_KEY = os.getenv("API_KEY")

model_name = "gpt2"

# Load the model and tokenizer with the updated parameter
model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=API_KEY)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=API_KEY)

def generate_text(prompt, max_new_tokens=50, temperature=0.7, top_p=0.9):
    """
    Generates text based on the input prompt using the GPT-2 model.
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    # Generate output
    outputs = model.generate(
        inputs['input_ids'], 
        max_new_tokens=max_new_tokens,  # Limit only the generated text
        temperature=temperature, 
        top_p=top_p, 
        do_sample=True
    )
    # Decode and return generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

# Predefined intent-checking prompt
intent_checking_prompt = """
You are an intelligent customer service assistant. Your task is to understand the intent behind customer queries and respond accordingly. Here are the possible intents:
1. Learn about products/services
2. Questions about existing orders/subscriptions
3. Help with an issue or troubleshooting
4. Request for a refund or return
5. Interest in partnerships or collaborations
6. Other (specify)

Now, identify the intent for the following query and respond appropriately:
"""

try:
    # Get user input
    customer_query = input("Enter the customer's query: ")
    final_prompt = f"{intent_checking_prompt}\nCustomer Query: {customer_query}\nResponse:"

    # Generate and display response
    generated_text = generate_text(final_prompt, max_new_tokens=50)
    print("\nGenerated Response:")
    print(generated_text)

except KeyboardInterrupt:
    print("\nProcess interrupted by the user.")
