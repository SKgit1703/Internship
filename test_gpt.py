from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the model and tokenizer (replace with your chosen model name)
model_name = 'huggingface/llama-7b'  # Replace with the actual model name
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors='pt')
    outputs = model.generate(inputs['input_ids'], max_length=150, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def chat_with_bot():
    print("Chatbot is ready to talk! (type 'quit' to exit)")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break
        response = generate_response(user_input)
        print(f"Bot: {response}")

if __name__ == "__main__":
    chat_with_bot()
