import torch
from transformers import RagTokenizer, RagSequenceForGeneration, RagRetriever

def initialize_rag_model():
    # Load the tokenizer and model
    tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")
    model = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq")

    # Load the retriever with proper parameters
    retriever = RagRetriever.from_pretrained("facebook/rag-sequence-nq")
    
    return tokenizer, retriever, model

def generate_answer(question, tokenizer, retriever, model):
    inputs = tokenizer(question, return_tensors="pt")
    input_ids = inputs["input_ids"]

    # Generate response
    with torch.no_grad():
        generated_ids = model.generate(input_ids, num_beams=2, max_length=50)
        
    # Decode and return the answer
    answer = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return answer[0]

def main():
    tokenizer, retriever, model = initialize_rag_model()
    
    # Example question
    question = "What is the capital of France?"
    
    # Generate an answer
    answer = generate_answer(question, tokenizer, retriever, model)
    print("Answer:", answer)

if __name__ == "__main__":
    main()
