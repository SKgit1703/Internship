from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from datasets import load_dataset
from langchain.schema import Document
from whoosh.index import create_in
from whoosh.fields import Schema, TEXT, ID
from whoosh.qparser import QueryParser
from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration
import os
import random

# Global conversation history
conversation_history = []

# Define custom fallback prompts
FALLBACK_PROMPTS = [
    "I'm sorry, I couldn't find an answer to your question.",
    "Please try rephrasing your query.",
    "I don't have enough information to answer that right now."
]

# 1. Document Ingestion
def load_docs(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)

    docs = []
    for doc in documents:
        page_text = doc.page_content
        page_number = doc.metadata.get('page_number', 'Unknown')
        chunks = text_splitter.split_text(page_text)

        for chunk in chunks:
            docs.append({
                'page_content': chunk,
                'metadata': {'page_number': page_number}
            })

    return docs

# 2. Indexing
def create_vector_store(docs):
    # Convert each dict in docs to a Document object
    documents = [Document(page_content=doc['page_content'], metadata=doc.get('metadata', {})) for doc in docs]

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create FAISS vector store
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore

def create_whoosh_index(docs, index_dir):
    # Ensure the directory exists
    if not os.path.exists(index_dir):
        os.makedirs(index_dir)

    schema = Schema(id=ID(stored=True), content=TEXT(stored=True), page_number=ID(stored=True))
    ix = create_in(index_dir, schema)
    writer = ix.writer()
    for i, doc in enumerate(docs):
        writer.add_document(id=str(i), content=doc['page_content'], page_number=str(doc['metadata'].get('page_number', 'Unknown')))
    writer.commit()

# 3. Retrieval-Augmented Generation (RAG)
def initialize_rag_model():
    tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")
    retriever = RagRetriever.from_pretrained("facebook/rag-sequence-nq")  # Using dummy dataset
    model = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq")
    return tokenizer, retriever, model

def get_rag_answer(query, tokenizer, retriever, model):
    inputs = tokenizer(query, return_tensors="pt")
    retrieved_docs = retriever(inputs['input_ids'], return_tensors="pt")
    
    # Generate the answer using the RAG model
    outputs = model.generate(input_ids=inputs['input_ids'], 
                             attention_mask=inputs['attention_mask'],
                             context_input_ids=retrieved_docs['context_input_ids'],
                             context_attention_mask=retrieved_docs['context_attention_mask'])
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 4. Query Processing and Response Generation
def get_answer(query, vectorstore, ix, tokenizer, retriever, model):
    # Semantic search using FAISS
    retriever_semantic = vectorstore.as_retriever()
    retrieved_docs_semantic = retriever_semantic.get_relevant_documents(query)
    
    # Textual search using Whoosh
    with ix.searcher() as searcher:
        query_parser = QueryParser("content", ix.schema)
        parsed_query = query_parser.parse(query)
        results_textual = searcher.search(parsed_query, limit=5)
    
    # Combine results
    combined_results = {}
    for result in results_textual:
        doc_id = result['id']
        combined_results[doc_id] = {'content': result['content'], 'page_number': result['page_number'], 'score': 1.0}
    
    for doc in retrieved_docs_semantic:
        doc_id = doc.metadata.get('id', str(len(combined_results)))
        if doc_id not in combined_results:
            combined_results[doc_id] = {'content': doc.page_content, 'page_number': doc.metadata.get('page_number', 'Unknown'), 'score': 0.8}
    
    # Rank and merge results
    sorted_results = sorted(combined_results.items(), key=lambda x: x[1]['score'], reverse=True)
    
    # Generate context
    context = " ".join([res['content'] for _, res in sorted_results[:3]])
    
    # Use RAG model to generate the final answer
    answer = get_rag_answer(query, tokenizer, retriever, model)
    
    # Fallback logic
    if not answer.strip() or len(answer) < 10:  # Simple check for fallback
        answer = random.choice(FALLBACK_PROMPTS)
    
    # Extract relevant excerpts with page numbers
    excerpts = [f"Document ID: {doc_id}\nPage Number: {content['page_number']}\nExcerpt: {content['content']}" for doc_id, content in combined_results.items()]
    excerpts_text = "\n\n".join(excerpts)
    
    return answer, context, excerpts_text

# 5. Conversational Chain Logic
def update_conversation_history(query, answer, context):
    conversation_history.append({
        'query': query,
        'answer': answer,
        'context': context
    })

def get_conversation_context():
    return "\n".join([f"Q: {entry['query']}\nA: {entry['answer']}" for entry in conversation_history])

# Main Execution (Enhanced Output)
if __name__ == "__main__":
    file_path = "sample.pdf"  # Replace with your actual PDF file path
    index_dir = "whoosh_index"

    docs = load_docs(file_path)
    vectorstore = create_vector_store(docs)
    create_whoosh_index(docs, index_dir)

    tokenizer, retriever, model = initialize_rag_model()

    while True:
        query = input("Enter your query (or 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        
        # Append conversation history
        conversation_context = get_conversation_context()
        full_query = f"{conversation_context}\n\nCurrent Query: {query}" if conversation_context else query
        
        answer, context, excerpts = get_answer(full_query, vectorstore, index_dir, tokenizer, retriever, model)
        
        # Update conversation history
        update_conversation_history(query, answer, context)
        
        print("\n------------------------")
        print("Query:", query)
        print("------------------------")
        print("Context:\n", context)
        print("------------------------")
        print("Answer:\n", answer)
        print("------------------------")
        print("Source Documents:\n", excerpts)
        print("------------------------\n")
