import fitz  # PyMuPDF for PDF text extraction
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from whoosh.index import create_in, open_dir
from whoosh.fields import Schema, TEXT
from whoosh.qparser import QueryParser
from transformers import AutoModelForCausalLM, AutoTokenizer

# Initialize SentenceTransformer and GPT-Neo model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
gpt_model = AutoModelForCausalLM.from_pretrained('EleutherAI/gpt-neo-125M')
gpt_tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neo-125M')

# Define paths
index_dir = "indexdir"
pdf_path = "Sample.pdf"

# 1. Extract Text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    pdf_document = fitz.open(pdf_path)
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        text += page.get_text()
    return text

# 2. Create and Populate Whoosh Index
def create_whoosh_index(documents):
    schema = Schema(id=TEXT(stored=True), content=TEXT)
    if not os.path.exists(index_dir):
        os.mkdir(index_dir)
    ix = create_in(index_dir, schema)
    writer = ix.writer()
    for i, doc in enumerate(documents):
        writer.add_document(id=str(i), content=doc)
    writer.commit()

# 3. Encode Text and Create FAISS Index
def create_faiss_index(texts):
    embeddings = model.encode(texts, convert_to_numpy=True)
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    return index

# 4. Perform Textual Search with Whoosh
def search_whoosh(query_string):
    ix = open_dir(index_dir)
    with ix.searcher() as searcher:
        query = QueryParser("content", ix.schema).parse(query_string)
        results = searcher.search(query)
        return [hit['content'] for hit in results]

# 5. Perform Semantic Search with FAISS
def search_faiss(query, faiss_index, paragraphs):
    query_embedding = model.encode([query])
    distances, indices = faiss_index.search(query_embedding, k=5)
    return [paragraphs[i] for i in indices[0]]

# 6. Generate Answer with GPT-Neo
def generate_answer(context):
    # Tokenize input text
    input_ids = gpt_tokenizer.encode(context, return_tensors='pt', truncation=True, max_length=2048)
    # Generate output
    output = gpt_model.generate(input_ids, max_new_tokens=200)
    return gpt_tokenizer.decode(output[0], skip_special_tokens=True)

# Main Workflow
if __name__ == "__main__":
    text = extract_text_from_pdf(pdf_path)
    paragraphs = text.split("\n\n")  # Adjust based on PDF text structure

    # Create Whoosh index
    create_whoosh_index(paragraphs)

    # Create FAISS index
    faiss_index = create_faiss_index(paragraphs)

    # Search Query
    query = "what is networks?"
    
    # Textual Search with Whoosh
    relevant_paragraphs = search_whoosh(query)

    # Semantic Search with FAISS
    top_paragraphs = search_faiss(query, faiss_index, paragraphs)

    # Combine results (you might need to refine this step)
    final_paragraph = top_paragraphs[0] if top_paragraphs else relevant_paragraphs[0] if relevant_paragraphs else ""

    # Generate Answer
    answer = generate_answer(final_paragraph)
    print(answer)
