
import PyPDF2
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import logging
from transformers import logging as hf_logging

path = r"C:\Users\rsaba\Downloads\fastapi-demo-products-get-post1\simplified-machine-learning-dr-pooja-sharma.pdf"

with open(path, "rb") as f:
    reader = PyPDF2.PdfReader(f)
    content = ""

    for page in reader.pages:
        content += page.extract_text()

text_splitter=RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20
)

chunk=text_splitter.split_text(content)

#print(len(chunk))

model = SentenceTransformer("all-MiniLM-L6-v2")
vector = model.encode(chunk)
#print(vector.shape)

import chromadb
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="my_vectors")
collections = client.list_collections()

#for c in collections:
#    print(c.name)
collection.add(
    documents=chunk,
    metadatas=[{"chunk_id": i} for i in range(len(chunk))],
    embeddings=vector.tolist(),
    ids=[f"id_{i}" for i in range(len(chunk))]
)
all_data = collection.get(include=['embeddings', 'metadatas', 'documents'])
#for i in range(5):
#    print(all_data['ids'][i])
#    print("Document:", all_data['documents'][i])
#    print("Metadata:", all_data['metadatas'][i])
#    print("Embedding length:", len(all_data['embeddings'][i]))
#    print("-"*50)

query = input("Ask a question related to Machine Learning only: ")
model=SentenceTransformer('all-MiniLM-L6-v2')
query_vector=model.encode([query])[0].tolist()

result=collection.query(
    query_embeddings=query_vector,
    n_results=4
)

final_result=result['documents'][0]



model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    dtype="auto"
)


prompt = f"""
You are an expert assistant.
Answer the question using ONLY the context below.

Context:
{final_result}

Question:
{query}

Answer:
"""

inputs = tokenizer(prompt, return_tensors="pt")
output_ids = model.generate(**inputs, max_new_tokens=100)

generated_tokens = output_ids[0][inputs["input_ids"].shape[1]:]

answer = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
print(answer)
