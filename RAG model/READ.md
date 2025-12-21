# RAG-based Question Answering System using Ollama

This project implements a Retrieval-Augmented Generation (RAG) pipeline that allows users to ask questions about Machine Learning using content from a PDF document.

The system retrieves relevant document chunks using embeddings and ChromaDB, then generates answers using the LLaMA3 model via Ollama.

---

## Features

- PDF document ingestion
- Text chunking using LangChain text splitters
- Sentence embeddings using MiniLM
- Vector storage using ChromaDB (persistent)
- Question answering using Ollama (LLaMA3)
- Answers restricted strictly to document context
- Exactly 4-line structured responses

---

## Tech Stack

- Python
- Ollama (LLaMA3)
- PyPDF2
- Sentence Transformers
- ChromaDB
- LangChain Text Splitters

---

## Project Structure

.
├── rag_new.py  
├── requirements.txt  
├── README.md  
├── chroma_db/  
└── simplified-machine-learning-dr-pooja-sharma.pdf  

---

## Installation

1. Install Python dependencies:

pip install -r requirements.txt

# To run program:
python rag_new.py

# Sample question:
what is unsupervised learning?
