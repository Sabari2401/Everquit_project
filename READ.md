# PDF Question Answering using RAG

## About the Project

This project is a simple **PDF Question Answering system** built using the **Retrieval-Augmented Generation (RAG)** approach. It allows users to ask questions and get answers directly from the content of a PDF file.

---

## What This Project Does

* Reads text from a PDF file
* Splits the text into small chunks
* Converts chunks into embeddings
* Stores embeddings in ChromaDB
* Takes a question from the user
* Retrieves relevant chunks
* Uses an LLM to generate an answer

---

## Technologies Used

* Python
* PyPDF2
* Sentence Transformers
* ChromaDB
* TinyLlama (LLM)
* LangChain Text Splitter

---

## How to Run

1. Install required libraries

```bash
pip install -r requirements.txt
```

2. Update the PDF file path in the code

3. Run the program

```bash
python app.py
```

---

## Usage

* Run the script
* Enter your question in the terminal
* Get answers based on the PDF content

---

## Example

```
Enter your question: What are the acts mentioned in the document?
```

---

## Purpose

This project is created for learning and demonstrating **GenAI and RAG concepts** using open-source tools.
