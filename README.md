# Privacy Preserving Document Q&A System (RAG + Local LLM)

## Overview
This project is a Retrieval-Augmented Generation (RAG) based Document Question-Answering system.

It:
- Loads PDF/TXT/DOCX documents
- Splits into chunks
- Creates embeddings (MiniLM)
- Stores embeddings in FAISS
- Uses Local LLM (Ollama - Phi-3)
- Runs fully on CPU
- No OpenAI API used (Privacy Preserved)

## Tech Stack
- Python
- LangChain
- FAISS
- Sentence Transformers
- Ollama
- Gradio

## How to Run

```bash
pip install -r requirements.txt
python app.py
