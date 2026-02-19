"""
indexer.py
Loads ALL documents from the data/ folder,
splits text into chunks, creates embeddings using HuggingFace,
and stores them into a FAISS vector store.
"""

import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from load_docs import extract_text   # <-- make sure this already handles pdf/docx


# Constants
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
FAISS_INDEX_DIR = "faiss_index"
DATA_DIR = "data"
MODEL_NAME = "all-MiniLM-L6-v2"


def chunk_text(filename, text):
    """Split text into chunks and wrap them into Document objects."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE,
                                              chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_text(text)

    docs = [
        Document(page_content=chunk, metadata={"source": filename, "chunk": i})
        for i, chunk in enumerate(chunks)
    ]
    print(f"✂️ Created {len(docs)} chunks from {filename}")
    return docs


def build_faiss_index(docs, persist_dir=FAISS_INDEX_DIR):
    """Create and save FAISS index using HuggingFace embeddings."""
    embedding = HuggingFaceEmbeddings(model_name=MODEL_NAME)

    vectorstore = FAISS.from_texts(
        [doc.page_content for doc in docs],
        embedding=embedding,
        metadatas=[doc.metadata for doc in docs]
    )

    os.makedirs(persist_dir, exist_ok=True)
    vectorstore.save_local(persist_dir)
    print(f"✅ Saved FAISS index to: {persist_dir}")


def load_faiss_index(persist_dir=FAISS_INDEX_DIR):
    embedding = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    vectorstore = FAISS.load_local(persist_dir, embedding,
                                   allow_dangerous_deserialization=True)
    print("✅ Loaded FAISS index successfully!")
    return vectorstore


if __name__ == "__main__":
    all_docs = []

    print("📁 Loading all documents from /data folder ...")

    for filename in os.listdir(DATA_DIR):
        if filename.startswith("~$"):
            print(f"⚠️ Skipping temporary file: {filename}")
            continue
        file_path = os.path.join(DATA_DIR, filename)

        if not filename.lower().endswith((".pdf", ".docx", ".txt")):
            continue  # skip unsupported files

        print(f"📂 Processing: {filename}")
        text = extract_text(file_path)

        docs = chunk_text(filename, text)
        all_docs.extend(docs)

    print(f"📚 Total chunks created from ALL documents: {len(all_docs)}")

    build_faiss_index(all_docs)
