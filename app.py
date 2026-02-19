import os
import gradio as gr
import shutil

from indexer import chunk_text, build_faiss_index, load_faiss_index, DATA_DIR, FAISS_INDEX_DIR
from load_docs import extract_text
from qa_chain import ask_question

# Ensure data folder exists
os.makedirs(DATA_DIR, exist_ok=True)

# Global FAISS vectorstore
vectorstore = None


# --- UPLOAD FILES (Works for Gradio 6.0.2) ---
def upload_files(files):

    shutil.rmtree(DATA_DIR)
    os.makedirs(DATA_DIR, exist_ok=True)
    
    if not files:
        return "⚠️ No files uploaded."

    saved_files = []

    for file_path in files:     # file_path is a STRING
        file_name = os.path.basename(file_path)

        # Read from temp path
        with open(file_path, "rb") as f:
            file_data = f.read()

        # Save to your DATA_DIR
        dest_path = os.path.join(DATA_DIR, file_name)

        with open(dest_path, "wb") as f:
            f.write(file_data)

        saved_files.append(file_name)

    return f"✅ Saved files: {', '.join(saved_files)}"


# --- BUILD / RELOAD FAISS INDEX ---
def build_index():
    global vectorstore

    all_docs = []

    for filename in os.listdir(DATA_DIR):
        if filename.startswith("~$"):   # Skip temporary DOCX files
            continue

        file_path = os.path.join(DATA_DIR, filename)
        text = extract_text(file_path)

        if not text:
            continue

        docs = chunk_text(filename, text)
        all_docs.extend(docs)

    if not all_docs:
        return "⚠️ No documents available. Upload files first."

    build_faiss_index(all_docs)
    vectorstore = load_faiss_index()

    return f"✅ FAISS index built with {len(all_docs)} chunks."


# --- ANSWER QUESTIONS ---
def answer_question(question):
    global vectorstore

    if vectorstore is None:
        if os.path.exists(FAISS_INDEX_DIR):
            vectorstore = load_faiss_index()
        else:
            return "⚠️ Please upload documents and build the index first.", ""

    answer, docs = ask_question(vectorstore, question)

    sources = "\n\n".join([
        f"{d.metadata['source']} (chunk {d.metadata['chunk']}): {d.page_content[:200]}..."
        for d in docs
    ])

    return answer, sources


# --- BUILD GRADIO UI ---
with gr.Blocks() as demo:

    gr.Markdown("## 📚 RAG Question Answering System (FAISS + Ollama)")

    # Upload tab
    with gr.Tab("Upload Documents"):
        file_input = gr.File(label="Upload Files", file_count="multiple")
        upload_btn = gr.Button("Upload")
        upload_output = gr.Textbox(label="Status")
        upload_btn.click(upload_files, inputs=file_input, outputs=upload_output)

    # Index tab
    with gr.Tab("Build / Load Index"):
        build_btn = gr.Button("Build FAISS Index")
        build_output = gr.Textbox(label="Status")
        build_btn.click(build_index, outputs=build_output)

    # Ask question tab
    with gr.Tab("Ask Questions"):
        question_input = gr.Textbox(label="Enter your question")
        ask_btn = gr.Button("Get Answer")
        answer_output = gr.Textbox(label="Answer")
        sources_output = gr.Textbox(label="Sources")
        ask_btn.click(
            answer_question,
            inputs=question_input,
            outputs=[answer_output, sources_output]
        )

demo.launch()

