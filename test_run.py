# test_run.py
"""
Test script to check if indexer.py and load_docs.py work properly.
"""

from load_docs import extract_text as load_input_file

from indexer import chunk_text, build_faiss_index

print("👋 test_run.py started running!")

# Change file name as per what’s inside your 'data' folder
file_path = "data/datascience_tutorial.pdf"  # or data/sample.txt

print(f"➡️ Loading file: {file_path}")
filename, text = load_input_file(file_path)

if not text.strip():
    print("⚠️ No text extracted! Check if your file has readable text.")
else:
    print("✅ Text successfully extracted!")

print("➡️ Splitting text into chunks...")
docs = chunk_text(filename, text)
print(f"✅ Created {len(docs)} chunks.")

print("➡️ Building FAISS index...")
build_faiss_index(docs)

print("🎯 Done! FAISS index successfully built and saved.")
