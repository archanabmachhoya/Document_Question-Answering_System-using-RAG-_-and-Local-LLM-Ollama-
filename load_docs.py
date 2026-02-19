# ingest.py
import pdfplumber
import docx
import os


print("✅ ingest.py started loading!")

def extract_text(path):
    ext = os.path.splitext(path)[1].lower()
    print(f"📂 File extension detected: {ext}")

    if ext == ".pdf":
        text = ""
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                text += f"\n\n[PAGE {i}]\n" + (page.extract_text() or "")
        return text

    elif ext in [".docx", ".doc"]:
        document = docx.Document(path)
        return "\n".join(p.text for p in document.paragraphs if p.text.strip())

    elif ext == ".txt":
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    else:
        return "❌ Unsupported file type. Use PDF, DOCX, or TXT."

if __name__ == "__main__":
    import sys
    print("➡️ Script started!") 

    if len(sys.argv) < 2:
        print("Usage: python load_docs.py your_file.pdf")
        sys.exit()

    file_path = sys.argv[1]
    print(f"📄 Trying to open file: {file_path}")
    print(f"📁 Current working directory: {os.getcwd()}")

    text = extract_text(file_path)
    print(f"📄 Trying to open file: {file_path}")
    print(f"📂 Current working directory: {os.getcwd()}")
    print("\n✅ Extracted text (first 1000 chars):\n")
    print(text[:1000])
