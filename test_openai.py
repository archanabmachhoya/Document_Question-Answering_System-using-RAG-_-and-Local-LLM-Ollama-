# test_openai.py
# test_openai.py
from dotenv import load_dotenv
import os
from openai import OpenAI

print("👉 Script started")   # debug

load_dotenv()
print("👉 .env loaded")      # debug

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ OPENAI_API_KEY not found in .env!")

print("✅ API key loaded:", api_key[:8], "...")

client = OpenAI(api_key=api_key)

print("👉 Requesting embedding...")   # debug
resp = client.embeddings.create(model="text-embedding-3-small", input="hello")
print("✅ Embedding length:", len(resp.data[0].embedding))

