import os
import json
from tqdm import tqdm
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

# ===== Step 1: Load environment variables =====
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# ===== Step 2: Load JSON chunks =====
json_path = os.path.join(os.path.dirname(__file__), "split_chunks.json")

with open(json_path, "r", encoding="utf-8") as f:
    chunks_data = json.load(f)

docs = [Document(page_content=chunk["content"]) for chunk in chunks_data]
print(f"📄 Loaded {len(docs)} chunks (no metadata) from '{json_path}'")

# ===== Step 3: Initialize the OpenAI embedding model =====
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=openai_api_key
)

# ===== Step 4: Create / Connect to Chroma database =====
persist_dir = os.path.join(os.path.dirname(__file__), "chroma_db_openai")
db = Chroma(persist_directory=persist_dir, embedding_function=embeddings)

# ===== Step 5: Generate and store embeddings =====
print("\n🚀 Generating embeddings and storing in Chroma...\n")
for i in tqdm(range(0, len(docs), 50), desc="Embedding chunks", unit="batch"):
    batch = docs[i:i + 50]
    db.add_documents(batch)

# ===== Step 6: Persist Chroma DB =====
db.persist()
print(f"\n✅ Stored {len(docs)} chunks in Chroma vector database at '{persist_dir}'")
