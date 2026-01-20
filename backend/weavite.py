import os
import json
from tqdm import tqdm
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_weaviate.vectorstores import WeaviateVectorStore
import weaviate
from weaviate.classes.init import AdditionalConfig, Timeout
from weaviate.classes.config import Configure, VectorDistances, DataType, Property

# ===========================
# Step 1: Load environment variables
# ===========================
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
weaviate_url = os.getenv("WEAVIATE_URL", "http://localhost:9000")

# ===========================
# Step 2: Load JSON chunks
# ===========================
json_path = os.path.join(os.path.dirname(__file__), "split_chunks.json")

with open(json_path, "r", encoding="utf-8") as f:
    chunks_data = json.load(f)

docs = [
    Document(
        page_content=chunk["content"],
        metadata=chunk.get("metadata", {})
    ) 
    for chunk in chunks_data
]
print(f"📄 Loaded {len(docs)} chunks from '{json_path}'")

# ===========================
# Step 3: Initialize OpenAI Embeddings
# ===========================
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=openai_api_key,
)

# ===========================
# Step 4: Connect to Local Weaviate
# ===========================
client = weaviate.connect_to_local(
    host="localhost",
    port=9000,
    grpc_port=50051,
    additional_config=AdditionalConfig(
        timeout=Timeout(init=30, query=60, insert=120)
    )
)

print("🔗 Connected to LOCAL Weaviate!")

# ===========================
# Step 5: Create GPU-optimized schema
# ===========================
CLASS_NAME = "Document"

existing = client.collections.list_all()

if CLASS_NAME in existing:
    print(f"🗑️ Deleting existing collection '{CLASS_NAME}'...")
    client.collections.delete(CLASS_NAME)

print(f"🛠 Creating GPU-optimized schema '{CLASS_NAME}'...")

client.collections.create(
    name=CLASS_NAME,
    properties=[
        Property(name="content", data_type=DataType.TEXT),
        Property(name="source", data_type=DataType.TEXT),
    ],
    # GPU-optimized HNSW configuration
    vector_index_config=Configure.VectorIndex.hnsw(
        distance_metric=VectorDistances.COSINE,
        ef_construction=128,          # Higher = better quality, slower indexing
        ef=-1,                         # Dynamic search (auto-tuned)
        max_connections=64,            # Higher = better recall, more memory
        dynamic_ef_min=100,            # Minimum ef for dynamic search
        dynamic_ef_max=500,            # Maximum ef for dynamic search
        dynamic_ef_factor=8,           # Multiplier for dynamic ef
        vector_cache_max_objects=100000,  # Cache vectors in memory
        flat_search_cutoff=40000,     # Use flat search for small datasets
        quantizer=None                 # No quantization for maximum quality
    )
)

print(f"✅ GPU-optimized collection '{CLASS_NAME}' created!")

# ===========================
# Step 6: Initialize VectorStore
# ===========================
vector_store = WeaviateVectorStore(
    client=client,
    index_name=CLASS_NAME,
    text_key="content",
    embedding=embeddings,
)

# ===========================
# Step 7: Insert chunks in batches
# ===========================
print("\n🚀 Generating embeddings and inserting into Weaviate...\n")

batch_size = 100  # Larger batches for GPU
for i in tqdm(range(0, len(docs), batch_size), desc="Indexing batches", unit="batch"):
    batch_docs = docs[i:i + batch_size]
    vector_store.add_documents(batch_docs)

print(f"\n✅ Successfully stored {len(docs)} chunks in GPU-optimized Weaviate!")

# ===========================
# Step 8: Close connection
# ===========================
client.close()
print("🔌 Closed Weaviate connection.")