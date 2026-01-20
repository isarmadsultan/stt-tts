"""Simple runner for weaviate indexing"""
import asyncio
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.weaviate_optimized import OptimizedWeaviateIndexer

async def main():
    print("Starting optimized Weaviate indexing...")
    
    indexer = OptimizedWeaviateIndexer(
        collection_name="Document",
        use_gpu_embeddings=True
    )
    
    await indexer.run_full_index()
    
    print("Indexing complete!")

if __name__ == "__main__":
    asyncio.run(main())
