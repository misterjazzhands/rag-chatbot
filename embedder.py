import os
import requests
import chromadb
print("Imports successful.")
from chunker import load_and_chunk
print("Chunker imported.")

# Fetch your token from environment variables
HF_TOKEN = os.getenv("HF_TOKEN")
API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

def get_embedding(texts):
    """Fetch embeddings from Hugging Face Free Inference API."""
    if not HF_TOKEN:
        print("WARNING: HF_TOKEN environment variable not set.")
        # Fallback to a zero-vector array just to keep things from crashing during testing
        return [[0.0] * 384 for _ in texts]
        
    try:
        response = requests.post(API_URL, headers=headers, json={"inputs": texts})
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching embeddings from Hugging Face API: {e}")
        raise

# Create a ChromaDB client and collection
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("knowledge_base")
print("ChromaDB client and collection ready.")

def embed_and_store(pdf_path, user_id="anonymous"):
    try:
        print("Starting embedding pipeline...")

        # Get chunks from pdf
        chunks = load_and_chunk(pdf_path)
        print(f"Loaded {len(chunks)} chunks")

        if not chunks:
            return

        print("Requesting embeddings from Hugging Face API...")
        embeddings = get_embedding(chunks)
        print("Embedding complete.")
        
        # Store in ChromaDB
        ids = [f"{pdf_path}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [{"source": pdf_path, "user_id": user_id} for _ in chunks]

        try:
            initial_count = collection.count()
            collection.delete(where={"$and": [{"source": pdf_path}, {"user_id": user_id}]})
            cleared_count = initial_count - collection.count()
            if cleared_count > 0:
                print(f"Cleared {cleared_count} existing chunks for source '{pdf_path}'.")
        except Exception as e:
            print(f"No existing chunks to clear or delete error: {e}")

        collection.add(
            documents = chunks,
            embeddings = embeddings,
            ids = ids,
            metadatas = metadatas
        )
        print(f"Stored {len(chunks)} chunks in ChromaDB.")

        try:
            if os.path.exists(pdf_path):
                os.remove(pdf_path)
                print(f"[CLEANUP] Deleted temporary PDF file: {pdf_path}")
        except Exception as cleanup_err:
            print(f"[CLEANUP ERROR] Failed to delete temporary PDF {pdf_path}: {cleanup_err}")

    except Exception as e:
        print(f"Error in embedding pipeline: {e}")
        raise

if __name__ == "__main__":
    embed_and_store("test.pdf")