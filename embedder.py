import os
import requests
from pinecone import Pinecone
print("Imports successful.")
from chunker import load_and_chunk
print("Chunker imported.")

# Fetch credentials
HF_TOKEN = os.getenv("HF_TOKEN")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Initialize Pinecone Client
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("knowledge-base")

API_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

def get_embeddings(texts):
    """Fetch embeddings from Hugging Face Free Inference API."""
    try:
        response = requests.post(API_URL, headers=headers, json={"inputs": texts}, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching embeddings from Hugging Face API: {e}")
        raise

def embed_and_store(pdf_path, user_id="anonymous"):
    try:
        print("Starting Pinecone embedding pipeline...")
        chunks = load_and_chunk(pdf_path)
        print(f"Loaded {len(chunks)} chunks")

        if not chunks:
            return

        print("Requesting embeddings from Hugging Face API...")
        embeddings = get_embeddings(chunks)
        print("Embedding complete.")
        
        # Format payloads for Pinecone upsertion
        upsert_data = []
        for i, chunk in enumerate(chunks):
            # Clean non-ascii strings safely for metadata fields
            safe_id = f"{os.path.basename(pdf_path)}_chunk_{i}".replace(" ", "_")
            upsert_data.append({
                "id": safe_id,
                "values": embeddings[i],
                "metadata": {
                    "text": chunk,
                    "source": os.path.basename(pdf_path),
                    "user_id": user_id
                }
            })

        # Upsert vectors directly into cloud instance
        print("Upserting vectors into cloud Pinecone instance...")
        index.upsert(vectors=upsert_data)
        print("Stored safely in Pinecone.")

        # Local Cleanup
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
            print(f"[CLEANUP] Deleted temporary file: {pdf_path}")

    except Exception as e:
        print(f"Error in Pinecone embedding pipeline: {e}")
        raise