import os
import time
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
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "knowledge-base")
index = pc.Index(PINECONE_INDEX_NAME)

# HF Inference API for sentence-transformers (no local model, no RAM overhead)
HF_API_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
HF_HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}


def get_embeddings(texts, retries=5):
    """Fetch embeddings from Hugging Face Free Inference API with retries and batching."""
    if not texts:
        return []

    all_embeddings = []
    batch_size = 20

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_embeddings = _fetch_batch_with_retry(batch, retries)
        all_embeddings.extend(batch_embeddings)

    return all_embeddings


def _fetch_batch_with_retry(batch, retries):
    for attempt in range(retries):
        try:
            response = requests.post(
                HF_API_URL,
                headers=HF_HEADERS,
                json={"inputs": batch, "options": {"wait_for_model": True}},
                timeout=60
            )

            if response.status_code == 503:
                error_data = response.json()
                wait_time = error_data.get("estimated_time", 20)
                print(f"HF model loading, waiting {wait_time}s before retry...")
                time.sleep(wait_time)
                continue

            response.raise_for_status()

            result = response.json()
            if isinstance(result, dict) and "error" in result:
                raise ValueError(f"Hugging Face API Error: {result['error']}")

            return result

        except Exception as e:
            print(f"Attempt {attempt + 1}/{retries} failed: {e}")
            if attempt == retries - 1:
                raise ValueError(
                    f"Hugging Face Free API failed after {retries} attempts. "
                    f"Error: {e}. Ensure HF_TOKEN is set and has Inference permissions."
                )
            time.sleep(3 * (attempt + 1))  # Exponential-ish backoff


def embed_and_store(pdf_path, user_id="anonymous"):
    try:
        print("Starting Pinecone embedding pipeline...")
        chunks = load_and_chunk(pdf_path)
        print(f"Loaded {len(chunks)} chunks")

        if not chunks:
            raise ValueError(
                "No valid text could be extracted from this PDF. "
                "It may be an image-based scan or too short (less than 40 words)."
            )

        print("Requesting embeddings from Hugging Face API...")
        embeddings = get_embeddings(chunks)
        print("Embedding complete.")

        # Format payloads for Pinecone upsertion
        upsert_data = []
        for i, chunk in enumerate(chunks):
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

    except Exception as e:
        print(f"Error in Pinecone embedding pipeline: {e}")
        raise
    finally:
        # Local Cleanup
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
            print(f"[CLEANUP] Deleted temporary file: {pdf_path}")