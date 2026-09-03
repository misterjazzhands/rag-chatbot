import os
from pinecone import Pinecone
print("Imports successful.")
from chunker import load_and_chunk
print("Chunker imported.")
from sentence_transformers import SentenceTransformer

# Fetch credentials
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Initialize Pinecone Client
pc = Pinecone(api_key=PINECONE_API_KEY)
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "knowledge-base")
index = pc.Index(PINECONE_INDEX_NAME)

# Load the embedding model locally (downloads ~90MB on first run, then cached)
print("Loading local embedding model (all-MiniLM-L6-v2)...")
_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print("Embedding model loaded.")


def get_embeddings(texts):
    """Generate embeddings locally using the sentence-transformers model."""
    if not texts:
        return []
    embeddings = _model.encode(texts, convert_to_list=True, show_progress_bar=False)
    return embeddings


def embed_and_store(pdf_path, user_id="anonymous"):
    try:
        print("Starting Pinecone embedding pipeline...")
        chunks = load_and_chunk(pdf_path)
        print(f"Loaded {len(chunks)} chunks")

        if not chunks:
            raise ValueError("No valid text could be extracted from this PDF. It may be an image-based scan or too short (less than 40 words).")

        print("Generating embeddings locally...")
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