import os
from embedder import get_embeddings, index

def test_embed():
    print("Testing Pinecone Upsert & HuggingFace Embeddings...")
    
    chunks = ["This is a test chunk about Artificial Intelligence.", "Here is another test chunk."]
    
    print("Requesting embeddings from Hugging Face API...")
    try:
        embeddings = get_embeddings(chunks)
        print(f"Embedding complete. Retrieved {len(embeddings)} embeddings of dimension {len(embeddings[0]) if embeddings else 0}")
    except Exception as e:
        print(f"HuggingFace API failed: {e}")
        return

    # Format payloads for Pinecone upsertion
    upsert_data = []
    user_id = "test_user_123"
    for i, chunk in enumerate(chunks):
        safe_id = f"test_doc_chunk_{i}"
        upsert_data.append({
            "id": safe_id,
            "values": embeddings[i],
            "metadata": {
                "text": chunk,
                "source": "test_doc.txt",
                "user_id": user_id
            }
        })

    print("Upserting vectors into cloud Pinecone instance...")
    try:
        index.upsert(vectors=upsert_data)
        print("Stored safely in Pinecone.")
    except Exception as e:
        print(f"Pinecone Upsert failed: {e}")

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    test_embed()
