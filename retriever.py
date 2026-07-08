from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb

# =========================
# LOAD MODELS
# =========================

print("Loading embedding model...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

print("Loading reranker model...")
reranker_model = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

print("Models loaded successfully.")

# =========================
# CONNECT TO CHROMADB
# =========================

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(
    "knowledge_base"
)

print("Connected to ChromaDB.")

# =========================
# RETRIEVAL FUNCTION
# =========================

def retrieve(query, top_k=10):

    print(f"\nUser query: {query}")

    # Convert query to embedding
    query_embedding = embed_model.encode(query)

    # Search vector database
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k
    )

    return results

# =========================
# RERANKING FUNCTION
# =========================

def rerank(query, documents, top_n=3):

    print("\nReranking results...")

    # Create query-document pairs
    pairs = [(query, doc) for doc in documents]

    # Predict relevance scores
    scores = reranker_model.predict(pairs)

    # Combine scores with documents
    scored_docs = list(zip(scores, documents))

    # Sort by score descending
    scored_docs.sort(
        reverse=True,
        key=lambda x: x[0]
    )

    # Keep only top_n results
    top_docs = scored_docs[:top_n]

    return top_docs

# =========================
# MAIN PROGRAM
# =========================

if __name__ == "__main__":

    user_query = input("Ask a question: ")

    # Step 1: Retrieve candidate chunks
    results = retrieve(user_query, top_k=10)

    documents = results["documents"][0]

    # Step 2: Rerank retrieved chunks
    reranked_results = rerank(
        user_query,
        documents,
        top_n=3
    )

    # Step 3: Display results
    print("\nTop reranked chunks:\n")

    for i, (score, doc) in enumerate(reranked_results):

        print(f"Result {i + 1}")
        print(f"Relevance Score: {score:.4f}\n")

        print(doc)

        print("\n" + "=" * 70 + "\n")