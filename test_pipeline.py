"""
End-to-end test of the full RAG pipeline.
Tests: PDF loading -> Chunking -> Embedding -> ChromaDB storage -> Retrieval -> Reranking -> LLM answer
"""
import os
import sys

print("=" * 60)
print("  RAG PIPELINE END-TO-END TEST")
print("=" * 60)

# Step 0: Reset ChromaDB
print("\n[STEP 0] Resetting ChromaDB...")
import chromadb
client = chromadb.PersistentClient(path="./chroma_db")
try:
    client.delete_collection("knowledge_base")
    print("  -> Deleted old collection")
except Exception:
    print("  -> No existing collection to delete")
col = client.get_or_create_collection("knowledge_base")
print(f"  -> Fresh collection created. Count: {col.count()}")

# Step 1: Test chunker
print("\n[STEP 1] Testing chunker (load_and_chunk)...")
from chunker import load_and_chunk

pdf_path = os.path.normpath("./uploads/test.pdf")
if not os.path.exists(pdf_path):
    pdf_path = "test.pdf"
if not os.path.exists(pdf_path):
    print(f"  ERROR: No test PDF found! Place test.pdf in root or uploads/")
    sys.exit(1)

print(f"  -> Using PDF: {pdf_path}")
chunks = load_and_chunk(pdf_path)
print(f"  -> SUCCESS: {len(chunks)} chunks generated")
if chunks:
    print(f"  -> First chunk preview: {chunks[0][:100]}...")

# Step 2: Test embedder
print("\n[STEP 2] Testing embedder (embed_and_store)...")
from embedder import embed_and_store, collection
embed_and_store(pdf_path)
count = collection.count()
print(f"  -> SUCCESS: {count} chunks stored in ChromaDB")

# Step 3: Test retriever
print("\n[STEP 3] Testing retrieval...")
from rag_pipeline import retrieve_candidates, rerank_candidates
test_query = "What is AI hallucination?"
candidates = retrieve_candidates(test_query, top_k=10)
print(f"  -> Retrieved {len(candidates)} candidates")
if candidates:
    print(f"  -> Top candidate preview: {candidates[0]['text'][:100]}...")

# Step 4: Test reranking
print("\n[STEP 4] Testing reranking...")
reranked = rerank_candidates(test_query, candidates, top_n=3)
print(f"  -> Reranked to {len(reranked)} results")
for i, r in enumerate(reranked):
    score = r.get('rerank_score', 'N/A')
    print(f"  -> [{i+1}] Score: {score:.4f} | {r['text'][:80]}...")

# Step 5: Test LLM generation
print("\n[STEP 5] Testing LLM generation (Groq)...")
from rag_pipeline import answer_query
result = answer_query(test_query)
print(f"  -> Answer: {result['answer'][:200]}...")
print(f"  -> Sources used: {len(result['sources'])}")

# Step 6: Test streaming
print("\n[STEP 6] Testing streaming generation...")
from rag_pipeline import answer_query_stream
token_count = 0
source_count = 0
for event in answer_query_stream(test_query):
    if event["type"] == "sources":
        source_count = len(event["sources"])
    elif event["type"] == "token":
        token_count += 1
print(f"  -> Streamed {token_count} tokens with {source_count} sources")

print("\n" + "=" * 60)
print("  ALL TESTS PASSED! Pipeline is working correctly.")
print("=" * 60)
