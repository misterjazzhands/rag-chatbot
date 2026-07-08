import os
import sys
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from llm_router import (
    DEFAULT_MODEL,
    chat_completion,
    chat_completion_stream,
    simple_completion,
)

# Load environment variables
load_dotenv()

# =========================
# INITIALIZE MODELS & CLIENTS
# =========================

print("\n--- Initializing RAG Pipeline ---")

# 2. Load Embedding Model (all-MiniLM-L6-v2)
print("Loading local embedding model (all-MiniLM-L6-v2)...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# 3. Load Reranker Model (Cross-Encoder)
# print("Loading local reranker model (cross-encoder/ms-marco-MiniLM-L-6-v2)...")
# try:
#     reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
# except Exception as e:
#     print(f"Error loading CrossEncoder. Falling back to default retrieval. Details: {e}")
#     reranker_model = None
reranker_model = None
print("Reranker disabled to reduce memory usage.")

# 4. Connect to ChromaDB
print("Connecting to ChromaDB...")
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection("knowledge_base")

print("RAG Pipeline ready.\n")


# =========================
# CORE RAG FUNCTIONS
# =========================

def retrieve_candidates(query, top_k=10, user_id="anonymous"):
    """Retrieve top_k semantic candidate chunks from ChromaDB."""
    # Normalize query embedding to match indexed vectors
    query_embedding = embed_model.encode(query, normalize_embeddings=True)
    
    # Query ChromaDB collection
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k,
        where={"user_id": user_id}
    )
    
    # Extract documents and metadata
    if results and "documents" in results and results["documents"]:
        documents = results["documents"][0]
        metadatas = results["metadatas"][0] if "metadatas" in results and results["metadatas"] else [{}] * len(documents)
        ids = results["ids"][0] if "ids" in results and results["ids"] else []
        return [{"id": ids[i], "text": doc, "metadata": metadatas[i]} for i, doc in enumerate(documents)]
    
    return []


def rerank_candidates(query, candidates, top_n=3):
    """Use CrossEncoder to rerank candidate chunks for high-precision retrieval."""
    if not candidates:
        return []
    
    # If reranker model isn't loaded, fallback to top_n from semantic retrieval
    if not reranker_model:
        return candidates[:top_n]
    
    # Extract texts from candidates
    texts = [cand["text"] for cand in candidates]
    
    # Create query-document pairs
    pairs = [(query, text) for text in texts]
    
    # Predict relevance scores
    scores = reranker_model.predict(pairs)
    
    # Combine scores with candidates
    scored_candidates = []
    for i, score in enumerate(scores):
        cand = candidates[i].copy()
        cand["rerank_score"] = float(score)
        scored_candidates.append(cand)
    
    # Sort by score descending
    scored_candidates.sort(reverse=True, key=lambda x: x["rerank_score"])
    
    # Keep only the top_n results
    return scored_candidates[:top_n]


def safe_print(message):
    """Utility to print unicode messages safely on all OS platforms."""
    try:
        print(message)
    except UnicodeEncodeError:
        encoding = sys.stdout.encoding or 'utf-8'
        print(message.encode(encoding, errors='replace').decode(encoding))


# =========================
# CONVERSATIONAL RAG ENGINE
# =========================

def reformulate_query(query, chat_history, model_id=DEFAULT_MODEL, user_api_key=None):
    """Rewrite follow-up questions to be standalone queries based on chat history."""
    if not chat_history:
        return query
        
    prompt = (
        "Given the following conversation and a follow up question, rephrase the "
        "follow up question to be a standalone question, in its original language.\n"
        "Return ONLY the standalone question, with no conversational filler or explanation.\n\n"
        "Chat History:\n"
    )
    for msg in chat_history[-4:]:
        prompt += f"{msg['role'].capitalize()}: {msg['content']}\n"
        
    prompt += f"\nFollow Up Input: {query}\nStandalone question:"
    
    try:
        reformulated = simple_completion(
            prompt,
            model_id=model_id,
            user_api_key=user_api_key,
            temperature=0.0,
            max_tokens=256,
        )
        if reformulated.startswith('"') and reformulated.endswith('"'):
            reformulated = reformulated[1:-1]
        print(f"[REFORMULATION] '{query}' -> '{reformulated}'")
        return reformulated
    except Exception as e:
        print(f"Query reformulation failed: {e}")
        return query


def local_fallback_answer(query, candidates, top_n=3):
    """Term-frequency fallback when LLM APIs are unavailable."""
    terms = set(query.lower().split())
    scored = []
    for cand in candidates:
        text = cand.get("text", "").lower()
        score = sum(1 for t in terms if t in text)
        scored.append((score, cand))
    scored.sort(key=lambda x: x[0], reverse=True)
    top = [c for s, c in scored[:top_n] if s > 0]
    if not top:
        return "I cannot find relevant information in the uploaded documents.", []
    excerpt = "\n\n---\n\n".join(c["text"] for c in top)
    return (
        f"*Local extraction mode (LLM unavailable)*\n\n"
        f"Based on keyword matching, here are the most relevant excerpts:\n\n{excerpt}"
    ), top


def answer_query(
    query,
    chat_history=None,
    top_k=10,
    top_n=3,
    user_id="anonymous",
    model_id=DEFAULT_MODEL,
    user_api_key=None,
):
    """
    Main pipeline execution:
    1. Reformulate query
    2. Retrieve candidates
    3. Rerank candidates
    4. Construct prompt with context and history
    5. Fetch LLM answer
    """
    if chat_history is None:
        chat_history = []

    search_query = reformulate_query(query, chat_history, model_id, user_api_key)
    
    # Step 2: Retrieve candidates
    candidates = retrieve_candidates(search_query, top_k=top_k, user_id=user_id)
    
    # Step 3: Rerank
    relevant_chunks = rerank_candidates(search_query, candidates, top_n=top_n)
    
    # Combine context
    context_str = "\n\n".join([f"--- Context Segment ---\n{chunk['text']}" for chunk in relevant_chunks])
    
    # Step 4: Construct System Prompt
    system_prompt = (
        "You are an expert AI Assistant designed to answer questions strictly based on the provided document contexts.\n\n"
        "RULES FOR OPERATION:\n"
        "1. Answer the user's question using ONLY the provided context segments.\n"
        "2. Ground your answer with facts from the context. You MUST NOT make up facts, dates, names, or numbers.\n"
        "3. Quote directly from the text where appropriate to ensure maximum accuracy.\n"
        "4. If the answer is not present in the context, say 'I cannot find the answer in the uploaded document.' and do not extrapolate.\n"
        "5. Be professional, clear, and structured in your responses.\n\n"
        f"CONTEXT SEGMENTS:\n{context_str}"
    )
    
    # Step 5: Build Messages Payload (incorporating conversational history)
    messages = [{"role": "system", "content": system_prompt}]
    
    # Add chat history (limit to last 6 messages to keep context window manageable)
    for msg in chat_history[-6:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
        
    # Append the current query
    messages.append({"role": "user", "content": query})
    
    try:
        response = chat_completion(
            messages=messages,
            model_id=model_id,
            user_api_key=user_api_key,
            temperature=0.2,
            max_tokens=1024,
        )
        answer = response.choices[0].message.content
    except Exception as e:
        print(f"LLM call failed, using local fallback: {e}")
        answer, fallback_sources = local_fallback_answer(search_query, candidates, top_n)
        if fallback_sources:
            relevant_chunks = fallback_sources
        
    return {
        "answer": answer,
        "sources": relevant_chunks,
        "model": model_id,
    }


def answer_query_stream(
    query,
    chat_history=None,
    top_k=10,
    top_n=3,
    user_id="anonymous",
    model_id=DEFAULT_MODEL,
    user_api_key=None,
):
    """
    Main pipeline execution with streaming support:
    1. Reformulate query
    2. Retrieve candidates
    3. Rerank candidates
    4. Yield source citations immediately
    5. Stream LLM tokens
    """
    if chat_history is None:
        chat_history = []

    search_query = reformulate_query(query, chat_history, model_id, user_api_key)
    
    # Step 2: Retrieve candidates
    candidates = retrieve_candidates(search_query, top_k=top_k, user_id=user_id)
    
    # Step 3: Rerank
    relevant_chunks = rerank_candidates(search_query, candidates, top_n=top_n)
    
    # Format sources for frontend
    sources = []
    for src in relevant_chunks:
        sources.append({
            "id": src.get("id"),
            "text": src.get("text"),
            "source": src.get("metadata", {}).get("source", "Unknown"),
            "score": src.get("rerank_score", 0.0)
        })
        
    yield {"type": "sources", "sources": sources, "model": model_id}
    
    # Combine context for prompt
    context_str = "\n\n".join([f"--- Context Segment ---\n{chunk['text']}" for chunk in relevant_chunks])
    
    # Step 4: Construct System Prompt
    system_prompt = (
        "You are an expert AI Assistant designed to answer questions strictly based on the provided document contexts.\n\n"
        "RULES FOR OPERATION:\n"
        "1. Answer the user's question using ONLY the provided context segments.\n"
        "2. Ground your answer with facts from the context. You MUST NOT make up facts, dates, names, or numbers.\n"
        "3. Quote directly from the text where appropriate to ensure maximum accuracy.\n"
        "4. If the answer is not present in the context, say 'I cannot find the answer in the uploaded document.' and do not extrapolate.\n"
        "5. Be professional, clear, and structured in your responses.\n\n"
        f"CONTEXT SEGMENTS:\n{context_str}"
    )
    
    # Step 5: Build Messages Payload
    messages = [{"role": "system", "content": system_prompt}]
    for msg in chat_history[-6:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": query})
    
    try:
        for token in chat_completion_stream(
            messages=messages,
            model_id=model_id,
            user_api_key=user_api_key,
            temperature=0.2,
            max_tokens=1024,
        ):
            yield {"type": "token", "token": token}
    except Exception as e:
        print(f"LLM stream failed, using local fallback: {e}")
        answer, fallback_sources = local_fallback_answer(search_query, candidates, top_n)
        if fallback_sources:
            fb_sources = []
            for src in fallback_sources:
                fb_sources.append({
                    "id": src.get("id"),
                    "text": src.get("text"),
                    "source": src.get("metadata", {}).get("source", "Unknown"),
                    "score": src.get("rerank_score", 0.0),
                })
            yield {"type": "sources", "sources": fb_sources, "model": "local-fallback"}
        yield {"type": "token", "token": answer}


# =========================
# TERMINAL CHAT SIMULATOR
# =========================

if __name__ == "__main__":
    print("\n=========================================")
    print("   CONVERSATIONAL RAG CHATBOT SIMULATOR  ")
    print("=========================================")
    
    history = []
    
    print("\nChatbot ready! Ask questions about your document (type 'quit' to exit).\n")
    
    while True:
        try:
            user_input = input("You: ")
            if user_input.strip().lower() == "quit":
                print("Goodbye!")
                break
                
            if not user_input.strip():
                continue
                
            # Run the RAG pipeline
            result = answer_query(user_input, chat_history=history)
            
            # Print response
            print("\n" + "="*50)
            safe_print(f"Bot: {result['answer']}")
            print("="*50)
            
            # Print Sources used
            print("\nSources Used:")
            for idx, src in enumerate(result['sources']):
                score_str = f"Score: {src['rerank_score']:.4f}" if 'rerank_score' in src else "Similarity match"
                safe_print(f"[{idx+1}] Chunk: {src['id']} ({score_str})")
                # print a small snippet of the source text
                safe_print(f"   Snippet: {src['text'][:120]}...")
            print("\n" + "-"*50 + "\n")
            
            # Update history
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": result['answer']})
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")
