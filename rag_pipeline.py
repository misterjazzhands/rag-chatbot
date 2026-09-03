import os
import sys
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from llm_router import DEFAULT_MODEL, chat_completion, chat_completion_stream, simple_completion

load_dotenv()
print("\n--- Initializing Cloud Pinecone RAG Pipeline ---")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Initialize Cloud clients
pc = Pinecone(api_key=PINECONE_API_KEY)
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "knowledge-base")
index = pc.Index(PINECONE_INDEX_NAME)

# Load local embedding model (shared with embedder.py via import at startup)
print("Loading local embedding model for query encoding...")
_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print("Query embedding model ready.")

def get_single_embedding(text):
    """Generate a single embedding locally using sentence-transformers."""
    return _model.encode([text], convert_to_list=True)[0]

def retrieve_candidates(query, top_k=5, user_id="anonymous"):
    """Query Pinecone Cloud using a metadata filter for multi-tenancy isolation."""
    query_vector = get_single_embedding(query)
    
    # Query with user-specific isolation filter
    results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
        filter={"user_id": {"$eq": user_id}}
    )
    
    candidates = []
    if results and "matches" in results:
        for match in results["matches"]:
            meta = match.get("metadata", {})
            candidates.append({
                "id": match.get("id"),
                "text": meta.get("text", ""),
                "metadata": {"source": meta.get("source", "Unknown"), "user_id": user_id}
            })
    return candidates

def safe_print(message):
    try:
        print(message)
    except UnicodeEncodeError:
        encoding = sys.stdout.encoding or 'utf-8'
        print(message.encode(encoding, errors='replace').decode(encoding))

def reformulate_query(query, chat_history, model_id=DEFAULT_MODEL, user_api_key=None):
    if not chat_history:
        return query
    prompt = "Given the conversation and a follow up question, rephrase it to be a standalone question.\n\nChat History:\n"
    for msg in chat_history[-4:]:
        prompt += f"{msg['role'].capitalize()}: {msg['content']}\n"
    prompt += f"\nFollow Up Input: {query}\nStandalone question:"
    try:
        reformulated = simple_completion(prompt, model_id=model_id, user_api_key=user_api_key, temperature=0.0, max_tokens=256)
        return reformulated.strip('"')
    except Exception:
        return query

def answer_query(query, chat_history=None, top_k=5, user_id="anonymous", model_id=DEFAULT_MODEL, user_api_key=None):
    if chat_history is None: chat_history = []
    search_query = reformulate_query(query, chat_history, model_id, user_api_key)
    relevant_chunks = retrieve_candidates(search_query, top_k=top_k, user_id=user_id)
    
    context_str = "\n\n".join([f"--- Context Segment ---\n{chunk['text']}" for chunk in relevant_chunks])
    system_prompt = f"You are an expert AI Assistant. Answer strictly using only this context:\n\n{context_str}"
    
    messages = [{"role": "system", "content": system_prompt}]
    for msg in chat_history[-6:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": query})
    
    try:
        response = chat_completion(messages=messages, model_id=model_id, user_api_key=user_api_key, temperature=0.2, max_tokens=1024)
        answer = response.choices[0].message.content
    except Exception as e:
        answer = f"Error processing query: {e}"
        
    return {"answer": answer, "sources": relevant_chunks, "model": model_id}

def answer_query_stream(query, chat_history=None, top_k=5, user_id="anonymous", model_id=DEFAULT_MODEL, user_api_key=None):
    if chat_history is None: chat_history = []
    search_query = reformulate_query(query, chat_history, model_id, user_api_key)
    relevant_chunks = retrieve_candidates(search_query, top_k=top_k, user_id=user_id)
    
    sources = [{"id": c["id"], "text": c["text"], "source": c["metadata"]["source"]} for c in relevant_chunks]
    yield {"type": "sources", "sources": sources, "model": model_id}
    
    context_str = "\n\n".join([f"--- Context Segment ---\n{chunk['text']}" for chunk in relevant_chunks])
    system_prompt = f"You are an expert AI Assistant. Answer strictly using only this context:\n\n{context_str}"
    
    messages = [{"role": "system", "content": system_prompt}]
    for msg in chat_history[-6:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": query})
    
    try:
        for token in chat_completion_stream(messages=messages, model_id=model_id, user_api_key=user_api_key, temperature=0.2, max_tokens=1024):
            yield {"type": "token", "token": token}
    except Exception as e:
        yield {"type": "token", "token": f"Streaming error: {e}"}