import os
import sys
import requests
from dotenv import load_dotenv
from pinecone import Pinecone
from llm_router import DEFAULT_MODEL, chat_completion, chat_completion_stream, simple_completion

load_dotenv()
print("\n--- Initializing Cloud Pinecone RAG Pipeline ---")

HF_TOKEN = os.getenv("HF_TOKEN")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Initialize Cloud clients
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("knowledge-base")

API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

def get_single_embedding(text):
    try:
        response = requests.post(API_URL, headers=headers, json={"inputs": [text]})
        response.raise_for_status()
        return response.json()[0]
    except Exception as e:
        print(f"API Embedding failed: {e}")
        return [0.01] * 384

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