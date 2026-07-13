import os
import re
import shutil
import traceback
from typing import List, Dict, Optional
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Header, BackgroundTasks, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import json

import embedder
from embedder import embed_and_store
from rag_pipeline import answer_query, answer_query_stream
from llm_router import get_available_models, DEFAULT_MODEL

# Initialize FastAPI App
app = FastAPI(
    title="RAG Chatbot Backend API",
    description="FastAPI backend supporting dynamic PDF ingestion, vector search, reranking, and Groq LLM generation.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://192.168.210.217:3000", "https://rag-chatbot-six-liart.vercel.app"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure uploads directory exists
UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# =========================
# PYDANTIC SCHEMAS (DATA VALIDATION)
# =========================

class ChatMessage(BaseModel):
    role: str # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    query: str
    chat_history: List[Dict[str, str]] = []
    model_id: Optional[str] = None
    user_api_key: Optional[str] = None



# =========================
# API ENDPOINTS
# =========================

@app.get("/")
def read_root():
    return {
        "message": "Welcome to the Pinecone RAG Chatbot API!",
        "docs_url": "/docs",
        "status": "online"
    }


@app.get("/api/models")
async def list_models():
    """Return available LLM models and host key availability."""
    return {"models": get_available_models(), "default": DEFAULT_MODEL}


@app.post("/api/upload")
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    x_user_id: str = Header(None)
):
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="Invalid file format. Only PDF files are supported."
        )

    user_id_safe = x_user_id if x_user_id else "anonymous"
    
    # 1. Clean the filename to remove unsafe characters and spaces
    clean_filename = re.sub(r'[^a-zA-Z0-9._-]', '_', file.filename)
    
    # 2. Setup safe directories
    UPLOAD_DIR = "./uploads"
    user_dir = os.path.join(UPLOAD_DIR, user_id_safe)
    os.makedirs(user_dir, exist_ok=True)
    filepath = os.path.normpath(os.path.join(user_dir, clean_filename))
    
    try:
        # 3. Write file cleanly
        with open(filepath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        print(f"[UPLOAD SUCCESS] Saved structured file: {filepath}")
        
        # 4. Offload to your background pipeline
        background_tasks.add_task(embedder.embed_and_store, filepath, user_id=user_id_safe)
        
        return {"status": "success", "filename": clean_filename}
        
    except Exception as e:
        print(f"[UPLOAD CRASH]: {str(e)}")
        traceback.print_exc()
        if os.path.exists(filepath):
            os.remove(filepath)
        raise HTTPException(status_code=500, detail=f"Internal Server Pipeline Error: {str(e)}")


@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest, x_user_id: str = Header(None)):
    """
    Endpoint to submit a query and get a retrieved-context response.
    Accommodates chat history for conversational memory.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
        
    try:
        formatted_history = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in request.chat_history
        ]

        user_id_safe = x_user_id if x_user_id else "anonymous"
        model_id = request.model_id or DEFAULT_MODEL
        result = answer_query(
            query=request.query,
            chat_history=formatted_history,
            user_id=user_id_safe,
            model_id=model_id,
            user_api_key=request.user_api_key,
        )
        
        # Format sources for response (remove heavy tensors or raw list formatting)
        sources = []
        for src in result["sources"]:
            sources.append({
                "id": src.get("id"),
                "text": src.get("text"),
                "source": src.get("metadata", {}).get("source", "Unknown"),
                "score": src.get("rerank_score", 0.0)
            })
            
        return {
            "answer": result["answer"],
            "sources": sources
        }
        
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Error executing query: {e}")



@app.post("/api/chat/stream")
async def chat_stream_endpoint(request: ChatRequest, x_user_id: str = Header(None)):
    """
    SSE endpoint to query and get a streamed response from the RAG pipeline.
    Yields sources followed by token chunks.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
        
    try:
        formatted_history = [
            {"role": msg.role, "content": msg.content} 
            for msg in request.chat_history
        ]
        
        user_id_safe = x_user_id if x_user_id else "anonymous"
        model_id = request.model_id or DEFAULT_MODEL
        
        def event_generator():
            for event in answer_query_stream(
                query=request.query, 
                chat_history=formatted_history, 
                user_id=user_id_safe,
                model_id=model_id,
                user_api_key=request.user_api_key
            ):
                yield f"data: {json.dumps(event)}\n\n"
                
        return StreamingResponse(event_generator(), media_type="text/event-stream")
        
    except Exception as e:
        print(f"Error in chat stream endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Error executing streaming query: {e}")


@app.get("/api/status")
async def get_status(x_user_id: str = Header(None)):
    """Returns dynamic vector data validation status."""
    try:
        user_id_safe = x_user_id if x_user_id else "anonymous"
        # Secure query against Pinecone index
        res = embedder.index.query(
            vector=[0.0]*384, 
            top_k=10, 
            include_metadata=True, 
            filter={"user_id": {"$eq": user_id_safe}}
        )
        matches = res.get("matches", [])
        chunk_count = len(matches)
        unique_sources = list(set([m["metadata"]["source"] for m in matches if "metadata" in m]))
        
        return {
            "ready": chunk_count > 0,
            "chunk_count": chunk_count,
            "indexed_documents": unique_sources
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch database status: {e}")


@app.post("/api/reset")
async def reset_db(x_user_id: str = Header(None)):
    """Clears indexed chunks and documents for the current user from ChromaDB."""
    try:
        user_id_safe = x_user_id if x_user_id else "anonymous"
        
        # Delete user's collection chunks
        try:
            embedder.collection.delete(where={"user_id": user_id_safe})
        except Exception:
            pass # ignore if doesn't exist
            
        # Clear user's uploads folder
        user_dir = os.path.join(UPLOAD_DIR, user_id_safe)
        if os.path.exists(user_dir):
            shutil.rmtree(user_dir)
            
        return {
            "status": "success",
            "message": "User's data has been successfully reset."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reset ChromaDB: {e}")


if __name__ == "__main__":
    import uvicorn
    # Start the server on port 8000
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
