@echo off
echo =======================================================
echo Starting SynthesizeAI RAG Chatbot...
echo =======================================================

echo.
echo [1/2] Starting Python Backend Server...
echo Note: The backend will take 10-20 seconds to load the AI models into memory on first start.
start "RAG Backend" cmd /k ".\venv\Scripts\python.exe main.py"

echo.
echo [2/2] Starting Next.js Frontend Server...
cd rag-frontend
start "RAG Frontend" cmd /k "npm run dev"

echo.
echo =======================================================
echo Both servers have been launched in separate windows!
echo Please wait for the backend window to say "Application startup complete." before using the chat.
echo =======================================================
pause
