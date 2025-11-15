@echo off
REM SupplementsRx AI - Unified RAG Server Startup Script (Windows)

echo 🚀 Starting SupplementsRx AI Unified RAG Server...
echo.

REM Check if virtual environment exists
if not exist "venv" (
    echo ⚠️  Virtual environment not found. Creating one...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install/update dependencies
echo 📦 Installing dependencies...
pip install -q -r requirements.txt

REM Start the server
echo.
echo 🌐 Starting API server on http://localhost:8000
echo    Frontend: Open src\ui\index.html in your browser
echo    API Docs: http://localhost:8000/docs
echo.
echo Press Ctrl+C to stop the server
echo.

cd /d "%~dp0"
uvicorn src.api.unified_rag_app:app --host 0.0.0.0 --port 8000 --reload

pause

