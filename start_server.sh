#!/bin/bash

# SupplementsRx AI - Unified RAG Server Startup Script

echo "🚀 Starting SupplementsRx AI Unified RAG Server..."
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "⚠️  Virtual environment not found. Creating one..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install/update dependencies
echo "📦 Installing dependencies..."
pip install -q -r requirements.txt

# Check Neo4j connection
echo ""
echo "🔍 Checking Neo4j connection..."
python3 -c "
from neo4j import GraphDatabase
import os
from dotenv import load_dotenv
load_dotenv()

try:
    driver = GraphDatabase.driver(
        os.getenv('NEO4J_URI', 'bolt://localhost:7688'),
        auth=(os.getenv('NEO4J_USER', 'neo4j'), os.getenv('NEO4J_PASSWORD', 'supplements_pass'))
    )
    with driver.session() as session:
        session.run('RETURN 1')
    print('✓ Neo4j connection successful')
    driver.close()
except Exception as e:
    print('⚠️  Neo4j connection failed:', e)
    print('   Make sure Neo4j is running and credentials are correct')
"

# Check Ollama
echo ""
echo "🔍 Checking Ollama..."
if command -v ollama &> /dev/null; then
    if ollama list &> /dev/null; then
        echo "✓ Ollama is available"
        echo "  Available models:"
        ollama list | grep -v "^NAME" | awk '{print "    - " $1}'
    else
        echo "⚠️  Ollama is installed but not running"
        echo "   Start it with: ollama serve"
    fi
else
    echo "⚠️  Ollama not found"
    echo "   Install from: https://ollama.ai"
    echo "   Or the system will use fallback mode"
fi

# Start the server
echo ""
echo "🌐 Starting API server on http://localhost:8000"
echo "   Frontend: Open src/ui/index.html in your browser"
echo "   API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

cd "$(dirname "$0")"
uvicorn src.api.unified_rag_app:app --host 0.0.0.0 --port 8000 --reload

