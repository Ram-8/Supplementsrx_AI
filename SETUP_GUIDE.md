# SupplementsRx AI - Setup Guide

This guide will help you set up the unified RAG system that combines Neo4j knowledge graph and vector embeddings with an open-source LLM.

## Prerequisites

1. **Python 3.8+** installed
2. **Neo4j Database** running (default: `bolt://localhost:7688`)
3. **Ollama** (recommended) or Hugging Face transformers for LLM

## Installation Steps

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up Neo4j

Make sure Neo4j is running and accessible. Update your `.env` file or environment variables:

```bash
NEO4J_URI=bolt://localhost:7688
NEO4J_USER=neo4j
NEO4J_PASSWORD=supplements_pass
```

### 3. Set Up LLM (Choose One)

#### Option A: Ollama (Recommended - Easy to Use)

1. Install Ollama from https://ollama.ai
2. Pull a model:
   ```bash
   ollama pull llama3.2
   # or
   ollama pull mistral
   # or
   ollama pull qwen2.5
   ```
3. The system will automatically use Ollama if available.

#### Option B: Hugging Face Transformers

If you don't want to use Ollama, the system will fall back to Hugging Face transformers. Note that this requires more memory and may be slower.

The system will automatically detect and use available LLM options.

### 4. Verify Data

Make sure you have:
- Processed data in `data/processed/`
- Embeddings built in `data/embeddings/embedding_store.json`
- Neo4j knowledge graph loaded

If not, run:
```bash
# Build embeddings (if not done)
python src/embeddings/build_store.py

# Load Neo4j KG (if not done)
python src/knowledge_graph/load_kg.py
```

## Running the Application

### 1. Start the API Server

```bash
cd src/api
python unified_rag_app.py
```

Or using uvicorn directly:
```bash
uvicorn src.api.unified_rag_app:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`

### 2. Open the Frontend

Open `src/ui/index.html` in your web browser, or serve it using a simple HTTP server:

```bash
# Python 3
cd src/ui
python -m http.server 8080

# Then open http://localhost:8080 in your browser
```

Or use any other static file server.

### 3. Test the API

```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Is berberine helpful for type 2 diabetes?"}'
```

## Features

### Unified RAG Pipeline

- **Vector Embeddings**: Semantic search through supplement documents
- **Neo4j Knowledge Graph**: Structured relationships (supplements, conditions, interactions, dosages)
- **LLM Curation**: Combines both sources intelligently
- **Source Attribution**: Shows which NatMed sections were referenced
- **Fallback**: Uses LLM knowledge if database information is insufficient

### Frontend Features

- ChatGPT-like interface
- Real-time responses
- Source citations
- Precaution notices
- Thumbs up/down rating system
- Responsive design

## API Endpoints

### POST `/api/query`
Query the unified RAG system.

**Request:**
```json
{
  "question": "Is berberine helpful for type 2 diabetes?",
  "top_k": 5
}
```

**Response:**
```json
{
  "answer": "...",
  "reasoning": "...",
  "sources": [...],
  "used_llm_knowledge": false,
  "precaution_notice": "...",
  "natmed_sections": ["Effectiveness", "Safety"]
}
```

### POST `/api/rate`
Rate a response (thumbs up/down).

**Request:**
```json
{
  "question": "...",
  "answer": "...",
  "rating": 1,  // 1 for thumbs up, -1 for thumbs down
  "feedback": "Optional feedback"
}
```

### GET `/health`
Check API health and LLM availability.

## Troubleshooting

### LLM Not Available

If you see "fallback" in the health check:
- Install Ollama and pull a model, OR
- The system will use a simple template-based response

### Neo4j Connection Issues

- Verify Neo4j is running: `neo4j status`
- Check connection settings in `.env`
- Ensure the database has been loaded with supplement data

### Embeddings Not Found

- Run `python src/embeddings/build_store.py` to build embeddings

### CORS Issues

- The API allows all origins by default (for development)
- In production, update `allow_origins` in `unified_rag_app.py`

## Configuration

You can customize the pipeline in `src/rag/unified_pipeline.py`:

- `llm_model`: Change the Ollama model name
- `top_k_embeddings`: Number of embedding results to retrieve
- `top_k_kg`: Number of knowledge graph results to retrieve

## Next Steps

- Add database persistence for ratings
- Implement user authentication
- Add conversation history
- Fine-tune LLM prompts
- Add more sophisticated query understanding

