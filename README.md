# SupplementsRx AI - Unified RAG System

## Overview

This system integrates **Neo4j Knowledge Graph** and **Vector Embeddings** to provide comprehensive, evidence-based answers about dietary supplements. An open-source LLM curates and combines information from both sources.

## Architecture

```
User Query
    ↓
┌─────────────────────────────────────┐
│   Unified RAG Pipeline              │
├─────────────────────────────────────┤
│ 1. Vector Embeddings (Semantic)     │
│ 2. Neo4j Knowledge Graph (Structured)│
│ 3. LLM Curation & Synthesis         │
└─────────────────────────────────────┘
    ↓
Structured Answer with Sources
```

## Components

### 1. Vector Embeddings (`src/embeddings/`)
- Semantic search through supplement documents
- Uses sentence transformers
- Returns relevant text chunks with similarity scores

### 2. Neo4j Knowledge Graph (`src/knowledge_graph/`)
- Structured relationships:
  - Supplements ↔ Conditions
  - Supplements ↔ Drugs (interactions)
  - Dosage guidelines
- Full-text search capabilities

### 3. Unified Pipeline (`src/rag/unified_pipeline.py`)
- Combines both retrieval methods
- Uses open-source LLM (Ollama/Hugging Face) for curation
- Provides reasoning and source attribution

### 4. API (`src/api/unified_rag_app.py`)
- FastAPI REST endpoints
- Query and rating functionality
- CORS-enabled for frontend

### 5. Frontend (`src/ui/`)
- ChatGPT-like interface
- Real-time responses
- Source citations
- Rating system

## Key Features

✅ **Dual Retrieval**: Both semantic (embeddings) and structured (KG) search  
✅ **LLM Curation**: Intelligent combination of sources  
✅ **Source Attribution**: Shows which NatMed sections were used  
✅ **Fallback**: Uses LLM knowledge when database is insufficient  
✅ **Precaution Notices**: Safety warnings included  
✅ **Rating System**: User feedback collection  

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up Ollama (recommended):**
   ```bash
   ollama pull llama3.2
   ```

3. **Start API:**
   ```bash
   uvicorn src.api.unified_rag_app:app --reload
   ```

4. **Open frontend:**
   - Open `src/ui/index.html` in browser
   - Or serve with: `python -m http.server 8080` (from `src/ui/`)

## Example Queries

- "Is berberine helpful for type 2 diabetes?"
- "What are the side effects of chromium?"
- "Can I take magnesium with metformin?"
- "What's the recommended dosage of vitamin D for diabetes?"

## Response Format

Each response includes:
- **Answer**: Main response to the question
- **Reasoning**: How sources were combined
- **Sources**: List of sources (vector embeddings + KG)
- **NatMed Sections**: Which sections were referenced
- **Precaution Notice**: Safety warning
- **LLM Knowledge Flag**: Whether general knowledge was used

## Technology Stack

- **Backend**: FastAPI, Python
- **Vector Search**: Sentence Transformers
- **Graph Database**: Neo4j
- **LLM**: Ollama (llama3.2/mistral) or Hugging Face
- **Frontend**: HTML, CSS, JavaScript (vanilla)
- **Data Source**: Natural Medicines (NatMed) database

## File Structure

```
src/
├── api/
│   └── unified_rag_app.py      # FastAPI application
├── rag/
│   └── unified_pipeline.py     # Main RAG pipeline
├── embeddings/
│   └── semantic_search.py      # Vector search
├── knowledge_graph/
│   └── app.py                  # Neo4j queries
└── ui/
    ├── index.html              # Frontend
    ├── styles.css              # Styling
    └── app.js                  # Frontend logic
```

## Configuration

Environment variables (`.env`):
```
NEO4J_URI=bolt://localhost:7688
NEO4J_USER=neo4j
NEO4J_PASSWORD=supplements_pass
```

## License

[Your License Here]

