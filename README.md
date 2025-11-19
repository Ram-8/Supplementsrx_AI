##SupplementsRx AI - Unified RAG System

## Overview

This system integrates **Neo4j Knowledge Graph** and **Vector Embeddings** to provide comprehensive, evidence-based answers about dietary supplements. An open-source LLM (Gemini) curates and combines information from both sources.

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
- Uses open-source LLM (Gemini) for curation
- Provides reasoning and source attribution as well as structuring

### 4. API (`src/api/unified_rag_app.py`)
- FastAPI REST endpoints
- Query and rating functionality
- CORS-enabled for frontend

### 5. Frontend (`src/ui/`)
- Example questions
- Real-time responses
- Source citations
- Rating system (thumbs up / thumbs down)
- User can report inaccuracies	

## Key Features

**Dual Retrieval**: Both semantic (embeddings) and structured (KG) search  
**LLM Curation**: Combination  of both neo4j kg and vector embedding is done using gemini 
**Fallback**: Uses LLM knowledge when database is insufficient  
**Precaution Notices**: Safety warnings included  
**Rating System**: User feedback collection
**report inaccuracies**: users can report inaccuracies in data
**Fallback mechanism**: hugging	face transformers are used as a fallback mechanism  

## Steps to do:

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Setup docker:**
   ```
    docker start -t neo4j-supplements
   ```
3. *Run .env:*
   ```
    docker start -t neo4j-supplements
   ```

3. **Start API:**
   ```bash
   uvicorn src.api.unified_rag_app:app --reload
   ```

4. **Open frontend:**
   Open `src/ui/index.html` in browser


## Things used

- **Backend**: FastAPI, Python
- **Vector Search**: Sentence Transformers
- **Graph Database**: Neo4j
- **LLM**: Gemini flash 2.0
- **Frontend**: HTML, CSS, JavaScript (vanilla)
- **Data Source**: Natural Medicines (NatMed) database

## Configuration

Environment variables (`.env`):
```
NEO4J_URI=bolt://localhost:7688
NEO4J_USER=neo4j
NEO4J_PASSWORD=supplements_pass
```

## Evaluation Metrics Implementation Summary

Three evaluation metrics have been used in this project:

### 1. Diabetes Relevance F1 (Macro)
- **Location:** `src/evaluation/metrics.py` - `DiabetesRelevanceEvaluator` class
- **Purpose:** Measures how well the system identifies diabetes-related entities
- **Implementation:**
  - Extracts diabetes keywords (diabetes, blood sugar, glucose, insulin, etc.)
  - Extracts diabetes-related supplements (berberine, chromium, magnesium, etc.)
  - Calculates F1 score separately for keywords and supplements
  - Macro-averages the two F1 scores

### 2. nDCG@k (Normalized Discounted Cumulative Gain)
- **Location:** `src/evaluation/metrics.py` - `NDCGEvaluator` class
- **Purpose:** Measures ranking quality of retrieved sources
- **Implementation:**
  - Uses source similarity scores as relevance
  - Calculates DCG (Discounted Cumulative Gain) for top-k sources
  - Normalizes by Ideal DCG (perfect ranking)
  - Formula: nDCG@k = DCG@k / IDCG@k
- **Default k:** 5 (can be configured)

### 3. Groundedness (Faithfulness) Rate
- **Location:** `src/evaluation/metrics.py` - `GroundednessEvaluator` class
- **Purpose:** Measures if answer is properly grounded in sources
- **Implementation:**
  - Citation presence check (30%) - Detects [VE] and [KG] citations
  - Source coverage (50%) - Measures how well answer covers source information
  - Hallucination detection (20%) - Detects unsupported claims
  - Final result depends on the weighted combination of these three components

## Test Dataset

The test dataset includes 10 sample queries covering:
- Supplement effectiveness questions
- Dosage questions
- Drug interaction questions
- Safety questions
- General diabetes supplement questions

Each query includes optional ground truth with:
- Expected keywords
- Expected supplements
- Expected relevance scores

## File Structure

Supplementsrx_AI-main/
│
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── start_server.bat                   # Windows server startup script
├── .env                               # Environment variables (Neo4j, API keys)
│
├── config/                            # Configuration module
│   ├── __init__.py
│   └── config.py                      # Configuration settings
│
├── src/                               # Main source code
│   ├── __init__.py
│   │
│   ├── api/                           # FastAPI application
│   │   ├── __init__.py
│   │   ├── rag_app.py                 # Original RAG API (legacy)
│   │   └── unified_rag_app.py         # Unified RAG API with evaluation endpoints
│   │
│   ├── embeddings/                    # Vector embeddings module
│   │   ├── __init__.py
│   │   ├── build_store.py             # Build embedding store from processed data
│   │   └── semantic_search.py         # Semantic search using embeddings
│   │
│   ├── knowledge_graph/               # Neo4j knowledge graph module
│   │   ├── __init__.py
│   │   ├── app.py                     # Knowledge graph API endpoints
│   │   ├── load_kg.py                 # Load data into Neo4j
│   │   └── backups/
│   │       └── app.py.bak_limitfix_1762928261
│   │
│   ├── preprocessing/                 # Data preprocessing
│   │   ├── __init__.py
│   │   └── structure_extractor.py     # Extract structured data from raw JSON
│   │
│   ├── rag/                           # RAG pipeline
│   │   ├── __init__.py
│   │   ├── pipeline.py                # Original RAG pipeline (legacy)
│   │   └── unified_pipeline.py        # Unified RAG pipeline (Neo4j + embeddings)
│   │
│   ├── scrapers/                      # Web scraping
│   │   ├── __init__.py
│   │   └── natmed_scraper_accordion.py # Natural Medicines scraper
│   │
│   └── ui/                            # Frontend
│       ├── __init__.py
│       ├── index.html                 # Main HTML file
│       ├── app.js                     # Frontend JavaScript
│       └── styles.css                 # CSS styles
│
├── data/                              # Data directory
│   ├── raw/                           # Raw scraped data (25+ JSON files)
│   │
│   │
│   ├── processed/                     # Processed/structured data (25 JSON files)
│   │ 
│   │
│   ├── embeddings/                    # Vector embeddings
│   │   └── embedding_store.json       # Pre-computed embeddings store
│   │
│   └── neo4j_kg/                      # Neo4j knowledge graph exports
│       ├── schema.cypher              # Database schema
│       ├── supplements_kg.cypher      # Cypher export
│       └── supplements_kg.graphml     # GraphML export
│
├── tests/                             # Test and evaluation files
│   ├── __init__.py
│   ├── metrics.py                     # Evaluation metrics (F1, nDCG, Groundedness)
│   ├── evaluate.py                    # Evaluation script
│   ├── test_queries.json              # Test dataset with ground truth
│   └── evaluation_results.json        # Evaluation results output
│
├── venv/                              # Python virtual environment (excluded from version control)
│               
│
└── env/                               # Environment-related C files (system files)

