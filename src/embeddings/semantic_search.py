# src/embeddings/semantic_search.py
import os
import json
import numpy as np
from typing import Any, Dict, List

from sentence_transformers import SentenceTransformer


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
STORE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data",
    "embeddings",
    "embedding_store.json"
)


def _normalize(v: np.ndarray):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)


def _apply_pca(vec: np.ndarray, mean: np.ndarray, comps: np.ndarray):
    return (vec - mean) @ comps.T


class SemanticSearcher:

    def __init__(self, store_path: str = STORE_PATH):
        with open(store_path, "r", encoding="utf-8") as f:
            self.store = json.load(f)

        self.model_name = self.store["model_name"]
        self.model = SentenceTransformer(self.model_name)

        red = self.store["reduction"]
        self.pca_mean = np.array(red["mean"], dtype=np.float32)
        self.pca_components = np.array(red["components"], dtype=np.float32)

        self.docs = self.store["documents"]
        self.doc_matrix = np.array([d["vector"] for d in self.docs], dtype=np.float32)
        self.doc_matrix = _normalize(self.doc_matrix)

    def embed_query(self, query: str) -> np.ndarray:
        base = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=False)
        proj = _apply_pca(base, self.pca_mean, self.pca_components)
        proj = _normalize(proj)
        return proj[0]

    def search(self, query: str, top_k: int = 5):
        qv = self.embed_query(query)
        sims = self.doc_matrix @ qv

        top_k = min(top_k, len(self.docs))
        idx = np.argpartition(-sims, top_k - 1)[:top_k]
        idx = idx[np.argsort(-sims[idx])]

        results = []
        for i in idx:
            d = self.docs[i]
            results.append({
                "score": float(sims[i]),
                "supplement_name": d["supplement_name"],
                "section": d["section"],
                "chunk_index": d["chunk_index"],
                "text": d["text"],
                "source_url": d["source_url"]
            })
        return results

    def best_answer(self, query: str, top_k: int = 3):
        hits = self.search(query, top_k)
        context = "\n\n---\n\n".join([h["text"] for h in hits])

        return {
            "query": query,
            "context": context,
            "top_hit": hits[0] if hits else None,
            "hits": hits
        }


if __name__ == "__main__":
    s = SemanticSearcher()
    res = s.best_answer("Which supplements help with blood sugar during pregnancy?")
    print(res["top_hit"])
