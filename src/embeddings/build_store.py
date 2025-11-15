# src/embeddings/build_store.py
import os
import json
import glob
import hashlib
from typing import List, Dict, Any, Iterable, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA


# ----------------------------
# Config
# ----------------------------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data", "processed")
OUTPUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),  # go to project root
    "data",
    "embeddings",
    "embedding_store.json"
)

# Embedding model
BASE_MODEL_NAME = "thenlper/gte-small"  # 384-dim
TARGET_DIM = 256

# Chunk settings
MAX_CHARS_PER_CHUNK = 1200
MIN_CHARS_TO_MERGE = 400


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _stable_id(*parts: str) -> str:
    h = hashlib.sha256("||".join(parts).encode("utf-8")).hexdigest()
    return h[:16]


def _flatten_dosage_guidelines(items: List[Dict[str, Any]]) -> str:
    lines = []
    for it in items or []:
        line = []
        if it.get("condition"): line.append(f"Condition: {it['condition']}")
        if it.get("dosage"): line.append(f"Dosage: {it['dosage']}")
        if it.get("frequency"): line.append(f"Frequency: {it['frequency']}")
        if it.get("duration"): line.append(f"Duration: {it['duration']}")
        if it.get("form"): line.append(f"Form: {it['form']}")
        if it.get("notes"): line.append(f"Notes: {it['notes']}")
        lines.append(" | ".join(line))
    return "\n".join(lines)


def _flatten_safety(s: Dict[str, Any]) -> str:
    if not s: return ""
    out = []
    for k, v in s.items():
        if k == "warnings" and v:
            out.append("Warnings: " + "; ".join(v))
        else:
            out.append(f"{k.replace('_', ' ').title()}: {v}")
    return "\n".join(out)


def _yield_sections(rec: Dict[str, Any], src_path: str):
    longform = {
        "overview": rec.get("overview_text", ""),
        "effectiveness": rec.get("effectiveness_text", ""),
        "safety": rec.get("safety_text", ""),
        "dosing": rec.get("dosing_text", ""),
        "interactions": rec.get("interactions_text", ""),
        "mechanism": rec.get("mechanism_text", "")
    }

    for section, text in longform.items():
        if text.strip():
            yield section, text.strip()

    if rec.get("conditions"):
        yield "conditions", "Conditions: " + ", ".join(rec["conditions"])

    if rec.get("dosage_guidelines"):
        yield "dosage_guidelines", _flatten_dosage_guidelines(rec["dosage_guidelines"])

    if rec.get("safety_ratings"):
        yield "safety_ratings", _flatten_safety(rec["safety_ratings"])

    meta = rec.get("metadata", {})
    name = rec.get("supplement_name") or rec.get("scientific_name") or "Unknown"
    src_url = meta.get("source_url", src_path)

    yield "metadata", f"{name} | Source: {src_url}"


def _chunk_text(text: str) -> List[str]:
    text = " ".join(text.split())
    chunks = []
    cur, length = [], 0

    for token in text.split(" "):
        if length + len(token) + 1 > MAX_CHARS_PER_CHUNK:
            chunks.append(" ".join(cur))
            cur = [token]
            length = len(token)
        else:
            cur.append(token)
            length += len(token) + 1

    if cur:
        chunks.append(" ".join(cur))

    if len(chunks) > 1 and len(chunks[-1]) < MIN_CHARS_TO_MERGE:
        chunks[-2] += " " + chunks[-1]
        chunks.pop()

    return chunks


def _build_corpus():
    corpus = []
    for path in sorted(glob.glob(os.path.join(DATA_DIR, "*.json"))):
        if os.path.basename(path) == "embedding_store.json":
            continue

        data = _read_json(path)
        supplement = data.get("supplement_name") or data.get("scientific_name")

        for section, text in _yield_sections(data, path):
            for idx, chunk in enumerate(_chunk_text(text)):
                corpus.append({
                    "id": _stable_id(path, section, str(idx)),
                    "supplement_name": supplement,
                    "section": section,
                    "chunk_index": idx,
                    "text": chunk,
                    "source_path": path,
                    "source_url": data.get("metadata", {}).get("source_url")
                })

    return corpus


def _normalize(v: np.ndarray):
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)


def main():
    print("Scanning data/processed...")
    docs = _build_corpus()
    print("Chunks:", len(docs))

    model = SentenceTransformer(BASE_MODEL_NAME)
    texts = [d["text"] for d in docs]

    print("Embedding all chunks...")
    base_emb = model.encode(texts, convert_to_numpy=True, batch_size=64, show_progress_bar=True)

    print("Applying PCA → 256-dim...")
    pca = PCA(n_components=TARGET_DIM, random_state=42)
    reduced = pca.fit_transform(base_emb)
    reduced = _normalize(reduced).astype(np.float32)

    for i, d in enumerate(docs):
        d["vector"] = reduced[i].tolist()

    store = {
        "model_name": BASE_MODEL_NAME,
        "base_dim": base_emb.shape[1],
        "vector_dim": TARGET_DIM,
        "reduction": {
            "method": "pca",
            "mean": pca.mean_.astype(np.float32).tolist(),
            "components": pca.components_.astype(np.float32).tolist()
        },
        "documents": docs
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(store, f, ensure_ascii=False)

    print(f"✅ Embedding store written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
