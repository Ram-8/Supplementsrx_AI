# src/rag/pipeline.py

import google.generativeai as genai
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

from src.embeddings.semantic_search import SemanticSearcher

# ================== IMPORTANT ==================
# Replace THIS placeholder with your REAL Gemini key from:
# https://aistudio.google.com/apikey
#
# It MUST look like: AIzaSy......
#
GEMINI_API_KEY = "AIzaSyD35ipoIqrZL9bpG3zvlHYBkZJFzUIHqMg"
genai.configure(api_key=GEMINI_API_KEY)
# ===============================================


@dataclass
class RAGSource:
    supplement_name: Optional[str]
    section: str
    chunk_index: int
    text: str
    source_url: Optional[str]
    score: float


@dataclass
class RAGResult:
    question: str
    answer: str
    sources: List[RAGSource]


class RAGPipeline:
    """
    Usage:
        pipeline = RAGPipeline()
        result = pipeline.run("Is berberine helpful for type 2 diabetes?")
    """

    def __init__(
        self,
        searcher: Optional[SemanticSearcher] = None,
        model="gemini-2.0-flash",
        top_k: int = 5,
    ):
        self.searcher = searcher or SemanticSearcher()
        self.model_name = model
        self.model = genai.GenerativeModel(self.model_name)
        self.top_k = top_k

    # ---------------- SCOPE & FILTERS ----------------

    def _is_in_scope(self, question: str) -> bool:
        """
        Restrict to diabetes-related questions.
        Anything outside this returns a generic 'out of scope' answer.
        """
        q = (question or "").lower()
        diabetes_terms = [
            "diabetes",
            "diabetic",
            "blood sugar",
            "blood glucose",
            "glucose",
            "a1c",
            "hba1c",
            "metformin",
            "insulin",
            "prediabetes",
        ]
        return any(term in q for term in diabetes_terms)

    def _select_hits(
        self,
        hits: List[Dict[str, Any]],
        min_score: float = 0.55
    ) -> List[Dict[str, Any]]:
        """
        - Keep only chunks with non-empty text.
        - Prefer chunks with score >= min_score.
        - If none pass the score threshold, fall back to all non-empty hits.

        This guarantees *some* context for any diabetes-related question,
        so there's always something to summarize.
        """
        useful = [
            h for h in hits
            if h.get("text") and h["text"].strip()
        ]

        if not useful:
            return []

        high = [
            h for h in useful
            if float(h.get("score", 0.0)) >= min_score
        ]
        if high:
            return high

        # Fallback: use all useful hits (even if scores are low)
        return useful

    # ---------------- CONTEXT & LLM ----------------

    def _build_context(self, hits: List[Dict[str, Any]]) -> str:
        """
        Convert retrieved chunks into a readable context block.
        """
        if not hits:
            return "No matching supplement context found in the database."

        blocks = []
        for i, h in enumerate(hits, start=1):
            name = h.get("supplement_name") or "Unknown supplement"
            section = h.get("section", "unknown")
            src = h.get("source_url") or "N/A"
            text = h.get("text", "")

            blocks.append(
                f"[{i}] Supplement: {name}\n"
                f"Section: {section}\n"
                f"Source: {src}\n"
                f"Text:\n{text}"
            )
        return "\n\n---\n\n".join(blocks)

    def _call_llm(self, question: str, context: str) -> str:
        """
        Call Gemini with a prompt that:
        - ONLY uses the given context.
        - Sounds like a chatbot talking to a patient.
        - Always returns a short answer + bullet points + safety note
          for diabetes questions when info exists.
        """
        system_prompt = (
            "You are a friendly, evidence-based chatbot that helps people with diabetes "
            "understand dietary supplements.\n\n"
            "STRICT RULES:\n"
            "1. Use ONLY the information in the context. Do NOT add your own knowledge.\n"
            "2. If the asked supplement is NOT mentioned in the context, say clearly that "
            "   this database does not have information about it, and do NOT guess.\n"
            "3. If information IS available, respond in this structure:\n"
            "   - First line: 'Short answer: ...'\n"
            "   - Then 2–4 bullet points summarizing key details (benefits, how it works, "
            "     important numbers for blood sugar / HbA1c, and important cautions if present).\n"
            "   - End with a one-sentence safety note telling the user to check with their "
            "     diabetes healthcare provider.\n"
            "4. Use simple language suitable for patients, not clinicians."
        )

        full_prompt = (
            f"{system_prompt}\n\n"
            f"User question:\n{question}\n\n"
            f"Context from the supplement database:\n{context}\n\n"
            "Now generate the chatbot response in that format."
        )

        response = self.model.generate_content(
            full_prompt,
            generation_config={"temperature": 0.2}
        )

        return (response.text or "").strip()


    # ---------------- MAIN PIPELINE ----------------

    def run(self, question: str, top_k: Optional[int] = None) -> RAGResult:
        k = top_k or self.top_k

        # 1) Non-diabetes questions → generic response
        if not self._is_in_scope(question):
            generic = (
                "This system is specialized for diabetes-related supplement guidance. "
                "The current database does not cover this condition, so it cannot "
                "provide a reliable answer."
            )
            return RAGResult(
                question=question,
                answer=generic,
                sources=[],
            )

        # 2) Retrieve from embeddings for diabetes-related questions
        bundle = self.searcher.best_answer(question, top_k=k)
        raw_hits = bundle.get("hits", [])

        # 3) Filter hits (but always keep something if any text exists)
        hits = self._select_hits(raw_hits, min_score=0.55)

        # 4) Build context and call Gemini (even if hits is empty)
        context = self._build_context(hits)
        answer = self._call_llm(question, context)

        # 5) Convert hits to RAGSource objects (may be empty if no context)
        sources = [
            RAGSource(
                supplement_name=h.get("supplement_name"),
                section=h.get("section", ""),
                chunk_index=h.get("chunk_index", 0),
                text=h.get("text", ""),
                source_url=h.get("source_url"),
                score=float(h.get("score", 0.0)),
            )
            for h in hits
        ]

        return RAGResult(
            question=question,
            answer=answer,
            sources=sources,
        )

    def run_as_dict(self, question: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        result = self.run(question, top_k)
        return {
            "answer": result.answer,
        }


if __name__ == "__main__":
    pipe = RAGPipeline()

    while True:
        question = input("\nAsk a question (or type 'exit' to quit): ")

        if question.lower().strip() in ["exit", "quit"]:
            print("Goodbye!")
            break

        result = pipe.run_as_dict(question)
        print("\nANSWER:\n", result.answer)
