"""
Evaluation metrics for RAG pipeline:
1. Diabetes relevance F1 (macro) - Measures relevance to diabetes queries
2. nDCG@k - Normalized Discounted Cumulative Gain at k
3. Groundedness (faithfulness) rate - Measures if answer is grounded in sources
"""

import re
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass


@dataclass
class EvaluationResult:
    """Result of evaluating a single query"""
    question: str
    answer: str
    sources: List[Dict[str, Any]]
    diabetes_relevance_f1: float
    ndcg_at_k: float
    groundedness_score: float
    metrics: Dict[str, float]


class DiabetesRelevanceEvaluator:
    """
    Evaluates diabetes relevance using F1 score (macro-averaged).
    
    Checks if the answer contains diabetes-related information and
    if it correctly identifies diabetes-related supplements/conditions.
    """
    
    # Diabetes-related keywords
    DIABETES_KEYWORDS = [
        "diabetes", "diabetic", "blood sugar", "glucose", "insulin",
        "hba1c", "glycated hemoglobin", "type 2 diabetes", "type 1 diabetes",
        "prediabetes", "hyperglycemia", "hypoglycemia", "diabetic neuropathy",
        "diabetic retinopathy", "metformin", "glucose tolerance"
    ]
    
    # Diabetes-related supplements (from the database)
    DIABETES_SUPPLEMENTS = [
        "berberine", "chromium", "magnesium", "vitamin d", "alpha-lipoic acid",
        "cinnamon", "fenugreek", "gymnema", "bitter melon", "banaba",
        "vanadium", "inositol", "milk thistle", "green tea", "turmeric",
        "curcumin", "resveratrol", "ginger", "glucomannan", "psyllium"
    ]
    
    def __init__(self):
        self.diabetes_keywords_lower = [kw.lower() for kw in self.DIABETES_KEYWORDS]
        self.diabetes_supplements_lower = [s.lower() for s in self.DIABETES_SUPPLEMENTS]
    
    def extract_diabetes_entities(self, text: str) -> Tuple[List[str], List[str]]:
        """
        Extract diabetes-related keywords and supplements from text.
        Returns (keywords_found, supplements_found)
        """
        text_lower = text.lower()
        
        keywords_found = []
        for keyword in self.diabetes_keywords_lower:
            if keyword in text_lower:
                keywords_found.append(keyword)
        
        supplements_found = []
        for supplement in self.diabetes_supplements_lower:
            if supplement in text_lower:
                supplements_found.append(supplement)
        
        return keywords_found, supplements_found
    
    def calculate_f1_macro(
        self, 
        predicted_keywords: List[str], 
        predicted_supplements: List[str],
        ground_truth_keywords: List[str],
        ground_truth_supplements: List[str]
    ) -> float:
        """
        Calculate macro-averaged F1 score for diabetes relevance.
        
        F1 is calculated separately for:
        1. Diabetes keywords detection
        2. Diabetes-related supplements detection
        
        Then macro-averaged.
        """
        def calculate_f1(predicted: List[str], ground_truth: List[str]) -> float:
            """Calculate F1 score for a single category"""
            if not ground_truth:
                # If no ground truth, return 1.0 if no predictions, 0.0 otherwise
                return 1.0 if not predicted else 0.0
            
            predicted_set = set(predicted)
            ground_truth_set = set(ground_truth)
            
            if not predicted_set and not ground_truth_set:
                return 1.0
            
            # Calculate precision, recall, F1
            if not predicted_set:
                return 0.0
            
            tp = len(predicted_set & ground_truth_set)
            fp = len(predicted_set - ground_truth_set)
            fn = len(ground_truth_set - predicted_set)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
            if precision + recall == 0:
                return 0.0
            
            f1 = 2 * (precision * recall) / (precision + recall)
            return f1
        
        # Calculate F1 for keywords
        f1_keywords = calculate_f1(predicted_keywords, ground_truth_keywords)
        
        # Calculate F1 for supplements
        f1_supplements = calculate_f1(predicted_supplements, ground_truth_supplements)
        
        # Macro-average
        f1_macro = (f1_keywords + f1_supplements) / 2.0
        
        return f1_macro
    
    def evaluate(
        self, 
        question: str, 
        answer: str, 
        sources: List[Dict[str, Any]],
        ground_truth_keywords: Optional[List[str]] = None,
        ground_truth_supplements: Optional[List[str]] = None
    ) -> float:
        """
        Evaluate diabetes relevance F1 score.
        
        If ground truth is not provided, uses question to infer expected entities.
        """
        # Extract predicted entities from answer and sources
        answer_keywords, answer_supplements = self.extract_diabetes_entities(answer)
        
        # Also check sources
        source_text = " ".join([s.get("text", "") for s in sources])
        source_keywords, source_supplements = self.extract_diabetes_entities(source_text)
        
        # Combine predictions
        predicted_keywords = list(set(answer_keywords + source_keywords))
        predicted_supplements = list(set(answer_supplements + source_supplements))
        
        # If ground truth not provided, infer from question
        if ground_truth_keywords is None:
            question_keywords, question_supplements = self.extract_diabetes_entities(question)
            ground_truth_keywords = question_keywords
            ground_truth_supplements = question_supplements
        
        if ground_truth_supplements is None:
            _, question_supplements = self.extract_diabetes_entities(question)
            ground_truth_supplements = question_supplements
        
        # Calculate F1 macro
        f1_macro = self.calculate_f1_macro(
            predicted_keywords,
            predicted_supplements,
            ground_truth_keywords,
            ground_truth_supplements
        )
        
        return f1_macro


class NDCGEvaluator:
    """
    Evaluates ranking quality using Normalized Discounted Cumulative Gain at k.
    
    nDCG@k measures how well the retrieved sources are ranked by relevance.
    """
    
    def __init__(self, k: int = 5):
        self.k = k
    
    def calculate_dcg(self, relevance_scores: List[float]) -> float:
        """
        Calculate Discounted Cumulative Gain.
        
        DCG = sum(rel_i / log2(i + 1)) for i in [1, k]
        """
        dcg = 0.0
        for i, rel in enumerate(relevance_scores[:self.k], 1):
            if rel > 0:
                dcg += rel / np.log2(i + 1)
        return dcg
    
    def calculate_ndcg(
        self, 
        predicted_relevance: List[float],
        ideal_relevance: Optional[List[float]] = None
    ) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain at k.
        
        nDCG@k = DCG@k / IDCG@k
        
        Where IDCG is the ideal DCG (perfect ranking).
        """
        # Calculate DCG for predicted ranking
        dcg = self.calculate_dcg(predicted_relevance)
        
        # Calculate IDCG (ideal DCG)
        if ideal_relevance is None:
            # Sort predicted relevance in descending order for ideal
            ideal_relevance = sorted(predicted_relevance, reverse=True)
        else:
            ideal_relevance = sorted(ideal_relevance, reverse=True)
        
        idcg = self.calculate_dcg(ideal_relevance)
        
        # Avoid division by zero
        if idcg == 0:
            return 0.0 if dcg == 0 else 1.0
        
        ndcg = dcg / idcg
        return ndcg
    
    def evaluate(
        self,
        sources: List[Dict[str, Any]],
        ground_truth_relevance: Optional[List[float]] = None
    ) -> float:
        """
        Evaluate nDCG@k for source ranking.
        
        Uses source scores as relevance if ground truth not provided.
        """
        if not sources:
            return 0.0
        
        # Extract relevance scores from sources
        predicted_relevance = []
        for source in sources[:self.k]:
            score = source.get("score", 0.0)
            if score is None:
                score = 0.0
            predicted_relevance.append(float(score))
        
        # If no scores available, use binary relevance (1 if source exists)
        if not predicted_relevance or all(s == 0.0 for s in predicted_relevance):
            predicted_relevance = [1.0] * min(len(sources), self.k)
        
        # Calculate nDCG
        ndcg = self.calculate_ndcg(predicted_relevance, ground_truth_relevance)
        
        return ndcg


class GroundednessEvaluator:
    """
    Evaluates groundedness (faithfulness) - whether the answer is supported by sources.
    
    Measures:
    1. Citation presence - Are sources cited in the answer?
    2. Source coverage - Are the sources actually used?
    3. Hallucination detection - Does answer contain unsupported claims?
    """
    
    def __init__(self):
        # Citation patterns
        self.citation_patterns = [
            r'\[VE\]',  # Vector embedding citation
            r'\[KG\]',  # Knowledge graph citation
            r'\[.*?\]',  # Generic citation
        ]
    
    def check_citations(self, answer: str) -> Tuple[bool, int]:
        """
        Check if answer contains citations.
        Returns (has_citations, citation_count)
        """
        citation_count = 0
        for pattern in self.citation_patterns:
            matches = re.findall(pattern, answer, re.IGNORECASE)
            citation_count += len(matches)
        
        has_citations = citation_count > 0
        return has_citations, citation_count
    
    def check_source_coverage(
        self, 
        answer: str, 
        sources: List[Dict[str, Any]]
    ) -> float:
        """
        Check how well the answer covers information from sources.
        Returns coverage score (0.0 to 1.0).
        """
        if not sources:
            return 0.0
        
        # Extract key information from sources
        source_info = []
        for source in sources:
            text = source.get("text", "")
            supplement = source.get("supplement_name", "")
            section = source.get("section", "")
            
            # Extract key phrases (simple approach)
            if text:
                # Get first 100 chars as key info
                key_info = text[:100].lower()
                source_info.append(key_info)
            if supplement:
                source_info.append(supplement.lower())
            if section:
                source_info.append(section.lower())
        
        # Check if answer contains information from sources
        answer_lower = answer.lower()
        matches = 0
        total = len(source_info)
        
        if total == 0:
            return 0.0
        
        for info in source_info:
            if info and info in answer_lower:
                matches += 1
        
        coverage = matches / total if total > 0 else 0.0
        return coverage
    
    def check_hallucination(
        self,
        answer: str,
        sources: List[Dict[str, Any]],
        question: str
    ) -> float:
        """
        Simple hallucination detection.
        Checks if answer makes claims not supported by sources.
        Returns hallucination score (0.0 = no hallucination, 1.0 = high hallucination).
        """
        if not sources:
            # If no sources, check if answer says "no information"
            if "no information" in answer.lower() or "not available" in answer.lower():
                return 0.0  # Correctly states no information
            return 0.5  # Potential hallucination (claiming info without sources)
        
        # Extract all source text
        source_text = " ".join([s.get("text", "") for s in sources]).lower()
        
        # Check for unsupported claims (simple heuristic)
        # Look for specific numbers, dosages, or claims that might not be in sources
        answer_lower = answer.lower()
        
        # Extract numbers and dosages from answer
        numbers_in_answer = re.findall(r'\d+\.?\d*\s*(mg|g|mcg|iu|%|mg/dl|mmol/l)', answer_lower)
        
        # Check if these numbers appear in sources
        unsupported_count = 0
        for number_phrase in numbers_in_answer:
            if number_phrase not in source_text:
                unsupported_count += 1
        
        # Calculate hallucination score
        if not numbers_in_answer:
            return 0.0  # No specific claims to verify
        
        hallucination_score = min(unsupported_count / len(numbers_in_answer), 1.0)
        return hallucination_score
    
    def evaluate(
        self,
        question: str,
        answer: str,
        sources: List[Dict[str, Any]]
    ) -> float:
        """
        Calculate overall groundedness score.
        
        Combines:
        - Citation presence (30%)
        - Source coverage (50%)
        - Hallucination detection (20%)
        
        Returns score from 0.0 to 1.0 (higher is better).
        """
        # Check citations
        has_citations, citation_count = self.check_citations(answer)
        citation_score = 1.0 if has_citations else 0.0
        
        # Check source coverage
        coverage_score = self.check_source_coverage(answer, sources)
        
        # Check hallucination
        hallucination_score = self.check_hallucination(answer, sources, question)
        anti_hallucination_score = 1.0 - hallucination_score  # Invert (lower hallucination = higher score)
        
        # Weighted combination
        groundedness = (
            0.3 * citation_score +
            0.5 * coverage_score +
            0.2 * anti_hallucination_score
        )
        
        return groundedness


class RAGEvaluator:
    """
    Main evaluator that combines all metrics.
    """
    
    def __init__(self, ndcg_k: int = 5):
        self.diabetes_evaluator = DiabetesRelevanceEvaluator()
        self.ndcg_evaluator = NDCGEvaluator(k=ndcg_k)
        self.groundedness_evaluator = GroundednessEvaluator()
    
    def evaluate(
        self,
        question: str,
        answer: str,
        sources: List[Dict[str, Any]],
        ground_truth: Optional[Dict[str, Any]] = None
    ) -> EvaluationResult:
        """
        Evaluate a single query-response pair.
        
        Args:
            question: User question
            answer: Generated answer
            sources: List of source dictionaries
            ground_truth: Optional ground truth data with:
                - keywords: List of expected diabetes keywords
                - supplements: List of expected supplements
                - relevance: List of relevance scores for sources
        
        Returns:
            EvaluationResult with all metrics
        """
        # Diabetes relevance F1 (macro)
        diabetes_f1 = self.diabetes_evaluator.evaluate(
            question=question,
            answer=answer,
            sources=sources,
            ground_truth_keywords=ground_truth.get("keywords") if ground_truth else None,
            ground_truth_supplements=ground_truth.get("supplements") if ground_truth else None
        )
        
        # nDCG@k
        ndcg = self.ndcg_evaluator.evaluate(
            sources=sources,
            ground_truth_relevance=ground_truth.get("relevance") if ground_truth else None
        )
        
        # Groundedness
        groundedness = self.groundedness_evaluator.evaluate(
            question=question,
            answer=answer,
            sources=sources
        )
        
        # Compile metrics
        metrics = {
            "diabetes_relevance_f1_macro": diabetes_f1,
            "ndcg_at_k": ndcg,
            "groundedness_score": groundedness
        }
        
        return EvaluationResult(
            question=question,
            answer=answer,
            sources=sources,
            diabetes_relevance_f1=diabetes_f1,
            ndcg_at_k=ndcg,
            groundedness_score=groundedness,
            metrics=metrics
        )
    
    def evaluate_batch(
        self,
        queries: List[Dict[str, Any]],
        pipeline
    ) -> Dict[str, Any]:
        """
        Evaluate a batch of queries.
        
        Args:
            queries: List of query dicts with 'question' and optional 'ground_truth'
            pipeline: RAG pipeline instance
        
        Returns:
            Dictionary with aggregated metrics
        """
        results = []
        
        for query_data in queries:
            question = query_data["question"]
            ground_truth = query_data.get("ground_truth")
            
            # Run pipeline
            pipeline_result = pipeline.run(question)
            
            # Convert sources to dict format
            sources = []
            for source in pipeline_result.sources:
                sources.append({
                    "source_type": source.source_type,
                    "supplement_name": source.supplement_name,
                    "section": source.section,
                    "text": source.text,
                    "source_url": source.source_url,
                    "score": source.score,
                    "metadata": source.metadata
                })
            
            # Evaluate
            eval_result = self.evaluate(
                question=question,
                answer=pipeline_result.answer,
                sources=sources,
                ground_truth=ground_truth
            )
            
            results.append(eval_result)
        
        # Aggregate metrics
        diabetes_f1_scores = [r.diabetes_relevance_f1 for r in results]
        ndcg_scores = [r.ndcg_at_k for r in results]
        groundedness_scores = [r.groundedness_score for r in results]
        
        aggregated = {
            "total_queries": len(results),
            "diabetes_relevance_f1_macro": {
                "mean": np.mean(diabetes_f1_scores),
                "std": np.std(diabetes_f1_scores),
                "min": np.min(diabetes_f1_scores),
                "max": np.max(diabetes_f1_scores)
            },
            "ndcg_at_k": {
                "mean": np.mean(ndcg_scores),
                "std": np.std(ndcg_scores),
                "min": np.min(ndcg_scores),
                "max": np.max(ndcg_scores)
            },
            "groundedness_score": {
                "mean": np.mean(groundedness_scores),
                "std": np.std(groundedness_scores),
                "min": np.min(groundedness_scores),
                "max": np.max(groundedness_scores)
            },
            "detailed_results": [
                {
                    "question": r.question,
                    "diabetes_relevance_f1": r.diabetes_relevance_f1,
                    "ndcg_at_k": r.ndcg_at_k,
                    "groundedness_score": r.groundedness_score
                }
                for r in results
            ]
        }
        
        return aggregated

