"""
Evaluation script for RAG pipeline.
Runs evaluation metrics on test dataset.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tests.metrics import RAGEvaluator
from src.rag.unified_pipeline import UnifiedRAGPipeline


def load_test_queries(test_file: str) -> list:
    """Load test queries from JSON file"""
    test_path = PROJECT_ROOT / test_file
    with open(test_path, 'r', encoding='utf-8') as f:
        queries = json.load(f)
    return queries


def run_evaluation(
    test_file: str = "tests/test_queries.json",
    ndcg_k: int = 5,
    output_file: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run evaluation on test dataset.
    
    Args:
        test_file: Path to test queries JSON file (relative to project root)
        ndcg_k: k value for nDCG@k metric
        output_file: Optional path to save results (relative to project root)
    
    Returns:
        Dictionary with evaluation results
    """
    print("="*80)
    print("RAG Pipeline Evaluation")
    print("="*80)
    print()
    
    # Load test queries
    print(f"Loading test queries from {test_file}...")
    queries = load_test_queries(test_file)
    print(f"Loaded {len(queries)} test queries")
    print()
    
    # Initialize pipeline
    print("Initializing RAG pipeline...")
    pipeline = UnifiedRAGPipeline()
    print("Pipeline initialized")
    print()
    
    # Initialize evaluator
    print(f"Initializing evaluator (nDCG@k={ndcg_k})...")
    evaluator = RAGEvaluator(ndcg_k=ndcg_k)
    print("Evaluator initialized")
    print()
    
    # Run evaluation
    print("Running evaluation...")
    print("-"*80)
    results = evaluator.evaluate_batch(queries, pipeline)
    
    # Print results
    print()
    print("="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    print()
    
    print(f"Total Queries: {results['total_queries']}")
    print()
    
    # Diabetes Relevance F1 (Macro)
    diabetes_f1 = results['diabetes_relevance_f1_macro']
    print("Diabetes Relevance F1 (Macro):")
    print(f"  Mean: {diabetes_f1['mean']:.4f}")
    print(f"  Std:  {diabetes_f1['std']:.4f}")
    print(f"  Min:  {diabetes_f1['min']:.4f}")
    print(f"  Max:  {diabetes_f1['max']:.4f}")
    print()
    
    # nDCG@k
    ndcg = results['ndcg_at_k']
    print(f"nDCG@{ndcg_k}:")
    print(f"  Mean: {ndcg['mean']:.4f}")
    print(f"  Std:  {ndcg['std']:.4f}")
    print(f"  Min:  {ndcg['min']:.4f}")
    print(f"  Max:  {ndcg['max']:.4f}")
    print()
    
    # Groundedness Score
    groundedness = results['groundedness_score']
    print("Groundedness (Faithfulness) Score:")
    print(f"  Mean: {groundedness['mean']:.4f}")
    print(f"  Std:  {groundedness['std']:.4f}")
    print(f"  Min:  {groundedness['min']:.4f}")
    print(f"  Max:  {groundedness['max']:.4f}")
    print()
    
    # Detailed results
    print("Detailed Results per Query:")
    print("-"*80)
    for i, detail in enumerate(results['detailed_results'], 1):
        print(f"\nQuery {i}: {detail['question']}")
        print(f"  Diabetes Relevance F1: {detail['diabetes_relevance_f1']:.4f}")
        print(f"  nDCG@{ndcg_k}: {detail['ndcg_at_k']:.4f}")
        print(f"  Groundedness Score: {detail['groundedness_score']:.4f}")
    
    print()
    print("="*80)
    
    # Save results if output file specified
    if output_file:
        output_path = PROJECT_ROOT / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {output_path}")
    
    # Cleanup
    pipeline.close()
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate RAG pipeline")
    parser.add_argument(
        "--test-file",
        type=str,
        default="tests/test_queries.json",
        help="Path to test queries JSON file (relative to project root)"
    )
    parser.add_argument(
        "--ndcg-k",
        type=int,
        default=5,
        help="k value for nDCG@k metric"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file to save results (JSON, relative to project root)"
    )
    
    args = parser.parse_args()
    
    run_evaluation(
        test_file=args.test_file,
        ndcg_k=args.ndcg_k,
        output_file=args.output
    )

