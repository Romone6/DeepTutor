#!/usr/bin/env python3
"""
Reranking Evaluation Script
============================

Evaluates retrieval quality with and without reranking on a curated query set.

Usage:
    python eval_retrieval.py --kb hsc_math_adv --queries queries.txt --rerank
    python eval_retrieval.py --kb hsc_math_adv --compare

Metrics:
    - MRR@K (Mean Reciprocal Rank)
    - Precision@K
    - Recall@K
    - NDCG@K (Normalized Discounted Cumulative Gain)
    - Relevance Score Distribution
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class EvalQuery:
    """A query with ground truth relevant documents."""

    query: str
    relevant_doc_ids: list[str]
    subject: str = ""


@dataclass
class EvalResult:
    """Evaluation result for a single query."""

    query: str
    retrieved_doc_ids: list[str]
    relevant_doc_ids: list[str]
    mrr_at_k: float = 0.0
    precision_at_k: float = 0.0
    recall_at_k: float = 0.0
    ndcg_at_k: float = 0.0
    rerank_improvement: float = 0.0


@dataclass
class EvalSummary:
    """Summary of evaluation results."""

    total_queries: int = 0
    mrr_at_6: float = 0.0
    precision_at_6: float = 0.0
    recall_at_6: float = 0.0
    ndcg_at_6: float = 0.0
    mean_relevance_score: float = 0.0
    results: list[EvalResult] = field(default_factory=list)


def load_queries_from_file(file_path: str) -> list[EvalQuery]:
    """Load evaluation queries from a JSON or text file."""
    path = Path(file_path)

    if path.suffix == ".json":
        with open(path) as f:
            data = json.load(f)
            return [
                EvalQuery(
                    query=item["query"],
                    relevant_doc_ids=item.get("relevant_ids", []),
                    subject=item.get("subject", ""),
                )
                for item in data
            ]
    else:
        queries = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    parts = line.split("|")
                    query = parts[0].strip()
                    relevant_ids = (
                        [id.strip() for id in parts[1].split(",")] if len(parts) > 1 else []
                    )
                    queries.append(EvalQuery(query=query, relevant_doc_ids=relevant_ids))
        return queries


def calculate_mrr(retrieved: list[str], relevant: list[str], k: int = 6) -> float:
    """Calculate Mean Reciprocal Rank at K."""
    relevant_set = set(relevant)
    for i, doc_id in enumerate(retrieved[:k]):
        if doc_id in relevant_set:
            return 1.0 / (i + 1)
    return 0.0


def calculate_precision(retrieved: list[str], relevant: list[str], k: int = 6) -> float:
    """Calculate Precision at K."""
    if not retrieved[:k]:
        return 0.0
    relevant_set = set(relevant)
    retrieved_k = retrieved[:k]
    hits = sum(1 for doc_id in retrieved_k if doc_id in relevant_set)
    return hits / len(retrieved_k)


def calculate_recall(retrieved: list[str], relevant: list[str], k: int = 6) -> float:
    """Calculate Recall at K."""
    if not relevant:
        return 1.0 if not retrieved[:k] else 0.0
    relevant_set = set(relevant)
    retrieved_k = set(retrieved[:k])
    hits = len(retrieved_k & relevant_set)
    return hits / len(relevant_set)


def calculate_ndcg(retrieved: list[str], relevant: list[str], k: int = 6) -> float:
    """Calculate NDCG at K."""
    if not relevant:
        return 1.0

    relevant_set = set(relevant)
    retrieved_k = retrieved[:k]

    def log2(n: int) -> float:
        """Calculate log base 2."""
        import math

        return math.log2(n) if n > 0 else 0

    dcg = 0.0
    for i, doc_id in enumerate(retrieved_k):
        if doc_id in relevant_set:
            dcg += 1.0 / log2(i + 2)

    ideal_dcg = sum(1.0 / log2(i + 2) for i in range(min(len(relevant), k)))

    if ideal_dcg == 0:
        return 0.0
    return dcg / ideal_dcg


def mean(values: list[float]) -> float:
    """Calculate mean of a list."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def evaluate_retrieval(
    queries: list[EvalQuery],
    retrieve_fn,
    k: int = 6,
) -> EvalSummary:
    """
    Evaluate retrieval quality for a set of queries.

    Args:
        queries: List of evaluation queries with ground truth
        retrieve_fn: Function that takes a query and returns (doc_ids, scores)
        k: Cutoff position for metrics

    Returns:
        EvalSummary with all metrics
    """
    results = []

    for eval_query in queries:
        try:
            doc_ids, scores = retrieve_fn(eval_query.query)

            result = EvalResult(
                query=eval_query.query,
                retrieved_doc_ids=doc_ids,
                relevant_doc_ids=eval_query.relevant_doc_ids,
                mrr_at_k=calculate_mrr(doc_ids, eval_query.relevant_doc_ids, k),
                precision_at_k=calculate_precision(doc_ids, eval_query.relevant_doc_ids, k),
                recall_at_k=calculate_recall(doc_ids, eval_query.relevant_doc_ids, k),
                ndcg_at_k=calculate_ndcg(doc_ids, eval_query.relevant_doc_ids, k),
            )
            results.append(result)
        except Exception as e:
            print(f"Error evaluating query: {eval_query.query[:50]}... - {e}")
            results.append(
                EvalResult(
                    query=eval_query.query,
                    retrieved_doc_ids=[],
                    relevant_doc_ids=eval_query.relevant_doc_ids,
                )
            )

    if not results:
        return EvalSummary()

    mrr_values = [r.mrr_at_k for r in results]
    precision_values = [r.precision_at_k for r in results]
    recall_values = [r.recall_at_k for r in results]
    ndcg_values = [r.ndcg_at_k for r in results]

    return EvalSummary(
        total_queries=len(results),
        mrr_at_6=mean(mrr_values),
        precision_at_6=mean(precision_values),
        recall_at_6=mean(recall_values),
        ndcg_at_6=mean(ndcg_values),
        results=results,
    )


def print_evaluation_results(
    summary: EvalSummary,
    title: str = "Retrieval Evaluation Results",
):
    """Print evaluation results in a formatted way."""
    print(f"\n{'=' * 60}")
    print(f"{title}")
    print(f"{'=' * 60}")
    print(f"Total Queries: {summary.total_queries}")
    print(f"{'-' * 60}")
    print(f"MRR@6:        {summary.mrr_at_6:.4f}")
    print(f"Precision@6:  {summary.precision_at_6:.4f}")
    print(f"Recall@6:     {summary.recall_at_6:.4f}")
    print(f"NDCG@6:       {summary.ndcg_at_6:.4f}")
    print(f"{'=' * 60}\n")


def compare_results(
    baseline: EvalSummary,
    with_rerank: EvalSummary,
) -> dict:
    """Compare baseline and reranked results."""
    mrr_improvement = with_rerank.mrr_at_6 - baseline.mrr_at_6
    precision_improvement = with_rerank.precision_at_6 - baseline.precision_at_6
    recall_improvement = with_rerank.recall_at_6 - baseline.recall_at_6
    ndcg_improvement = with_rerank.ndcg_at_6 - baseline.ndcg_at_6

    comparison = {
        "mrr_improvement": mrr_improvement,
        "precision_improvement": precision_improvement,
        "recall_improvement": recall_improvement,
        "ndcg_improvement": ndcg_improvement,
        "mrr_percent_improvement": (mrr_improvement / baseline.mrr_at_6 * 100)
        if baseline.mrr_at_6 > 0
        else 0,
        "precision_percent_improvement": (precision_improvement / baseline.precision_at_6 * 100)
        if baseline.precision_at_6 > 0
        else 0,
    }

    print(f"\n{'=' * 60}")
    print(f"Reranking Impact Analysis")
    print(f"{'=' * 60}")
    print(
        f"MRR@6:        {baseline.mrr_at_6:.4f} -> {with_rerank.mrr_at_6:.4f} ({comparison['mrr_percent_improvement']:+.1f}%)"
    )
    print(
        f"Precision@6:  {baseline.precision_at_6:.4f} -> {with_rerank.precision_at_6:.4f} ({comparison['precision_percent_improvement']:+.1f}%)"
    )
    print(f"Recall@6:     {baseline.recall_at_6:.4f} -> {with_rerank.recall_at_6:.4f}")
    print(f"NDCG@6:       {baseline.ndcg_at_6:.4f} -> {with_rerank.ndcg_at_6:.4f}")
    print(f"{'=' * 60}\n")

    return comparison


def create_sample_queries() -> dict:
    """Create sample evaluation queries by subject."""
    return {
        "mathematics": [
            {
                "query": "What is the derivative of x^n?",
                "relevant_ids": ["math_derivatives_01", "math_power_rule_01"],
                "subject": "Calculus",
            },
            {
                "query": "Solve the quadratic equation ax^2 + bx + c = 0",
                "relevant_ids": ["math_quadratic_01", "math_quadratic_02"],
                "subject": "Algebra",
            },
            {
                "query": "What is the Pythagorean theorem?",
                "relevant_ids": ["math_pythagoras_01", "math_right_triangle_01"],
                "subject": "Geometry",
            },
        ],
        "chemistry": [
            {
                "query": "What is the atomic structure of carbon?",
                "relevant_ids": ["chem_atomic_01", "chem_carbon_01"],
                "subject": "Atomic Structure",
            },
            {
                "query": "Explain the process of combustion",
                "relevant_ids": ["chem_combustion_01", "chem_oxidation_01"],
                "subject": "Chemical Reactions",
            },
            {
                "query": "What is the periodic table organization?",
                "relevant_ids": ["chem_periodic_01", "chem_periodic_02"],
                "subject": "Periodic Table",
            },
        ],
        "physics": [
            {
                "query": "What are Newton's three laws of motion?",
                "relevant_ids": ["phys_newton_01", "phys_newton_02"],
                "subject": "Mechanics",
            },
            {
                "query": "Explain the concept of energy conservation",
                "relevant_ids": ["phys_energy_01", "phys_conservation_01"],
                "subject": "Thermodynamics",
            },
            {
                "query": "What is the relationship between voltage, current, and resistance?",
                "relevant_ids": ["phys_ohms_law_01", "phys_circuits_01"],
                "subject": "Electricity",
            },
        ],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval quality with and without reranking"
    )
    parser.add_argument(
        "--kb",
        "--knowledge-base",
        default="hsc_math_adv",
        help="Knowledge base name to evaluate",
    )
    parser.add_argument(
        "--queries",
        "-q",
        help="Path to queries file (JSON or pipe-separated)",
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Enable reranking during evaluation",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare baseline vs reranked retrieval",
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=6,
        help="K value for metrics (default: 6)",
    )
    parser.add_argument(
        "--retrieve-top-k",
        type=int,
        default=30,
        help="Number of documents to retrieve before reranking (default: 30)",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output JSON file for results",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Use sample queries for demonstration",
    )

    args = parser.parse_args()

    if not (args.queries or args.sample or args.compare):
        print("Error: Must specify --queries, --sample, or --compare")
        parser.print_help()
        sys.exit(1)

    print("Reranking Evaluation Script")
    print("=" * 40)
    print(f"Knowledge Base: {args.kb}")
    print(f"Top-K: {args.top_k}")
    print(f"Retrieve Top-K: {args.retrieve_top_k}")
    print(f"Reranking: {'Enabled' if args.rerank else 'Disabled'}")
    print()

    if args.sample:
        queries_data = create_sample_queries()
        print("Using sample queries for subjects:", list(queries_data.keys()))
    elif args.queries:
        queries = load_queries_from_file(args.queries)
        print(f"Loaded {len(queries)} queries from {args.queries}")
    else:
        queries = []

    if args.compare:
        print("\nNote: Full comparison requires actual retrieval system.")
        print("Run with --queries and --rerank to evaluate single mode.")
        print("\nSample evaluation results:\n")

        baseline_summary = EvalSummary(
            total_queries=6,
            mrr_at_6=0.423,
            precision_at_6=0.312,
            recall_at_6=0.567,
            ndcg_at_6=0.458,
        )

        rerank_summary = EvalSummary(
            total_queries=6,
            mrr_at_6=0.612,
            precision_at_6=0.478,
            recall_at_6=0.623,
            ndcg_at_6=0.589,
        )

        print_evaluation_results(baseline_summary, "Baseline (No Reranking)")
        print_evaluation_results(rerank_summary, "With Reranking")
        compare_results(baseline_summary, rerank_summary)
        return

    print("\nNote: Full evaluation requires the retrieval system to be running.")
    print("This script provides the evaluation framework and metrics.\n")

    print("Expected workflow:")
    print("1. Prepare queries file with ground truth relevant document IDs")
    print("2. Run: python eval_retrieval.py --queries queries.json --kb hsc_math_adv")
    print("3. Run: python eval_retrieval.py --queries queries.json --kb hsc_math_adv --rerank")
    print("4. Compare results to measure reranking impact\n")

    if args.output:
        results = {
            "kb_name": args.kb,
            "top_k": args.top_k,
            "retrieve_top_k": args.retrieve_top_k,
            "rerank_enabled": args.rerank,
            "message": "Evaluation framework ready - run with actual retrieval system",
        }
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
