"""
MTEB Evaluation Script for Sparse Encoders

Evaluate trained sparse encoder models on Scandinavian MTEB tasks:
- NorQuAD: Norwegian question answering retrieval
- SNL: Norwegian language retrieval

Usage:
    uv run python scripts/evaluate_sparse_mteb.py --model-path models/splade-norbert3-multidataset/final
"""

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path

import mteb
from sentence_transformers import SparseEncoder

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# Scandinavian MTEB tasks
SCANDINAVIAN_TASKS = {
    "norwegian": [
        "NorQuAD",  # Norwegian question answering
        "SNL",      # Norwegian language retrieval
    ],
    "danish": [
        # Add Danish tasks if available
    ],
    "swedish": [
        # Add Swedish tasks if available
    ],
}


def evaluate_model(
    model_path: str,
    output_dir: str = "results/mteb",
    tasks: list = None,
    batch_size: int = 32,
):
    """
    Evaluate sparse encoder model on MTEB tasks.
    
    Args:
        model_path: Path to trained sparse encoder model
        output_dir: Directory to save evaluation results
        tasks: List of MTEB task names (default: Norwegian tasks)
        batch_size: Batch size for encoding
    """
    logger.info("=" * 80)
    logger.info("SPARSE ENCODER MTEB EVALUATION")
    logger.info("=" * 80)
    
    # Load model
    logger.info(f"Loading model from: {model_path}")
    model = SparseEncoder(model_path)
    logger.info(f"Model loaded: {model}")
    
    # Default to Norwegian tasks if none specified
    if tasks is None:
        tasks = SCANDINAVIAN_TASKS["norwegian"]
        logger.info(f"Using default Norwegian tasks: {tasks}")
    else:
        logger.info(f"Using specified tasks: {tasks}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Run evaluation
    results = {}
    
    for task_name in tasks:
        logger.info("\n" + "=" * 80)
        logger.info(f"EVALUATING: {task_name}")
        logger.info("=" * 80)
        
        try:
            # Load task
            task = mteb.get_task(task_name)
            
            # Run evaluation
            evaluation = mteb.MTEB(tasks=[task])
            task_results = evaluation.run(
                model,
                output_folder=output_dir,
                batch_size=batch_size,
                overwrite_results=False,
            )
            
            results[task_name] = task_results
            
            # Log key metrics
            if task_name in task_results:
                task_result = task_results[task_name]
                if 'test' in task_result:
                    test_scores = task_result['test']
                    
                    # Log NDCG@10 if available (common for retrieval)
                    if 'ndcg_at_10' in test_scores:
                        logger.info(f"  NDCG@10: {test_scores['ndcg_at_10']:.4f}")
                    
                    # Log other metrics
                    for metric, value in test_scores.items():
                        if isinstance(value, (int, float)):
                            logger.info(f"  {metric}: {value:.4f}")
            
            logger.info(f"✓ {task_name} evaluation complete")
            
        except Exception as e:
            logger.error(f"✗ Failed to evaluate {task_name}: {e}")
            results[task_name] = {"error": str(e)}
    
    # Save results summary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = Path(model_path).parent.name
    results_file = os.path.join(output_dir, f"{model_name}_summary_{timestamp}.json")
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Results saved to: {results_file}")
    
    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 80)
    
    for task_name, task_result in results.items():
        if "error" in task_result:
            logger.info(f"{task_name}: ERROR - {task_result['error']}")
        elif task_name in task_result and 'test' in task_result[task_name]:
            test_scores = task_result[task_name]['test']
            
            # Print NDCG@10 prominently
            if 'ndcg_at_10' in test_scores:
                logger.info(f"{task_name} - NDCG@10: {test_scores['ndcg_at_10']:.4f}")
            else:
                # Print first available metric
                first_metric = next(iter(test_scores.items()))
                logger.info(f"{task_name} - {first_metric[0]}: {first_metric[1]:.4f}")
    
    return results


def compare_with_baseline(results_file: str, baseline_file: str = None):
    """
    Compare evaluation results with baseline (dense model).
    
    Args:
        results_file: Path to sparse model results JSON
        baseline_file: Path to baseline (dense) model results JSON
    """
    logger.info("=" * 80)
    logger.info("COMPARISON WITH BASELINE")
    logger.info("=" * 80)
    
    with open(results_file, 'r') as f:
        sparse_results = json.load(f)
    
    if baseline_file and os.path.exists(baseline_file):
        with open(baseline_file, 'r') as f:
            baseline_results = json.load(f)
        
        logger.info("\n{:<20} {:<15} {:<15} {:<15}".format(
            "Task", "Sparse", "Dense", "Difference"
        ))
        logger.info("-" * 70)
        
        for task_name in sparse_results:
            if task_name in baseline_results:
                sparse_score = sparse_results[task_name].get('test', {}).get('ndcg_at_10')
                baseline_score = baseline_results[task_name].get('test', {}).get('ndcg_at_10')
                
                if sparse_score and baseline_score:
                    diff = sparse_score - baseline_score
                    diff_pct = (diff / baseline_score) * 100
                    
                    logger.info("{:<20} {:<15.4f} {:<15.4f} {:<15}".format(
                        task_name,
                        sparse_score,
                        baseline_score,
                        f"{diff:+.4f} ({diff_pct:+.1f}%)"
                    ))
    else:
        logger.info("No baseline file provided or found. Skipping comparison.")


def analyze_sparsity(model_path: str, sample_texts: list = None):
    """
    Analyze sparsity of the sparse encoder model.
    
    Args:
        model_path: Path to trained sparse encoder model
        sample_texts: Optional list of sample texts to encode
    """
    logger.info("=" * 80)
    logger.info("SPARSITY ANALYSIS")
    logger.info("=" * 80)
    
    model = SparseEncoder(model_path)
    
    if sample_texts is None:
        sample_texts = [
            "Hva er hovedstaden i Norge?",
            "Oslo er hovedstaden i Norge",
            "Python er et programmeringsspråk",
            "Kunstig intelligens og maskinlæring",
        ]
    
    logger.info(f"Analyzing sparsity on {len(sample_texts)} sample texts\n")
    
    # Encode samples
    embeddings = model.encode(sample_texts, convert_to_sparse_tensor=True)
    
    vocab_size = embeddings.shape[1]
    
    for i, (text, embedding) in enumerate(zip(sample_texts, embeddings)):
        # Count non-zero dimensions
        active_dims = (embedding != 0).sum()
        sparsity = (1 - active_dims / vocab_size) * 100
        
        logger.info(f"Sample {i+1}: {text}")
        logger.info(f"  Active dimensions: {active_dims} / {vocab_size}")
        logger.info(f"  Sparsity: {sparsity:.2f}%")
        
        # Decode top-k tokens
        decoded = model.decode(embedding, top_k=10)
        logger.info(f"  Top-10 tokens: {decoded}")
        logger.info("")
    
    # Overall statistics
    total_active = sum((emb != 0).sum() for emb in embeddings)
    avg_active = total_active / len(embeddings)
    avg_sparsity = (1 - avg_active / vocab_size) * 100
    
    logger.info("=" * 80)
    logger.info(f"AVERAGE SPARSITY: {avg_sparsity:.2f}%")
    logger.info(f"AVERAGE ACTIVE DIMENSIONS: {avg_active:.1f} / {vocab_size}")
    logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Evaluate sparse encoder on MTEB tasks")
    parser.add_argument(
        "--model-path",
        dest="model_path",
        type=str,
        required=True,
        help="Path to trained sparse encoder model",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/mteb",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=None,
        help="List of MTEB tasks to evaluate (default: Norwegian tasks)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for encoding",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default=None,
        help="Path to baseline (dense model) results JSON for comparison",
    )
    parser.add_argument(
        "--analyze_sparsity",
        action="store_true",
        help="Run sparsity analysis on sample texts",
    )
    
    args = parser.parse_args()
    
    # Run evaluation
    evaluate_model(
        model_path=args.model_path,
        output_dir=args.output_dir,
        tasks=args.tasks,
        batch_size=args.batch_size,
    )
    
    # Analyze sparsity if requested
    if args.analyze_sparsity:
        analyze_sparsity(args.model_path)
    
    # Compare with baseline if provided
    if args.baseline:
        # Find the most recent results file
        results_files = sorted(Path(args.output_dir).glob("*_summary_*.json"))
        if results_files:
            compare_with_baseline(str(results_files[-1]), args.baseline)


if __name__ == "__main__":
    main()
