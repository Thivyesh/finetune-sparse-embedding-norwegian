"""
Sparse Data Formatter

Converts datasets from dense embedding format to sparse encoder format.
Ensures compatibility with SparseEncoderTrainer expectations.

Key differences:
- Dense: {query, positive, negative} -> embeddings
- Sparse: {anchor, positive, negative} -> sparse vectors with sparsity regularization

This module handles column naming and format conversions.
"""

from datasets import Dataset
from typing import Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def format_nli_for_sparse(dataset: Dataset) -> Dataset:
    """
    Format NLI dataset for sparse encoder training.
    
    Input format: {anchor, positive, negative}
    Output format: Same (already compatible)
    
    Args:
        dataset: Dataset with NLI triplets
        
    Returns:
        Formatted dataset
    """
    # NLI data is already in correct format (anchor, positive, negative)
    logger.info(f"NLI dataset already in correct format: {dataset.column_names}")
    return dataset


def format_qa_for_sparse(dataset: Dataset) -> Dataset:
    """
    Format QA dataset for sparse encoder training.
    
    Input format: {query, positive} or {query, answer}
    Output format: {anchor, positive}
    
    Args:
        dataset: Dataset with QA pairs
        
    Returns:
        Formatted dataset with renamed columns
    """
    # Rename columns for consistency
    column_mapping = {}
    
    if 'query' in dataset.column_names:
        column_mapping['query'] = 'anchor'
    
    if 'answer' in dataset.column_names:
        column_mapping['answer'] = 'positive'
    elif 'positive' not in dataset.column_names and 'document' in dataset.column_names:
        column_mapping['document'] = 'positive'
    
    if column_mapping:
        dataset = dataset.rename_columns(column_mapping)
        logger.info(f"Renamed QA columns: {column_mapping}")
    
    return dataset


def format_ddsc_for_sparse(dataset: Dataset) -> Dataset:
    """
    Format DDSC retrieval dataset for sparse encoder training.
    
    Input format: {query, positive, negative, instruction, task, language}
    Output format: {anchor, positive, negative}
    
    Note: Keeps instruction/task/language if present for logging,
    but they won't be used in training.
    
    Args:
        dataset: Dataset with DDSC retrieval data
        
    Returns:
        Formatted dataset with renamed columns
    """
    column_mapping = {}
    
    if 'query' in dataset.column_names:
        column_mapping['query'] = 'anchor'
    
    # Keep positive and negative as-is (already correct names)
    
    if column_mapping:
        dataset = dataset.rename_columns(column_mapping)
        logger.info(f"Renamed DDSC columns: {column_mapping}")
    
    return dataset


def prepare_multidataset_for_sparse(
    datasets_dict: Dict[str, Dataset],
    dataset_types: Dict[str, str]
) -> Dict[str, Dataset]:
    """
    Prepare multiple datasets for sparse encoder training.
    
    Args:
        datasets_dict: Dictionary of dataset name -> Dataset
        dataset_types: Dictionary of dataset name -> type ('nli', 'qa', 'ddsc')
        
    Returns:
        Dictionary of formatted datasets
    """
    formatted = {}
    
    for name, dataset in datasets_dict.items():
        dataset_type = dataset_types.get(name, 'unknown')
        
        if dataset_type == 'nli':
            formatted[name] = format_nli_for_sparse(dataset)
        elif dataset_type == 'qa':
            formatted[name] = format_qa_for_sparse(dataset)
        elif dataset_type == 'ddsc':
            formatted[name] = format_ddsc_for_sparse(dataset)
        else:
            logger.warning(f"Unknown dataset type '{dataset_type}' for {name}, keeping as-is")
            formatted[name] = dataset
        
        logger.info(f"Formatted {name} ({dataset_type}): {formatted[name].column_names}")
    
    return formatted


def validate_sparse_dataset(dataset: Dataset, name: str) -> bool:
    """
    Validate that dataset has correct format for sparse encoder training.
    
    Required columns: At least 'anchor' and 'positive'
    Optional: 'negative'
    
    Args:
        dataset: Dataset to validate
        name: Dataset name for logging
        
    Returns:
        True if valid, False otherwise
    """
    required_cols = ['anchor', 'positive']
    missing = [col for col in required_cols if col not in dataset.column_names]
    
    if missing:
        logger.error(f"Dataset '{name}' missing required columns: {missing}")
        logger.error(f"Available columns: {dataset.column_names}")
        return False
    
    has_negatives = 'negative' in dataset.column_names
    logger.info(f"Dataset '{name}' validated: anchor, positive{', negative' if has_negatives else ''}")
    
    return True


def remove_none_negatives(dataset: Dataset) -> Dataset:
    """
    Remove samples with None negatives (in-batch negatives only).
    
    For sparse encoders, we typically want explicit negatives or none at all.
    This function filters out None negatives to use only in-batch negatives.
    
    Args:
        dataset: Dataset with optional 'negative' column
        
    Returns:
        Filtered dataset
    """
    if 'negative' not in dataset.column_names:
        return dataset
    
    # Filter out None negatives
    original_size = len(dataset)
    dataset = dataset.filter(lambda x: x['negative'] is not None)
    filtered_size = len(dataset)
    
    if filtered_size < original_size:
        logger.info(f"Filtered {original_size - filtered_size} samples with None negatives")
    
    return dataset


def add_dataset_name_column(dataset: Dataset, name: str) -> Dataset:
    """
    Add a column indicating which dataset each sample came from.
    Useful for debugging and analysis.
    
    Args:
        dataset: Dataset to add column to
        name: Name of the dataset
        
    Returns:
        Dataset with added 'dataset_name' column
    """
    dataset = dataset.add_column('dataset_name', [name] * len(dataset))
    logger.info(f"Added 'dataset_name' column to {name}")
    return dataset


if __name__ == "__main__":
    # Test formatting functions
    from datasets import Dataset as HFDataset
    
    # Test NLI format
    nli_data = {
        'anchor': ['What is AI?', 'Python is great'],
        'positive': ['Artificial Intelligence', 'Python programming language'],
        'negative': ['Cooking recipes', 'Java programming'],
    }
    nli_dataset = HFDataset.from_dict(nli_data)
    formatted_nli = format_nli_for_sparse(nli_dataset)
    print(f"NLI formatted: {formatted_nli.column_names}")
    
    # Test QA format
    qa_data = {
        'query': ['What is the capital of Norway?', 'Who wrote Python?'],
        'answer': ['Oslo', 'Guido van Rossum'],
    }
    qa_dataset = HFDataset.from_dict(qa_data)
    formatted_qa = format_qa_for_sparse(qa_dataset)
    print(f"QA formatted: {formatted_qa.column_names}")
    
    # Test validation
    validate_sparse_dataset(formatted_nli, 'test_nli')
    validate_sparse_dataset(formatted_qa, 'test_qa')
