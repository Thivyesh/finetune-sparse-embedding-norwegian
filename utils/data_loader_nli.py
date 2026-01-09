"""
Data loading and preprocessing for NLI embedding model training.

This module handles:
1. Loading the NLI dataset from HuggingFace
2. Formatting it correctly for triplet training
3. Creating train/eval/test splits
"""

from datasets import load_dataset, Dataset
from typing import Optional, Tuple, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_triplet_dataset(
    dataset_name: str,
    train_split: str = "train",
    eval_split: str = "dev",
    test_split: str = "test",
    anchor_column: str = "anchor",
    positive_column: str = "positive",
    negative_column: str = "negative",
    max_train_samples: Optional[int] = None,
    max_eval_samples: Optional[int] = None,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Load and prepare triplet dataset for embedding training.

    Args:
        dataset_name: HuggingFace dataset identifier (e.g., "Fremtind/all-nli-norwegian")
        train_split: Name of training split in the dataset
        eval_split: Name of evaluation/validation split
        test_split: Name of test split
        anchor_column: Name of column containing anchor sentences
        positive_column: Name of column containing positive (similar) sentences
        negative_column: Name of column containing negative (dissimilar) sentences
        max_train_samples: Optional limit on training samples (for quick testing)
        max_eval_samples: Optional limit on eval samples (for quick testing)

    Returns:
        Tuple of (train_dataset, eval_dataset, test_dataset)
    """
    logger.info(f"Loading dataset: {dataset_name}")

    # Load dataset from HuggingFace
    try:
        dataset = load_dataset(dataset_name)
    except Exception as e:
        raise ValueError(
            f"Failed to load dataset '{dataset_name}'. "
            f"Please check that the dataset name is correct and you have internet access.\n"
            f"Error: {e}"
        )

    # Extract splits
    if train_split not in dataset:
        raise ValueError(
            f"Train split '{train_split}' not found in dataset. "
            f"Available splits: {list(dataset.keys())}"
        )

    train_dataset = dataset[train_split]
    eval_dataset = dataset.get(eval_split, None)
    test_dataset = dataset.get(test_split, None)

    logger.info(f"✓ Loaded training split: {len(train_dataset):,} samples")
    if eval_dataset:
        logger.info(f"✓ Loaded evaluation split: {len(eval_dataset):,} samples")
    if test_dataset:
        logger.info(f"✓ Loaded test split: {len(test_dataset):,} samples")

    # Verify required columns exist
    required_columns = [anchor_column, positive_column, negative_column]
    missing_columns = [col for col in required_columns if col not in train_dataset.column_names]

    if missing_columns:
        raise ValueError(
            f"Required columns {missing_columns} not found in dataset.\n"
            f"Available columns: {train_dataset.column_names}\n"
            f"Please check your column names."
        )

    # Limit dataset size if requested (useful for quick testing)
    if max_train_samples is not None and max_train_samples < len(train_dataset):
        logger.info(f"Limiting training samples to {max_train_samples:,}")
        train_dataset = train_dataset.select(range(max_train_samples))

    if eval_dataset and max_eval_samples is not None and max_eval_samples < len(eval_dataset):
        logger.info(f"Limiting evaluation samples to {max_eval_samples:,}")
        eval_dataset = eval_dataset.select(range(max_eval_samples))

    # Show example for verification
    logger.info("\n" + "="*70)
    logger.info("EXAMPLE TRIPLET FROM DATASET:")
    logger.info("="*70)
    example = train_dataset[0]
    logger.info(f"Anchor:   {example[anchor_column]}")
    logger.info(f"Positive: {example[positive_column]}")
    logger.info(f"Negative: {example[negative_column]}")
    logger.info("="*70 + "\n")

    return train_dataset, eval_dataset, test_dataset


def load_nli_data(
    dataset_name: str = "Fremtind/all-nli-norwegian",
    languages: List[str] = ["norwegian"],
    split_ratio: Tuple[float, float, float] = (0.98, 0.01, 0.01),
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Load NLI (Natural Language Inference) dataset for Norwegian.
    
    This is a convenience function that loads the Norwegian NLI dataset
    with sensible defaults for sparse encoder training.
    
    Args:
        dataset_name: HuggingFace dataset identifier
        languages: List of languages (for compatibility, currently only Norwegian)
        split_ratio: Train/dev/test split ratio
        
    Returns:
        Tuple of (train_dataset, dev_dataset, test_dataset)
    """
    return load_triplet_dataset(
        dataset_name=dataset_name,
        train_split="train",
        eval_split="dev",
        test_split="test",
        anchor_column="anchor",
        positive_column="positive",
        negative_column="negative",
    )


if __name__ == "__main__":
    # Test the loader
    print("\n" + "="*70)
    print("TESTING NLI DATA LOADER")
    print("="*70 + "\n")

    train, dev, test = load_nli_data()

    print("\nDataset sizes:")
    print(f"  Train: {len(train):,}")
    print(f"  Dev: {len(dev):,}")
    print(f"  Test: {len(test):,}")

    print("\nFirst few samples:")
    for i in range(min(3, len(train))):
        sample = train[i]
        print(f"\nSample {i+1}:")
        print(f"  Anchor: {sample['anchor']}")
        print(f"  Positive: {sample['positive']}")
        print(f"  Negative: {sample['negative']}")

    print("\n✓ NLI data loader test complete!")
