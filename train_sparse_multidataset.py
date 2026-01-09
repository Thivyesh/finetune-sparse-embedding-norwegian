# Full implementation follows. Edit this root-level file to change training behavior.
"""
Multi-Dataset Sparse Encoder Training Script

Trains SPLADE (inference-free or regular) sparse encoder models using multiple
Scandinavian datasets with round-robin sampling.

Reuses data loaders from dense embedding training project.

Based on: https://huggingface.co/blog/train-sparse-encoder
"""

import logging
import os
import sys
import yaml
from typing import Dict, Optional
import mlflow

from datasets import DatasetDict
from sentence_transformers import (
    SparseEncoder,
    SparseEncoderModelCardData,
    SparseEncoderTrainer,
    SparseEncoderTrainingArguments,
)
from sentence_transformers.models import Router
from sentence_transformers.sparse_encoder.evaluation import SparseNanoBEIREvaluator
from sentence_transformers.sparse_encoder.losses import (
    SparseMultipleNegativesRankingLoss,
    SpladeLoss,
    CSRLoss,
)
from transformers import EarlyStoppingCallback
from sentence_transformers.sparse_encoder.models import (
    MLMTransformer,
    SpladePooling,
    SparseStaticEmbedding,
    SparseAutoEncoder,
)
from sentence_transformers.models import Transformer, Pooling
from sentence_transformers.training_args import BatchSamplers, MultiDatasetBatchSamplers

# Import data loaders (copied from dense training project)
from utils.data_loader_nli import load_nli_data
from utils.data_loader_ddsc import load_ddsc_data
from utils.data_loader_scandi_qa import load_scandi_qa_data

# Import local utilities
from utils.sparse_data_formatter import (
    format_nli_for_sparse,
    format_qa_for_sparse,
    format_ddsc_for_sparse,
    validate_sparse_dataset,
)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    logger.info(f"Loaded config from {config_path}")
    return config


def create_sparse_model(config: dict) -> SparseEncoder:
    """
    Create sparse encoder model based on configuration.
    
    Supports:
    - Regular SPLADE: MLMTransformer + SpladePooling
    - Inference-free SPLADE: Router with query (static) + document (MLM) paths
    - CSR: Dense Transformer + Pooling + SparseAutoEncoder
    """
    model_config = config['model']
    base_model = model_config['base_model']
    architecture = model_config['architecture']
    
    logger.info(f"Creating {architecture} model from {base_model}")
    
    if architecture == "inference-free-splade":
        # Inference-free: Static embeddings for queries, MLM for documents
        pooling_strategy = model_config.get('pooling_strategy', 'max')
        mlm_transformer = MLMTransformer.load(
            base_model,
            trust_remote_code=True,
            tokenizer_kwargs={"model_max_length": config['training']['max_seq_length']}
        )
        
        splade_pooling = SpladePooling(
            pooling_strategy=pooling_strategy,
            word_embedding_dimension=mlm_transformer.get_sentence_embedding_dimension()
        )
        
        router = Router.for_query_document(
            query_modules=[
                SparseStaticEmbedding(
                    tokenizer=mlm_transformer.tokenizer,
                    frozen=model_config.get('freeze_static_embedding', False)
                )
            ],
            document_modules=[mlm_transformer, splade_pooling],
        )
        
        model = SparseEncoder(
            modules=[router],
            similarity_fn_name="dot",
            model_card_data=create_model_card_data(config),
        )
        
        logger.info("Created inference-free SPLADE model with Router")
        
    elif architecture == "csr":
        # CSR: Sparsify existing dense embedding model
        # Load dense model (Transformer + Pooling already included)
        transformer = Transformer(base_model, model_args={"trust_remote_code": True})
        pooling = Pooling(transformer.get_word_embedding_dimension(), pooling_mode="mean")
        
        # Get CSR hyperparameters
        input_dim = transformer.get_word_embedding_dimension()
        hidden_dim = model_config.get('csr_hidden_dim', input_dim * 4)
        k = model_config.get('csr_k', 256)  # Top-k active dimensions
        k_aux = model_config.get('csr_k_aux', 512)  # For auxiliary loss
        
        sparse_autoencoder = SparseAutoEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            k=k,
            k_aux=k_aux,
        )
        
        model = SparseEncoder(
            modules=[transformer, pooling, sparse_autoencoder],
            model_card_data=create_model_card_data(config),
        )
        
        logger.info(f"Created CSR model with k={k}, hidden_dim={hidden_dim}")
        
    else:
        # Regular SPLADE: MLMTransformer + SpladePooling
        model = SparseEncoder(
            base_model,
            model_card_data=create_model_card_data(config),
            model_kwargs={"trust_remote_code": True}
        )
        
        logger.info("Created regular SPLADE model")
    
    return model


def create_model_card_data(config: dict) -> SparseEncoderModelCardData:
    """Create model card metadata."""
    card_config = config.get('model_card', {})
    
    return SparseEncoderModelCardData(
        language=card_config.get('language', ['no', 'da', 'sv']),
        license=card_config.get('license', 'mit'),
        model_name=card_config.get('model_name', 'SPLADE Scandinavian Multi-Dataset'),
    )


def load_datasets(config: dict) -> Dict[str, DatasetDict]:
    """
    Load all enabled datasets using existing data loaders.
    
    Returns:
        Dictionary mapping dataset name to DatasetDict with 'train', 'eval', 'test' splits
    """
    datasets = {}
    dataset_config = config['datasets']
    
    # Load NLI data
    if dataset_config['nli']['enabled']:
        logger.info("Loading NLI dataset...")
        nli_config = dataset_config['nli']
        
        nli_train, nli_dev, nli_test = load_nli_data(
            dataset_name=nli_config.get('source', 'Fremtind/all-nli-norwegian'),
            languages=nli_config.get('languages', ['norwegian']),
            split_ratio=tuple(nli_config.get('split_ratio', [0.98, 0.01, 0.01])),
        )
        
        # Format for sparse encoder
        nli_train = format_nli_for_sparse(nli_train)
        nli_dev = format_nli_for_sparse(nli_dev)
        nli_test = format_nli_for_sparse(nli_test)
        
        datasets['nli'] = DatasetDict({
            'train': nli_train,
            'eval': nli_dev,
            'test': nli_test,
        })
        
        logger.info(f"NLI dataset loaded: {len(nli_train)} train, {len(nli_dev)} dev, {len(nli_test)} test")
    
    # Load Scandi QA data
    if dataset_config['scandi_qa']['enabled']:
        logger.info("Loading Scandi QA dataset...")
        qa_config = dataset_config['scandi_qa']
        
        qa_train, qa_dev, qa_test = load_scandi_qa_data(
            use_norquad=qa_config.get('use_norquad', True),
            use_openbookqa=qa_config.get('use_openbookqa', True),
            use_scandiqa=qa_config.get('use_scandiqa', True),
            use_supervised_da=qa_config.get('use_supervised_da', True),
            use_paws=qa_config.get('use_paws', True),
            scandiqa_languages=qa_config.get('scandiqa_languages', ['no', 'da', 'sv']),
            paws_data_dir=qa_config.get('paws_data_dir', 'data/paws-x/x-final'),
        )
        
        # Format for sparse encoder
        qa_train = format_qa_for_sparse(qa_train)
        qa_dev = format_qa_for_sparse(qa_dev)
        qa_test = format_qa_for_sparse(qa_test)
        
        datasets['scandi_qa'] = DatasetDict({
            'train': qa_train,
            'eval': qa_dev,
            'test': qa_test,
        })
        
        logger.info(f"Scandi QA loaded: {len(qa_train)} train, {len(qa_dev)} dev, {len(qa_test)} test")
    
    # Load DDSC data
    if dataset_config['ddsc']['enabled']:
        logger.info("Loading DDSC dataset...")
        ddsc_config = dataset_config['ddsc']
        
        ddsc_train, ddsc_dev, ddsc_test = load_ddsc_data(
            languages=ddsc_config.get('languages', ['norwegian', 'danish', 'swedish']),
            tasks=ddsc_config.get('tasks', None),
            split_ratio=tuple(ddsc_config.get('split_ratio', [0.98, 0.01, 0.01])),
        )
        
        # Format for sparse encoder
        ddsc_train = format_ddsc_for_sparse(ddsc_train)
        ddsc_dev = format_ddsc_for_sparse(ddsc_dev)
        ddsc_test = format_ddsc_for_sparse(ddsc_test)
        
        datasets['ddsc'] = DatasetDict({
            'train': ddsc_train,
            'eval': ddsc_dev,
            'test': ddsc_test,
        })
        
        logger.info(f"DDSC loaded: {len(ddsc_train)} train, {len(ddsc_dev)} dev, {len(ddsc_test)} test")
    
    # Validate all datasets
    for name, dataset_dict in datasets.items():
        for split_name, dataset in dataset_dict.items():
            if not validate_sparse_dataset(dataset, f"{name}_{split_name}"):
                raise ValueError(f"Dataset validation failed for {name}_{split_name}")
    
    return datasets


def create_training_args(config: dict) -> SparseEncoderTrainingArguments:
    """Create training arguments from config."""
    train_config = config['training']
    loss_config = config['loss']
    router_config = config.get('router', {})
    hardware_config = config.get('hardware', {})
    
    # Determine batch sampler
    batch_sampler_str = train_config.get('batch_sampler', 'no_duplicates')
    batch_sampler = BatchSamplers.NO_DUPLICATES if batch_sampler_str == 'no_duplicates' else BatchSamplers.BATCH_SAMPLER
    
    # Determine multi-dataset batch sampler
    sampling_strategy = config['datasets'].get('sampling_strategy', 'round_robin')
    multi_dataset_sampler = (
        MultiDatasetBatchSamplers.ROUND_ROBIN if sampling_strategy == 'round_robin'
        else MultiDatasetBatchSamplers.PROPORTIONAL
    )
    
    args = SparseEncoderTrainingArguments(
        # Required
        output_dir=train_config['output_dir'],
        
        # Training parameters
        num_train_epochs=train_config['num_train_epochs'],
        per_device_train_batch_size=train_config['per_device_train_batch_size'],
        per_device_eval_batch_size=train_config['per_device_eval_batch_size'],
        gradient_accumulation_steps=train_config.get('gradient_accumulation_steps', 1),
        
        learning_rate=train_config['learning_rate'],
        warmup_ratio=train_config.get('warmup_ratio', 0.1),
        weight_decay=train_config.get('weight_decay', 0.01),
        max_grad_norm=train_config.get('max_grad_norm', 1.0),
        
        # Precision
        fp16=train_config.get('fp16', False),
        bf16=train_config.get('bf16', False),
        
        # Memory optimization
        gradient_checkpointing=hardware_config.get('gradient_checkpointing', False),
        
        # Batch sampling
        batch_sampler=batch_sampler,
        multi_dataset_batch_sampler=multi_dataset_sampler,
        
        # Evaluation
        eval_strategy=train_config.get('eval_strategy', 'steps'),
        eval_steps=train_config.get('eval_steps', 1000),
        
    # Logging
    logging_steps=train_config.get('logging_steps', 100),
    logging_dir=train_config.get('logging_dir', None),
        run_name=train_config.get('run_name', 'splade-training'),
        report_to=train_config.get('report_to', ['tensorboard']),
        
        # Saving
        save_strategy=train_config.get('save_strategy', 'steps'),
        save_steps=train_config.get('save_steps', 1000),
        save_total_limit=train_config.get('save_total_limit', 1),
        load_best_model_at_end=train_config.get('load_best_model_at_end', True),
        metric_for_best_model=train_config.get('metric_for_best_model', 'eval_loss'),
        greater_is_better=train_config.get('greater_is_better', False),  # True for metrics like NDCG, False for loss
        
        # HuggingFace Hub integration
        push_to_hub=train_config.get('push_to_hub', False),
        hub_model_id=train_config.get('hub_model_id', None),
        hub_strategy=train_config.get('hub_strategy', 'checkpoint'),  # "checkpoint" or "end"
        hub_private_repo=train_config.get('hub_private_repo', False),
                
        # Data loading (MUST be 0 on macOS MPS to avoid multiprocessing errors)
        dataloader_num_workers=train_config.get('dataloader_num_workers', 0),
        dataloader_pin_memory=train_config.get('dataloader_pin_memory', False),
    )
    
    # Add router mapping if using inference-free SPLADE
    if config['model'].get('use_router', False):
        router_mapping = router_config.get('mapping', {
            'anchor': 'query',
            'positive': 'document',
            'negative': 'document',
        })
        args.router_mapping = router_mapping
        
        # Add learning rate mapping for SparseStaticEmbedding (query path)
        # Higher LR for query embeddings, lower for document path MLM transformer
        lr_patterns = router_config.get('learning_rate_patterns', {})
        if lr_patterns:
            args.learning_rate_mapping = lr_patterns
            logger.info(f"Learning rate mapping: {lr_patterns}")
        
        logger.info(f"Router mapping: {router_mapping}")
    
    logger.info(f"Multi-dataset sampling: {sampling_strategy}")
    
    # Log hub configuration if enabled
    if train_config.get('push_to_hub'):
        logger.info(f"HuggingFace Hub push enabled: {train_config.get('hub_model_id')}")
        logger.info(f"Hub strategy: {train_config.get('hub_strategy', 'checkpoint')}")

    # Ensure logging_dir lives under the model output_dir (the model repo)
    logging_dir = train_config.get('logging_dir')
    if not logging_dir:
        # Default to output_dir/runs
        logging_dir = os.path.join(train_config['output_dir'], 'runs')
        args.logging_dir = logging_dir
        logger.info(f"No logging_dir set in config; defaulting to: {logging_dir}")
    else:
        # If a relative path was provided, make it relative to the output_dir
        if not os.path.isabs(logging_dir):
            logging_dir = os.path.join(train_config['output_dir'], logging_dir)
        args.logging_dir = logging_dir
        logger.info(f"Resolved logging_dir to: {logging_dir}")
    
    return args


def create_loss(model: SparseEncoder, config: dict):
    """Create loss function from config."""
    loss_config = config['loss']
    loss_type = loss_config.get('type', 'splade_loss')
    
    if loss_type == 'splade_loss':
        inner_loss = SparseMultipleNegativesRankingLoss(model=model)
        
        loss = SpladeLoss(
            model=model,
            loss=inner_loss,
            query_regularizer_weight=loss_config.get('query_regularizer_weight', 0),
            document_regularizer_weight=loss_config.get('document_regularizer_weight', 0.003),
        )
        
        logger.info(
            f"Created SpladeLoss with query_reg={loss_config.get('query_regularizer_weight')}, "
            f"doc_reg={loss_config.get('document_regularizer_weight')}"
        )
        
    elif loss_type == 'csr_loss':
        inner_loss = SparseMultipleNegativesRankingLoss(model=model)
        
        loss = CSRLoss(
            model=model,
            loss=inner_loss,
            aux_loss_weight=loss_config.get('aux_loss_weight', 0.1),
        )
        
        logger.info(f"Created CSRLoss with aux_loss_weight={loss_config.get('aux_loss_weight', 0.1)}")
        
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    
    return loss


def create_evaluator(config: dict) -> Optional[SparseNanoBEIREvaluator]:
    """Create evaluator for training."""
    eval_config = config.get('evaluation', {})
    evaluator_type = eval_config.get('evaluator', None)
    
    if evaluator_type == 'sparse_nanobeir':
        datasets = eval_config.get('nanobeir_datasets', ['msmarco', 'nfcorpus'])
        
        evaluator = SparseNanoBEIREvaluator(
            dataset_names=datasets,
            batch_size=config['training']['per_device_eval_batch_size'],
        )
        
        logger.info(f"Created NanoBEIR evaluator with datasets: {datasets}")
        return evaluator
    
    logger.info("No evaluator configured")
    return None


def prepare_multi_dataset_dict(datasets: Dict[str, DatasetDict]) -> tuple:
    """
    Prepare datasets for multi-dataset training.
    
    Filters out None values which cause issues with model card generation.
    
    Returns:
        (train_dataset_dict, eval_dataset_dict)
    """
    def has_no_none_values(example):
        """Check if sample has any None values."""
        return all(v is not None for v in example.values())
    
    train_dict = {}
    eval_dict = {}
    
    for name, dataset_dict in datasets.items():
        # Filter out None values for model card compatibility
        train_filtered = dataset_dict['train'].filter(has_no_none_values)
        eval_filtered = dataset_dict['eval'].filter(has_no_none_values)
        
        # Log filtering stats
        train_orig = len(dataset_dict['train'])
        train_final = len(train_filtered)
        if train_orig != train_final:
            logger.info(f"  {name} train: Filtered {train_orig - train_final} samples with None values ({train_final}/{train_orig} kept)")
        
        train_dict[name] = train_filtered
        eval_dict[name] = eval_filtered
    
    logger.info(f"Prepared multi-dataset training with {len(train_dict)} datasets:")
    for name, dataset in train_dict.items():
        logger.info(f"  - {name}: {len(dataset)} training samples")
    
    return DatasetDict(train_dict), DatasetDict(eval_dict)


def setup_mlflow(config: dict):
    """
    Setup MLflow experiment tracking.
    
    Args:
        config: Training configuration dictionary
    """
    
    train_config = config['training']
    mlflow_config = config.get('mlflow', {})
    
    # Set MLflow tracking URI (default: local mlruns directory)
    tracking_uri = mlflow_config.get('tracking_uri', 'mlruns')
    mlflow.set_tracking_uri(tracking_uri)
    logger.info(f"MLflow tracking URI: {tracking_uri}")
    
    # Set experiment name
    experiment_name = mlflow_config.get('experiment_name', 'sparse-encoder-training')
    mlflow.set_experiment(experiment_name)
    logger.info(f"MLflow experiment: {experiment_name}")
    
    # Start run with run name
    run_name = train_config.get('run_name', 'splade-training')
    mlflow.start_run(run_name=run_name)
    logger.info(f"MLflow run: {run_name}")
    
    # Log configuration parameters
    logger.info("Logging configuration to MLflow...")
    
    # Flatten nested config for MLflow params
    def flatten_dict(d, parent_key='', sep='_'):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, (list, tuple)):
                # Convert lists to strings for MLflow
                items.append((new_key, str(v)))
            else:
                items.append((new_key, v))
        return dict(items)
    
    flat_params = flatten_dict(config)
    
    # Filter out None values and complex objects
    # Also sanitize keys to remove invalid characters for MLflow
    def sanitize_mlflow_key(key):
        """Remove characters not allowed in MLflow parameter names."""
        # Replace invalid characters with underscores
        import re
        # MLflow allows: alphanumerics, _, -, ., space, :, /
        # Remove regex special chars like * and other problematic characters
        sanitized = re.sub(r'[^\w\-\.\s:/]', '_', key)
        # Truncate to 250 chars (MLflow limit)
        return sanitized[:250]
    
    mlflow_params = {
        sanitize_mlflow_key(k): v for k, v in flat_params.items()
        if v is not None and isinstance(v, (str, int, float, bool))
    }
    
    mlflow.log_params(mlflow_params)
    logger.info(f"Logged {len(mlflow_params)} parameters to MLflow")
    
    # Log tags
    tags = {
        'model_architecture': config['model']['architecture'],
        'base_model': config['model']['base_model'],
        'sampling_strategy': config['datasets'].get('sampling_strategy', 'round_robin'),
    }
    mlflow.set_tags(tags)
    
    return True


def log_dataset_stats_to_mlflow(datasets: Dict[str, DatasetDict]):
    """Log dataset statistics to MLflow."""

    metrics = {}
    for name, dataset_dict in datasets.items():
        metrics[f"dataset_{name}_train_size"] = len(dataset_dict['train'])
        metrics[f"dataset_{name}_eval_size"] = len(dataset_dict['eval'])
        metrics[f"dataset_{name}_test_size"] = len(dataset_dict['test'])
    
    mlflow.log_metrics(metrics)
    logger.info("Logged dataset statistics to MLflow")


def main(config_path: str):
    """Main training function."""
    logger.info("=" * 80)
    logger.info("SPARSE ENCODER MULTI-DATASET TRAINING")
    logger.info("=" * 80)
    
    # Load config
    config = load_config(config_path)
    
    # Setup MLflow experiment tracking
    setup_mlflow(config)
    
    # Create output directory
    os.makedirs(config['training']['output_dir'], exist_ok=True)
    
    # Load datasets
    logger.info("\n" + "=" * 80)
    logger.info("LOADING DATASETS")
    logger.info("=" * 80)
    datasets = load_datasets(config)
    
    # Log dataset statistics to MLflow
    log_dataset_stats_to_mlflow(datasets)
    
    # Prepare multi-dataset format
    train_dataset, eval_dataset = prepare_multi_dataset_dict(datasets)
    
    # Create model
    logger.info("\n" + "=" * 80)
    logger.info("CREATING MODEL")
    logger.info("=" * 80)
    model = create_sparse_model(config)
    logger.info(f"Model architecture:\n{model}")
    
    # Create loss
    logger.info("\n" + "=" * 80)
    logger.info("CREATING LOSS FUNCTION")
    logger.info("=" * 80)
    loss = create_loss(model, config)
    
    # Create training args
    logger.info("\n" + "=" * 80)
    logger.info("CREATING TRAINING ARGUMENTS")
    logger.info("=" * 80)
    args = create_training_args(config)
    
    # Create evaluator
    logger.info("\n" + "=" * 80)
    logger.info("CREATING EVALUATOR")
    logger.info("=" * 80)
    evaluator = create_evaluator(config)
    
    # Create trainer
    logger.info("\n" + "=" * 80)
    logger.info("CREATING TRAINER")
    logger.info("=" * 80)
    
    # Add early stopping callback if configured
    callbacks = []
    if config['training'].get('early_stopping_patience'):
        early_stopping_callback = EarlyStoppingCallback(
            early_stopping_patience=config['training']['early_stopping_patience'],
            early_stopping_threshold=config['training'].get('early_stopping_threshold', 0.0),
        )
        callbacks.append(early_stopping_callback)
        logger.info(f"Early stopping enabled: patience={config['training']['early_stopping_patience']}")
    
    trainer = SparseEncoderTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        evaluator=evaluator,
        callbacks=callbacks if callbacks else None,
    )
    
    logger.info("Trainer created successfully")
    
    # Train
    logger.info("\n" + "=" * 80)
    logger.info("STARTING TRAINING")
    logger.info("=" * 80)
    trainer.train()
    
    # Evaluate on test set
    if evaluator:
        logger.info("\n" + "=" * 80)
        logger.info("FINAL EVALUATION")
        logger.info("=" * 80)
        results = evaluator(model)
        logger.info(f"Final evaluation results: {results}")
    
    # Save final model
    final_model_path = os.path.join(config['training']['output_dir'], 'final')
    logger.info("\n" + "=" * 80)
    logger.info(f"SAVING FINAL MODEL to {final_model_path}")
    logger.info("=" * 80)
    model.save_pretrained(final_model_path)
    
    # Log model to MLflow
    if mlflow.active_run():
        logger.info("Logging model to MLflow...")
        mlflow.log_artifact(final_model_path, artifact_path="model")
        
        # Log final metrics if available
        if evaluator and results:
            mlflow.log_metrics({f"final_{k}": v for k, v in results.items() if isinstance(v, (int, float))})
    
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"Model saved to: {final_model_path}")
    
    if mlflow.active_run():
        logger.info(f"MLflow run ID: {mlflow.active_run().info.run_id}")
        logger.info(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
        mlflow.end_run()
    
    logger.info("Next steps:")
    logger.info("  1. Evaluate on MTEB: python evaluate_sparse_mteb.py")
    logger.info("  2. Test hybrid retrieval with dense model")
    logger.info("  3. Push to HuggingFace Hub")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train sparse encoder with multi-dataset")
    parser.add_argument(
        "config",
        type=str,
        help="Path to training configuration YAML file",
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)
    
    main(args.config)
