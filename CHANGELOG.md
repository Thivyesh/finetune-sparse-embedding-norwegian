# Changelog

All notable changes to this project will be documented in this file.

## [2026-01-09] - Repository Restructuring & Complete Implementation

### Added
- **Three sparse encoder architectures**:
  - Regular SPLADE (MLMTransformer + SpladePooling)
  - Inference-free SPLADE (Router with query/document paths)
  - CSR (Contrastive Sparse Representation for dense model sparsification)
- **Complete training pipeline** with multi-dataset support (NLI, QA, DDSC)
- **MLflow integration** for experiment tracking
- **HuggingFace Hub auto-push** after each checkpoint
- **Early stopping** with configurable patience
- **Mac M-series optimizations** (bf16, MPS support, multiprocessing fixes)
- **Comprehensive documentation**:
  - README.md with quick start guide
  - docs/QUICKSTART.md for detailed setup
  - docs/SPARSE_ARCHITECTURES.md for architecture comparison

### Changed
- **Restructured project**:
  - Moved training scripts to `scripts/`
  - Moved documentation to `docs/`
  - Cleaned up root directory
- **Updated configs** with proper metric tracking (NDCG@10)
- **Fixed learning rate patterns** for Router architecture
- **Optimized checkpoint saving** (save_total_limit: 1)

### Removed
- `main.py` (unnecessary boilerplate)
- `setup_data_loaders.py` (data loaders already in utils/)
- `train_sparse_embedding/` folder (outdated structure)

### Fixed
- Learning rate YAML parsing (scientific notation → decimals)
- Router parameter naming for inference-free SPLADE
- `dataloader_num_workers: 0` for Mac MPS compatibility
- `greater_is_better: true` for NDCG metric tracking
- Early stopping callback integration

## Architecture Details

### Regular SPLADE
```
Input → MLMTransformer → SpladePooling → Sparse Vector (30k dims, 0.6% active)
```

### Inference-free SPLADE
```
Query: Input → SparseStaticEmbedding → Sparse Vector (~8 active dims)
Document: Input → MLMTransformer → SpladePooling → Sparse Vector (~180 active dims)
```

### CSR
```
Input → Dense Transformer → Pooling → SparseAutoEncoder → Sparse Vector (256 active dims)
```

## Training Configuration Summary

All configs now include:
- `save_total_limit: 1` (save disk space)
- `push_to_hub: true` (automatic backup)
- `hub_strategy: "checkpoint"` (push after each save)
- `metric_for_best_model: "eval_NanoBEIR_mean_dot_ndcg@10"`
- `greater_is_better: true`
- `bf16: true` (M4 Max optimization)
- `dataloader_num_workers: 0` (Mac compatibility)

## Performance Expectations (1 epoch, M4 Max)

| Architecture | Training Time | Query Speed | NDCG@10 | Active Dims |
|--------------|---------------|-------------|---------|-------------|
| Regular SPLADE | 8-12h | 20-50ms | ~0.52 | 180 (0.6%) |
| Inference-free | 10-14h | 1-2ms | ~0.50 | 8 queries |
| CSR | 2-3h | 10-20ms | ~0.48 | 256 (33%) |
