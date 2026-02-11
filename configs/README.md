# Configuration Files

This directory contains training configurations for sparse encoder models.

## Files

### `training_config_splade.yaml`
**Recommended starter configuration**

- Base model: `ltg/norbert3-xs` (42M parameters)
- Architecture: Inference-free SPLADE
- Sequence length: 256 tokens
- Batch size: 16
- Multi-dataset training with ROUND_ROBIN sampling
- Conservative regularization: 0 query, 3e-3 document

**Use this for:**
- Fast iteration and experimentation
- Testing hyperparameters
- Limited compute resources

**Training time:** ~8-12 hours on M2 Pro

### `training_config_splade_large.yaml`
**For production-quality models**

- Base model: `ltg/norbert4-base` (150M parameters)
- Architecture: Inference-free SPLADE
- Sequence length: 512 tokens
- Batch size: 8 (gradient accumulation: 2)
- Same multi-dataset setup
- Tuned regularization

**Use this for:**
- Best performance
- Final model for deployment
- After hyperparameter tuning with small model

**Training time:** ~24-30 hours on M2 Pro

### `training_config_splade_h100.yaml`
**H100-optimized inference-free SPLADE**

- Base model: `ltg/norbert4-large` (355M parameters)
- Architecture: Inference-free SPLADE (Router + SparseStaticEmbedding + MLMTransformer)
- Batch size: 32, eval batch: 64
- 8 dataloader workers, pinned memory, prefetch factor 4
- TF32 enabled, BF16 mixed precision, torch.compile

**Use this for:**
- Best quality sparse embeddings on H100 / CUDA GPUs
- Production deployment

**Training time:** ~2-3 hours on H100

### `training_config_splade_regular_h100.yaml`
**H100-optimized regular SPLADE (no router)**

- Base model: `ltg/norbert4-large` (355M parameters)
- Architecture: Regular SPLADE (MLMTransformer + SpladePooling)
- Batch size: 32, eval batch: 64
- All H100 optimizations (TF32, BF16, torch.compile, workers, pinning)

**Use this for:**
- Comparing regular SPLADE vs inference-free on H100
- When you want runtime query encoding

**Training time:** ~2-3 hours on H100

### `training_config_splade_large_h100.yaml`
**H100-optimized large inference-free SPLADE**

- Base model: `ltg/norbert4-large` (355M parameters)
- Architecture: Inference-free SPLADE
- Batch size: 32, eval batch: 64
- Gradient accumulation: 1 (batches already fit)
- All H100 optimizations

**Use this for:**
- Best performance production model on H100
- Largest model for maximum quality

**Training time:** ~2-3 hours on H100

### `training_config_csr_h100.yaml`
**H100-optimized CSR (Contextualized Sparse Representations)**

- Base model: `thivy/norbert4-base-scandinavian-embedding` (150M params + autoencoder)
- Architecture: CSR (Transformer + Pooling + SparseAutoEncoder)
- Batch size: 128, eval batch: 256
- All H100 optimizations

**Use this for:**
- Dense-to-sparse conversion approach
- Leveraging pre-trained dense embeddings

**Training time:** ~30-60 min on H100

## Multi-Dataset Configuration

All configs use the same three datasets:

1. **NLI** (~556k samples): Norwegian NLI triplets
2. **Scandi QA** (~100k samples): Multi-source QA pairs
3. **DDSC** (~949k samples): Nordic retrieval tasks

**Sampling:** ROUND_ROBIN ensures equal representation from each dataset.

## Key Parameters to Tune

### Regularization Weights
Control sparsity of embeddings:

```yaml
loss:
  query_regularizer_weight: 0      # 0 for inference-free, 3e-4 to 5e-4 for regular SPLADE
  document_regularizer_weight: 3e-3  # Higher = sparser (less storage, possibly lower performance)
```

**Target sparsity:**
- Queries (inference-free): ~5-10 active dimensions (99%+ sparse)
- Documents: ~1500-3000 active dimensions (90-95% sparse out of 30k vocab)

### Learning Rates
Inference-free SPLADE uses two learning rates:

```yaml
training:
  learning_rate: 2e-5           # For MLM transformer (document encoder)
  query_learning_rate: 1e-3     # For static query embeddings (50x higher!)
```

### Sequence Length
Longer sequences = better context, but more memory:

```yaml
training:
  max_seq_length: 256   # Start here
  # max_seq_length: 512   # Increase if needed and memory allows
```

## Customization

To create a custom config:

1. Copy `training_config_splade.yaml`
2. Modify parameters:
   - `model.base_model`: Choose base model
   - `training.per_device_train_batch_size`: Adjust for your GPU memory
   - `loss.*_regularizer_weight`: Tune sparsity
   - `datasets.*.enabled`: Enable/disable datasets
3. Save with descriptive name
4. Train: `uv run python train_sparse_multidataset.py configs/your_config.yaml`

## Inference-free vs Regular SPLADE

### Inference-free SPLADE (Recommended)
```yaml
model:
  architecture: "inference-free-splade"
  use_router: true
```

**Pros:**
- Near-instant query encoding (no neural network)
- Best for production search systems
- Lower query latency

**Cons:**
- Slightly lower performance (~1-2% NDCG@10)
- More complex architecture

### Regular SPLADE
```yaml
model:
  architecture: "splade"
  use_router: false

loss:
  query_regularizer_weight: 3e-4  # Must be non-zero!
```

**Pros:**
- Simpler architecture
- Potentially higher performance

**Cons:**
- Slower query encoding
- Both queries and documents need neural network

## Hardware Requirements

| Config | Model Size | VRAM (H100) | Training Time (H100) |
|--------|-----------|------------|----------------------|
| `training_config_splade_h100.yaml` | 355M params (norbert4-large) | ~50-60 GB | ~2-3 hours |
| `training_config_splade_regular_h100.yaml` | 355M params (norbert4-large) | ~50-60 GB | ~2-3 hours |
| `training_config_splade_large_h100.yaml` | 355M params (norbert4-large) | ~50-60 GB | ~2-3 hours |
| `training_config_csr_h100.yaml` | 150M+AE | ~30 GB | ~30-60 min |

### Mac (MPS) Configs
Use `training_config_splade.yaml`, `training_config_splade_regular.yaml`, `training_config_splade_large.yaml`, `training_config_csr.yaml` (legacy, smaller models).

### H100 / CUDA GPU Configs
Use `*_h100.yaml` variants with **norbert4-large** (355M params). Key differences from Mac configs:
- **device**: `cuda` instead of `mps`
- **Base model**: norbert4-large for best quality
- **Batch sizes**: 32-128 (leverages 96GB VRAM, norbert4-large needs ~2GB per sample)
- **Data loading**: 8 workers + pinned memory + prefetching (Mac MPS requires 0 workers)
- **TF32**: Enabled for ~3x faster FP32 matmuls on Ampere+ GPUs
- **torch.compile**: Fuses kernels, reduces CPU launch overhead

**Note:** Training time ~2-3 hours per epoch. Most datasets run 1 epoch, adjust `num_train_epochs` as needed.

## Monitoring Training

Track these metrics during training:

1. **Eval loss**: Should decrease steadily
2. **Sparsity**: Log active dimensions (see evaluation config)
3. **NDCG@10**: On NanoBEIR datasets (msmarco, nfcorpus)
4. **Memory usage**: Ensure no OOM errors

Use TensorBoard:
```bash
tensorboard --logdir models/splade-norbert3-multidataset
```

## After Training

1. **Evaluate on MTEB:**
  ```bash
  uv run python scripts/evaluate_sparse_mteb.py --model-path models/splade-norbert3-multidataset/final
  ```

2. **Analyze sparsity:**
  ```bash
  uv run python scripts/evaluate_sparse_mteb.py --model-path models/splade-norbert3-multidataset/final --analyze_sparsity
  ```

3. **Compare with dense baseline:**
   ```bash
   uv run python scripts/evaluate_sparse_mteb.py \
     --model-path models/splade-norbert3-multidataset/final \
     --baseline ../finetune-embedding-norwegian/results/mteb/norbert4_base_results.json
   ```

4. **Test hybrid retrieval** (combine sparse + dense)

5. **Push to HuggingFace Hub**
