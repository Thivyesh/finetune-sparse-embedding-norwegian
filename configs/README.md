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

| Config | Model Size | Memory | Training Time |
|--------|-----------|--------|---------------|
| `training_config_splade.yaml` | 42M params | 8-16 GB | 8-12 hours |
| `training_config_splade_large.yaml` | 150M params | 24-32 GB | 24-30 hours |

**Note:** Times are for M2 Pro / modern CPU. With GPU, training is 3-5x faster.

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
