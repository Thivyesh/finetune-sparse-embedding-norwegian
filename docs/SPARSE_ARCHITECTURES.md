# Sparse Encoder Architectures - Quick Comparison

This project supports three sparse encoder architectures. Choose based on your needs:

## 1. Regular SPLADE (Simplest)
**Config:** `configs/training_config_splade_regular.yaml`

### Architecture
```
Query:    MLMTransformer → SpladePooling → Sparse Vector
Document: MLMTransformer → SpladePooling → Sparse Vector
```

### Characteristics
- ✅ **Simplest to train** - single learning rate, no router
- ✅ **Best accuracy** - full transformer expressiveness for queries
- ✅ **Interpretable** - can decode which tokens are active
- ❌ Query inference ~20-50ms (transformer forward pass)

### Use When
- You want the simplest setup
- Query latency of 20-50ms is acceptable
- You want best possible accuracy

### Training
```bash
uv run python train_sparse_multidataset.py configs/training_config_splade_regular.yaml
```

---

## 2. Inference-free SPLADE (Fastest Queries)
**Config:** `configs/training_config_splade.yaml`

### Architecture
```
Query:    SparseStaticEmbedding (lookup table) → Sparse Vector  [~1ms]
Document: MLMTransformer → SpladePooling → Sparse Vector        [offline]
```

### Characteristics
- ✅ **Instant query inference** - just lookup + multiply (~1ms)
- ✅ **Documents indexed offline** - speed doesn't matter
- ✅ **Production-ready** - optimal for real-time search
- ❌ More complex (Router with 2 paths + different learning rates)
- ❌ Slightly lower accuracy (queries less expressive)

### Use When
- Production search where query latency is critical
- You can index documents offline
- You want ~50x faster query inference

### Training
```bash
uv run python train_sparse_multidataset.py configs/training_config_splade.yaml
```

---

## 3. CSR - Contrastive Sparse Representation (Leverage Existing Model)
**Config:** `configs/training_config_csr.yaml`

### Architecture
```
Dense Transformer → Pooling → SparseAutoEncoder → Sparse Vector
(thivy/norbert4-base-scandinavian-embedding + sparse layer)
```

### Characteristics
- ✅ **Leverages your existing trained model** - reuses all learned knowledge
- ✅ **Faster training** - only training autoencoder layer
- ✅ **Better performance** - starts from already-optimized dense embeddings
- ✅ **Hybrid search ready** - works great with your dense model
- ❌ Not interpretable (no token-level breakdown like SPLADE)
- ❌ Query inference same speed as dense (~10-20ms)

### Use When
- You already have a good dense model trained on your data
- You want sparse embeddings that complement your dense model
- You want faster training than SPLADE from scratch
- Hybrid search (dense + sparse) is your goal

### Training
```bash
uv run python train_sparse_multidataset.py configs/training_config_csr.yaml
```

---

## Recommendation for Your Setup

Given that you have:
- ✅ **`thivy/norbert4-base-scandinavian-embedding`** - already trained on same data
- ✅ Multi-dataset training infrastructure
- ✅ M4 Max with bf16 support

### I recommend starting with CSR:

1. **Train CSR first** (fastest, leverages existing model)
   ```bash
   uv run python train_sparse_multidataset.py configs/training_config_csr.yaml
   ```

2. **Then train Regular SPLADE** (for comparison & interpretability)
   ```bash
   uv run python train_sparse_multidataset.py configs/training_config_splade_regular.yaml
   ```

3. **Optionally train Inference-free** (if query latency critical)
   ```bash
   uv run python train_sparse_multidataset.py configs/training_config_splade.yaml
   ```

---

## Architecture Details

### Model Sizes
- **Regular SPLADE**: Same size as base MLM model (~124M params for norbert3-base)
- **Inference-free SPLADE**: MLM + 50k-dim static embedding (~124M + 50k params)
- **CSR**: Dense model + autoencoder (~150M + small autoencoder)

### Embedding Dimensions
- **SPLADE models**: Vocabulary size (typically 30k-50k dimensions, ~1-5% non-zero)
- **CSR**: Configurable (default 256 active out of 768/1024 dense dims)

### Training Time Estimates (M4 Max, 1 epoch on ~1.6M samples)
- **CSR**: ~2-3 hours (only training autoencoder)
- **Regular SPLADE**: ~8-12 hours (training full transformer)
- **Inference-free SPLADE**: ~10-14 hours (training transformer + static embeddings)

### Query Inference Speed
- **Dense model**: ~10-20ms
- **CSR**: ~10-20ms (same as dense)
- **Regular SPLADE**: ~20-50ms (MLM forward pass)
- **Inference-free SPLADE**: ~1-2ms (just lookup + multiply)

---

## Hybrid Search Strategy

The ideal setup for production:
1. **Dense model** (already have): `thivy/norbert4-base-scandinavian-embedding`
2. **CSR sparse model** (train next): Works perfectly with your dense model
3. **Combine with RRF** (Reciprocal Rank Fusion): Fuse dense + sparse rankings

This gives you:
- Best of both worlds (semantic + lexical matching)
- 10-20% better retrieval than either alone
- Compatible models (same base model, trained on same data)
