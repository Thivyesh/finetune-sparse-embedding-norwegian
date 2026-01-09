# Sparse Embedding Training - Quick Start

New to sparse encoders? Start here!

## What You'll Build

A SPLADE sparse encoder that:
- Produces interpretable embeddings (see which words matter)
- Uses 10-20x less storage than dense models
- Works great in hybrid search (combine with dense models)
- Achieves 95-98% of dense model performance

## Prerequisites

```bash
# Python 3.9+
python --version

# Install dependencies
pip install -r requirements.txt
```

**Note:** Data loaders are already included (copied from dense training project). No additional setup needed!

### For Learning & Experimentation
**Use:** `configs/training_config_splade.yaml`
- Small model (ltg/norbert3-xs, 42M params)
- Fast training (~8-12 hours)
- Low memory (8-16 GB)

### For Production
**Use:** `configs/training_config_splade_large.yaml`
- Large model (ltg/norbert4-base, 150M params)
- Best performance (~24-30 hours)
- Higher memory (24-32 GB)

## Train

```bash
# Start training (small model)
uv run python train_sparse_multidataset.py configs/training_config_splade.yaml

# On macOS, prevent sleep during training
caffeinate -i uv run python train_sparse_multidataset.py configs/training_config_splade.yaml
```

## Monitor Training

### TensorBoard
Watch TensorBoard:
```bash
tensorboard --logdir models/splade-norbert3-multidataset
```

### MLflow (Recommended)
View MLflow UI for richer experiment tracking:
```bash
mlflow ui
```

Then open http://localhost:5000 in your browser.

MLflow tracks:
- All hyperparameters
- Training metrics over time
- Dataset statistics
- Final model artifacts
- Comparison across runs

Key metrics to watch:
- **Eval loss**: Should steadily decrease
- **Sparsity**: Check active dimensions (target: <5% for queries, <10% for docs)
- **NDCG@10**: On NanoBEIR (msmarco, nfcorpus)

## Evaluate

After training completes:

```bash
# MTEB evaluation on Norwegian tasks
uv run python scripts/evaluate_sparse_mteb.py \
  --model-path models/splade-norbert3-multidataset/final

# Analyze sparsity
uv run python scripts/evaluate_sparse_mteb.py \
  --model-path models/splade-norbert3-multidataset/final \
  --analyze_sparsity

# Compare with dense baseline (adjust baseline path if needed)
uv run python scripts/evaluate_sparse_mteb.py \
  --model-path models/splade-norbert3-multidataset/final \
  --baseline ../finetune-embedding-norwegian/results/mteb/norbert4_base_results.json
```

## Use Your Model

### Basic Usage

```python
from sentence_transformers import SparseEncoder

# Load your trained model
model = SparseEncoder("models/splade-norbert3-multidataset/final")

# Encode texts
texts = ["Hva er hovedstaden i Norge?", "Oslo er hovedstaden"]
embeddings = model.encode(texts)

print(embeddings.shape)  # (2, 30522) - sparse vectors

# Decode to see which tokens are active
decoded = model.decode(embeddings[0], top_k=10)
print(decoded)  # Top 10 active tokens with weights
```

### Hybrid Search (Combine Sparse + Dense)

```python
from sentence_transformers import SparseEncoder, SentenceTransformer

# Load both models
sparse_model = SparseEncoder("models/splade-norbert3-multidataset/final")
dense_model = SentenceTransformer("thivy/norbert4-base-scandinavian-embedding")

# Encode documents with both
docs = ["Document 1...", "Document 2..."]
sparse_doc_embeddings = sparse_model.encode_document(docs)
dense_doc_embeddings = dense_model.encode(docs)

# Encode query with both
query = "Your search query"
sparse_query_embedding = sparse_model.encode_query(query)
dense_query_embedding = dense_model.encode(query)

# Retrieve with both
# ... combine results with Reciprocal Rank Fusion (RRF)
# See HuggingFace guide for full implementation
```

## Troubleshooting

### Issue: Import errors when training
**Solution:** Data loaders should already be in `utils/` in this repository. If you see import errors, verify that `utils/` contains the data loader files (`data_loader_nli.py`, `data_loader_ddsc.py`, `data_loader_scandi_qa.py`, `data_loader_paws.py`, `sparse_data_formatter.py`).

If you used an earlier project copy, ensure files were moved into `utils/` or copy them from the dense training repo.

### Issue: OOM (Out of Memory)
**Solution:** Reduce batch size in config:
```yaml
training:
  per_device_train_batch_size: 8  # or 4, or 2
  gradient_accumulation_steps: 2  # Keep effective batch size
```

### Issue: Training too slow
**Solution:**
1. Use smaller model (norbert3-xs instead of norbert4-base)
2. Reduce sequence length (256 instead of 512)
3. Enable FP16 if you have GPU: `fp16: true`

### Issue: Model is not sparse enough
**Solution:** Increase regularization in config:
```yaml
loss:
  document_regularizer_weight: 5e-3  # Increase from 3e-3
```

### Issue: Model performance is low
**Solution:**
1. Check if datasets loaded correctly (see training logs)
2. Train for more steps/epochs
3. Use larger base model
4. Reduce regularization (less sparse = better performance)

## Next Steps

1. **Tune Hyperparameters**: Adjust regularization weights, learning rates
2. **Scale Up**: Try larger model (norbert4-base) after tuning on small model
3. **Deploy**: Push to HuggingFace Hub, integrate into your application
4. **Hybrid Search**: Combine with dense model for best results

## Resources

- [HuggingFace Sparse Encoder Guide](https://huggingface.co/blog/train-sparse-encoder)
- [Sentence Transformers Docs](https://sbert.net/docs/sparse_encoder/training_overview.html)
- [SPLADE Paper](https://arxiv.org/abs/2109.10086)
- Dense training repo: `../finetune-embedding-norwegian/`

## Questions?

Check the main README.md for detailed documentation.
