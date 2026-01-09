# Quick Reference Card

## 🚀 Train Models

```bash
# Regular SPLADE (best accuracy, interpretable)
uv run python train_sparse_multidataset.py configs/training_config_splade_regular.yaml

# Inference-free SPLADE (fastest queries ~1ms)
uv run python train_sparse_multidataset.py configs/training_config_splade.yaml

# CSR (leverage existing dense model)
uv run python train_sparse_multidataset.py configs/training_config_csr.yaml

# Large model (NorBERT4)
uv run python train_sparse_multidataset.py configs/training_config_splade_large.yaml
```

## 📊 Monitor Training

```bash
# MLflow (recommended)
mlflow ui
# → http://localhost:5000

# TensorBoard
tensorboard --logdir models/
```

## 🧪 Evaluate

```bash
# MTEB Norwegian tasks
uv run python scripts/evaluate_sparse_mteb.py --model-path models/your-model/final
```

## ⚙️ Key Config Parameters

```yaml
# Model
model:
  base_model: "ltg/norbert3-base"  # or "ltg/norbert4-base" or "thivy/norbert4-scandinavian"
  architecture: "inference-free-splade"  # or "splade" or "csr"

# Training
training:
  per_device_train_batch_size: 16  # Adjust for your GPU/memory
  learning_rate: 0.00002  # 2e-5 for SPLADE, 1e-4 for CSR
  bf16: true  # M4 Max: true, older GPUs: fp16
  num_train_epochs: 1
  
  # Checkpointing
  save_steps: 1000
  eval_steps: 1000
  metric_for_best_model: "eval_NanoBEIR_mean_dot_ndcg@10"
  greater_is_better: true
  
  # Hub
  push_to_hub: true
  hub_model_id: "thivy/your-model-name"
  hub_strategy: "checkpoint"  # or "end"

# Loss (SPLADE/Inference-free)
loss:
  query_regularizer_weight: 0  # 0 for inference-free, 0.0005 for regular
  document_regularizer_weight: 0.003

# Loss (CSR)
loss:
  type: "csr_loss"
  aux_loss_weight: 0.1
```

## 🎯 Architecture Decision Tree

```
Do you have a trained dense model?
├─ YES → Use CSR (fastest training, leverages existing model)
└─ NO
   └─ Is query latency critical?
      ├─ YES → Use Inference-free SPLADE (~1ms queries)
      └─ NO → Use Regular SPLADE (best accuracy)
```

## 📈 Performance Metrics

| Metric | Formula | Goal |
|--------|---------|------|
| NDCG@10 | Normalized ranking quality | Maximize |
| Query Sparsity | % non-zero dims in query | Minimize (storage) |
| Document Sparsity | % non-zero dims in doc | Minimize (storage) |
| Active Dims | Avg non-zero dims | Lower = sparser |

## 🔧 Troubleshooting

### Mac MPS Issues
```yaml
dataloader_num_workers: 0  # MUST be 0
dataloader_pin_memory: false  # MUST be false
```

### Out of Memory
```yaml
per_device_train_batch_size: 8  # Reduce batch size
gradient_accumulation_steps: 2  # Simulate larger batches
gradient_checkpointing: true  # Trade speed for memory
```

### Learning Rate Pattern Error
```yaml
# Ensure patterns match actual parameter names
learning_rate_patterns:
  ".*\\.query\\..*": 0.001  # For inference-free SPLADE
```

### HuggingFace Hub Push Fails
```bash
# Login first
huggingface-cli login

# Or set token
export HF_TOKEN="your_token_here"
```

## 📁 Important Paths

```
configs/              # Edit YAML configs here
scripts/              # Training and evaluation scripts
utils/                # Data loaders (usually don't need to edit)
models/               # Saved checkpoints (gitignored)
mlruns/               # MLflow tracking data
docs/                 # Full documentation
```

## 🔗 Quick Links

- **Architecture comparison**: [docs/SPARSE_ARCHITECTURES.md](docs/SPARSE_ARCHITECTURES.md)
- **Setup guide**: [docs/QUICKSTART.md](docs/QUICKSTART.md)
- **Training guide**: [HuggingFace Blog](https://huggingface.co/blog/train-sparse-encoder)
- **API docs**: [Sentence Transformers](https://sbert.net/docs/sparse_encoder/)
