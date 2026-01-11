# Sparse Embedding Training for Norwegian/Scandinavian Languages


Train state-of-the-art sparse embedding models (SPLADE, Inference-free SPLADE, CSR) for Norwegian, Danish, and Swedish using multi-dataset training. Train SPLADE sparse encoder models for Norwegian, Danish, and Swedish using multi-dataset training with round-robin sampling.



## 🎯 Features

## Overview



- **3 Sparse Architectures**: Regular SPLADE, Inference-free SPLADE, CSR. This project trains sparse embedding models (SPLADE architecture) using high-quality Scandinavian datasets. Sparse models offer:
[comment]: <> (Clean README: accurate project overview, conformed to current repo layout and commands)

## Highlights

- Three architectures: Regular SPLADE, Inference-free SPLADE, CSR
- Multi-dataset training (NLI, Scandi QA, DDSC)
- MLflow experiment tracking and HuggingFace Hub checkpoint push
- Mac M-series optimizations (bf16, MPS, dataloader settings)

## Project layout

Top-level layout (important files/folders):

```
finetune-sparse-embedding-norwegian/
├── configs/                # YAML training configs
├── scripts/                # Training and evaluation scripts
├── utils/                  # Data loaders and data formatters
├── docs/                   # Quickstart and architecture docs
├── models/                 # Saved checkpoints (gitignored)
├── requirements.txt
└── README.md
```

See `docs/QUICKSTART.md` and `docs/SPARSE_ARCHITECTURES.md` for more detailed guides.

## Quickstart

1) Install dependencies

```bash
# Using uv (if you manage envs with uv):
uv sync
# Or with pip:
pip install -r requirements.txt
```

2) Login to HuggingFace (if you will push checkpoints)

```bash
huggingface-cli login
```

3) Train a model

The canonical training script is `train_sparse_multidataset.py` at the repository root. Example runs:

```bash
# Regular SPLADE (best accuracy)
uv run python train_sparse_multidataset.py configs/training_config_splade_regular.yaml

# Inference-free SPLADE (fast query encoding)
uv run python train_sparse_multidataset.py configs/training_config_splade.yaml

# CSR (sparsify an existing dense model)
uv run python train_sparse_multidataset.py configs/training_config_csr.yaml
```

Notes:
- Use the `configs/` YAMLs to change hyperparameters, batch sizes and Hub settings.
- For macOS M-series, make sure to set `dataloader_num_workers: 0` and `dataloader_pin_memory: false` in the config.

4) Monitor training

```bash
# MLflow UI (recommended)
mlflow ui  # opens on http://localhost:5000 by default

# TensorBoard (optional)
tensorboard --logdir models/
```

## Evaluation

Use the evaluation script in `scripts/` to run MTEB or NanoBEIR-style evaluations:

```bash
uv run python scripts/evaluate_sparse_mteb.py --model-path models/your-checkpoint/final
```

### Resuming training

You can resume interrupted training from a checkpoint saved under the configured `training.output_dir`.

- Automatically resume from the latest checkpoint:

```bash
uv run python train_sparse_multidataset.py --resume configs/training_config_splade.yaml
```

- Resume from an explicit checkpoint directory:

```bash
uv run python train_sparse_multidataset.py --resume-from models/your-model/checkpoint-1000 configs/training_config_splade.yaml
```

Notes:
- The script will look for directories named `checkpoint-<num>` under the configured `training.output_dir` and pick the one with the highest numeric suffix when `--resume` is used.
- If no checkpoints are found, training will start from scratch and a warning will be logged.

## Configuration

Edit the YAML files in `configs/` to customize training. Important fields:

- `model.base_model`: base HuggingFace model id (e.g. `ltg/norbert3-base`)
- `model.architecture`: `splade`, `inference-free-splade`, or `csr`
- `training.per_device_train_batch_size`, `training.learning_rate`, `training.num_train_epochs`
- `training.push_to_hub`, `training.hub_model_id`, `training.hub_strategy`
- `loss.query_regularizer_weight`, `loss.document_regularizer_weight` (SPLADE)
- `loss.type`, `loss.aux_loss_weight` (CSR)

Primary metric used for model selection is `eval_NanoBEIR_mean_dot_ndcg@10` (higher is better).

## Datasets

The training pipeline combines several datasets (NLI, Scandi QA, DDSC). All datasets will be auto-downloaded on first run if not present locally.

Summary (approx):

- NLI: ~556k triplets
- Scandi QA: ~121k QA pairs (NorQuAD, OpenBookQA, ScandiQA, PAWS-X etc.)
- DDSC retrieval: ~949k pairs

Total: ~1.6M training samples across Norwegian, Danish and Swedish.

## Tips & Troubleshooting

- macOS M-series: set `bf16: true` in configs (if supported), `dataloader_num_workers: 0` and `dataloader_pin_memory: false`.
- If you hit OOM: reduce `per_device_train_batch_size`, enable `gradient_checkpointing` or use `gradient_accumulation_steps`.
- Make sure you have logged into HuggingFace before running a job that pushes checkpoints to the Hub.

## Example: training flow

1. Edit `configs/training_config_splade.yaml` (or other config) for hyperparameters and Hub settings.
2. Start training with one of the `uv run python scripts/...` commands shown above.
3. Monitor with MLflow and check that the best model (NDCG@10) is being saved and optionally pushed to the Hub.

## Code examples

Load models for retrieval:

```python
from sentence_transformers import SparseEncoder, SentenceTransformer

# Sparse (SPLADE/CSR) and dense encoders used for hybrid retrieval
sparse = SparseEncoder("thivy/splade-norbert3-scandinavian")
dense = SentenceTransformer("thivy/norbert4-scandinavian")

texts = ["Dette er en test.", "Hvordan går det?"]
sparse_embeddings = sparse.encode(texts)
dense_embeddings = dense.encode(texts)
```

## Resources

- HuggingFace Sparse Encoder Training Guide: https://huggingface.co/blog/train-sparse-encoder
- Sentence Transformers sparse encoder docs: https://sbert.net/docs/sparse_encoder/

## License

MIT License
