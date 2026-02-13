# SPLADE NorBERT4 Sparsity Issue & Solutions

## The Problem: 0% Sparsity Metric

The fine-tuned model reports **0% sparsity** (all 51,200 dimensions active) even though it's functionally sparse.

### Root Cause

NorBERT4's MLM head applies a sigmoid activation that forces all logits into the (0, 30) range:

```python
# In modeling_gptbert.py line 738
logits = 30 * torch.sigmoid(subword_prediction / 7.5)
```

SPLADE uses ReLU-based sparsity: `ReLU(log(1+exp(x)))` which produces exact zeros only from negative inputs. Since sigmoid always outputs positive values, ReLU can never produce zeros—the metric shows 0% sparsity despite the model learning small weights.

**This is not a bug.** The model works correctly; it just needs a threshold at inference.

---

## Fix #1: Testing (Current Model)

The model is **functionally sparse** when you apply a threshold at inference time.

### Quick Test

```python
from sentence_transformers import SparseEncoder

model = SparseEncoder('thivy/norbert4-base-splade-finetuned-scand', trust_remote_code=True)

query = "Hva er hovedstaden i Norge?"
doc = "Oslo er hovedstaden og den største byen i Norge."

q_emb = model.encode_query([query], convert_to_sparse_tensor=False)
d_emb = model.encode_document([doc], convert_to_sparse_tensor=False)

# Apply threshold for ~99% sparsity
threshold = 0.05
q_emb[q_emb < threshold] = 0
d_emb[d_emb < threshold] = 0

print(f"Query active dims: {(q_emb > 0).sum().item()} / 51200")
print(f"Doc active dims: {(d_emb > 0).sum().item()} / 51200")
```

### Full Verification

Run the included test script:

```bash
cd /home/azureuser/localfiles/finetune-sparse-embedding-norwegian
uv run python3 test_hf_model.py
```

**Expected results:**
- ✅ All 3 test queries rank correct documents first
- ✅ 99.1% query sparsity with threshold=0.05
- ✅ 99.7% document sparsity with threshold=0.05
- ✅ Top rankings preserved after thresholding

---

## Fix #2: Future Fine-tuning (Proper Sparsity)

To get correct 0% metric sparsity in future training, **remove the sigmoid** from NorBERT4:

### Step 1: Patch the Base Model

Download NorBERT4 and remove line 738 from `modeling_gptbert.py`:

```python
# BEFORE (line 738)
logits = 30 * torch.sigmoid(subword_prediction / 7.5)

# AFTER (line 738)
logits = subword_prediction  # Use raw logits instead
```

### Step 2: Use Patched Model for Training

Update your training config:

```yaml
model:
  base_model: "thivy/norbert4-base-splade"  # Use patched version
  architecture: "splade"
  pooling_strategy: "max"
  use_router: false
```

### Step 3: Train Normally

The patched model will produce negative logits, allowing ReLU to generate exact zeros and proper metric sparsity.

---

## Files & Resources

| File | Purpose |
|------|---------|
| `test_hf_model.py` | Comprehensive verification script (6 tests) |
| `models/splade-norbert4-base-regular-multidataset/final/` | Local trained model files |
| `thivy/norbert4-base-splade` | Patched base model on HuggingFace (for future training) |
| `thivy/norbert4-base-splade-finetuned-scand` | Current fine-tuned model (use with threshold) |

---

## Summary

| Aspect | Current Model | After Patching Base |
|--------|---------------|---------------------|
| Metric Sparsity | 0% (false) | 98-99% (accurate) |
| Functional Sparsity | 99%+ with threshold | 99%+ without threshold |
| Inference Requirement | Apply threshold=0.05 | No threshold needed |
| Ease of Use | Slightly harder | Simpler |

Both approaches work. Use the **current model** if you need it now; use the **patched approach** for cleaner metrics in future training.
