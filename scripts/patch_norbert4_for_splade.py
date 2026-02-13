"""
Patch NorBERT4-base for SPLADE compatibility.

Problem:
  NorBERT4's GptBertForMaskedLM.forward() applies `30 * sigmoid(x/7.5)` to the
  MLM logits (line 738 in modeling_gptbert.py). This maps all logits to (0, 30),
  making them strictly positive. SPLADE relies on ReLU(log(1+exp(logits))) to
  produce sparse vectors — ReLU can only zero out negative inputs. Since NorBERT4
  logits are always positive, SPLADE cannot achieve any sparsity.

Fix:
  Remove the `30 * sigmoid(x/7.5)` line so the model outputs raw logits (unbounded,
  can be negative), just like standard BERT/DistilBERT. This makes the model
  compatible with SPLADE's sparsity mechanism.

Usage:
  uv run python3 scripts/patch_norbert4_for_splade.py
"""

import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import snapshot_download


def patch_modeling_file(modeling_path: str) -> None:
    """
    Patch modeling_gptbert.py to remove the sigmoid activation from
    GptBertForMaskedLM.forward().
    
    The line `subword_prediction = 30 * torch.sigmoid(subword_prediction / 7.5)`
    is removed so that raw logits pass through directly.
    """
    with open(modeling_path, "r") as f:
        content = f.read()
    
    # The problematic line in GptBertForMaskedLM.forward()
    sigmoid_line = "        subword_prediction = 30 * torch.sigmoid(subword_prediction / 7.5)"
    
    # Count occurrences — there are 2: one in MaskedLM, one in CausalLM
    count = content.count(sigmoid_line)
    print(f"Found {count} occurrence(s) of sigmoid activation line")
    
    if count == 0:
        raise ValueError(
            "Could not find the sigmoid line in modeling_gptbert.py. "
            "The file may have already been patched or the format changed."
        )
    
    # We need to replace ONLY the one in GptBertForMaskedLM, not GptBertForCausalLM.
    # Strategy: find the line that comes right after `self.classifier(sequence_output)`
    # within the GptBertForMaskedLM class.
    
    # Replace in context: the classifier call + sigmoid line in MaskedLM
    old_block = (
        "        subword_prediction = self.classifier(sequence_output)\n"
        "        subword_prediction = 30 * torch.sigmoid(subword_prediction / 7.5)\n"
    )
    
    new_block = (
        "        subword_prediction = self.classifier(sequence_output)\n"
        "        # PATCHED for SPLADE: removed `30 * sigmoid(x/7.5)` activation.\n"
        "        # Raw logits (unbounded, can be negative) are needed for SPLADE's\n"
        "        # ReLU(log(1+exp(logits))) to produce sparse vectors.\n"
    )
    
    # Replace only the first occurrence (which is in GptBertForMaskedLM)
    patched_content = content.replace(old_block, new_block, 1)
    
    if patched_content == content:
        raise ValueError("Replacement failed — old block not found exactly as expected")
    
    with open(modeling_path, "w") as f:
        f.write(patched_content)
    
    print("✅ Patched GptBertForMaskedLM.forward() — removed sigmoid activation")


def main():
    hub_repo_id = "thivy/norbert4-base-splade"
    source_model = "ltg/norbert4-base"
    local_dir = Path("models/norbert4-base-splade-patched")
    
    print(f"=== Patching {source_model} for SPLADE compatibility ===\n")
    
    # Step 1: Download the original model
    print(f"1. Downloading {source_model}...")
    downloaded_path = snapshot_download(
        repo_id=source_model,
        local_dir=str(local_dir),
        ignore_patterns=["*.onnx", "onnx/*"],
    )
    print(f"   Downloaded to: {downloaded_path}\n")
    
    # Step 2: Patch the modeling file
    print("2. Patching modeling_gptbert.py...")
    modeling_path = local_dir / "modeling_gptbert.py"
    if not modeling_path.exists():
        raise FileNotFoundError(f"modeling_gptbert.py not found at {modeling_path}")
    patch_modeling_file(str(modeling_path))
    print()
    
    # Step 3: Verify the patch works by loading the model
    print("3. Verifying patched model loads correctly...")
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(str(local_dir), trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(str(local_dir), trust_remote_code=True)
    model.eval()
    
    # Quick test: check that logits can be negative
    test_text = "Hovedstaden i Norge er Oslo"
    inputs = tokenizer(test_text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    print(f"   Input: '{test_text}'")
    print(f"   Logits shape: {logits.shape}")
    print(f"   Logits min: {logits.min().item():.4f}")
    print(f"   Logits max: {logits.max().item():.4f}")
    print(f"   Logits mean: {logits.mean().item():.4f}")
    
    if logits.min().item() >= 0:
        print("   ⚠️  WARNING: Logits are still all non-negative! Patch may not have worked.")
    else:
        print("   ✅ Logits contain negative values — SPLADE compatible!")
    
    # Check SPLADE-style sparsity
    max_logits = logits.max(dim=1).values[0]
    splade_output = torch.relu(torch.log1p(torch.exp(max_logits)))
    nonzero = (splade_output > 0).sum().item()
    total = splade_output.shape[0]
    sparsity = 1.0 - nonzero / total
    print(f"   SPLADE nonzero dims: {nonzero}/{total} ({sparsity*100:.1f}% sparsity)")
    print()
    
    # Step 4: Push to HuggingFace Hub
    print(f"4. Pushing patched model to {hub_repo_id}...")
    from huggingface_hub import HfApi
    
    api = HfApi()
    api.create_repo(hub_repo_id, exist_ok=True, repo_type="model")
    
    # Upload all files
    api.upload_folder(
        folder_path=str(local_dir),
        repo_id=hub_repo_id,
        commit_message=(
            "Patch NorBERT4-base for SPLADE: remove sigmoid from MLM head\n\n"
            "Removed `30 * sigmoid(x/7.5)` from GptBertForMaskedLM.forward() so that\n"
            "raw logits (unbounded, can be negative) are output. This is required for\n"
            "SPLADE's ReLU(log(1+exp(logits))) to produce truly sparse vectors.\n\n"
            "Based on ltg/norbert4-base. Only modeling_gptbert.py was modified."
        ),
        ignore_patterns=[".git*", "__pycache__", "*.pyc"],
    )
    print(f"   ✅ Pushed to https://huggingface.co/{hub_repo_id}")
    print()
    
    print("=== Done! ===")
    print(f"Use '{hub_repo_id}' as base_model in your training config.")


if __name__ == "__main__":
    main()
