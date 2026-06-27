---
name: embedding-reconstruction-and-repair
description: Diagnose and repair warped pre-trained text embeddings in adapter checkpoints by comparing with pristine base model weights and copying matching indices.
---

# Embedding Reconstruction & Repair

This skill provides diagnostic scripts to check if training has warped pre-trained model embeddings (which degrades general text logic) and run weight surgery to restore them.

## Diagnostics (Cosine Similarity)
Measure the cosine similarity between the adapter embedding weights and the pristine base model weights for the pre-trained vocabulary boundary (e.g. index 0 to 151,665):

```python
import torch

def check_embedding_corruption(base_embed, adapter_embed, limit=151665):
    cos = torch.nn.CosineSimilarity(dim=1)
    similarity = cos(base_embed[:limit], adapter_embed[:limit])
    print(f"Mean similarity: {similarity.mean().item():.6f}")
    print(f"Min similarity: {similarity.min().item():.6f}")
```

## Repair Surgery
If the minimum similarity drops below `0.99`, run in-place weight surgery to overwrite the corrupted rows with pristine base model weights:

```python
@torch.no_grad()
def repair_embeddings(model, clean_base_embed, limit=151665):
    # Overwrite only the pre-trained vocabulary range
    model.get_input_embeddings().weight.data[:limit].copy_(clean_base_embed[:limit])
    print(f"Successfully restored first {limit} rows of embedding weights.")
```
