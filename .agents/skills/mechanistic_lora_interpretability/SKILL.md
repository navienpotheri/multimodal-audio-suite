---
name: mechanistic-lora-interpretability
description: Analyze fine-tuned LoRA checkpoints to inspect update magnitudes, run SVD, check for rank collapse, and locate layer drift.
---

# Mechanistic LoRA Interpretability

This skill enables agents to inspect and evaluate fine-tuned LoRA checkpoints to understand how representation learning was distributed across layers and modules.

## Analysis Workflow

1. **Frobenius Norm Computation**:
   Calculate the Frobenius norm ($\|\Delta W\|_F$) for the LoRA update matrix to locate where the largest weight shifts occurred:
   $$\Delta W = W_{up} \times W_{down}$$
   $$\|\Delta W\|_F = \sqrt{\sum_{i} \sum_{j} |\Delta W_{ij}|^2}$$

2. **Singular Value Decomposition (SVD)**:
   Decompose the update matrix to obtain singular values:
   $$\Delta W = U \Sigma V^T$$

3. **Effective Rank (Participation Ratio)**:
   Evaluate the dimensionality of the update subspace. If the Participation Ratio ($PR$) is close to 1, the update has collapsed into a 1-dimensional line:
   $$PR = \frac{(\sum_{i} \sigma_i)^2}{\sum_{i} \sigma_i^2}$$
   where $\sigma_i$ are the singular values.

## Implementation Guide

Use the following Python blueprint to run the analysis:

```python
import torch

def analyze_lora_layer(lora_A, lora_B, alpha, r):
    # lora_A: [r, in_dim]
    # lora_B: [out_dim, r]
    scaling = alpha / r
    delta_w = (lora_B @ lora_A) * scaling
    
    # Frobenius Norm
    frob_norm = torch.norm(delta_w, p='fro').item()
    
    # SVD
    U, S, V = torch.svd(delta_w)
    
    # Participation Ratio
    sum_s = torch.sum(S)
    sum_s2 = torch.sum(S**2)
    effective_rank = ((sum_s ** 2) / sum_s2).item()
    
    return frob_norm, effective_rank, S.tolist()
```
