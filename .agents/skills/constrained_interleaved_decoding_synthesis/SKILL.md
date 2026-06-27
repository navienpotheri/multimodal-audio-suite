---
name: constrained-interleaved-decoding-synthesis
description: Implement constrained interleaved decoding for multi-codebook speech models, apply CFG on CPU with KV-caching, and post-process audio with low-pass filters.
---

# Constrained Interleaved Decoding & Audio Synthesis

This skill controls the autoregressive generation of interleaved audio sequences from causal language models, enforcing strict vocabulary routing constraints and performing high-quality post-processing.

## Core Implementations

### 1. Interleaving Logits Constraint
For multi-codebook speech models (like interleaved 2-codebook EnCodec representations), implement logits masking to alternate between allowed codebook tokens at each generation step:
```python
# Even step: only codebook 0 tokens are allowed
if step % 2 == 0:
    allowed = torch.cat([c0_token_ids, torch.tensor([audio_end_id])])
# Odd step: only codebook 1 tokens are allowed
else:
    allowed = torch.cat([c1_token_ids, torch.tensor([audio_end_id])])

mask = torch.ones_like(logits, dtype=torch.bool)
mask[allowed] = False
logits[mask] = -float('inf')
```

### 2. Classifier-Free Guidance (CFG) with KV-Caching
Run parallel conditional and unconditional flows, reusing past key-values to accelerate CPU generation:
```python
# Apply CFG formula
cfg_logits = uncond_logits + guidance_scale * (cond_logits - uncond_logits)
```

### 3. Audio Post-Processing & Normalization
*   **Peak Normalization**: Scale peak amplitudes to **0.8** to prevent clipping.
*   **Butterworth Low-pass Filter**: Apply a 6th-order Butterworth low-pass filter with a **7.5 kHz** cutoff to clean high-frequency digital noise and quantization artifacts from the EnCodec reconstructor.
