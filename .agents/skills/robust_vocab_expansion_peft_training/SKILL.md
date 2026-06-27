---
name: robust-vocab-expansion-peft-training
description: Expand model tokenizer vocabulary with custom modality tokens, configure PEFT LoRA adapters, and freeze pre-trained text embeddings in-place during training.
---

# Robust Vocabulary Expansion & PEFT Training

This skill manages vocabulary expansion for multimodal tokens (such as EnCodec audio states) and implements safe, robust PEFT training loops that protect original text processing capabilities from degradation.

## Workflow Steps

1. **Vocabulary Expansion**:
   Register custom special tokens in the tokenizer and resize the model's token embeddings:
   ```python
   tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
   model.resize_token_embeddings(len(tokenizer))
   ```

2. **PEFT/LoRA Configuration**:
   Apply LoRA to target modules (such as all linear layers: `q_proj`, `k_proj`, etc.) and set up `embed_tokens` and `lm_head` in `modules_to_save` to enable parameter updates on expanded indices.

3. **In-Place Gradient Masking (Crucial)**:
   To prevent warping pre-trained text knowledge, zero the gradients for all original base text rows (indices `0` to `151,665` for Qwen) in the embedding and output head weight gradients right before the optimizer steps:
   ```python
   # Inside training loop after loss.backward():
   embed_grad = model.get_input_embeddings().weight.grad
   if embed_grad is not None:
       embed_grad[:original_vocab_size] = 0.0
       
   lm_head_grad = model.get_output_embeddings().weight.grad
   if lm_head_grad is not None:
       lm_head_grad[:original_vocab_size] = 0.0
       
   optimizer.step()
   ```
