---
name: offline-multimodal-cpu-caching
description: Decouple and cache multi-modal acoustic and visual features offline to optimize local CPU training pipelines and prevent CPU thrashing.
---

# Offline Multimodal CPU Caching

This skill coordinates local feature extraction and caching for visual and acoustic inputs to bypass expensive CPU feature-generation overhead during active training epochs.

## Extraction Blueprints

### 1. Acoustic Features
*   **Continuous States (Whisper)**: Resample audio to 16kHz, pass through Whisper encoder, extract hidden states, and cache to disk:
    ```python
    import torch
    import scipy.signal
    from transformers import WhisperProcessor, WhisperForConditionalGeneration

    def cache_whisper_states(wav_path, processor_path, model_path):
        # Load and resample audio to 16000Hz
        # Extract features and feed to WhisperForConditionalGeneration
        # Cache outputs: torch.save(encoder_outputs.last_hidden_state, cache_path)
    ```
*   **Discrete Tokens (EnCodec)**: Resample audio to 24kHz, pass through EnCodec, extract target codebooks, and cache:
    ```python
    from transformers import EncodecModel

    def cache_encodec_tokens(wav_path, model_path, num_codebooks=2):
        # Extract codes: codes = model.encode(audio_values, padding_mask)[0][0]
        # Cache codes: torch.save(codes[:num_codebooks, :], cache_path)
    ```

### 2. Visual Features
*   **Patch Embeddings (CLIP)**: Load CLIP vision tower, normalize images, extract non-CLS token patch embeddings, and cache to disk.

## Guidelines
*   **Decoupled Design**: Do not perform features extraction inside the dataloader `__getitem__` call.
*   **Local-Only Pathing**: Write output cache directories locally (e.g. `cache_bootstrapped/`) and add them to `.gitignore`.
