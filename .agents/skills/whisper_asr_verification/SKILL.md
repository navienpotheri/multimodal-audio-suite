---
name: whisper-asr-verification
description: Configure automated speech-to-text loopback verification using local Whisper models to transcribe generated voice waveforms and compute semantic matches.
---

# Whisper ASR Loopback Verification

This skill configures automated speech-to-text loopback verification pipelines to measure the semantic and phonetic accuracy of generated speech files.

## Workflow

1.  **Audio Ingest**: Resample generated speech to 16kHz and average to mono.
2.  **ASR Forward Pass**: Feed the resampled float32 array directly into `openai/whisper-tiny` locally.
3.  **Decoding**: Decode logits with `skip_special_tokens=True` to extract the transcript.
4.  **Semantic Match Comparison**: Run exact or fuzzy string comparison between the generated transcript and the target prompt text.

## Python Implementation

```python
import scipy.signal
import numpy as np
from scipy.io import wavfile
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

def run_asr_loopback(wav_path, model_id="openai/whisper-tiny"):
    # Read audio
    sr, audio_int16 = wavfile.read(wav_path)
    audio_f32 = audio_int16.astype(np.float32) / 32768.0
    if len(audio_f32.shape) > 1:
        audio_f32 = audio_f32.mean(axis=-1)
        
    # Resample to 16kHz for Whisper
    num_samples = int(len(audio_f32) * 16000 / sr)
    audio_16k = scipy.signal.resample(audio_f32, num_samples)
    
    # Run Whisper ASR
    processor = WhisperProcessor.from_pretrained(model_id, local_files_only=True)
    model = WhisperForConditionalGeneration.from_pretrained(model_id, local_files_only=True)
    
    input_features = processor(audio_16k, sampling_rate=16000, return_tensors="pt").input_features
    with torch.no_grad():
        predicted_ids = model.generate(input_features)
    
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription.strip()
```
