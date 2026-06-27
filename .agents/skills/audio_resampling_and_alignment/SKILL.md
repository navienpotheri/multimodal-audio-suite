---
name: audio-resampling-and-alignment
description: Resample audio files locally using pure NumPy and SciPy linear interpolation to avoid heavy dependency overhead and optimize CPU digital signal processing (DSP).
---

# Audio Resampling & DSP Alignment

This skill outlines how to efficiently resample audio waveforms locally using native NumPy and SciPy linear interpolation, bypassing heavy library dependencies like `librosa` or `torchaudio` on CPU.

## Resampling Implementation

Use the following Python pattern to execute linear resampling for Whisper (16kHz) and EnCodec (24kHz) inputs:

```python
import numpy as np
import scipy.signal
from scipy.io import wavfile

def resample_audio(wav_path, target_sr=16000):
    # Read raw audio file
    orig_sr, audio_int16 = wavfile.read(wav_path)
    
    # Convert to float32 representation normalized to [-1.0, 1.0]
    audio_f32 = audio_int16.astype(np.float32) / 32768.0
    
    # If stereo, average channels to mono
    if len(audio_f32.shape) > 1:
        audio_f32 = audio_f32.mean(axis=-1)
        
    # Calculate target number of samples
    num_samples = int(len(audio_f32) * target_sr / orig_sr)
    
    # Execute linear resampling
    audio_resampled = scipy.signal.resample(audio_f32, num_samples)
    
    return audio_resampled, target_sr
```

## Guidelines
*   **Mono Normalization**: Always average multi-channel arrays into mono arrays to feed speech towers.
*   **Amplitude Scaling**: Maintain floating-point values in the range $[-1.0, 1.0]$ to prevent clipping or numerical instability in subsequent encoder blocks.
