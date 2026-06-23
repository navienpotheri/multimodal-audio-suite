# 🎙️ Local CPU Multimodal Audio Suite

<div align="center">

[![Python Version](https://img.shields.io/badge/Python-3.11+-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-CPU_Only-EE4C2C.svg?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Models-orange.svg?style=for-the-badge)](https://huggingface.co)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

**An offline, CPU-optimized training and inference pipeline for Unified Vision+Audio-to-Text Fusion and Discrete Text-to-Speech (TTS).**

</div>

---

## 📖 Overview

The **Multimodal Audio Suite** is a 100% local framework designed for consumer-grade laptop CPUs. It provides:
1.  **Unified Fusion Model (Image+Audio-to-Text)**: Merges visual representation (CLIP patch embeddings) and acoustic context (Whisper continuous encoder states) through learned projector matrices to feed a text generation LLM (`Qwen/Qwen2.5-0.5B-Instruct`).
2.  **Discrete Text-to-Speech (TTS)**: Translates input text into discrete acoustic codes (`EnCodec` tokens) using a fine-tuned causal language adapter.
3.  **ASR Loopback Verification**: Automatically feeds synthesized speech back into a local Automatic Speech Recognition (ASR) model (`Whisper-Tiny`) to compute transcripts and measure performance.

---

## 🗂️ Repository Split (GitHub vs. Local)

To prevent bloating the Git history with gigabytes of binary weight files, caches, and datasets, we split the project files as follows:

| Target Location | File / Directory | Description |
| :--- | :--- | :--- |
| **🌐 GitHub (Tracked)** | `bootstrap_dataset.py` | Synthesizes dataset, extracts Whisper embeddings and EnCodec tokens |
| | `preprocess_audio.py` | Implements continuous Whisper and discrete EnCodec tokenizers |
| | `preprocess_vision.py` | Implements CLIP vision embedding extraction |
| | `model.py` | Defines projector layers, alignment models, and custom collators |
| | `train_fusion.py` | Training loops for the Image+Audio-to-Text Fusion model |
| | `train_v2v.py` | Training loops for the Discrete TTS (Text-to-Audio) model |
| | `manifest_bootstrapped.json` | Manifest pointing to the 1,000-sample dataset mappings |
| | `.gitignore` | Ignores heavy models, databases, caches, and temp audio outputs |
| **💻 Local Only (Ignored)** | `bootstrapped_ljspeech/` | 1,000 synthesized WAV voice files (~350MB) |
| | `cache/` | Downloaded HuggingFace model weights (Qwen, Whisper, EnCodec) |
| | `cache_bootstrapped/` | Extracted tensor features saved to disk to avoid CPU recalculation |
| | `*_adapter/` | LoRA training checkpoints and checkpoints |
| | `*.pt` | Projector weights (`audio_projector.pt`, `vision_projector.pt`) |
| | `*.wav` | Generated test or intermediate audios |

---

## 🛠️ Installation & Setup

### 1. Set Up Virtual Environment
Activate your shell and initialize Python dependencies:
```bash
# Create environment
python -m venv venv

# Activate (Windows CMD)
.\venv\Scripts\activate

# Install CPU PyTorch, Transformers, SoundFile, and Librosa
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers librosa soundfile accelerate peft pyttsx3
```

### 2. Run Dataset Bootstrapping
Generate the synthetic audio database locally using the operating system's native TTS engine, and cache the multi-modal features:
```bash
python bootstrap_dataset.py
```
*   **Step 2a (TTS Generation)**: Generates 1,000 WAV files matching sentences from the corpus in `bootstrapped_ljspeech/`.
*   **Step 2b (Feature Caching)**: Processes the WAV files to extract Whisper encoder states and EnCodec discrete audio tokens, storing them under `cache_bootstrapped/`.

---

## 🚀 Running Training and Verification

### 1. Train the Unified Fusion Model (Step 3)
Trains the projection matrices aligning visual and acoustic encodings with the Qwen text representation space:
```bash
python train_fusion.py
```

### 2. Train the Discrete TTS Model (Step 4)
Fine-tunes the autoregressive discrete token predictor to convert raw text into discrete EnCodec indices:
```bash
python train_v2v.py
```

### 3. End-to-End ASR Loopback Verification
Validates that generated voice files match the semantic content of the target text:
```bash
python scratch/test_asr.py
```

---

## 🔒 Audio Post-Processing Guidelines
All output waveforms are post-processed locally to guarantee high playback safety:
- **Peak Amplitude Normalization**: Hard-capped at **0.8** to prevent digital clipping.
- **Low-pass Filtering**: Audio frequencies are filtered at **7.5 kHz** to remove high-frequency digital noise and quantization artifacts from the EnCodec reconstructor.
