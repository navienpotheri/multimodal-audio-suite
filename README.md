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

---

## ⚡ CPU Performance Optimizations & Improvisations

To make model training and generation feasible on a local, consumer-grade CPU system (22 logical cores, 32GB RAM), we introduced several critical performance optimizations and improvisations:

### 1. Offline Pre-Processing & Feature Caching
*   **The Problem**: Extracting continuous Whisper states and discrete EnCodec tokens during the training step requires running the Whisper encoder and EnCodec models on CPU in addition to Qwen. This increases the CPU overhead per step by over **15x**.
*   **Improvisation**: We decoupled feature extraction. The script `bootstrap_dataset.py` processes all WAV files once, caching Whisper and EnCodec tensors to disk (`cache_bootstrapped/` and `cache_trimmed/`). The training dataloader simply loads the pre-computed tensors, reducing step overhead to practically zero.

### 2. Quadratic Attention Scaling Mitigation (Duration Filtering)
*   **The Problem**: Self-attention compute scales quadratically $O(N^2)$ with sequence length. Longer audio samples (e.g. 10s-15s) create very long EnCodec token sequences, causing step times to explode on CPU.
*   **Improvisation**: We filtered the training manifest to samples with a duration $\le 4.5\text{s}$ (retaining 3,083 out of 8,000 samples). This capped the maximum sequence length to ~800 tokens, maintaining step times in a fast 8s-12s envelope.

### 3. Padding Elimination via Batch Size 1
*   **The Problem**: Batching sequences of different lengths requires padding them to the maximum length in the batch. Computing self-attention over padded tokens wastes massive CPU cycles on redundant calculations.
*   **Improvisation**: We set `batch_size = 1`. This entirely eliminates padding tokens, ensuring that 100% of the computed attention matrix belongs to actual target tokens.

### 4. KV-Cached Inference
*   **The Problem**: Generating 450 tokens autoregressively requires passing the entire accumulated prompt through Qwen at every step, which takes several minutes per generation.
*   **Improvisation**: We implemented key-value (KV) caching for both the conditional and unconditional generation flows. Instead of recalculating past keys and values, we retrieve them from memory, reducing inference time to **under 40 seconds** on CPU.

### 5. In-Place Gradient Masking
*   **The Problem**: PEFT's `modules_to_save` config duplicates the entire embedding and output projection head matrices, creating gradients and optimizer states for all 151,665 text tokens, consuming an extra 4.4 GB of RAM.
*   **Improvisation**: We froze the pre-trained base text tokens by directly zeroing their gradients in-place in the optimizer step, avoiding any text model weight degradation while focusing backpropagation resources solely on LoRA layers and the 2,048 new audio tokens.

### 6. Threading Optimization
*   **Improvisation**: We benchmarked CPU thread allocation, identifying that 8 physical threads is the memory-bandwidth sweet spot. This maximizes training efficiency without triggering scheduling overhead or thread thrashing.

---

## 🧪 Generalization Limitations & Project Conclusion (Final Scope)

This version of the local Multimodal Audio Suite concludes at the **Testing Novel Prompts & Generalization Failure** milestone. This represents the empirical limit of executing local text-to-audio cross-modal alignments on consumer CPU architectures using LoRA fine-tuning.

### 1. Generalization Test Results
Evaluating the model on unseen (out-of-distribution) prompts reveals that while the model learns to synthesize structured speech sequences for the training corpus, it fails to generalize to novel text configurations. During generation, the autoregressive feedback loop accumulates minor prediction errors and drifts into alternative semantic states.

Applying **Classifier-Free Guidance (CFG)** forces the model away from collapsed static/silence, steering it to construct alternative English sentences, but it remains unaligned with the target text:

#### Prompt 1: *"A young woman is sitting in a library reading a book."*
*   **Greedy Decoding**: *"and see you later."*
*   **CFG 1.5**: *"In that way, we can have our pain being so quick to record."*
*   **CFG 3.0**: *"I'll open it in that direction, there it is, remember, keep it."*

#### Prompt 2: *"There are no people visible in this image."*
*   **Greedy Decoding**: *"This idiom, clobinfig, not says..."*
*   **CFG 1.5**: *"The subject is simple and my shepherds understand things."*
*   **CFG 3.0**: *"The people who have been following this second video, they saw"*

### 2. Current Project Scope Limits
*   **Hardware Ceiling**: Generalization on a 3,000+ sample size corpus requires 10-15+ epochs of training to pull average loss down and resolve exposure bias. On CPU, this takes ~70-100 hours, setting a clear operational boundary for this local phase.
*   **Alignment baseline**: The codebase successfully implements local feature extraction, constrained interleaving logits control, scale-free EnCodec decoding, and loopback Whisper ASR verification, establishing a complete baseline pipeline.
