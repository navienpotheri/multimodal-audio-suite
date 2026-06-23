"""
Phase 1: Preprocessing and Caching Script for CPU Multimodal Audio Suite.
Processes LJSpeech (real or synthetic) and caches:
- Continuous representations: Whisper-Tiny encoder states (shape [1500, 384])
- Discrete representations: EnCodec 24kHz codebook tokens (shape [num_codebooks, seq_len])
"""

import os
import sys
import json
import csv
import argparse
import torch
import numpy as np
from scipy.io import wavfile
from pathlib import Path
from tqdm import tqdm

# Force stdout to UTF-8 for Windows terminal safety
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

from transformers import (
    WhisperProcessor,
    WhisperModel,
    AutoProcessor,
    EncodecModel
)

def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess and cache audio features for CPU multimodal training.")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=None,
        help="Path to the LJSpeech directory (e.g. C:\\ljspeech\\LJSpeech-1.1). If omitted, a synthetic dataset is generated."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=r"C:\Users\Navie\.gemini\antigravity\scratch\multimodal-audio-suite",
        help="Target directory for workspace and cache output"
    )
    parser.add_argument(
        "--whisper_model",
        type=str,
        default="openai/whisper-tiny",
        help="Whisper model version (default: openai/whisper-tiny for speed on CPU)"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=30,
        help="Number of samples to process (useful to limit dataset size for fast CPU verification)"
    )
    return parser.parse_args()

def generate_synthetic_dataset(output_dir, num_samples=30):
    """
    Generates a synthetic speech dataset of 22050Hz WAV files and metadata.csv
    mimicking the structure of the LJSpeech-1.1 dataset.
    This guarantees the project runs immediately and locally on CPU without massive downloads.
    """
    print(f"Generating synthetic LJSpeech-style dataset ({num_samples} samples)...")
    dataset_path = Path(output_dir) / "synthetic_ljspeech"
    wav_path = dataset_path / "wavs"
    wav_path.mkdir(parents=True, exist_ok=True)
    
    # Simple sentences to synthesize
    sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "Generative audio models represent complex acoustic manifolds.",
        "By aligning speech encoders with language models, we hear natively.",
        "Discrete audio tokenization allows predicting sound autoregressively.",
        "We are training a multimodal frontier model on local CPU.",
        "Flow matching outputs enable ultra low latency speech synthesis.",
        "Parameters are efficient when using low rank adaptation strategies.",
        "Attention mechanisms route information across different sequence modalities.",
        "Deep learning architectures process sound as continuous frequency spectrograms.",
        "The neural codec compresses raw speech waveforms into discrete codebook indices."
    ]
    
    metadata = []
    
    for i in range(num_samples):
        file_id = f"LJ{i+1:03d}-0001"
        sentence = sentences[i % len(sentences)]
        
        # Generate a synthetic audio wave (modulated sine wave representing speech pitch)
        sr = 22050
        duration = 1.5 + (i % 3) * 0.5  # 1.5, 2.0, 2.5 seconds
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        
        # Core pitch frequency (varies to sound slightly dynamic)
        freq = 120 + (i % 4) * 20
        # Modulated wave: carrier wave modulated by low frequency envelope
        carrier = np.sin(2 * np.pi * freq * t)
        modulator = 0.5 * (1.0 + np.sin(2 * np.pi * 3 * t))
        audio = carrier * modulator
        
        # Normalize to 16-bit integer range
        audio = (audio * 32767).astype(np.int16)
        
        # Save WAV file
        file_path = wav_path / f"{file_id}.wav"
        wavfile.write(str(file_path), sr, audio)
        
        # Add to metadata (ID|Transcription|Normalized Transcription)
        metadata.append([file_id, sentence, sentence])
        
    # Write metadata.csv
    csv_path = dataset_path / "metadata.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="|")
        writer.writerows(metadata)
        
    print(f"Synthetic dataset created at: {dataset_path}")
    return dataset_path

def resample_audio(audio_data, orig_sr, target_sr):
    """Resamples mono audio data to target_sr using fast linear interpolation."""
    if orig_sr == target_sr:
        return audio_data
        
    duration = len(audio_data) / orig_sr
    num_samples = int(round(duration * target_sr))
    
    x_old = np.linspace(0, duration, len(audio_data))
    x_new = np.linspace(0, duration, num_samples)
    resampled = np.interp(x_new, x_old, audio_data)
    return resampled

def main():
    args = parse_args()
    
    workspace_dir = Path(args.output_dir)
    workspace_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Setup dataset path
    dataset_dir = args.dataset_dir
    if not dataset_dir:
        dataset_path = generate_synthetic_dataset(workspace_dir, args.num_samples)
    else:
        dataset_path = Path(dataset_dir)
        if not dataset_path.exists():
            print(f"Error: Specified dataset path does not exist: {dataset_dir}")
            return
            
    wav_dir = dataset_path / "wavs"
    metadata_csv = dataset_path / "metadata.csv"
    
    if not wav_dir.exists() or not metadata_csv.exists():
        print(f"Error: {dataset_path} does not match LJSpeech structure (missing wavs/ or metadata.csv)")
        return
        
    # Setup cache directories
    cache_dir = workspace_dir / "cache"
    cont_cache_dir = cache_dir / "continuous"
    disc_cache_dir = cache_dir / "discrete"
    
    cont_cache_dir.mkdir(parents=True, exist_ok=True)
    disc_cache_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("                    INITIALIZING FEATURE EXTRACTORS")
    print("="*60)
    
    # Load Whisper models
    print(f"Loading Whisper encoder ({args.whisper_model})...")
    whisper_proc = WhisperProcessor.from_pretrained(args.whisper_model, local_files_only=True)
    whisper_model = WhisperModel.from_pretrained(args.whisper_model, local_files_only=True)
    whisper_model.eval()
    
    # Load Encodec models
    print("Loading Encodec model (facebook/encodec_24khz)...")
    encodec_proc = AutoProcessor.from_pretrained("facebook/encodec_24khz", local_files_only=True)
    encodec_model = EncodecModel.from_pretrained("facebook/encodec_24khz", local_files_only=True)
    encodec_model.eval()
    
    print("Models loaded successfully.\n")
    
    # 2. Parse metadata
    print("Parsing dataset metadata...")
    samples = []
    with open(metadata_csv, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="|")
        for row in reader:
            if not row or len(row) < 2:
                continue
            file_id = row[0]
            transcript = row[1]
            wav_file = wav_dir / f"{file_id}.wav"
            if wav_file.exists():
                samples.append({
                    "id": file_id,
                    "transcript": transcript,
                    "wav_path": wav_file
                })
                
    # Limit samples
    samples = samples[:args.num_samples]
    print(f"Found {len(samples)} valid audio samples to process.")
    
    # 3. Process and Cache Features
    manifest = []
    
    print("\n" + "="*60)
    print("                    EXTRACTING & CACHING FEATURES")
    print("="*60)
    
    for item in tqdm(samples, desc="Preprocessing"):
        file_id = item["id"]
        wav_path = item["wav_path"]
        transcript = item["transcript"]
        
        # Load audio wave
        try:
            orig_sr, audio_int16 = wavfile.read(str(wav_path))
            # Convert to float32 normalized [-1.0, 1.0]
            audio_f32 = audio_int16.astype(np.float32) / 32768.0
            # If stereo, convert to mono
            if len(audio_f32.shape) > 1:
                audio_f32 = audio_f32.mean(axis=-1)
        except Exception as e:
            print(f"\nWarning: Could not read {wav_path}. Error: {e}")
            continue
            
        duration = len(audio_f32) / orig_sr
        
        # --- 3a. Extract Whisper features (16kHz) ---
        audio_16k = resample_audio(audio_f32, orig_sr, 16000)
        # Whisper processor returns padded features to 30s
        w_inputs = whisper_proc(audio_16k, sampling_rate=16000, return_tensors="pt")
        
        with torch.no_grad():
            w_output = whisper_model.encoder(w_inputs.input_features).last_hidden_state
            # w_output shape: [1, 1500, hidden_dim]
            # Squeeze batch size to save space
            w_output = w_output.squeeze(0) # shape [1500, hidden_dim]
            
        cont_file = cont_cache_dir / f"{file_id}.pt"
        torch.save(w_output, cont_file)
        
        # --- 3b. Extract Encodec features (24kHz) ---
        audio_24k = resample_audio(audio_f32, orig_sr, 24000)
        e_inputs = encodec_proc(raw_audio=audio_24k, sampling_rate=24000, return_tensors="pt")
        
        with torch.no_grad():
            # Get discrete token codes
            e_output = encodec_model.encode(e_inputs.input_values, e_inputs.padding_mask)
            # audio_codes shape: [1, 1, num_codebooks, seq_len]
            # Squeeze batch and frame indices
            audio_codes = e_output.audio_codes.squeeze(0).squeeze(0) # shape [num_codebooks, seq_len]
            
        disc_file = disc_cache_dir / f"{file_id}.pt"
        torch.save(audio_codes, disc_file)
        
        # Add to manifest list
        manifest.append({
            "id": file_id,
            "text": transcript,
            "duration": float(duration),
            "continuous_path": str(cont_file),
            "discrete_path": str(disc_file),
            "whisper_dim": int(w_output.shape[-1]),
            "num_codebooks": int(audio_codes.shape[0]),
            "codec_seq_len": int(audio_codes.shape[1])
        })
        
    # Write manifest.json
    manifest_path = workspace_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        
    print("\n" + "="*60)
    print("                    PREPROCESSING COMPLETED")
    print("="*60)
    print(f"Manifest saved to: {manifest_path}")
    print(f"Cached files saved to: {cache_dir}")
    print(f"Continuous Whisper states: {len(list(cont_cache_dir.glob('*.pt')))} files")
    print(f"Discrete Encodec codes:     {len(list(disc_cache_dir.glob('*.pt')))} files")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
