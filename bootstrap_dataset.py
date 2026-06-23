"""
Bootstrap Dataset Script: Converts the 52k multimodal image-caption dataset into a
text-to-speech dataset, extracts continuous (Whisper) & discrete (EnCodec) features,
and caches them for CPU training.
Split into Stage 1 (TTS file generation) and Stage 2 (Feature extraction) to avoid SAPI5 COM conflicts.
"""

import os
import sys
import json
import csv
import argparse
import time
import numpy as np
from scipy.io import wavfile
from pathlib import Path
from tqdm import tqdm
import pyttsx3

# Force stdout to UTF-8 for Windows terminal safety
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

def parse_args():
    parser = argparse.ArgumentParser(description="Bootstrap 52k dataset into TTS representations.")
    parser.add_argument(
        "--source_manifest",
        type=str,
        default=r"C:\Users\Navie\ideogram_finetune\dataset\manifest.json",
        help="Path to the source 52k manifest.json"
    )
    parser.add_argument(
        "--source_dataset_dir",
        type=str,
        default=r"C:\Users\Navie\ideogram_finetune\dataset",
        help="Root directory of the 52k dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=r"C:\Users\Navie\.gemini\antigravity\scratch\multimodal-audio-suite",
        help="Workspace directory for caching output"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=50,
        help="Number of samples to bootstrap (default: 50 for quick verification)"
    )
    parser.add_argument(
        "--whisper_model",
        type=str,
        default="openai/whisper-tiny",
        help="Whisper model path"
    )
    return parser.parse_args()

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
    
    source_manifest_path = Path(args.source_manifest)
    source_dir = Path(args.source_dataset_dir)
    workspace_dir = Path(args.output_dir)
    
    if not source_manifest_path.exists():
        print(f"Error: Source manifest.json not found at: {source_manifest_path}")
        return
        
    # Setup directories
    bootstrapped_dir = workspace_dir / "bootstrapped_ljspeech"
    wav_dir = bootstrapped_dir / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)
    
    cache_dir = workspace_dir / "cache_bootstrapped"
    cont_cache_dir = cache_dir / "continuous"
    disc_cache_dir = cache_dir / "discrete"
    
    cont_cache_dir.mkdir(parents=True, exist_ok=True)
    disc_cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Load manifest
    print(f"\nLoading 52k dataset manifest: {source_manifest_path}...")
    with open(source_manifest_path, "r", encoding="utf-8") as f:
        source_manifest = json.load(f)
        
    print(f"Found {len(source_manifest)} source items in manifest.")
    samples_to_process = source_manifest[:args.num_samples]
    print(f"Processing the first {len(samples_to_process)} samples...")
    
    # ============================================================
    # STAGE 1: Text-to-Speech WAV Generation (pyttsx3 / SAPI5)
    # ============================================================
    print("\n" + "="*60)
    print("                    STAGE 1: GENERATING TTS WAV FILES")
    print("="*60)
    
    print("Initializing pyttsx3 Text-to-Speech Engine...")
    tts_engine = pyttsx3.init()
    tts_engine.setProperty('rate', 155)
    
    samples_data = []
    
    for idx, item in enumerate(tqdm(samples_to_process, desc="Synthesizing Speech")):
        caption_rel = item["caption"]
        caption_path = source_dir / caption_rel
        file_id = f"BOOT{idx+1:05d}-0001"
        
        # Load high-level description
        if not caption_path.exists():
            continue
            
        try:
            with open(caption_path, "r", encoding="utf-8") as f:
                cap_data = json.load(f)
                text = cap_data.get("high_level_description", "")
        except Exception as e:
            print(f"\nError reading caption file {caption_path}: {e}")
            continue
            
        if not text:
            vlm_desc = item.get("vlm_desc", "")
            if vlm_desc:
                text = vlm_desc.split(".")[0] + "."
            else:
                continue
                
        # Clean text
        text = text.replace("\n", " ").strip()
        if len(text) > 150:
            space_idx = text[:150].rfind(" ")
            text = text[:space_idx] + "." if space_idx > 0 else text[:150] + "."
            
        temp_wav = wav_dir / f"{file_id}.wav"
        
        # Queue saving job
        tts_engine.save_to_file(text, str(temp_wav))
        
        samples_data.append({
            "id": file_id,
            "text": text,
            "wav_path": temp_wav,
            "original_image": item["image"]
        })
        
    print("Writing WAV files to disk via SAPI5 loop...")
    tts_engine.runAndWait()
    
    # Delete SAPI5 engine object to release COM bindings cleanly
    del tts_engine
    print("Stage 1 complete. SAPI5 resources released.")
    
    # ============================================================
    # STAGE 2: Feature Extraction (PyTorch / Whisper / Encodec)
    # ============================================================
    print("\n" + "="*60)
    print("                    STAGE 2: EXTRACTING & CACHING FEATURES")
    print("="*60)
    
    # Import deep learning libraries inside Stage 2 to isolate from SAPI5
    print("Importing PyTorch and Transformers...")
    import torch
    from transformers import WhisperProcessor, WhisperModel, AutoProcessor, EncodecModel
    
    print(f"Loading Whisper encoder ({args.whisper_model})...")
    whisper_proc = WhisperProcessor.from_pretrained(args.whisper_model, local_files_only=True)
    whisper_model = WhisperModel.from_pretrained(args.whisper_model, local_files_only=True)
    whisper_model.eval()
    
    print("Loading Encodec model (facebook/encodec_24khz)...")
    encodec_proc = AutoProcessor.from_pretrained("facebook/encodec_24khz", local_files_only=True)
    encodec_model = EncodecModel.from_pretrained("facebook/encodec_24khz", local_files_only=True)
    encodec_model.eval()
    
    manifest_bootstrapped = []
    
    for item in tqdm(samples_data, desc="Extracting Features"):
        file_id = item["id"]
        temp_wav = item["wav_path"]
        text = item["text"]
        
        if not temp_wav.exists():
            continue
            
        # Load audio wave
        try:
            orig_sr, audio_int16 = wavfile.read(str(temp_wav))
            audio_f32 = audio_int16.astype(np.float32) / 32768.0
            if len(audio_f32.shape) > 1:
                audio_f32 = audio_f32.mean(axis=-1)
        except Exception as e:
            print(f"\nWarning: Could not read WAV {temp_wav}. Error: {e}")
            continue
            
        duration = len(audio_f32) / orig_sr
        
        # Extract Whisper features (16kHz)
        audio_16k = resample_audio(audio_f32, orig_sr, 16000)
        w_inputs = whisper_proc(audio_16k, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            w_output = whisper_model.encoder(w_inputs.input_features).last_hidden_state
            w_output = w_output.squeeze(0) # shape [1500, hidden_dim]
            
        cont_file = cont_cache_dir / f"{file_id}.pt"
        torch.save(w_output, cont_file)
        
        # Extract Encodec features (24kHz)
        audio_24k = resample_audio(audio_f32, orig_sr, 24000)
        e_inputs = encodec_proc(raw_audio=audio_24k, sampling_rate=24000, return_tensors="pt")
        with torch.no_grad():
            e_output = encodec_model.encode(e_inputs.input_values, e_inputs.padding_mask)
            audio_codes = e_output.audio_codes.squeeze(0).squeeze(0) # shape [num_codebooks, seq_len]
            
        disc_file = disc_cache_dir / f"{file_id}.pt"
        torch.save(audio_codes, disc_file)
        
        # Add to manifest list
        manifest_bootstrapped.append({
            "id": file_id,
            "text": text,
            "duration": float(duration),
            "continuous_path": str(cont_file),
            "discrete_path": str(disc_file),
            "whisper_dim": int(w_output.shape[-1]),
            "num_codebooks": int(audio_codes.shape[0]),
            "codec_seq_len": int(audio_codes.shape[1]),
            "original_image": item["original_image"]
        })
        
    # Save the new manifest_bootstrapped.json
    manifest_path = workspace_dir / "manifest_bootstrapped.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_bootstrapped, f, indent=2)
        
    print("\n" + "="*60)
    print("                    BOOTSTRAP COMPLETED")
    print("="*60)
    print(f"Bootstrapped manifest saved to: {manifest_path}")
    print(f"Cached files saved to:          {cache_dir}")
    print(f"Total Bootstrapped items:       {len(manifest_bootstrapped)} samples")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
