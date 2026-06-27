import os
import sys
import json
import re
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

def resample_audio(audio_data, orig_sr, target_sr):
    if orig_sr == target_sr:
        return audio_data
    duration = len(audio_data) / orig_sr
    num_samples = int(round(duration * target_sr))
    x_old = np.linspace(0, duration, len(audio_data))
    x_new = np.linspace(0, duration, num_samples)
    resampled = np.interp(x_new, x_old, audio_data)
    return resampled

def trim_silence(audio_f32, threshold=0.005, pad_ms=50, sr=22050):
    # Find indices where amplitude exceeds threshold
    active_idx = np.where(np.abs(audio_f32) > threshold)[0]
    if len(active_idx) == 0:
        return audio_f32
    start = max(0, active_idx[0] - int(pad_ms * sr / 1000))
    end = min(len(audio_f32), active_idx[-1] + int(pad_ms * sr / 1000))
    return audio_f32[start:end]

def main():
    source_manifest_path = Path(r"C:\Users\Navie\ideogram_finetune\dataset\manifest.json")
    source_dir = Path(r"C:\Users\Navie\ideogram_finetune\dataset")
    workspace_dir = Path(r"C:\Users\Navie\.gemini\antigravity\scratch\multimodal-audio-suite")
    
    bootstrapped_dir = workspace_dir / "bootstrapped_ljspeech_trimmed"
    wav_dir = bootstrapped_dir / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)
    
    cache_dir = workspace_dir / "cache_trimmed"
    cont_cache_dir = cache_dir / "continuous"
    disc_cache_dir = cache_dir / "discrete"
    cont_cache_dir.mkdir(parents=True, exist_ok=True)
    disc_cache_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nLoading 52k dataset manifest: {source_manifest_path}...", flush=True)
    with open(source_manifest_path, "r", encoding="utf-8") as f:
        source_manifest = json.load(f)
        
    print(f"Found {len(source_manifest)} source items.", flush=True)
    
    # Extract unique clean texts
    seen_texts = set()
    candidate_samples = []
    
    for idx, item in enumerate(source_manifest):
        caption_rel = item["caption"]
        caption_path = source_dir / caption_rel
        
        if not caption_path.exists():
            continue
            
        try:
            with open(caption_path, "r", encoding="utf-8") as f:
                cap_data = json.load(f)
                full_text = cap_data.get("high_level_description", "")
        except Exception:
            continue
            
        if not full_text:
            vlm_desc = item.get("vlm_desc", "")
            if vlm_desc:
                full_text = vlm_desc
            else:
                continue
                
        # Split on sentence boundaries
        raw_sentences = re.split(r'(?<=[.!?])\s+', full_text)
        for s in raw_sentences:
            s_clean = s.replace("\n", " ").strip()
            # Basic filters: length 20-120 chars, no code-like text
            if 20 <= len(s_clean) <= 120 and "addCriterion" not in s_clean and s_clean not in seen_texts:
                seen_texts.add(s_clean)
                candidate_samples.append({
                    "text": s_clean,
                    "original_image": item["image"]
                })
            
    print(f"Extracted {len(candidate_samples)} unique clean candidate sentences.", flush=True)
    
    # Limit to target dataset size of 8,000 samples
    target_samples = candidate_samples[:8000]
    print(f"Selected first {len(target_samples)} samples for bootstrapping.", flush=True)
    
    # ============================================================
    # STAGE 1: Text-to-Speech WAV Generation & Silence Trimming
    # ============================================================
    print("\n" + "="*60, flush=True)
    print("                    STAGE 1: GENERATING TTS WAV FILES", flush=True)
    print("="*60, flush=True)
    
    print("Initializing pyttsx3 Text-to-Speech Engine...", flush=True)
    tts_engine = pyttsx3.init()
    tts_engine.setProperty('rate', 155)
    
    samples_data = []
    for idx, item in enumerate(tqdm(target_samples, desc="Queueing SAPI5 synthesis")):
        file_id = f"TRIM{idx+1:05d}-0001"
        text = item["text"]
        temp_wav = wav_dir / f"{file_id}.wav"
        
        tts_engine.save_to_file(text, str(temp_wav))
        
        samples_data.append({
            "id": file_id,
            "text": text,
            "wav_path": temp_wav,
            "original_image": item["original_image"]
        })
        
    print("Writing WAV files to disk...", flush=True)
    tts_engine.runAndWait()
    del tts_engine
    print("Stage 1 complete: WAV files generated.", flush=True)
    
    # --- Dynamic Silence Trimming Stage ---
    print("\nTrimming leading and trailing silence from synthesized WAVs...", flush=True)
    for item in tqdm(samples_data, desc="Trimming silence"):
        wav_path = item["wav_path"]
        if not wav_path.exists():
            continue
        try:
            sr, audio_int16 = wavfile.read(str(wav_path))
            audio_f32 = audio_int16.astype(np.float32) / 32768.0
            if len(audio_f32.shape) > 1:
                audio_f32 = audio_f32.mean(axis=-1)
                
            # Trim leading/trailing silence
            trimmed_audio = trim_silence(audio_f32, threshold=0.005, pad_ms=50, sr=sr)
            
            # Convert back to int16 and save over the original file
            audio_int16_trimmed = (trimmed_audio * 32767).clip(-32768, 32767).astype(np.int16)
            wavfile.write(str(wav_path), sr, audio_int16_trimmed)
        except Exception as e:
            print(f"\nWarning: Could not trim {wav_path}: {e}", flush=True)
            
    print("Silence trimming complete.", flush=True)
    
    # ============================================================
    # STAGE 2: Feature Extraction (Whisper / Encodec)
    # ============================================================
    print("\n" + "="*60, flush=True)
    print("                    STAGE 2: EXTRACTING & CACHING FEATURES", flush=True)
    print("="*60, flush=True)
    
    print("Importing PyTorch and Transformers...", flush=True)
    import torch
    from transformers import WhisperProcessor, WhisperModel, AutoProcessor, EncodecModel
    
    print("Loading Whisper encoder (openai/whisper-tiny)...", flush=True)
    whisper_proc = WhisperProcessor.from_pretrained("openai/whisper-tiny", local_files_only=True)
    whisper_model = WhisperModel.from_pretrained("openai/whisper-tiny", local_files_only=True)
    whisper_model.eval()
    
    print("Loading Encodec model (facebook/encodec_24khz)...", flush=True)
    encodec_proc = AutoProcessor.from_pretrained("facebook/encodec_24khz", local_files_only=True)
    encodec_model = EncodecModel.from_pretrained("facebook/encodec_24khz", local_files_only=True)
    encodec_model.eval()
    
    manifest_trimmed = []
    
    for item in tqdm(samples_data, desc="Extracting Features"):
        file_id = item["id"]
        temp_wav = item["wav_path"]
        text = item["text"]
        
        if not temp_wav.exists():
            continue
            
        try:
            orig_sr, audio_int16 = wavfile.read(str(temp_wav))
            audio_f32 = audio_int16.astype(np.float32) / 32768.0
            if len(audio_f32.shape) > 1:
                audio_f32 = audio_f32.mean(axis=-1)
        except Exception:
            continue
            
        duration = len(audio_f32) / orig_sr
        
        # Whisper features (16kHz)
        audio_16k = resample_audio(audio_f32, orig_sr, 16000)
        w_inputs = whisper_proc(audio_16k, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            w_output = whisper_model.encoder(w_inputs.input_features).last_hidden_state
            w_output = w_output.squeeze(0) # shape [1500, hidden_dim]
            
        cont_file = cont_cache_dir / f"{file_id}.pt"
        torch.save(w_output, cont_file)
        
        # EnCodec features (24kHz)
        audio_24k = resample_audio(audio_f32, orig_sr, 24000)
        e_inputs = encodec_proc(raw_audio=audio_24k, sampling_rate=24000, return_tensors="pt")
        with torch.no_grad():
            e_output = encodec_model.encode(e_inputs.input_values, e_inputs.padding_mask)
            audio_codes = e_output.audio_codes.squeeze(0).squeeze(0) # shape [num_codebooks, seq_len]
            
        disc_file = disc_cache_dir / f"{file_id}.pt"
        torch.save(audio_codes, disc_file)
        
        manifest_trimmed.append({
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
        
    # Save the manifest_trimmed.json
    manifest_path = workspace_dir / "manifest_trimmed.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_trimmed, f, indent=2)
        
    print("\n" + "="*60, flush=True)
    print("                    PREPROCESSING COMPLETE", flush=True)
    print("="*60, flush=True)
    print(f"Manifest saved to:         {manifest_path}", flush=True)
    print(f"Total Bootstrapped items:  {len(manifest_trimmed)} samples", flush=True)

if __name__ == "__main__":
    main()
