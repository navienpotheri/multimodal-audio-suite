import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
import re
import argparse
import numpy as np
import torch
from transformers import Qwen2ForCausalLM, AutoTokenizer, EncodecModel, WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel
from scipy.io import wavfile
import scipy.signal

import sys

# Force stdout to UTF-8 for Windows terminal safety
try:
    sys.stdout.reconfigure(encoding='utf-8')
except NameError:
    pass
except AttributeError:
    pass

def run_asr(wav_path):
    print(f"\n--- Transcribing {os.path.basename(wav_path)} using Whisper-Tiny ---")
    if not os.path.exists(wav_path):
        print("Error: File not found.")
        return ""
        
    orig_sr, audio_int16 = wavfile.read(wav_path)
    audio_f32 = audio_int16.astype(np.float32) / 32768.0
    if len(audio_f32.shape) > 1:
        audio_f32 = audio_f32.mean(axis=-1)
        
    num_samples = int(len(audio_f32) * 16000 / orig_sr)
    audio_16k = scipy.signal.resample(audio_f32, num_samples)
    
    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny", local_files_only=True)
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny", local_files_only=True)
    model.eval()
    
    input_features = processor(audio_16k, sampling_rate=16000, return_tensors="pt").input_features
    with torch.no_grad():
        predicted_ids = model.generate(input_features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    print(f"ASR Transcription: '{transcription.strip()}'")
    return transcription.strip()

def generate_speech_cfg(model, tokenizer, encodec_model, text, output_path, guidance_scale=3.0):
    print(f"\n--- Generating Speech for: '{text}' (CFG={guidance_scale}) ---")
    
    # Conditioned prompt
    prompt_str = f"<|im_start|>user\nRead the following text: {text}<|im_end|>\n<|im_start|>assistant\n<audio_start>"
    # Unconditioned prompt
    uncond_prompt_str = f"<|im_start|>user\nRead the following text: <|im_end|>\n<|im_start|>assistant\n<audio_start>"
    
    input_ids = tokenizer(prompt_str, return_tensors="pt").input_ids
    uncond_ids = tokenizer(uncond_prompt_str, return_tensors="pt").input_ids
    
    audio_end_id = tokenizer.convert_tokens_to_ids("<audio_end>")
    
    audio_tokens_c0 = [f"<audio_c0_{i}>" for i in range(1024)]
    audio_tokens_c1 = [f"<audio_c1_{i}>" for i in range(1024)]
    c0_ids = torch.tensor([tokenizer.convert_tokens_to_ids(t) for t in audio_tokens_c0])
    c1_ids = torch.tensor([tokenizer.convert_tokens_to_ids(t) for t in audio_tokens_c1])
    
    # Warmup KV cache
    with torch.no_grad():
        cond_outputs = model(input_ids=input_ids, use_cache=True)
        cond_past = cond_outputs.past_key_values
        cond_logits = cond_outputs.logits[0, -1, :]
        
        uncond_outputs = model(input_ids=uncond_ids, use_cache=True)
        uncond_past = uncond_outputs.past_key_values
        uncond_logits = uncond_outputs.logits[0, -1, :]
        
    new_tokens = []
    max_new_tokens = 450
    
    for step in range(max_new_tokens):
        # Apply CFG formula: L = L_u + g * (L_c - L_u)
        cfg_logits = uncond_logits + guidance_scale * (cond_logits - uncond_logits)
        
        # Restrict tokens based on codebook interleaving
        if step % 2 == 0:
            allowed = torch.cat([c0_ids, torch.tensor([audio_end_id])])
        else:
            allowed = torch.cat([c1_ids, torch.tensor([audio_end_id])])
            
        mask = torch.ones_like(cfg_logits, dtype=torch.bool)
        mask[allowed] = False
        cfg_logits[mask] = -float('inf')
        
        # Greedy choice
        next_token = torch.argmax(cfg_logits).item()
        
        if next_token == audio_end_id:
            break
            
        new_tokens.append(next_token)
        
        # Forward pass with KV cache
        next_token_tensor = torch.tensor([[next_token]])
        with torch.no_grad():
            cond_outputs = model(input_ids=next_token_tensor, past_key_values=cond_past, use_cache=True)
            cond_past = cond_outputs.past_key_values
            cond_logits = cond_outputs.logits[0, -1, :]
            
            uncond_outputs = model(input_ids=next_token_tensor, past_key_values=uncond_past, use_cache=True)
            uncond_past = uncond_outputs.past_key_values
            uncond_logits = uncond_outputs.logits[0, -1, :]
            
    c0_codes = []
    c1_codes = []
    gen_text = tokenizer.decode(new_tokens, skip_special_tokens=False)
    tokens_parsed = re.findall(r"<(audio_c[01])_(\d+)>", gen_text)
    for codebook, val in tokens_parsed:
        if codebook == "audio_c0":
            c0_codes.append(int(val))
        elif codebook == "audio_c1":
            c1_codes.append(int(val))
            
    min_len = min(len(c0_codes), len(c1_codes))
    print(f"Generated {len(new_tokens)} total tokens. Aligned length: {min_len} Encodec frames.")
    
    if min_len > 0:
        codes_tensor = torch.zeros(1, 1, 2, min_len, dtype=torch.long)
        codes_tensor[0, 0, 0, :] = torch.tensor(c0_codes[:min_len], dtype=torch.long)
        codes_tensor[0, 0, 1, :] = torch.tensor(c1_codes[:min_len], dtype=torch.long)
        
        with torch.no_grad():
            reconstructed_audio = encodec_model.decode(codes_tensor, [None], padding_mask=None).audio_values
            audio_data = reconstructed_audio.squeeze().cpu().numpy()
            
        audio_int16 = (audio_data * 32767).clip(-32768, 32767).astype(np.int16)
        wavfile.write(output_path, 24000, audio_int16)
        print(f"Saved generated audio to: {output_path}")
        return True
    else:
        print("Error: No audio codes were generated.")
        return False

def parse_args():
    parser = argparse.ArgumentParser(description="Generate speech from robust trained adapter on CPU.")
    parser.add_argument("--prompt", type=str, required=True, help="Text sentence to read")
    parser.add_argument("--output", type=str, required=True, help="Output path for the .wav file")
    parser.add_argument("--guidance_scale", type=float, default=3.0, help="CFG guidance scale (default: 3.0)")
    return parser.parse_args()

def main():
    args = parse_args()
    workspace_dir = r"C:\Users\Navie\.gemini\antigravity\scratch\multimodal-audio-suite"
    save_path = os.path.join(workspace_dir, "tts_v2v_robust_adapter")
    
    if not os.path.exists(save_path):
        print(f"Error: Adapter path {save_path} does not exist. Run train_v2v_robust.py first.")
        return
        
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(save_path, local_files_only=True)
    model = Qwen2ForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct",
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        local_files_only=True
    )
    model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(model, save_path)
    model.eval()
    
    print("Loading Encodec model...")
    encodec_model = EncodecModel.from_pretrained("facebook/encodec_24khz", local_files_only=True)
    encodec_model.eval()
    
    success = generate_speech_cfg(model, tokenizer, encodec_model, args.prompt, args.output, guidance_scale=args.guidance_scale)
    if success:
        run_asr(args.output)

if __name__ == "__main__":
    main()
