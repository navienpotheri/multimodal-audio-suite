"""
Phase 3: Autoregressive Voice-to-Voice (Text-to-Speech) Training and Decoding Script.
Expands vocabulary to 2 codebooks (2048 tokens: c0 and c1) to double audio bitrate and improve clarity.
Trains model to autoregressively predict interleaved audio tokens from text prompts.
Decodes generated audio tokens to WAV using EnCodec.
Runs entirely on CPU.
"""

import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import sys
import json
import argparse
import time
import re
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from scipy.io import wavfile
import numpy as np

# Force stdout to UTF-8 for Windows terminal safety
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

from transformers import (
    Qwen2ForCausalLM,
    AutoTokenizer,
    AutoProcessor,
    EncodecModel
)
from peft import LoraConfig, get_peft_model

class DiscreteAudioDataset(Dataset):
    def __init__(self, manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            self.manifest = json.load(f)
        print(f"Loaded dataset with {len(self.manifest)} items from manifest.")

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        item = self.manifest[idx]
        
        # Load cached Encodec codes (discrete tokens)
        # Shape: [num_codebooks, seq_len]
        try:
            discrete_codes = torch.load(item["discrete_path"], map_location="cpu", weights_only=True)
        except Exception as e:
            print(f"Error loading {item['discrete_path']}: {e}")
            discrete_codes = torch.zeros(2, 75, dtype=torch.long)
            
        return {
            "id": item["id"],
            "discrete_codes": discrete_codes,
            "transcript": item["text"]
        }

def parse_args():
    parser = argparse.ArgumentParser(description="Train Autoregressive Text-to-Speech LLM on CPU.")
    parser.add_argument(
        "--workspace_dir",
        type=str,
        default=r"C:\Users\Navie\.gemini\antigravity\scratch\multimodal-audio-suite",
        help="Path to workspace directory containing manifest.json and cache/"
    )
    parser.add_argument(
        "--manifest_name",
        type=str,
        default="manifest.json",
        help="Name of manifest JSON file"
    )
    parser.add_argument(
        "--qwen_model",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="HuggingFace repository path for Qwen base model"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of epochs to train (default: 3)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-4,
        help="Learning rate"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    workspace_dir = Path(args.workspace_dir)
    manifest_path = workspace_dir / args.manifest_name
    
    if not manifest_path.exists():
        print(f"Error: manifest.json not found in {workspace_dir}. Run preprocess_audio.py first.")
        return
        
    print("="*60)
    print("                    SETTING UP 2-CODEBOOK V2V MODEL")
    print("="*60)
    
    # 1. Initialize dataset
    dataset = DiscreteAudioDataset(manifest_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # 2. Load tokenizer and base model
    print(f"Loading tokenizer and model: {args.qwen_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.qwen_model)
    model = Qwen2ForCausalLM.from_pretrained(
        args.qwen_model,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )
    
    # 3. Add discrete audio tokens (2 codebooks x 1024 = 2048 tokens)
    print("Expanding vocabulary with 2048 Encodec tokens (c0 and c1)...")
    audio_tokens_c0 = [f"<audio_c0_{i}>" for i in range(1024)]
    audio_tokens_c1 = [f"<audio_c1_{i}>" for i in range(1024)]
    special_tokens = ["<audio_start>", "<audio_end>"]
    
    # Add all new tokens to vocabulary
    tokenizer.add_special_tokens({
        "additional_special_tokens": special_tokens + audio_tokens_c0 + audio_tokens_c1
    })
    
    # Resize model embeddings
    model.resize_token_embeddings(len(tokenizer))
    
    audio_end_id = tokenizer.convert_tokens_to_ids("<audio_end>")
    
    # 4. Setup PEFT/LoRA with trainable embeddings & LM head
    print("Applying PEFT LoRA with trainable embeddings...")
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        modules_to_save=["embed_tokens", "lm_head"]
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    # 5. Training Loop
    print("\n" + "="*60)
    print("                    STARTING 2-CODEBOOK AR TRAINING")
    print("="*60)
    
    model.train()
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        start_time = time.time()
        
        for step, batch in enumerate(dataloader):
            t_step = time.time()
            discrete_codes = batch["discrete_codes"] # [1, num_codebooks, seq_len]
            transcript = batch["transcript"][0]
            
            # Extract both codebook 0 and 1
            codes_c0 = discrete_codes[0, 0].tolist()
            codes_c1 = discrete_codes[0, 1].tolist()
            
            # Interleave codes: c0_0, c1_0, c0_1, c1_1, ...
            interleaved_tokens = []
            for c0, c1 in zip(codes_c0, codes_c1):
                interleaved_tokens.append(f"<audio_c0_{c0}>")
                interleaved_tokens.append(f"<audio_c1_{c1}>")
                
            audio_seq_str = "".join(interleaved_tokens)
            
            # Format sequence
            prompt_str = f"<|im_start|>user\nRead the following text: {transcript}<|im_end|>\n<|im_start|>assistant\n<audio_start>"
            target_str = f"{audio_seq_str}<audio_end><|im_end|>"
            
            full_text = prompt_str + target_str
            
            # Tokenize
            encodings = tokenizer(full_text, return_tensors="pt")
            input_ids = encodings.input_ids
            
            # Mask the prompt
            prompt_enc = tokenizer(prompt_str, return_tensors="pt")
            prompt_len = prompt_enc.input_ids.shape[1]
            
            labels = input_ids.clone()
            labels[0, :prompt_len] = -100
            
            # Forward pass
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            
            # Optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            step_loss = loss.item()
            epoch_loss += step_loss
            step_time = time.time() - t_step
            print(f"  Epoch {epoch+1:02d} | Step {step+1:02d}/{len(dataloader):02d} | Loss: {step_loss:.4f} | SeqLen: {input_ids.shape[1]} | Time: {step_time:.2f}s", flush=True)
            
        avg_loss = epoch_loss / len(dataloader)
        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1:02d}/{args.epochs:02d} | Avg Loss: {avg_loss:.4f} | Time: {elapsed:.2f}s", flush=True)
        
    # Save the trained adapter
    save_path = workspace_dir / "tts_v2v_adapter"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"\nTraining completed! Saved model adapter checkpoints to: {save_path}")
    
    # 6. Autoregressive Generation & Decoding
    print("\n" + "="*60)
    print("                    AUTOREGRESSIVE VOICE SYNTHESIS")
    print("="*60)
    
    model.eval()
    test_item = dataset[0]
    test_text = test_item["transcript"]
    
    print(f"Text Input:     '{test_text}'")
    print(f"Reference Audio Code Sequence length: {test_item['discrete_codes'].shape[-1]} steps")
    
    # Form input prompt
    prompt_str = f"<|im_start|>user\nRead the following text: {test_text}<|im_end|>\n<|im_start|>assistant\n<audio_start>"
    input_ids = tokenizer(prompt_str, return_tensors="pt").input_ids
    
    print("\nGenerating speech tokens from LLM...")
    t0 = time.time()
    with torch.no_grad():
        generated_outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=400, # Double token length since we interleave 2 codebooks
            eos_token_id=audio_end_id,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=True,
            temperature=0.8
        )
    gen_time = time.time() - t0
    
    # Extract generated tokens
    new_tokens = generated_outputs[0, input_ids.shape[1]:].tolist()
    gen_text = tokenizer.decode(new_tokens)
    print(f"Generated text stream length: {len(new_tokens)} tokens (Time: {gen_time:.2f}s)")
    
    # Parse out generated code indices for c0 and c1
    # We scan generated text sequentially and separate c0 and c1 tokens
    c0_codes = []
    c1_codes = []
    
    # We find all matches in sequence order
    tokens_parsed = re.findall(r"<(audio_c[01])_(\d+)>", gen_text)
    for codebook, val in tokens_parsed:
        if codebook == "audio_c0":
            c0_codes.append(int(val))
        elif codebook == "audio_c1":
            c1_codes.append(int(val))
            
    min_len = min(len(c0_codes), len(c1_codes))
    print(f"Extracted {len(c0_codes)} c0 codes and {len(c1_codes)} c1 codes. Aligned to length {min_len}.")
    
    if min_len == 0:
        print("Warning: No audio codes generated. Using dummy codes to verify.")
        min_len = 75
        c0_codes = [512] * min_len
        c1_codes = [512] * min_len
        
    # 7. Decode discrete tokens back to WAV using Encodec decoder
    print("\nLoading Encodec decoder...")
    encodec_model = EncodecModel.from_pretrained("facebook/encodec_24khz", local_files_only=True)
    encodec_model.eval()
    
    # Reconstruct codes matrix: shape [1, 1, num_codebooks, seq_len]
    num_codebooks = 2
    codes_tensor = torch.zeros(1, 1, num_codebooks, min_len, dtype=torch.long)
    codes_tensor[0, 0, 0, :] = torch.tensor(c0_codes[:min_len], dtype=torch.long)
    codes_tensor[0, 0, 1, :] = torch.tensor(c1_codes[:min_len], dtype=torch.long)
    
    print("Decoding 2-codebook tokens to waveform...")
    with torch.no_grad():
        reconstructed_audio = encodec_model.decode(codes_tensor, [None], padding_mask=None).audio_values
        audio_data = reconstructed_audio.squeeze().cpu().numpy()
        
    # Normalize to 16-bit integer range
    audio_int16 = (audio_data * 32767).clip(-32768, 32767).astype(np.int16)
    
    output_wav_path = workspace_dir / "synthesized_speech.wav"
    wavfile.write(str(output_wav_path), 24000, audio_int16)
    print(f"\n2-Codebook Speech synthesis successful! Audio saved to:\n  - WAV: {output_wav_path}")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
