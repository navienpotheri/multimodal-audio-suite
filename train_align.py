"""
Phase 2: LLaVA-Audio Projection Alignment Training Script.
Loads pre-computed Whisper continuous features and trains the linear projection layer
to align speech features with Qwen-2.5-0.5B-Instruct. Runs entirely on CPU.
"""

import os
import sys
import json
import argparse
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# Force stdout to UTF-8 for Windows terminal safety
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

from model import AudioLLM

class CachedAudioDataset(Dataset):
    def __init__(self, manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            self.manifest = json.load(f)
        print(f"Loaded dataset with {len(self.manifest)} items from manifest.")

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        item = self.manifest[idx]
        
        # Load cached Whisper states
        # Shape: [1500, whisper_dim]
        try:
            whisper_features = torch.load(item["continuous_path"], map_location="cpu", weights_only=True)
        except Exception as e:
            print(f"Error loading {item['continuous_path']}: {e}")
            whisper_features = torch.zeros(1500, 384)
            
        return {
            "id": item["id"],
            "whisper_features": whisper_features,
            "transcript": item["text"]
        }

def parse_args():
    parser = argparse.ArgumentParser(description="Train Audio-LLM projection layer on CPU.")
    parser.add_argument(
        "--workspace_dir",
        type=str,
        default=r"C:\Users\Navie\.gemini\antigravity\scratch\multimodal-audio-suite",
        help="Path to workspace directory containing manifest.json and cache/"
    )
    parser.add_argument(
        "--qwen_model",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="HuggingFace repository path for Qwen base model (default: 0.5B for CPU speed)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of epochs to train (default: 5)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for projection layer"
    )
    parser.add_argument(
        "--pooling_factor",
        type=int,
        default=10,
        help="Pooling factor to downsample audio sequence (default: 10)"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    workspace_dir = Path(args.workspace_dir)
    manifest_path = workspace_dir / "manifest.json"
    
    if not manifest_path.exists():
        print(f"Error: manifest.json not found in {workspace_dir}. Run preprocess_audio.py first.")
        return
        
    print("="*60)
    print("                    SETTING UP LLAVA-AUDIO TRAINING")
    print("="*60)
    
    # 1. Initialize dataset
    dataset = CachedAudioDataset(manifest_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # 2. Get audio dimension from first item
    first_item = dataset[0]
    audio_dim = first_item["whisper_features"].shape[-1]
    print(f"Detected Whisper audio dimension: {audio_dim}")
    
    # 3. Load Model
    print(f"Loading AudioLLM with base model {args.qwen_model}...")
    # Using float32 for CPU training stability
    model = AudioLLM(
        qwen_model_path=args.qwen_model,
        audio_dim=audio_dim,
        pooling_factor=args.pooling_factor,
        torch_dtype=torch.float32
    )
    
    # Verify trainable params
    model.print_trainable_parameters()
    
    # Optimizer for projection layer only
    trainable_params = model.get_trainable_parameters()
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)
    
    print("\n" + "="*60)
    print("                    STARTING TRAINING LOOP")
    print("="*60)
    
    model.train()
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        start_time = time.time()
        
        for step, batch in enumerate(dataloader):
            whisper_features = batch["whisper_features"] # [1, 1500, audio_dim]
            transcript = batch["transcript"][0]
            
            # Format conversational Chat template matching Qwen-2.5-Instruct
            prompt_before = "<|im_start|>user\n<audio_start>"
            prompt_after = f"<audio_end>\nTranscribe the following speech.<|im_end|>\n<|im_start|>assistant\n"
            response = f"{transcript}<|im_end|>"
            
            # Tokenize segments
            prefix_ids = model.tokenizer(prompt_before, return_tensors="pt").input_ids
            suffix_ids = model.tokenizer(prompt_after, return_tensors="pt").input_ids
            label_ids = model.tokenizer(response, return_tensors="pt").input_ids
            
            # Forward pass
            outputs = model(
                whisper_features=whisper_features,
                prefix_ids=prefix_ids,
                suffix_ids=suffix_ids,
                labels=label_ids
            )
            
            loss = outputs.loss
            
            # Optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(dataloader)
        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1:02d}/{args.epochs:02d} | Avg Loss: {avg_loss:.4f} | Time: {elapsed:.2f}s")
        
    # Save the trained projection weights
    save_path = workspace_dir / "audio_projector.pt"
    torch.save(model.proj.state_dict(), save_path)
    print(f"\nTraining completed! Saved projection weights to: {save_path}")
    
    # 4. Run Verification Inference
    print("\n" + "="*60)
    print("                    RUNNING VERIFICATION INFERENCE")
    print("="*60)
    
    model.eval()
    test_item = dataset[0]
    whisper_features = test_item["whisper_features"].unsqueeze(0) # [1, 1500, audio_dim]
    real_transcript = test_item["transcript"]
    
    prompt_before = "<|im_start|>user\n<audio_start>"
    prompt_after = f"<audio_end>\nTranscribe the following speech.<|im_end|>\n<|im_start|>assistant\n"
    
    prefix_ids = model.tokenizer(prompt_before, return_tensors="pt").input_ids
    suffix_ids = model.tokenizer(prompt_after, return_tensors="pt").input_ids
    
    print(f"Input prompt template: {prompt_before}<audio_tokens>{prompt_after}")
    print(f"Target Reference:      {real_transcript}")
    
    print("\nGenerating response...")
    generated_text = model.generate(
        whisper_features=whisper_features,
        prefix_ids=prefix_ids,
        suffix_ids=suffix_ids,
        max_new_tokens=60,
        temperature=0.0 # Greedy decoding
    )
    
    print(f"Generated Output:      {generated_text}")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
