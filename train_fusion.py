"""
Phase 4: Unified Visual-Audio Understanding Model.
Trains a visual projection layer and Qwen LoRA weights on CPU
using interleaved image patch embeddings, audio hidden states, and text.
"""

import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import sys
import json
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image

# Force stdout to UTF-8
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

from transformers import Qwen2ForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

class MultimodalDataset(Dataset):
    def __init__(self, manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            self.manifest = json.load(f)
        print(f"Loaded fusion dataset with {len(self.manifest)} items.")

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        item = self.manifest[idx]
        
        # Load cached features
        vision_features = torch.load(item["vision_path"], map_location="cpu", weights_only=True) # [256, 1024]
        
        # Load and average-pool continuous audio features to length 150
        audio_features_raw = torch.load(item["continuous_path"], map_location="cpu", weights_only=True) # [1500, 384]
        seq_len, dim = audio_features_raw.shape
        audio_features = audio_features_raw.view(seq_len // 10, 10, dim).mean(dim=1) # [150, 384]
        
        return {
            "id": item["id"],
            "vision_features": vision_features,
            "audio_features": audio_features,
            "transcript": item["text"]
        }

def main():
    workspace_dir = Path(r"C:\Users\Navie\.gemini\antigravity\scratch\multimodal-audio-suite")
    manifest_path = workspace_dir / "manifest_bootstrapped.json"
    
    if not manifest_path.exists():
        print(f"Error: manifest_bootstrapped.json not found. Run preprocess_vision.py first.")
        return
        
    print("="*60)
    print("                INITIALIZING MULTIMODAL FUSION MODEL")
    print("="*60)
    
    dataset = MultimodalDataset(manifest_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    model = Qwen2ForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct",
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )
    
    # Add multimodal tokens
    special_tokens = ["<image>", "<audio>"]
    tokenizer.add_special_tokens({
        "additional_special_tokens": special_tokens
    })
    model.resize_token_embeddings(len(tokenizer))
    
    image_token_id = tokenizer.convert_tokens_to_ids("<image>")
    audio_token_id = tokenizer.convert_tokens_to_ids("<audio>")
    
    # Define and load projectors
    print("Loading projectors...")
    # Audio projector maps 384 -> 896
    audio_projector = nn.Linear(384, 896)
    audio_proj_path = workspace_dir / "audio_projector.pt"
    if audio_proj_path.exists():
        audio_projector.load_state_dict(torch.load(audio_proj_path, map_location="cpu"))
        print("Loaded pre-trained audio projector weights.")
    else:
        print("Warning: Pre-trained audio projector not found. Initializing randomly.")
        
    # Vision projector maps 1024 -> 896
    vision_projector = nn.Linear(1024, 896)
    
    # Setup LoRA on Qwen
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
    
    # Optimizer updates vision projector, audio projector, and LoRA weights
    params = list(model.parameters()) + list(vision_projector.parameters()) + list(audio_projector.parameters())
    optimizer = torch.optim.AdamW(params, lr=1e-3)
    
    epochs = 5
    model.train()
    audio_projector.train()
    vision_projector.train()
    
    print("\n" + "="*60)
    print("                    STARTING FUSION TRAINING")
    print("="*60)
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        start_time = time.time()
        
        for step, batch in enumerate(dataloader):
            t_step = time.time()
            vision_features = batch["vision_features"][0] # [256, 1024]
            audio_features = batch["audio_features"][0]   # [150, 384]
            transcript = batch["transcript"][0]
            
            # Format inputs: Repeat tokens to match visual and audio feature sizes
            prompt_str = (
                "<|im_start|>user\n"
                + "<image>" * 256 
                + "<audio>" * 150 
                + "\nTranscribe the speech and describe the matching image.<|im_end|>\n"
                + "<|im_start|>assistant\n"
            )
            target_str = f"{transcript}<|im_end|>"
            full_text = prompt_str + target_str
            
            encodings = tokenizer(full_text, return_tensors="pt")
            input_ids = encodings.input_ids
            
            # Labels: mask prompt tokens
            prompt_enc = tokenizer(prompt_str, return_tensors="pt")
            prompt_len = prompt_enc.input_ids.shape[1]
            labels = input_ids.clone()
            labels[0, :prompt_len] = -100
            
            # Project features to Qwen dim (896)
            proj_vision = vision_projector(vision_features) # [256, 896]
            proj_audio = audio_projector(audio_features)   # [150, 896]
            
            # Get default text embeddings
            embed_tokens = model.get_input_embeddings()
            inputs_embeds = embed_tokens(input_ids).clone()
            
            # Replace placeholder embeddings with projected continuous features
            img_mask = (input_ids == image_token_id)
            aud_mask = (input_ids == audio_token_id)
            
            # Verify shapes match before assignment
            assert img_mask.sum().item() == 256, f"Expected 256 image tokens, found {img_mask.sum().item()}"
            assert aud_mask.sum().item() == 150, f"Expected 150 audio tokens, found {aud_mask.sum().item()}"
            
            inputs_embeds[img_mask] = proj_vision.view(-1, 896)
            inputs_embeds[aud_mask] = proj_audio.view(-1, 896)
            
            # Forward pass
            outputs = model(inputs_embeds=inputs_embeds, labels=labels)
            loss = outputs.loss
            
            # Backward and step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            step_loss = loss.item()
            epoch_loss += step_loss
            step_time = time.time() - t_step
            print(f"  Epoch {epoch+1:02d} | Step {step+1:02d}/{len(dataloader):02d} | Loss: {step_loss:.4f} | Time: {step_time:.2f}s", flush=True)
            
        avg_loss = epoch_loss / len(dataloader)
        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1:02d}/{epochs:02d} | Avg Loss: {avg_loss:.4f} | Time: {elapsed:.2f}s", flush=True)
        
    # Save the trained components
    save_dir = workspace_dir / "fusion_adapter"
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    torch.save(vision_projector.state_dict(), workspace_dir / "vision_projector.pt")
    torch.save(audio_projector.state_dict(), workspace_dir / "audio_projector_fusion.pt")
    print(f"\nTraining completed! Saved checkpoints.")
    
    # 7. Evaluate on a test sample
    print("\n" + "="*60)
    print("                    MODEL GENERATION TESTING")
    print("="*60)
    
    model.eval()
    vision_projector.eval()
    audio_projector.eval()
    
    # Test on the first item
    test_item = dataset[0]
    vision_features = test_item["vision_features"]
    audio_features = test_item["audio_features"]
    
    prompt_str = (
        "<|im_start|>user\n"
        + "<image>" * 256 
        + "<audio>" * 150 
        + "\nTranscribe the speech and describe the matching image.<|im_end|>\n"
        + "<|im_start|>assistant\n"
    )
    input_ids = tokenizer(prompt_str, return_tensors="pt").input_ids
    
    # We do autoregressive generation using inputs_embeds!
    # Because we are using custom embeddings, we manually run the generation step-by-step
    generated_ids = []
    curr_input_ids = input_ids.clone()
    
    # Pre-project visual/audio features
    proj_vision = vision_projector(vision_features)
    proj_audio = audio_projector(audio_features)
    
    print("Generating output tokens (greedy)...")
    for _ in range(50): # Generate up to 50 tokens
        embed_tokens = model.get_input_embeddings()
        inputs_embeds = embed_tokens(curr_input_ids).clone()
        
        # Replace image/audio placeholders
        img_mask = (curr_input_ids == image_token_id)
        aud_mask = (curr_input_ids == audio_token_id)
        inputs_embeds[img_mask] = proj_vision.view(-1, 896)
        inputs_embeds[aud_mask] = proj_audio.view(-1, 896)
        
        with torch.no_grad():
            outputs = model(inputs_embeds=inputs_embeds)
            next_token_logits = outputs.logits[0, -1, :]
            next_token = torch.argmax(next_token_logits).unsqueeze(0).unsqueeze(0)
            
        generated_ids.append(next_token.item())
        curr_input_ids = torch.cat([curr_input_ids, next_token], dim=-1)
        
        if next_token.item() == tokenizer.eos_token_id or next_token.item() == tokenizer.convert_tokens_to_ids("<|im_end|>"):
            break
            
    decoded_output = tokenizer.decode(generated_ids)
    print(f"\nGround Truth Transcript:  '{test_item['transcript']}'")
    print(f"Generated Description:   '{decoded_output}'")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
