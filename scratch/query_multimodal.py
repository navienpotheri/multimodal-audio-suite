"""
Phase 4 Verification Script.
Allows interactive selection of one of the 15 bootstrapped samples,
loads the trained visual-audio fusion model, and prints the generated caption.
"""

import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import sys
import json
import torch
import torch.nn as nn
from pathlib import Path
from transformers import Qwen2ForCausalLM, AutoTokenizer
from peft import PeftModel

# Force stdout to UTF-8
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

def main():
    workspace_dir = Path(r"C:\Users\Navie\.gemini\antigravity\scratch\multimodal-audio-suite")
    manifest_path = workspace_dir / "manifest_bootstrapped.json"
    adapter_path = workspace_dir / "fusion_adapter"
    vision_proj_path = workspace_dir / "vision_projector.pt"
    audio_proj_path = workspace_dir / "audio_projector_fusion.pt"
    
    if not all([manifest_path.exists(), adapter_path.exists(), vision_proj_path.exists(), audio_proj_path.exists()]):
        print("Error: Missing checkpoints or manifest files. Please ensure you ran train_fusion.py successfully.")
        return
        
    # Load manifest
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
        
    print(f"Loaded {len(manifest)} available visual-audio items.")
    print("="*60)
    for idx, item in enumerate(manifest):
        print(f"[{idx+1:02d}] ID: {item['id']} | Image: {Path(item['original_image']).name}")
    print("="*60)
    
    # Prompt user for index selection
    try:
        selection = input(f"Select a sample number to test (1-{len(manifest)}) [default: 1]: ").strip()
        if not selection:
            sel_idx = 0
        else:
            sel_idx = int(selection) - 1
            if sel_idx < 0 or sel_idx >= len(manifest):
                print("Invalid selection. Using sample 1.")
                sel_idx = 0
    except ValueError:
        print("Invalid input. Using sample 1.")
        sel_idx = 0
        
    selected_item = manifest[sel_idx]
    print(f"\nSelected item: {selected_item['id']}")
    print(f"Image File:     {selected_item['original_image']}")
    print(f"Ground Truth:   '{selected_item['text']}'")
    
    print("\nLoading model checkpoints (this may take a few seconds on CPU)...")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    model = Qwen2ForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct",
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )
    model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    
    # Load projectors
    vision_projector = nn.Linear(1024, 896)
    vision_projector.load_state_dict(torch.load(vision_proj_path, map_location="cpu"))
    vision_projector.eval()
    
    audio_projector = nn.Linear(384, 896)
    audio_projector.load_state_dict(torch.load(audio_proj_path, map_location="cpu"))
    audio_projector.eval()
    
    # Load features
    vision_features = torch.load(selected_item["vision_path"], map_location="cpu") # [256, 1024]
    
    audio_features_raw = torch.load(selected_item["continuous_path"], map_location="cpu") # [1500, 384]
    seq_len, dim = audio_features_raw.shape
    audio_features = audio_features_raw.view(seq_len // 10, 10, dim).mean(dim=1) # [150, 384]
    
    image_token_id = tokenizer.convert_tokens_to_ids("<image>")
    audio_token_id = tokenizer.convert_tokens_to_ids("<audio>")
    
    prompt_str = (
        "<|im_start|>user\n"
        + "<image>" * 256 
        + "<audio>" * 150 
        + "\nTranscribe the speech and describe the matching image.<|im_end|>\n"
        + "<|im_start|>assistant\n"
    )
    curr_input_ids = tokenizer(prompt_str, return_tensors="pt").input_ids
    
    # Pre-project visual/audio features
    proj_vision = vision_projector(vision_features)
    proj_audio = audio_projector(audio_features)
    
    print("\nGenerating caption from visual and audio embeddings...")
    generated_ids = []
    
    for step in range(50):
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
        
        # Stop on EOS or end of turn
        if next_token.item() == tokenizer.eos_token_id or next_token.item() == tokenizer.convert_tokens_to_ids("<|im_end|>"):
            break
            
    decoded_output = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print("\n" + "="*60)
    print("                    FUSION INFERENCE RESULT")
    print("="*60)
    print(f"Generated Description:\n  '{decoded_output.strip()}'")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
