"""
Phase 4: Visual Feature Extraction and Caching.
Loads openai/clip-vit-large-patch14 to extract vision patch embeddings
for bootstrapped images, saving features to cache_bootstrapped/vision/
and updating manifest_bootstrapped.json.
Runs entirely offline with manual torchvision preprocessing.
"""

import os
import sys
import json
import time
import torch
import torchvision.transforms as T
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from transformers import CLIPVisionModel

# Force stdout to UTF-8
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

# Force transformers offline
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

def main():
    workspace_dir = Path(r"C:\Users\Navie\.gemini\antigravity\scratch\multimodal-audio-suite")
    manifest_path = workspace_dir / "manifest_bootstrapped.json"
    
    if not manifest_path.exists():
        print(f"Error: manifest_bootstrapped.json not found at: {manifest_path}")
        return
        
    # Setup cache directory
    vision_cache_dir = workspace_dir / "cache_bootstrapped" / "vision"
    vision_cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Load manifest
    print(f"Loading manifest: {manifest_path}...")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
        
    print(f"Found {len(manifest)} items. Initializing CLIP vision tower...")
    
    model_name = "openai/clip-vit-large-patch14"
    try:
        model = CLIPVisionModel.from_pretrained(model_name, local_files_only=True)
        print("Loaded CLIPVisionModel from local cache successfully.")
    except Exception as e:
        print(f"Error loading model from local cache: {e}")
        return
        
    model.eval()
    
    # Manual torchvision preprocessing mimicking CLIPProcessor
    # Bicubic resize to 224x224, normalize with standard CLIP mean and std
    transform = T.Compose([
        T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
    ])
    
    updated_manifest = []
    
    # Base dataset path for images
    dataset_dir = Path(r"C:\Users\Navie\ideogram_finetune\dataset")
    
    print("\n" + "="*60)
    print("                    EXTRACTING CLIP VISION FEATURES (MANUAL PREPROCESS)")
    print("="*60)
    
    for item in tqdm(manifest, desc="Processing Images"):
        image_rel_path = item.get("original_image", "")
        file_id = item["id"]
        
        image_path = dataset_dir / image_rel_path
        if not image_path.exists():
            print(f"\nWarning: Image not found at {image_path}. Skipping.")
            continue
            
        # Extract features
        try:
            image = Image.open(image_path).convert("RGB")
            # Apply torchvision transforms and add batch dimension
            image_tensor = transform(image).unsqueeze(0) # shape [1, 3, 224, 224]
            
            with torch.no_grad():
                outputs = model(image_tensor)
                # CLIP output last_hidden_state shape: [1, 257, 1024]
                # Squeeze batch size and discard the CLS token (first token)
                # This leaves 256 patch tokens of shape [256, 1024]
                last_hidden_state = outputs.last_hidden_state.squeeze(0)[1:, :] # shape [256, 1024]
                
            # Save features
            save_file = vision_cache_dir / f"{file_id}.pt"
            torch.save(last_hidden_state, save_file)
            
            # Update manifest item
            item["vision_path"] = str(save_file)
            item["vision_dim"] = int(last_hidden_state.shape[-1])
            item["vision_seq_len"] = int(last_hidden_state.shape[0])
            
        except Exception as e:
            print(f"\nError processing image {image_path}: {e}")
            continue
            
        updated_manifest.append(item)
        
    # Overwrite manifest_bootstrapped.json
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(updated_manifest, f, indent=2)
        
    print("\n" + "="*60)
    print("                    VISION CACHING COMPLETED")
    print("="*60)
    print(f"Updated manifest saved to: {manifest_path}")
    print(f"Cached vision files: {len(list(vision_cache_dir.glob('*.pt')))} files")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
