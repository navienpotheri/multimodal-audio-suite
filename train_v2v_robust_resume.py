import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import sys
import json
import time
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from transformers import Qwen2ForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from peft import PeftModel

# Force stdout to UTF-8 for Windows terminal safety
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

class TrimmedAudioDataset(Dataset):
    def __init__(self, manifest_path, max_duration=4.5):
        with open(manifest_path, "r", encoding="utf-8") as f:
            full_manifest = json.load(f)
        self.manifest = [x for x in full_manifest if x.get("duration", 0) <= max_duration]
        print(f"Loaded trimmed dataset and filtered to duration <= {max_duration}s: retained {len(self.manifest)}/{len(full_manifest)} items.")

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        item = self.manifest[idx]
        discrete_codes = torch.load(item["discrete_path"], map_location="cpu", weights_only=True)
        return {
            "id": item["id"],
            "discrete_codes": discrete_codes,
            "transcript": item["text"]
        }

def main():
    workspace_dir = Path(r"C:\Users\Navie\.gemini\antigravity\scratch\multimodal-audio-suite")
    manifest_path = workspace_dir / "manifest_trimmed.json"
    save_path = workspace_dir / "tts_v2v_robust_adapter"
    
    print("="*60, flush=True)
    print("  ROBUST TTS TRAINING RESUMPTION (EPOCH 2)", flush=True)
    print("="*60, flush=True)
    
    dataset = TrimmedAudioDataset(manifest_path, max_duration=4.5)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    print("Loading tokenizer from existing checkpoint...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(save_path, local_files_only=True)
    
    print("Loading base Qwen model...", flush=True)
    model = Qwen2ForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct",
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        local_files_only=True
    )
    model.resize_token_embeddings(len(tokenizer))
    
    print("Loading PEFT model weights from existing adapter...", flush=True)
    model = PeftModel.from_pretrained(model, save_path, is_trainable=True)
    model.print_trainable_parameters()
    
    # Resume learning rate: decay from 2.5e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.5e-4)
    
    epochs = 1  # We only need to run the remaining 1 epoch (Epoch 2)
    num_training_steps = len(dataloader) * epochs
    num_warmup_steps = int(0.05 * num_training_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    
    original_vocab_size = 151665
    model.train()
    
    epoch_loss = 0.0
    start_time = time.time()
    
    for step, batch in enumerate(dataloader):
        t_step = time.time()
        discrete_codes = batch["discrete_codes"]
        transcript = batch["transcript"][0]
        
        codes_c0 = discrete_codes[0, 0].tolist()
        codes_c1 = discrete_codes[0, 1].tolist()
        
        interleaved_tokens = []
        for c0, c1 in zip(codes_c0, codes_c1):
            interleaved_tokens.append(f"<audio_c0_{c0}>")
            interleaved_tokens.append(f"<audio_c1_{c1}>")
            
        audio_seq_str = "".join(interleaved_tokens)
        
        prompt_str = f"<|im_start|>user\nRead the following text: {transcript}<|im_end|>\n<|im_start|>assistant\n<audio_start>"
        target_str = f"{audio_seq_str}<audio_end><|im_end|>"
        
        full_text = prompt_str + target_str
        
        encodings = tokenizer(full_text, return_tensors="pt")
        input_ids = encodings.input_ids
        
        prompt_enc = tokenizer(prompt_str, return_tensors="pt")
        prompt_len = prompt_enc.input_ids.shape[1]
        
        labels = input_ids.clone()
        labels[0, :prompt_len] = -100
        
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        
        optimizer.zero_grad()
        loss.backward()
        
        # Freeze original vocab embeddings and LM head rows
        embed_grad = model.get_input_embeddings().weight.grad
        if embed_grad is not None:
            embed_grad[:original_vocab_size] = 0.0
            
        lm_head_grad = model.get_output_embeddings().weight.grad
        if lm_head_grad is not None:
            lm_head_grad[:original_vocab_size] = 0.0
            
        optimizer.step()
        scheduler.step()
        
        step_loss = loss.item()
        epoch_loss += step_loss
        step_time = time.time() - t_step
        
        if (step + 1) % 50 == 0:
            print(f"  Epoch 02 | Step {step+1:04d}/{len(dataloader):04d} | Loss: {step_loss:.4f} | SeqLen: {input_ids.shape[1]} | LR: {scheduler.get_last_lr()[0]:.2e} | Time: {step_time:.2f}s", flush=True)
            
        if (step + 1) % 1000 == 0:
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            print(f"  Saved step {step+1:04d} checkpoints to: {save_path}", flush=True)
            
    avg_loss = epoch_loss / len(dataloader)
    elapsed = time.time() - start_time
    print(f"Epoch 02/02 | Avg Loss: {avg_loss:.4f} | Time: {elapsed:.2f}s", flush=True)
    
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"\nResumed training completed! Final model adapter checkpoints saved to: {save_path}", flush=True)

if __name__ == "__main__":
    main()
