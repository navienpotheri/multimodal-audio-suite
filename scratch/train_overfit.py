import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import sys
import json
import time
import re
import torch
import shutil
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from scipy.io import wavfile
import numpy as np

# Force stdout to UTF-8
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
from transformers import LogitsProcessor, LogitsProcessorList
from peft import LoraConfig, get_peft_model

class SingleItemDataset(Dataset):
    def __init__(self, manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            self.manifest = json.load(f)
        # Only keep the very first sample
        self.item = self.manifest[0]
        print(f"Overfitting dataset initialized with item: {self.item['id']}")
        print(f"Transcript: '{self.item['text']}'")

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        discrete_codes = torch.load(self.item["discrete_path"], map_location="cpu", weights_only=True)
        return {
            "id": self.item["id"],
            "discrete_codes": discrete_codes,
            "transcript": self.item["text"]
        }

def main():
    workspace_dir = Path(r"C:\Users\Navie\.gemini\antigravity\scratch\multimodal-audio-suite")
    manifest_path = workspace_dir / "manifest_bootstrapped.json"
    
    print("="*60)
    print("                STARTING SINGLE-SAMPLE OVERFITTING")
    print("="*60)
    
    dataset = SingleItemDataset(manifest_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    model = Qwen2ForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct",
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )
    
    # Add special tokens
    audio_tokens_c0 = [f"<audio_c0_{i}>" for i in range(1024)]
    audio_tokens_c1 = [f"<audio_c1_{i}>" for i in range(1024)]
    special_tokens = ["<audio_start>", "<audio_end>"]
    
    tokenizer.add_special_tokens({
        "additional_special_tokens": special_tokens + audio_tokens_c0 + audio_tokens_c1
    })
    model.resize_token_embeddings(len(tokenizer))
    
    audio_end_id = tokenizer.convert_tokens_to_ids("<audio_end>")
    c0_ids = [tokenizer.convert_tokens_to_ids(t) for t in audio_tokens_c0]
    c1_ids = [tokenizer.convert_tokens_to_ids(t) for t in audio_tokens_c1]
    
    # Setup LoRA
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
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3) # Higher learning rate for quick memorization
    
    # Train for 60 epochs
    epochs = 60
    model.train()
    print("\nTraining...")
    for epoch in range(epochs):
        t0 = time.time()
        for step, batch in enumerate(dataloader):
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
            optimizer.step()
            
        elapsed = time.time() - t0
        print(f"  Epoch {epoch+1:02d}/{epochs:02d} | Loss: {loss.item():.4f} | Time: {elapsed:.2f}s", flush=True)
        
    print("\nSaving overfitted adapter...")
    overfit_path = workspace_dir / "tts_overfit_adapter"
    model.save_pretrained(overfit_path)
    tokenizer.save_pretrained(overfit_path)
    
    # 6. Constrained Generation
    print("\nGenerating speech via Constrained Decoding...")
    model.eval()
    test_item = dataset[0]
    test_text = test_item["transcript"]
    
    prompt_str = f"<|im_start|>user\nRead the following text: {test_text}<|im_end|>\n<|im_start|>assistant\n<audio_start>"
    input_ids = tokenizer(prompt_str, return_tensors="pt").input_ids
    prompt_len = input_ids.shape[1]
    
    class InterleavedAudioLogitsProcessor(LogitsProcessor):
        def __init__(self, prompt_len, c0_ids, c1_ids, audio_end_id):
            self.prompt_len = prompt_len
            self.c0_ids = torch.tensor(c0_ids, dtype=torch.long)
            self.c1_ids = torch.tensor(c1_ids, dtype=torch.long)
            self.audio_end_id = audio_end_id

        def __call__(self, input_ids, scores):
            for i in range(input_ids.shape[0]):
                gen_len = input_ids[i].shape[0] - self.prompt_len
                if gen_len % 2 == 0:
                    allowed_ids = torch.cat([self.c0_ids, torch.tensor([self.audio_end_id])])
                else:
                    allowed_ids = torch.cat([self.c1_ids, torch.tensor([self.audio_end_id])])
                mask = torch.ones(scores.shape[1], dtype=torch.bool)
                mask[allowed_ids] = False
                scores[i, mask] = -float('inf')
            return scores

    logits_processors = LogitsProcessorList([
        InterleavedAudioLogitsProcessor(prompt_len, c0_ids, c1_ids, audio_end_id)
    ])
    
    with torch.no_grad():
        generated_outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=1500, # Generate full length of codes (712 * 2 = 1424)
            eos_token_id=audio_end_id,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=False, # Use greedy decoding for best reproduction
            logits_processor=logits_processors
        )
        
    new_tokens = generated_outputs[0, prompt_len:].tolist()
    print(f"Generated {len(new_tokens)} tokens.")
    
    gen_text = tokenizer.decode(new_tokens, skip_special_tokens=False)
    
    c0_codes = []
    c1_codes = []
    tokens_parsed = re.findall(r"<(audio_c[01])_(\d+)>", gen_text)
    for codebook, val in tokens_parsed:
        if codebook == "audio_c0":
            c0_codes.append(int(val))
        elif codebook == "audio_c1":
            c1_codes.append(int(val))
            
    min_len = min(len(c0_codes), len(c1_codes))
    print(f"Extracted {len(c0_codes)} c0 codes and {len(c1_codes)} c1 codes. Aligned to length {min_len}.")
    
    if min_len == 0:
        print("Error: No audio codes generated.")
        return
        
    print("\nDecoding with Encodec...")
    encodec_model = EncodecModel.from_pretrained("facebook/encodec_24khz", local_files_only=True)
    encodec_model.eval()
    
    codes_tensor = torch.zeros(1, 1, 2, min_len, dtype=torch.long)
    codes_tensor[0, 0, 0, :] = torch.tensor(c0_codes[:min_len], dtype=torch.long)
    codes_tensor[0, 0, 1, :] = torch.tensor(c1_codes[:min_len], dtype=torch.long)
    
    with torch.no_grad():
        reconstructed_audio = encodec_model.decode(codes_tensor, [None], padding_mask=None).audio_values
        audio_data = reconstructed_audio.squeeze().cpu().numpy()
        
    # Normalize
    max_amp = np.abs(audio_data).max()
    print(f"Original Decoded Max Amplitude: {max_amp:.6f}")
    if max_amp > 1e-5:
        audio_data = audio_data * (0.8 / max_amp)
        print("Normalized amplitude to peak at 0.8.")
        
    audio_int16 = (audio_data * 32767).clip(-32768, 32767).astype(np.int16)
    
    local_wav = workspace_dir / "synthesized_speech.wav"
    home_wav = Path(r"C:\Users\Navie\synthesized_speech.wav")
    
    wavfile.write(str(local_wav), 24000, audio_int16)
    shutil.copy(str(local_wav), str(home_wav))
    print(f"Speech saved and copied to: {home_wav}")
    
    # Open folder
    os.startfile(r"C:\Users\Navie")
    print("Done!")

if __name__ == "__main__":
    main()
