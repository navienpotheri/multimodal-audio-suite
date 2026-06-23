import torch
from transformers import Qwen2ForCausalLM, AutoTokenizer, EncodecModel
from peft import PeftModel
import re
import os
import numpy as np

# Set offline environment variables
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

workspace_dir = r"C:\Users\Navie\.gemini\antigravity\scratch\multimodal-audio-suite"
save_path = os.path.join(workspace_dir, "tts_v2v_adapter")

print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(save_path)
model = Qwen2ForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct",
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True
)
model.resize_token_embeddings(len(tokenizer))
model = PeftModel.from_pretrained(model, save_path)
model.eval()

# Let's inspect the test text
test_text = "The image features a highly detailed and intricate design with a symmetrical, geometric, and symbolic aesthetic. The overall color scheme is."
prompt_str = f"<|im_start|>user\nRead the following text: {test_text}<|im_end|>\n<|im_start|>assistant\n<audio_start>"
input_ids = tokenizer(prompt_str, return_tensors="pt").input_ids

audio_end_id = tokenizer.convert_tokens_to_ids("<audio_end>")
print("Generating speech tokens from LLM...")
with torch.no_grad():
    generated_outputs = model.generate(
        input_ids=input_ids,
        max_new_tokens=400,
        eos_token_id=audio_end_id,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=True,
        temperature=0.8
    )

new_tokens = generated_outputs[0, input_ids.shape[1]:].tolist()
print(f"Generated {len(new_tokens)} tokens.")

# Decode with skip_special_tokens=False to ensure special tokens are decoded
gen_text = tokenizer.decode(new_tokens, skip_special_tokens=False)
print("Decoded text snippet (first 300 chars):")
print(gen_text[:300])

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
print("First 20 c0 codes:", c0_codes[:20])
print("First 20 c1 tokens:", c1_codes[:20])

if min_len > 0:
    print("Loading Encodec decoder...")
    encodec_model = EncodecModel.from_pretrained("facebook/encodec_24khz", local_files_only=True)
    encodec_model.eval()
    
    codes_tensor = torch.zeros(1, 1, 2, min_len, dtype=torch.long)
    codes_tensor[0, 0, 0, :] = torch.tensor(c0_codes[:min_len], dtype=torch.long)
    codes_tensor[0, 0, 1, :] = torch.tensor(c1_codes[:min_len], dtype=torch.long)
    
    print("Decoding 2-codebook tokens to waveform...")
    with torch.no_grad():
        reconstructed_audio = encodec_model.decode(codes_tensor, [None], padding_mask=None).audio_values
        audio_data = reconstructed_audio.squeeze().cpu().numpy()
        
    print(f"Audio values Max: {audio_data.max():.6f}")
    print(f"Audio values Min: {audio_data.min():.6f}")
    print(f"Audio values Std: {audio_data.std():.6f}")
    print("First 20 raw audio sample values:", audio_data[:20])
