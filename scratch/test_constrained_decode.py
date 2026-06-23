import torch
from transformers import Qwen2ForCausalLM, AutoTokenizer, EncodecModel
from transformers import LogitsProcessor, LogitsProcessorList
from peft import PeftModel
import re
import os
import shutil
import numpy as np
from scipy.io import wavfile

# Set offline environment variables
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

workspace_dir = r"C:\Users\Navie\.gemini\antigravity\scratch\multimodal-audio-suite"
save_path = os.path.join(workspace_dir, "tts_v2v_adapter")

print("1. Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(save_path)
model = Qwen2ForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct",
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True
)
model.resize_token_embeddings(len(tokenizer))
model = PeftModel.from_pretrained(model, save_path)
model.eval()

# Prompt text
test_text = "The image features a highly detailed and intricate design with a symmetrical, geometric, and symbolic aesthetic. The overall color scheme is."
prompt_str = f"<|im_start|>user\nRead the following text: {test_text}<|im_end|>\n<|im_start|>assistant\n<audio_start>"
input_ids = tokenizer(prompt_str, return_tensors="pt").input_ids
prompt_len = input_ids.shape[1]

# Set up token IDs
audio_tokens_c0 = [f"<audio_c0_{i}>" for i in range(1024)]
audio_tokens_c1 = [f"<audio_c1_{i}>" for i in range(1024)]
c0_ids = [tokenizer.convert_tokens_to_ids(t) for t in audio_tokens_c0]
c1_ids = [tokenizer.convert_tokens_to_ids(t) for t in audio_tokens_c1]
audio_end_id = tokenizer.convert_tokens_to_ids("<audio_end>")

class InterleavedAudioLogitsProcessor(LogitsProcessor):
    def __init__(self, prompt_len, c0_ids, c1_ids, audio_end_id):
        self.prompt_len = prompt_len
        self.c0_ids = torch.tensor(c0_ids, dtype=torch.long)
        self.c1_ids = torch.tensor(c1_ids, dtype=torch.long)
        self.audio_end_id = audio_end_id

    def __call__(self, input_ids, scores):
        # input_ids shape: [batch_size, seq_len]
        for i in range(input_ids.shape[0]):
            gen_len = input_ids[i].shape[0] - self.prompt_len
            
            # Determine which tokens are allowed
            if gen_len % 2 == 0:
                # Even step: must generate c0 or audio_end
                allowed_ids = torch.cat([self.c0_ids, torch.tensor([self.audio_end_id])])
            else:
                # Odd step: must generate c1 or audio_end
                allowed_ids = torch.cat([self.c1_ids, torch.tensor([self.audio_end_id])])
                
            # Create a mask for allowed tokens
            mask = torch.ones(scores.shape[1], dtype=torch.bool)
            mask[allowed_ids] = False
            scores[i, mask] = -float('inf')
            
        return scores

logits_processors = LogitsProcessorList([
    InterleavedAudioLogitsProcessor(prompt_len, c0_ids, c1_ids, audio_end_id)
])

print("2. Generating speech tokens from LLM with interleaved logits constraint...")
with torch.no_grad():
    generated_outputs = model.generate(
        input_ids=input_ids,
        max_new_tokens=400,
        eos_token_id=audio_end_id,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=True,
        temperature=0.8,
        logits_processor=logits_processors
    )

new_tokens = generated_outputs[0, prompt_len:].tolist()
print(f"Generated {len(new_tokens)} tokens.")

# Decode text
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
print("First 10 c0 codes:", c0_codes[:10])
print("First 10 c1 codes:", c1_codes[:10])

if min_len == 0:
    print("Error: No audio codes generated.")
    exit(1)

print("3. Loading Encodec decoder...")
encodec_model = EncodecModel.from_pretrained("facebook/encodec_24khz", local_files_only=True)
encodec_model.eval()

codes_tensor = torch.zeros(1, 1, 2, min_len, dtype=torch.long)
codes_tensor[0, 0, 0, :] = torch.tensor(c0_codes[:min_len], dtype=torch.long)
codes_tensor[0, 0, 1, :] = torch.tensor(c1_codes[:min_len], dtype=torch.long)

print("4. Decoding tokens to waveform...")
with torch.no_grad():
    reconstructed_audio = encodec_model.decode(codes_tensor, [None], padding_mask=None).audio_values
    audio_data = reconstructed_audio.squeeze().cpu().numpy()

max_amp = np.abs(audio_data).max()
print(f"Original Decoded Max Amplitude: {max_amp:.6f}")

# Normalize to peak at 0.8
if max_amp > 1e-5:
    scale = 0.8 / max_amp
    audio_data = audio_data * scale
    print(f"Normalized Max Amplitude to: {np.abs(audio_data).max():.6f}")

audio_int16 = (audio_data * 32767).clip(-32768, 32767).astype(np.int16)

local_wav = os.path.join(workspace_dir, "synthesized_speech.wav")
home_wav = r"C:\Users\Navie\synthesized_speech.wav"

wavfile.write(local_wav, 24000, audio_int16)
print(f"Saved locally to: {local_wav}")

shutil.copy(local_wav, home_wav)
print(f"Copied to home directory: {home_wav}")

# Open home directory in File Explorer
print("Opening folder in File Explorer...")
os.startfile(r"C:\Users\Navie")
print("Done!")
