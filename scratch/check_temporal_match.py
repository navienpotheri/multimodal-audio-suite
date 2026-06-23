import torch
import json
import re
import os

workspace_dir = r"C:\Users\Navie\.gemini\antigravity\scratch\multimodal-audio-suite"
save_path = os.path.join(workspace_dir, "tts_overfit_adapter")
manifest_path = os.path.join(workspace_dir, "manifest_bootstrapped.json")

# Load original cached codes
with open(manifest_path, "r", encoding="utf-8") as f:
    manifest = json.load(f)
orig_path = manifest[0]["discrete_path"]
orig_codes = torch.load(orig_path, map_location="cpu")

# Load generated codes from a greedy run
from transformers import Qwen2ForCausalLM, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(save_path)
model = Qwen2ForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct",
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True
)
model.resize_token_embeddings(len(tokenizer))
from peft import PeftModel
model = PeftModel.from_pretrained(model, save_path)
model.eval()

test_text = manifest[0]["text"]
prompt_str = f"<|im_start|>user\nRead the following text: {test_text}<|im_end|>\n<|im_start|>assistant\n<audio_start>"
input_ids = tokenizer(prompt_str, return_tensors="pt").input_ids
prompt_len = input_ids.shape[1]

audio_tokens_c0 = [f"<audio_c0_{i}>" for i in range(1024)]
audio_tokens_c1 = [f"<audio_c1_{i}>" for i in range(1024)]
audio_end_id = tokenizer.convert_tokens_to_ids("<audio_end>")
c0_ids = [tokenizer.convert_tokens_to_ids(t) for t in audio_tokens_c0]
c1_ids = [tokenizer.convert_tokens_to_ids(t) for t in audio_tokens_c1]

from transformers import LogitsProcessor, LogitsProcessorList
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
        max_new_tokens=1500,
        eos_token_id=audio_end_id,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=False,
        logits_processor=logits_processors
    )

new_tokens = generated_outputs[0, prompt_len:].tolist()
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
orig_len = orig_codes.shape[1]
compare_len = min(min_len, orig_len)

# Divide into 5 segments
segment_size = compare_len // 5
print(f"Dividing {compare_len} steps into 5 segments of size {segment_size}...")

for seg_idx in range(5):
    start = seg_idx * segment_size
    end = (seg_idx + 1) * segment_size if seg_idx < 4 else compare_len
    
    seg_len = end - start
    c0_match = sum(1 for i in range(start, end) if c0_codes[i] == orig_codes[0, i].item())
    c1_match = sum(1 for i in range(start, end) if c1_codes[i] == orig_codes[1, i].item())
    
    print(f"Segment {seg_idx+1} ({start}-{end}) | c0 match: {c0_match}/{seg_len} ({c0_match/seg_len*100:.1f}%) | c1 match: {c1_match}/{seg_len} ({c1_match/seg_len*100:.1f}%)")
