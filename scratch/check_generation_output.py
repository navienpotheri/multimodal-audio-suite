import torch
from transformers import Qwen2ForCausalLM, AutoTokenizer
import re
import os

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

# Resize model embeddings to match expanded tokenizer vocabulary
model.resize_token_embeddings(len(tokenizer))

# Load the PEFT adapter
from peft import PeftModel
model = PeftModel.from_pretrained(model, save_path)
model.eval()

# Let's inspect the test text
test_text = "The image features a highly detailed and intricate design with a symmetrical, geometric, and symbolic aesthetic. The overall color scheme is."
prompt_str = f"<|im_start|>user\nRead the following text: {test_text}<|im_end|>\n<|im_start|>assistant\n<audio_start>"
input_ids = tokenizer(prompt_str, return_tensors="pt").input_ids

print("Generating tokens...")
audio_end_id = tokenizer.convert_tokens_to_ids("<audio_end>")
with torch.no_grad():
    generated_outputs = model.generate(
        input_ids=input_ids,
        max_new_tokens=100,
        eos_token_id=audio_end_id,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=True,
        temperature=0.8
    )

new_tokens = generated_outputs[0, input_ids.shape[1]:].tolist()
print(f"Generated {len(new_tokens)} tokens.")
token_strings = [tokenizer.decode([t]) for t in new_tokens]
print("First 50 token strings generated:")
print(token_strings[:50])

# Count how many c0/c1 tokens were generated
c0_count = sum(1 for ts in token_strings if "audio_c0" in ts)
c1_count = sum(1 for ts in token_strings if "audio_c1" in ts)
print(f"c0 tokens count: {c0_count}")
print(f"c1 tokens count: {c1_count}")

# Check the actual token IDs and their values
print("First 20 token IDs generated:", new_tokens[:20])
