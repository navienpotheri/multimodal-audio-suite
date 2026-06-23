import sys
import time
import torch
from transformers import Qwen2ForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

print("1. Loading tokenizer and model...")
t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
model = Qwen2ForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct",
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True
)
print(f"Loaded in {time.time() - t0:.2f}s")

print("2. Resizing embeddings...")
t0 = time.time()
audio_tokens = [f"<audio_c0_{i}>" for i in range(1024)] + [f"<audio_c1_{i}>" for i in range(1024)] + ["<audio_start>", "<audio_end>"]
tokenizer.add_special_tokens({"additional_special_tokens": audio_tokens})
model.resize_token_embeddings(len(tokenizer))
print(f"Resized in {time.time() - t0:.2f}s")

print("3. Setup PEFT...")
t0 = time.time()
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    modules_to_save=["embed_tokens", "lm_head"]
)
model = get_peft_model(model, peft_config)
print(f"PEFT setup in {time.time() - t0:.2f}s")

print("4. Printing parameters...")
model.print_trainable_parameters()

print("5. Benchmarking forward and backward pass for different sequence lengths...")
for seq_len in [100, 300, 700, 1400]:
    t0 = time.time()
    dummy_input = torch.randint(0, len(tokenizer), (1, seq_len))
    dummy_labels = dummy_input.clone()
    dummy_labels[0, :seq_len//2] = -100
    
    # Forward pass
    outputs = model(input_ids=dummy_input, labels=dummy_labels)
    loss = outputs.loss
    forward_time = time.time() - t0
    
    # Backward pass
    t0_back = time.time()
    loss.backward()
    backward_time = time.time() - t0_back
    
    print(f"Seq Len: {seq_len} | Forward: {forward_time:.2f}s | Backward: {backward_time:.2f}s | Total Step: {forward_time + backward_time:.2f}s")

