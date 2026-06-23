"""
End-to-End Image-to-Speech Pipeline.
1. Takes an image from the bootstrapped set.
2. Uses the Phase 4 Fusion Model to generate a textual caption/description.
3. Uses the Phase 3 V2V Model to synthesize 2-codebook speech from that description.
4. Reconstructs and low-pass filters the audio, saving it to home.
"""

import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import sys
import json
import torch
import torch.nn as nn
import torchvision.transforms as T
import re
import shutil
from pathlib import Path
from PIL import Image
from transformers import Qwen2ForCausalLM, AutoTokenizer, CLIPVisionModel, EncodecModel, LogitsProcessor, LogitsProcessorList
from peft import PeftModel
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, lfilter

# Force stdout to UTF-8
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

# Filter helpers
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def main():
    workspace_dir = Path(r"C:\Users\Navie\.gemini\antigravity\scratch\multimodal-audio-suite")
    manifest_path = workspace_dir / "manifest_bootstrapped.json"
    
    fusion_adapter_path = workspace_dir / "fusion_adapter"
    vision_proj_path = workspace_dir / "vision_projector.pt"
    audio_proj_path = workspace_dir / "audio_projector_fusion.pt"
    
    if not all([manifest_path.exists(), fusion_adapter_path.exists()]):
        print("Error: Missing checkpoints. Please run train_fusion.py and train_overfit.py first.")
        return
        
    # Load manifest
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
        
    print("="*60)
    print("                IMAGE-TO-SPEECH (ANY-TO-AUDIO) PIPELINE")
    print("="*60)
    for idx, item in enumerate(manifest):
        print(f"[{idx+1:02d}] ID: {item['id']} | Image: {Path(item['original_image']).name}")
    print("="*60)
    
    try:
        selection = input(f"Select an image to describe and speak (1-{len(manifest)}) [default: 1]: ").strip()
        sel_idx = int(selection) - 1 if selection else 0
        if sel_idx < 0 or sel_idx >= len(manifest):
            sel_idx = 0
    except ValueError:
        sel_idx = 0
        
    selected_item = manifest[sel_idx]
    print(f"\nProcessing: {selected_item['id']} | Image: {selected_item['original_image']}")
    
    if selected_item["id"] == "BOOT00001-0001":
        v2v_adapter_path = workspace_dir / "tts_overfit_adapter"
    elif selected_item["id"] == "BOOT00002-0001":
        v2v_adapter_path = workspace_dir / "tts_overfit_adapter_2"
    else:
        v2v_adapter_path = workspace_dir / "tts_two_samples_adapter"
        
    print(f"Using TTS model adapter: {v2v_adapter_path.name}")
    if not v2v_adapter_path.exists():
        print(f"Error: TTS adapter path does not exist: {v2v_adapter_path}")
        return
    
    # 1. Vision Feature Extraction
    print("\n--- STAGE 1: Extracting Vision Features ---")
    dataset_dir = Path(r"C:\Users\Navie\ideogram_finetune\dataset")
    image_path = dataset_dir / selected_item["original_image"]
    
    clip_model_name = "openai/clip-vit-large-patch14"
    clip_model = CLIPVisionModel.from_pretrained(clip_model_name, local_files_only=True)
    clip_model.eval()
    
    transform = T.Compose([
        T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
    ])
    
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        outputs = clip_model(image_tensor)
        vision_features = outputs.last_hidden_state.squeeze(0)[1:, :] # [256, 1024]
        
    # 2. Describe Image via Fusion Model (Phase 4)
    print("\n--- STAGE 2: Generating Text Description ---")
    tokenizer_f = AutoTokenizer.from_pretrained(fusion_adapter_path)
    model_f = Qwen2ForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct",
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )
    model_f.resize_token_embeddings(len(tokenizer_f))
    model_f = PeftModel.from_pretrained(model_f, fusion_adapter_path)
    model_f.eval()
    
    vision_projector = nn.Linear(1024, 896)
    vision_projector.load_state_dict(torch.load(vision_proj_path, map_location="cpu"))
    vision_projector.eval()
    
    image_token_id = tokenizer_f.convert_tokens_to_ids("<image>")
    
    # We describe the image using ONLY the visual features (audio features are passed as zero embeddings)
    # This checks if the visual embeddings can describe the image independently
    prompt_str = (
        "<|im_start|>user\n"
        + "<image>" * 256 
        + "\nTranscribe the speech and describe the matching image.<|im_end|>\n"
        + "<|im_start|>assistant\n"
    )
    curr_input_ids = tokenizer_f(prompt_str, return_tensors="pt").input_ids
    
    proj_vision = vision_projector(vision_features) # [256, 896]
    
    generated_ids_f = []
    print("Generating description...")
    for _ in range(50):
        embed_tokens = model_f.get_input_embeddings()
        inputs_embeds = embed_tokens(curr_input_ids).clone()
        
        img_mask = (curr_input_ids == image_token_id)
        inputs_embeds[img_mask] = proj_vision.view(-1, 896)
        
        with torch.no_grad():
            outputs = model_f(inputs_embeds=inputs_embeds)
            next_token_logits = outputs.logits[0, -1, :]
            next_token = torch.argmax(next_token_logits).unsqueeze(0).unsqueeze(0)
            
        generated_ids_f.append(next_token.item())
        curr_input_ids = torch.cat([curr_input_ids, next_token], dim=-1)
        
        if next_token.item() == tokenizer_f.eos_token_id or next_token.item() == tokenizer_f.convert_tokens_to_ids("<|im_end|>"):
            break
            
    generated_description = tokenizer_f.decode(generated_ids_f, skip_special_tokens=True).strip()
    # Clean description
    generated_description = generated_description.replace("\n", " ").strip()
    print(f"Generated Description: '{generated_description}'")
    
    # Clean memory
    del model_f
    del clip_model
    
    # 3. Text to Speech via V2V Model (Phase 3)
    print("\n--- STAGE 3: Synthesizing Audio from Description ---")
    tokenizer_v = AutoTokenizer.from_pretrained(v2v_adapter_path)
    model_v = Qwen2ForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct",
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )
    model_v.resize_token_embeddings(len(tokenizer_v))
    model_v = PeftModel.from_pretrained(model_v, v2v_adapter_path)
    model_v.eval()
    
    audio_tokens_c0 = [f"<audio_c0_{i}>" for i in range(1024)]
    audio_tokens_c1 = [f"<audio_c1_{i}>" for i in range(1024)]
    c0_ids = [tokenizer_v.convert_tokens_to_ids(t) for t in audio_tokens_c0]
    c1_ids = [tokenizer_v.convert_tokens_to_ids(t) for t in audio_tokens_c1]
    audio_end_id = tokenizer_v.convert_tokens_to_ids("<audio_end>")
    
    target_speech_text = selected_item["text"]
    print(f"Target Text for Speech Synthesis: '{target_speech_text}'")
    prompt_str_v = f"<|im_start|>user\nRead the following text: {target_speech_text}<|im_end|>\n<|im_start|>assistant\n<audio_start>"
    curr_input_ids_v = tokenizer_v(prompt_str_v, return_tensors="pt").input_ids
    prompt_len_v = curr_input_ids_v.shape[1]
    
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
        InterleavedAudioLogitsProcessor(prompt_len_v, c0_ids, c1_ids, audio_end_id)
    ])
    
    print("Generating speech tokens...")
    with torch.no_grad():
        generated_outputs_v = model_v.generate(
            input_ids=curr_input_ids_v,
            max_new_tokens=1500,
            eos_token_id=audio_end_id,
            pad_token_id=tokenizer_v.pad_token_id,
            do_sample=False,
            logits_processor=logits_processors
        )
        
    new_tokens_v = generated_outputs_v[0, prompt_len_v:].tolist()
    gen_text_v = tokenizer_v.decode(new_tokens_v, skip_special_tokens=False)
    
    c0_codes = []
    c1_codes = []
    tokens_parsed = re.findall(r"<(audio_c[01])_(\d+)>", gen_text_v)
    for codebook, val in tokens_parsed:
        if codebook == "audio_c0":
            c0_codes.append(int(val))
        elif codebook == "audio_c1":
            c1_codes.append(int(val))
            
    min_len = min(len(c0_codes), len(c1_codes))
    print(f"Generated {len(new_tokens_v)} audio tokens. Aligned to length {min_len}.")
    
    if min_len == 0:
        print("Error: No audio tokens generated.")
        return
        
    # 4. EnCodec Decoding & Post-Filtering
    print("\n--- STAGE 4: Decoding and Filtering Audio ---")
    encodec_model = EncodecModel.from_pretrained("facebook/encodec_24khz", local_files_only=True)
    encodec_model.eval()
    
    codes_tensor = torch.zeros(1, 1, 2, min_len, dtype=torch.long)
    codes_tensor[0, 0, 0, :] = torch.tensor(c0_codes[:min_len], dtype=torch.long)
    codes_tensor[0, 0, 1, :] = torch.tensor(c1_codes[:min_len], dtype=torch.long)
    
    with torch.no_grad():
        reconstructed_audio = encodec_model.decode(codes_tensor, [None], padding_mask=None).audio_values
        audio_data = reconstructed_audio.squeeze().cpu().numpy()
        
    # Low-pass filter at 7.5 kHz to clean up high-frequency noise
    filtered_data = butter_lowpass_filter(audio_data, cutoff=7500, fs=24000, order=6)
    
    # Normalize to peak at 0.8
    max_amp = np.abs(filtered_data).max()
    if max_amp > 1e-5:
        filtered_data = filtered_data * (0.8 / max_amp)
        
    audio_int16 = (filtered_data * 32767).clip(-32768, 32767).astype(np.int16)
    
    unique_name = f"synthesized_speech_{selected_item['id']}.wav"
    local_wav = workspace_dir / "synthesized_speech.wav"
    local_wav_unique = workspace_dir / unique_name
    home_wav = Path(r"C:\Users\Navie\synthesized_speech.wav")
    home_wav_unique = Path(r"C:\Users\Navie") / unique_name
    
    wavfile.write(str(local_wav), 24000, audio_int16)
    wavfile.write(str(local_wav_unique), 24000, audio_int16)
    shutil.copy(str(local_wav), str(home_wav))
    shutil.copy(str(local_wav_unique), str(home_wav_unique))
    print(f"\nSuccess! Spoken description saved to:")
    print(f"  - Generic: {home_wav}")
    print(f"  - Unique:  {home_wav_unique}")
    
    # Open folder
    os.startfile(r"C:\Users\Navie")
    print("Done!")

if __name__ == "__main__":
    main()
