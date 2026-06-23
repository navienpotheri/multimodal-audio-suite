import torch
from transformers import EncodecModel
from scipy.io import wavfile
import numpy as np
import os

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

print("Loading original cached codes...")
path = r"C:\Users\Navie\.gemini\antigravity\scratch\multimodal-audio-suite\cache_bootstrapped\discrete\BOOT00001-0001.pt"
codes = torch.load(path, map_location="cpu")
print("Original codes shape:", codes.shape)

# Reconstruct codes matrix: shape [1, 1, num_codebooks, seq_len]
codes_tensor = codes.unsqueeze(0).unsqueeze(0)
print("Reshaped codes tensor for Encodec:", codes_tensor.shape)

print("Loading Encodec model...")
encodec_model = EncodecModel.from_pretrained("facebook/encodec_24khz", local_files_only=True)
encodec_model.eval()

print("Decoding original codes...")
with torch.no_grad():
    reconstructed_audio = encodec_model.decode(codes_tensor, [None], padding_mask=None).audio_values
    audio_data = reconstructed_audio.squeeze().cpu().numpy()

print(f"Decoded original audio shape: {audio_data.shape}")
print(f"Decoded original audio Max: {audio_data.max():.6f}")
print(f"Decoded original audio Min: {audio_data.min():.6f}")
print(f"Decoded original audio Std: {audio_data.std():.6f}")

# Normalize to 16-bit integer range
audio_int16 = (audio_data * 32767).clip(-32768, 32767).astype(np.int16)
print(f"Decoded int16 original audio Max: {audio_int16.max()}")
print(f"Decoded int16 original audio Min: {audio_int16.min()}")

# Write to test wav file
wavfile.write("test_reconstructed_original.wav", 24000, audio_int16)
print("Saved to test_reconstructed_original.wav")
