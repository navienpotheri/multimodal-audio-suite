import torch
from transformers import EncodecModel
import numpy as np

c0_codes = [62, 408, 62, 779, 738, 780, 1019, 388, 906, 727, 727, 465, 502, 363, 502, 191, 563, 875, 228, 133, 73, 677, 537]
c1_codes = [913, 544, 544, 544, 913, 646, 424, 580, 913, 544, 601, 498, 446, 489, 1003, 105, 838, 105, 838, 60, 573, 573, 249]
min_len = min(len(c0_codes), len(c1_codes))

codes_tensor = torch.zeros(1, 1, 2, min_len, dtype=torch.long)
codes_tensor[0, 0, 0, :] = torch.tensor(c0_codes[:min_len], dtype=torch.long)
codes_tensor[0, 0, 1, :] = torch.tensor(c1_codes[:min_len], dtype=torch.long)

print("Loading Encodec model...")
encodec_model = EncodecModel.from_pretrained("facebook/encodec_24khz", local_files_only=True)
encodec_model.eval()

print("Decoding test generated codes...")
with torch.no_grad():
    reconstructed_audio = encodec_model.decode(codes_tensor, [None], padding_mask=None).audio_values
    audio_data = reconstructed_audio.squeeze().cpu().numpy()

print(f"Decoded test audio shape: {audio_data.shape}")
print(f"Decoded test audio Max: {audio_data.max():.6f}")
print(f"Decoded test audio Min: {audio_data.min():.6f}")
print(f"Decoded test audio Std: {audio_data.std():.6f}")
