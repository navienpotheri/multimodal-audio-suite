import torch

path = r"C:\Users\Navie\.gemini\antigravity\scratch\multimodal-audio-suite\cache_bootstrapped\discrete\BOOT00001-0001.pt"
codes = torch.load(path, map_location="cpu")
print("Codes shape:", codes.shape)
print("Dtype:", codes.dtype)
print("Codebook 0 - Min:", codes[0].min().item(), "Max:", codes[0].max().item())
print("Codebook 1 - Min:", codes[1].min().item(), "Max:", codes[1].max().item())
print("Codebook 0 - first 20 tokens:")
print(codes[0, :20].tolist())
print("Codebook 1 - first 20 tokens:")
print(codes[1, :20].tolist())
