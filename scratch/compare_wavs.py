from scipy.io import wavfile
import numpy as np

def check_file(path):
    print(f"Checking {path}...")
    try:
        sr, data = wavfile.read(path)
        print(f"  Sampling Rate: {sr}")
        print(f"  Shape: {data.shape}")
        print(f"  Dtype: {data.dtype}")
        print(f"  Max value: {data.max()}")
        print(f"  Min value: {data.min()}")
        print(f"  Std dev: {data.std():.6f}")
        print(f"  First 10 samples: {data[:10]}")
    except Exception as e:
        print(f"  Error reading file: {e}")

check_file(r"C:\Users\Navie\.gemini\antigravity\scratch\multimodal-audio-suite\synthesized_speech.wav")
check_file(r"C:\Users\Navie\synthesized_speech.wav")
