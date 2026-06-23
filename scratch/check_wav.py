from scipy.io import wavfile
import numpy as np

try:
    sr, data = wavfile.read(r"C:\Users\Navie\synthesized_speech.wav")
    print(f"Sampling Rate: {sr}")
    print(f"Shape: {data.shape}")
    print(f"Dtype: {data.dtype}")
    print(f"Max value: {data.max()}")
    print(f"Min value: {data.min()}")
    print(f"Mean value: {data.mean():.6f}")
    print(f"Std dev: {data.std():.6f}")
    print(f"First 20 samples: {data[:20]}")
    
    # Check if all elements are the same
    if np.all(data == data[0]):
        print(f"CRITICAL: Audio data is completely flat (all values are {data[0]}).")
    else:
        print("Audio data contains varying values.")
except Exception as e:
    print(f"Error checking WAV file: {e}")
