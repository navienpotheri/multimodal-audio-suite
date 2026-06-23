import scipy.io.wavfile as wavfile
from scipy.signal import butter, lfilter
import numpy as np
import shutil
import os

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

local_wav = r"C:\Users\Navie\.gemini\antigravity\scratch\multimodal-audio-suite\synthesized_speech.wav"
filtered_local_wav = r"C:\Users\Navie\.gemini\antigravity\scratch\multimodal-audio-suite\synthesized_speech_filtered.wav"
home_wav = r"C:\Users\Navie\synthesized_speech_filtered.wav"

print("Reading WAV file...")
sr, data = wavfile.read(local_wav)

# Normalize data to float32 first
if data.dtype == np.int16:
    data_f32 = data.astype(np.float32) / 32768.0
else:
    data_f32 = data.astype(np.float32)

print("Applying low-pass filter at 7500 Hz...")
# Cutoff at 7.5 kHz is ideal for human voice (speech energy is mostly below 8 kHz)
filtered_data = butter_lowpass_filter(data_f32, cutoff=7500, fs=sr, order=6)

# Normalize amplitude to peak at 0.8
max_amp = np.abs(filtered_data).max()
if max_amp > 1e-5:
    filtered_data = filtered_data * (0.8 / max_amp)

# Convert back to int16
filtered_int16 = (filtered_data * 32767).clip(-32768, 32767).astype(np.int16)

print("Saving filtered file...")
wavfile.write(filtered_local_wav, sr, filtered_int16)
shutil.copy(filtered_local_wav, home_wav)
print(f"Filtered file saved to home directory: {home_wav}")

# Open home directory
os.startfile(r"C:\Users\Navie")
print("Done!")
