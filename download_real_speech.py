"""
Downloads 10 actual human speech WAV files from the Free Spoken Digit Dataset
and replaces the synthetic synthesizer waveforms to enable training on real spoken voice.
"""

import os
import urllib.request
from pathlib import Path

def main():
    workspace_dir = Path(r"C:\Users\Navie\.gemini\antigravity\scratch\multimodal-audio-suite")
    dataset_dir = workspace_dir / "synthetic_ljspeech"
    wav_dir = dataset_dir / "wavs"
    
    # Create directories if missing
    wav_dir.mkdir(parents=True, exist_ok=True)
    
    # We will download 10 speech recordings of digits 0 to 9 spoken by speaker 'jackson'
    digit_words = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    
    print("Downloading 10 real human speech samples from Jakobovski/free-spoken-digit-dataset...")
    
    metadata = []
    
    import sys
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

    for i, word in enumerate(digit_words):
        file_id = f"LJ{i+1:03d}-0001"
        url = f"https://raw.githubusercontent.com/Jakobovski/free-spoken-digit-dataset/master/recordings/{i}_jackson_0.wav"
        dest_path = wav_dir / f"{file_id}.wav"
        
        print(f"  Downloading: {url} -> {dest_path.name}")
        try:
            urllib.request.urlretrieve(url, str(dest_path))
            metadata.append([file_id, word, word])
        except Exception as e:
            print(f"Error downloading {url}: {e}")
            
    # Write metadata.csv
    csv_path = dataset_dir / "metadata.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        import csv
        writer = csv.writer(f, delimiter="|")
        writer.writerows(metadata)
        
    print(f"\nReal speech dataset successfully downloaded and configured at: {dataset_path if 'dataset_path' in locals() else dataset_dir}")
    print(f"Transcripts metadata written to: {csv_path}")

if __name__ == "__main__":
    main()
