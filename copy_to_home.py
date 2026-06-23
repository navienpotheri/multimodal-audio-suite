"""
Copies the synthesized speech WAV file to the user's visible home directory
and opens the directory in File Explorer.
"""

import os
import shutil
from pathlib import Path

def main():
    source_file = Path(r"C:\Users\Navie\.gemini\antigravity\scratch\multimodal-audio-suite\synthesized_speech.wav")
    dest_dir = Path(r"C:\Users\Navie")
    dest_file = dest_dir / "synthesized_speech.wav"
    
    if not source_file.exists():
        print(f"Error: Synthesized audio file not found at: {source_file}")
        return
        
    print(f"Copying {source_file.name} to {dest_file}...")
    try:
        shutil.copy(source_file, dest_file)
        print("Copy successful!")
        
        print(f"Opening folder {dest_dir} in File Explorer...")
        os.startfile(str(dest_dir))
        print("Done.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
