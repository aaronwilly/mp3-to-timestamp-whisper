#!/usr/bin/env python3
"""
Example usage of whisper_timestamp.py
This script demonstrates how to use the whisper timestamp generator programmatically.
"""

from whisper_timestamp import whisper_to_json, aeneas_align
from pathlib import Path
import json


def example_whisper_transcription():
    """Example: Transcribe MP3 using Whisper"""
    print("Example 1: Whisper Transcription")
    print("-" * 50)
    
    # Replace with your actual MP3 file path
    audio_file = "your_audio.mp3"
    
    if not Path(audio_file).exists():
        print(f"Note: {audio_file} not found. Please replace with your actual MP3 file.")
        return
    
    # Transcribe with Whisper (using base model)
    result = whisper_to_json(audio_file, model_name="base")
    
    # Save to JSON
    output_file = "example_output.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Transcribed {len(result['segments'])} segments")
    print(f"✓ Saved to {output_file}")
    print(f"  Duration: {result['duration']} seconds")
    print(f"  Language: {result['language']}\n")


def example_text_alignment():
    """Example: Align existing text with audio using aeneas"""
    print("Example 2: Text Alignment with aeneas")
    print("-" * 50)
    
    # Replace with your actual files
    audio_file = "your_audio.mp3"
    text_file = "your_text.txt"
    
    if not Path(audio_file).exists() or not Path(text_file).exists():
        print(f"Note: Files not found. Please replace with your actual files.")
        return
    
    # Align text with audio
    result = aeneas_align(audio_file, text_file)
    
    # Save to JSON
    output_file = "example_aligned_output.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Aligned {len(result['segments'])} segments")
    print(f"✓ Saved to {output_file}\n")


if __name__ == "__main__":
    print("Whisper Timestamp Generator - Examples\n")
    
    # Run examples (uncomment the one you want to test)
    # example_whisper_transcription()
    # example_text_alignment()
    
    print("To run examples, uncomment the function calls in this file.")
    print("Or use the command line interface:")
    print("  python whisper_timestamp.py your_audio.mp3")

