![App Screenshot](./screenshot.png)
# Whisper Timestamp Generator

A simple Python project to convert MP3 audio files to timestamped JSON files using OpenAI Whisper.

## Features

- Transcribe MP3 audio files and generate JSON with timestamps
- Support for word-level timestamps
- Optional text alignment using aeneas (if you already have the text)
- Multiple Whisper model sizes (tiny, base, small, medium, large)

## Installation

1. Activate your virtual environment (if using one):
```bash
# Windows PowerShell
.\venv\Scripts\Activate.ps1

# Linux/Mac
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Optional: Installing aeneas (for text alignment)

The `aeneas` package is optional and only needed if you want to align existing text files with audio. The main Whisper transcription feature works without it.

If you need aeneas, it can be tricky to install on Windows. Try these steps:

```bash
# Make sure numpy is installed first
pip install numpy

# Then try installing aeneas
pip install aeneas
```

Note: On some systems, you may need to install additional dependencies for aeneas:
- FFmpeg (for audio processing)
- eSpeak (for text-to-speech synthesis)

If aeneas installation fails, you can still use the main Whisper transcription feature without it.

## Usage

### Basic Usage (MP3 only - Whisper transcription)

```bash
python whisper_timestamp.py your_audio.mp3
```

This will create `your_audio_timestamps.json` with transcribed text and timestamps.

### Specify Output File

```bash
python whisper_timestamp.py your_audio.mp3 -o output.json
```

### Use Different Whisper Model

```bash
python whisper_timestamp.py your_audio.mp3 -m small
```

Available models: `tiny`, `base`, `small`, `medium`, `large`
- `tiny`: Fastest, least accurate
- `base`: Good balance (default)
- `large`: Most accurate, slowest

### Specify Language

```bash
python whisper_timestamp.py your_audio.mp3 -l en
```

### Use Existing Text File (aeneas alignment)

If you already have the text and want to align it with the audio:

```bash
python whisper_timestamp.py your_audio.mp3 -t your_text.txt
```

## Output Format

The generated JSON file has the following structure:

```json
{
  "language": "en",
  "duration": 120.5,
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 5.2,
      "text": "Hello, this is a test.",
      "words": [
        {
          "word": "Hello,",
          "start": 0.0,
          "end": 0.8
        },
        {
          "word": "this",
          "start": 0.8,
          "end": 1.2
        }
      ]
    }
  ]
}
```

## Requirements

- Python 3.8+
- openai-whisper
- aeneas (optional, for text alignment)

## Notes

- First run will download the Whisper model (can be large, especially for 'large' model)
- Processing time depends on audio length and model size
- For best accuracy, use the 'large' model, but it requires more memory and time

