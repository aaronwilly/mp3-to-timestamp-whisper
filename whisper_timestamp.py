#!/usr/bin/env python3
"""
Simple Python project to convert MP3 audio to timestamped JSON using Whisper.
Supports both transcription (MP3 only) and text alignment (MP3 + text file).
"""

import json
import sys
import argparse
import subprocess
import tempfile
from pathlib import Path
import whisper


def whisper_to_json(audio_path, model_name="base", language=None):
    """
    Convert MP3 audio to timestamped JSON using Whisper.
    
    Args:
        audio_path: Path to the MP3 audio file
        model_name: Whisper model to use (tiny, base, small, medium, large)
        language: Language code (e.g., 'en', 'es'). If None, auto-detects.
    
    Returns:
        Dictionary with timestamped segments
    """
    # Load Whisper model
    print(f"Loading Whisper model: {model_name}...")
    model = whisper.load_model(model_name)
    
    # Transcribe audio
    print(f"Transcribing audio: {audio_path}...")
    result = model.transcribe(
        str(audio_path),
        language=language,
        verbose=False,
        word_timestamps=True
    )
    
    # Extract segments with timestamps
    segments = []
    for segment in result["segments"]:
        segment_data = {
            "id": segment["id"],
            "start": round(segment["start"], 2),
            "end": round(segment["end"], 2),
            "text": segment["text"].strip()
        }
        
        # Include word-level timestamps if available
        if "words" in segment:
            segment_data["words"] = [
                {
                    "word": word["word"],
                    "start": round(word["start"], 2),
                    "end": round(word["end"], 2)
                }
                for word in segment["words"]
            ]
        
        segments.append(segment_data)
    
    # Create output structure
    output = {
        "language": result.get("language", "unknown"),
        "duration": round(result.get("duration", 0), 2),
        "segments": segments
    }
    
    return output


def process_aeneas_syncmap(task, paragraphs, find_paragraph_for_text):
    """Process aeneas sync map directly for word-level alignment"""
    segments = []
    segment_id = 0
    current_paragraph = 0
    
    # Access sync map fragments - SyncMap has fragments property
    sync_map = task.sync_map
    if not sync_map:
        return {"language": "english", "duration": 0, "segments": []}
    
    # Access fragments properly - SyncMap might have fragments attribute or be accessed via len/index
    try:
        # Try accessing via fragments attribute
        if hasattr(sync_map, 'fragments'):
            fragments = sync_map.fragments
        elif hasattr(sync_map, '__len__'):
            # Access by index
            fragments = [sync_map[i] for i in range(len(sync_map))]
        else:
            # Try to iterate if it has __iter__
            fragments = [f for f in sync_map]
    except:
        # If all else fails, try to get fragments from task output
        print("Could not access sync map fragments directly, using subprocess method")
        return None
    
    for fragment in fragments:
        try:
            begin = float(fragment.begin)
            end = float(fragment.end)
            text = fragment.text.strip() if hasattr(fragment, 'text') else ""
            
            if not text:
                continue
            
            # Find paragraph
            paragraph_num = find_paragraph_for_text(text, current_paragraph)
            if paragraph_num >= current_paragraph:
                current_paragraph = paragraph_num
            paragraph_num = current_paragraph
            
            # Check if fragment has word-level data
            # In aeneas, word-level data might be in fragment.children or similar
            has_words = False
            if hasattr(fragment, 'children') and fragment.children:
                # Word-level fragments might be in children
                for word_frag in fragment.children:
                    try:
                        word_begin = float(word_frag.begin)
                        word_end = float(word_frag.end)
                        word_text = word_frag.text.strip() if hasattr(word_frag, 'text') else ""
                        
                        if word_text:
                            segments.append({
                                "id": segment_id,
                                "start": round(word_begin, 2),
                                "end": round(word_end, 2),
                                "text": word_text,
                                "paragraph": paragraph_num
                            })
                            segment_id += 1
                            has_words = True
                    except:
                        continue
            
            # If no word-level data found, split fragment text with natural overlaps
            if not has_words:
                words = text.split()
                if words:
                    # Calculate word positions based on character positions for better distribution
                    total_chars = sum(len(word) for word in words) + (len(words) - 1)  # Include spaces
                    if total_chars > 0:
                        char_positions = []
                        current_char = 0
                        for word in words:
                            char_positions.append((current_char, current_char + len(word)))
                            current_char += len(word) + 1  # +1 for space
                        
                        # Distribute words allowing natural overlaps
                        for i, word in enumerate(words):
                            original_word = word.strip()
                            if not original_word:
                                continue
                            
                            # Calculate word position based on character ratio
                            char_start, char_end = char_positions[i]
                            word_start_ratio = char_start / total_chars
                            word_end_ratio = char_end / total_chars
                            
                            # Calculate timestamps
                            word_start = begin + (end - begin) * word_start_ratio
                            word_end = begin + (end - begin) * word_end_ratio
                            
                            # Ensure minimum duration
                            if word_end <= word_start:
                                word_end = word_start + 0.15
                            
                            segments.append({
                                "id": segment_id,
                                "start": round(word_start, 2),
                                "end": round(word_end, 2),
                                "text": original_word,
                                "paragraph": paragraph_num
                            })
                            segment_id += 1
        except Exception as e:
            print(f"Error processing fragment: {e}")
            continue
    
    # Post-process segments to allow natural overlaps
    if segments:
        segments = apply_natural_overlaps(segments)
    
    duration = segments[-1]["end"] if segments else 0
    return {
        "language": "english",
        "duration": round(duration, 2),
        "segments": segments
    }


def process_aeneas_fragments(aeneas_data, paragraphs, find_paragraph_for_text):
    """Process aeneas fragments into word-level segments with paragraphs"""
    segments = []
    fragments = aeneas_data.get("fragments", [])
    segment_id = 0
    current_paragraph = 0
    
    for fragment in fragments:
        begin = float(fragment.get("begin", "0"))
        end = float(fragment.get("end", "0"))
        lines = fragment.get("lines", [])
        text = lines[0].strip() if lines else ""
        
        if not text:
            continue
        
        # Find paragraph
        paragraph_num = find_paragraph_for_text(text, current_paragraph)
        if paragraph_num >= current_paragraph:
            current_paragraph = paragraph_num
        paragraph_num = current_paragraph
        
        # Check for word-level data - handle both list and dict formats
        if "words" in fragment:
            words_data = fragment["words"]
            
            # Handle list format
            if isinstance(words_data, list) and len(words_data) > 0:
                for word_data in words_data:
                    # Handle both dict and string formats
                    if isinstance(word_data, dict):
                        word_begin = float(word_data.get("begin", word_data.get("start", begin)))
                        word_end = float(word_data.get("end", end))
                        word_text = word_data.get("word", word_data.get("text", "")).strip()
                    elif isinstance(word_data, str):
                        # If it's just a string, we can't get timestamps
                        word_text = word_data.strip()
                        word_begin = begin
                        word_end = end
                    else:
                        continue
                    
                    if word_text:
                        segments.append({
                            "id": segment_id,
                            "start": round(word_begin, 2),
                            "end": round(word_end, 2),
                            "text": word_text,
                            "paragraph": paragraph_num
                        })
                        segment_id += 1
            # Handle dict format
            elif isinstance(words_data, dict) and len(words_data) > 0:
                for key, word_data in words_data.items():
                    if isinstance(word_data, dict):
                        word_begin = float(word_data.get("begin", word_data.get("start", begin)))
                        word_end = float(word_data.get("end", end))
                        word_text = word_data.get("word", word_data.get("text", key)).strip()
                    else:
                        word_text = str(word_data).strip()
                        word_begin = begin
                        word_end = end
                    
                    if word_text:
                        segments.append({
                            "id": segment_id,
                            "start": round(word_begin, 2),
                            "end": round(word_end, 2),
                            "text": word_text,
                            "paragraph": paragraph_num
                        })
                        segment_id += 1
        else:
            # Fallback: distribute words with natural overlaps
            words = text.split()
            if words:
                total_chars = sum(len(word) for word in words)
                if total_chars > 0:
                    # Calculate cumulative character positions for better distribution
                    char_positions = []
                    current_char = 0
                    for word in words:
                        char_positions.append((current_char, current_char + len(word)))
                        current_char += len(word) + 1  # +1 for space
                    
                    # Distribute words with natural overlaps
                    for i, word in enumerate(words):
                        original_word = word.strip()
                        if not original_word:
                            continue
                        
                        # Calculate word position based on character ratio
                        char_start, char_end = char_positions[i]
                        word_start_ratio = char_start / total_chars
                        word_end_ratio = char_end / total_chars
                        
                        # Calculate timestamps with overlap
                        word_start = begin + (end - begin) * word_start_ratio
                        word_end = begin + (end - begin) * word_end_ratio
                        
                        # Ensure minimum duration
                        if word_end <= word_start:
                            word_end = word_start + 0.15
                        
                        # Allow natural overlap: next word can start before this one ends
                        # Overlap by ~20% of word duration for natural speech flow
                        word_duration = word_end - word_start
                        overlap = word_duration * 0.2
                        
                        segments.append({
                            "id": segment_id,
                            "start": round(word_start, 2),
                            "end": round(word_end, 2),
                            "text": original_word,
                            "paragraph": paragraph_num
                        })
                        segment_id += 1
    
    # Post-process segments to allow natural overlaps
    if segments:
        segments = apply_natural_overlaps(segments)
    
    duration = segments[-1]["end"] if segments else 0
    return {
        "language": "english",
        "duration": round(duration, 2),
        "segments": segments
    }


def apply_natural_overlaps(segments):
    """Post-process segments to allow natural speech overlaps"""
    if len(segments) < 2:
        return segments
    
    processed = []
    for i, seg in enumerate(segments):
        start = seg["start"]
        end = seg["end"]
        word_duration = end - start
        
        # Allow next word to start slightly before this word ends (natural speech overlap)
        if i < len(segments) - 1:
            next_seg = segments[i + 1]
            next_start = next_seg["start"]
            gap = next_start - end
            
            # Create natural overlap: next word should start before current word ends
            # Overlap by 20-30% of the shorter word's duration
            if gap < 0.5:  # Only for small gaps (not real pauses)
                overlap_ratio = 0.25  # 25% overlap
                next_word_duration = next_seg["end"] - next_start
                overlap = min(word_duration * overlap_ratio, next_word_duration * overlap_ratio, gap + 0.05)
                
                # Adjust next word start to create overlap
                if gap >= 0:
                    # There's a gap, create overlap
                    next_seg["start"] = max(start, end - overlap)
                else:
                    # Already overlapping, but might need adjustment
                    if gap < -0.1:  # Too much overlap, reduce it
                        next_seg["start"] = end - overlap
        
        processed.append(seg)
    
    return processed


def aeneas_align(audio_path, text_path):
    """
    Align existing text with audio using aeneas.
    
    Args:
        audio_path: Path to the MP3 audio file
        text_path: Path to the text file
    
    Returns:
        Dictionary with timestamped segments
    """
    # Use subprocess method (more reliable than direct import)
    # This works even if aeneas is installed via exe
    print(f"Aligning text from {text_path} with audio {audio_path}...")
    
    # Read text file to identify paragraphs
    with open(text_path, 'r', encoding='utf-8') as f:
        text_content = f.read()
    
    # Split into paragraphs (empty lines separate paragraphs)
    paragraphs = [p.strip() for p in text_content.split('\n\n') if p.strip()]
    print(f"Found {len(paragraphs)} paragraphs in text file")
    
    # Create a function to find which paragraph contains a given text fragment
    def find_paragraph_for_text(fragment_text, current_para=0):
        """Find which paragraph contains this fragment text"""
        fragment_text_clean = fragment_text.strip()
        if not fragment_text_clean:
            return current_para
        
        # Try exact match first
        for para_num, para_text in enumerate(paragraphs):
            if fragment_text_clean in para_text:
                return para_num
        
        # Try matching by first few words
        fragment_words = fragment_text_clean.split()[:5]
        if fragment_words:
            fragment_start = " ".join(fragment_words)
            for para_num, para_text in enumerate(paragraphs):
                if fragment_start in para_text:
                    return para_num
        
        # Default: stay in current paragraph
        return current_para
    
    # Create temporary output file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
        output_file = tmp_file.name
    
    try:
        # Try to use aeneas Python API directly for word-level data
        # This is more reliable than JSON output format
        try:
            from aeneas.executetask import ExecuteTask
            from aeneas.task import Task
            
            print("Attempting to use aeneas Python API for word-level alignment...")
            # Use finer-grained configuration for better word-level alignment
            config_string = "task_language=eng|is_text_type=plain|os_task_file_format=json|task_adjust_boundary_nonspeech_min=0.000|task_adjust_boundary_nonspeech_max=0.050"
            task = Task(config_string=config_string)
            task.audio_file_path_absolute = str(Path(audio_path).absolute())
            task.text_file_path_absolute = str(Path(text_path).absolute())
            
            # Execute task
            ExecuteTask(task).execute()
            
            # Check if we have word-level sync map
            if task.sync_map:
                # Try to get word-level fragments
                print("[OK] Got sync map from aeneas API")
                # Convert sync map to our format
                result = process_aeneas_syncmap(task, paragraphs, find_paragraph_for_text)
                if result:
                    return result
                else:
                    print("Could not process sync map, falling back to subprocess method...")
            else:
                print("No sync map from aeneas API, falling back to subprocess method...")
        except ImportError:
            print("aeneas Python API not available, using subprocess method...")
        except Exception as e:
            print(f"Error using aeneas API: {e}")
            print("Falling back to subprocess method...")
        
        # Fallback: Call aeneas via subprocess with word-level alignment
        # Use finer-grained configuration for better word-level alignment
        config_string = "task_language=eng|is_text_type=plain|os_task_file_format=json|task_adjust_boundary_nonspeech_min=0.000|task_adjust_boundary_nonspeech_max=0.050"
        result = subprocess.run(
            [
                sys.executable, "-m", "aeneas.tools.execute_task",
                str(Path(audio_path).absolute()),
                str(Path(text_path).absolute()),
                config_string,
                output_file,
                "--presets-word"
            ],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Read the JSON output
        with open(output_file, 'r', encoding='utf-8') as f:
            aeneas_data = json.load(f)
        
        # Check if we have word-level data
        fragments = aeneas_data.get("fragments", [])
        has_word_level = False
        word_count = 0
        for frag in fragments:
            if "words" in frag:
                words_data = frag.get("words")
                if isinstance(words_data, list) and len(words_data) > 0:
                    has_word_level = True
                    word_count += len(words_data)
                elif isinstance(words_data, dict) and len(words_data) > 0:
                    has_word_level = True
                    word_count += len(words_data)
        
        if has_word_level:
            print(f"[OK] Using aeneas word-level timestamps ({word_count} words found)")
        else:
            print("Using fragment-level with proportional word distribution...")
        
        # Process fragments
        return process_aeneas_fragments(aeneas_data, paragraphs, find_paragraph_for_text)
        
    except subprocess.CalledProcessError as e:
        print(f"Error running aeneas: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        sys.exit(1)
    except FileNotFoundError:
        print("Error: aeneas is not installed or not found in PATH.")
        print("Install it with: pip install aeneas")
        sys.exit(1)
    finally:
        # Clean up temp file
        try:
            Path(output_file).unlink()
        except:
            pass


def main():
    parser = argparse.ArgumentParser(
        description="Convert MP3 audio to timestamped JSON using Whisper"
    )
    parser.add_argument(
        "audio",
        type=str,
        help="Path to the MP3 audio file"
    )
    parser.add_argument(
        "-t", "--text",
        type=str,
        default=None,
        help="Optional: Path to text file (uses aeneas for alignment if provided)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output JSON file path (default: audio_filename_timestamps.json)"
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model to use (default: base)"
    )
    parser.add_argument(
        "-l", "--language",
        type=str,
        default=None,
        help="Language code (e.g., 'en', 'es'). Auto-detects if not specified."
    )
    
    args = parser.parse_args()
    
    # Validate input file
    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"Error: Audio file not found: {audio_path}")
        sys.exit(1)
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = audio_path.parent / f"{audio_path.stem}_timestamps.json"
    
    # Process audio
    try:
        if args.text:
            # Use aeneas for exact text alignment
            text_path = Path(args.text)
            if not text_path.exists():
                print(f"Error: Text file not found: {text_path}")
                sys.exit(1)
            output_data = aeneas_align(audio_path, text_path)
        else:
            # Use Whisper for transcription
            output_data = whisper_to_json(audio_path, model_name=args.model, language=args.language)
        
        # Save to JSON file
        print(f"Saving timestamps to: {output_path}")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"[OK] Success! Generated {len(output_data['segments'])} segments.")
        print(f"  Duration: {output_data['duration']} seconds")
        print(f"  Language: {output_data['language']}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

