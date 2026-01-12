#!/usr/bin/env python3
"""
Test script to verify timestamp accuracy of generated JSON files.
Checks for common issues like overlaps, gaps, ordering problems, and duration limits.
"""

import json
import random
import sys
from pathlib import Path


def load_json(json_path):
    """Load the timestamp JSON file"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def check_timestamp_accuracy(json_data, audio_duration=None):
    """Check for timestamp accuracy issues"""
    segments = json_data.get("segments", [])
    if not segments:
        print("❌ No segments found in JSON file")
        return
    
    print(f"Analyzing {len(segments)} segments...")
    print(f"   Duration: {json_data.get('duration', 0):.2f} seconds")
    if audio_duration:
        print(f"   Audio duration: {audio_duration:.2f} seconds")
    
    issues = []
    
    # Check 1: Sequential ordering
    print("\n[1] Checking sequential ordering...")
    prev_end = 0
    ordering_issues = 0
    for i, seg in enumerate(segments):
        start = seg.get("start", 0)
        end = seg.get("end", 0)
        
        if start < prev_end:
            ordering_issues += 1
            if ordering_issues <= 5:  # Show first 5 issues
                issues.append({
                    "type": "ordering",
                    "id": seg.get("id"),
                    "word": seg.get("text"),
                    "start": start,
                    "prev_end": prev_end,
                    "gap": start - prev_end
                })
        prev_end = end
    
    if ordering_issues > 0:
        print(f"   [WARNING] Found {ordering_issues} ordering issues (start time < previous end time)")
        for issue in issues[:5]:
            print(f"      ID {issue['id']}: '{issue['word']}' starts at {issue['start']:.2f}s (prev ended at {issue['prev_end']:.2f}s, gap: {issue['gap']:.2f}s)")
    else:
        print("   [OK] All timestamps are sequential")
    
    # Check 2: Overlaps
    print("\n[2] Checking for overlaps...")
    overlaps = 0
    overlap_issues = []
    for i in range(len(segments) - 1):
        curr = segments[i]
        next_seg = segments[i + 1]
        
        curr_end = curr.get("end", 0)
        next_start = next_seg.get("start", 0)
        
        if curr_end > next_start:
            overlaps += 1
            if overlaps <= 5:
                overlap_issues.append({
                    "id1": curr.get("id"),
                    "word1": curr.get("text"),
                    "end1": curr_end,
                    "id2": next_seg.get("id"),
                    "word2": next_seg.get("text"),
                    "start2": next_start,
                    "overlap": curr_end - next_start
                })
    
    if overlaps > 0:
        print(f"   [WARNING] Found {overlaps} overlaps")
        for issue in overlap_issues[:5]:
            print(f"      '{issue['word1']}' (ID {issue['id1']}) ends at {issue['end1']:.2f}s")
            print(f"      '{issue['word2']}' (ID {issue['id2']}) starts at {issue['start2']:.2f}s")
            print(f"      Overlap: {issue['overlap']:.2f}s")
    else:
        print("   [OK] No overlaps found")
    
    # Check 3: Large gaps
    print("\n[3] Checking for large gaps...")
    large_gaps = 0
    gap_issues = []
    for i in range(len(segments) - 1):
        curr = segments[i]
        next_seg = segments[i + 1]
        
        curr_end = curr.get("end", 0)
        next_start = next_seg.get("start", 0)
        gap = next_start - curr_end
        
        if gap > 1.0:  # Gaps larger than 1 second
            large_gaps += 1
            if large_gaps <= 5:
                gap_issues.append({
                    "id1": curr.get("id"),
                    "word1": curr.get("text"),
                    "end1": curr_end,
                    "id2": next_seg.get("id"),
                    "word2": next_seg.get("text"),
                    "start2": next_start,
                    "gap": gap
                })
    
    if large_gaps > 0:
        print(f"   [WARNING] Found {large_gaps} large gaps (>1s)")
        for issue in gap_issues[:5]:
            print(f"      '{issue['word1']}' (ID {issue['id1']}) ends at {issue['end1']:.2f}s")
            print(f"      '{issue['word2']}' (ID {issue['id2']}) starts at {issue['start2']:.2f}s")
            print(f"      Gap: {issue['gap']:.2f}s")
    else:
        print("   [OK] No large gaps found")
    
    # Check 4: Duration limits
    if audio_duration:
        print(f"\n[4] Checking if timestamps exceed audio duration ({audio_duration:.2f}s)...")
        exceeded = 0
        exceed_issues = []
        for seg in segments:
            start = seg.get("start", 0)
            end = seg.get("end", 0)
            
            if start > audio_duration or end > audio_duration:
                exceeded += 1
                if exceeded <= 5:
                    exceed_issues.append({
                        "id": seg.get("id"),
                        "word": seg.get("text"),
                        "start": start,
                        "end": end
                    })
        
        if exceeded > 0:
            print(f"   [WARNING] Found {exceeded} segments exceeding audio duration")
            for issue in exceed_issues[:5]:
                print(f"      ID {issue['id']}: '{issue['word']}' ({issue['start']:.2f}s - {issue['end']:.2f}s)")
        else:
            print("   [OK] All timestamps within audio duration")
    
    # Check 5: Word duration (too short or too long)
    print("\n[5] Checking word durations...")
    too_short = 0
    too_long = 0
    duration_issues = []
    for seg in segments:
        start = seg.get("start", 0)
        end = seg.get("end", 0)
        duration = end - start
        
        if duration < 0.05:  # Less than 50ms
            too_short += 1
            if too_short <= 5:
                duration_issues.append({
                    "type": "short",
                    "id": seg.get("id"),
                    "word": seg.get("text"),
                    "duration": duration
                })
        elif duration > 2.0:  # More than 2 seconds
            too_long += 1
            if too_long <= 5:
                duration_issues.append({
                    "type": "long",
                    "id": seg.get("id"),
                    "word": seg.get("text"),
                    "duration": duration
                })
    
    if too_short > 0:
        print(f"   [WARNING] Found {too_short} words with very short duration (<0.05s)")
        for issue in [d for d in duration_issues if d["type"] == "short"][:5]:
            print(f"      ID {issue['id']}: '{issue['word']}' ({issue['duration']:.3f}s)")
    if too_long > 0:
        print(f"   [WARNING] Found {too_long} words with very long duration (>2.0s)")
        for issue in [d for d in duration_issues if d["type"] == "long"][:5]:
            print(f"      ID {issue['id']}: '{issue['word']}' ({issue['duration']:.2f}s)")
    if too_short == 0 and too_long == 0:
        print("   [OK] Word durations look reasonable")
    
    # Show random samples
    print("\nRandom sample words (for manual verification):")
    print("   Format: [ID] 'word' | Start: X.XXs | End: Y.YYs | Duration: Z.ZZs | Paragraph: N")
    print("   " + "-" * 80)
    
    sample_size = min(10, len(segments))
    sample_indices = random.sample(range(len(segments)), sample_size)
    sample_indices.sort()
    
    for idx in sample_indices:
        seg = segments[idx]
        start = seg.get("start", 0)
        end = seg.get("end", 0)
        duration = end - start
        word = seg.get("text", "")
        para = seg.get("paragraph", 0)
        seg_id = seg.get("id", idx)
        
        # Show context (previous and next word)
        context = ""
        if idx > 0:
            prev_word = segments[idx - 1].get("text", "")
            prev_end = segments[idx - 1].get("end", 0)
            gap = start - prev_end
            context += f"< '{prev_word}' ({prev_end:.2f}s) [gap: {gap:.2f}s] "
        
        context += f"'{word}'"
        
        if idx < len(segments) - 1:
            next_word = segments[idx + 1].get("text", "")
            next_start = segments[idx + 1].get("start", 0)
            gap = next_start - end
            context += f" > '{next_word}' ({next_start:.2f}s) [gap: {gap:.2f}s]"
        
        print(f"   [{seg_id:4d}] {context}")
        print(f"        Start: {start:7.2f}s | End: {end:7.2f}s | Duration: {duration:5.2f}s | Para: {para}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    total_issues = ordering_issues + overlaps + large_gaps + (exceeded if audio_duration else 0) + too_short + too_long
    if total_issues == 0:
        print("[OK] No issues detected! Timestamps look accurate.")
    else:
        print(f"[WARNING] Found {total_issues} potential issues:")
        if ordering_issues > 0:
            print(f"   - {ordering_issues} ordering issues")
        if overlaps > 0:
            print(f"   - {overlaps} overlaps")
        if large_gaps > 0:
            print(f"   - {large_gaps} large gaps")
        if audio_duration and exceeded > 0:
            print(f"   - {exceeded} segments exceeding duration")
        if too_short > 0:
            print(f"   - {too_short} words too short")
        if too_long > 0:
            print(f"   - {too_long} words too long")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test timestamp accuracy of generated JSON files"
    )
    parser.add_argument(
        "json_file",
        type=str,
        help="Path to the generated timestamp JSON file"
    )
    parser.add_argument(
        "-a", "--audio-duration",
        type=float,
        default=None,
        help="Actual audio duration in seconds (optional, for validation)"
    )
    
    args = parser.parse_args()
    
    json_path = Path(args.json_file)
    if not json_path.exists():
        print(f"Error: JSON file not found: {json_path}")
        sys.exit(1)
    
    print("=" * 80)
    print("TIMESTAMP ACCURACY TEST")
    print("=" * 80)
    print(f"Testing: {json_path}")
    print()
    
    try:
        json_data = load_json(json_path)
        check_timestamp_accuracy(json_data, args.audio_duration)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

