import json
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')
log_path = r"C:\Users\ThanhNhan\.gemini\antigravity-ide\brain\c2fd1c4e-1335-4cae-a577-12dfb221980c\.system_generated\logs\transcript.jsonl"

with open(log_path, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        content = data.get("content", "")
        if "good_y_snaps" in content or "estimated_pitch" in content:
            print("Step:", data["step_index"], "Type:", data["type"])
            # Print matching lines with context
            lines = content.split("\n")
            for i, l in enumerate(lines):
                if any(x in l for x in ("good_y_snaps", "estimated_pitch", "pitch_std", "snapped_horiz_gaps.append")):
                    # Print 5 lines before and after
                    start = max(0, i - 4)
                    end = min(len(lines), i + 10)
                    print(f"--- Context for step {data['step_index']}, line {i} ---")
                    for j in range(start, end):
                        print(f"{j:4d}: {lines[j][:120]}")
                    print("-" * 50)
