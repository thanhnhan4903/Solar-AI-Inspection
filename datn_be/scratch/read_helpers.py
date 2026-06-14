import json
import sys

sys.stdout.reconfigure(encoding='utf-8')
log_path = r"C:\Users\ThanhNhan\.gemini\antigravity-ide\brain\c2fd1c4e-1335-4cae-a577-12dfb221980c\.system_generated\logs\transcript.jsonl"

with open(log_path, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        content = data.get("content", "")
        if "def evaluate_angle_candidate" in content:
            print("Step:", data["step_index"], "Type:", data["type"])
            # Dump to file
            with open(f"scratch/step_{data['step_index']}_helper.txt", "w", encoding="utf-8") as out:
                out.write(content)
            print("Wrote to scratch/step_", data["step_index"], "_helper.txt")
