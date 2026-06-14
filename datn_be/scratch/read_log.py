import json
import sys

sys.stdout.reconfigure(encoding='utf-8')
log_path = r"C:\Users\ThanhNhan\.gemini\antigravity-ide\brain\c2fd1c4e-1335-4cae-a577-12dfb221980c\.system_generated\logs\transcript.jsonl"

with open(log_path, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        idx = data.get("step_index", -1)
        if 150 <= idx <= 175:
            preview = data.get("content", "")[:200].replace("\n", " ")
            print(f"Step {idx}: Type={data['type']}, Length={len(data.get('content', ''))}, Preview={preview}")
