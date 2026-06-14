import json
import sys

sys.stdout.reconfigure(encoding='utf-8')
log_path = r"C:\Users\ThanhNhan\.gemini\antigravity-ide\brain\c2fd1c4e-1335-4cae-a577-12dfb221980c\.system_generated\logs\transcript.jsonl"

with open(log_path, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        idx = data.get("step_index", -1)
        if idx == 216 and data["type"] == "VIEW_FILE":
            print(f"Writing Step {idx} (Length: {len(data['content'])})")
            with open(f"scratch/step_{idx}_content.txt", "w", encoding="utf-8") as out:
                out.write(data["content"])
