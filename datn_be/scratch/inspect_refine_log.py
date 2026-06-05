import json
import os

refine_path = "data/results/debug_logs/DJI_0029_R_string_lattice_refine.jsonl"
if os.path.exists(refine_path):
    with open(refine_path, "r", encoding="utf-8") as f:
        count = 0
        for line in f:
            rec = json.loads(line)
            if rec.get("record_type") == "panel":
                print(f"record keys: {list(rec.keys())}")
                print(f"block_id: {rec.get('block_id')}")
                print(f"string_id: {rec.get('string_id')}")
                count += 1
                if count >= 3:
                    break
else:
    print("Refine log not found")
