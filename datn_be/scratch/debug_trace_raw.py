import json
import os

trace_path = "data/results/debug_logs/DJI_0029_R_final_polygon_trace.jsonl"
if os.path.exists(trace_path):
    with open(trace_path, "r", encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            if p["raw_idx"] == 25:
                print(json.dumps(p, indent=2))
else:
    print("Trace not found")
