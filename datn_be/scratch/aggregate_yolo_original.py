# Temporary script to aggregate yolo_original panels from final polygon trace
# Usage: python scratch/aggregate_yolo_original.py

import json
import pathlib
from collections import defaultdict

# Paths (adjust if needed)
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
TRACE_PATH = PROJECT_ROOT / "data" / "results" / "debug_logs" / "DJI_0843_R_final_polygon_trace.jsonl"

# Aggregation containers
total_yolo = 0
by_stage = defaultdict(int)
by_reason = defaultdict(int)
by_string_id = defaultdict(int)
by_block_id = defaultdict(int)
examples = []

with TRACE_PATH.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        # Only consider panel records (they have 'final_polygon_source')
        if "final_polygon_source" not in rec:
            continue
        source = rec.get("final_polygon_source")
        if source == "yolo_original":
            total_yolo += 1
            stage = rec.get("final_polygon_stage")
            if stage:
                by_stage[stage] += 1
            reason = rec.get("final_polygon_reason")
            if reason:
                by_reason[reason] += 1
            sid = rec.get("string_id")
            if sid is not None:
                by_string_id[sid] += 1
            bid = rec.get("block_id")
            if bid is not None:
                by_block_id[bid] += 1
            if len(examples) < 10:
                examples.append({
                    "raw_idx": rec.get("raw_idx"),
                    "bbox": rec.get("bbox"),
                    "area": rec.get("area"),
                    "final_polygon_stage": stage,
                    "final_polygon_reason": reason,
                    "string_id": sid,
                    "block_id": bid,
                })

# Output results
print(f"total yolo_original: {total_yolo}\n")
print("Grouped by final_polygon_stage:")
for k, v in by_stage.items():
    print(f"  {k}: {v}")
print("\nGrouped by final_polygon_reason:")
for k, v in by_reason.items():
    print(f"  {k}: {v}")
print("\nGrouped by string_id:")
for k, v in sorted(by_string_id.items()):
    print(f"  {k}: {v}")
print("\nGrouped by block_id:")
for k, v in sorted(by_block_id.items()):
    print(f"  {k}: {v}")

print("\nFirst 10 examples:")
for ex in examples:
    print(ex)
