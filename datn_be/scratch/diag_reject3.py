import json
from pathlib import Path

trace = Path("data/results/debug_logs/DJI_0029_R_final_polygon_trace.jsonl")
TARGET = {46, 52, 77}
fields = [
    "raw_idx", "original_yolo_bbox", "bbox", "polygon",
    "line_consensus_rescue_reject_reason",
    "line_consensus_row_support", "line_consensus_col_support",
    "line_consensus_row_dist", "line_consensus_col_dist",
    "line_consensus_iou", "line_consensus_center_shift",
    "line_consensus_area_ratio", "line_consensus_max_overlap",
    "geometry_gate_relaxed_accept", "final_polygon_reason",
    "block_id", "string_id",
]

with open(trace, encoding="utf-8") as f:
    for line in f:
        p = json.loads(line)
        if p.get("raw_idx") in TARGET:
            rid = p["raw_idx"]
            print(f"\n=== raw_idx={rid} ===")
            for k in fields:
                v = p.get(k, "N/A")
                print(f"  {k}: {v}")
