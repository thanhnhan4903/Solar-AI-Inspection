import json
import os

trace_path = "data/results/debug_logs/DJI_0029_R_final_polygon_trace.jsonl"
target_idxs = [25, 40, 46, 47, 52, 62, 65, 67, 68, 77, 78, 86]

if not os.path.exists(trace_path):
    print("Trace path not found.")
else:
    with open(trace_path, "r", encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            if p["raw_idx"] in target_idxs:
                print(f"raw_idx: {p['raw_idx']}")
                print(f"  is_endpoint: {p.get('is_endpoint') or p.get('is_endpoint_panel')}")
                print(f"  geometry_gate_accept: {p.get('geometry_gate_accept')}")
                print(f"  geometry_gate_reason: {p.get('geometry_gate_reason')}")
                print(f"  final_polygon_source: {p.get('final_polygon_source')}")
                print(f"  final_polygon_reason: {p.get('final_polygon_reason')}")
                print(f"  bbox: {p.get('bbox')}")
                print(f"  original_yolo_bbox: {p.get('original_yolo_bbox')}")
