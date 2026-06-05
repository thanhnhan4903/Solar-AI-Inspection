import json
from pathlib import Path

trace = Path("data/results/debug_logs/DJI_0029_R_final_polygon_trace.jsonl")
TARGET = {25, 40, 46, 47, 52, 62, 65, 67, 68, 77, 78, 86}

with open(trace, encoding="utf-8") as f:
    for line in f:
        p = json.loads(line)
        ridx = p.get("raw_idx")
        if ridx in TARGET:
            row_src = p.get("line_consensus_row_source", "N/A")
            global_used = p.get("line_consensus_global_row_used", "N/A")
            iou = p.get("line_consensus_iou", 0.0) or 0.0
            row_d = p.get("line_consensus_row_dist", 0.0) or 0.0
            col_d = p.get("line_consensus_col_dist", 0.0) or 0.0
            accept = p.get("line_consensus_rescue_accept")
            print(f"  {ridx:>3}  accept={str(accept):<5}  row_src={row_src:<6}  global_used={str(global_used):<5}  iou={iou:.4f}  row_d={row_d:.2f}  col_d={col_d:.2f}")
