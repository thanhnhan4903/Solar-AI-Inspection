import json

trace_path = "data/results/debug_logs/DJI_0029_R_final_polygon_trace.jsonl"
targets = {25, 40, 46, 47, 52, 62, 65, 67, 68, 77, 78, 86}

with open(trace_path, "r", encoding="utf-8") as f:
    for line in f:
        rec = json.loads(line.strip())
        raw_idx = rec.get("raw_idx")
        if raw_idx in targets:
            print(f"Index: {raw_idx}")
            print(f"  Source: {rec.get('final_polygon_source')}")
            print(f"  Reason: {rec.get('final_polygon_reason')}")
            print(f"  Rescue Accept: {rec.get('line_consensus_rescue_accept')}")
            print(f"  Rescue Reject Reason: {rec.get('line_consensus_rescue_reject_reason')}")
            print(f"  Row Dist / Supp: {rec.get('line_consensus_row_dist')} / {rec.get('line_consensus_row_support')}")
            print(f"  Col Dist / Supp: {rec.get('line_consensus_col_dist')} / {rec.get('line_consensus_col_support')}")
            print(f"  IoU / Center Shift: {rec.get('line_consensus_iou')} / {rec.get('line_consensus_center_shift')}")
            print(f"  Area Ratio / Max Overlap: {rec.get('line_consensus_area_ratio')} / {rec.get('line_consensus_max_overlap')}")
            print(f"  Overlap Approx: {rec.get('line_consensus_overlap_approximate')}")
            print("-" * 50)
