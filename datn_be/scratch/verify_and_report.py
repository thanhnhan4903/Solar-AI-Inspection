import json
import os

trace_path = "data/results/debug_logs/DJI_0029_R_final_polygon_trace.jsonl"
target_idxs = [25, 40, 46, 47, 52, 62, 65, 67, 68, 77, 78, 86]

if not os.path.exists(trace_path):
    print("Error: Trace file not found.")
else:
    print("=" * 100)
    print("TARGET PANELS REPORT FOR DJI_0029_R")
    print("=" * 100)
    print(f"| raw_idx | source_before | final_source | final_reason | rescue_accept | rescue_reject_reason | row_supp | col_supp | row_dist | col_dist | IoU | center_shift | area_ratio | max_overlap | approx |")
    print(f"|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    
    with open(trace_path, "r", encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            if p["raw_idx"] in target_idxs:
                idx = p["raw_idx"]
                final_source = p.get("final_polygon_source")
                final_reason = p.get("final_polygon_reason")
                rescue_accept = p.get("line_consensus_rescue_accept")
                rescue_reject = p.get("line_consensus_rescue_reject_reason")
                row_supp = p.get("line_consensus_row_support")
                col_supp = p.get("line_consensus_col_support")
                row_dist = p.get("line_consensus_row_dist")
                col_dist = p.get("line_consensus_col_dist")
                iou = p.get("line_consensus_iou")
                shift = p.get("line_consensus_center_shift")
                area_ratio = p.get("line_consensus_area_ratio")
                overlap = p.get("line_consensus_max_overlap")
                approx = p.get("line_consensus_overlap_approximate")
                
                # Format floats
                row_dist_str = f"{row_dist:.2f}" if row_dist is not None else "None"
                col_dist_str = f"{col_dist:.2f}" if col_dist is not None else "None"
                iou_str = f"{iou:.4f}" if iou is not None else "None"
                shift_str = f"{shift:.2f}" if shift is not None else "None"
                area_ratio_str = f"{area_ratio:.4f}" if area_ratio is not None else "None"
                overlap_str = f"{overlap:.4f}" if overlap is not None else "None"
                
                print(f"| {idx} | yolo_original | {final_source} | {final_reason} | {rescue_accept} | {rescue_reject} | {row_supp} | {col_supp} | {row_dist_str} | {col_dist_str} | {iou_str} | {shift_str} | {area_ratio_str} | {overlap_str} | {approx} |")
    print("=" * 100)
