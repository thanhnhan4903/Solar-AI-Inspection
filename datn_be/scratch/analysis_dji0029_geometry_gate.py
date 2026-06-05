import json
import os
import numpy as np

img = "DJI_0029_R"
trace_path = f"data/results/debug_logs/{img}_final_polygon_trace.jsonl"
refine_path = f"data/results/debug_logs/{img}_string_lattice_refine.jsonl"

target_idxs = [25, 40, 46, 47, 52, 62, 65, 67, 68, 77, 78, 86]

trace_data = {}
if os.path.exists(trace_path):
    with open(trace_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = json.loads(line)
            r_idx = p.get("raw_idx")
            if r_idx in target_idxs:
                trace_data[r_idx] = p

refine_data = {}
if os.path.exists(refine_path):
    with open(refine_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = json.loads(line)
            if p.get("record_type") == "panel":
                p_idx = p.get("panel_idx")
                if p_idx in target_idxs:
                    refine_data[p_idx] = p

results = []

for idx in target_idxs:
    t_rec = trace_data.get(idx, {})
    r_rec = refine_data.get(idx, {})
    
    bbox = t_rec.get("bbox", r_rec.get("bbox", [0, 0, 0, 0]))
    ox1, oy1, ox2, oy2 = bbox
    orig_h = oy2 - oy1
    orig_w = ox2 - ox1
    orig_area = orig_w * orig_h
    near_edge = (ox1 <= 8 or oy1 <= 8 or ox2 >= 632 or oy2 >= 504)
    
    cand_poly = r_rec.get("candidate_polygon") or r_rec.get("candidate_polygon_before_snap")
    
    iou = 0.0
    center_shift = 0.0
    area_ratio = 0.0
    exp_x = 0.0
    exp_y = 0.0
    inside_image = False
    cand_bbox = None
    
    if cand_poly:
        xs = [pt[0] for pt in cand_poly]
        ys = [pt[1] for pt in cand_poly]
        cand_x1, cand_x2 = min(xs), max(xs)
        cand_y1, cand_y2 = min(ys), max(ys)
        cand_bbox = [round(cand_x1), round(cand_y1), round(cand_x2), round(cand_y2)]
        inside_image = (0 <= cand_x1 and cand_x2 <= 640 and 0 <= cand_y1 and cand_y2 <= 512)
        
        inter_x1 = max(ox1, cand_x1)
        inter_y1 = max(oy1, cand_y1)
        inter_x2 = min(ox2, cand_x2)
        inter_y2 = min(oy2, cand_y2)
        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        cand_bbox_area = (cand_x2 - cand_x1) * (cand_y2 - cand_y1)
        union_area = orig_area + cand_bbox_area - inter_area
        iou = inter_area / union_area if union_area > 0 else 0.0
        
        ox_c = (ox1 + ox2) / 2.0
        oy_c = (oy1 + oy2) / 2.0
        cx_c = (cand_x1 + cand_x2) / 2.0
        cy_c = (cand_y1 + cand_y2) / 2.0
        center_shift_px = ((ox_c - cx_c) ** 2 + (oy_c - cy_c) ** 2) ** 0.5
        orig_diag = (orig_w ** 2 + orig_h ** 2) ** 0.5
        center_shift = center_shift_px / max(orig_diag, 1.0)
        area_ratio = cand_bbox_area / max(orig_area, 1.0)
        exp_x = max(0.0, cand_x2 - ox2) + max(0.0, ox1 - cand_x1)
        exp_y = max(0.0, cand_y2 - oy2) + max(0.0, oy1 - cand_y1)

    reason = t_rec.get("final_polygon_reason", "")
    geom_gate_accept = "missing"
    geom_gate_reason = ""
    if reason.startswith("geometry_gate_rejected_"):
        geom_gate_accept = False
        geom_gate_reason = reason.replace("geometry_gate_rejected_", "")
    elif t_rec.get("final_polygon_source") == "string_lattice_middle":
        geom_gate_accept = True
        
    prop_accept = "missing"
    prop_reject_reason = ""
    if t_rec.get("final_polygon_source") == "string_lattice_middle":
        prop_accept = True
    elif "string_lattice_propagation_rejected_" in reason:
        prop_accept = False
        prop_reject_reason = reason.replace("string_lattice_propagation_rejected_", "")
    elif geom_gate_accept is False:
        prop_accept = False
        prop_reject_reason = "gate_false"

    results.append({
        "raw_idx": idx,
        "bbox": bbox,
        "polygon": t_rec.get("polygon"),
        "final_polygon_source": t_rec.get("final_polygon_source"),
        "final_polygon_reason": reason,
        "geometry_gate_accept": geom_gate_accept,
        "geometry_gate_reason": geom_gate_reason,
        "string_lattice_propagation_accept": prop_accept,
        "string_lattice_propagation_reject_reason": prop_reject_reason,
        "string_id": t_rec.get("string_id"),
        "block_id": t_rec.get("block_id"),
        "candidate_polygon": cand_poly,
        "candidate_bbox": cand_bbox,
        "iou": iou,
        "center_shift": center_shift,
        "area_ratio": area_ratio,
        "expansion_x": exp_x,
        "expansion_y": exp_y,
        "inside_image": inside_image,
        "max_overlap": r_rec.get("max_overlap"),
        "near_edge": near_edge
    })

print(f"| raw_idx | string_id | block_id | Original Bbox | Candidate Bbox | IoU | Center Shift | Area Ratio | Exp X | Exp Y | Inside Img | Near Edge | Gate Reason |")
print(f"|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in results:
    cand_bbox_str = str(r['candidate_bbox']) if r['candidate_bbox'] else "None"
    print(f"| {r['raw_idx']} | {r['string_id']} | {r['block_id']} | {r['bbox']} | {cand_bbox_str} | {r['iou']:.4f} | {r['center_shift']:.4f} | {r['area_ratio']:.4f} | {r['expansion_x']:.1f} | {r['expansion_y']:.1f} | {r['inside_image']} | {r['near_edge']} | {r['geometry_gate_reason']} |")
