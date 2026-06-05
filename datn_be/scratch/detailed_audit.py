import os
import json
import glob

TRACE_DIR = "data/results/debug_logs"
images = ["DJI_0029_R", "DJI_0087_R", "DJI_0843_R", "DJI_0845_R"]

def analyze_image(img_name):
    filepath = os.path.join(TRACE_DIR, f"{img_name}_final_polygon_trace.jsonl")
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return

    panels = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line.strip())
            
            # Derive geometry_gate_accept and geometry_gate_reason
            reason = rec.get("final_polygon_reason", "")
            source = rec.get("final_polygon_source", "")
            
            gga = "missing"
            ggr = ""
            slpa = "missing"
            slprr = ""
            
            # Reconstruct geometry gate values
            if isinstance(reason, str) and reason.startswith("geometry_gate_rejected_"):
                gga = False
                ggr = reason.replace("geometry_gate_rejected_", "")
            elif source == "string_lattice_middle":
                gga = True
                ggr = ""
                
            # Reconstruct propagation values
            if source == "string_lattice_middle":
                slpa = True
                slprr = ""
            elif isinstance(reason, str) and reason.startswith("string_lattice_propagation_rejected_"):
                slpa = False
                slprr = reason.replace("string_lattice_propagation_rejected_", "")
            elif gga is False:
                slpa = False
                slprr = "gate_false"
                
            # Reconstruct near_edge
            bbox = rec.get("bbox", [0, 0, 0, 0])
            x1, y1, x2, y2 = bbox
            near_edge = (x1 <= 8 or y1 <= 8 or x2 >= 632 or y2 >= 504)
            
            panels.append({
                "image_name": img_name,
                "raw_idx": rec.get("raw_idx"),
                "bbox": bbox,
                "polygon": rec.get("polygon"),
                "final_polygon_source": source,
                "final_polygon_stage": rec.get("final_polygon_stage"),
                "final_polygon_reason": reason,
                "geometry_gate_accept": gga,
                "geometry_gate_reason": ggr,
                "string_lattice_propagation_accept": slpa,
                "string_lattice_propagation_reject_reason": slprr,
                "final_polygon_clamped": rec.get("final_polygon_clamped", False),
                "final_polygon_clamp_delta_max": rec.get("final_polygon_clamp_delta_max", 0.0),
                "near_edge": near_edge,
                "string_id": rec.get("string_id"),
                "block_id": rec.get("block_id")
            })

    # Stats
    source_counts = {}
    reason_counts = {}
    gga_counts = {"True": 0, "False": 0, "missing": 0}
    slpa_counts = {"True": 0, "False": 0, "missing": 0}
    near_edge_count = 0
    clamp_count = 0
    max_clamp_delta = 0.0

    for p in panels:
        source = p["final_polygon_source"]
        source_counts[source] = source_counts.get(source, 0) + 1
        
        reason = p["final_polygon_reason"]
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        
        gga_counts[str(p["geometry_gate_accept"])] += 1
        slpa_counts[str(p["string_lattice_propagation_accept"])] += 1
        
        if p["near_edge"]:
            near_edge_count += 1
            
        if p["final_polygon_clamped"]:
            clamp_count += 1
            max_clamp_delta = max(max_clamp_delta, p["final_polygon_clamp_delta_max"])

    print(f"\n======================================================================")
    print(f"IMAGE: {img_name}")
    print(f"======================================================================")
    print("1. final_polygon_source counts:")
    for k, v in sorted(source_counts.items()):
        print(f"   - {k}: {v}")
        
    print("\n2. final_polygon_reason counts:")
    for k, v in sorted(reason_counts.items()):
        print(f"   - {k}: {v}")
        
    print(f"\n3. geometry_gate_accept counts: True={gga_counts['True']} / False={gga_counts['False']} / missing={gga_counts['missing']}")
    print(f"4. string_lattice_propagation_accept counts: True={slpa_counts['True']} / False={slpa_counts['False']} / missing={slpa_counts['missing']}")
    print(f"5. number of near_edge panels: {near_edge_count}")
    print(f"6. number of clamped panels: {clamp_count} (max_delta: {max_clamp_delta:.1f} px)")

    # Sort to find 30 most suspicious panels
    # Priority:
    # A. final_polygon_source == "yolo_original"
    # B. final_polygon_source contains "endpoint"
    # C. near_edge == True
    # D. geometry_gate_accept == False
    # E. string_lattice_propagation_accept == False
    # F. final_polygon_clamped == True
    def get_sort_key(p):
        return (
            p["final_polygon_source"] == "yolo_original",
            "endpoint" in str(p["final_polygon_source"]),
            p["near_edge"] is True,
            p["geometry_gate_accept"] is False,
            p["string_lattice_propagation_accept"] is False,
            p["final_polygon_clamped"] is True
        )

    suspicious = sorted(panels, key=get_sort_key, reverse=True)
    
    print(f"\n7. TOP 30 SUSPICIOUS PANELS:")
    for i, p in enumerate(suspicious[:30]):
        print(f"\n   [{i+1}] raw_idx={p['raw_idx']}")
        print(f"       - bbox: {p['bbox']}")
        print(f"       - polygon: {p['polygon']}")
        print(f"       - final_polygon_source: {p['final_polygon_source']}")
        print(f"       - final_polygon_stage: {p['final_polygon_stage']}")
        print(f"       - final_polygon_reason: {p['final_polygon_reason']}")
        print(f"       - geometry_gate_accept: {p['geometry_gate_accept']}")
        print(f"       - geometry_gate_reason: {p['geometry_gate_reason']}")
        print(f"       - string_lattice_propagation_accept: {p['string_lattice_propagation_accept']}")
        print(f"       - string_lattice_propagation_reject_reason: {p['string_lattice_propagation_reject_reason']}")
        print(f"       - final_polygon_clamped: {p['final_polygon_clamped']}")
        print(f"       - final_polygon_clamp_delta_max: {p['final_polygon_clamp_delta_max']}")
        print(f"       - near_edge: {p['near_edge']}")
        print(f"       - string_id: {p['string_id']}")
        print(f"       - block_id: {p['block_id']}")

for img in images:
    analyze_image(img)
