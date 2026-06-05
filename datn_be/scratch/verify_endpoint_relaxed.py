import json
import os
import collections

test_images = ["DJI_0843_R", "DJI_0845_R", "DJI_0087_R", "DJI_0029_R"]

for img in test_images:
    trace_path = f"data/results/debug_logs/{img}_final_polygon_trace.jsonl"
    if not os.path.exists(trace_path):
        print(f"Skipping {img}: trace file not found.")
        continue

    print(f"\n=======================================================")
    print(f"DIAGNOSTICS FOR IMAGE: {img}")
    print(f"=======================================================")

    endpoint_total = 0
    endpoint_failed_neighbor_overlap_count = 0
    endpoint_relaxed_accept_count = 0
    endpoint_relaxed_reject_count = 0
    endpoint_relaxed_reject_reasons = collections.Counter()
    source_counts = collections.Counter()
    
    accepted_examples = []
    rejected_examples = []

    with open(trace_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = json.loads(line)
            
            # Count final polygon sources
            source = p.get("final_polygon_source", "yolo_original")
            source_counts[source] += 1
            
            # Identify endpoint panels in trace log
            # We can check is_endpoint_panel from the trace, or from the logs.
            # In panel_processor.py: p["endpoint_relaxed_accept"] is set if accepted or rejected.
            # Let's count based on the flags we added.
            relaxed_accept = p.get("endpoint_relaxed_accept")
            
            # Original endpoint fallback or relaxed accept means it is an endpoint checked by us
            is_endpoint_checked = (
                relaxed_accept is not None or 
                source == "endpoint_original_fallback" or
                p.get("final_polygon_reason", "").startswith("string_endpoint_failed_")
            )
            
            if is_endpoint_checked:
                endpoint_total += 1
                
                # Was it failed due to neighbor overlap?
                reason = p.get("final_polygon_reason", "")
                is_overlap_fail = (
                    "neighbor_overlap" in reason or 
                    "neighbor_overlap" in str(p.get("endpoint_relaxed_reject_reason", "")) or
                    "neighbor_overlap" in str(p.get("string_reason", ""))
                )
                if is_overlap_fail:
                    endpoint_failed_neighbor_overlap_count += 1
                
                if relaxed_accept is True:
                    endpoint_relaxed_accept_count += 1
                    if len(accepted_examples) < 10:
                        accepted_examples.append({
                            "raw_idx": p.get("raw_idx"),
                            "bbox": p.get("bbox"),
                            "polygon": p.get("polygon"),
                            "source": source,
                            "reason": p.get("final_polygon_reason"),
                            "overlap": p.get("endpoint_relaxed_overlap"),
                            "IoU": p.get("endpoint_relaxed_iou"),
                            "center_shift": p.get("endpoint_relaxed_center_shift"),
                            "near_edge": (p.get("bbox", [0,0,0,0])[0] <= 8 or p.get("bbox", [0,0,0,0])[1] <= 8 or p.get("bbox", [0,0,0,0])[2] >= 632 or p.get("bbox", [0,0,0,0])[3] >= 504)
                        })
                elif relaxed_accept is False:
                    endpoint_relaxed_reject_count += 1
                    reject_code = p.get("endpoint_relaxed_reject_reason", "unknown")
                    endpoint_relaxed_reject_reasons[reject_code] += 1
                    if len(rejected_examples) < 10:
                        rejected_examples.append({
                            "raw_idx": p.get("raw_idx"),
                            "bbox": p.get("bbox"),
                            "polygon": p.get("polygon"),
                            "source": source,
                            "reason": p.get("final_polygon_reason"),
                            "overlap": p.get("endpoint_relaxed_overlap"),
                            "IoU": p.get("endpoint_relaxed_iou"),
                            "center_shift": p.get("endpoint_relaxed_center_shift"),
                            "near_edge": (p.get("bbox", [0,0,0,0])[0] <= 8 or p.get("bbox", [0,0,0,0])[1] <= 8 or p.get("bbox", [0,0,0,0])[2] >= 632 or p.get("bbox", [0,0,0,0])[3] >= 504),
                            "reject_reason": reject_code
                        })

    print(f"1. endpoint_total: {endpoint_total}")
    print(f"2. endpoint_failed_neighbor_overlap_count: {endpoint_failed_neighbor_overlap_count}")
    print(f"3. endpoint_relaxed_accept_count: {endpoint_relaxed_accept_count}")
    print(f"4. endpoint_relaxed_reject_count: {endpoint_relaxed_reject_count}")
    print(f"5. endpoint_relaxed_reject_reason counts: {dict(endpoint_relaxed_reject_reasons)}")
    print(f"6. final_polygon_source counts: {dict(source_counts)}")
    
    print(f"\n7. FIRST 10 ACCEPTED ENDPOINT EXAMPLES:")
    for i, ex in enumerate(accepted_examples):
        print(f"   [{i+1}] raw_idx={ex['raw_idx']}")
        print(f"       - bbox: {ex['bbox']}")
        print(f"       - polygon: {ex['polygon']}")
        print(f"       - source: {ex['source']}")
        print(f"       - reason: {ex['reason']}")
        print(f"       - overlap: {ex['overlap']:.4f}")
        print(f"       - IoU: {ex['IoU']:.4f}")
        print(f"       - center_shift: {ex['center_shift']:.4f}")
        print(f"       - near_edge: {ex['near_edge']}")
        
    print(f"\n8. FIRST 10 REJECTED ENDPOINT EXAMPLES:")
    for i, ex in enumerate(rejected_examples):
        print(f"   [{i+1}] raw_idx={ex['raw_idx']}")
        print(f"       - bbox: {ex['bbox']}")
        print(f"       - polygon: {ex['polygon']}")
        print(f"       - source: {ex['source']}")
        print(f"       - reason: {ex['reason']}")
        print(f"       - reject_reason: {ex['reject_reason']}")
        print(f"       - near_edge: {ex['near_edge']}")
