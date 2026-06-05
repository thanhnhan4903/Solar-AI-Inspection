import os
import json

TRACE_DIR = "data/results/debug_logs"
images = ["DJI_0029_R", "DJI_0843_R", "DJI_0845_R", "DJI_0087_R"]
target_idxs = [25, 40, 46, 47, 52, 62, 65, 67, 68, 77, 78, 86]

def analyze_image_relaxed(img_name):
    filepath = os.path.join(TRACE_DIR, f"{img_name}_final_polygon_trace.jsonl")
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}\n")
        return None

    panels = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line.strip())
            panels.append(rec)

    print("=" * 60)
    print(f"IMAGE: {img_name}")
    print("=" * 60)

    # Calculate final polygon source counts
    src_counts = {}
    for p in panels:
        src = p.get("final_polygon_source", "yolo_original")
        src_counts[src] = src_counts.get(src, 0) + 1

    print("Source counts:")
    for src, count in sorted(src_counts.items()):
        print(f"  - {src}: {count}")

    # Inspect target panels for DJI_0029_R
    if img_name == "DJI_0029_R":
        print("\nTarget panel audit (12 raw_idx):")
        print(f"| raw_idx | Bbox | Source | Reason | Relaxed Accept | Relaxed Reject Reason | IoU | Center Shift | Area Ratio |")
        print(f"|---|---|---|---|---|---|---|---|---|")
        for idx in target_idxs:
            p = next((x for x in panels if x.get("raw_idx") == idx), None)
            if p:
                print(f"| {idx} | {p.get('bbox')} | {p.get('final_polygon_source')} | {p.get('final_polygon_reason')} | {p.get('geometry_gate_relaxed_accept')} | {p.get('geometry_gate_relaxed_reject_reason')} | {p.get('geometry_gate_relaxed_iou', 0.0):.4f} | {p.get('geometry_gate_relaxed_center_shift', 0.0):.4f} | {p.get('geometry_gate_relaxed_area_ratio', 0.0):.4f} |")
            else:
                print(f"| {idx} | NOT FOUND | | | | | | | |")

    # Relaxed gate summary stats
    relaxed_accept_cnt = sum(1 for p in panels if p.get("geometry_gate_relaxed_accept") is True)
    relaxed_reject_cnt = sum(1 for p in panels if p.get("geometry_gate_relaxed_accept") is False)
    relaxed_missing = sum(1 for p in panels if p.get("geometry_gate_relaxed_accept") is None)
    
    print(f"\nRelaxed Gate: Accept={relaxed_accept_cnt} | Reject={relaxed_reject_cnt} | Missing={relaxed_missing}")
    if relaxed_reject_cnt > 0:
        reasons = {}
        for p in panels:
            if p.get("geometry_gate_relaxed_accept") is False:
                r = p.get("geometry_gate_relaxed_reject_reason")
                reasons[r] = reasons.get(r, 0) + 1
        print(f"Relaxed Reject Reasons: {reasons}")
    print()
    return src_counts

if __name__ == "__main__":
    results = {}
    for img in images:
        results[img] = analyze_image_relaxed(img)
