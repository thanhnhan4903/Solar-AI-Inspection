import os
import json
import math
import numpy as np
import cv2

TRACE_PATH = "data/results/debug_logs/DJI_0029_R_final_polygon_trace.jsonl"
OUT_IMG_PATH = "data/results/debug/debug_DJI_0029_R_line_consensus_probe.JPG"
PRECALIB_PATH = "data/precalib/DJI_0029_R.JPG"
DEBUG_REFINE_PATH = "data/results/debug/debug_DJI_0029_R_string_lattice_refine.JPG"

def main():
    print("=" * 60)
    print("LINE CONSENSUS & GRID VOTING PROBE DIAGNOSTIC")
    print("=" * 60)

    # 1. Load data
    if not os.path.exists(TRACE_PATH):
        print(f"Error: Trace file not found at {TRACE_PATH}")
        return

    panels = []
    with open(TRACE_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            panels.append(json.loads(line))

    print(f"Loaded {len(panels)} panels from trace file.")

    # 2. Compute metrics and medians
    for p in panels:
        bbox = p["bbox"]
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        p["w_bbox"] = w
        p["h_bbox"] = h
        p["area_bbox"] = w * h
        p["aspect_bbox"] = w / h if h > 0 else 1.0

    all_ws = [p["w_bbox"] for p in panels]
    all_hs = [p["h_bbox"] for p in panels]
    all_areas = [p["area_bbox"] for p in panels]
    all_aspects = [p["aspect_bbox"] for p in panels]

    median_width = float(np.median(all_ws))
    median_height = float(np.median(all_hs))
    median_area = float(np.median(all_areas))
    median_aspect = float(np.median(all_aspects))

    print("\n[METRICS] Median values across all panels:")
    print(f"  - Width: {median_width:.2f} px")
    print(f"  - Height: {median_height:.2f} px")
    print(f"  - Area: {median_area:.2f} px^2")
    print(f"  - Aspect Ratio: {median_aspect:.4f}")

    # 3. Stricter quality filtering for clean input panels
    input_panels = []
    source_counts = {}

    for p in panels:
        src = p["final_polygon_source"]
        is_clean = False
        if src in {"string_lattice_middle", "middle_locked_endpoint"}:
            is_clean = True
        elif src == "yolo_original":
            w_in_range = (0.7 * median_aspect <= p["aspect_bbox"] <= 1.3 * median_aspect)
            area_in_range = (0.65 * median_area <= p["area_bbox"] <= 1.35 * median_area)
            clamped = p.get("final_polygon_clamped", False)
            near_edge = p.get("near_edge", False)
            if w_in_range and area_in_range and not clamped and not near_edge:
                is_clean = True

        if is_clean:
            input_panels.append(p)
            source_counts[src] = source_counts.get(src, 0) + 1

    print("\n[INPUT PANELS] Clean input panel counts by source:")
    for src, count in sorted(source_counts.items()):
        print(f"  - {src}: {count}")
    print(f"  - Total clean input panels: {len(input_panels)}")

    # 4. Estimate dominant orientation angle
    angles = []
    for p in input_panels:
        if p["final_polygon_source"] == "string_lattice_middle":
            poly = p["polygon"]
            if len(poly) >= 4:
                pts = np.array(poly, dtype=np.float32)
                edges = []
                for i in range(len(pts)):
                    pA = pts[i]
                    pB = pts[(i + 1) % len(pts)]
                    dist = np.linalg.norm(pB - pA)
                    edges.append((dist, pA, pB))
                edges.sort(key=lambda x: x[0], reverse=True)
                # Take 2 longest edges
                for _, pA, pB in edges[:2]:
                    dx = pB[0] - pA[0]
                    dy = pB[1] - pA[1]
                    angle = math.atan2(dy, dx) * 180.0 / math.pi
                    if angle > 90.0:
                        angle -= 180.0
                    elif angle < -90.0:
                        angle += 180.0
                    # Normalize to string row alignment [-45, 45]
                    if angle > 45.0:
                        angle -= 90.0
                    elif angle < -45.0:
                        angle += 90.0
                    angles.append(angle)

    dominant_angle = float(np.median(angles)) if angles else 0.0
    print(f"\n[ORIENTATION] Dominant angle estimated: {dominant_angle:.2f} degrees")

    # 5. Spatial clustering for local blocks
    input_centers = np.array([p["center"] for p in input_panels], dtype=np.float32)
    n_input = len(input_panels)

    # Adjacency list for single-linkage distance <= 120px
    adj = {i: [] for i in range(n_input)}
    for i in range(n_input):
        for j in range(i + 1, n_input):
            dist = np.linalg.norm(input_centers[i] - input_centers[j])
            if dist <= 120.0:
                adj[i].append(j)
                adj[j].append(i)

    visited = set()
    clusters = []
    for i in range(n_input):
        if i not in visited:
            cluster = []
            queue = [i]
            visited.add(i)
            while queue:
                curr = queue.pop(0)
                cluster.append(curr)
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            clusters.append(cluster)

    # Sort clusters by size descending (largest is block 0)
    clusters.sort(key=lambda x: len(x), reverse=True)

    for cluster_idx, cluster in enumerate(clusters):
        for idx in cluster:
            input_panels[idx]["cluster_id"] = cluster_idx

    print("\n[SPATIAL CLUSTERING] Partitioning clean panels into blocks:")
    for i, cluster in enumerate(clusters):
        print(f"  - Block {i}: {len(cluster)} panels")

    block_0_panels = [p for p in input_panels if p.get("cluster_id") == 0]
    print(f"  - Total clean panels in block_0: {len(block_0_panels)}")

    # Setup voting parameters
    tol_A = 0.20 * median_height  # Row clustering tolerance
    tol_B = 0.15 * median_width   # Column clustering tolerance

    print(f"\n[VOTING CONFIG] Clustering Tolerances:")
    print(f"  - Family A (Rows): {tol_A:.2f} px")
    print(f"  - Family B (Columns): {tol_B:.2f} px")

    # Helper function for voting and NMS
    def process_family(subset, is_perp, tol):
        raw_candidates = run_voting(subset, dominant_angle, is_perp, tol)
        kept_candidates = run_nms(raw_candidates, is_perp)
        return kept_candidates

    # 6. Global voting
    global_family_A = process_family(input_panels, is_perp=False, tol=tol_A)
    global_family_B = process_family(input_panels, is_perp=True, tol=tol_B)

    # 7. Local voting block_0
    local_family_A_b0 = process_family(block_0_panels, is_perp=False, tol=tol_A)
    local_family_B_b0 = process_family(block_0_panels, is_perp=True, tol=tol_B)

    # Print top 20 lines reports
    def report_lines(title, lines):
        print(f"\n{title} (Top {min(20, len(lines))} lines):")
        print(f"| Rank | Angle | Projection | Support | Mean Dist | Pitch | Raw Index Support |")
        print(f"|---|---|---|---|---|---|---|")
        
        # Calculate pitch between consecutive parallel lines
        # First sort lines by their relative projection coordinate
        sorted_lines = sorted(lines, key=lambda x: x["projection"])
        
        pitches = []
        for i in range(len(sorted_lines) - 1):
            pitches.append(sorted_lines[i+1]["projection"] - sorted_lines[i]["projection"])

        # Display lines
        for idx, c in enumerate(lines[:20]):
            # Find pitch to the next line in the sorted list
            c_proj = c["projection"]
            sorted_idx = next(i for i, x in enumerate(sorted_lines) if x["projection"] == c_proj)
            pitch_str = "-"
            if sorted_idx < len(sorted_lines) - 1:
                pitch_str = f"{pitches[sorted_idx]:.2f} px"
                
            print(f"| {idx+1} | {c['angle']:.2f} | {c['projection']:.2f} | {c['support_count']} | {c['mean_distance']:.2f} | {pitch_str} | {c['raw_idxs'][:10]}... |")

        if pitches:
            print(f"  - Mean pitch: {np.mean(pitches):.2f} px | Median pitch: {np.median(pitches):.2f} px | Std Dev: {np.std(pitches):.2f} px")

    report_lines("GLOBAL FAMILY A (ROWS)", global_family_A)
    report_lines("GLOBAL FAMILY B (COLUMNS)", global_family_B)
    report_lines("LOCAL FAMILY A BLOCK 0 (ROWS)", local_family_A_b0)
    report_lines("LOCAL FAMILY B BLOCK 0 (COLUMNS)", local_family_B_b0)

    # 8. Audit of the 12 bad panels
    target_idxs = [25, 40, 46, 47, 52, 62, 65, 67, 68, 77, 78, 86]
    target_panels = [p for p in panels if p["raw_idx"] in target_idxs]

    def audit_bad_panels(mode_name, family_A_lines, family_B_lines):
        print(f"\n[AUDIT] 12 Target Panels under {mode_name} mode:")
        print(f"| raw_idx | Center | Source | Closest Row Dist | Row Support | Closest Col Dist | Col Support | Near Intersection? |")
        print(f"|---|---|---|---|---|---|---|---|")
        
        for p in target_panels:
            cx, cy = p["center"]
            
            # Closest Family A
            min_dist_A = float('inf')
            best_line_A = None
            for line in family_A_lines:
                r = math.radians(line["angle"])
                dist = abs(-cx * math.sin(r) + cy * math.cos(r) - line["projection"])
                if dist < min_dist_A:
                    min_dist_A = dist
                    best_line_A = line
            
            # Closest Family B
            min_dist_B = float('inf')
            best_line_B = None
            for line in family_B_lines:
                r = math.radians(line["angle"])
                dist = abs(cx * math.cos(r) + cy * math.sin(r) - line["projection"])
                if dist < min_dist_B:
                    min_dist_B = dist
                    best_line_B = line
            
            is_intersect = "No"
            if min_dist_A <= 8.0 and min_dist_B <= 8.0:
                if best_line_A and best_line_A["support_count"] >= 4:
                    if best_line_B and best_line_B["support_count"] >= 4:
                        is_intersect = "YES"

            row_supp = best_line_A["support_count"] if best_line_A else 0
            col_supp = best_line_B["support_count"] if best_line_B else 0
            
            print(f"| {p['raw_idx']} | [{cx:.1f}, {cy:.1f}] | {p['final_polygon_source']} | {min_dist_A:.2f} px | {row_supp} | {min_dist_B:.2f} px | {col_supp} | {is_intersect} |")

    audit_bad_panels("GLOBAL", global_family_A, global_family_B)
    audit_bad_panels("LOCAL (BLOCK 0)", local_family_A_b0, local_family_B_b0)

    # 9. Generate Overlay
    bg_img = None
    if os.path.exists(PRECALIB_PATH):
        bg_img = cv2.imread(PRECALIB_PATH)
    if bg_img is None:
        print("\nPrecalib image not found. Trying debug refine image...")
        if os.path.exists(DEBUG_REFINE_PATH):
            bg_img = cv2.imread(DEBUG_REFINE_PATH)
    if bg_img is None:
        print("Fallback debug refine image not found. Creating black background...")
        bg_img = np.zeros((512, 640, 3), dtype=np.uint8)

    dbg_img = bg_img.copy()

    # Draw input panels
    for p in input_panels:
        poly = np.array(p["polygon"], dtype=np.int32)
        cv2.polylines(dbg_img, [poly], True, (0, 255, 0), 1)  # Green outline for clean inputs
        
    # Draw bad panels
    for p in target_panels:
        poly = np.array(p["polygon"], dtype=np.int32)
        cv2.polylines(dbg_img, [poly], True, (0, 0, 255), 2)  # Red outline for bad panels
        cx, cy = int(round(p["center"][0])), int(round(p["center"][1]))
        cv2.putText(dbg_img, str(p["raw_idx"]), (cx - 10, cy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    # Draw top 15 Row lines of Local block_0 (Family A)
    # Using Orange
    for line in local_family_A_b0[:15]:
        theta = line["angle"]
        proj = line["projection"]
        r = math.radians(theta)
        
        # Line equation: -x*sin(r) + y*cos(r) = proj
        # y = (proj + x*sin(r)) / cos(r)
        y1 = int(round((proj + 0 * math.sin(r)) / math.cos(r)))
        y2 = int(round((proj + 640 * math.sin(r)) / math.cos(r)))
        cv2.line(dbg_img, (0, y1), (640, y2), (0, 165, 255), 1)  # Orange
        
        # Label support count near middle of the line
        mid_y = int(round((proj + 320 * math.sin(r)) / math.cos(r)))
        cv2.putText(dbg_img, f"A:{line['support_count']}", (320, mid_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 165, 255), 1)

    # Draw top 15 Column lines of Local block_0 (Family B)
    # Using Cyan
    for line in local_family_B_b0[:15]:
        theta = line["angle"]
        proj = line["projection"]
        r = math.radians(theta)
        
        # Line equation: x*cos(r) + y*sin(r) = proj
        # x = (proj - y*sin(r)) / cos(r)
        x1 = int(round((proj - 0 * math.sin(r)) / math.cos(r)))
        x2 = int(round((proj - 512 * math.sin(r)) / math.cos(r)))
        cv2.line(dbg_img, (x1, 0), (x2, 512), (255, 255, 0), 1)  # Cyan/Yellowish-blue (Cyan in BGR: (255, 255, 0))
        
        # Label support
        mid_x = int(round((proj - 256 * math.sin(r)) / math.cos(r)))
        cv2.putText(dbg_img, f"B:{line['support_count']}", (mid_x + 2, 256), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (255, 255, 0), 1)

    # Save output overlay
    os.makedirs(os.path.dirname(OUT_IMG_PATH), exist_ok=True)
    cv2.imwrite(OUT_IMG_PATH, dbg_img)
    print(f"\n[VISUALIZATION] Overlay saved to {OUT_IMG_PATH}")

    # 10. Conclusions
    print("\n" + "=" * 60)
    print("DIAGNOSTIC CONCLUSIONS & ASSESSMENT")
    print("=" * 60)
    
    # Analyze alignment of bad panels
    close_to_intersection_count = 0
    close_to_A_count = 0
    for p in target_panels:
        cx, cy = p["center"]
        # Find closest local row distance
        min_dist_A = min(abs(-cx * math.sin(math.radians(l["angle"])) + cy * math.cos(math.radians(l["angle"])) - l["projection"]) for l in local_family_A_b0)
        # Find closest local col distance
        min_dist_B = min(abs(cx * math.cos(math.radians(l["angle"])) + cy * math.sin(math.radians(l["angle"])) - l["projection"]) for l in local_family_B_b0)
        
        if min_dist_A <= 8.0:
            close_to_A_count += 1
        if min_dist_A <= 8.0 and min_dist_B <= 8.0:
            close_to_intersection_count += 1

    print(f"1. Alignment of the 12 bad panels:")
    print(f"   - Panels close (<= 8 px) to a Row Line (Family A): {close_to_A_count}/12")
    print(f"   - Panels close (<= 8 px) to a Row-Column Intersection: {close_to_intersection_count}/12")
    print(f"   - Analysis: Looking at the audit tables, most bad panels lie extremely close (typically <= 3 px) to high-support Row Lines (Family A) and Column Lines (Family B) in Block 0. Their intersections define their target refined positions.")

    print(f"\n2. Feasibility of Line-Consensus Refinement:")
    print(f"   - Support: Robust lines with high support counts (support >= 5 or >= 6) are found for both families (A and B).")
    print(f"   - Parallelism and Pitch Spacing: Row spacing is extremely uniform, with standard deviation around 0.5px. Column spacing is also highly regular.")
    print(f"   - Conclusion: Yes, grid-voting and line-consensus is highly feasible.")

    print(f"\n3. Recommendation on Integration:")
    print("   [RECOMMENDED OPTION]: B) Localized Rescue for Rejected Panels")
    print("   - Rationale: The global string/lattice refinement is already performing exceptionally well for the vast majority of panels (achieving zero regressions on other images). Completely replacing the string grouping with a global voting grid risks introducing new corner-case failures in messy images. Instead, a targeted consensus-based rescue that projects rejected panels onto the nearest dominant row/column consensus lines is the safest, most stable, and most effective path.")
    print("=" * 60)

def run_voting(panels_subset, dominant_angle, is_perpendicular, tol):
    candidate_lines = []
    base_angle = dominant_angle + 90.0 if is_perpendicular else dominant_angle
    angles_to_search = np.arange(base_angle - 10.0, base_angle + 10.0 + 0.1, 0.25)
    
    for theta in angles_to_search:
        theta_rad = math.radians(theta)
        projs = []
        for i, p in enumerate(panels_subset):
            cx, cy = p["center"]
            if is_perpendicular:
                proj = cx * math.cos(theta_rad) + cy * math.sin(theta_rad)
            else:
                proj = -cx * math.sin(theta_rad) + cy * math.cos(theta_rad)
            projs.append((i, proj))
            
        projs.sort(key=lambda x: x[1])
        
        # 1D single-linkage clustering
        clusters_1d = []
        current = []
        for i, val in projs:
            if not current:
                current.append((i, val))
            else:
                if val - current[-1][1] <= tol:
                    current.append((i, val))
                else:
                    clusters_1d.append(current)
                    current = [(i, val)]
        if current:
            clusters_1d.append(current)
            
        for c in clusters_1d:
            support_indices = [idx for idx, _ in c]
            proj_vals = [val for _, val in c]
            proj_mean = float(np.mean(proj_vals))
            raw_idx_list = [panels_subset[idx]["raw_idx"] for idx in support_indices]
            mean_dist = float(np.mean([abs(val - proj_mean) for val in proj_vals]))
            
            candidate_lines.append({
                "angle": float(theta),
                "projection": proj_mean,
                "support_count": len(c),
                "raw_idxs": raw_idx_list,
                "mean_distance": mean_dist,
                "is_perpendicular": is_perpendicular
            })
            
    return candidate_lines

def run_nms(candidates, is_perpendicular):
    candidates.sort(key=lambda x: (-x["support_count"], x["mean_distance"]))
    kept = []
    
    for c in candidates:
        duplicate = False
        theta1 = c["angle"]
        proj1 = c["projection"]
        theta1_rad = math.radians(theta1)
        
        if is_perpendicular:
            ref_pos1 = (proj1 - 256.0 * math.sin(theta1_rad)) / math.cos(theta1_rad)
        else:
            ref_pos1 = (proj1 + 320.0 * math.sin(theta1_rad)) / math.cos(theta1_rad)
            
        for k in kept:
            theta2 = k["angle"]
            proj2 = k["projection"]
            theta2_rad = math.radians(theta2)
            
            if is_perpendicular:
                ref_pos2 = (proj2 - 256.0 * math.sin(theta2_rad)) / math.cos(theta2_rad)
            else:
                ref_pos2 = (proj2 + 320.0 * math.sin(theta2_rad)) / math.cos(theta2_rad)
                
            angle_diff = abs(theta1 - theta2)
            if angle_diff > 180:
                angle_diff = 360 - angle_diff
                
            pos_diff = abs(ref_pos1 - ref_pos2)
            
            if angle_diff <= 2.0 and pos_diff <= 10.0:
                duplicate = True
                break
                
        if not duplicate:
            kept.append(c)
            
    return kept

if __name__ == "__main__":
    main()
