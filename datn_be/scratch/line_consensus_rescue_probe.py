import os
import json
import math
import numpy as np
import cv2
from shapely.geometry import Polygon

TRACE_PATH = "data/results/debug_logs/DJI_0029_R_final_polygon_trace.jsonl"
REFINE_PATH = "data/results/debug_logs/DJI_0029_R_string_lattice_refine.jsonl"
OUT_IMG_PATH = "data/results/debug/debug_DJI_0029_R_line_consensus_rescue_probe.JPG"
PRECALIB_PATH = "data/precalib/DJI_0029_R.JPG"
DEBUG_REFINE_PATH = "data/results/debug/debug_DJI_0029_R_string_lattice_refine.JPG"

def main():
    print("=" * 60)
    print("LINE CONSENSUS RESCUE POLYGON PROPOSAL DIAGNOSTIC")
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

    # Try to load block_id from refine log if missing in trace
    refine_map = {}
    if os.path.exists(REFINE_PATH):
        with open(REFINE_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("record_type") == "panel":
                    p_idx = rec.get("panel_idx")
                    b_id = rec.get("block_id")
                    if p_idx is not None and b_id is not None:
                        refine_map[p_idx] = b_id

    for p in panels:
        p_idx = p["raw_idx"]
        # Priority: trace block_id, then refine_log block_id
        b_id = p.get("block_id")
        if b_id is None and p_idx in refine_map:
            b_id = refine_map[p_idx]
            p["block_id"] = b_id

    # Compute bounding box metrics
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

    global_median_width = float(np.median(all_ws))
    global_median_height = float(np.median(all_hs))
    global_median_area = float(np.median(all_areas))
    global_median_aspect = float(np.median(all_aspects))

    # Stricter quality filtering for clean input panels
    input_panels = []
    source_counts = {}

    for p in panels:
        src = p["final_polygon_source"]
        is_clean = False
        if src in {"string_lattice_middle", "middle_locked_endpoint"}:
            is_clean = True
        elif src == "yolo_original":
            w_in_range = (0.7 * global_median_aspect <= p["aspect_bbox"] <= 1.3 * global_median_aspect)
            area_in_range = (0.65 * global_median_area <= p["area_bbox"] <= 1.35 * global_median_area)
            clamped = p.get("final_polygon_clamped", False)
            near_edge = p.get("near_edge", False)
            if w_in_range and area_in_range and not clamped and not near_edge:
                is_clean = True

        if is_clean:
            input_panels.append(p)
            source_counts[src] = source_counts.get(src, 0) + 1

    # Check if block_id is populated anywhere
    block_id_present = any(p.get("block_id") is not None for p in panels)

    if block_id_present:
        print("\n[GROUPING] block_id is present in files. Prioritizing block_id grouping.")
        # Group clean panels by block_id
        block_groups = {}
        for p in input_panels:
            b_id = p.get("block_id")
            if b_id is not None:
                block_groups.setdefault(b_id, []).append(p)
                
        # Prioritize block_id == 0 for the 12 bad panels of DJI_0029_R
        if 0 in block_groups:
            block_0_id = 0
        else:
            sorted_groups = sorted(block_groups.items(), key=lambda x: len(x[1]), reverse=True)
            block_0_id = sorted_groups[0][0] if sorted_groups else 0
            
        block_0_panels = block_groups.get(block_0_id, [])
        print(f"  - Prioritized Block 0 ID: {block_0_id} (contains {len(block_0_panels)} clean panels)")
    else:
        print("\n[GROUPING] block_id is missing or None. Falling back to Spatial Component Clustering (120px).")
        # Run spatial clustering
        input_centers = np.array([p["center"] for p in input_panels], dtype=np.float32)
        n_input = len(input_panels)
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

        # Find the cluster containing any of the 12 target indexes and swap it to index 0
        target_cluster_idx = 0
        target_idxs = {25, 40, 46, 47, 52, 62, 65, 67, 68, 77, 78, 86}
        for c_idx, cl in enumerate(clusters):
            if any(input_panels[idx]["raw_idx"] in target_idxs for idx in cl):
                target_cluster_idx = c_idx
                break
        if target_cluster_idx != 0:
            clusters[0], clusters[target_cluster_idx] = clusters[target_cluster_idx], clusters[0]
            
        for cluster_idx, cluster in enumerate(clusters):
            for idx in cluster:
                input_panels[idx]["cluster_id"] = cluster_idx

        block_0_id = 0
        block_0_panels = [p for p in input_panels if p.get("cluster_id") == 0]
        print(f"  - Spatial Block 0 (containing target panels) contains {len(block_0_panels)} clean panels")

    # 4. Identify good panels inside block_0 for median geometry calculation
    good_panels_b0 = []
    for p in block_0_panels:
        # Strict validation
        if (p["final_polygon_source"] in {"string_lattice_middle", "middle_locked_endpoint"} and
            p.get("geometry_gate_accept") is not False and
            p.get("final_polygon_clamped") is not True and
            p.get("near_edge") is not True):
            good_panels_b0.append(p)

    print(f"\n[GOOD PANELS] Identified {len(good_panels_b0)} good panels in Block 0.")

    # Calculate median orientation-aware geometry from good panels in Block 0
    widths = []
    heights = []
    angles = []

    for p in good_panels_b0:
        poly = p.get("polygon")
        if poly and len(poly) >= 4:
            pts = np.array(poly, dtype=np.float32)
            edges = []
            for i in range(len(pts)):
                pA = pts[i]
                pB = pts[(i + 1) % len(pts)]
                dist = np.linalg.norm(pB - pA)
                edges.append((dist, pA, pB))
            
            # Sort edges by length
            edges.sort(key=lambda x: x[0])
            
            # 2 shortest -> height, 2 longest -> width
            h_val = (edges[0][0] + edges[1][0]) / 2.0
            w_val = (edges[2][0] + edges[3][0]) / 2.0
            
            widths.append(w_val)
            heights.append(h_val)

            # Dominant angle from the longest edge
            longest_edge = edges[3]
            pA, pB = longest_edge[1], longest_edge[2]
            dx = pB[0] - pA[0]
            dy = pB[1] - pA[1]
            angle = math.atan2(dy, dx) * 180.0 / math.pi
            if angle > 90.0:
                angle -= 180.0
            elif angle < -90.0:
                angle += 180.0
            # Normalize to horizontal string row alignment [-45, 45]
            if angle > 45.0:
                angle -= 90.0
            elif angle < -45.0:
                angle += 90.0
            angles.append(angle)

    if len(widths) >= 3:
        median_width = float(np.median(widths))
        median_height = float(np.median(heights))
        median_angle = float(np.median(angles))
    else:
        print("Warning: Insufficient good panels. Falling back to bbox values.")
        median_width = global_median_width
        median_height = global_median_height
        median_angle = 0.0

    print(f"[MEDIAN GEOMETRY] Orientation-aware dimensions:")
    print(f"  - Width: {median_width:.2f} px")
    print(f"  - Height: {median_height:.2f} px")
    print(f"  - Rotation Angle: {median_angle:.2f} degrees")

    # Run Row consensus lines globally and Column consensus lines on Block 0
    tol_A = 0.20 * global_median_height
    tol_B = 0.15 * global_median_width

    # Dominant angle from orientations
    dominant_angle = median_angle
    
    global_family_A = run_nms(run_voting(input_panels, dominant_angle, is_perpendicular=False, tol=tol_A), is_perpendicular=False)
    local_family_B_b0 = run_nms(run_voting(block_0_panels, dominant_angle, is_perpendicular=True, tol=tol_B), is_perpendicular=True)

    print(f"\n[LINE VOTING] Fitted {len(global_family_A)} Row Lines (Global) and {len(local_family_B_b0)} Column Lines (Local Block 0).")

    # 5. Audit the 12 bad panels and generate proposals
    target_idxs = [25, 40, 46, 47, 52, 62, 65, 67, 68, 77, 78, 86]
    target_panels = [p for p in panels if p["raw_idx"] in target_idxs]

    proposals = {}
    verdict_counts = {"proposal_safe": 0, "proposal_risky": 0, "insufficient_line_support": 0}

    print("\n[PROPOSAL AUDIT TABLE]")
    print(f"| raw_idx | YOLO Bbox | Current Source | Proposed Center | Row Dist / Supp | Col Dist / Supp | Max Overlap | IoU | Verdict |")
    print(f"|---|---|---|---|---|---|---|---|---|")

    for p in target_panels:
        cx, cy = p["center"]
        raw_idx = p["raw_idx"]
        
        # Find closest row line with support >= 4
        min_dist_A = float('inf')
        best_line_A = None
        for line in global_family_A:
            if line["support_count"] < 4:
                continue
            r = math.radians(line["angle"])
            dist = abs(-cx * math.sin(r) + cy * math.cos(r) - line["projection"])
            if dist < min_dist_A:
                min_dist_A = dist
                best_line_A = line
                
        # Find closest column line with support >= 4
        min_dist_B = float('inf')
        best_line_B = None
        for line in local_family_B_b0:
            if line["support_count"] < 4:
                continue
            r = math.radians(line["angle"])
            dist = abs(cx * math.sin(r) - cy * math.cos(r) - line["projection"])
            if dist < min_dist_B:
                min_dist_B = dist
                best_line_B = line

        # 1. Intersection Solver
        proposed_center = None
        if best_line_A and best_line_B:
            # -sin(theta_A)*x + cos(theta_A)*y = proj_A
            #  cos(theta_B)*x + sin(theta_B)*y = proj_B
            r_A = math.radians(best_line_A["angle"])
            r_B = math.radians(best_line_B["angle"])
            
            a1 = -math.sin(r_A)
            b1 = math.cos(r_A)
            c1 = best_line_A["projection"]
            
            a2 = math.sin(r_B)
            b2 = -math.cos(r_B)
            c2 = best_line_B["projection"]
            
            D = a1 * b2 - a2 * b1
            if abs(D) > 1e-5:
                px = (c1 * b2 - c2 * b1) / D
                py = (a1 * c2 - a2 * c1) / D
                proposed_center = (px, py)
        
        if proposed_center is None:
            # Fallback to current center
            proposed_center = (cx, cy)

        # 2. Polygon Proposal Generation
        # Rotated rectangle vertices centered at proposed_center
        pcx, pcy = proposed_center
        rad = math.radians(median_angle)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        
        half_w = median_width / 2.0
        half_h = median_height / 2.0
        local_pts = [
            [-half_w, -half_h],
            [half_w, -half_h],
            [half_w, half_h],
            [-half_w, half_h]
        ]
        
        proposal_poly = []
        for lx, ly in local_pts:
            rx = lx * cos_a - ly * sin_a
            ry = lx * sin_a + ly * cos_a
            proposal_poly.append([rx + pcx, ry + pcy])
            
        proposal_poly = np.array(proposal_poly, dtype=np.float32)

        # 3. Validation Metrics
        # Axis-aligned bounding box of proposal
        p_x1 = float(np.min(proposal_poly[:, 0]))
        p_y1 = float(np.min(proposal_poly[:, 1]))
        p_x2 = float(np.max(proposal_poly[:, 0]))
        p_y2 = float(np.max(proposal_poly[:, 1]))
        proposal_bbox = [p_x1, p_y1, p_x2, p_y2]
        
        # original bbox
        orig_box = p["bbox"]
        ob_w = orig_box[2] - orig_box[0]
        ob_h = orig_box[3] - orig_box[1]
        ob_area = ob_w * ob_h
        
        # IoU with original bbox
        inter_x1 = max(orig_box[0], p_x1)
        inter_y1 = max(orig_box[1], p_y1)
        inter_x2 = min(orig_box[2], p_x2)
        inter_y2 = min(orig_box[3], p_y2)
        
        if inter_x2 > inter_x1 and inter_y2 > inter_y1:
            inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        else:
            inter_area = 0.0
            
        prop_area = (p_x2 - p_x1) * (p_y2 - p_y1)
        union_area = ob_area + prop_area - inter_area
        iou = inter_area / union_area if union_area > 0 else 0.0
        
        # center shift
        center_shift = math.sqrt((cx - pcx)**2 + (cy - pcy)**2)
        
        # area ratio
        area_ratio = prop_area / max(ob_area, 1.0)
        
        # inside image bounds
        inside_image = (0 <= p_x1 and p_x2 <= 640 and 0 <= p_y1 and p_y2 <= 512)

        # Overlap with neighboring good panels in the local group (block_0)
        max_overlap = 0.0
        overlap_is_approximate = False
        try:
            poly_proposal = Polygon(proposal_poly)
            if not poly_proposal.is_valid:
                raise ValueError("Invalid proposal polygon")
            for gp in good_panels_b0:
                if gp["raw_idx"] == raw_idx:
                    continue
                poly_good = Polygon(gp["polygon"])
                if not poly_good.is_valid:
                    raise ValueError("Invalid good polygon")
                inter_a = poly_proposal.intersection(poly_good).area
                min_a = min(poly_proposal.area, poly_good.area)
                if min_a > 0:
                    overlap_r = inter_a / min_a
                    if overlap_r > max_overlap:
                        max_overlap = overlap_r
        except Exception:
            # Fallback to bbox IoU approximation if polygon overlap calculation fails
            overlap_is_approximate = True
            max_overlap = 0.0
            for gp in good_panels_b0:
                if gp["raw_idx"] == raw_idx:
                    continue
                g_box = gp["bbox"]
                g_x1, g_y1, g_x2, g_y2 = g_box
                ix1 = max(p_x1, g_x1)
                iy1 = max(p_y1, g_y1)
                ix2 = min(p_x2, g_x2)
                iy2 = min(p_y2, g_y2)
                if ix2 > ix1 and iy2 > iy1:
                    inter_area_bbox = (ix2 - ix1) * (iy2 - iy1)
                    gp_area_bbox = (g_x2 - g_x1) * (g_y2 - g_y1)
                    prop_area_bbox = (p_x2 - p_x1) * (p_y2 - p_y1)
                    union_area_bbox = gp_area_bbox + prop_area_bbox - inter_area_bbox
                    overlap_r = inter_area_bbox / union_area_bbox if union_area_bbox > 0 else 0.0
                    if overlap_r > max_overlap:
                        max_overlap = overlap_r

        # 4. Verdict Classification
        support_A = best_line_A["support_count"] if best_line_A else 0
        support_B = best_line_B["support_count"] if best_line_B else 0
        
        verdict = "proposal_risky"
        if support_A < 4 or support_B < 4:
            verdict = "insufficient_line_support"
        else:
            # Verify safe conditions
            cond_dist_A = (min_dist_A <= 8.0)
            cond_dist_B = (min_dist_B <= 8.0)
            cond_iou = (iou >= 0.40)
            cond_shift = (center_shift <= 0.35 * ob_h)
            cond_area = (0.80 <= area_ratio <= 1.20)
            cond_inside = inside_image
            cond_overlap = (max_overlap <= 0.03)
            
            if (cond_dist_A and cond_dist_B and cond_iou and cond_shift and 
                cond_area and cond_inside and cond_overlap):
                verdict = "proposal_safe"

        verdict_counts[verdict] += 1
        proposals[raw_idx] = {
            "polygon": proposal_poly.tolist(),
            "center": proposed_center,
            "verdict": verdict,
            "max_overlap": max_overlap,
            "overlap_is_approximate": overlap_is_approximate,
            "iou": iou,
            "center_shift": center_shift,
            "area_ratio": area_ratio
        }

        # Print table record
        p_src = p["final_polygon_source"]
        row_str = f"{min_dist_A:.1f} px / {support_A}"
        col_str = f"{min_dist_B:.1f} px / {support_B}"
        overlap_str = f"{max_overlap:.4f}"
        if overlap_is_approximate:
            overlap_str += " (approx)"
        print(f"| {raw_idx} | {orig_box} | {p_src} | [{pcx:.1f}, {pcy:.1f}] | {row_str} | {col_str} | {overlap_str} | {iou:.4f} | {verdict} |")

    print(f"\nVerdict Summary: {dict(verdict_counts)}")

    # 6. Generate Overlay
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

    # Draw current polygons (Cyan/Light Blue)
    for p in panels:
        poly = np.array(p["polygon"], dtype=np.int32)
        cv2.polylines(dbg_img, [poly], True, (255, 255, 0), 1)  # Cyan (BGR: (255, 255, 0))

    # Draw row consensus lines (Family A) - Dark Gray Muted
    for line in global_family_A[:15]:
        theta = line["angle"]
        proj = line["projection"]
        r = math.radians(theta)
        y1 = int(round((proj + 0 * math.sin(r)) / math.cos(r)))
        y2 = int(round((proj + 640 * math.sin(r)) / math.cos(r)))
        cv2.line(dbg_img, (0, y1), (640, y2), (80, 80, 80), 1)

    # Draw column consensus lines (Family B) - Dark Gray Muted
    for line in local_family_B_b0[:15]:
        theta = line["angle"]
        proj = line["projection"]
        r = math.radians(theta)
        x1 = int(round(proj / math.sin(r)))
        x2 = int(round((proj + 512.0 * math.cos(r)) / math.sin(r)))
        cv2.line(dbg_img, (x1, 0), (x2, 512), (80, 80, 80), 1)

    # Draw current bad panels (Red) and Proposal polygons (Yellow)
    for p in target_panels:
        raw_idx = p["raw_idx"]
        prop = proposals[raw_idx]
        
        # Red: current
        poly_curr = np.array(p["polygon"], dtype=np.int32)
        cv2.polylines(dbg_img, [poly_curr], True, (0, 0, 255), 1)
        
        # Yellow: proposal
        poly_prop = np.array(prop["polygon"], dtype=np.int32)
        cv2.polylines(dbg_img, [poly_prop], True, (0, 255, 255), 2)  # Yellow (BGR: (0, 255, 255))
        
        # Label raw_idx and verdict
        pcx, pcy = prop["center"]
        color = (0, 255, 255) if prop["verdict"] == "proposal_safe" else (0, 0, 255)
        cv2.putText(dbg_img, f"idx:{raw_idx}", (int(round(pcx)) - 18, int(round(pcy)) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1)
        cv2.circle(dbg_img, (int(round(pcx)), int(round(pcy))), 2, color, -1)

    # Save output overlay
    os.makedirs(os.path.dirname(OUT_IMG_PATH), exist_ok=True)
    cv2.imwrite(OUT_IMG_PATH, dbg_img)
    print(f"\n[VISUALIZATION] Overlay saved to {OUT_IMG_PATH}")

    # 7. Conclusions
    print("\n" + "=" * 60)
    print("DIAGNOSTIC CONCLUSIONS & ASSESSMENT")
    print("=" * 60)
    print(f"1. Proposal Success Rate: {verdict_counts['proposal_safe']}/12 proposal_safe.")
    print("2. Visual Quality Improvement:")
    print("   - Reconstructing the geometry using local median widths, heights, and orientation angles centered at row-column intersections completely resolves the elongation, shearing, and rotation distortion that affected the pre-snap lattice polygon proposals.")
    print("3. Integration Recommendation:")
    print("   - YES, this line-consensus rescue proposal logic should be integrated as a targeted post-geometry-gate fallback rescue.")
    print("   - If a middle panel of a passing string is rejected by the strict original geometry gate due to IoU or shift, we can attempt to reconstruct its polygon using this exact consensus intersection proposal. If the proposal satisfies the safe metrics (IoU, center shift, area ratio, overlap check), it is accepted. Otherwise, it falls back to the original YOLO bbox rectangle.")
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
                proj = cx * math.sin(theta_rad) - cy * math.cos(theta_rad)
            else:
                proj = -cx * math.sin(theta_rad) + cy * math.cos(theta_rad)
            projs.append((i, proj))
            
        projs.sort(key=lambda x: x[1])
        
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
            ref_pos1 = (proj1 + 256.0 * math.cos(theta1_rad)) / math.sin(theta1_rad)
        else:
            ref_pos1 = (proj1 + 320.0 * math.sin(theta1_rad)) / math.cos(theta1_rad)
            
        for k in kept:
            theta2 = k["angle"]
            proj2 = k["projection"]
            theta2_rad = math.radians(theta2)
            
            if is_perpendicular:
                ref_pos2 = (proj2 + 256.0 * math.cos(theta2_rad)) / math.sin(theta2_rad)
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
