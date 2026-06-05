import json
import math
import numpy as np

trace_path = "data/results/debug_logs/DJI_0029_R_final_polygon_trace.jsonl"
panels = []
with open(trace_path, "r", encoding="utf-8") as f:
    for line in f:
        panels.append(json.loads(line.strip()))

# Assign block_id via spatial component clustering
centers = np.array([p["center"] for p in panels], dtype=np.float32)
n_p = len(panels)
adj = {i: [] for i in range(n_p)}
for i in range(n_p):
    for j in range(i + 1, n_p):
        dist = np.linalg.norm(centers[i] - centers[j])
        if dist <= 120.0:
            adj[i].append(j)
            adj[j].append(i)

visited = set()
cluster_id = 0
for i in range(n_p):
    if i not in visited:
        queue = [i]
        visited.add(i)
        while queue:
            curr = queue.pop(0)
            panels[curr]["block_id"] = cluster_id
            for neighbor in adj[curr]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        cluster_id += 1

# Identify good panels
good_panels_all = []
for p in panels:
    clamped = p.get("final_polygon_clamped", False)
    src = p.get("final_polygon_source")
    if src in {"string_lattice_middle", "middle_locked_endpoint"} and not clamped:
        good_panels_all.append(p)

print(f"Total good panels: {len(good_panels_all)}")

def run_voting(panels_subset, dominant_angle, is_perpendicular, tol):
    candidate_lines = []
    base_angle = dominant_angle + 90.0 if is_perpendicular else dominant_angle
    angles_to_search = np.arange(base_angle - 10.0, base_angle + 10.0 + 0.1, 0.25)
    for theta in angles_to_search:
        theta_rad = math.radians(theta)
        projs = []
        for i, gp in enumerate(panels_subset):
            cx, cy = gp["center"]
            proj = cx * math.sin(theta_rad) - cy * math.cos(theta_rad) if is_perpendicular else -cx * math.sin(theta_rad) + cy * math.cos(theta_rad)
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
            raw_idx_list = [panels_subset[idx].get("raw_idx") for idx in support_indices]
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
        ref_pos1 = (proj1 + 256.0 * math.cos(theta1_rad)) / math.sin(theta1_rad) if is_perpendicular else (proj1 + 320.0 * math.sin(theta1_rad)) / math.cos(theta1_rad)
        for k in kept:
            theta2 = k["angle"]
            proj2 = k["projection"]
            theta2_rad = math.radians(theta2)
            ref_pos2 = (proj2 + 256.0 * math.cos(theta2_rad)) / math.sin(theta2_rad) if is_perpendicular else (proj2 + 320.0 * math.sin(theta2_rad)) / math.cos(theta2_rad)
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

targets = [25, 47, 52, 68, 78]
for idx in targets:
    p = [x for x in panels if x["raw_idx"] == idx][0]
    b_id = p["block_id"]
    local_good = [gp for gp in good_panels_all if gp["block_id"] == b_id]
    print(f"Target index {idx}: block_id={b_id}, local good panels count={len(local_good)}")
    
    # Calculate median geometry
    widths, heights, angles = [], [], []
    for gp in local_good:
        poly = gp.get("polygon")
        if poly and len(poly) >= 4:
            pts = np.array(poly, dtype=np.float32)
            edges = []
            for i in range(len(pts)):
                pA, pB = pts[i], pts[(i + 1) % len(pts)]
                edges.append((np.linalg.norm(pB - pA), pA, pB))
            edges.sort(key=lambda x: x[0])
            widths.append((edges[2][0] + edges[3][0]) / 2.0)
            heights.append((edges[0][0] + edges[1][0]) / 2.0)
            
            longest = edges[3]
            dx, dy = longest[2][0] - longest[1][0], longest[2][1] - longest[1][1]
            ang = math.atan2(dy, dx) * 180.0 / math.pi
            if ang > 90.0: ang -= 180.0
            elif ang < -90.0: ang += 180.0
            if ang > 45.0: ang -= 90.0
            elif ang < -45.0: ang += 90.0
            angles.append(ang)
            
    m_w = float(np.median(widths))
    m_h = float(np.median(heights))
    m_a = float(np.median(angles))
    print(f"  Median width={m_w:.2f}, height={m_h:.2f}, angle={m_a:.2f}")
    
    tol_A = 0.20 * m_h
    tol_B = 0.15 * m_w
    row_lines_local = run_nms(run_voting(local_good, m_a, is_perpendicular=False, tol=tol_A), is_perpendicular=False)
    col_lines_local = run_nms(run_voting(local_good, m_a, is_perpendicular=True, tol=tol_B), is_perpendicular=True)
    
    cx, cy = p["center"]
    best_line_A = None
    min_dist_A = float('inf')
    for line in row_lines_local:
        if line["support_count"] < 4: continue
        r = math.radians(line["angle"])
        dist = abs(-cx * math.sin(r) + cy * math.cos(r) - line["projection"])
        if dist < min_dist_A:
            min_dist_A = dist
            best_line_A = line
            
    if best_line_A is None or min_dist_A > 8.0:
        row_lines_global = run_nms(run_voting(good_panels_all, m_a, is_perpendicular=False, tol=tol_A), is_perpendicular=False)
        min_dist_A_g = float('inf')
        best_line_A_g = None
        for line in row_lines_global:
            if line["support_count"] < 4: continue
            r = math.radians(line["angle"])
            dist = abs(-cx * math.sin(r) + cy * math.cos(r) - line["projection"])
            if dist < min_dist_A_g:
                min_dist_A_g = dist
                best_line_A_g = line
        if best_line_A_g is not None:
            best_line_A = best_line_A_g
            min_dist_A = min_dist_A_g
            
    best_line_B = None
    min_dist_B = float('inf')
    for line in col_lines_local:
        if line["support_count"] < 4: continue
        r = math.radians(line["angle"])
        dist = abs(cx * math.sin(r) - cy * math.cos(r) - line["projection"])
        if dist < min_dist_B:
            min_dist_B = dist
            best_line_B = line
            
    print(f"  Best Line A (Row): dist={min_dist_A:.2f}, supp={best_line_A['support_count'] if best_line_A else None}")
    print(f"  Best Line B (Col): dist={min_dist_B:.2f}, supp={best_line_B['support_count'] if best_line_B else None}")
    
    if best_line_A and best_line_B:
        r_A, r_B = math.radians(best_line_A["angle"]), math.radians(best_line_B["angle"])
        a1, b1, c1 = -math.sin(r_A), math.cos(r_A), best_line_A["projection"]
        a2, b2, c2 = math.sin(r_B), -math.cos(r_B), best_line_B["projection"]
        D = a1 * b2 - a2 * b1
        px, py = ((c1 * b2 - c2 * b1) / D, (a1 * c2 - a2 * c1) / D) if abs(D) > 1e-5 else (cx, cy)
        
        # Build proposal polygon
        rad = math.radians(m_a)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        hw, hh = m_w / 2.0, m_h / 2.0
        prop_poly = []
        for lx, ly in [[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]]:
            prop_poly.append([lx * cos_a - ly * sin_a + px, lx * sin_a + ly * cos_a + py])
        prop_poly = np.array(prop_poly, dtype=np.float32)
        p_x1, p_y1 = float(np.min(prop_poly[:, 0])), float(np.min(prop_poly[:, 1]))
        p_x2, p_y2 = float(np.max(prop_poly[:, 0])), float(np.max(prop_poly[:, 1]))
        
        orig_box = p["bbox"]
        ob_w, ob_h = orig_box[2] - orig_box[0], orig_box[3] - orig_box[1]
        ob_area = ob_w * ob_h
        
        inter_x1, inter_y1 = max(orig_box[0], p_x1), max(orig_box[1], p_y1)
        inter_x2, inter_y2 = min(orig_box[2], p_x2), min(orig_box[3], p_y2)
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1) if (inter_x2 > inter_x1 and inter_y2 > inter_y1) else 0.0
        prop_area = (p_x2 - p_x1) * (p_y2 - p_y1)
        union_area = ob_area + prop_area - inter_area
        iou = inter_area / union_area if union_area > 0 else 0.0
        center_shift = math.sqrt((cx - px)**2 + (cy - py)**2)
        area_ratio = prop_area / max(ob_area, 1.0)
        
        print(f"  IoU={iou:.4f}, Shift={center_shift:.2f} (limit {0.35 * ob_h:.2f}), Area Ratio={area_ratio:.4f}")
