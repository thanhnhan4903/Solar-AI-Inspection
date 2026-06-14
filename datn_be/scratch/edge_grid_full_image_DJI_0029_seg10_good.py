import os
import json
import math
import numpy as np
import cv2
from pathlib import Path

def extract_input_image():
    """Extracts DJI_0029_R.JPG from Dataset.zip if it does not exist."""
    target_path = Path("data/precalib/DJI_0029_R.JPG")
    if target_path.exists():
        return target_path

    target_path.parent.mkdir(parents=True, exist_ok=True)
    zip_path = Path("../Dataset.zip")
    if not zip_path.exists():
        zip_path = Path("Dataset.zip")
    if not zip_path.exists():
        for p in Path(".").resolve().parents:
            candidate = p / "Dataset.zip"
            if candidate.exists():
                zip_path = candidate
                break
    if not zip_path.exists():
        raise FileNotFoundError("Could not locate Dataset.zip")

    import zipfile
    with zipfile.ZipFile(zip_path, 'r') as z:
        match = None
        for name in z.namelist():
            if "DJI_0029_R_JPG" in name and name.endswith(".jpg"):
                match = name
                break
        if not match:
            raise ValueError("Could not find DJI_0029_R in Dataset.zip")
        with open(target_path, "wb") as f:
            f.write(z.read(match))
    return target_path

def load_panels_from_logs():
    """Loads panel detections from the log file."""
    log_path = Path("data/results/debug_logs/DJI_0029_R_panel_refine.jsonl")
    if not log_path.exists():
        raise FileNotFoundError(f"Missing YOLO panel detection log at {log_path}")
    
    panels = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            panels.append(json.loads(line))
    return panels

def group_into_blocks(panels):
    """Groups panels into 1-column or 2-column blocks based on X coordinate."""
    panels_with_cx = []
    for p in panels:
        x1, y1, x2, y2 = p["bbox"]
        cx = (x1 + x2) / 2.0
        panels_with_cx.append((cx, p))
    panels_with_cx.sort(key=lambda x: x[0])
    
    columns = []
    for cx, p in panels_with_cx:
        if not columns:
            columns.append([p])
        else:
            mean_x = np.mean([(pt["bbox"][0] + pt["bbox"][2]) / 2.0 for pt in columns[-1]])
            if cx - mean_x > 40:
                columns.append([p])
            else:
                columns[-1].append(p)
                
    for col in columns:
        col.sort(key=lambda p: (p["bbox"][1] + p["bbox"][3]) / 2.0)
    columns.sort(key=lambda col: np.mean([(pt["bbox"][0] + pt["bbox"][2]) / 2.0 for pt in col]))
    
    blocks = []
    i = 0
    n_cols = len(columns)
    while i < n_cols:
        col_curr = columns[i]
        cx_curr = np.mean([(pt["bbox"][0] + pt["bbox"][2]) / 2.0 for pt in col_curr])
        
        if i + 1 < n_cols:
            col_next = columns[i + 1]
            cx_next = np.mean([(pt["bbox"][0] + pt["bbox"][2]) / 2.0 for pt in col_next])
            if cx_next - cx_curr < 70:
                blocks.append({
                    "columns": [col_curr, col_next],
                    "panels": col_curr + col_next
                })
                i += 2
                continue
                
        blocks.append({
            "columns": [col_curr],
            "panels": col_curr
        })
        i += 1
        
    return blocks

def fit_local_vertical_line(edges, x_base, y_start, y_end, search_half_width=8):
    h_img, w_img = edges.shape[:2]
    ys, xs = [], []
    y_start_int = int(max(0, math.floor(y_start)))
    y_end_int = int(min(h_img - 1, math.ceil(y_end)))
    
    for y in range(y_start_int, y_end_int + 1):
        x_min = int(max(0, math.floor(x_base - search_half_width)))
        x_max = int(min(w_img - 1, math.ceil(x_base + search_half_width)))
        for x in range(x_min, x_max + 1):
            if edges[y, x] == 255:
                ys.append(y)
                xs.append(x)
                
    if len(xs) >= 5:
        try:
            coeffs = np.polyfit(ys, xs, 1)
            a, b = coeffs[0], coeffs[1]
            angle_deg = abs(np.degrees(np.arctan(1.0 / a if a != 0 else 9999)))
            if 65.0 <= angle_deg <= 115.0:
                return a, b
        except Exception:
            pass
    return 0.0, float(x_base)

def fit_boundary_lines(edges, panels, x_baselines, y_min, y_max, n_segments=10):
    y_step = (y_max - y_min) / n_segments
    boundaries_local = [[] for _ in x_baselines]
    
    for k in range(n_segments):
        seg_y_start = y_min + k * y_step
        seg_y_end = seg_y_start + y_step
        
        seg_panels = [p for p in panels if (p["bbox"][1]+p["bbox"][3])/2.0 >= seg_y_start and (p["bbox"][1]+p["bbox"][3])/2.0 <= seg_y_end]
        if not seg_panels:
            seg_panels = panels
            
        for i, get_x_base in enumerate(x_baselines):
            x_base = get_x_base(seg_panels)
            a, b = fit_local_vertical_line(edges, x_base, seg_y_start, seg_y_end, search_half_width=8)
            boundaries_local[i].append((a, b, seg_y_start, seg_y_end))
            
    return boundaries_local

def get_boundary_x_at_y(boundary_pts, y_val):
    if y_val <= boundary_pts[0][1]:
        return boundary_pts[0][0]
    if y_val >= boundary_pts[-1][1]:
        return boundary_pts[-1][0]
    
    for i in range(len(boundary_pts) - 1):
        y0, x0 = boundary_pts[i][1], boundary_pts[i][0]
        y1, x1 = boundary_pts[i+1][1], boundary_pts[i+1][0]
        if y0 <= y_val <= y1:
            t = (y_val - y0) / (y1 - y0)
            return x0 + t * (x1 - x0)
    return boundary_pts[-1][0]

def evaluate_vertical_segment_support(gray, sobel_x_abs, a, b, y_start, y_end, boundary_key, n_samples=8, max_offset=8):
    ys = np.linspace(y_start, y_end, n_samples)
    offsets = []
    support_points = []
    
    for y in ys:
        y_idx = int(round(y))
        x_est = a * y + b
        
        profile = []
        for dx in range(-max_offset, max_offset + 1):
            px = int(round(x_est + dx))
            px_clamped = max(0, min(sobel_x_abs.shape[1] - 1, px))
            profile.append(sobel_x_abs[y_idx, px_clamped])
            
        if len(profile) == 2 * max_offset + 1:
            peak_idx = np.argmax(profile)
            offset = peak_idx - max_offset
            offsets.append(offset)
            support_points.append((x_est + offset, y_idx))
            
    if not offsets:
        return 0.0, 99.0, 99.0, [], False
        
    good_offsets = [o for o in offsets if abs(o) <= 5]
    support_ratio = len(good_offsets) / len(offsets)
    mean_abs_offset = np.mean([abs(o) for o in offsets])
    max_abs_offset = np.max([abs(o) for o in offsets])
    
    texture_ok = True
    if boundary_key in ("left_outer", "right_outer"):
        direction = 1 if boundary_key == "left_outer" else -1
        in_vals = []
        out_vals = []
        for y in ys:
            y_idx = int(round(y))
            x_fit = a * y + b
            x_in = int(round(x_fit + direction * 6))
            x_out = int(round(x_fit - direction * 6))
            if 0 <= x_in < gray.shape[1] and 0 <= x_out < gray.shape[1] and 0 <= y_idx < gray.shape[0]:
                in_vals.append(gray[y_idx, x_in])
                out_vals.append(gray[y_idx, x_out])
        if in_vals and out_vals:
            mean_in = np.mean(in_vals)
            mean_out = np.mean(out_vals)
            if mean_in < mean_out + 5.0:
                texture_ok = False
                
    is_good = support_ratio >= 0.65 and mean_abs_offset <= 6.5 and texture_ok
    return support_ratio, mean_abs_offset, max_abs_offset, support_points, is_good

def fit_and_diagnose_boundary_seg10(gray, edges, sobel_x_abs, panels, x_baseline_fn, y_min, y_max, boundary_key, n_segments=10):
    y_step = (y_max - y_min) / n_segments
    local_segment_fits = []
    segment_diagnostics = []
    
    max_dev_baseline = 8.0
    min_sup_ratio = 0.65
    
    for k in range(n_segments):
        seg_y_start = y_min + k * y_step
        seg_y_end = seg_y_start + y_step
        
        seg_panels = [p for p in panels if (p["bbox"][1]+p["bbox"][3])/2.0 >= seg_y_start and (p["bbox"][1]+p["bbox"][3])/2.0 <= seg_y_end]
        if not seg_panels:
            seg_panels = panels
            
        x_base = x_baseline_fn(seg_panels)
        a, b = fit_local_vertical_line(edges, x_base, seg_y_start, seg_y_end, search_half_width=8)
        
        sup_ratio, mean_off, max_off, sup_pts, is_good = evaluate_vertical_segment_support(
            gray, sobel_x_abs, a, b, seg_y_start, seg_y_end, boundary_key
        )
        
        y_mid = (seg_y_start + seg_y_end) / 2.0
        x_fit_mid = a * y_mid + b
        
        dev_ok = abs(x_fit_mid - x_base) <= max_dev_baseline
        angle_deg = np.degrees(np.arctan(abs(a)))
        angle_ok = angle_deg <= 15.0
        
        if boundary_key == "middle_divider":
            is_good = sup_ratio >= 0.5 and mean_off <= 6.5
        else:
            is_good = is_good and dev_ok and angle_ok and (sup_ratio >= min_sup_ratio)
        
        local_segment_fits.append((a, b, seg_y_start, seg_y_end))
        segment_diagnostics.append({
            "idx": k,
            "support_ratio": sup_ratio,
            "mean_offset": mean_off,
            "max_offset": max_off,
            "support_points": sup_pts,
            "is_good": is_good,
            "fallback_used": False,
            "baseline_x": x_base
        })
        
    good_indices = [idx for idx, diag in enumerate(segment_diagnostics) if diag["is_good"]]
    fallback_count = 0
    
    for k in range(n_segments):
        if not segment_diagnostics[k]["is_good"]:
            fallback_count += 1
            segment_diagnostics[k]["fallback_used"] = True
            
            if good_indices:
                closest_idx = min(good_indices, key=lambda idx: abs(idx - k))
                a_good, b_good, _, _ = local_segment_fits[closest_idx]
                y_mid = (y_min + k * y_step + y_min + (k + 1) * y_step) / 2.0
                baseline_x = segment_diagnostics[k]["baseline_x"]
                b_adj = baseline_x - a_good * y_mid
                local_segment_fits[k] = (a_good, b_adj, y_min + k * y_step, y_min + (k + 1) * y_step)
            else:
                y_mid = (y_min + k * y_step + y_min + (k + 1) * y_step) / 2.0
                baseline_x = segment_diagnostics[k]["baseline_x"]
                local_segment_fits[k] = (0.0, baseline_x, y_min + k * y_step, y_min + (k + 1) * y_step)
                
    # Create 11 polyline points
    y_boundaries = [y_min + k * y_step for k in range(n_segments + 1)]
    poly_pts = []
    for idx, y in enumerate(y_boundaries):
        if idx == 0:
            a, b, _, _ = local_segment_fits[0]
            x = a * y + b
        elif idx == n_segments:
            a, b, _, _ = local_segment_fits[-1]
            x = a * y + b
        else:
            a1, b1, _, _ = local_segment_fits[idx - 1]
            a2, b2, _, _ = local_segment_fits[idx]
            x = ((a1 * y + b1) + (a2 * y + b2)) / 2.0
        poly_pts.append((x, y))
        
    # Smooth with Moving Average of window size 3
    smoothed_xs = []
    for idx in range(len(poly_pts)):
        x_vals = []
        for offset in [-1, 0, 1]:
            curr_idx = idx + offset
            if 0 <= curr_idx < len(poly_pts):
                x_vals.append(poly_pts[curr_idx][0])
        smoothed_xs.append(np.mean(x_vals))
        
    # Endpoint top/bottom keeps within 4px, other vertex stays within 8px of baseline
    final_pts = []
    for idx, (x, y) in enumerate(poly_pts):
        x_smooth = smoothed_xs[idx]
        if idx == 0 or idx == n_segments:
            if abs(x_smooth - x) > 4.0:
                x_smooth = x + np.sign(x_smooth - x) * 4.0
        seg_idx = min(idx, n_segments - 1)
        base_x = segment_diagnostics[seg_idx]["baseline_x"]
        if abs(x_smooth - base_x) > 8.0:
            x_smooth = base_x + np.sign(x_smooth - base_x) * 8.0
        final_pts.append((x_smooth, y))
        
    max_v_jump = 0.0
    for idx in range(1, len(final_pts)):
        max_v_jump = max(max_v_jump, abs(final_pts[idx][0] - final_pts[idx-1][0]))
        
    max_turn_angle = 0.0
    for idx in range(1, len(final_pts) - 1):
        dx1 = final_pts[idx][0] - final_pts[idx-1][0]
        dy1 = final_pts[idx][1] - final_pts[idx-1][1]
        dx2 = final_pts[idx+1][0] - final_pts[idx][0]
        dy2 = final_pts[idx+1][1] - final_pts[idx][1]
        ang1 = np.arctan2(dy1, dx1)
        ang2 = np.arctan2(dy2, dx2)
        turn = abs(np.degrees(ang1 - ang2))
        turn = min(turn, 360.0 - turn)
        max_turn_angle = max(max_turn_angle, turn)
        
    return final_pts, local_segment_fits, segment_diagnostics, fallback_count, max_v_jump, max_turn_angle

def intersect_horizontal_line_with_vertical_boundary(m, c, boundary_pts):
    for i in range(len(boundary_pts) - 1):
        x0, y0 = boundary_pts[i]
        x1, y1 = boundary_pts[i+1]
        dy = y1 - y0
        dx = x1 - x0
        if abs(dy) < 1e-5:
            continue
        denom = 1.0 - m * (dx / dy)
        if abs(denom) < 1e-5:
            continue
        y_intersect = (m * x0 - m * y0 * (dx / dy) + c) / denom
        if y0 <= y_intersect <= y1:
            x_intersect = x0 + (y_intersect - y0) * (dx / dy)
            return x_intersect, y_intersect
            
    x0, y0 = boundary_pts[0]
    x1, y1 = boundary_pts[1]
    dy = y1 - y0
    dx = x1 - x0
    y_intersect = (m * x0 - m * y0 * (dx / (dy or 1.0)) + c) / (1.0 - m * (dx / (dy or 1.0)))
    x_intersect = x0 + (y_intersect - y0) * (dx / (dy or 1.0))
    return x_intersect, y_intersect

def draw_polyline(img, pts, color, thickness=2):
    for idx in range(len(pts) - 1):
        p1 = (int(round(pts[idx][0])), int(round(pts[idx][1])))
        p2 = (int(round(pts[idx+1][0])), int(round(pts[idx+1][1])))
        cv2.line(img, p1, p2, color, thickness, lineType=cv2.LINE_AA)

def detect_horizontal_gap_lines(gray, panels):
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    sobel_y_abs = np.abs(sobel_y)
    
    gap_lines = []
    rows = []
    sorted_panels = sorted(panels, key=lambda p: (p["bbox"][1]+p["bbox"][3])/2.0)
    for p in sorted_panels:
        yc = (p["bbox"][1] + p["bbox"][3]) / 2.0
        placed = False
        for r in rows:
            r_yc = np.mean([(pt["bbox"][1]+pt["bbox"][3])/2.0 for pt in r])
            if abs(yc - r_yc) < 20:
                r.append(p)
                placed = True
                break
        if not placed:
            rows.append([p])
            
    rows.sort(key=lambda r: np.mean([(pt["bbox"][1]+pt["bbox"][3])/2.0 for pt in r]))
    
    search_regions = []
    # Top edge of first row
    r_top_y = min(p["bbox"][1] for p in rows[0])
    search_regions.append((r_top_y - 6, r_top_y + 6, rows[0]))
    
    # Gaps between rows
    for i in range(len(rows) - 1):
        r_bot_y = max(p["bbox"][3] for p in rows[i])
        r_top_y_next = min(p["bbox"][1] for p in rows[i+1])
        y_mid = (r_bot_y + r_top_y_next) / 2.0
        combined_panels = rows[i] + rows[i+1]
        search_regions.append((y_mid - 8, y_mid + 8, combined_panels))
        
    # Bottom edge of last row
    r_bot_y_last = max(p["bbox"][3] for p in rows[-1])
    search_regions.append((r_bot_y_last - 6, r_bot_y_last + 6, rows[-1]))
    
    h_img, w_img = gray.shape[:2]
    for idx_reg, (y_start, y_end, reg_panels) in enumerate(search_regions):
        y_start_int = int(max(0, math.floor(y_start)))
        y_end_int = int(min(h_img - 1, math.ceil(y_end)))
        if y_start_int >= y_end_int:
            y_mid = int(round((y_start + y_end) / 2.0))
            x_min = int(max(0, min(p["bbox"][0] for p in reg_panels) - 5))
            x_max = int(min(w_img - 1, max(p["bbox"][2] for p in reg_panels) + 5))
            gap_lines.append((y_mid, x_min, x_max))
            continue
            
        x_min = int(max(0, min(p["bbox"][0] for p in reg_panels) - 5))
        x_max = int(min(w_img - 1, max(p["bbox"][2] for p in reg_panels) + 5))
        
        intensities = []
        for y_coord in range(y_start_int, y_end_int + 1):
            val = np.mean(sobel_y_abs[y_coord, x_min:x_max])
            intensities.append((val, y_coord))
            
        best_y = max(intensities, key=lambda x: x[0])[1]
        gap_lines.append((best_y, x_min, x_max))
        
    return gap_lines

def evaluate_angle_candidate(gray, raw_gaps, x_mid, theta, n_samples=25):
    m = math.tan(theta)
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    total_depth = 0.0
    
    max_offset = 6
    for y_anchor, x_start, x_end in raw_gaps:
        xs = np.linspace(x_start, x_end, n_samples)
        profile_sums = np.zeros(2 * max_offset + 1)
        profile_counts = np.zeros(2 * max_offset + 1)
        
        for dn in range(-max_offset, max_offset + 1):
            for x in xs:
                x_sample = x - dn * sin_theta
                y_sample = m * (x - x_mid) + y_anchor + dn * cos_theta
                x_idx = int(round(x_sample))
                y_idx = int(round(y_sample))
                if 0 <= x_idx < gray.shape[1] and 0 <= y_idx < gray.shape[0]:
                    profile_sums[dn + max_offset] += gray[y_idx, x_idx]
                    profile_counts[dn + max_offset] += 1
                    
        profile = profile_sums / np.maximum(profile_counts, 1)
        smoothed = np.convolve(profile, np.ones(3)/3.0, mode="same")
        
        best_depth = 0.0
        for v in range(1, len(smoothed) - 1):
            if smoothed[v] < smoothed[v-1] and smoothed[v] < smoothed[v+1]:
                left_peak = np.max(smoothed[:v])
                right_peak = np.max(smoothed[v+1:]) if v+1 < len(smoothed) else smoothed[v]
                depth = min(left_peak, right_peak) - smoothed[v]
                if depth > best_depth:
                    best_depth = depth
        total_depth += best_depth
    return total_depth

def get_oriented_line_y_at_x(m, x_mid, y_anchor, x_val):
    return m * (x_val - x_mid) + y_anchor

def intersect_oriented_line_with_boundary(m_perp, x_mid, y_anchor, boundary_pts):
    c = y_anchor - m_perp * x_mid
    return intersect_horizontal_line_with_vertical_boundary(m_perp, c, boundary_pts)

def main():
    img_path = extract_input_image()
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Could not read image at {img_path}")
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray)
    blur = cv2.GaussianBlur(gray_clahe, (5, 5), 0)
    edges = cv2.Canny(blur, 40, 120, apertureSize=3)
    
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_x_abs = np.abs(sobel_x)
    
    panels = load_panels_from_logs()
    blocks = group_into_blocks(panels)
    
    img_blocks = img.copy()
    img_lines = img.copy()
    img_panels = img.copy()
    img_support = img.copy()
    img_snap = img.copy()
    img_panels_verified = img.copy()
    
    stats = []
    
    for b_idx, b in enumerate(blocks):
        block_name = f"block_{b_idx + 1}"
        n_cols = len(b["columns"])
        
        y_min = min(p["bbox"][1] for p in b["panels"])
        y_max = max(p["bbox"][3] for p in b["panels"])
        x_min_roi = min(p["bbox"][0] for p in b["panels"]) - 10
        x_max_roi = max(p["bbox"][2] for p in b["panels"]) + 10
        
        # Draw block ROI
        cv2.rectangle(img_blocks, (x_min_roi, y_min), (x_max_roi, y_max), (0, 255, 0), 2)
        cv2.putText(img_blocks, block_name, (x_min_roi + 5, y_min + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Fit boundaries
        if n_cols == 1:
            col = b["columns"][0]
            x_baselines = [
                lambda seg: np.median([p["bbox"][0] for p in seg]),
                lambda seg: np.median([p["bbox"][2] for p in seg])
            ]
            boundary_keys = ["left_outer", "right_outer"]
        else:
            col1 = b["columns"][0]
            col2 = b["columns"][1]
            x_baselines = [
                lambda seg: np.median([p["bbox"][0] for p in seg if p in col1] or [p["bbox"][0] for p in seg]),
                lambda seg: np.median([((p1["bbox"][2]+p2["bbox"][0])/2.0) for p1 in seg if p1 in col1 for p2 in seg if p2 in col2] or [(p["bbox"][2]+p["bbox"][0])/2.0 for p in seg]),
                lambda seg: np.median([p["bbox"][2] for p in seg if p in col2] or [p["bbox"][2] for p in seg])
            ]
            boundary_keys = ["left_outer", "middle_divider", "right_outer"]
            
        boundaries = {}
        boundaries_local_fits = {}
        boundaries_diags = {}
        boundaries_fallbacks = {}
        boundaries_max_jumps = {}
        boundaries_max_turns = {}
        
        for idx_bk, bkey in enumerate(boundary_keys):
            pts, local_fits, diags, fb_cnt, max_jmp, max_trn = fit_and_diagnose_boundary_seg10(
                gray, edges, sobel_x_abs, b["panels"], x_baselines[idx_bk], y_min, y_max, bkey
            )
            boundaries[bkey] = pts
            boundaries_local_fits[bkey] = local_fits
            boundaries_diags[bkey] = diags
            boundaries_fallbacks[bkey] = fb_cnt
            boundaries_max_jumps[bkey] = max_jmp
            boundaries_max_turns[bkey] = max_trn
            
        left_pts = boundaries["left_outer"]
        right_pts = boundaries["right_outer"]
        
        # Determine vertical direction of the block
        all_slopes = []
        for bkey in boundary_keys:
            fits = boundaries_local_fits[bkey]
            diags = boundaries_diags[bkey]
            for k, f in enumerate(fits):
                if not diags[k]["fallback_used"]:
                    all_slopes.append(f[0])
        a_avg = np.mean(all_slopes) if all_slopes else 0.0
        
        # Base horizontal direction perpendicular to the vertical boundary
        theta_base = math.atan(-a_avg)
        
        # Detect initial horizontal gap anchors (y-coordinates)
        raw_horiz_gaps = detect_horizontal_gap_lines(gray, b["panels"])
        x_mid = (x_min_roi + x_max_roi) / 2.0
        
        # 1. Search for best shared angle within +/- 2 degrees of base
        best_theta = theta_base
        max_score = -1.0
        for delta_deg in np.linspace(-2.0, 2.0, 21):
            theta_cand = theta_base + math.radians(delta_deg)
            score = evaluate_angle_candidate(gray, raw_horiz_gaps, x_mid, theta_cand, n_samples=30)
            if score > max_score:
                max_score = score
                best_theta = theta_cand
                
        m_perp = math.tan(best_theta)
        vertical_angle_deg = np.degrees(np.arctan(1.0 / a_avg if a_avg != 0 else 9999))
        horizontal_angle_deg = np.degrees(best_theta)
        
        # 2. Build profile projection along normal for each raw gap and find dark-valley center
        candidate_y_snaps = []
        for y, x_start, x_end in raw_horiz_gaps:
            xs = np.linspace(x_start, x_end, 30)
            profile_sums = np.zeros(2 * 8 + 1)
            profile_counts = np.zeros(2 * 8 + 1)
            for dn in range(-8, 9):
                for x in xs:
                    x_sample = x - dn * math.sin(best_theta)
                    y_sample = m_perp * (x - x_mid) + y + dn * math.cos(best_theta)
                    x_idx = int(round(x_sample))
                    y_idx = int(round(y_sample))
                    if 0 <= x_idx < gray.shape[1] and 0 <= y_idx < gray.shape[0]:
                        profile_sums[dn + 8] += gray[y_idx, x_idx]
                        profile_counts[dn + 8] += 1
            profile = profile_sums / np.maximum(profile_counts, 1)
            smoothed = np.convolve(profile, np.ones(3)/3.0, mode="same")
            
            # Find best valley
            best_valley_idx = -1
            best_depth = -1.0
            for v in range(1, len(smoothed) - 1):
                if smoothed[v] < smoothed[v-1] and smoothed[v] < smoothed[v+1]:
                    left_peak = np.max(smoothed[:v])
                    right_peak = np.max(smoothed[v+1:])
                    depth = min(left_peak, right_peak) - smoothed[v]
                    if depth > best_depth:
                        best_depth = depth
                        best_valley_idx = v
            
            # Snap limit check
            snap_offset = best_valley_idx - 8 if best_valley_idx != -1 else 999.0
            is_valley = (best_valley_idx != -1 and best_depth >= 2.0 and abs(snap_offset) <= 3.0)
            y_snap = y + snap_offset if is_valley else y
            
            candidate_y_snaps.append({
                "y_raw": y,
                "y_snap": y_snap,
                "depth": best_depth if best_valley_idx != -1 else 0.0,
                "is_valley": is_valley,
                "x_start": x_start,
                "x_end": x_end
            })
            
        # 3. Fit evenly-spaced Pitch Grid
        good_y_snaps = [c["y_snap"] for c in candidate_y_snaps if c["is_valley"]]
        indices = np.array([idx for idx, c in enumerate(candidate_y_snaps) if c["is_valley"]])
        if len(good_y_snaps) >= 2:
            coeffs = np.polyfit(indices, good_y_snaps, 1)
            estimated_pitch = coeffs[0]
            y_start_grid = coeffs[1]
            fitted_ys = estimated_pitch * indices + y_start_grid
            pitch_std = np.std(np.array(good_y_snaps) - fitted_ys)
        else:
            raw_ys = np.array([c["y_raw"] for c in candidate_y_snaps])
            raw_indices = np.arange(len(raw_ys))
            coeffs = np.polyfit(raw_indices, raw_ys, 1)
            estimated_pitch = coeffs[0]
            y_start_grid = coeffs[1]
            pitch_std = 0.0

        # 4. Refined snap search around grid position
        snapped_horiz_gaps = []
        rejected_count = 0
        interpolated_missing_count = 0
        
        for idx, (y_raw, x_start, x_end) in enumerate(raw_horiz_gaps):
            y_grid = estimated_pitch * idx + y_start_grid
            
            xs = np.linspace(x_start, x_end, 30)
            profile_sums = np.zeros(2 * 3 + 1)
            profile_counts = np.zeros(2 * 3 + 1)
            for dn in range(-3, 4):
                for x in xs:
                    x_sample = x - dn * math.sin(best_theta)
                    y_sample = m_perp * (x - x_mid) + y_grid + dn * math.cos(best_theta)
                    x_idx = int(round(x_sample))
                    y_idx = int(round(y_sample))
                    if 0 <= x_idx < gray.shape[1] and 0 <= y_idx < gray.shape[0]:
                        profile_sums[dn + 3] += gray[y_idx, x_idx]
                        profile_counts[dn + 3] += 1
            profile = profile_sums / np.maximum(profile_counts, 1)
            smoothed = np.convolve(profile, np.ones(3)/3.0, mode="same")
            
            best_valley_idx = -1
            best_depth = -1.0
            for v in range(1, len(smoothed) - 1):
                if smoothed[v] < smoothed[v-1] and smoothed[v] < smoothed[v+1]:
                    left_peak = np.max(smoothed[:v])
                    right_peak = np.max(smoothed[v+1:])
                    depth = min(left_peak, right_peak) - smoothed[v]
                    if depth > best_depth:
                        best_depth = depth
                        best_valley_idx = v
                        
            if best_valley_idx != -1 and best_depth >= 2.0:
                snap_offset = best_valley_idx - 3
                y_snap = y_grid + snap_offset
                dark_valley_score = best_depth
            else:
                snap_offset = 0
                y_snap = y_grid
                dark_valley_score = 0.0
                
            # Support ratio calculation
            sup_cnt = 0
            support_pts = []
            for x in xs:
                loc_profile = []
                for dy in range(-3, 4):
                    y_est = m_perp * (x - x_mid) + y_snap
                    py = int(round(y_est + dy))
                    py_clamped = max(0, min(gray.shape[0] - 1, py))
                    loc_profile.append(gray[py_clamped, int(round(x))])
                
                loc_smoothed = np.convolve(loc_profile, np.ones(3)/3.0, mode="same")
                loc_valley_idx = -1
                loc_best_depth = -1.0
                for v in range(1, len(loc_smoothed) - 1):
                    if loc_smoothed[v] < loc_smoothed[v-1] and loc_smoothed[v] < loc_smoothed[v+1]:
                        lp = np.max(loc_smoothed[:v])
                        rp = np.max(loc_smoothed[v+1:])
                        dp = min(lp, rp) - loc_smoothed[v]
                        if dp > loc_best_depth:
                            loc_best_depth = dp
                            loc_valley_idx = v
                
                loc_offset = loc_valley_idx - 3 if loc_valley_idx != -1 else 999.0
                if abs(loc_offset) <= 1:
                    sup_cnt += 1
                support_pts.append((int(round(x)), m_perp * (x - x_mid) + y_snap + (loc_offset if abs(loc_offset) <= 3 else 0)))
                
            support_ratio = sup_cnt / len(xs)
            mean_abs_offset = abs(y_snap - y_grid)
            pitch_deviation = abs(y_snap - y_grid)
            
            # Check gate
            is_gate_passed = (
                dark_valley_score >= 2.0 and
                support_ratio >= 0.65 and
                mean_abs_offset <= 2.5 and
                pitch_deviation <= 3.0
            )
            
            if not is_gate_passed:
                rejected_count += 1
                y_snap = y_grid
                is_gate_passed = False
                interpolated_missing_count += 1
                
            snapped_horiz_gaps.append({
                "y_snap": y_snap,
                "is_good": is_gate_passed,
                "support_ratio": support_ratio,
                "mean_offset": mean_abs_offset,
                "pitch_deviation": pitch_deviation,
                "support_pts": support_pts,
                "x_start": x_start,
                "x_end": x_end,
                "dark_valley_score": dark_valley_score
            })
            
        # --- DRAW ON IMAGE 1: SUPPORT ---
        for bkey in boundary_keys:
            pts = boundaries[bkey]
            diag_list = boundaries_diags[bkey]
            for idx_diag, diag in enumerate(diag_list):
                ys = diag_list[idx_diag]["idx"] * ((y_max - y_min) / 10) + y_min
                ye = ys + ((y_max - y_min) / 10)
                a_fit, b_fit, _, _ = boundaries_local_fits[bkey][idx_diag]
                x_s = int(round(a_fit * ys + b_fit))
                x_e = int(round(a_fit * ye + b_fit))
                color = (0, 255, 0) if not diag["fallback_used"] else (0, 165, 255)
                cv2.line(img_support, (x_s, int(ys)), (x_e, int(ye)), color, 2)
                for pt in diag["support_points"]:
                    cv2.circle(img_support, (int(round(pt[0])), pt[1]), 1, (255, 0, 255), -1)
                    
        for g in snapped_horiz_gaps:
            y_snap_anchor = g["y_snap"]
            x_start = g["x_start"]
            x_end = g["x_end"]
            is_good = g["is_good"]
            
            y_start_draw = int(round(get_oriented_line_y_at_x(m_perp, x_mid, y_snap_anchor, x_start)))
            y_end_draw = int(round(get_oriented_line_y_at_x(m_perp, x_mid, y_snap_anchor, x_end)))
            
            color = (0, 255, 0) if is_good else (0, 0, 255)
            cv2.line(img_support, (x_start, y_start_draw), (x_end, y_end_draw), color, 2)
            for pt in g["support_pts"]:
                cv2.circle(img_support, (int(round(pt[0])), int(round(pt[1]))), 1, (255, 0, 255), -1)
                
        # --- DRAW ON IMAGE 2: LINES ---
        for g in snapped_horiz_gaps:
            y_snap_anchor = g["y_snap"]
            x_start = g["x_start"]
            x_end = g["x_end"]
            y_start_draw = int(round(get_oriented_line_y_at_x(m_perp, x_mid, y_snap_anchor, x_start)))
            y_end_draw = int(round(get_oriented_line_y_at_x(m_perp, x_mid, y_snap_anchor, x_end)))
            cv2.line(img_lines, (x_start, y_start_draw), (x_end, y_end_draw), (0, 255, 255), 2)
            
        draw_polyline(img_lines, boundaries["left_outer"], (180, 0, 0), 2)
        draw_polyline(img_lines, boundaries["right_outer"], (180, 0, 0), 2)
        if "middle_divider" in boundaries:
            draw_polyline(img_lines, boundaries["middle_divider"], (255, 255, 0), 2)
            
        # --- DRAW ON IMAGE 3: PANELS (Original) & SNAP ---
        for g in snapped_horiz_gaps:
            y_snap_anchor = g["y_snap"]
            x_start = g["x_start"]
            x_end = g["x_end"]
            y_start_draw = int(round(get_oriented_line_y_at_x(m_perp, x_mid, y_snap_anchor, x_start)))
            y_end_draw = int(round(get_oriented_line_y_at_x(m_perp, x_mid, y_snap_anchor, x_end)))
            cv2.line(img_snap, (x_start, y_start_draw), (x_end, y_end_draw), (0, 255, 255), 2)
            
        draw_polyline(img_snap, boundaries["left_outer"], (180, 0, 0), 2)
        draw_polyline(img_snap, boundaries["right_outer"], (180, 0, 0), 2)
        if "middle_divider" in boundaries:
            draw_polyline(img_snap, boundaries["middle_divider"], (255, 255, 0), 2)
            
        # Draw vertical boundaries on panels
        draw_polyline(img_panels, boundaries["left_outer"], (180, 0, 0), 2)
        draw_polyline(img_panels, boundaries["right_outer"], (180, 0, 0), 2)
        if "middle_divider" in boundaries:
            draw_polyline(img_panels, boundaries["middle_divider"], (180, 0, 0), 2)
            
        draw_polyline(img_panels_verified, boundaries["left_outer"], (180, 0, 0), 2)
        draw_polyline(img_panels_verified, boundaries["right_outer"], (180, 0, 0), 2)
        if "middle_divider" in boundaries:
            draw_polyline(img_panels_verified, boundaries["middle_divider"], (180, 0, 0), 2)
            
        # --- BUILD & VERIFY POLYGONS ---
        verified_panel_count = 0
        skipped_panel_count = 0
        
        for col_idx, col in enumerate(b["columns"]):
            if n_cols == 1:
                c_left = left_pts
                c_right = right_pts
            else:
                mid_pts = boundaries["middle_divider"]
                if col_idx == 0:
                    c_left = left_pts
                    c_right = mid_pts
                else:
                    c_left = mid_pts
                    c_right = right_pts
                    
            for row_idx in range(len(snapped_horiz_gaps) - 1):
                g_t = snapped_horiz_gaps[row_idx]
                g_b = snapped_horiz_gaps[row_idx + 1]
                
                y_snap_t = g_t["y_snap"]
                y_snap_b = g_b["y_snap"]
                
                tl = intersect_oriented_line_with_boundary(m_perp, x_mid, y_snap_t, c_left)
                tr = intersect_oriented_line_with_boundary(m_perp, x_mid, y_snap_t, c_right)
                br = intersect_oriented_line_with_boundary(m_perp, x_mid, y_snap_b, c_right)
                bl = intersect_oriented_line_with_boundary(m_perp, x_mid, y_snap_b, c_left)
                
                poly_proposal = np.array([tl, tr, br, bl], dtype=np.int32)
                
                is_convex = cv2.isContourConvex(poly_proposal)
                area_prop = cv2.contourArea(poly_proposal)
                
                matched_panel = col[row_idx] if row_idx < len(col) else None
                area_ok = True
                shift_ok = True
                
                if matched_panel:
                    w_orig = matched_panel["bbox"][2] - matched_panel["bbox"][0]
                    h_orig = matched_panel["bbox"][3] - matched_panel["bbox"][1]
                    area_orig = w_orig * h_orig
                    area_ratio = area_prop / area_orig if area_orig > 0 else 0.0
                    area_ok = 0.65 <= area_ratio <= 1.45
                    
                    cx_prop = np.mean(poly_proposal[:, 0])
                    cy_prop = np.mean(poly_proposal[:, 1])
                    cx_orig = (matched_panel["bbox"][0] + matched_panel["bbox"][2]) / 2.0
                    cy_orig = (matched_panel["bbox"][1] + matched_panel["bbox"][3]) / 2.0
                    c_shift = math.sqrt((cx_prop - cx_orig)**2 + (cy_prop - cy_orig)**2)
                    shift_ok = c_shift <= 30.0
                    
                order_ok = tl[1] < bl[1] and tr[1] < br[1]
                lr_ok = tl[0] < tr[0] and bl[0] < br[0]
                
                is_valid = is_convex and area_ok and shift_ok and order_ok and lr_ok
                
                if is_valid:
                    is_both_snapped = g_t["is_good"] and g_b["is_good"]
                    color = (0, 255, 255) if is_both_snapped else (0, 165, 255)
                    cv2.polylines(img_panels_verified, [poly_proposal], True, color, 1, cv2.LINE_AA)
                    cv2.polylines(img_panels, [poly_proposal], True, color, 1, cv2.LINE_AA)
                    verified_panel_count += 1
                else:
                    skipped_panel_count += 1
                    if matched_panel:
                        orig_poly = np.array(matched_panel["refined_polygon"], dtype=np.int32)
                        cv2.polylines(img_panels_verified, [orig_poly], True, (100, 100, 100), 1, cv2.LINE_AA)
                        cv2.polylines(img_panels, [orig_poly], True, (100, 100, 100), 1, cv2.LINE_AA)
                        
        stats.append({
            "name": block_name,
            "horizontal_angle_deg": horizontal_angle_deg,
            "estimated_pitch_px": estimated_pitch,
            "pitch_std_px": pitch_std,
            "number_of_interpolated_missing_lines": interpolated_missing_count,
            "rejected_horizontal_lines_count": rejected_count,
            "verified_panel_count": verified_panel_count,
            "skipped_panel_count": skipped_panel_count
        })
        
    # Save outputs
    out_dir = Path("data/results/debug")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    cv2.imwrite(str(out_dir / "debug_DJI_0029_R_full_image_blocks.JPG"), img_blocks)
    cv2.imwrite(str(out_dir / "debug_DJI_0029_R_full_image_lines.JPG"), img_lines)
    cv2.imwrite(str(out_dir / "debug_DJI_0029_R_full_image_panels.JPG"), img_panels)
    cv2.imwrite(str(out_dir / "debug_DJI_0029_R_full_image_line_support.JPG"), img_support)
    cv2.imwrite(str(out_dir / "debug_DJI_0029_R_full_image_line_snap.JPG"), img_snap)
    cv2.imwrite(str(out_dir / "debug_DJI_0029_R_full_image_panels_verified_seg10.JPG"), img_panels_verified)
    
    # Print stats
    print("\n" + "="*95)
    print("DIAGNOSTIC REPORT PER BLOCK (THERMAL GAP SNAPPED + SEGMENTED10)")
    print("="*95)
    for s in stats:
        print(f"Block Name: {s['name']}")
        print(f"  Horizontal Angle:                    {s['horizontal_angle_deg']:.2f} deg")
        print(f"  Estimated Pitch:                     {s['estimated_pitch_px']:.2f} px")
        print(f"  Pitch Std Dev:                       {s['pitch_std_px']:.4f} px")
        print(f"  Number of Interpolated/Grid Lines:   {s['number_of_interpolated_missing_lines']}")
        print(f"  Rejected Horizontal Lines Count:     {s['rejected_horizontal_lines_count']}")
        print(f"  Verified Panel Polygons Created:     {s['verified_panel_count']}")
        print(f"  Skipped Panel Polygons:              {s['skipped_panel_count']}")
        print("-" * 95)

if __name__ == "__main__":
    main()

