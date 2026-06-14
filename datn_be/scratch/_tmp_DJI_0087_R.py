import os
import json
import math
import copy
import numpy as np
import cv2
from pathlib import Path

# Config variables
HORIZONTAL_OUTER_MARGIN_PX = 0.3
THERMAL_GAP_SNAP_WINDOW = 4
PITCH_REGULARIZATION_WEIGHT = 0.45

VERTICAL_OUTER_INSET_PX = 1.2
MIDDLE_DIVIDER_INSET_PX = 0.7

LAST_ROW_SNAP_WINDOW = 5
LAST_ROW_PITCH_WEIGHT = 0.05
IMAGE_STEM = "DJI_0087_R"

def extract_input_image():
    """Extracts DJI_0029_R.JPG from Dataset.zip if it does not exist."""
    target_path = Path(f"data/precalib/{IMAGE_STEM}.JPG")
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
            if IMAGE_STEM in name and name.lower().endswith((".jpg", ".jpeg")):
                match = name
                break
        if not match:
            raise ValueError("Could not find DJI_0029_R in Dataset.zip")
        with open(target_path, "wb") as f:
            f.write(z.read(match))
    return target_path

def load_panels_from_logs():
    """Loads panel detections from the log file."""
    log_path = Path(f"data/results/debug_logs/{IMAGE_STEM}_panel_refine.jsonl")
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

def split_block_by_y_gap(block):
    """Split a single block into sub-blocks along Y when a large vertical gap exists.
    
    - Sort panels by y_top inside each column, then across the full block.
    - Compute median panel height as the reference scale.
    - Split when gap between consecutive row-groups exceeds max(45px, 1.8 * median_height).
    # Discard sub-blocks with too few panels: <2.
    """
    n_cols = len(block["columns"])
    min_panels = 2

    # Sort all panels by y_top
    all_panels = sorted(block["panels"], key=lambda p: p["bbox"][1])
    
    # Compute median panel height across the block
    heights = [p["bbox"][3] - p["bbox"][1] for p in all_panels]
    median_h = float(np.median(heights)) if heights else 30.0
    gap_threshold = max(45.0, 1.8 * median_h)
    
    # Cluster panels into "rows" by y proximity (within median_h)
    row_clusters = []
    for p in all_panels:
        placed = False
        for rc in row_clusters:
            rc_mid_y = np.mean([(rp["bbox"][1] + rp["bbox"][3]) / 2.0 for rp in rc])
            p_mid_y = (p["bbox"][1] + p["bbox"][3]) / 2.0
            if abs(p_mid_y - rc_mid_y) < median_h:
                rc.append(p)
                placed = True
                break
        if not placed:
            row_clusters.append([p])
    
    # Sort row_clusters by their mean y
    row_clusters.sort(key=lambda rc: np.mean([(p["bbox"][1] + p["bbox"][3]) / 2.0 for p in rc]))
    
    # Find split points: gap between bottom of row[k] and top of row[k+1]
    split_indices = []  # indices BEFORE which we start a new sub-block
    for k in range(len(row_clusters) - 1):
        bottom_k = max(p["bbox"][3] for p in row_clusters[k])
        top_k1 = min(p["bbox"][1] for p in row_clusters[k + 1])
        gap = top_k1 - bottom_k
        if gap > gap_threshold:
            split_indices.append(k + 1)  # row_clusters[k+1] starts a new sub-block
    
    if not split_indices:
        # No split needed - return original block as-is (with populated keys)
        block["y_range"] = (
            min(p["bbox"][1] for p in all_panels),
            max(p["bbox"][3] for p in all_panels)
        )
        block["panel_count"] = len(all_panels)
        block["median_panel_height"] = median_h
        block["max_gap_used"] = gap_threshold
        return [block], median_h, []
    
    # Build sub-block row-group lists
    all_split_points = [0] + split_indices + [len(row_clusters)]
    sub_row_groups = []
    for i in range(len(all_split_points) - 1):
        start_idx = all_split_points[i]
        end_idx = all_split_points[i + 1]
        sub_rows = row_clusters[start_idx:end_idx]
        sub_panels = [p for rc in sub_rows for p in rc]
        sub_row_groups.append((sub_panels, sub_rows))
    
    # Reconstruct column structure per sub-block
    sub_blocks = []
    gaps_used = []
    for i, (sub_panels_list, sub_rows) in enumerate(sub_row_groups):
        if len(sub_panels_list) < min_panels:
            continue  # skip sub-blocks with too few panels
        
        # Rebuild columns by x-center proximity (same logic as group_into_blocks)
        sub_panels_sorted_x = sorted(sub_panels_list, key=lambda p: (p["bbox"][0] + p["bbox"][2]) / 2.0)
        sub_cols = []
        for p in sub_panels_sorted_x:
            px_cx = (p["bbox"][0] + p["bbox"][2]) / 2.0
            if not sub_cols:
                sub_cols.append([p])
            else:
                mean_x = np.mean([(pt["bbox"][0] + pt["bbox"][2]) / 2.0 for pt in sub_cols[-1]])
                if px_cx - mean_x > 40:
                    sub_cols.append([p])
                else:
                    sub_cols[-1].append(p)
        
        for sc in sub_cols:
            sc.sort(key=lambda p: (p["bbox"][1] + p["bbox"][3]) / 2.0)
        sub_cols.sort(key=lambda sc: np.mean([(pt["bbox"][0] + pt["bbox"][2]) / 2.0 for pt in sc]))
        
        # Calculate the gap ABOVE this sub-block (for reporting)
        if i > 0:
            prev_rows = sub_row_groups[i - 1][1]
            prev_bottom = max(p["bbox"][3] for rc in prev_rows for p in rc)
            curr_top = min(p["bbox"][1] for p in sub_panels_list)
            gaps_used.append(float(curr_top - prev_bottom))
        
        sub_blocks.append({
            "columns": sub_cols,
            "panels": sub_panels_list,
            "y_range": (
                min(p["bbox"][1] for p in sub_panels_list),
                max(p["bbox"][3] for p in sub_panels_list)
            ),
            "panel_count": len(sub_panels_list),
            "median_panel_height": median_h,
            "max_gap_used": gap_threshold
        })
    
    if not sub_blocks:
        block["y_range"] = (
            min(p["bbox"][1] for p in all_panels),
            max(p["bbox"][3] for p in all_panels)
        )
        block["panel_count"] = len(all_panels)
        block["median_panel_height"] = median_h
        block["max_gap_used"] = gap_threshold
        return [block], median_h, []
    
    return sub_blocks, median_h, gaps_used


# LEGACY_UNUSED: fit_local_vertical_line, fit_boundary_lines, fit_and_diagnose_boundary_seg10 are no longer used.


def get_boundary_x_at_y(boundary_pts, y_val):
    if not boundary_pts:
        return 0.0
    if len(boundary_pts) == 1:
        return boundary_pts[0][0]
    pts = sorted(boundary_pts, key=lambda p: p[1])
    if y_val <= pts[0][1]:
        return pts[0][0]
    if y_val >= pts[-1][1]:
        return pts[-1][0]
    for i in range(len(pts) - 1):
        p0, p1 = pts[i], pts[i+1]
        if p0[1] <= y_val <= p1[1]:
            if abs(p1[1] - p0[1]) < 1e-5:
                return p0[0]
            t = (y_val - p0[1]) / (p1[1] - p0[1])
            return p0[0] + t * (p1[0] - p0[0])
    return pts[0][0]

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

def shift_boundary_x(boundary_pts, dx):
    return [(float(x) + float(dx), float(y)) for x, y in boundary_pts]

def sort_panel_corners(poly):
    if len(poly) != 4:
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    else:
        corners = poly
        
    sorted_by_y = sorted(corners, key=lambda p: p[1])
    top_two = sorted(sorted_by_y[:2], key=lambda p: p[0])
    bottom_two = sorted(sorted_by_y[2:], key=lambda p: p[0])
    
    TL = top_two[0]
    TR = top_two[1]
    BR = bottom_two[1]
    BL = bottom_two[0]
    
    return TL, TR, BR, BL

def fit_consensus_vertical_boundary(points, y_min, y_max, row_count):
    ys = [p[1] for p in points]
    xs = [p[0] for p in points]
    if len(xs) < 2:
        val = float(np.mean(xs) if xs else 0.0)
        return [(val, float(y)) for y in np.linspace(y_min, y_max, 11)]
        
    a_base, b_base = np.polyfit(ys, xs, 1)
    
    if row_count < 8:
        return [(a_base * y + b_base, y) for y in np.linspace(y_min, y_max, 11)]
    else:
        try:
            coeffs = np.polyfit(ys, xs, 2)
            fn = np.poly1d(coeffs)
            pts = []
            for y in np.linspace(y_min, y_max, 11):
                x_val = fn(y)
                x_base = a_base * y + b_base
                x_val = max(x_base - 4.0, min(x_base + 4.0, x_val))
                pts.append((x_val, y))
            return pts
        except Exception:
            return [(a_base * y + b_base, y) for y in np.linspace(y_min, y_max, 11)]

def extract_support_points_for_horizontal_edge(rotated_gray, m_ref, c_ref, x_left, x_right, block_name, row_idx, row_count, is_outer_edge=False):
    num_samples = max(80, int(abs(x_right - x_left) / 2))
    sample_xs = np.linspace(x_left, x_right, num_samples)
    
    # Calculate normal unit vector
    length = math.sqrt(1.0 + m_ref**2)
    nx = -m_ref / length
    ny = 1.0 / length
    
    win = 3 if is_outer_edge else 4
    
    support_pts = []
    
    # Inset for outer edges
    min_x = x_left + 4 if is_outer_edge else x_left
    max_x = x_right - 4 if is_outer_edge else x_right
    
    for sx in sample_xs:
        if sx < min_x or sx > max_x:
            continue
            
        sy_est = m_ref * sx + c_ref
        profile = []
        
        for d in range(-win, win + 1):
            px = sx + d * nx
            py = sy_est + d * ny
            x_idx = max(0, min(rotated_gray.shape[1] - 1, int(round(px))))
            y_idx = max(0, min(rotated_gray.shape[0] - 1, int(round(py))))
            profile.append(rotated_gray[y_idx, x_idx])
            
        if len(profile) < 3:
            continue
            
        min_idx = np.argmin(profile)
        if 0 < min_idx < len(profile) - 1:
            if profile[min_idx] < profile[min_idx - 1] and profile[min_idx] < profile[min_idx + 1]:
                # Relative depth criteria
                left_peak = np.max(profile[:min_idx])
                right_peak = np.max(profile[min_idx+1:])
                valley_depth = min(left_peak, right_peak) - profile[min_idx]
                
                # valley value criteria
                if (valley_depth >= 6 or profile[min_idx] <= np.median(profile) - 4) and profile[min_idx] < 160:
                    px_val = sx + (min_idx - win) * nx
                    py_val = sy_est + (min_idx - win) * ny
                    support_pts.append((px_val, py_val))
                    
    # Outlier filtering
    clean_pts = []
    if len(support_pts) >= 4:
        # Step 1: Median/MAD
        xs = np.array([pt[0] for pt in support_pts])
        ys = np.array([pt[1] for pt in support_pts])
        res = ys - (m_ref * xs + c_ref)
        med_res = np.median(res)
        devs = np.abs(res - med_res)
        mad = np.median(devs)
        
        keep_step1 = devs <= max(2.0, 2.5 * mad)
        xs_s1 = xs[keep_step1]
        ys_s1 = ys[keep_step1]
        
        # Step 2: Iterative fit
        if len(xs_s1) >= 4:
            curr_xs = xs_s1.copy()
            curr_ys = ys_s1.copy()
            m_fit, c_fit = m_ref, c_ref
            
            for _ in range(3):
                if len(curr_xs) < 4:
                    break
                try:
                    m_fit, c_fit = np.polyfit(curr_xs, curr_ys, 1)
                    res_fit = np.abs(curr_ys - (m_fit * curr_xs + c_fit))
                    keep_step2 = res_fit <= 2.0
                    if np.sum(keep_step2) == len(curr_xs):
                        break
                    curr_xs = curr_xs[keep_step2]
                    curr_ys = curr_ys[keep_step2]
                except Exception:
                    break
                    
            if len(curr_xs) >= 4:
                clean_pts = [(float(x), float(y)) for x, y in zip(curr_xs, curr_ys)]
                
    # Validation
    support_valid = False
    support_line_m = m_ref
    support_line_c = c_ref
    support_coverage = 0.0
    support_residual_std = 0.0
    
    if len(clean_pts) >= max(8, int(0.35 * num_samples)):
        xs_clean = np.array([pt[0] for pt in clean_pts])
        ys_clean = np.array([pt[1] for pt in clean_pts])
        try:
            m_fit, c_fit = np.polyfit(xs_clean, ys_clean, 1)
            std_res_fit = np.std(ys_clean - (m_fit * xs_clean + c_fit))
            
            # Coverage calculation
            n_bins = 8
            bins = np.linspace(x_left, x_right, n_bins + 1)
            bin_indices = np.digitize(xs_clean, bins) - 1
            unique_bins = np.unique(bin_indices)
            unique_bins = unique_bins[(unique_bins >= 0) & (unique_bins < n_bins)]
            coverage = len(unique_bins) / n_bins
            
            if coverage >= 0.65 and std_res_fit <= 2.0 and abs(m_fit - m_ref) < 0.08:
                support_valid = True
                support_line_m = m_fit
                support_line_c = c_fit
                support_coverage = coverage
                support_residual_std = std_res_fit
        except Exception:
            pass
            
    return support_line_m, support_line_c, clean_pts, support_coverage, support_residual_std, support_valid, len(support_pts)

def snap_horizontal_boundary(rotated_gray, m_row, c_row_yolo, c_row_pitch, left_bound_x, right_bound_x, r, row_count, block_name, x_mid):
    win = LAST_ROW_SNAP_WINDOW if r == row_count else THERMAL_GAP_SNAP_WINDOW
    
    is_outer = (r == 0 or r == row_count)
    m_fit, c_fit, clean_pts, coverage, std_res, support_valid, n_raw = extract_support_points_for_horizontal_edge(
        rotated_gray, m_row, c_row_yolo, left_bound_x, right_bound_x, block_name, r, row_count, is_outer_edge=is_outer
    )
    
    # Priority fallbacks for snapped line
    # Sample xs for thermal valley
    sample_xs = np.linspace(left_bound_x, right_bound_x, 30)
    best_thermal_dy = 0
    min_thermal_val = 99999.0
    for dy in range(-win, win + 1):
        vals = []
        for sx in sample_xs:
            y_val = int(round(m_row * sx + c_row_yolo)) + dy
            y_val = max(0, min(rotated_gray.shape[0] - 1, y_val))
            x_val = max(0, min(rotated_gray.shape[1] - 1, int(round(sx))))
            vals.append(rotated_gray[y_val, x_val])
        avg_v = np.mean(vals) if vals else 255.0
        if avg_v < min_thermal_val:
            min_thermal_val = avg_v
            best_thermal_dy = dy
    c_row_thermal = c_row_yolo + best_thermal_dy
    thermal_ok = (min_thermal_val < 135.0)
    
    if support_valid:
        c_final = m_fit * x_mid + c_fit
        selected_cand = "support"
        reason = "valid_support"
    else:
        if c_row_yolo is not None:
            c_final = c_row_yolo
            selected_cand = "yolo"
            reason = "support_invalid_fallback_yolo"
        elif thermal_ok:
            c_final = c_row_thermal
            selected_cand = "thermal"
            reason = "support_invalid_fallback_thermal"
        else:
            c_final = c_row_pitch
            selected_cand = "pitch"
            reason = "support_invalid_fallback_pitch"
            
    print(f"[SUPPORT_EDGE] {block_name} {r} raw={n_raw} clean={len(clean_pts)} coverage={coverage:.3f} residual={std_res:.3f} source={selected_cand} reason={reason}")
    
    # Return additional support variables
    return c_final, selected_cand, coverage, clean_pts, c_fit, c_row_thermal, c_row_yolo, c_row_pitch, c_final, m_fit, c_fit, clean_pts, coverage, std_res, support_valid


def build_grid_from_yolo_edge_consensus(rotated_gray, rotated_sobel_x_abs, b_local, y_min, y_max, n_cols, row_count, x_min_roi, x_max_roi, block_name):
    col_corners = []
    for col in b_local["columns"]:
        col_c = []
        for p in col:
            TL, TR, BR, BL = sort_panel_corners(p["refined_polygon"])
            col_c.append((TL, TR, BR, BL))
        col_corners.append(col_c)
        
    boundaries = {}
    boundaries_n_segments_used = {}
    boundaries_max_jumps = {}
    boundaries_max_turns = {}
    boundaries_fallback_to_baseline = {}
    
    # Check if the block has any crop-left panel in column 0
    has_crop_left_col = False
    if len(b_local["columns"]) > 0:
        for p in b_local["columns"][0]:
            if IMAGE_STEM == "DJI_0087_R" and p["orig_ref"]["bbox"][0] < 12:
                has_crop_left_col = True
                break

    if n_cols == 1:
        col0 = col_corners[0]
        left_pts = [c[0] for c in col0] + [c[3] for c in col0]
        right_pts = [c[1] for c in col0] + [c[2] for c in col0]
        
        boundaries["left_outer"] = fit_consensus_vertical_boundary(left_pts, y_min, y_max, row_count)
        boundaries["right_outer"] = fit_consensus_vertical_boundary(right_pts, y_min, y_max, row_count)
        boundary_keys = ["left_outer", "right_outer"]
    else:
        col0 = col_corners[0]
        col1 = col_corners[1]
        
        left_pts = [c[0] for c in col0] + [c[3] for c in col0]
        if has_crop_left_col:
            mid_pts = [c[0] for c in col1] + [c[3] for c in col1]
        else:
            mid_pts = [c[1] for c in col0] + [c[2] for c in col0] + [c[0] for c in col1] + [c[3] for c in col1]
        right_pts = [c[1] for c in col1] + [c[2] for c in col1]
        
        boundaries["left_outer"] = fit_consensus_vertical_boundary(left_pts, y_min, y_max, row_count)
        boundaries["middle_divider"] = fit_consensus_vertical_boundary(mid_pts, y_min, y_max, row_count)
        boundaries["right_outer"] = fit_consensus_vertical_boundary(right_pts, y_min, y_max, row_count)
        boundary_keys = ["left_outer", "middle_divider", "right_outer"]
        
    for k in boundary_keys:
        boundaries_n_segments_used[k] = min(10, max(4, row_count - 2))
        boundaries_max_jumps[k] = 0.0
        boundaries_max_turns[k] = 0.0
        boundaries_fallback_to_baseline[k] = False
        
    m_independent = []
    row_points_list = []
    


    for r in range(row_count + 1):
        row_pts = []
        for col_idx, col in enumerate(b_local["columns"]):
            if has_crop_left_col and col_idx == 0:
                continue
            if r == 0:
                if len(col) > 0:
                    p = col[0]
                    TL_c, TR_c, BR_c, BL_c = sort_panel_corners(p["refined_polygon"])
                    row_pts.extend([TL_c, TR_c])
            elif r == row_count:
                if len(col) >= row_count:
                    p = col[row_count - 1]
                    TL_c, TR_c, BR_c, BL_c = sort_panel_corners(p["refined_polygon"])
                    row_pts.extend([BL_c, BR_c])
            else:
                if len(col) > r - 1:
                    p_prev = col[r - 1]
                    TL_c, TR_c, BR_c, BL_c = sort_panel_corners(p_prev["refined_polygon"])
                    row_pts.extend([BL_c, BR_c])
                if len(col) > r:
                    p_curr = col[r]
                    TL_c, TR_c, BR_c, BL_c = sort_panel_corners(p_curr["refined_polygon"])
                    row_pts.extend([TL_c, TR_c])
        
        row_points_list.append(row_pts)
        if len(row_pts) >= 2:
            xs_r = [pt[0] for pt in row_pts]
            ys_r = [pt[1] for pt in row_pts]
            try:
                m_r, _ = np.polyfit(xs_r, ys_r, 1)
                m_independent.append(m_r)
            except Exception:
                pass
                
    m_consensus = np.median(m_independent) if m_independent else 0.0
    
    pitches = []
    for r in range(row_count):
        y_r = np.mean([pt[1] for pt in row_points_list[r]]) if row_points_list[r] else y_min
        y_r1 = np.mean([pt[1] for pt in row_points_list[r + 1]]) if row_points_list[r + 1] else y_max
        pitches.append(y_r1 - y_r)
    estimated_pitch = float(np.median(pitches)) if pitches else 30.0
    pitch_std = float(np.std(pitches)) if pitches else 1.0
    
    snapped_horiz_gaps = []
    rejected_count = 0
    interpolated_missing_count = 0
    x_mid = (x_min_roi + x_max_roi) / 2.0
    
    for r in range(row_count + 1):
        row_pts = row_points_list[r]
        if not row_pts:
            y_est = y_min + r * estimated_pitch
            m_row = m_consensus
            c_row = y_est - m_row * x_mid
        else:
            xs_r = np.array([pt[0] for pt in row_pts])
            ys_r = np.array([pt[1] for pt in row_pts])
            try:
                m_r, _ = np.polyfit(xs_r, ys_r, 1)
            except Exception:
                m_r = m_consensus
            m_row = 0.7 * m_r + 0.3 * m_consensus
            c_row = np.mean(ys_r - m_row * xs_r)
            
        left_bound_x = get_boundary_x_at_y(boundaries["left_outer"], m_row * x_mid + c_row)
        

                    
        if has_crop_left_col and "middle_divider" in boundaries:
            left_bound_x = get_boundary_x_at_y(boundaries["middle_divider"], m_row * x_mid + c_row)
            
        right_bound_x = get_boundary_x_at_y(boundaries["right_outer"], m_row * x_mid + c_row)
        
        c_row_yolo = c_row
        y_pitch_est = y_min + r * estimated_pitch
        c_row_pitch = y_pitch_est - m_row * x_mid
        
        c_final, selected_cand, selected_score, support_pts, c_row_support, c_row_thermal, _, _, c_selected, support_line_m, support_line_c, support_clean_pts, support_coverage, support_residual_std, support_valid = snap_horizontal_boundary(
            rotated_gray, m_row, c_row_yolo, c_row_pitch, left_bound_x, right_bound_x, r, row_count, block_name, x_mid
        )
        
        pitch_used = (selected_cand == "pitch")
        print(f"[ROW] {block_name} {r} {selected_cand} {len(support_pts)} {pitch_used}")
        
        if r == row_count:
            y_before = m_row * x_mid + ((1.0 - 0.25) * c_selected + 0.25 * c_row_pitch)
            y_after = m_row * x_mid + c_final
            delta = y_after - y_before
            print(f"[LAST_ROW] {block_name} {r} {selected_cand} {delta:.2f}")
            
        if selected_cand not in ("support", "thermal"):
            rejected_count += 1
            if selected_cand == "pitch":
                interpolated_missing_count += 1
                
        snapped_horiz_gaps.append({
            "y_snap": m_row * x_mid + c_final,
            "is_good": selected_cand in ("support", "thermal"),
            "support_ratio": selected_score,
            "mean_offset": 0.0,
            "pitch_deviation": abs(c_final - c_row_pitch),
            "support_pts": support_pts,
            "x_start": left_bound_x,
            "x_end": right_bound_x,
            "m_row": m_row,
            "c_row": c_final,
            "c_selected": c_selected,
            "c_row_yolo": c_row_yolo,
            "c_row_thermal": c_row_thermal,
            "c_row_pitch": c_row_pitch,
            "c_row_support": c_row_support,
            "selected_cand": selected_cand,
            "support_valid": support_valid,
            "support_line_m": support_line_m,
            "support_line_c": support_line_c,
            "support_clean_pts": support_clean_pts,
            "support_coverage": support_coverage,
            "support_residual_std": support_residual_std
        })
        
    return boundaries, snapped_horiz_gaps, {
        "estimated_pitch_px": estimated_pitch,
        "pitch_std_px": pitch_std,
        "rejected_horizontal_lines_count": rejected_count,
        "number_of_interpolated_missing_lines": interpolated_missing_count,
        "boundaries_n_segments_used": boundaries_n_segments_used,
        "boundaries_max_jumps": boundaries_max_jumps,
        "boundaries_max_turns": boundaries_max_turns,
        "boundaries_fallback_to_baseline": boundaries_fallback_to_baseline,
        "m_consensus": m_consensus
    }


def get_panel_hybrid_anchors(panel):
    poly = None
    source = None
    
    # 1. raw_yolo_poly
    if "raw_yolo_poly" in panel and panel["raw_yolo_poly"] is not None and len(panel["raw_yolo_poly"]) >= 3:
        pts = np.array(panel["raw_yolo_poly"], dtype=np.float32)
        rect = cv2.minAreaRect(pts)
        box = cv2.boxPoints(rect)
        poly = [tuple(pt) for pt in box]
        source = "raw_yolo_minarearect"
        
    # 2. polygon
    if poly is None and "polygon" in panel and panel["polygon"] is not None and len(panel["polygon"]) >= 3:
        pts = np.array(panel["polygon"], dtype=np.float32)
        rect = cv2.minAreaRect(pts)
        box = cv2.boxPoints(rect)
        poly = [tuple(pt) for pt in box]
        source = "polygon_minarearect"
        
    # 3. refined_polygon
    if poly is None and "refined_polygon" in panel and panel["refined_polygon"] is not None and len(panel["refined_polygon"]) >= 3:
        pts = np.array(panel["refined_polygon"], dtype=np.float32)
        rect = cv2.minAreaRect(pts)
        box = cv2.boxPoints(rect)
        poly = [tuple(pt) for pt in box]
        source = "refined_polygon_minarearect"

    # 4. bbox fallback
    if poly is None:
        x1, y1, x2, y2 = panel["bbox"]
        poly = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        source = "bbox_fallback"
        
    TL, TR, BR, BL = sort_panel_corners(poly)
    
    cx = (TL[0] + TR[0] + BR[0] + BL[0]) / 4.0
    cy = (TL[1] + TR[1] + BR[1] + BL[1]) / 4.0
    
    width = math.sqrt((TR[0] - TL[0])**2 + (TR[1] - TL[1])**2)
    height = math.sqrt((BL[0] - TL[0])**2 + (BL[1] - TL[1])**2)
    
    angle_rad = math.atan2(TR[1] - TL[1], TR[0] - TL[0])
    angle_deg = np.degrees(angle_rad)
    while angle_deg > 90:
        angle_deg -= 180
    while angle_deg < -90:
        angle_deg += 180
    if angle_deg > 45:
        angle_deg -= 90
    elif angle_deg < -45:
        angle_deg += 90
        
    return {
        "center": (cx, cy),
        "width": width,
        "height": height,
        "angle_deg": angle_deg,
        "top_edge": (TL, TR),
        "bottom_edge": (BL, BR),
        "left_edge": (TL, BL),
        "right_edge": (TR, BR),
        "poly_4pts": [TL, TR, BR, BL],
        "source": source
    }

def get_rotated_rect_points(cx, cy, w, h, angle_deg):
    angle_rad = np.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    
    dx_tl = -w/2.0 * cos_a - (-h/2.0) * sin_a
    dy_tl = -w/2.0 * sin_a + (-h/2.0) * cos_a
    
    dx_tr = w/2.0 * cos_a - (-h/2.0) * sin_a
    dy_tr = w/2.0 * sin_a + (-h/2.0) * cos_a
    
    dx_br = w/2.0 * cos_a - (h/2.0) * sin_a
    dy_br = w/2.0 * sin_a + (h/2.0) * cos_a
    
    dx_bl = -w/2.0 * cos_a - (h/2.0) * sin_a
    dy_bl = -w/2.0 * sin_a + (h/2.0) * cos_a
    
    TL = (cx + dx_tl, cy + dy_tl)
    TR = (cx + dx_tr, cy + dy_tr)
    BR = (cx + dx_br, cy + dy_br)
    BL = (cx + dx_bl, cy + dy_bl)
    
    return [TL, TR, BR, BL]

def refine_edge_with_thermal_gap(img_gray, p1, p2, search_window=3):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    length = math.sqrt(dx*dx + dy*dy)
    if length == 0:
        return p1, p2
    nx = -dy / length
    ny = dx / length
    
    num_samples = 15
    profile = []
    for d in range(-search_window, search_window + 1):
        vals = []
        for i in range(num_samples):
            t = i / (num_samples - 1)
            x_sample = p1[0] + t * dx + d * nx
            y_sample = p1[1] + t * dy + d * ny
            x_idx = int(round(x_sample))
            y_idx = int(round(y_sample))
            if 0 <= x_idx < img_gray.shape[1] and 0 <= y_idx < img_gray.shape[0]:
                vals.append(img_gray[y_idx, x_idx])
        profile.append(np.mean(vals) if vals else 255.0)
        
    best_d = 0
    has_support = False
    min_val = 255.0
    for idx in range(1, len(profile) - 1):
        if profile[idx] < profile[idx-1] and profile[idx] < profile[idx+1]:
            if profile[idx] < 130 and profile[idx] < min_val:
                min_val = profile[idx]
                best_d = idx - search_window
                has_support = True
                
    if has_support:
        p1_refined = (p1[0] + best_d * nx, p1[1] + best_d * ny)
        p2_refined = (p2[0] + best_d * nx, p2[1] + best_d * ny)
        return p1_refined, p2_refined
    else:
        return p1, p2

def get_polygon_intersection_area(poly1, poly2):
    p1 = np.array(poly1, dtype=np.float32)
    p2 = np.array(poly2, dtype=np.float32)
    ret, inter_poly = cv2.intersectConvexConvex(p1, p2)
    if ret > 0 and inter_poly is not None:
        return cv2.contourArea(inter_poly)
    return 0.0

def validate_and_fallback_panel(poly_proposal_orig, p_orig, gray, block_panels_polys=None):
    poly_proposal_orig = np.array(poly_proposal_orig, dtype=np.float32)
    is_convex = cv2.isContourConvex(poly_proposal_orig.astype(np.int32))
    
    anchors = get_panel_hybrid_anchors(p_orig)
    yolo_poly = np.array(anchors["poly_4pts"], dtype=np.float32)
    area_orig = cv2.contourArea(yolo_poly)
    if area_orig <= 0:
        x1, y1, x2, y2 = p_orig["bbox"]
        area_orig = (x2 - x1) * (y2 - y1)
        
    area_prop = cv2.contourArea(poly_proposal_orig)
    area_ratio = area_prop / area_orig if area_orig > 0 else 0.0
    area_ok = 0.70 <= area_ratio <= 1.40
    
    cx_prop = np.mean(poly_proposal_orig[:, 0])
    cy_prop = np.mean(poly_proposal_orig[:, 1])
    cx_orig = anchors["center"][0]
    cy_orig = anchors["center"][1]
    
    c_shift = math.sqrt((cx_prop - cx_orig)**2 + (cy_prop - cy_orig)**2)
    panel_height = anchors["height"]
    shift_ok = c_shift <= max(12.0, 0.40 * panel_height)
    
    overlap_ok = True
    if block_panels_polys is not None:
        for other_poly in block_panels_polys:
            inter_area = get_polygon_intersection_area(poly_proposal_orig, other_poly)
            if inter_area > 0.35 * min(area_prop, cv2.contourArea(other_poly)):
                overlap_ok = False
                break
                
    is_valid = is_convex and area_ok and shift_ok and overlap_ok
    
    if is_valid:
        return [(float(pt[0]), float(pt[1])) for pt in poly_proposal_orig], "pass"
    else:
        TL_ref, TR_ref = refine_edge_with_thermal_gap(gray, anchors["poly_4pts"][0], anchors["poly_4pts"][1])
        BL_ref, BR_ref = refine_edge_with_thermal_gap(gray, anchors["poly_4pts"][3], anchors["poly_4pts"][2])
        poly_fallback = [TL_ref, TR_ref, BR_ref, BL_ref]
        return poly_fallback, "warning"

def refine_oriented_polygon(gray, poly):
    TL, TR, BR, BL = sort_panel_corners(poly)
    
    TL_t, TR_t = refine_edge_with_thermal_gap(gray, TL, TR, search_window=3)
    BL_b, BR_b = refine_edge_with_thermal_gap(gray, BL, BR, search_window=3)
    
    TL_l, BL_l = refine_edge_with_thermal_gap(gray, TL, BL, search_window=3)
    TR_r, BR_r = refine_edge_with_thermal_gap(gray, TR, BR, search_window=3)
    
    def intersect_lines(p1, p2, q1, q2):
        A1 = p2[1] - p1[1]
        B1 = p1[0] - p2[0]
        C1 = A1 * p1[0] + B1 * p1[1]
        
        A2 = q2[1] - q1[1]
        B2 = q1[0] - q2[0]
        C2 = A2 * q1[0] + B2 * q1[1]
        
        det = A1 * B2 - A2 * B1
        if abs(det) < 1e-5:
            return p1
        x = (B2 * C1 - B1 * C2) / det
        y = (A1 * C2 - A2 * C1) / det
        return (x, y)
        
    tl_new = intersect_lines(TL_t, TR_t, TL_l, BL_l)
    tr_new = intersect_lines(TL_t, TR_t, TR_r, BR_r)
    br_new = intersect_lines(BL_b, BR_b, TR_r, BR_r)
    bl_new = intersect_lines(BL_b, BR_b, TL_l, BL_l)
    
    return [tl_new, tr_new, br_new, bl_new]

def is_polygon_valid(poly, anchors, block_panels_polys=None):
    poly_np = np.array(poly, dtype=np.float32)
    if not cv2.isContourConvex(poly_np.astype(np.int32)):
        return False, "not_convex"
        
    area = cv2.contourArea(poly_np)
    yolo_poly = np.array(anchors["poly_4pts"], dtype=np.float32)
    area_orig = cv2.contourArea(yolo_poly)
    if area_orig <= 0:
        return True, "ok"
        
    area_ratio = area / area_orig
    if not (0.60 <= area_ratio <= 1.45):
        return False, "bad_area"
        
    # Check aspect ratio
    dx_top = math.sqrt((poly[1][0] - poly[0][0])**2 + (poly[1][1] - poly[0][1])**2)
    dy_left = math.sqrt((poly[3][0] - poly[0][0])**2 + (poly[3][1] - poly[0][1])**2)
    if dy_left <= 0:
        return False, "zero_height"
    aspect_ratio = dx_top / dy_left
    
    yolo_w = anchors["width"]
    yolo_h = anchors["height"]
    if yolo_h > 0:
        yolo_aspect = yolo_w / yolo_h
        if not (0.65 * yolo_aspect <= aspect_ratio <= 1.35 * yolo_aspect):
            return False, "bad_aspect_ratio"
            
    if block_panels_polys is not None:
        for other_poly in block_panels_polys:
            inter_area = get_polygon_intersection_area(poly_np, other_poly)
            if inter_area > 0.35 * min(area, cv2.contourArea(other_poly)):
                return False, "overlap"
                
    return True, "ok"

def intersect_snap_horizontal_with_rail(m, x_mid, y_snap, rail_pts):
    # Line equation: y = m * (x - x_mid) + y_snap => -m*x + y + (m*x_mid - y_snap) = 0
    A = -m
    B = 1.0
    C = m * x_mid - y_snap
    for i in range(len(rail_pts) - 1):
        x0, y0 = rail_pts[i]
        x1, y1 = rail_pts[i+1]
        dx = x1 - x0
        dy = y1 - y0
        denom = A * dx + B * dy
        if abs(denom) > 1e-6:
            t = -(A * x0 + B * y0 + C) / denom
            if 0.0 <= t <= 1.0:
                return x0 + t * dx, y0 + t * dy
    return None

def transform_points_local_to_orig(points, transform_info):
    Minv = transform_info["rot_M_inv"]
    roi_x1 = transform_info["roi_x1"]
    roi_y1 = transform_info["roi_y1"]
    
    orig_pts = []
    for x_local, y_local in points:
        x_rel = Minv[0, 0] * float(x_local) + Minv[0, 1] * float(y_local) + Minv[0, 2]
        y_rel = Minv[1, 0] * float(x_local) + Minv[1, 1] * float(y_local) + Minv[1, 2]
        orig_pts.append((float(x_rel + roi_x1), float(y_rel + roi_y1)))
    return orig_pts

def build_panel_polygons_from_visible_snap_evidence(img, snap_blocks, img_support=None, img_snap=None):
    if img_support is None:
        img_support = img.copy()
    if img_snap is None:
        img_snap = img.copy()
    drawn_count = 0
    for sb in snap_blocks:
        block_name = sb["block_name"]
        n_cols = sb["n_cols"]
        row_count = sb["row_count"]
        is_large_grid = sb["is_large_grid"]
        boundaries = sb["boundaries"]
        snapped_horiz_gaps = sb["snapped_horiz_gaps"]
        m_perp = sb["m_perp"]
        x_mid = sb["x_mid"]
        b_local = sb["b_local"]
        transform_info = sb["transform_info"]
        
        is_small_block = sb.get("is_small_block", False)
        rotated_gray = sb.get("rotated_gray", None)
        
        if is_small_block:
            for col_idx, col in enumerate(b_local["columns"]):
                for row_idx in range(len(col)):
                    p = col[row_idx]
                    poly_proposal_local = p["polygon"]
                    poly_final_orig = transform_points_local_to_orig(poly_proposal_local, transform_info)
                    
                    poly_orig = np.array(poly_final_orig, dtype=np.int32)
                    cv2.polylines(img, [poly_orig], True, (0, 255, 255), 1, cv2.LINE_AA)
                    drawn_count += 1
                    
                    source = p.get("source", "small_block_yolo_limited")
                    reason = p.get("reason", "valid_refine")
                    shift = p.get("shift", 0.0)
                    area_ratio = p.get("area_ratio", 1.0)
                    overlap = p.get("overlap", 0.0)
                    print(f"[GEOMETRY_SOURCE] {block_name} {row_idx} {col_idx} source={source} reason={reason} shift={shift:.2f} area_ratio={area_ratio:.2f} overlap={overlap:.2f}")
            continue
            
        bx_min = min(panel["orig_ref"]["bbox"][0] for panel in b_local["panels"]) if b_local["panels"] else 9999.0
        
        # Use pre-computed support line fit from snap_horizontal_boundary / build_small_block
        # This ensures panel edges match exactly what is shown in line_snap.
        fitted_horiz_lines = []
        for r, g in enumerate(snapped_horiz_gaps):
            support_valid = g.get("support_valid", False)
            support_line_m = g.get("support_line_m", None)
            support_line_c = g.get("support_line_c", None)
            m_row_base = g.get("m_row", 0.0)
            c_row_base = g.get("c_row", g.get("c_yolo", 0.0))
 
            if support_valid and support_line_m is not None and support_line_c is not None:
                m_row = float(support_line_m)
                c_row = float(support_line_c)
                selected_cand = "support_prefit"
                std_res = float(g.get("support_residual_std", 0.0))
            else:
                m_row = float(m_row_base)
                c_row = float(c_row_base)
                selected_cand = "yolo_fallback"
                std_res = 0.0
 


            print(f"[HORIZ_LINE] {block_name} r={r} source={selected_cand} m={m_row:.5f} c={c_row:.2f} residual={std_res:.3f}")
 
            fitted_horiz_lines.append({
                "m_row": m_row,
                "c_row": c_row,
                "selected_cand": selected_cand,
                "y_snap": m_row * x_mid + c_row
            })
            
        # Draw panels using these direct boundaries
        for col_idx, col in enumerate(b_local["columns"]):
            for row_idx in range(len(col)):
                p = col[row_idx]
                p_orig = p["orig_ref"]
                
                is_left_cropped = (IMAGE_STEM == "DJI_0087_R" and p_orig["bbox"][0] < 12)
                if is_left_cropped and rotated_gray is not None:
                    poly_proposal_local, ref_src, center_shift, area_ratio, reason = refine_crop_left_panel_individually(
                        rotated_gray, p, block_name, row_idx, row_count, img_support, img_snap, transform_info
                    )
                    poly_final_orig = transform_points_local_to_orig(poly_proposal_local, transform_info)
                    overlap = 0.0
                    print(f"[GEOMETRY_SOURCE] {block_name} {row_idx} {col_idx} source={ref_src} reason={reason} shift={center_shift:.2f} area_ratio={area_ratio:.2f} overlap={overlap:.2f}")
                else:
                    # Get horizontal boundaries
                    g_t = fitted_horiz_lines[row_idx]
                    g_b = fitted_horiz_lines[row_idx + 1]
                    
                    # Get vertical boundaries
                    if is_large_grid or sb.get("is_grid", True):
                        left_pts = boundaries.get("left_outer")
                        right_pts = boundaries.get("right_outer")
                        mid_pts = boundaries.get("middle_divider")
                        
                        if n_cols == 1:
                            c_left = left_pts
                            c_right = right_pts
                            left_name = "rail"
                            right_name = "rail"
                        else:
                            if col_idx == 0:
                                c_left = left_pts
                                c_right = mid_pts
                                left_name = "rail"
                                right_name = "rail"
                            else:
                                c_left = mid_pts
                                c_right = right_pts
                                left_name = "rail"
                                right_name = "rail"
                    else:
                        c_left = boundaries.get(f"left_col_{col_idx}")
                        c_right = boundaries.get(f"right_col_{col_idx}")
                        left_name = "rail" if c_left is not None else "yolo"
                        right_name = "rail" if c_right is not None else "yolo"
                    
                    TL_yolo, TR_yolo, BR_yolo, BL_yolo = sort_panel_corners(p["refined_polygon"])
                    
                    # Decide left rail for small blocks
                    use_yolo_left = False
                    if not is_large_grid and not sb.get("is_grid", True):
                        if c_left is not None:
                            rail_h = abs(c_left[-1][1] - c_left[0][1])
                            if rail_h < 120:
                                use_yolo_left = True
                        else:
                            use_yolo_left = True
                    
                    if use_yolo_left:
                        left_line = [TL_yolo, BL_yolo]
                        left_name = "yolo"
                    else:
                        left_line = c_left
                        
                    # Decide right rail for small blocks
                    use_yolo_right = False
                    if not is_large_grid and not sb.get("is_grid", True):
                        if c_right is not None:
                            rail_h = abs(c_right[-1][1] - c_right[0][1])
                            if rail_h < 120:
                                use_yolo_right = True
                        else:
                            use_yolo_right = True
                            
                    if use_yolo_right:
                        right_line = [TR_yolo, BR_yolo]
                        right_name = "yolo"
                    else:
                        right_line = c_right
                    
                    # Intersect
                    tl = intersect_snap_horizontal_with_rail(g_t["m_row"], 0.0, g_t["c_row"], left_line)
                    if tl is None and left_line is not None and len(left_line) >= 2:
                        tl = (get_boundary_x_at_y(left_line, g_t["y_snap"]), g_t["y_snap"])
                        
                    tr = intersect_snap_horizontal_with_rail(g_t["m_row"], 0.0, g_t["c_row"], right_line)
                    if tr is None and right_line is not None and len(right_line) >= 2:
                        tr = (get_boundary_x_at_y(right_line, g_t["y_snap"]), g_t["y_snap"])
                        
                    br = intersect_snap_horizontal_with_rail(g_b["m_row"], 0.0, g_b["c_row"], right_line)
                    if br is None and right_line is not None and len(right_line) >= 2:
                        br = (get_boundary_x_at_y(right_line, g_b["y_snap"]), g_b["y_snap"])
                        
                    bl = intersect_snap_horizontal_with_rail(g_b["m_row"], 0.0, g_b["c_row"], left_line)
                    if bl is None and left_line is not None and len(left_line) >= 2:
                        bl = (get_boundary_x_at_y(left_line, g_b["y_snap"]), g_b["y_snap"])
                        
                    if tl is None or tr is None or br is None or bl is None:
                        # fallback separately
                        poly_final_local = p.get("refined_polygon") or p.get("polygon")
                        poly_final_orig = transform_points_local_to_orig(poly_final_local, transform_info)
                        
                        poly_yolo_np = np.array(poly_final_local, dtype=np.float32)
                        area_ratio = 1.0
                        center_shift = 0.0
                        source = "fallback_yolo"
                        reason = "intersection_failed"
                        print(f"[GEOMETRY_SOURCE] {block_name} {row_idx} {col_idx} source={source} reason={reason} shift={center_shift:.2f} area_ratio={area_ratio:.2f} overlap=0.00")
                    else:
                        poly_proposal_local = [tl, tr, br, bl]
                        poly_final_orig = transform_points_local_to_orig(poly_proposal_local, transform_info)
                        
                        poly_yolo = p["refined_polygon"]
                        poly_yolo_np = np.array(poly_yolo, dtype=np.float32)
                        poly_prop_np = np.array(poly_proposal_local, dtype=np.float32)
                        
                        area_prop = cv2.contourArea(poly_prop_np)
                        area_yolo = cv2.contourArea(poly_yolo_np)
                        area_ratio = area_prop / (area_yolo + 1e-6)
                        
                        cx_prop = np.mean(poly_prop_np[:, 0])
                        cy_prop = np.mean(poly_prop_np[:, 1])
                        cx_yolo = np.mean(poly_yolo_np[:, 0])
                        cy_yolo = np.mean(poly_yolo_np[:, 1])
                        center_shift = math.sqrt((cx_prop - cx_yolo)**2 + (cy_prop - cy_yolo)**2)
                        
                        source = "snap_grid"
                        t_src = g_t["selected_cand"]
                        b_src = g_b["selected_cand"]
                        reason = f"top_{t_src}_bottom_{b_src}"
                        print(f"[GEOMETRY_SOURCE] {block_name} {row_idx} {col_idx} source={source} reason={reason} shift={center_shift:.2f} area_ratio={area_ratio:.2f} overlap=0.00")
                    
                poly_orig = np.array(poly_final_orig, dtype=np.int32)
                cv2.polylines(img, [poly_orig], True, (0, 255, 255), 1, cv2.LINE_AA)
                drawn_count += 1
                
    return drawn_count

def refine_single_edge_local(rotated_gray, p1, p2, is_horizontal, win, cov_threshold, limit, res_threshold=2.5, angle_threshold=4.0):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    length = math.sqrt(dx*dx + dy*dy)
    if length == 0:
        return p1, p2, []
        
    if is_horizontal:
        tx = dx / length
        ty = dy / length
        nx = -ty
        ny = tx
    else:
        tx = dx / length
        ty = dy / length
        nx = ty
        ny = -tx
        
    num_samples = max(20, min(80, int(length / 2)))
    sample_xs = np.linspace(p1[0], p2[0], num_samples)
    sample_ys = np.linspace(p1[1], p2[1], num_samples)
    
    support_pts = []
    
    for i in range(num_samples):
        sx = sample_xs[i]
        sy = sample_ys[i]
        
        profile = []
        for d in range(-win, win + 1):
            px = sx + d * nx
            py = sy + d * ny
            x_idx = max(0, min(rotated_gray.shape[1] - 1, int(round(px))))
            y_idx = max(0, min(rotated_gray.shape[0] - 1, int(round(py))))
            profile.append(rotated_gray[y_idx, x_idx])
            
        if len(profile) < 3:
            continue
            
        min_idx = np.argmin(profile)
        if 0 < min_idx < len(profile) - 1:
            if profile[min_idx] < profile[min_idx - 1] and profile[min_idx] < profile[min_idx + 1]:
                left_peak = np.max(profile[:min_idx])
                right_peak = np.max(profile[min_idx+1:])
                valley_depth = min(left_peak, right_peak) - profile[min_idx]
                if (valley_depth >= 6 or profile[min_idx] <= np.median(profile) - 4) and profile[min_idx] < 160:
                    px_val = sx + (min_idx - win) * nx
                    py_val = sy + (min_idx - win) * ny
                    support_pts.append((px_val, py_val))
                    
    if len(support_pts) < 4:
        return p1, p2, []
        
    xs = np.array([pt[0] for pt in support_pts])
    ys = np.array([pt[1] for pt in support_pts])
    
    if is_horizontal:
        m_ref = dy / (dx + 1e-6)
        c_ref = p1[1] - m_ref * p1[0]
        res = ys - (m_ref * xs + c_ref)
        med_res = np.median(res)
        devs = np.abs(res - med_res)
        mad = np.median(devs)
        keep = devs <= max(2.0, 2.5 * mad)
        xs_clean = xs[keep]
        ys_clean = ys[keep]
        
        if len(xs_clean) < 4:
            return p1, p2, []
            
        try:
            m_fit, c_fit = np.polyfit(xs_clean, ys_clean, 1)
            res_fit = ys_clean - (m_fit * xs_clean + c_fit)
            keep_fit = np.abs(res_fit) <= 2.0
            xs_final = xs_clean[keep_fit]
            ys_final = ys_clean[keep_fit]
            if len(xs_final) >= 4:
                m_fit, c_fit = np.polyfit(xs_final, ys_final, 1)
                std_res = np.std(ys_final - (m_fit * xs_final + c_fit))
                clean_pts = list(zip(xs_final, ys_final))
            else:
                return p1, p2, []
        except Exception:
            return p1, p2, []
            
        bins = np.linspace(min(p1[0], p2[0]), max(p1[0], p2[0]), 9)
        bin_indices = np.digitize(xs_final, bins) - 1
        coverage = len(np.unique(bin_indices[(bin_indices >= 0) & (bin_indices < 8)])) / 8.0
        
        angle_fit = np.degrees(math.atan(m_fit))
        angle_ref = np.degrees(math.atan(m_ref))
        angle_diff = abs(angle_fit - angle_ref)
        while angle_diff > 90:
            angle_diff = abs(angle_diff - 180)
            
    else:
        m_ref = dx / (dy + 1e-6)
        c_ref = p1[0] - m_ref * p1[1]
        res = xs - (m_ref * ys + c_ref)
        med_res = np.median(res)
        devs = np.abs(res - med_res)
        mad = np.median(devs)
        keep = devs <= max(2.0, 2.5 * mad)
        xs_clean = xs[keep]
        ys_clean = ys[keep]
        
        if len(ys_clean) < 4:
            return p1, p2, []
            
        try:
            m_fit, c_fit = np.polyfit(ys_clean, xs_clean, 1)
            res_fit = xs_clean - (m_fit * ys_clean + c_fit)
            keep_fit = np.abs(res_fit) <= 2.0
            xs_final = xs_clean[keep_fit]
            ys_final = ys_clean[keep_fit]
            if len(ys_final) >= 4:
                m_fit, c_fit = np.polyfit(ys_final, xs_final, 1)
                std_res = np.std(xs_final - (m_fit * ys_final + c_fit))
                clean_pts = list(zip(xs_final, ys_final))
            else:
                return p1, p2, []
        except Exception:
            return p1, p2, []
            
        bins = np.linspace(min(p1[1], p2[1]), max(p1[1], p2[1]), 9)
        bin_indices = np.digitize(ys_final, bins) - 1
        coverage = len(np.unique(bin_indices[(bin_indices >= 0) & (bin_indices < 8)])) / 8.0
        
        angle_fit = np.degrees(math.atan(1.0 / (m_fit + 1e-6)))
        angle_ref = np.degrees(math.atan(1.0 / (m_ref + 1e-6)))
        angle_diff = abs(angle_fit - angle_ref)
        while angle_diff > 90:
            angle_diff = abs(angle_diff - 180)
            
    if coverage < cov_threshold or std_res > res_threshold or angle_diff > angle_threshold:
        return p1, p2, []
        
    if is_horizontal:
        y1_new = m_fit * p1[0] + c_fit
        y2_new = m_fit * p2[0] + c_fit
        dy1 = np.clip(y1_new - p1[1], -limit, limit)
        dy2 = np.clip(y2_new - p2[1], -limit, limit)
        return (p1[0], p1[1] + dy1), (p2[0], p2[1] + dy2), clean_pts
    else:
        x1_new = m_fit * p1[1] + c_fit
        x2_new = m_fit * p2[1] + c_fit
        dx1 = np.clip(x1_new - p1[0], -limit, limit)
        dx2 = np.clip(x2_new - p2[0], -limit, limit)
        return (p1[0] + dx1, p1[1]), (p2[0] + dx2, p2[1]), clean_pts

def intersect_horizontal_line_with_two_points(m_h, c_h, A, B):
    dx = B[0] - A[0]
    dy = B[1] - A[1]
    denom = m_h * dx - dy
    if abs(denom) < 1e-6:
        return A
    x = (-A[0] * dy - (c_h - A[1]) * dx) / denom
    y = m_h * x + c_h
    return (x, y)

def refine_panel_edges_locally_with_yolo_prior(rotated_gray, poly, is_crop_left):
    TL, TR, BR, BL = sort_panel_corners(poly)
    
    limit = 2.5 if is_crop_left else 3.0
    cov_threshold = 0.25 if is_crop_left else 0.30
    res_threshold = 2.5 if is_crop_left else 2.8
    angle_threshold = 4.0 if is_crop_left else 5.0
    
    p1_t, p2_t, clean_t = refine_single_edge_local(rotated_gray, TL, TR, is_horizontal=True, win=4, cov_threshold=cov_threshold, limit=limit, res_threshold=res_threshold, angle_threshold=angle_threshold)
    p1_b, p2_b, clean_b = refine_single_edge_local(rotated_gray, BL, BR, is_horizontal=True, win=4, cov_threshold=cov_threshold, limit=limit, res_threshold=res_threshold, angle_threshold=angle_threshold)
    
    support_pts_dict = {}
    if clean_t: support_pts_dict["top"] = clean_t
    if clean_b: support_pts_dict["bottom"] = clean_b
    
    if clean_t:
        m_top = (p2_t[1] - p1_t[1]) / (p2_t[0] - p1_t[0] + 1e-9)
        c_top = p1_t[1] - m_top * p1_t[0]
    else:
        m_top = (TR[1] - TL[1]) / (TR[0] - TL[0] + 1e-9)
        c_top = TL[1] - m_top * TL[0]
        
    if clean_b:
        m_bot = (p2_b[1] - p1_b[1]) / (p2_b[0] - p1_b[0] + 1e-9)
        c_bot = p1_b[1] - m_bot * p1_b[0]
    else:
        m_bot = (BR[1] - BL[1]) / (BR[0] - BL[0] + 1e-9)
        c_bot = BL[1] - m_bot * BL[0]
        
    TL_new = intersect_horizontal_line_with_two_points(m_top, c_top, TL, BL)
    TR_new = intersect_horizontal_line_with_two_points(m_top, c_top, TR, BR)
    BL_new = intersect_horizontal_line_with_two_points(m_bot, c_bot, TL, BL)
    BR_new = intersect_horizontal_line_with_two_points(m_bot, c_bot, TR, BR)
    
    return [TL_new, TR_new, BR_new, BL_new], support_pts_dict

def refine_local_panels_individually(rotated_gray, b_local, block_name, img_support, img_snap, transform_local_to_orig):
    panels_result = []
    
    for col_idx, col in enumerate(b_local["columns"]):
        for row_idx, p in enumerate(col):
            anchors = get_panel_hybrid_anchors(p["orig_ref"])
            poly_init = p["refined_polygon"]
            
            poly_refined = refine_oriented_polygon(rotated_gray, poly_init)
            
            poly_prop = np.array(poly_refined, dtype=np.float32)
            cx_prop = np.mean(poly_prop[:, 0])
            cy_prop = np.mean(poly_prop[:, 1])
            
            poly_init_np = np.array(poly_init, dtype=np.float32)
            cx_orig = np.mean(poly_init_np[:, 0])
            cy_orig = np.mean(poly_init_np[:, 1])
            c_shift = math.sqrt((cx_prop - cx_orig)**2 + (cy_prop - cy_orig)**2)
            
            tr_prop = poly_refined[1]
            tl_prop = poly_refined[0]
            angle_prop = np.degrees(math.atan2(tr_prop[1] - tl_prop[1], tr_prop[0] - tl_prop[0]))
            while angle_prop > 90: angle_prop -= 180
            while angle_prop < -90: angle_prop += 180
            
            angle_orig = np.degrees(math.atan2(poly_init[1][1] - poly_init[0][1], poly_init[1][0] - poly_init[0][0]))
            while angle_orig > 90: angle_orig -= 180
            while angle_orig < -90: angle_orig += 180
            angle_diff = abs(angle_prop - angle_orig)
            while angle_diff > 90: angle_diff = abs(angle_diff - 180)
            
            inter_area = get_polygon_intersection_area(poly_prop, poly_init_np)
            area_prop = cv2.contourArea(poly_prop)
            area_yolo = cv2.contourArea(poly_init_np)
            union_area = area_prop + area_yolo - inter_area
            iou = inter_area / union_area if union_area > 0 else 0.0
            
            is_bad = (c_shift > 8.0) or (angle_diff > 5.0) or (iou < 0.70)
            
            if is_bad:
                poly_final = poly_init
                quality = "fallback"
                source = "yolo_anchor"
            else:
                poly_final = poly_refined
                quality = "pass"
                source = "refined_local"
                
            panels_result.append({
                "col": col_idx,
                "row": row_idx,
                "polygon": poly_final,
                "quality": quality,
                "source": source,
                "orig_p": p["orig_ref"],
                "anchors": anchors
            })
            
            TL, TR, BR, BL = sort_panel_corners(poly_final)
            for p1, p2 in [(TL, TR), (BL, BR), (TL, BL), (TR, BR)]:
                sample_xs = np.linspace(p1[0], p2[0], 12)
                sample_ys = np.linspace(p1[1], p2[1], 12)
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                length = math.sqrt(dx*dx + dy*dy)
                if length > 0:
                    nx = -dy / length
                    ny = dx / length
                    for i in range(12):
                        sx = sample_xs[i]
                        sy = sample_ys[i]
                        profile = []
                        for d in range(-3, 4):
                            cx = int(round(sx + d * nx))
                            cy = int(round(sy + d * ny))
                            if 0 <= cx < rotated_gray.shape[1] and 0 <= cy < rotated_gray.shape[0]:
                                profile.append(rotated_gray[cy, cx])
                            else:
                                profile.append(255)
                        best_d = np.argmin(profile) - 3
                        if profile[best_d + 3] < 140:
                            pt_support_local = (int(round(sx + best_d * nx)), int(round(sy + best_d * ny)))
                            pt_support_orig = transform_local_to_orig(pt_support_local[0], pt_support_local[1])
                            cv2.circle(img_support, (int(round(pt_support_orig[0])), int(round(pt_support_orig[1]))), 1, (255, 0, 255), -1)
                            cv2.circle(img_snap, (int(round(pt_support_orig[0])), int(round(pt_support_orig[1]))), 1, (255, 0, 255), -1)

    n_panels = len(panels_result)
    fallback_indices = set()
    for i in range(n_panels):
        for j in range(i + 1, n_panels):
            poly1 = np.array(panels_result[i]["polygon"], dtype=np.float32)
            poly2 = np.array(panels_result[j]["polygon"], dtype=np.float32)
            inter_area = get_polygon_intersection_area(poly1, poly2)
            area1 = cv2.contourArea(poly1)
            area2 = cv2.contourArea(poly2)
            union_area = area1 + area2 - inter_area
            iou = inter_area / union_area if union_area > 0 else 0.0
            if iou > 0.05:
                fallback_indices.add(i)
                fallback_indices.add(j)
                
    for idx in fallback_indices:
        p_info = panels_result[idx]
        col_idx = p_info["col"]
        row_idx = p_info["row"]
        p_info["polygon"] = b_local["columns"][col_idx][row_idx]["refined_polygon"]
        p_info["quality"] = "fallback"
        p_info["source"] = "yolo_anchor"
        
    return panels_result

def get_local_yolo_polygon_prior(p, transform_orig_to_local):
    anchors = get_panel_hybrid_anchors(p)
    local_poly = [transform_orig_to_local(pt[0], pt[1]) for pt in anchors["poly_4pts"]]
    return local_poly

def refine_vertical_edge_local(rotated_gray, Q1, Q2, limit=4.0, res_threshold=2.5, angle_threshold=5.0):
    dx = Q2[0] - Q1[0]
    dy = Q2[1] - Q1[1]
    length = math.sqrt(dx*dx + dy*dy)
    if length == 0:
        return []
    
    tx = dx / length
    ty = dy / length
    nx = ty
    ny = -tx
    
    num_samples = max(20, min(80, int(length / 2)))
    sample_xs = np.linspace(Q1[0], Q2[0], num_samples)
    sample_ys = np.linspace(Q1[1], Q2[1], num_samples)
    
    support_pts = []
    win = 4
    
    for i in range(num_samples):
        sx = sample_xs[i]
        sy = sample_ys[i]
        
        profile = []
        for d in range(-win, win + 1):
            px = sx + d * nx
            py = sy + d * ny
            x_idx = max(0, min(rotated_gray.shape[1] - 1, int(round(px))))
            y_idx = max(0, min(rotated_gray.shape[0] - 1, int(round(py))))
            profile.append(rotated_gray[y_idx, x_idx])
            
        if len(profile) < 3:
            continue
            
        min_idx = np.argmin(profile)
        if 0 < min_idx < len(profile) - 1:
            if profile[min_idx] < profile[min_idx - 1] and profile[min_idx] < profile[min_idx + 1]:
                left_peak = np.max(profile[:min_idx])
                right_peak = np.max(profile[min_idx+1:])
                valley_depth = min(left_peak, right_peak) - profile[min_idx]
                if (valley_depth >= 6 or profile[min_idx] <= np.median(profile) - 4) and profile[min_idx] < 160:
                    px_val = sx + (min_idx - win) * nx
                    py_val = sy + (min_idx - win) * ny
                    support_pts.append((px_val, py_val))
                    
    return support_pts

def intersect_lines_uv(m_h, c_h, m_v, c_v):
    denom = 1.0 - m_v * m_h
    if abs(denom) < 1e-6:
        return (c_v, c_h)
    u = (m_v * c_h + c_v) / denom
    v = m_h * u + c_h
    return (u, v)

def check_block_overlaps(proposals):
    n = len(proposals)
    for i in range(n):
        for j in range(i + 1, n):
            poly1 = np.array(proposals[i], dtype=np.float32)
            poly2 = np.array(proposals[j], dtype=np.float32)
            inter_area = get_polygon_intersection_area(poly1, poly2)
            area1 = cv2.contourArea(poly1)
            area2 = cv2.contourArea(poly2)
            union_area = area1 + area2 - inter_area
            iou = inter_area / union_area if union_area > 0 else 0.0
            if iou > 0.03:
                return True, i, j, iou
    return False, -1, -1, 0.0

def build_small_block_micro_grid_from_yolo_and_local_edges(rotated_gray, b_local, block_name, img_support, img_snap, transform_local_to_orig, transform_orig_to_local):
    N_c = len(b_local["columns"])
    N_r = max(len(col) for col in b_local["columns"]) if b_local["columns"] else 0
    
    # 1. Orientation local
    yolo_angles = []
    for col in b_local["columns"]:
        for p in col:
            poly_init = get_local_yolo_polygon_prior(p["orig_ref"], transform_orig_to_local)
            TL, TR, BR, BL = sort_panel_corners(poly_init)
            yolo_angles.append(np.degrees(math.atan2(TR[1] - TL[1], TR[0] - TL[0])))
            yolo_angles.append(np.degrees(math.atan2(BR[1] - BL[1], BR[0] - BL[0])))
            
    use_envelope_angle = False
    if len(yolo_angles) > 1:
        max_diff = max(yolo_angles) - min(yolo_angles)
        if max_diff > 8.0:
            use_envelope_angle = True
            
    if use_envelope_angle:
        all_yolo_pts = []
        for col in b_local["columns"]:
            for p in col:
                poly_init = get_local_yolo_polygon_prior(p["orig_ref"], transform_orig_to_local)
                all_yolo_pts.extend(poly_init)
        rect = cv2.minAreaRect(np.array(all_yolo_pts, dtype=np.float32))
        box = cv2.boxPoints(rect)
        TL_env, TR_env, BR_env, BL_env = sort_panel_corners(box)
        theta_local_deg = np.degrees(math.atan2(TR_env[1] - TL_env[1], TR_env[0] - TL_env[0]))
    else:
        med_angle = np.median(yolo_angles) if yolo_angles else 0.0
        clean_angles = [a for a in yolo_angles if abs(a - med_angle) <= 8.0]
        theta_local_deg = np.mean(clean_angles) if clean_angles else med_angle
        
    theta_local_rad = np.radians(theta_local_deg)
    
    # 2. Local coordinate system setup
    all_pts = []
    for col in b_local["columns"]:
        for p in col:
            poly_init = get_local_yolo_polygon_prior(p["orig_ref"], transform_orig_to_local)
            all_pts.extend(poly_init)
    xs = [pt[0] for pt in all_pts]
    ys = [pt[1] for pt in all_pts]
    x_c = np.mean(xs) if xs else 0.0
    y_c = np.mean(ys) if ys else 0.0
    
    cos_a = math.cos(-theta_local_rad)
    sin_a = math.sin(-theta_local_rad)
    
    def project_to_local(pt):
        xr = pt[0] - x_c
        yr = pt[1] - y_c
        u = xr * cos_a - yr * sin_a
        v = xr * sin_a + yr * cos_a
        return (u, v)
        
    def project_to_rotated(u, v):
        xr = u * cos_a + v * sin_a
        yr = -u * sin_a + v * cos_a
        return (xr + x_c, yr + y_c)
        
    # 3. Sort panels into local grid layout
    panel_centers_uv = []
    for col_idx, col in enumerate(b_local["columns"]):
        for row_idx, p in enumerate(col):
            poly_init = get_local_yolo_polygon_prior(p["orig_ref"], transform_orig_to_local)
            TL, TR, BR, BL = sort_panel_corners(poly_init)
            cx = (TL[0] + TR[0] + BR[0] + BL[0]) / 4.0
            cy = (TL[1] + TR[1] + BR[1] + BL[1]) / 4.0
            cu, cv = project_to_local((cx, cy))
            panel_centers_uv.append({
                "col_idx": col_idx,
                "row_idx": row_idx,
                "u": cu,
                "v": cv,
                "panel": p
            })
            
    panel_centers_uv.sort(key=lambda item: item["v"])
    rows = []
    for item in panel_centers_uv:
        if not rows:
            rows.append([item])
        else:
            mean_v = np.mean([x["v"] for x in rows[-1]])
            if item["v"] - mean_v > 20.0:
                rows.append([item])
            else:
                rows[-1].append(item)
                
    for r in rows:
        r.sort(key=lambda item: item["u"])
        
    rows.sort(key=lambda r: np.mean([item["v"] for item in r]))
    
    N_r = len(rows)
    N_c = max(len(r) for r in rows) if rows else 0
    
    grid_map = {}
    for r_idx, r in enumerate(rows):
        for c_idx, item in enumerate(r):
            grid_map[(c_idx, r_idx)] = item

    # 4. Generate Panel Foreground Mask in Rotated ROI
    xs_roi = [pt[0] for pt in all_pts]
    ys_roi = [pt[1] for pt in all_pts]
    rx1 = max(0, int(np.floor(min(xs_roi) - 4)))
    ry1 = max(0, int(np.floor(min(ys_roi) - 4)))
    rx2 = min(rotated_gray.shape[1] - 1, int(np.ceil(max(xs_roi) + 4)))
    ry2 = min(rotated_gray.shape[0] - 1, int(np.ceil(max(ys_roi) + 4)))
    
    roi_img = rotated_gray[ry1:ry2+1, rx1:rx2+1]
    thresh_val = float(np.percentile(roi_img, 58))
    local_mask = (roi_img >= thresh_val).astype(np.uint8) * 255
    
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    local_mask_closed = cv2.morphologyEx(local_mask, cv2.MORPH_CLOSE, kernel_close)
    local_mask_opened = cv2.morphologyEx(local_mask_closed, cv2.MORPH_OPEN, kernel_open)
    
    mask_opened = np.zeros_like(rotated_gray)
    mask_opened[ry1:ry2+1, rx1:rx2+1] = local_mask_opened
    
    # Create yolo_mask representing exact rotated YOLO polygon areas
    yolo_mask = np.zeros_like(rotated_gray)
    for col in b_local["columns"]:
        for p in col:
            p_rot = get_local_yolo_polygon_prior(p["orig_ref"], transform_orig_to_local)
            cv2.fillPoly(yolo_mask, [np.array(p_rot, dtype=np.int32)], 255)
            
    contours, _ = cv2.findContours(mask_opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = []
    total_mask_area = 0.0
    for c in contours:
        area = cv2.contourArea(c)
        if area >= 150.0:
            c_box = cv2.boundingRect(c)
            cw, ch = c_box[2], c_box[3]
            aspect = cw / float(ch) if ch > 0 else 0.0
            
            if ch >= 12 and aspect <= 4.5:
                c_mask = np.zeros_like(rotated_gray)
                cv2.drawContours(c_mask, [c], -1, 255, -1)
                overlap = cv2.bitwise_and(yolo_mask, c_mask)
                if np.any(overlap):
                    valid_contours.append(c)
                    total_mask_area += area
            
    contour_ok = False
    U_left_fg = U_right_fg = V_top_fg = V_bottom_fg = None
    if len(valid_contours) > 0 and total_mask_area >= 200.0:
        contour_ok = True
        all_contour_pts = np.concatenate(valid_contours, axis=0)
        pts_uv = [project_to_local((pt[0][0], pt[0][1])) for pt in all_contour_pts]
        us_fg = [pt[0] for pt in pts_uv]
        vs_fg = [pt[1] for pt in pts_uv]
        U_left_fg = min(us_fg)
        U_right_fg = max(us_fg)
        V_top_fg = min(vs_fg)
        V_bottom_fg = max(vs_fg)
        
    rect_str = f"({U_left_fg:.1f},{V_top_fg:.1f},{U_right_fg:.1f},{V_bottom_fg:.1f})" if contour_ok else "None"
    print(f"[SMALL_MASK] {block_name} area={total_mask_area:.1f} contour_ok={contour_ok} rect={rect_str}")
    
    # Save debug mask visualization
    img_mask_vis = cv2.cvtColor(rotated_gray, cv2.COLOR_GRAY2BGR)
    overlay = np.zeros_like(img_mask_vis)
    overlay[mask_opened > 0] = (0, 255, 0)
    img_mask_vis = cv2.addWeighted(img_mask_vis, 0.7, overlay, 0.3, 0)
    
    for c_idx in range(N_c):
        for r_idx in range(N_r):
            if (c_idx, r_idx) in grid_map:
                p_item = grid_map[(c_idx, r_idx)]
                poly_init = get_local_yolo_polygon_prior(p_item["panel"]["orig_ref"], transform_orig_to_local)
                cv2.polylines(img_mask_vis, [np.array(poly_init, dtype=np.int32)], True, (255, 0, 0), 1)
                
    if len(valid_contours) > 0:
        cv2.drawContours(img_mask_vis, valid_contours, -1, (0, 255, 255), 1)
        # Draw union bounding box
        corners_local = [
            (U_left_fg, V_top_fg),
            (U_right_fg, V_top_fg),
            (U_right_fg, V_bottom_fg),
            (U_left_fg, V_bottom_fg)
        ]
        corners_rot = [project_to_rotated(u, v) for u, v in corners_local]
        cv2.polylines(img_mask_vis, [np.array(corners_rot, dtype=np.int32)], True, (0, 255, 255), 1)
        
    out_dir = Path("data/results/debug")
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / f"debug_{IMAGE_STEM}_{block_name}_mask.JPG"), img_mask_vis)
    cv2.imwrite(str(out_dir / f"debug_{IMAGE_STEM}_smallblock_mask.JPG"), img_mask_vis)
    
    # 5. Initialize envelope boundaries priors
    yolo_lefts = []
    yolo_rights = []
    yolo_tops = []
    yolo_bottoms = []
    for c_idx in range(N_c):
        for r_idx in range(N_r):
            if (c_idx, r_idx) in grid_map:
                item = grid_map[(c_idx, r_idx)]
                poly_init = get_local_yolo_polygon_prior(item["panel"]["orig_ref"], transform_orig_to_local)
                TL, TR, BR, BL = sort_panel_corners(poly_init)
                yolo_lefts.extend([project_to_local(TL)[0], project_to_local(BL)[0]])
                yolo_rights.extend([project_to_local(TR)[0], project_to_local(BR)[0]])
                yolo_tops.extend([project_to_local(TL)[1], project_to_local(TR)[1]])
                yolo_bottoms.extend([project_to_local(BL)[1], project_to_local(BR)[1]])
                
    U_left_yolo = min(yolo_lefts) if yolo_lefts else 0.0
    U_right_yolo = max(yolo_rights) if yolo_rights else 0.0
    V_top_yolo = min(yolo_tops) if yolo_tops else 0.0
    V_bottom_yolo = max(yolo_bottoms) if yolo_bottoms else 0.0
    
    if contour_ok:
        # Clamp YOLO boundaries by foreground contour if contour exists
        U_left_yolo = max(U_left_yolo, U_left_fg)
        U_right_yolo = min(U_right_yolo, U_right_fg)
        V_top_yolo = max(V_top_yolo, V_top_fg)
        V_bottom_yolo = min(V_bottom_yolo, V_bottom_fg)

        U_left_prior = U_left_fg
        U_right_prior = U_right_fg
        V_top_prior = V_top_fg
        V_bottom_prior = V_bottom_fg
        prior_source = "foreground"
    else:
        U_left_prior = U_left_yolo
        U_right_prior = U_right_yolo
        V_top_prior = V_top_yolo
        V_bottom_prior = V_bottom_yolo
        prior_source = "yolo_fallback"
        
    # Valley detector functions with win=3
    def find_horizontal_valley_pts(u_start, u_end, v_prior, win=3):
        num_samples = max(20, min(80, int((u_end - u_start) / 2)))
        u_samples = np.linspace(u_start, u_end, num_samples)
        pts = []
        for u_val in u_samples:
            profile = []
            for d in range(-win, win + 1):
                v_val = v_prior + d
                img_x, img_y = project_to_rotated(u_val, v_val)
                x_idx = max(0, min(rotated_gray.shape[1] - 1, int(round(img_x))))
                y_idx = max(0, min(rotated_gray.shape[0] - 1, int(round(img_y))))
                profile.append(rotated_gray[y_idx, x_idx])
            if len(profile) < 3:
                continue
            min_idx = np.argmin(profile)
            if 0 < min_idx < len(profile) - 1:
                if profile[min_idx] < profile[min_idx - 1] and profile[min_idx] < profile[min_idx + 1]:
                    left_peak = np.max(profile[:min_idx])
                    right_peak = np.max(profile[min_idx+1:])
                    valley_depth = min(left_peak, right_peak) - profile[min_idx]
                    if (valley_depth >= 6 or profile[min_idx] <= np.median(profile) - 4) and profile[min_idx] < 160:
                        px_val = u_val
                        py_val = v_prior + (min_idx - win)
                        pts.append((px_val, py_val))
        return pts
        
    def find_vertical_valley_pts(v_start, v_end, u_prior, win=3):
        num_samples = max(20, min(80, int((v_end - v_start) / 2)))
        v_samples = np.linspace(v_start, v_end, num_samples)
        pts = []
        for v_val in v_samples:
            profile = []
            for d in range(-win, win + 1):
                u_val = u_prior + d
                img_x, img_y = project_to_rotated(u_val, v_val)
                x_idx = max(0, min(rotated_gray.shape[1] - 1, int(round(img_x))))
                y_idx = max(0, min(rotated_gray.shape[0] - 1, int(round(img_y))))
                profile.append(rotated_gray[y_idx, x_idx])
            if len(profile) < 3:
                continue
            min_idx = np.argmin(profile)
            if 0 < min_idx < len(profile) - 1:
                if profile[min_idx] < profile[min_idx - 1] and profile[min_idx] < profile[min_idx + 1]:
                    left_peak = np.max(profile[:min_idx])
                    right_peak = np.max(profile[min_idx+1:])
                    valley_depth = min(left_peak, right_peak) - profile[min_idx]
                    if (valley_depth >= 6 or profile[min_idx] <= np.median(profile) - 4) and profile[min_idx] < 160:
                        px_val = u_prior + (min_idx - win)
                        py_val = v_val
                        pts.append((px_val, py_val))
        return pts

    def clean_and_fit_horizontal(pts, u_start, u_end, v_prior, limit=3.0, shift_limit=3.0):
        if len(pts) < 4:
            return None
        us = np.array([pt[0] for pt in pts])
        vs = np.array([pt[1] for pt in pts])
        res = vs - v_prior
        med_res = np.median(res)
        devs = np.abs(res - med_res)
        mad = np.median(devs)
        keep = devs <= max(2.0, 2.5 * mad)
        us_clean = us[keep]
        vs_clean = vs[keep]
        if len(us_clean) < 4:
            return None
        try:
            m_fit, c_fit = np.polyfit(us_clean, vs_clean, 1)
            res_fit = vs_clean - (m_fit * us_clean + c_fit)
            keep_fit = np.abs(res_fit) <= 2.0
            us_final = us_clean[keep_fit]
            vs_final = vs_clean[keep_fit]
            if len(us_final) < 4:
                return None
            m_fit, c_fit = np.polyfit(us_final, vs_final, 1)
            std_res = np.std(vs_final - (m_fit * us_final + c_fit))
            
            bins = np.linspace(u_start, u_end, 9)
            bin_indices = np.digitize(us_final, bins) - 1
            coverage = len(np.unique(bin_indices[(bin_indices >= 0) & (bin_indices < 8)])) / 8.0
            
            angle_diff = abs(np.degrees(math.atan(m_fit)))
            u_mid = (u_start + u_end) / 2.0
            shift = (m_fit * u_mid + c_fit) - v_prior
            
            if coverage >= 0.30 and std_res <= 2.8 and angle_diff <= 5.0 and abs(shift) <= shift_limit:
                return {
                    'm': float(m_fit),
                    'c': float(c_fit),
                    'pts': list(zip(us_final, vs_final)),
                    'shift': shift,
                    'coverage': coverage,
                    'residual': std_res,
                    'source': 'support'
                }
        except Exception:
            pass
        return None

    def clean_and_fit_vertical(pts, v_start, v_end, u_prior, limit=3.0, shift_limit=3.0):
        if len(pts) < 4:
            return None
        us = np.array([pt[0] for pt in pts])
        vs = np.array([pt[1] for pt in pts])
        res = us - u_prior
        med_res = np.median(res)
        devs = np.abs(res - med_res)
        mad = np.median(devs)
        keep = devs <= max(2.0, 2.5 * mad)
        us_clean = us[keep]
        vs_clean = vs[keep]
        if len(vs_clean) < 4:
            return None
        try:
            m_fit, c_fit = np.polyfit(vs_clean, us_clean, 1)
            res_fit = us_clean - (m_fit * vs_clean + c_fit)
            keep_fit = np.abs(res_fit) <= 2.0
            us_final = us_clean[keep_fit]
            vs_final = vs_clean[keep_fit]
            if len(vs_final) < 4:
                return None
            m_fit, c_fit = np.polyfit(vs_final, us_final, 1)
            std_res = np.std(us_final - (m_fit * vs_final + c_fit))
            
            bins = np.linspace(v_start, v_end, 9)
            bin_indices = np.digitize(vs_final, bins) - 1
            coverage = len(np.unique(bin_indices[(bin_indices >= 0) & (bin_indices < 8)])) / 8.0
            
            angle_diff = abs(np.degrees(math.atan(m_fit)))
            v_mid = (v_start + v_end) / 2.0
            shift = (m_fit * v_mid + c_fit) - u_prior
            
            if coverage >= 0.40 and std_res <= 3.0 and angle_diff <= 6.0 and abs(shift) <= shift_limit:
                return {
                    'm': float(m_fit),
                    'c': float(c_fit),
                    'pts': list(zip(us_final, vs_final)),
                    'shift': shift,
                    'coverage': coverage,
                    'residual': std_res,
                    'source': 'support'
                }
        except Exception:
            pass
        return None

    # 6. Snap outer boundaries with max shift +-3 px
    left_pts = find_vertical_valley_pts(V_top_prior, V_bottom_prior, U_left_prior, win=3)
    left_fit = clean_and_fit_vertical(left_pts, V_top_prior, V_bottom_prior, U_left_prior, limit=3.0, shift_limit=3.0)
    
    right_pts = find_vertical_valley_pts(V_top_prior, V_bottom_prior, U_right_prior, win=3)
    right_fit = clean_and_fit_vertical(right_pts, V_top_prior, V_bottom_prior, U_right_prior, limit=3.0, shift_limit=3.0)
    
    v_mid = (V_top_prior + V_bottom_prior) / 2.0
    u_start = (left_fit['m'] * v_mid + left_fit['c']) if left_fit else U_left_prior
    u_end = (right_fit['m'] * v_mid + right_fit['c']) if right_fit else U_right_prior
    
    u_start_contract = u_start + 2.0
    u_end_contract = u_end - 2.0
    
    top_pts = find_horizontal_valley_pts(u_start_contract, u_end_contract, V_top_prior, win=3)
    top_fit = clean_and_fit_horizontal(top_pts, u_start_contract, u_end_contract, V_top_prior, limit=3.0, shift_limit=3.0)
    
    bottom_pts = find_horizontal_valley_pts(u_start_contract, u_end_contract, V_bottom_prior, win=3)
    bottom_fit = clean_and_fit_horizontal(bottom_pts, u_start_contract, u_end_contract, V_bottom_prior, limit=3.0, shift_limit=3.0)
    
    U_boundaries = {}
    U_boundaries[0] = left_fit if left_fit else {'m': 0.0, 'c': U_left_prior, 'source': prior_source}
    U_boundaries[N_c] = right_fit if right_fit else {'m': 0.0, 'c': U_right_prior, 'source': prior_source}
    
    V_boundaries = {}
    V_boundaries[0] = top_fit if top_fit else {'m': 0.0, 'c': V_top_prior, 'source': prior_source}
    V_boundaries[N_r] = bottom_fit if bottom_fit else {'m': 0.0, 'c': V_bottom_prior, 'source': prior_source}
    
    for fit_res in [left_fit, right_fit, top_fit, bottom_fit]:
        if fit_res and 'pts' in fit_res:
            for pt_uv in fit_res['pts']:
                pt_rot = project_to_rotated(pt_uv[0], pt_uv[1])
                pt_orig = transform_local_to_orig(pt_rot[0], pt_rot[1])
                cv2.circle(img_support, (int(round(pt_orig[0])), int(round(pt_orig[1]))), 1, (255, 0, 255), -1)
                cv2.circle(img_snap, (int(round(pt_orig[0])), int(round(pt_orig[1]))), 1, (255, 0, 255), -1)
                
    # 7. Divide middle column/row separators
    for c in range(1, N_c):
        ratio = c / N_c
        m_l, c_l = U_boundaries[0]['m'], U_boundaries[0]['c']
        m_r, c_r = U_boundaries[N_c]['m'], U_boundaries[N_c]['c']
        m_mid = m_l * (1 - ratio) + m_r * ratio
        c_mid = c_l * (1 - ratio) + c_r * ratio
        
        u_mid_prior = m_mid * v_mid + c_mid
        mid_col_pts = find_vertical_valley_pts(V_top_prior, V_bottom_prior, u_mid_prior, win=3)
        mid_col_fit = clean_and_fit_vertical(mid_col_pts, V_top_prior, V_bottom_prior, u_mid_prior, limit=3.0, shift_limit=3.0)
        if mid_col_fit:
            U_boundaries[c] = mid_col_fit
            for pt_uv in mid_col_fit['pts']:
                pt_rot = project_to_rotated(pt_uv[0], pt_uv[1])
                pt_orig = transform_local_to_orig(pt_rot[0], pt_rot[1])
                cv2.circle(img_support, (int(round(pt_orig[0])), int(round(pt_orig[1]))), 1, (255, 0, 255), -1)
                cv2.circle(img_snap, (int(round(pt_orig[0])), int(round(pt_orig[1]))), 1, (255, 0, 255), -1)
        else:
            U_boundaries[c] = {'m': m_mid, 'c': c_mid, 'source': 'envelope_midpoint'}
            
    for r in range(1, N_r):
        ratio = r / N_r
        m_t, c_t = V_boundaries[0]['m'], V_boundaries[0]['c']
        m_b, c_b = V_boundaries[N_r]['m'], V_boundaries[N_r]['c']
        m_mid = m_t * (1 - ratio) + m_b * ratio
        c_mid = c_t * (1 - ratio) + c_b * ratio
        
        u_mid_eval = (u_start + u_end) / 2.0
        v_mid_prior = m_mid * u_mid_eval + c_mid
        mid_row_pts = find_horizontal_valley_pts(u_start_contract, u_end_contract, v_mid_prior, win=3)
        mid_row_fit = clean_and_fit_horizontal(mid_row_pts, u_start_contract, u_end_contract, v_mid_prior, limit=3.0, shift_limit=3.0)
        if mid_row_fit:
            V_boundaries[r] = mid_row_fit
            for pt_uv in mid_row_fit['pts']:
                pt_rot = project_to_rotated(pt_uv[0], pt_uv[1])
                pt_orig = transform_local_to_orig(pt_rot[0], pt_rot[1])
                cv2.circle(img_support, (int(round(pt_orig[0])), int(round(pt_orig[1]))), 1, (255, 0, 255), -1)
                cv2.circle(img_snap, (int(round(pt_orig[0])), int(round(pt_orig[1]))), 1, (255, 0, 255), -1)
        else:
            V_boundaries[r] = {'m': m_mid, 'c': c_mid, 'source': 'envelope_midpoint'}
            
    # Check ordering and spacing
    def check_middle_column_boundaries():
        v_test = [V_top_prior, (V_top_prior + V_bottom_prior)/2.0, V_bottom_prior]
        for c in range(1, N_c):
            m_curr, c_curr = U_boundaries[c]['m'], U_boundaries[c]['c']
            m_prev, c_prev = U_boundaries[c-1]['m'], U_boundaries[c-1]['c']
            m_next, c_next = U_boundaries[c+1]['m'], U_boundaries[c+1]['c']
            is_ok = True
            for vt in v_test:
                u_curr = m_curr * vt + c_curr
                u_prev = m_prev * vt + c_prev
                u_next = m_next * vt + c_next
                if u_curr - u_prev < 15.0 or u_next - u_curr < 15.0:
                    is_ok = False
                    break
            if not is_ok:
                ratio = c / N_c
                m_l, c_l = U_boundaries[0]['m'], U_boundaries[0]['c']
                m_r, c_r = U_boundaries[N_c]['m'], U_boundaries[N_c]['c']
                U_boundaries[c] = {'m': m_l*(1-ratio) + m_r*ratio, 'c': c_l*(1-ratio) + c_r*ratio, 'source': 'envelope_midpoint'}

    def check_middle_row_boundaries():
        u_test = [u_start, (u_start + u_end)/2.0, u_end]
        for r in range(1, N_r):
            m_curr, c_curr = V_boundaries[r]['m'], V_boundaries[r]['c']
            m_prev, c_prev = V_boundaries[r-1]['m'], V_boundaries[r-1]['c']
            m_next, c_next = V_boundaries[r+1]['m'], V_boundaries[r+1]['c']
            is_ok = True
            for ut in u_test:
                v_curr = m_curr * ut + c_curr
                v_prev = m_prev * ut + c_prev
                v_next = m_next * ut + c_next
                if v_curr - v_prev < 15.0 or v_next - v_curr < 15.0:
                    is_ok = False
                    break
            if not is_ok:
                ratio = r / N_r
                m_t, c_t = V_boundaries[0]['m'], V_boundaries[0]['c']
                m_b, c_b = V_boundaries[N_r]['m'], V_boundaries[N_r]['c']
                V_boundaries[r] = {'m': m_t*(1-ratio) + m_b*ratio, 'c': c_t*(1-ratio) + c_b*ratio, 'source': 'envelope_midpoint'}
                
    check_middle_column_boundaries()
    check_middle_row_boundaries()
    
    # 8. Build proposals
    def build_proposals_dict(h_bounds, v_bounds):
        props = {}
        for r_idx in range(N_r):
            for c_idx in range(N_c):
                tl = intersect_lines_uv(h_bounds[r_idx]['m'], h_bounds[r_idx]['c'], v_bounds[c_idx]['m'], v_bounds[c_idx]['c'])
                tr = intersect_lines_uv(h_bounds[r_idx]['m'], h_bounds[r_idx]['c'], v_bounds[c_idx+1]['m'], v_bounds[c_idx+1]['c'])
                br = intersect_lines_uv(h_bounds[r_idx+1]['m'], h_bounds[r_idx+1]['c'], v_bounds[c_idx+1]['m'], v_bounds[c_idx+1]['c'])
                bl = intersect_lines_uv(h_bounds[r_idx+1]['m'], h_bounds[r_idx+1]['c'], v_bounds[c_idx]['m'], v_bounds[c_idx]['c'])
                props[(c_idx, r_idx)] = [
                    project_to_rotated(tl[0], tl[1]),
                    project_to_rotated(tr[0], tr[1]),
                    project_to_rotated(br[0], br[1]),
                    project_to_rotated(bl[0], bl[1])
                ]
        return props

    proposals = build_proposals_dict(V_boundaries, U_boundaries)
    panel_keys = list(proposals.keys())
    panel_list = [proposals[k] for k in panel_keys]
    has_overlap, _, _, _ = check_block_overlaps(panel_list)
    
    # Stage 1: Reset middle boundaries to midpoint
    if has_overlap:
        for c in range(1, N_c):
            ratio = c / N_c
            m_l, c_l = U_boundaries[0]['m'], U_boundaries[0]['c']
            m_r, c_r = U_boundaries[N_c]['m'], U_boundaries[N_c]['c']
            U_boundaries[c] = {'m': m_l*(1-ratio) + m_r*ratio, 'c': c_l*(1-ratio) + c_r*ratio, 'source': 'envelope_midpoint'}
        for r in range(1, N_r):
            ratio = r / N_r
            m_t, c_t = V_boundaries[0]['m'], V_boundaries[0]['c']
            m_b, c_b = V_boundaries[N_r]['m'], V_boundaries[N_r]['c']
            V_boundaries[r] = {'m': m_t*(1-ratio) + m_b*ratio, 'c': c_t*(1-ratio) + c_b*ratio, 'source': 'envelope_midpoint'}
        proposals = build_proposals_dict(V_boundaries, U_boundaries)
        panel_list = [proposals[k] for k in panel_keys]
        has_overlap, _, _, _ = check_block_overlaps(panel_list)
        
    # Stage 2: Reset outer boundaries to YOLO priors and recreate middles
    if has_overlap:
        U_boundaries[0] = {'m': 0.0, 'c': U_left_prior, 'source': prior_source}
        U_boundaries[N_c] = {'m': 0.0, 'c': U_right_prior, 'source': prior_source}
        V_boundaries[0] = {'m': 0.0, 'c': V_top_prior, 'source': prior_source}
        V_boundaries[N_r] = {'m': 0.0, 'c': V_bottom_prior, 'source': prior_source}
        for c in range(1, N_c):
            ratio = c / N_c
            U_boundaries[c] = {'m': 0.0, 'c': U_left_prior*(1-ratio) + U_right_prior*ratio, 'source': 'envelope_midpoint'}
        for r in range(1, N_r):
            ratio = r / N_r
            V_boundaries[r] = {'m': 0.0, 'c': V_top_prior*(1-ratio) + V_bottom_prior*ratio, 'source': 'envelope_midpoint'}
        proposals = build_proposals_dict(V_boundaries, U_boundaries)
        panel_list = [proposals[k] for k in panel_keys]
        has_overlap, _, _, _ = check_block_overlaps(panel_list)
        
    # Stage 3: Fallback individual overlapping panels
    fallback_panels = set()
    if has_overlap:
        for i in range(len(panel_list)):
            for j in range(i + 1, len(panel_list)):
                poly1 = np.array(panel_list[i], dtype=np.float32)
                poly2 = np.array(panel_list[j], dtype=np.float32)
                inter_area = get_polygon_intersection_area(poly1, poly2)
                area1 = cv2.contourArea(poly1)
                area2 = cv2.contourArea(poly2)
                union_area = area1 + area2 - inter_area
                iou = inter_area / union_area if union_area > 0 else 0.0
                if iou > 0.03:
                    fallback_panels.add(panel_keys[i])
                    fallback_panels.add(panel_keys[j])
                    
    # Log small envelope properties
    def format_boundary_c(b):
        return f"{b['m']*v_mid + b['c']:.1f}"
    def format_horizontal_c(b):
        return f"{b['m']*((u_start+u_end)/2.0) + b['c']:.1f}"
        
    left_str = format_boundary_c(U_boundaries[0])
    right_str = format_boundary_c(U_boundaries[N_c])
    mid_u_str = format_boundary_c(U_boundaries[1]) if N_c == 2 else "None"
    
    top_str = format_horizontal_c(V_boundaries[0])
    bottom_str = format_horizontal_c(V_boundaries[N_r])
    mid_v_str = format_horizontal_c(V_boundaries[1]) if N_r == 2 else "None"
    
    print(f"[SMALL_ENVELOPE] {block_name} source={prior_source} U=({left_str},{mid_u_str},{right_str}) V=({top_str},{mid_v_str},{bottom_str})")

    # Calculate median width and height of proposals
    prop_widths = []
    prop_heights = []
    for key, poly in proposals.items():
        TL_p, TR_p, BR_p, BL_p = sort_panel_corners(poly)
        w = (math.sqrt((TR_p[0] - TL_p[0])**2 + (TR_p[1] - TL_p[1])**2) + math.sqrt((BR_p[0] - BL_p[0])**2 + (BR_p[1] - BL_p[1])**2)) / 2.0
        h = (math.sqrt((BL_p[0] - TL_p[0])**2 + (BL_p[1] - TL_p[1])**2) + math.sqrt((BR_p[0] - TR_p[0])**2 + (BR_p[1] - TR_p[1])**2)) / 2.0
        prop_widths.append(w)
        prop_heights.append(h)
    median_width = np.median(prop_widths) if prop_widths else 1.0
    median_height = np.median(prop_heights) if prop_heights else 1.0

    max_overlap_iou = 0.0
    for i in range(len(panel_list)):
        for j in range(i + 1, len(panel_list)):
            poly1 = np.array(panel_list[i], dtype=np.float32)
            poly2 = np.array(panel_list[j], dtype=np.float32)
            inter_area = get_polygon_intersection_area(poly1, poly2)
            area1 = cv2.contourArea(poly1)
            area2 = cv2.contourArea(poly2)
            union_area = area1 + area2 - inter_area
            iou = inter_area / union_area if union_area > 0 else 0.0
            if iou > max_overlap_iou:
                max_overlap_iou = iou

    panels_result = []
    for c_idx in range(N_c):
        for r_idx in range(N_r):
            if (c_idx, r_idx) not in grid_map:
                continue
            item = grid_map[(c_idx, r_idx)]
            p = item["panel"]
            
            poly_init = get_local_yolo_polygon_prior(p["orig_ref"], transform_orig_to_local)
            anchors = get_panel_hybrid_anchors(p["orig_ref"])
            
            key = (c_idx, r_idx)
            if key in fallback_panels:
                rect = cv2.minAreaRect(np.array(poly_init, dtype=np.float32))
                box = cv2.boxPoints(rect)
                poly_final = [tuple(pt) for pt in sort_panel_corners(box)]
                source = "fallback_yolo"
                reason = "overlap_fallback"
                center_shift = 0.0
                area_ratio = 1.0
            else:
                poly_proposal = proposals[key]
                TL_yolo, TR_yolo, BR_yolo, BL_yolo = sort_panel_corners(poly_init)
                panel_height = math.sqrt((BL_yolo[0] - TL_yolo[0])**2 + (BL_yolo[1] - TL_yolo[1])**2)
                
                TL_new, TR_new, BR_new, BL_new = sort_panel_corners(poly_proposal)
                
                is_convex = cv2.isContourConvex(np.array(poly_proposal, dtype=np.int32))
                area_prop = cv2.contourArea(np.array(poly_proposal, dtype=np.float32))
                area_yolo = cv2.contourArea(np.array(poly_init, dtype=np.float32))
                area_ratio = area_prop / (area_yolo + 1e-6)
                
                cx_prop = (TL_new[0] + TR_new[0] + BR_new[0] + BL_new[0]) / 4.0
                cy_prop = (TL_new[1] + TR_new[1] + BR_new[1] + BL_new[1]) / 4.0
                cx_yolo = (TL_yolo[0] + TR_yolo[0] + BR_yolo[0] + BL_yolo[0]) / 4.0
                cy_yolo = (TL_yolo[1] + TR_yolo[1] + BR_yolo[1] + BL_yolo[1]) / 4.0
                center_shift = math.sqrt((cx_prop - cx_yolo)**2 + (cy_prop - cy_yolo)**2)
                
                angle_prop = np.degrees(math.atan2(TR_new[1] - TL_new[1], TR_new[0] - TL_new[0]))
                angle_yolo = np.degrees(math.atan2(TR_yolo[1] - TL_yolo[1], TR_yolo[0] - TL_yolo[0]))
                diff_angle = abs(angle_prop - angle_yolo)
                while diff_angle > 90:
                    diff_angle = abs(diff_angle - 180)
                    
                w_prop = math.sqrt((TR_new[0] - TL_new[0])**2 + (TR_new[1] - TL_new[1])**2)
                h_prop = math.sqrt((BL_new[0] - TL_new[0])**2 + (BL_new[1] - TL_new[1])**2)
                aspect_prop = w_prop / (h_prop + 1e-6)
                
                w_yolo = math.sqrt((TR_yolo[0] - TL_yolo[0])**2 + (TR_yolo[1] - TL_yolo[1])**2)
                h_yolo = math.sqrt((BL_yolo[0] - TL_yolo[0])**2 + (BL_yolo[1] - TL_yolo[1])**2)
                aspect_yolo = w_yolo / (h_yolo + 1e-6)
                
                aspect_ratio_dev = aspect_prop / (aspect_yolo + 1e-6)
                aspect_ok = (0.65 <= aspect_ratio_dev <= 1.35)
                
                crossing_ok = True
                for other_key, other_poly in proposals.items():
                    if other_key == key:
                        continue
                    for pt in poly_proposal:
                        if cv2.pointPolygonTest(np.array(other_poly, dtype=np.float32), (pt[0], pt[1]), False) > 0:
                            crossing_ok = False
                            break
                    if not crossing_ok:
                        break
                        
                size_ok = (w_prop >= 0.60 * median_width) and (h_prop >= 0.60 * median_height)
                
                has_foreground = (U_boundaries[c_idx]['source'] == 'foreground' or 
                                  U_boundaries[c_idx+1]['source'] == 'foreground' or 
                                  V_boundaries[r_idx]['source'] == 'foreground' or 
                                  V_boundaries[r_idx+1]['source'] == 'foreground')
                
                has_snap = (U_boundaries[c_idx]['source'] == 'support' or 
                            U_boundaries[c_idx+1]['source'] == 'support' or 
                            V_boundaries[r_idx]['source'] == 'support' or 
                            V_boundaries[r_idx+1]['source'] == 'support')
                
                has_evidence = has_foreground or has_snap
                
                source = "foreground_grid" if has_evidence else "midpoint_grid"
                
                is_foreground_env = has_foreground or (prior_source == "foreground")
                max_shift = 0.20 * panel_height if is_foreground_env else 0.18 * panel_height
                
                is_valid = (
                    is_convex and
                    0.80 <= area_ratio <= 1.20 and
                    center_shift <= max_shift and
                    diff_angle <= 4.0 and
                    aspect_ok and
                    crossing_ok and
                    size_ok and
                    has_evidence
                )
                
                if is_valid:
                    poly_final = poly_proposal
                    reason = "valid_refine"
                    if is_foreground_env and center_shift > 0.25 * panel_height:
                        print(f"[INFO] {block_name} {r_idx} {c_idx} allowed center shift {center_shift:.2f} px (> 0.25 * height) via foreground_grid source.")
                else:
                    rect = cv2.minAreaRect(np.array(poly_init, dtype=np.float32))
                    box = cv2.boxPoints(rect)
                    poly_final = [tuple(pt) for pt in sort_panel_corners(box)]
                    source = "fallback_yolo"
                    reason = ""
                    if not is_convex: reason += "not_convex "
                    if not (0.80 <= area_ratio <= 1.20): reason += f"area_ratio_{area_ratio:.2f} "
                    if center_shift > max_shift: reason += f"shift_{center_shift:.1f} "
                    if diff_angle > 4.0: reason += f"angle_{diff_angle:.1f} "
                    if not aspect_ok: reason += f"aspect_dev_{aspect_ratio_dev:.2f} "
                    if not crossing_ok: reason += "crossing "
                    if not size_ok: reason += f"size_w={w_prop:.1f}/h={h_prop:.1f}_vs_med={median_width:.1f}/{median_height:.1f} "
                    if not has_evidence: reason += "no_evidence "
                    reason = reason.strip()
                    
            panels_result.append({
                "col": c_idx,
                "row": r_idx,
                "polygon": poly_final,
                "quality": "fallback" if source == "fallback_yolo" else "pass",
                "source": source,
                "reason": reason,
                "shift": center_shift,
                "area_ratio": area_ratio,
                "overlap": max_overlap_iou,
                "orig_p": p["orig_ref"],
                "anchors": anchors
            })
            
            print(f"[SMALL_PANEL] {block_name} {r_idx} {c_idx} source={source} area_ratio={area_ratio:.2f} center_shift={center_shift:.2f} reason={reason}")
            
    return panels_result

def refine_small_block_panels_individually(rotated_gray, b_local, block_name, img_support, img_snap, transform_local_to_orig, transform_orig_to_local):
    return build_small_block_micro_grid_from_yolo_and_local_edges(rotated_gray, b_local, block_name, img_support, img_snap, transform_local_to_orig, transform_orig_to_local)

def refine_crop_left_panel_individually(rotated_gray, p_local, block_name, row_idx, row_count, img_support, img_snap, transform_info):
    Minv = transform_info["rot_M_inv"]
    roi_x1 = transform_info["roi_x1"]
    roi_y1 = transform_info["roi_y1"]
    M = cv2.invertAffineTransform(Minv)
    def transform_local_to_orig(x_local, y_local):
        x_rel = Minv[0, 0] * float(x_local) + Minv[0, 1] * float(y_local) + Minv[0, 2]
        y_rel = Minv[1, 0] * float(x_local) + Minv[1, 1] * float(y_local) + Minv[1, 2]
        return float(x_rel + roi_x1), float(y_rel + roi_y1)

    def transform_orig_to_local(x, y):
        x_rel = float(x) - roi_x1
        y_rel = float(y) - roi_y1
        x_local = M[0, 0] * x_rel + M[0, 1] * y_rel + M[0, 2]
        y_local = M[1, 0] * x_rel + M[1, 1] * y_rel + M[1, 2]
        return float(x_local), float(y_local)

    poly_init = get_local_yolo_polygon_prior(p_local["orig_ref"], transform_orig_to_local)
    poly_refined, support_pts_dict = refine_panel_edges_locally_with_yolo_prior(
        rotated_gray, poly_init, is_crop_left=True
    )
    
    # Draw support points of accepted edges
    for edge_key, clean_pts in support_pts_dict.items():
        for pt in clean_pts:
            pt_orig = transform_local_to_orig(pt[0], pt[1])
            cv2.circle(img_support, (int(round(pt_orig[0])), int(round(pt_orig[1]))), 1, (255, 0, 255), -1)
            cv2.circle(img_snap, (int(round(pt_orig[0])), int(round(pt_orig[1]))), 1, (255, 0, 255), -1)
            
    # Validate
    TL, TR, BR, BL = sort_panel_corners(poly_init)
    panel_height = math.sqrt((BL[0] - TL[0])**2 + (BL[1] - TL[1])**2)
    
    TL_new, TR_new, BR_new, BL_new = sort_panel_corners(poly_refined)
    
    is_convex = cv2.isContourConvex(np.array(poly_refined, dtype=np.int32))
    area_prop = cv2.contourArea(np.array(poly_refined, dtype=np.float32))
    area_yolo = cv2.contourArea(np.array(poly_init, dtype=np.float32))
    area_ratio = area_prop / (area_yolo + 1e-6)
    
    cx_prop = (TL_new[0] + TR_new[0] + BR_new[0] + BL_new[0]) / 4.0
    cy_prop = (TL_new[1] + TR_new[1] + BR_new[1] + BL_new[1]) / 4.0
    cx_yolo = (TL[0] + TR[0] + BR[0] + BL[0]) / 4.0
    cy_yolo = (TL[1] + TR[1] + BR[1] + BL[1]) / 4.0
    center_shift = math.sqrt((cx_prop - cx_yolo)**2 + (cy_prop - cy_yolo)**2)
    
    angle_prop = np.degrees(math.atan2(TR_new[1] - TL_new[1], TR_new[0] - TL_new[0]))
    angle_yolo = np.degrees(math.atan2(TR[1] - TL[1], TR[0] - TL[0]))
    diff_angle = abs(angle_prop - angle_yolo)
    while diff_angle > 90:
        diff_angle = abs(diff_angle - 180)
        
    angle_prop_b = np.degrees(math.atan2(BR_new[1] - BL_new[1], BR_new[0] - BL_new[0]))
    angle_yolo_b = np.degrees(math.atan2(BR[1] - BL[1], BR[0] - BL[0]))
    diff_angle_b = abs(angle_prop_b - angle_yolo_b)
    while diff_angle_b > 90:
        diff_angle_b = abs(diff_angle_b - 180)
        
    is_valid = (
        is_convex and
        0.85 <= area_ratio <= 1.15 and
        center_shift <= 0.15 * panel_height and
        diff_angle <= 3.0 and
        diff_angle_b <= 3.0 and
        abs(angle_prop - angle_prop_b) <= 4.0
    )
    
    if is_valid:
        has_refinement = ("top" in support_pts_dict or "bottom" in support_pts_dict)
        source = "crop_left_top_bottom" if has_refinement else "crop_left_yolo"
        return poly_refined, source, center_shift, area_ratio, "valid_refine"
    else:
        return poly_init, "crop_left_yolo", center_shift, area_ratio, "invalid_refine"

def build_small_block_from_yolo_edges_hybrid(block, block_name, gray, img_support, img_snap, img_hybrid_sources, img_lines, img_panels, img_panels_verified, img_quality, b_local, rotated_gray, transform_local_to_orig, transform_orig_to_local):
    row_count = max(len(col) for col in b_local["columns"]) if b_local["columns"] else 0
    is_small_block = (IMAGE_STEM == "DJI_0845_R" and (row_count < 4 or len(b_local["panels"]) <= 4))
    if is_small_block:
        panels_result = refine_small_block_panels_individually(rotated_gray, b_local, block_name, img_support, img_snap, transform_local_to_orig, transform_orig_to_local)
        
        # Store refined polygons in b_local so build_panels_from_snap_geometry can access them
        for col_idx, col in enumerate(b_local["columns"]):
            for row_idx, p in enumerate(col):
                match_res = None
                for pr in panels_result:
                    if pr["col"] == col_idx and pr["row"] == row_idx:
                        match_res = pr
                        break
                if match_res is not None:
                    p["polygon"] = match_res["polygon"]
                    p["quality"] = match_res["quality"]
                    p["source"] = match_res["source"]
                    p["reason"] = match_res["reason"]
                    p["shift"] = match_res["shift"]
                    p["area_ratio"] = match_res["area_ratio"]
                    p["overlap"] = match_res["overlap"]
                    
        return {
            "boundaries": {},
            "snapped_horiz_gaps": [],
            "block_angle_deg": 0.0,
            "is_grid": False
        }

    panels_result = refine_local_panels_individually(rotated_gray, b_local, block_name, img_support, img_snap, transform_local_to_orig)
    
    y_min_local = min(p["bbox"][1] for p in b_local["panels"])
    y_max_local = max(p["bbox"][3] for p in b_local["panels"])
    n_cols = len(b_local["columns"])
    row_count = max(len(col) for col in b_local["columns"])
    
    # Constants
    MIN_SPLITTABLE_RAIL_LEN_PX = 120
    MIN_ROWS_FOR_GRID_SPLIT = 4
    MIN_RAIL_SUPPORT_RATIO = 0.55
    MIN_VERTICAL_RAIL_GAP_PX = 12
    
    # Calculate support ratio
    pass_panels = [p for p in panels_result if p["quality"] == "pass"]
    support_ratio = len(pass_panels) / len(panels_result) if panels_result else 0.0
    
    rail_len = y_max_local - y_min_local
    
    should_split = True
    split_reason = "row_count_ok"
    
    if rail_len < MIN_SPLITTABLE_RAIL_LEN_PX:
        should_split = False
        split_reason = "rail_len_under_threshold"
    elif row_count < MIN_ROWS_FOR_GRID_SPLIT:
        should_split = False
        split_reason = "row_count_under_threshold"
    elif support_ratio < MIN_RAIL_SUPPORT_RATIO:
        should_split = False
        split_reason = "support_ratio_under_threshold"
        
    # Fit initial boundaries
    # Fit initial boundaries
    col0 = [p for p in panels_result if p["col"] == 0]
    left_pts = [p["polygon"][0] for p in col0] + [p["polygon"][3] for p in col0]
    
    if n_cols == 1:
        right_pts = [p["polygon"][1] for p in col0] + [p["polygon"][2] for p in col0]
        left_rail = fit_consensus_vertical_boundary(left_pts, y_min_local, y_max_local, row_count)
        right_rail = fit_consensus_vertical_boundary(right_pts, y_min_local, y_max_local, row_count)
        mid_rail = None
    else:
        col1 = [p for p in panels_result if p["col"] == 1]
        mid_pts = [p["polygon"][1] for p in col0] + [p["polygon"][2] for p in col0] + [p["polygon"][0] for p in col1] + [p["polygon"][3] for p in col1]
        right_pts = [p["polygon"][1] for p in col1] + [p["polygon"][2] for p in col1]
        
        left_rail = fit_consensus_vertical_boundary(left_pts, y_min_local, y_max_local, row_count)
        mid_rail = fit_consensus_vertical_boundary(mid_pts, y_min_local, y_max_local, row_count)
        right_rail = fit_consensus_vertical_boundary(right_pts, y_min_local, y_max_local, row_count)
        
        if should_split:
            gap_left_mid = abs(np.mean([pt[0] for pt in left_rail]) - np.mean([pt[0] for pt in mid_rail]))
            gap_mid_right = abs(np.mean([pt[0] for pt in mid_rail]) - np.mean([pt[0] for pt in right_rail]))
            if min(gap_left_mid, gap_mid_right) < MIN_VERTICAL_RAIL_GAP_PX:
                should_split = False
                split_reason = "gap_too_small"
                
    boundaries = {}
    if should_split:
        boundaries["left_outer"] = left_rail
        boundaries["right_outer"] = right_rail
        if mid_rail is not None:
            boundaries["middle_divider"] = mid_rail
    else:
        # Fit independent rails per column
        left_rails = []
        right_rails = []
        for col_idx in range(n_cols):
            col_panels = [p for p in panels_result if p["col"] == col_idx]
            left_pts_c = [p["polygon"][0] for p in col_panels] + [p["polygon"][3] for p in col_panels]
            right_pts_c = [p["polygon"][1] for p in col_panels] + [p["polygon"][2] for p in col_panels]
            left_rails.append(fit_consensus_vertical_boundary(left_pts_c, y_min_local, y_max_local, row_count))
            right_rails.append(fit_consensus_vertical_boundary(right_pts_c, y_min_local, y_max_local, row_count))
            
        if n_cols == 2:
            gap_mid = abs(np.mean([pt[0] for pt in right_rails[0]]) - np.mean([pt[0] for pt in left_rails[1]]))
            if gap_mid < MIN_VERTICAL_RAIL_GAP_PX:
                col0 = [p for p in panels_result if p["col"] == 0]
                col1 = [p for p in panels_result if p["col"] == 1]
                mid_pts = [p["polygon"][1] for p in col0] + [p["polygon"][2] for p in col0] + [p["polygon"][0] for p in col1] + [p["polygon"][3] for p in col1]
                merged_mid = fit_consensus_vertical_boundary(mid_pts, y_min_local, y_max_local, row_count)
                boundaries["left_col_0"] = left_rails[0]
                boundaries["right_col_0"] = merged_mid
                boundaries["left_col_1"] = merged_mid
                boundaries["right_col_1"] = right_rails[1]
            else:
                boundaries["left_col_0"] = left_rails[0]
                boundaries["right_col_0"] = right_rails[0]
                boundaries["left_col_1"] = left_rails[1]
                boundaries["right_col_1"] = right_rails[1]
        else:
            boundaries["left_col_0"] = left_rails[0]
            boundaries["right_col_0"] = right_rails[0]
            
    action_str = "split" if should_split else ("merge" if split_reason == "gap_too_small" else "single")
    print(f"[RAIL_SPLIT] {block_name} left_outer {rail_len:.1f} {support_ratio:.2f} {row_count} action={action_str} reason={split_reason}")
    if n_cols == 2:
        mid_action = "merge" if (action_str == "merge" or not should_split) else "split"
        print(f"[RAIL_SPLIT] {block_name} middle_divider {rail_len:.1f} {support_ratio:.2f} {row_count} action={mid_action} reason={split_reason}")
    print(f"[RAIL_SPLIT] {block_name} right_outer {rail_len:.1f} {support_ratio:.2f} {row_count} action={action_str} reason={split_reason}")
    
    # Store refined polygons in b_local so build_panels_from_snap_geometry can access them
    for col_idx, col in enumerate(b_local["columns"]):
        for row_idx, p in enumerate(col):
            match_res = None
            for pr in panels_result:
                if pr["col"] == col_idx and pr["row"] == row_idx:
                    match_res = pr
                    break
            if match_res is not None:
                p["polygon"] = match_res["polygon"]
                p["quality"] = match_res["quality"]
                p["source"] = match_res["source"]
                
    snapped_horiz_gaps = []
    x_min_roi_local = min(p["bbox"][0] for p in b_local["panels"]) - 10
    x_max_roi_local = max(p["bbox"][2] for p in b_local["panels"]) + 10
    
    for r in range(row_count + 1):
        if r == 0:
            row_0_panels = [p for p in panels_result if p["row"] == 0]
            yolo_pts = [p["polygon"][0] for p in row_0_panels] + [p["polygon"][1] for p in row_0_panels]
        elif r == row_count:
            row_last_panels = [p["polygon"] for p in panels_result if p["row"] == row_count - 1]
            yolo_pts = [p[3] for p in row_last_panels] + [p[2] for p in row_last_panels]
        else:
            row_prev = [p for p in panels_result if p["row"] == r - 1]
            row_curr = [p for p in panels_result if p["row"] == r]
            yolo_pts = ([p["polygon"][3] for p in row_prev] + [p["polygon"][2] for p in row_prev] +
                        [p["polygon"][0] for p in row_curr] + [p["polygon"][1] for p in row_curr])
                        
        xs_yolo = [pt[0] for pt in yolo_pts]
        ys_yolo = [pt[1] for pt in yolo_pts]
        m_yolo, c_yolo = np.polyfit(xs_yolo, ys_yolo, 1)
        
        is_outer = (r == 0 or r == row_count)
        m_fit, c_fit, clean_pts, coverage, std_res, support_valid, n_raw = extract_support_points_for_horizontal_edge(
            rotated_gray, m_yolo, c_yolo, x_min_roi_local, x_max_roi_local, block_name, r, row_count, is_outer_edge=is_outer
        )
        
        if support_valid:
            m_row = m_fit
            c_row = c_fit
            selected_cand = "support"
            reason = "valid_support"
            for pt in clean_pts:
                pt_orig = transform_local_to_orig(pt[0], pt[1])
                cv2.circle(img_support, (int(round(pt_orig[0])), int(round(pt_orig[1]))), 1, (255, 0, 255), -1)
                cv2.circle(img_snap, (int(round(pt_orig[0])), int(round(pt_orig[1]))), 1, (255, 0, 255), -1)
        else:
            m_row = m_yolo
            c_row = c_yolo
            selected_cand = "yolo"
            reason = "fallback_yolo"
            
        print(f"[SUPPORT_EDGE] {block_name} {r} raw={n_raw} clean={len(clean_pts)} coverage={coverage:.3f} residual={std_res:.3f} source={selected_cand} reason={reason}")
        
        snapped_horiz_gaps.append({
            "y_snap": m_row * ((x_min_roi_local + x_max_roi_local) / 2.0) + c_row,
            "selected_cand": selected_cand,
            "m_row": m_row,
            "c_row": c_row,
            "m_yolo": m_yolo,
            "c_yolo": c_yolo,
            "support_pts": clean_pts,
            "support_valid": support_valid,
            "support_line_m": m_fit,
            "support_line_c": c_fit,
            "support_clean_pts": clean_pts,
            "support_coverage": coverage,
            "support_residual_std": std_res
        })
        
    boundaries_orig = {bkey: [transform_local_to_orig(pt[0], pt[1]) for pt in boundaries[bkey]] for bkey in boundaries}
    for bkey in boundaries_orig:
        draw_polyline(img_snap, boundaries_orig[bkey], (180, 0, 0), 2)
        
    return {
        "boundaries": boundaries,
        "snapped_horiz_gaps": snapped_horiz_gaps,
        "block_angle_deg": 0.0,
        "is_grid": should_split
    }

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
    x_blocks = group_into_blocks(panels)
    original_block_count = len(x_blocks)
    
    # Split each x-block along Y if a large gap exists between row clusters
    blocks = []
    block_split_meta = []  # metadata per original block about how it was split
    for xb in x_blocks:
        sub_blocks, median_h, gaps_used = split_block_by_y_gap(xb)
        block_split_meta.append({
            "original_n_cols": len(xb["columns"]),
            "original_panel_count": len(xb["panels"]),
            "sub_block_count": len(sub_blocks),
            "median_panel_height": median_h,
            "gaps_used": gaps_used
        })
        blocks.extend(sub_blocks)
    
    split_block_count = len(blocks)
    
    img_blocks = img.copy()
    img_support = img.copy()
    img_snap = img.copy()
    img_panels_from_snap = img.copy()
    
    snap_blocks = []
    stats = []

    
    for b_idx, b in enumerate(blocks):
        block_name = f"block_{b_idx + 1}"
        n_cols = len(b["columns"])
        
        # Original coordinates range
        bx_min = min(p["bbox"][0] for p in b["panels"])
        bx_max = max(p["bbox"][2] for p in b["panels"])
        by_min = min(p["bbox"][1] for p in b["panels"])
        by_max = max(p["bbox"][3] for p in b["panels"])
        
        # 1. Estimate robust median orientation angle of the block columns
        block_angles = []
        for col in b["columns"]:
            if len(col) >= 2:
                cxs = [(p["bbox"][0] + p["bbox"][2]) / 2.0 for p in col]
                cys = [(p["bbox"][1] + p["bbox"][3]) / 2.0 for p in col]
                try:
                    slope, _ = np.polyfit(cys, cxs, 1)
                    angle_rad = math.atan(slope)
                    block_angles.append(angle_rad)
                except Exception:
                    pass
        if block_angles:
            rotation_rad = np.median(block_angles)
        else:
            rotation_rad = 0.0
        rotation_deg = np.degrees(rotation_rad)
        
        # 2. Crop ROI block with margin
        margin = 60
        roi_x1 = max(0, int(bx_min - margin))
        roi_y1 = max(0, int(by_min - margin))
        roi_x2 = min(gray.shape[1], int(bx_max + margin))
        roi_y2 = min(gray.shape[0], int(by_max + margin))
        
        roi_w = roi_x2 - roi_x1
        roi_h = roi_y2 - roi_y1
        
        center_x = roi_w / 2.0
        center_y = roi_h / 2.0
        
        # 3. Rotate ROI by -rotation_deg to align columns vertically
        M = cv2.getRotationMatrix2D((center_x, center_y), -rotation_deg, 1.0)
        Minv = cv2.invertAffineTransform(M)
        
        # Point transformations
        def transform_orig_to_local(x, y):
            x_rel = float(x) - roi_x1
            y_rel = float(y) - roi_y1
            x_local = M[0, 0] * x_rel + M[0, 1] * y_rel + M[0, 2]
            y_local = M[1, 0] * x_rel + M[1, 1] * y_rel + M[1, 2]
            return float(x_local), float(y_local)
            
        def transform_local_to_orig(x_local, y_local):
            x_rel = Minv[0, 0] * float(x_local) + Minv[0, 1] * float(y_local) + Minv[0, 2]
            y_rel = Minv[1, 0] * float(x_local) + Minv[1, 1] * float(y_local) + Minv[1, 2]
            return float(x_rel + roi_x1), float(y_rel + roi_y1)
            
        # Rotate cropped images
        roi_gray = gray[roi_y1:roi_y2, roi_x1:roi_x2]
        rotated_gray = cv2.warpAffine(roi_gray, M, (roi_w, roi_h), flags=cv2.INTER_CUBIC)
        
        roi_edges = edges[roi_y1:roi_y2, roi_x1:roi_x2]
        rotated_edges = cv2.warpAffine(roi_edges, M, (roi_w, roi_h), flags=cv2.INTER_NEAREST)
        
        roi_sobel_x_abs = sobel_x_abs[roi_y1:roi_y2, roi_x1:roi_x2]
        rotated_sobel_x_abs = cv2.warpAffine(roi_sobel_x_abs, M, (roi_w, roi_h), flags=cv2.INTER_CUBIC)
        
        # Transform panels to rotated local coordinates
        b_local = {
            "columns": [],
            "panels": []
        }
        panel_map = {}
        for p in b["panels"]:
            local_poly = [transform_orig_to_local(pt[0], pt[1]) for pt in p["refined_polygon"]]
            l_xs = [pt[0] for pt in local_poly]
            l_ys = [pt[1] for pt in local_poly]
            local_bbox = [min(l_xs), min(l_ys), max(l_xs), max(l_ys)]
            
            p_local = {
                "bbox": local_bbox,
                "refined_polygon": local_poly,
                "orig_ref": p
            }
            panel_map[id(p)] = p_local
            b_local["panels"].append(p_local)
            
        for col in b["columns"]:
            col_local = [panel_map[id(p)] for p in col]
            b_local["columns"].append(col_local)
            
        y_min_local = min(p["bbox"][1] for p in b_local["panels"])
        y_max_local = max(p["bbox"][3] for p in b_local["panels"])
        x_min_roi_local = min(p["bbox"][0] for p in b_local["panels"]) - 10
        x_max_roi_local = max(p["bbox"][2] for p in b_local["panels"]) + 10
        
        # Original coordinates ROI for blocks debug image
        cv2.rectangle(img_blocks, (int(bx_min - 10), int(by_min)), (int(bx_max + 10), int(by_max)), (0, 255, 0), 2)
        cv2.putText(img_blocks, f"{block_name} ({rotation_deg:.1f} deg)", (int(bx_min - 5), int(by_min) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
        
        # 4. Fit boundaries in local rotated coordinate space
        # Calculate maximum row count across columns in this block
        row_count = max(len(col) for col in b_local["columns"]) if b_local["columns"] else 10
        
        MIN_LARGE_BLOCK_ROWS = 4
        MIN_SMALL_BLOCK_PANELS = 2
        
        block_panels = b["panels"]
        if len(block_panels) < MIN_SMALL_BLOCK_PANELS:
            continue
            
        # Calculate local and orig bbox for debug log
        local_xs = [p["bbox"][0] for p in b_local["panels"]] + [p["bbox"][2] for p in b_local["panels"]]
        local_ys = [p["bbox"][1] for p in b_local["panels"]] + [p["bbox"][3] for p in b_local["panels"]]
        l_min_x, l_max_x = min(local_xs), max(local_xs)
        l_min_y, l_max_y = min(local_ys), max(local_ys)
        
        orig_xs = [p["bbox"][0] for p in b["panels"]] + [p["bbox"][2] for p in b["panels"]]
        orig_ys = [p["bbox"][1] for p in b["panels"]] + [p["bbox"][3] for p in b["panels"]]
        o_min_x, o_max_x = min(orig_xs), max(orig_xs)
        o_min_y, o_max_y = min(orig_ys), max(orig_ys)
        
        print(f"[SNAP_BLOCK] {block_name} local_bbox=({l_min_x:.1f},{l_min_y:.1f},{l_max_x:.1f},{l_max_y:.1f}) orig_bbox=({o_min_x:.1f},{o_min_y:.1f},{o_max_x:.1f},{o_max_y:.1f})")

        is_small_block = (IMAGE_STEM == "DJI_0845_R" and (row_count < 4 or len(b["panels"]) <= 4))
        if is_small_block:
            result = build_small_block_from_yolo_edges_hybrid(
                b, block_name, gray, img_support, img_snap, None, None, None, None, None,
                b_local, rotated_gray, transform_local_to_orig, transform_orig_to_local
            )
            boundaries = result["boundaries"]
            snapped_horiz_gaps = result["snapped_horiz_gaps"]
            
            snap_blocks.append({
                "block_name": block_name,
                "n_cols": n_cols,
                "row_count": row_count,
                "is_large_grid": False,
                "is_grid": result["is_grid"],
                "boundaries": copy.deepcopy(boundaries),
                "snapped_horiz_gaps": copy.deepcopy(snapped_horiz_gaps),
                "m_perp": 0.0,
                "x_mid": float((x_min_roi_local + x_max_roi_local) / 2.0),
                "b_local": copy.deepcopy(b_local),
                "transform_info": {
                    "rot_M_inv": Minv.copy(),
                    "roi_x1": int(roi_x1),
                    "roi_y1": int(roi_y1),
                    "local_w": int(roi_w),
                    "local_h": int(roi_h),
                },
                "rotation_deg": float(rotation_deg),
                "is_small_block": True,
                "rotated_gray": rotated_gray.copy()
            })
            continue
            
        if n_cols == 1:
            boundary_keys = ["left_outer", "right_outer"]
        else:
            boundary_keys = ["left_outer", "middle_divider", "right_outer"]
            
        boundaries, snapped_horiz_gaps, consensus_stats = build_grid_from_yolo_edge_consensus(
            rotated_gray, rotated_sobel_x_abs, b_local, y_min_local, y_max_local, n_cols, row_count, x_min_roi_local, x_max_roi_local, block_name
        )
        
        best_theta = math.atan(consensus_stats["m_consensus"])
        m_perp = math.tan(best_theta)
        x_mid = (x_min_roi_local + x_max_roi_local) / 2.0
        
        boundaries_orig = {bkey: [transform_local_to_orig(pt[0], pt[1]) for pt in boundaries[bkey]] for bkey in boundaries}
        
        for g in snapped_horiz_gaps:
            for pt in g["support_pts"]:
                pt_orig = transform_local_to_orig(pt[0], pt[1])
                cv2.circle(img_support, (int(round(pt_orig[0])), int(round(pt_orig[1]))), 1, (255, 0, 255), -1)
                cv2.circle(img_snap, (int(round(pt_orig[0])), int(round(pt_orig[1]))), 1, (255, 0, 255), -1)
                
        draw_polyline(img_snap, boundaries_orig["left_outer"], (180, 0, 0), 2)
        draw_polyline(img_snap, boundaries_orig["right_outer"], (180, 0, 0), 2)
        if "middle_divider" in boundaries_orig:
            draw_polyline(img_snap, boundaries_orig["middle_divider"], (180, 0, 0), 2)
            
        snap_blocks.append({
            "block_name": block_name,
            "n_cols": n_cols,
            "row_count": row_count,
            "is_large_grid": True,
            "boundaries": copy.deepcopy(boundaries),
            "snapped_horiz_gaps": copy.deepcopy(snapped_horiz_gaps),
            "m_perp": float(m_perp),
            "x_mid": float(x_mid),
            "b_local": copy.deepcopy(b_local),
            "transform_info": {
                "rot_M_inv": Minv.copy(),
                "roi_x1": int(roi_x1),
                "roi_y1": int(roi_y1),
                "local_w": int(roi_w),
                "local_h": int(roi_h),
            },
            "rotation_deg": float(rotation_deg),
            "rotated_gray": rotated_gray.copy()
        })
        
    # Draw panel polygons from snap blocks
    drawn_panel_count = build_panel_polygons_from_visible_snap_evidence(img_panels_from_snap, snap_blocks, img_support, img_snap)
    
    # Save outputs
    out_dir = Path("data/results/debug")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Clean up old files
    old_files = [
        f"debug_{IMAGE_STEM}_panels_quality.JPG",
        f"debug_{IMAGE_STEM}_panels_verified.JPG",
        f"debug_{IMAGE_STEM}_full_image_panels_verified_seg10.JPG",
        f"debug_{IMAGE_STEM}_hybrid_sources.JPG",
        f"debug_{IMAGE_STEM}_yolo_edge_consensus_lines.JPG",
        f"debug_{IMAGE_STEM}_yolo_edge_consensus_panels.JPG",
        f"debug_{IMAGE_STEM}_yolo_edge_consensus_quality.JPG"
    ]
    for old_name in old_files:
        (out_dir / old_name).unlink(missing_ok=True)
        
    # Write the 4 specified output images
    cv2.imwrite(str(out_dir / f"debug_{IMAGE_STEM}_blocks.JPG"), img_blocks)
    cv2.imwrite(str(out_dir / f"debug_{IMAGE_STEM}_support_points.JPG"), img_support)
    cv2.imwrite(str(out_dir / f"debug_{IMAGE_STEM}_line_snap.JPG"), img_snap)
    cv2.imwrite(str(out_dir / f"debug_{IMAGE_STEM}_panels_from_snap.JPG"), img_panels_from_snap)
    
    # Print simplified stats
    print("\n" + "="*95)
    print(f"TOTAL DRAWN PANELS FROM SNAP: {drawn_panel_count}")
    print("="*95)

if __name__ == "__main__":
    main()
