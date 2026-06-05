import os
import json
import math
import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass

@dataclass
class EdgeGridConfig:
    roi_margin: int = 45
    vertical_mode: str = "segmented"
    n_segments: int = 4
    max_x_jump_px: float = 10.0
    max_angle_delta_deg: float = 8.0
    search_half_width: int = 10
    min_points_per_segment: int = 6
    max_dev_baseline_px: float = 12.0
    smooth_window: int = 3

# Paths
IMAGE_PATH = Path("data/precalib/DJI_0029_R.JPG")
TRACE_PATH = Path("data/results/debug_logs/DJI_0029_R_final_polygon_trace.jsonl")

OUT_PATH_PROBE = Path("data/results/debug/debug_DJI_0029_R_edge_line_probe_refactor.JPG")
OUT_PATH_PIECEWISE = Path("data/results/debug/debug_DJI_0029_R_edge_grid_proposal_piecewise_refactor.JPG")

# 12 Target indices
TARGET_IDXS = {25, 40, 46, 47, 52, 62, 65, 67, 68, 77, 78, 86}

def intersect_h_v(y_int_h, ang_h, x_int_v, ang_v, x_mid, y_mid):
    """Compute the intersection of a horizontal-ish line and a vertical-ish line."""
    tan_h = math.tan(math.radians(ang_h))
    if abs(ang_v) == 90.0:
        x = x_int_v
        y = y_int_h + tan_h * (x - x_mid)
    else:
        tan_v = math.tan(math.radians(ang_v))
        # From equations:
        # y = tan_h * (x - x_mid) + y_int_h
        # x = x_int_v + (y - y_mid) / tan_v
        # Solve for x:
        # x = (x_int_v * tan_v - tan_h * x_mid + y_int_h - y_mid) / (tan_v - tan_h)
        x = (x_int_v * tan_v - tan_h * x_mid + y_int_h - y_mid) / (tan_v - tan_h)
        y = y_int_h + tan_h * (x - x_mid)
    return (float(x), float(y))

def get_boundary_x(boundary_points, y_val):
    """Interpolate the X position on the boundary for a given Y."""
    ys = [pt[0] for pt in boundary_points]
    xs = [pt[1] for pt in boundary_points]
    return float(np.interp(y_val, ys, xs))

def get_intersection_piecewise(y_int_h, ang_h, boundary_points, x_mid):
    """Compute intersection of a horizontal-ish line and a piecewise vertical boundary."""
    x_est = get_boundary_x(boundary_points, y_int_h)
    tan_h = math.tan(math.radians(ang_h))
    y_ref = y_int_h + tan_h * (x_est - x_mid)
    x_ref = get_boundary_x(boundary_points, y_ref)
    y_final = y_int_h + tan_h * (x_ref - x_mid)
    return (x_ref, y_final)


def load_image_and_trace(image_path, trace_path, target_idxs):
    """1. Load image and trace file."""
    assert trace_path.exists(), f"Trace not found: {trace_path}"
    with open(trace_path, encoding="utf-8") as f:
        panels = [json.loads(line.strip()) for line in f if line.strip()]
    targets = [p for p in panels if p.get("raw_idx") in target_idxs]
    assert targets, "No target panels found in trace!"
    
    assert image_path.exists(), f"Image not found: {image_path}"
    img = cv2.imread(str(image_path))
    return panels, targets, img

def build_roi_from_target_panels(targets, img_w, img_h, margin=40):
    """2. Build ROI bounding coordinates from targets with a margin."""
    all_x = []
    all_y = []
    for p in targets:
        box = p.get("bbox") or p.get("original_yolo_bbox")
        if box:
            all_x.extend([box[0], box[2]])
            all_y.extend([box[1], box[3]])
    min_x = max(0, min(all_x) - margin)
    max_x = min(img_w, max(all_x) + margin)
    min_y = max(0, min(all_y) - margin)
    max_y = min(img_h, max(all_y) + margin)
    return min_x, max_x, min_y, max_y

def preprocess_roi_for_edges(img, roi_coords):
    """3. Preprocess the ROI to extract Canny edges."""
    min_x, max_x, min_y, max_y = roi_coords
    roi_img = img[min_y:max_y, min_x:max_x]
    gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
    return edges

def cluster_segments(segments, threshold=12.0):
    """Helper: Cluster 1D segments by intercept values."""
    if not segments:
        return []
    segments.sort(key=lambda x: x[0])
    clusters = []
    current = [segments[0]]
    for s in segments[1:]:
        if s[0] - current[-1][0] <= threshold:
            current.append(s)
        else:
            clusters.append(current)
            current = [s]
    clusters.append(current)
    return clusters

def detect_horizontal_gaps(edges, roi_coords):
    """4. Detect and cluster horizontal consensus gap lines."""
    min_x, max_x, min_y, max_y = roi_coords
    x_mid = (min_x + max_x) / 2.0
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=25, minLineLength=20, maxLineGap=10)
    
    horiz_segments = []
    if lines is not None:
        for line in lines:
            x1_roi, y1_roi, x2_roi, y2_roi = line[0]
            x1, y1 = x1_roi + min_x, y1_roi + min_y
            x2, y2 = x2_roi + min_x, y2_roi + min_y
            dx = x2 - x1
            dy = y2 - y1
            length = math.sqrt(dx*dx + dy*dy)
            if length < 1e-3:
                continue
            angle = math.degrees(math.atan2(dy, dx))
            if angle > 90.0:
                angle -= 180.0
            elif angle < -90.0:
                angle += 180.0
                
            if abs(angle) <= 25.0:
                y_int = y1 + (x_mid - x1) * (dy / dx) if abs(dx) > 1e-3 else y1
                horiz_segments.append((y_int, angle, length, (x1, y1, x2, y2)))
                
    horiz_clusters = cluster_segments(horiz_segments, 12.0)
    final_horiz_lines = []
    for c in horiz_clusters:
        total_len = sum(s[2] for s in c)
        if total_len < 25.0:
            continue
        med_y_int = float(np.median([s[0] for s in c]))
        med_ang = float(np.median([s[1] for s in c]))
        final_horiz_lines.append((med_y_int, med_ang, total_len))
    return final_horiz_lines

def detect_initial_vertical_boundaries(edges, roi_coords):
    """5. Detect, cluster, and filter the initial straight vertical boundaries."""
    min_x, max_x, min_y, max_y = roi_coords
    y_mid = (min_y + max_y) / 2.0
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=25, minLineLength=20, maxLineGap=10)
    
    vert_segments = []
    if lines is not None:
        for line in lines:
            x1_roi, y1_roi, x2_roi, y2_roi = line[0]
            x1, y1 = x1_roi + min_x, y1_roi + min_y
            x2, y2 = x2_roi + min_x, y2_roi + min_y
            dx = x2 - x1
            dy = y2 - y1
            length = math.sqrt(dx*dx + dy*dy)
            if length < 1e-3:
                continue
            angle = math.degrees(math.atan2(dy, dx))
            if angle > 90.0:
                angle -= 180.0
            elif angle < -90.0:
                angle += 180.0
                
            if abs(angle) >= 65.0:
                x_int = x1 + (y_mid - y1) * (dx / dy) if abs(dy) > 1e-3 else x1
                vert_segments.append((x_int, angle, length, (x1, y1, x2, y2)))
                
    vert_clusters = cluster_segments(vert_segments, 12.0)
    final_vert_lines = []
    for c in vert_clusters:
        total_len = sum(s[2] for s in c)
        if total_len < 25.0:
            continue
        med_x_int = float(np.median([s[0] for s in c]))
        med_ang = float(np.median([s[1] for s in c]))
        final_vert_lines.append((med_x_int, med_ang, total_len))
        
    # Filter left vertical line and sort
    final_vert_lines = [line for line in final_vert_lines if line[0] >= 480.0]
    final_vert_lines.sort(key=lambda x: x[0])
    return final_vert_lines

def sample_piecewise_vertical_boundary(edges, roi_coords, initial_vert_line, search_window=15, max_dev_px=6, smooth_window=7):
    """6. Sample Canny edges along the vertical line and regularize/smooth them to make a piecewise polyline."""
    min_x, max_x, min_y, max_y = roi_coords
    y_mid = (min_y + max_y) / 2.0
    x_int, ang_v, _ = initial_vert_line
    raw_samples = []
    
    for y in range(int(min_y), int(max_y), 5):
        if abs(ang_v) == 90.0:
            x_base = x_int
        else:
            x_base = x_int + (y - y_mid) / math.tan(math.radians(ang_v))
            
        y_roi = int(round(y - min_y))
        y_roi = max(0, min(edges.shape[0] - 1, y_roi))
        x_pred_roi = x_base - min_x
        
        best_x_roi = None
        min_dist = float('inf')
        x_start = int(max(0, math.floor(x_pred_roi - search_window)))
        x_end = int(min(edges.shape[1] - 1, math.ceil(x_pred_roi + search_window)))
        
        for x_r in range(x_start, x_end + 1):
            if edges[y_roi, x_r] == 255:
                dist = abs(x_r - x_pred_roi)
                if dist < min_dist:
                    min_dist = dist
                    best_x_roi = x_r
                    
        x_raw = best_x_roi + min_x if best_x_roi is not None else x_base
        
        # Clip to baseline +/- max_dev_px
        x_clipped = max(x_base - max_dev_px, min(x_base + max_dev_px, x_raw))
        raw_samples.append((y, x_base, x_clipped))
        
    # Smooth with moving average window of size `smooth_window`
    smoothed = []
    n_samples = len(raw_samples)
    half_w = smooth_window // 2
    for idx in range(n_samples):
        w_start = max(0, idx - half_w)
        w_end = min(n_samples, idx + half_w + 1)
        mean_x = float(np.mean([raw_samples[k][2] for k in range(w_start, w_end)]))
        
        # Clip again to guarantee constraint post-smoothing
        y_val = raw_samples[idx][0]
        x_base_val = raw_samples[idx][1]
        x_final = max(x_base_val - max_dev_px, min(x_base_val + max_dev_px, mean_x))
        smoothed.append((y_val, x_final))
    return smoothed

def fit_segmented_vertical_boundary(
    edge_img,
    initial_line,
    y_min,
    y_max,
    n_segments=8,
    search_half_width=10,
    min_points_per_segment=6,
    max_angle_delta_deg=8,
    max_x_jump_px=10,
    min_x=0,
    max_dev_baseline_px=None,
    smooth_window=3
):
    """Fit a vertical boundary with n_segments line segments, limiting orientation deltas and X jumps."""
    x_int, ang_v, _ = initial_line
    y_mid = (y_min + y_max) / 2.0
    if abs(ang_v) == 90.0:
        a_init = 0.0
        b_init = x_int
    else:
        a_init = 1.0 / math.tan(math.radians(ang_v))
        b_init = x_int - a_init * y_mid

    y_step = (y_max - y_min) / n_segments
    segment_lines = []

    for k in range(n_segments):
        seg_y_start = y_min + k * y_step
        seg_y_end = seg_y_start + y_step
        
        seg_ys = []
        seg_xs = []
        
        y_start_int = int(math.floor(seg_y_start))
        y_end_int = int(math.ceil(seg_y_end))
        
        for y in range(y_start_int, y_end_int):
            y_roi = int(round(y - y_min))
            if y_roi < 0 or y_roi >= edge_img.shape[0]:
                continue
                
            x_base = a_init * y + b_init
            x_min_search = int(math.floor(x_base - search_half_width))
            x_max_search = int(math.ceil(x_base + search_half_width))
            
            for x in range(x_min_search, x_max_search + 1):
                x_roi = int(round(x - min_x))
                if x_roi < 0 or x_roi >= edge_img.shape[1]:
                    continue
                if edge_img[y_roi, x_roi] == 255:
                    seg_ys.append(y)
                    seg_xs.append(x)
                    
        # Fit segment using least squares with outlier rejection
        a_fit, b_fit = a_init, b_init
        fitted = False
        
        if len(seg_ys) >= min_points_per_segment:
            try:
                # Initial fit
                a_c, b_c = np.polyfit(seg_ys, seg_xs, 1)
                
                # Outlier rejection
                residuals = np.abs(np.array(seg_xs) - (a_c * np.array(seg_ys) + b_c))
                threshold = np.median(residuals) * 1.5
                threshold = max(threshold, 2.0)
                
                valid_indices = residuals <= threshold
                ys_clean = np.array(seg_ys)[valid_indices]
                xs_clean = np.array(seg_xs)[valid_indices]
                
                if len(ys_clean) >= min_points_per_segment:
                    a_c, b_c = np.polyfit(ys_clean, xs_clean, 1)
                    
                # Calculate angle of the fitted segment
                ang_fit = math.degrees(math.atan2(1.0, a_c))
                if ang_fit > 90.0:
                    ang_fit -= 180.0
                elif ang_fit < -90.0:
                    ang_fit += 180.0
                    
                ang_diff = abs(ang_fit - ang_v)
                if ang_diff > 180.0:
                    ang_diff = 360.0 - ang_diff
                    
                if ang_diff <= max_angle_delta_deg:
                    a_fit = a_c
                    b_fit = b_c
                    fitted = True
            except Exception:
                pass
                
        segment_lines.append((a_fit, b_fit, fitted))

    # Check for abrupt slope breaks (gãy đột ngột) and fallback
    for k in range(1, n_segments - 1):
        a_prev, b_prev, fit_p = segment_lines[k-1]
        a_curr, b_curr, fit_c = segment_lines[k]
        a_next, b_next, fit_n = segment_lines[k+1]
        
        if fit_c:
            ang_prev = math.degrees(math.atan2(1.0, a_prev))
            ang_curr = math.degrees(math.atan2(1.0, a_curr))
            ang_next = math.degrees(math.atan2(1.0, a_next))
            
            ang_neighbor_avg = (ang_prev + ang_next) / 2.0
            if abs(ang_curr - ang_neighbor_avg) > max_angle_delta_deg:
                if fit_p and fit_n:
                    a_fit = (a_prev + a_next) / 2.0
                    b_fit = (b_prev + b_next) / 2.0
                else:
                    a_fit = a_init
                    b_fit = b_init
                segment_lines[k] = (a_fit, b_fit, False)

    # Smooth slope/intercept by moving average window
    slopes = [line[0] for line in segment_lines]
    intercepts = [line[1] for line in segment_lines]
    smoothed_slopes = []
    smoothed_intercepts = []
    half_w = smooth_window // 2
    for k in range(n_segments):
        w_start = max(0, k - half_w)
        w_end = min(n_segments, k + half_w + 1)
        s = float(np.mean(slopes[w_start:w_end]))
        i = float(np.mean(intercepts[w_start:w_end]))
        smoothed_slopes.append(s)
        smoothed_intercepts.append(i)
        
    for k in range(n_segments):
        segment_lines[k] = (smoothed_slopes[k], smoothed_intercepts[k], segment_lines[k][2])

    # Limit X jump between consecutive segments
    for k in range(1, n_segments):
        a_prev, b_prev, _ = segment_lines[k-1]
        a_curr, b_curr, fitted = segment_lines[k]
        
        y_junc = y_min + k * y_step
        x_prev = a_prev * y_junc + b_prev
        x_curr = a_curr * y_junc + b_curr
        
        if abs(x_curr - x_prev) > max_x_jump_px:
            x_curr_adj = np.clip(x_curr, x_prev - max_x_jump_px, x_prev + max_x_jump_px)
            b_curr = x_curr_adj - a_curr * y_junc
            segment_lines[k] = (a_curr, b_curr, fitted)

    # Output continuous polyline: n_segments + 1 points
    y_junctions = [y_min + k * y_step for k in range(n_segments + 1)]
    polyline = []
    for k in range(n_segments + 1):
        y_j = y_junctions[k]
        if k == 0:
            a, b, _ = segment_lines[0]
            x_j = a * y_j + b
        elif k == n_segments:
            a, b, _ = segment_lines[-1]
            x_j = a * y_j + b
        else:
            a_p, b_p, _ = segment_lines[k-1]
            a_c, b_c, _ = segment_lines[k]
            x_left = a_p * y_j + b_p
            x_right = a_c * y_j + b_c
            x_j = (x_left + x_right) / 2.0
            
        # Clip x around initial baseline
        if max_dev_baseline_px is not None:
            x_base = a_init * y_j + b_init
            x_j = np.clip(x_j, x_base - max_dev_baseline_px, x_base + max_dev_baseline_px)
            
        polyline.append((y_j, x_j))
        
    return polyline


def build_panel_polygons_from_boundaries(
    targets, final_horiz_lines, left_boundary, right_boundary, roi_coords,
    left_line_straight=None, right_line_straight=None, vertical_mode="straight"
):
    """7. Intersect horizontal lines and vertical boundaries (straight or piecewise) to form proposals."""
    min_x, max_x, min_y, max_y = roi_coords
    x_mid = (min_x + max_x) / 2.0
    y_mid = (min_y + max_y) / 2.0
    
    ref_ious = {
        25: 0.7809, 40: 0.4744, 46: 0.4778, 47: 0.8173,
        52: 0.7792, 62: 0.5986, 65: 0.5438, 67: 0.4892,
        68: 0.8791, 77: 0.4663, 78: 0.8958, 86: 0.5261
    }
    
    report_data = []
    for p in targets:
        ridx = p["raw_idx"]
        yolo_box = p.get("original_yolo_bbox") or p.get("bbox")
        ox1, oy1, ox2, oy2 = yolo_box
        cx_o = (ox1 + ox2) / 2.0
        cy_o = (oy1 + oy2) / 2.0
        
        # Find horizontal lines
        top_line = None
        bot_line = None
        min_top_err = float('inf')
        min_bot_err = float('inf')
        
        for h_line in final_horiz_lines:
            y_int, ang, _ = h_line
            y_at_cx = y_int + math.tan(math.radians(ang)) * (cx_o - x_mid)
            if y_at_cx < cy_o:
                err = abs(y_at_cx - oy1)
                if err < min_top_err:
                    min_top_err = err
                    top_line = h_line
            elif y_at_cx > cy_o:
                err = abs(y_at_cx - oy2)
                if err < min_bot_err:
                    min_bot_err = err
                    bot_line = h_line
                    
        proposal_ok = True
        reason = "ok"
        
        has_vert = (left_line_straight and right_line_straight) if vertical_mode == "straight" else (left_boundary and right_boundary)
        if not (top_line and bot_line and has_vert):
            proposal_ok = False
            reason = "missing_boundary_line"
        else:
            y_top_at_cx = top_line[0] + math.tan(math.radians(top_line[1])) * (cx_o - x_mid)
            y_bot_at_cx = bot_line[0] + math.tan(math.radians(bot_line[1])) * (cx_o - x_mid)
            if abs(y_top_at_cx - oy1) > 15.0 or abs(y_bot_at_cx - oy2) > 15.0:
                proposal_ok = False
                reason = "missing_boundary_line"
                
        iou = 0.0
        center_shift = 0.0
        area_ratio = 0.0
        proposal_poly = []
        
        if proposal_ok:
            if vertical_mode == "straight":
                pt_tl = intersect_h_v(top_line[0], top_line[1], left_line_straight[0], left_line_straight[1], x_mid, y_mid)
                pt_tr = intersect_h_v(top_line[0], top_line[1], right_line_straight[0], right_line_straight[1], x_mid, y_mid)
                pt_br = intersect_h_v(bot_line[0], bot_line[1], right_line_straight[0], right_line_straight[1], x_mid, y_mid)
                pt_bl = intersect_h_v(bot_line[0], bot_line[1], left_line_straight[0], left_line_straight[1], x_mid, y_mid)
            else:
                pt_tl = get_intersection_piecewise(top_line[0], top_line[1], left_boundary, x_mid)
                pt_tr = get_intersection_piecewise(top_line[0], top_line[1], right_boundary, x_mid)
                pt_br = get_intersection_piecewise(bot_line[0], bot_line[1], right_boundary, x_mid)
                pt_bl = get_intersection_piecewise(bot_line[0], bot_line[1], left_boundary, x_mid)
                
            proposal_poly = [pt_tl, pt_tr, pt_br, pt_bl]
            
            p_x1 = min(pt[0] for pt in proposal_poly)
            p_y1 = min(pt[1] for pt in proposal_poly)
            p_x2 = max(pt[0] for pt in proposal_poly)
            p_y2 = max(pt[1] for pt in proposal_poly)
            
            ix1, iy1 = max(p_x1, ox1), max(p_y1, oy1)
            ix2, iy2 = min(p_x2, ox2), min(p_y2, oy2)
            inter = (ix2 - ix1) * (iy2 - iy1) if ix2 > ix1 and iy2 > iy1 else 0.0
            area_p = (p_x2 - p_x1) * (p_y2 - p_y1)
            area_o = (ox2 - ox1) * (oy2 - oy1)
            union = area_p + area_o - inter
            iou = inter / union if union > 0 else 0.0
            
            cx_p = sum(pt[0] for pt in proposal_poly) / 4.0
            cy_p = sum(pt[1] for pt in proposal_poly) / 4.0
            center_shift = math.sqrt((cx_p - cx_o)**2 + (cy_p - cy_o)**2)
            area_ratio = area_p / max(area_o, 1.0)
            
        ref_iou = ref_ious.get(ridx, 0.0)
        if proposal_ok:
            if iou > ref_iou + 0.005:
                note = f"IoU improved by +{(iou - ref_iou):.3f}"
            elif iou < ref_iou - 0.005:
                note = f"IoU decreased by -{(ref_iou - iou):.3f}"
            else:
                note = "IoU matches straight line"
        else:
            note = "failed"
            
        report_data.append({
            "raw_idx": ridx,
            "proposal_ok": proposal_ok,
            "reason": reason,
            "proposal_poly": proposal_poly,
            "iou": iou,
            "center_shift": center_shift,
            "area_ratio": area_ratio,
            "note": note
        })
    return report_data

def draw_debug_overlay(img, targets, final_horiz_lines, final_vert_lines, vertical_boundaries, report_data, roi_coords, out_path_probe, out_path_proposal, vertical_mode="straight"):
    """8. Draw overlays and save images."""
    min_x, max_x, min_y, max_y = roi_coords
    x_mid = (min_x + max_x) / 2.0
    y_mid = (min_y + max_y) / 2.0

    # A. Draw edge_line_probe
    img_probe = img.copy()
    for y_int, ang, _ in final_horiz_lines:
        rad = math.radians(ang)
        tan_a = math.tan(rad)
        pt1_x = min_x
        pt1_y = int(round(y_int + tan_a * (pt1_x - x_mid)))
        pt2_x = max_x
        pt2_y = int(round(y_int + tan_a * (pt2_x - x_mid)))
        cv2.line(img_probe, (pt1_x, pt1_y), (pt2_x, pt2_y), (0, 255, 255), 2, cv2.LINE_AA)
        
    for x_int, ang, _ in final_vert_lines:
        rad = math.radians(ang)
        if abs(ang) == 90.0:
            pt1_x = int(round(x_int))
            pt1_y = min_y
            pt2_x = int(round(x_int))
            pt2_y = max_y
        else:
            tan_a = math.tan(rad)
            pt1_y = min_y
            pt1_x = int(round(x_int + (pt1_y - y_mid) / tan_a))
            pt2_y = max_y
            pt2_x = int(round(x_int + (pt2_y - y_mid) / tan_a))
        cv2.line(img_probe, (pt1_x, pt1_y), (pt2_x, pt2_y), (255, 0, 0), 2, cv2.LINE_AA)
        
    for p in targets:
        ridx = p["raw_idx"]
        yolo_box = p.get("original_yolo_bbox") or p.get("bbox")
        poly = p.get("polygon")
        if yolo_box:
            cv2.rectangle(img_probe, (yolo_box[0], yolo_box[1]), (yolo_box[2], yolo_box[3]), (0, 0, 255), 1, cv2.LINE_AA)
        if poly and len(poly) >= 3:
            pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(img_probe, [pts], isClosed=True, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
        cx, cy = p.get("center", [0, 0])
        cv2.putText(img_probe, str(ridx), (int(cx) - 8, int(cy) + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1, cv2.LINE_AA)
                    
    OUT_PATH_PROBE.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path_probe), img_probe, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"Saved: {out_path_probe}")

    # B. Draw edge_grid_proposal
    img_proposal = img.copy()
    for y_int, ang, _ in final_horiz_lines:
        rad = math.radians(ang)
        tan_a = math.tan(rad)
        pt1_x = min_x
        pt1_y = int(round(y_int + tan_a * (pt1_x - x_mid)))
        pt2_x = max_x
        pt2_y = int(round(y_int + tan_a * (pt2_x - x_mid)))
        cv2.line(img_proposal, (pt1_x, pt1_y), (pt2_x, pt2_y), (200, 255, 255), 1, cv2.LINE_AA)
        
    if vertical_mode == "straight":
        for x_int, ang, _ in final_vert_lines:
            rad = math.radians(ang)
            if abs(ang) == 90.0:
                pt1_x = int(round(x_int))
                pt1_y = min_y
                pt2_x = int(round(x_int))
                pt2_y = max_y
            else:
                tan_a = math.tan(rad)
                pt1_y = min_y
                pt1_x = int(round(x_int + (pt1_y - y_mid) / tan_a))
                pt2_y = max_y
                pt2_x = int(round(x_int + (pt2_y - y_mid) / tan_a))
            cv2.line(img_proposal, (pt1_x, pt1_y), (pt2_x, pt2_y), (255, 0, 0), 2, cv2.LINE_AA)
    else:
        for b_pts in vertical_boundaries:
            pts_draw = np.array([[pt[1], pt[0]] for pt in b_pts], dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(img_proposal, [pts_draw], isClosed=False, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        
    for r in report_data:
        ridx = r["raw_idx"]
        p = [p for p in targets if p["raw_idx"] == ridx][0]
        yolo_box = p.get("original_yolo_bbox") or p.get("bbox")
        ox1, oy1, ox2, oy2 = yolo_box
        cx_o = (ox1 + ox2) / 2.0
        cy_o = (oy1 + oy2) / 2.0
        
        cv2.rectangle(img_proposal, (ox1, oy1), (ox2, oy2), (255, 255, 255), 1, cv2.LINE_AA)
        
        poly = p.get("polygon")
        if poly and len(poly) >= 3:
            pts_green = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(img_proposal, [pts_green], isClosed=True, color=(0, 180, 0), thickness=1, lineType=cv2.LINE_AA)
            
        if r["proposal_ok"]:
            pts_draw = np.array(r["proposal_poly"], dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(img_proposal, [pts_draw], isClosed=True, color=(0, 255, 255), thickness=2, lineType=cv2.LINE_AA)
            
        cv2.putText(img_proposal, str(ridx), (int(cx_o) - 8, int(cy_o) + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255) if r["proposal_ok"] else (0, 0, 255), 1, cv2.LINE_AA)
                    
    cv2.imwrite(str(out_path_proposal), img_proposal, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"Saved: {out_path_proposal}")

def propose_edge_grid_polygons(
    image_path,
    trace_path,
    target_raw_indices,
    config: EdgeGridConfig
) -> dict:
    """Public helper function to propose edge grid polygons for the target panels."""
    image_path = Path(image_path)
    trace_path = Path(trace_path)
    panels, targets, img = load_image_and_trace(image_path, trace_path, set(target_raw_indices))
    h, w, c = img.shape
    
    # 2. Build ROI
    roi_coords = build_roi_from_target_panels(targets, w, h, margin=config.roi_margin)
    
    # 3. Preprocess edges
    edges = preprocess_roi_for_edges(img, roi_coords)
    
    # 4. Detect horizontal lines
    final_horiz_lines = detect_horizontal_gaps(edges, roi_coords)
    
    # 5. Detect initial straight vertical lines
    final_vert_lines = detect_initial_vertical_boundaries(edges, roi_coords)
    
    # 6. Sample piecewise or segmented vertical boundaries
    vertical_boundaries = []
    vertical_mode = config.vertical_mode
    for line_v in final_vert_lines:
        if vertical_mode == "segmented": # 4 segments (default)
            smoothed_pts = fit_segmented_vertical_boundary(
                edges, line_v, roi_coords[2], roi_coords[3],
                n_segments=config.n_segments,
                search_half_width=config.search_half_width,
                min_points_per_segment=config.min_points_per_segment,
                max_angle_delta_deg=config.max_angle_delta_deg,
                max_x_jump_px=config.max_x_jump_px,
                min_x=roi_coords[0],
                max_dev_baseline_px=config.max_dev_baseline_px,
                smooth_window=config.smooth_window
            )
        elif vertical_mode == "segmented16": # 16 segments
            smoothed_pts = fit_segmented_vertical_boundary(
                edges, line_v, roi_coords[2], roi_coords[3],
                n_segments=16, search_half_width=8, min_points_per_segment=6,
                max_angle_delta_deg=5, max_x_jump_px=5, min_x=roi_coords[0],
                max_dev_baseline_px=8, smooth_window=5
            )
        elif vertical_mode == "segmented8": # 8 segments
            smoothed_pts = fit_segmented_vertical_boundary(
                edges, line_v, roi_coords[2], roi_coords[3],
                n_segments=8, search_half_width=10, min_points_per_segment=6,
                max_angle_delta_deg=8, max_x_jump_px=10, min_x=roi_coords[0],
                max_dev_baseline_px=None, smooth_window=3
            )
        else:
            smoothed_pts = sample_piecewise_vertical_boundary(
                edges, roi_coords, line_v, max_dev_px=6, smooth_window=7
            )
        vertical_boundaries.append(smoothed_pts)
        
    left_boundary = vertical_boundaries[0] if len(vertical_boundaries) > 0 else []
    right_boundary = vertical_boundaries[1] if len(vertical_boundaries) > 1 else []
    
    # 7. Build panel polygons
    left_line_straight = final_vert_lines[0] if len(final_vert_lines) > 0 else None
    right_line_straight = final_vert_lines[1] if len(final_vert_lines) > 1 else None
    
    report_data = build_panel_polygons_from_boundaries(
        targets, final_horiz_lines, left_boundary, right_boundary, roi_coords,
        left_line_straight=left_line_straight,
        right_line_straight=right_line_straight,
        vertical_mode=vertical_mode
    )
    
    # Format return dictionary
    proposals = []
    for r in report_data:
        proposals.append({
            "raw_idx": r["raw_idx"],
            "polygon": r["proposal_poly"],
            "source": f"edge_grid_segmented{config.n_segments}" if vertical_mode == "segmented" else f"edge_grid_{vertical_mode}",
            "ok": r["proposal_ok"],
            "metrics": {
                "iou": r["iou"],
                "center_shift": r["center_shift"],
                "area_ratio": r["area_ratio"],
                "note": r["note"]
            }
        })
        
    debug_dict = {
        "roi": roi_coords,
        "left_boundary": left_boundary,
        "right_boundary": right_boundary,
        "horizontal_lines": final_horiz_lines,
        "final_vert_lines": final_vert_lines,
        "vertical_boundaries": vertical_boundaries,
        "img": img,
        "targets": targets,
        "report_data": report_data
    }
    
    return {
        "proposals": proposals,
        "debug": debug_dict
    }

def run_edge_grid_piecewise_proposal(
    image_path,
    trace_path,
    target_raw_indices,
    output_prefix,
    roi_margin=45,
    max_dev_px=6,
    smooth_window=7,
    vertical_mode="segmented"
):
    # Construct EdgeGridConfig from the parameters passed
    config = EdgeGridConfig(
        roi_margin=roi_margin,
        vertical_mode=vertical_mode
    )
    # If the vertical_mode is segmented, we use our optimal 4-segment parameters
    if vertical_mode == "segmented":
        config.n_segments = 4
        config.max_x_jump_px = 10.0
        config.max_angle_delta_deg = 8.0
        config.search_half_width = 10
        config.min_points_per_segment = 6
        config.max_dev_baseline_px = 12.0
        config.smooth_window = 3

    # Call public API
    res = propose_edge_grid_polygons(image_path, trace_path, target_raw_indices, config)
    
    # Extract debug info for drawing overlays
    debug = res["debug"]
    img = debug["img"]
    targets = debug["targets"]
    final_horiz_lines = debug["horizontal_lines"]
    final_vert_lines = debug["final_vert_lines"]
    vertical_boundaries = debug["vertical_boundaries"]
    roi_coords = debug["roi"]
    report_data = debug["report_data"]
    
    # Suffix logic
    if vertical_mode == "segmented":
        mode_suffix = "segmented"
    elif vertical_mode == "segmented8":
        mode_suffix = "segmented8"
    elif vertical_mode == "segmented16":
        mode_suffix = "segmented16"
    else:
        mode_suffix = vertical_mode
        
    out_path_probe = Path(f"{output_prefix}_edge_line_probe_{mode_suffix}.JPG")
    out_path_proposal = Path(f"{output_prefix}_edge_grid_proposal_{mode_suffix}.JPG")
    
    draw_debug_overlay(img, targets, final_horiz_lines, final_vert_lines, vertical_boundaries, report_data, roi_coords, out_path_probe, out_path_proposal, vertical_mode)
    
    # Output report
    print("\n" + "="*80)
    print(f"vertical_mode: {vertical_mode} (suffix: {mode_suffix})")
    print(f"{'idx':>5}  {'proposal_ok':<12}  {'IoU':>6}  {'shift':>6}  {'area_r':>6}  {'Note'}")
    print("-"*80)
    for p in sorted(res["proposals"], key=lambda x: x["raw_idx"]):
        m = p["metrics"]
        print(
            f"{p['raw_idx']:>5}  "
            f"{str(p['ok']):<12}  "
            f"{m['iou']:>6.4f}  "
            f"{m['center_shift']:>6.2f}  "
            f"{m['area_ratio']:>6.2f}  "
            f"{m['note']}"
        )
    print("="*80)
    print(f"Sample points on Left vertical boundary: {len(debug['left_boundary'])}")
    print(f"Sample points on Right vertical boundary: {len(debug['right_boundary'])}")
    print(f"Output path (Probe): {out_path_probe}")
    print(f"Output path (Proposal): {out_path_proposal}")

if __name__ == "__main__":
    # Run segmented 4-segment mode (new default)
    run_edge_grid_piecewise_proposal(
        image_path="data/precalib/DJI_0029_R.JPG",
        trace_path="data/results/debug_logs/DJI_0029_R_final_polygon_trace.jsonl",
        target_raw_indices=[25, 40, 46, 47, 52, 62, 65, 67, 68, 77, 78, 86],
        output_prefix="data/results/debug/debug_DJI_0029_R_refactor_generalized",
        roi_margin=45,
        vertical_mode="segmented"
    )



