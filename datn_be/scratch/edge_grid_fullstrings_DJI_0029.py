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

def cluster_panels_into_strings(panels, dist_threshold=40):
    """Clusters panels into columns/strings using greedy X-center grouping."""
    # Compute X center for each panel
    panels_with_cx = []
    for p in panels:
        x1, y1, x2, y2 = p["bbox"]
        cx = (x1 + x2) / 2.0
        panels_with_cx.append((cx, p))
        
    # Sort by X center
    panels_with_cx.sort(key=lambda x: x[0])
    
    strings = []
    for cx, p in panels_with_cx:
        if not strings:
            strings.append([p])
        else:
            # Check distance to mean X of last group
            mean_x = np.mean([(pt["bbox"][0] + pt["bbox"][2]) / 2.0 for pt in strings[-1]])
            if cx - mean_x > dist_threshold:
                strings.append([p])
            else:
                strings[-1].append(p)
                
    # Sort panels in each string by Y coordinate (top-to-bottom)
    for s in strings:
        s.sort(key=lambda p: (p["bbox"][1] + p["bbox"][3]) / 2.0)
        
    # Sort the strings themselves from left to right
    strings.sort(key=lambda s: np.mean([(pt["bbox"][0] + pt["bbox"][2]) / 2.0 for pt in s]))
    
    return strings

def fit_local_vertical_line(edges, x_base, y_start, y_end, search_half_width=12):
    """
    Fits a local vertical line x = a*y + b in the search band around x_base.
    Returns: (a, b) coefficient or None if failed.
    """
    h_img, w_img = edges.shape[:2]
    ys = []
    xs = []
    
    y_start_int = int(max(0, math.floor(y_start)))
    y_end_int = int(min(h_img - 1, math.ceil(y_end)))
    
    for y in range(y_start_int, y_end_int + 1):
        x_min = int(max(0, math.floor(x_base - search_half_width)))
        x_max = int(min(w_img - 1, math.ceil(x_base + search_half_width)))
        
        for x in range(x_min, x_max + 1):
            if edges[y, x] == 255:
                ys.append(y)
                xs.append(x)
                
    if len(xs) >= 10:
        # Fit linear regression x = a * y + b
        try:
            coeffs = np.polyfit(ys, xs, 1)
            a, b = coeffs[0], coeffs[1]
            angle_deg = abs(np.degrees(np.arctan(1.0 / a if a != 0 else 9999)))
            # The line should be close to vertical (e.g. angle in [65, 115] degrees)
            if 65.0 <= angle_deg <= 115.0:
                return a, b
        except Exception:
            pass
            
    # Fallback to straight vertical
    return 0.0, float(x_base)

def fit_string_vertical_edges(edges, string_panels, n_segments=4):
    """
    Divides the string into n_segments along Y and fits local vertical lines for left/right edges.
    Returns: (left_local_lines, right_local_lines)
    Each line is represented as a tuple of (a, b, y_start, y_end)
    """
    y_min = min(p["bbox"][1] for p in string_panels)
    y_max = max(p["bbox"][3] for p in string_panels)
    y_step = (y_max - y_min) / n_segments
    
    left_local_lines = []
    right_local_lines = []
    
    for k in range(n_segments):
        seg_y_start = y_min + k * y_step
        seg_y_end = seg_y_start + y_step
        
        # Get panels in this segment to determine baseline X
        seg_panels = [p for p in string_panels if (p["bbox"][1]+p["bbox"][3])/2.0 >= seg_y_start and (p["bbox"][1]+p["bbox"][3])/2.0 <= seg_y_end]
        if not seg_panels:
            seg_panels = string_panels # fallback
            
        base_x_left = np.median([p["bbox"][0] for p in seg_panels])
        base_x_right = np.median([p["bbox"][2] for p in seg_panels])
        
        a_l, b_l = fit_local_vertical_line(edges, base_x_left, seg_y_start, seg_y_end)
        a_r, b_r = fit_local_vertical_line(edges, base_x_right, seg_y_start, seg_y_end)
        
        left_local_lines.append((a_l, b_l, seg_y_start, seg_y_end))
        right_local_lines.append((a_r, b_r, seg_y_start, seg_y_end))
        
    return left_local_lines, right_local_lines

def build_final_boundary(local_lines, y_min, y_max, n_segments=4, max_dev_px=4.0):
    """
    Combines 4 local vertical lines into a single smooth boundary (straight/slanted or segmented polyline).
    Returns: (polyline_points, boundary_type)
    polyline_points is a list of (x, y) coordinates.
    boundary_type is either "straight/slanted" or "segmented"
    """
    y_step = (y_max - y_min) / n_segments
    y_boundaries = [y_min + k * y_step for k in range(n_segments + 1)]
    
    # 1. Fit a single global straight line through the midpoint of each local line
    mid_points = []
    for a, b, ys, ye in local_lines:
        y_mid = (ys + ye) / 2.0
        x_mid = a * y_mid + b
        mid_points.append((x_mid, y_mid))
        
    ys_arr = np.array([p[1] for p in mid_points])
    xs_arr = np.array([p[0] for p in mid_points])
    global_a, global_b = np.polyfit(ys_arr, xs_arr, 1)
    
    # Check max deviation of the local midpoints from the global line
    max_dev = 0.0
    for x_mid, y_mid in mid_points:
        x_glob = global_a * y_mid + global_b
        max_dev = max(max_dev, abs(x_mid - x_glob))
        
    if max_dev <= max_dev_px:
        # Use single straight line
        pts = [(int(round(global_a * y + global_b)), int(round(y))) for y in y_boundaries]
        return pts, "straight/slanted"
    else:
        # Use segmented polyline
        pts = []
        for idx, y in enumerate(y_boundaries):
            if idx == 0:
                # Top of first segment
                a, b, _, _ = local_lines[0]
                x = a * y + b
            elif idx == n_segments:
                # Bottom of last segment
                a, b, _, _ = local_lines[-1]
                x = a * y + b
            else:
                # Boundary between segment idx-1 and idx: average the two fits
                a1, b1, _, _ = local_lines[idx - 1]
                a2, b2, _, _ = local_lines[idx]
                x = ((a1 * y + b1) + (a2 * y + b2)) / 2.0
            pts.append((int(round(x)), int(round(y))))
        return pts, "segmented"

def detect_horizontal_gap_lines(gray, string_panels):
    """
    Detects clean, evenly-spaced horizontal gap lines matching panel boundaries.
    Uses Sobel-Y projection peaks within localized search bands.
    """
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    sobel_y_abs = np.abs(sobel_y)
    
    gap_lines = []
    
    # Sort panels from top to bottom
    n_panels = len(string_panels)
    
    # We will search for:
    # 1. Top edge of the first panel
    # 2. Gaps between consecutive panels
    # 3. Bottom edge of the last panel
    
    search_regions = []
    
    # Top edge of first panel
    y_top = string_panels[0]["bbox"][1]
    search_regions.append((y_top - 6, y_top + 6))
    
    # Gaps between panels
    for i in range(n_panels - 1):
        y_bottom_curr = string_panels[i]["bbox"][3]
        y_top_next = string_panels[i+1]["bbox"][1]
        y_mid = (y_bottom_curr + y_top_next) / 2.0
        search_regions.append((y_mid - 8, y_mid + 8))
        
    # Bottom edge of last panel
    y_bottom_last = string_panels[-1]["bbox"][3]
    search_regions.append((y_bottom_last - 6, y_bottom_last + 6))
    
    h_img, w_img = gray.shape[:2]
    
    for y_start, y_end in search_regions:
        y_start_int = int(max(0, math.floor(y_start)))
        y_end_int = int(min(h_img - 1, math.ceil(y_end)))
        
        if y_start_int >= y_end_int:
            continue
            
        # Determine average X span of the string in this Y range
        overlapping_panels = [p for p in string_panels if not (p["bbox"][3] < y_start_int or p["bbox"][1] > y_end_int)]
        if not overlapping_panels:
            overlapping_panels = string_panels
            
        x_min = int(max(0, min(p["bbox"][0] for p in overlapping_panels) - 5))
        x_max = int(min(w_img - 1, max(p["bbox"][2] for p in overlapping_panels) + 5))
        
        # Calculate Y projection intensity in this band
        intensities = []
        for y_coord in range(y_start_int, y_end_int + 1):
            val = np.mean(sobel_y_abs[y_coord, x_min:x_max])
            intensities.append((val, y_coord))
            
        # Select the Y coordinate with maximum gradient intensity
        if intensities:
            best_y = max(intensities, key=lambda x: x[0])[1]
            gap_lines.append((best_y, x_min, x_max))
            
    return gap_lines

def draw_polyline(img, pts, color, thickness=2):
    for idx in range(len(pts) - 1):
        cv2.line(img, pts[idx], pts[idx+1], color, thickness, lineType=cv2.LINE_AA)

def main():
    # 1. Load image and detections
    img_path = extract_input_image()
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Could not read image at {img_path}")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray)
    blur = cv2.GaussianBlur(gray_clahe, (5, 5), 0)
    edges = cv2.Canny(blur, 40, 120, apertureSize=3)
    
    panels = load_panels_from_logs()
    strings = cluster_panels_into_strings(panels)
    
    # 2. Initialize debug images
    img_raw = img.copy()
    img_final = img.copy()
    img_colored = img.copy()
    
    console_stats = []
    
    # Process each string independently
    for str_idx, string_panels in enumerate(strings):
        string_name = f"string_{str_idx + 1}"
        y_min = min(p["bbox"][1] for p in string_panels)
        y_max = max(p["bbox"][3] for p in string_panels)
        
        # Determine ROI for visual aid
        x_min_roi = min(p["bbox"][0] for p in string_panels) - 10
        x_max_roi = max(p["bbox"][2] for p in string_panels) + 10
        
        # A. Fit local vertical lines (4 segments each side)
        left_locals, right_locals = fit_string_vertical_edges(edges, string_panels, n_segments=4)
        
        # B. Combine into final left and right boundaries
        left_boundary, left_type = build_final_boundary(left_locals, y_min, y_max, n_segments=4)
        right_boundary, right_type = build_final_boundary(right_locals, y_min, y_max, n_segments=4)
        
        # C. Find horizontal gap lines
        final_horiz_gaps = detect_horizontal_gap_lines(gray, string_panels)
        
        # --- DRAW ON IMAGE 1: RAW ---
        # Draw string ROI
        cv2.rectangle(img_raw, (x_min_roi, y_min), (x_max_roi, y_max), (0, 128, 0), 1)
        # Draw 4 local vertical lines for left/right edges
        for (a, b, ys, ye) in left_locals:
            x_s, x_e = int(round(a*ys+b)), int(round(a*ye+b))
            cv2.line(img_raw, (x_s, int(ys)), (x_e, int(ye)), (255, 0, 255), 1)
        for (a, b, ys, ye) in right_locals:
            x_s, x_e = int(round(a*ys+b)), int(round(a*ye+b))
            cv2.line(img_raw, (x_s, int(ys)), (x_e, int(ye)), (255, 0, 255), 1)
        # Draw all horizontal candidates in this string
        for y, x_start, x_end in final_horiz_gaps:
            cv2.line(img_raw, (x_start, y), (x_end, y), (0, 255, 255), 1)
        # Label string
        cv2.putText(img_raw, string_name, (x_min_roi, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
            
        # --- DRAW ON IMAGE 2: FINAL ---
        # Draw final horizontal lines
        for y, x_start, x_end in final_horiz_gaps:
            cv2.line(img_final, (x_start, y), (x_end, y), (0, 255, 0), 2)
        # Draw final left/right boundaries
        draw_polyline(img_final, left_boundary, (255, 0, 0), 2)
        draw_polyline(img_final, right_boundary, (255, 0, 0), 2)
        # Label string
        cv2.putText(img_final, string_name, (x_min_roi, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        
        # --- DRAW ON IMAGE 3: COLORED ---
        # Horizontal gap lines: Yellow (0, 255, 255)
        for y, x_start, x_end in final_horiz_gaps:
            cv2.line(img_colored, (x_start, y), (x_end, y), (0, 255, 255), 2)
        # Local vertical 4 segments: Light Cyan (255, 255, 200)
        for (a, b, ys, ye) in left_locals:
            x_s, x_e = int(round(a*ys+b)), int(round(a*ye+b))
            cv2.line(img_colored, (x_s, int(ys)), (x_e, int(ye)), (255, 255, 200), 1)
        for (a, b, ys, ye) in right_locals:
            x_s, x_e = int(round(a*ys+b)), int(round(a*ye+b))
            cv2.line(img_colored, (x_s, int(ys)), (x_e, int(ye)), (255, 255, 200), 1)
        # Final vertical boundaries: Dark Blue (180, 0, 0)
        draw_polyline(img_colored, left_boundary, (180, 0, 0), 2)
        draw_polyline(img_colored, right_boundary, (180, 0, 0), 2)
        # Label string
        cv2.putText(img_colored, string_name, (x_min_roi, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 0, 0), 1, cv2.LINE_AA)
        
        console_stats.append({
            "name": string_name,
            "n_horizontal": len(final_horiz_gaps),
            "left_local_count": len(left_locals),
            "right_local_count": len(right_locals),
            "left_type": left_type,
            "right_type": right_type
        })
        
    # 3. Save images
    out_dir = Path("data/results/debug")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    cv2.imwrite(str(out_dir / "debug_DJI_0029_R_string_lines_raw.JPG"), img_raw)
    cv2.imwrite(str(out_dir / "debug_DJI_0029_R_string_lines_final.JPG"), img_final)
    cv2.imwrite(str(out_dir / "debug_DJI_0029_R_string_lines_colored.JPG"), img_colored)
    
    # 4. Print stats
    print("\n================ STATISTICS ================")
    print(f"Total strings detected: {len(strings)}")
    print("--------------------------------------------")
    for s in console_stats:
        print(f"String: {s['name']}")
        print(f"  - Final horizontal lines: {s['n_horizontal']}")
        print(f"  - Left edge local fits:  {s['left_local_count']} (Type: {s['left_type']})")
        print(f"  - Right edge local fits: {s['right_local_count']} (Type: {s['right_type']})")
    print("============================================")

if __name__ == "__main__":
    main()
