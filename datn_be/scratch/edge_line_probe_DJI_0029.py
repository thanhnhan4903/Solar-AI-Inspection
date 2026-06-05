import os
import json
import math
import cv2
import numpy as np
from pathlib import Path

# Paths
IMAGE_PATH = Path("data/precalib/DJI_0029_R.JPG")
TRACE_PATH = Path("data/results/debug_logs/DJI_0029_R_final_polygon_trace.jsonl")
OUT_PATH = Path("data/results/debug/debug_DJI_0029_R_edge_line_probe.JPG")

# 12 Target indices
TARGET_IDXS = {25, 40, 46, 47, 52, 62, 65, 67, 68, 77, 78, 86}

def main():
    # Load trace
    assert TRACE_PATH.exists(), f"Trace not found: {TRACE_PATH}"
    panels = []
    with open(TRACE_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                panels.append(json.loads(line))
                
    targets = [p for p in panels if p.get("raw_idx") in TARGET_IDXS]
    assert targets, "No target panels found in trace!"
    
    # Load image
    assert IMAGE_PATH.exists(), f"Image not found: {IMAGE_PATH}"
    img = cv2.imread(str(IMAGE_PATH))
    h, w, c = img.shape
    
    # Compute ROI with 40px margin
    all_x = []
    all_y = []
    for p in targets:
        box = p.get("bbox") or p.get("original_yolo_bbox")
        if box:
            all_x.extend([box[0], box[2]])
            all_y.extend([box[1], box[3]])
            
    min_x = max(0, min(all_x) - 40)
    max_x = min(w, max(all_x) + 40)
    min_y = max(0, min(all_y) - 40)
    max_y = min(h, max(all_y) + 40)
    
    print(f"ROI coordinates: x=[{min_x}, {max_x}], y=[{min_y}, {max_y}]")
    
    # Crop ROI
    roi_img = img[min_y:max_y, min_x:max_x]
    
    # Edge detection
    gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
    
    # Hough Lines P
    # rho=1, theta=1 deg, threshold=25, minLineLength=20, maxLineGap=10
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=25, minLineLength=20, maxLineGap=10)
    
    horiz_segments = []
    vert_segments = []
    
    x_mid = (min_x + max_x) / 2.0
    y_mid = (min_y + max_y) / 2.0
    
    if lines is not None:
        for line in lines:
            x1_roi, y1_roi, x2_roi, y2_roi = line[0]
            # Convert to global coordinates
            x1, y1 = x1_roi + min_x, y1_roi + min_y
            x2, y2 = x2_roi + min_x, y2_roi + min_y
            
            dx = x2 - x1
            dy = y2 - y1
            length = math.sqrt(dx*dx + dy*dy)
            if length < 1e-3:
                continue
                
            angle = math.degrees(math.atan2(dy, dx))
            # Normalize to [-90, 90]
            if angle > 90.0:
                angle -= 180.0
            elif angle < -90.0:
                angle += 180.0
                
            # Horizontal-ish: angle in [-25, 25]
            if abs(angle) <= 25.0:
                # Calculate Y-intercept at x_mid
                # y_mid = y1 + (x_mid - x1) * (dy / dx)
                y_int = y1 + (x_mid - x1) * (dy / dx) if abs(dx) > 1e-3 else y1
                horiz_segments.append((y_int, angle, length, (x1, y1, x2, y2)))
                
            # Vertical-ish: angle in [65, 90] or [-90, -65]
            elif abs(angle) >= 65.0:
                # Calculate X-intercept at y_mid
                # x_mid = x1 + (y_mid - y1) * (dx / dy)
                x_int = x1 + (y_mid - y1) * (dx / dy) if abs(dy) > 1e-3 else x1
                vert_segments.append((x_int, angle, length, (x1, y1, x2, y2)))
                
    # Clustering helper
    def cluster_segments(segments, threshold):
        if not segments:
            return []
        # Sort by intercept
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

    # Group close parallel segments (threshold = 12 px)
    horiz_clusters = cluster_segments(horiz_segments, 12.0)
    vert_clusters = cluster_segments(vert_segments, 12.0)
    
    final_horiz_lines = []
    final_vert_lines = []
    
    # Process horizontal clusters
    for c in horiz_clusters:
        total_len = sum(s[2] for s in c)
        if total_len < 25.0:  # Filter noise
            continue
        # Weighted or median values
        med_y_int = float(np.median([s[0] for s in c]))
        med_ang = float(np.median([s[1] for s in c]))
        final_horiz_lines.append((med_y_int, med_ang, total_len))
        
    # Process vertical clusters
    for c in vert_clusters:
        total_len = sum(s[2] for s in c)
        if total_len < 25.0:  # Filter noise
            continue
        med_x_int = float(np.median([s[0] for s in c]))
        med_ang = float(np.median([s[1] for s in c]))
        final_vert_lines.append((med_x_int, med_ang, total_len))
        
    print(f"Found {len(final_horiz_lines)} horizontal lines and {len(final_vert_lines)} vertical lines after filtering.")
    
    # Overlay on original image
    dbg_img = img.copy()
    
    # Draw horizontal consensus lines (Yellow)
    for y_int, ang, _ in final_horiz_lines:
        rad = math.radians(ang)
        tan_a = math.tan(rad)
        # Find points at x=min_x and x=max_x
        pt1_x = min_x
        pt1_y = int(round(y_int + tan_a * (pt1_x - x_mid)))
        pt2_x = max_x
        pt2_y = int(round(y_int + tan_a * (pt2_x - x_mid)))
        cv2.line(dbg_img, (pt1_x, pt1_y), (pt2_x, pt2_y), (0, 255, 255), 2, cv2.LINE_AA)
        
    # Draw vertical consensus lines (Blue)
    for x_int, ang, _ in final_vert_lines:
        rad = math.radians(ang)
        # Find points at y=min_y and y=max_y
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
            
        cv2.line(dbg_img, (pt1_x, pt1_y), (pt2_x, pt2_y), (255, 0, 0), 2, cv2.LINE_AA)
        
    # Draw original YOLO bbox (Red) and final polygon (Green)
    for p in targets:
        ridx = p["raw_idx"]
        yolo_box = p.get("original_yolo_bbox") or p.get("bbox")
        poly = p.get("polygon")
        
        # Draw YOLO box in Red
        if yolo_box:
            bx1, by1, bx2, by2 = yolo_box
            cv2.rectangle(dbg_img, (bx1, by1), (bx2, by2), (0, 0, 255), 1, cv2.LINE_AA)
            
        # Draw current final polygon in Green
        if poly and len(poly) >= 3:
            pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(dbg_img, [pts], isClosed=True, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
            
        # Label raw_idx
        cx, cy = p.get("center", [0, 0])
        cv2.putText(dbg_img, str(ridx), (int(cx) - 8, int(cy) + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1, cv2.LINE_AA)
                    
    # Save image
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT_PATH), dbg_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"Saved: {OUT_PATH}")
    
    # Calculate median angles
    h_angles = [line[1] for line in final_horiz_lines]
    v_angles = [line[1] for line in final_vert_lines]
    
    med_h_angle = float(np.median(h_angles)) if h_angles else 0.0
    med_v_angle = float(np.median(v_angles)) if v_angles else 0.0
    
    # Output metrics
    print("\n--- Diagnostic Report ---")
    print(f"Horizontal lines found: {len(final_horiz_lines)}")
    print(f"Vertical lines found: {len(final_vert_lines)}")
    print(f"Median horizontal angle: {med_h_angle:.2f} degrees")
    print(f"Median vertical angle: {med_v_angle:.2f} degrees")
    print(f"Output path: {OUT_PATH}")

if __name__ == "__main__":
    main()
