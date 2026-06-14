import os
import zipfile
import numpy as np
import cv2
from pathlib import Path

def extract_input_image():
    """Extracts DJI_0029_R.JPG from Dataset.zip if it does not exist."""
    target_path = Path("data/precalib/DJI_0029_R.JPG")
    if target_path.exists():
        print(f"Input image already exists at: {target_path}")
        return target_path

    # Ensure parent directory exists
    target_path.parent.mkdir(parents=True, exist_ok=True)

    zip_path = Path("../Dataset.zip")
    if not zip_path.exists():
        zip_path = Path("Dataset.zip") # Try current directory

    if not zip_path.exists():
        # Search parent directories
        for p in Path(".").resolve().parents:
            candidate = p / "Dataset.zip"
            if candidate.exists():
                zip_path = candidate
                break

    if not zip_path.exists():
        raise FileNotFoundError("Could not locate Dataset.zip in workspace to extract the test image.")

    print(f"Extracting DJI_0029_R from {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        # Find the 0029 image
        match = None
        for name in z.namelist():
            if "DJI_0029_R_JPG" in name and name.endswith(".jpg"):
                match = name
                break
        
        if not match:
            raise ValueError("Could not find DJI_0029_R image in Dataset.zip")

        # Read and write
        img_data = z.read(match)
        with open(target_path, "wb") as f:
            f.write(img_data)
        print(f"Successfully extracted and saved to {target_path}")
    
    return target_path

def get_line_angle(line):
    x1, y1, x2, y2 = line
    dx = x2 - x1
    dy = y2 - y1
    angle = np.arctan2(dy, dx) * 180.0 / np.pi
    # Normalize to [0, 180)
    return angle % 180.0

def line_projection_dist_and_gap(line1, line2):
    """
    Returns:
      perp_dist: average perpendicular distance from line2 endpoints to the infinite line of line1
      gap: gap distance between projections of line1 and line2 on line1's direction
    """
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    # Vector of line1
    v = np.array([x2 - x1, y2 - y1], dtype=np.float32)
    len_v = np.linalg.norm(v)
    if len_v < 1e-5:
        return 999, 999
    u = v / len_v # unit vector

    # Normal vector of line1
    n = np.array([-u[1], u[0]], dtype=np.float32)

    # Projections of endpoints of line1 onto u
    p1 = np.array([x1, y1], dtype=np.float32)
    p2 = np.array([x2, y2], dtype=np.float32)
    proj1_1 = np.dot(p1, u)
    proj1_2 = np.dot(p2, u)
    min_proj1 = min(proj1_1, proj1_2)
    max_proj1 = max(proj1_1, proj1_2)

    # Projections of endpoints of line2 onto u and n
    p3 = np.array([x3, y3], dtype=np.float32)
    p4 = np.array([x4, y4], dtype=np.float32)
    
    dist3 = np.dot(p3 - p1, n)
    dist4 = np.dot(p4 - p1, n)
    perp_dist = (np.abs(dist3) + np.abs(dist4)) / 2.0

    proj2_1 = np.dot(p3, u)
    proj2_2 = np.dot(p4, u)
    min_proj2 = min(proj2_1, proj2_2)
    max_proj2 = max(proj2_1, proj2_2)

    # Calculate gap
    if max_proj1 < min_proj2:
        gap = min_proj2 - max_proj1
    elif max_proj2 < min_proj1:
        gap = min_proj1 - max_proj2
    else:
        gap = 0.0 # overlap

    return perp_dist, gap

def merge_two_lines(line1, line2, is_horizontal=True):
    # Sort endpoints to find extreme points
    pts = [line1[:2], line1[2:], line2[:2], line2[2:]]
    if is_horizontal:
        pts = sorted(pts, key=lambda p: p[0])
    else:
        pts = sorted(pts, key=lambda p: p[1])
    return [int(pts[0][0]), int(pts[0][1]), int(pts[-1][0]), int(pts[-1][1])]

def merge_lines_list(lines, is_horizontal=True, perp_dist_thresh=12, gap_thresh=60, angle_diff_thresh=10):
    if len(lines) == 0:
        return []
        
    merged_any = True
    current_lines = [list(l) for l in lines]
    
    while merged_any:
        merged_any = False
        n_lines = len(current_lines)
        used = [False] * n_lines
        new_lines = []
        
        for i in range(n_lines):
            if used[i]:
                continue
            for j in range(i + 1, n_lines):
                if used[j]:
                    continue
                
                # Check angle diff
                ang1 = get_line_angle(current_lines[i])
                ang2 = get_line_angle(current_lines[j])
                ang_diff = abs(ang1 - ang2)
                ang_diff = min(ang_diff, 180.0 - ang_diff)
                
                if ang_diff > angle_diff_thresh:
                    continue
                
                # Check perpendicular distance and gap
                d, g = line_projection_dist_and_gap(current_lines[i], current_lines[j])
                if d < perp_dist_thresh and g < gap_thresh:
                    # Merge them
                    merged = merge_two_lines(current_lines[i], current_lines[j], is_horizontal)
                    current_lines[i] = merged
                    used[j] = True
                    merged_any = True
                    
            new_lines.append(current_lines[i])
            
        current_lines = [l for idx, l in enumerate(new_lines) if not used[idx]]
        
    return current_lines

def main():
    # 1. Load image
    img_path = extract_input_image()
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Could not read image at {img_path}")
    
    print(f"Loaded image size: {img.shape}")
    
    # 2. Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 3. CLAHE and Gaussian Blur
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray)
    blur = cv2.GaussianBlur(gray_clahe, (5, 5), 0)
    
    # 4. Canny Edge Detection
    edges = cv2.Canny(blur, 40, 120, apertureSize=3)
    
    # 5. Detect raw line segments
    # Use HoughLinesP
    raw_lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=40, minLineLength=25, maxLineGap=10)
    if raw_lines is None:
        raw_lines = []
    else:
        raw_lines = [l[0] for l in raw_lines]
        
    print(f"Total raw segments detected: {len(raw_lines)}")
    
    # 6. Separate into horizontal-ish and vertical-ish
    raw_horiz = []
    raw_vert = []
    
    for l in raw_lines:
        ang = get_line_angle(l)
        # Horizontal: close to 0 or 180
        # Vertical: close to 90
        diff_horiz = min(ang, 180.0 - ang)
        diff_vert = abs(ang - 90.0)
        
        if diff_horiz < 30.0:
            raw_horiz.append(l)
        elif diff_vert < 30.0:
            raw_vert.append(l)
            
    # 7. Merge lines
    # Horizontal line merging parameters
    merged_horiz = merge_lines_list(raw_horiz, is_horizontal=True, perp_dist_thresh=8, gap_thresh=50, angle_diff_thresh=8)
    # Vertical line merging parameters
    merged_vert = merge_lines_list(raw_vert, is_horizontal=False, perp_dist_thresh=12, gap_thresh=40, angle_diff_thresh=8)
    
    # 8. Save output directories
    out_dir = Path("data/results/debug")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # - Debug Image 1: Raw segments
    img_raw = img.copy()
    for l in raw_lines:
        cv2.line(img_raw, (l[0], l[1]), (l[2], l[3]), (0, 255, 0), 1)
    raw_out_path = out_dir / "debug_DJI_0029_R_all_lines_raw.JPG"
    cv2.imwrite(str(raw_out_path), img_raw)
    
    # - Debug Image 2: Merged segments
    img_merged = img.copy()
    for l in merged_horiz:
        cv2.line(img_merged, (l[0], l[1]), (l[2], l[3]), (0, 255, 0), 2)
    for l in merged_vert:
        cv2.line(img_merged, (l[0], l[1]), (l[2], l[3]), (255, 0, 0), 2)
    merged_out_path = out_dir / "debug_DJI_0029_R_all_lines_merged.JPG"
    cv2.imwrite(str(merged_out_path), img_merged)
    
    # - Debug Image 3: Colored segments (Horizontal = Yellow, Vertical = Blue)
    img_colored = img.copy()
    for l in merged_horiz:
        cv2.line(img_colored, (l[0], l[1]), (l[2], l[3]), (0, 255, 255), 2) # Yellow
    for l in merged_vert:
        cv2.line(img_colored, (l[0], l[1]), (l[2], l[3]), (255, 0, 0), 2) # Blue (BGR Red is Blue in custom color request, wait. BGR is Blue, Green, Red. Yellow is (0, 255, 255) in BGR. Blue is (255, 0, 0) in BGR.)
    colored_out_path = out_dir / "debug_DJI_0029_R_all_lines_colored.JPG"
    cv2.imwrite(str(colored_out_path), img_colored)
    
    # 9. Print stats
    print("\n================ STATS ================")
    print(f"Raw Horizontal Segments: {len(raw_horiz)}")
    print(f"Raw Vertical Segments:   {len(raw_vert)}")
    print(f"Merged Horizontal Lines: {len(merged_horiz)}")
    print(f"Merged Vertical Lines:   {len(merged_vert)}")
    print("=======================================")
    print(f"Saved raw debug:     {raw_out_path}")
    print(f"Saved merged debug:  {merged_out_path}")
    print(f"Saved colored debug: {colored_out_path}")

if __name__ == "__main__":
    main()
