import cv2
import sys
from pathlib import Path
import json
import math
import numpy as np

# Load panels
IMAGE_STEM = "DJI_0845_R"
log_path = Path(f"data/results/debug_logs/{IMAGE_STEM}_panel_refine.jsonl")
panels = []
with open(log_path, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            panels.append(json.loads(line))

# group into columns
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

x_blocks = []
i = 0
n_cols = len(columns)
while i < n_cols:
    col_curr = columns[i]
    cx_curr = np.mean([(pt["bbox"][0] + pt["bbox"][2]) / 2.0 for pt in col_curr])
    if i + 1 < n_cols:
        col_next = columns[i + 1]
        cx_next = np.mean([(pt["bbox"][0] + pt["bbox"][2]) / 2.0 for pt in col_next])
        if cx_next - cx_curr < 70:
            x_blocks.append({
                "columns": [col_curr, col_next],
                "panels": col_curr + col_next
            })
            i += 2
            continue
    x_blocks.append({
        "columns": [col_curr],
        "panels": col_curr
    })
    i += 1

def split_block_by_y_gap(block):
    all_panels = sorted(block["panels"], key=lambda p: p["bbox"][1])
    heights = [p["bbox"][3] - p["bbox"][1] for p in all_panels]
    median_h = float(np.median(heights)) if heights else 30.0
    gap_threshold = max(45.0, 1.8 * median_h)
    
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
            
    row_clusters.sort(key=lambda rc: np.mean([(p["bbox"][1] + p["bbox"][3]) / 2.0 for p in rc]))
    
    split_indices = []
    for k in range(len(row_clusters) - 1):
        bottom_k = max(p["bbox"][3] for p in row_clusters[k])
        top_k1 = min(p["bbox"][1] for p in row_clusters[k + 1])
        gap = top_k1 - bottom_k
        if gap > gap_threshold:
            split_indices.append(k + 1)
            
    if not split_indices:
        return [block]
        
    all_split_points = [0] + split_indices + [len(row_clusters)]
    sub_row_groups = []
    for i in range(len(all_split_points) - 1):
        start_idx = all_split_points[i]
        end_idx = all_split_points[i + 1]
        sub_rows = row_clusters[start_idx:end_idx]
        sub_panels = [p for rc in sub_rows for p in rc]
        sub_row_groups.append(sub_panels)
        
    sub_blocks = []
    for sub_panels_list in sub_row_groups:
        if len(sub_panels_list) < 2:
            continue
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
        sub_blocks.append({
            "columns": sub_cols,
            "panels": sub_panels_list
        })
    return sub_blocks

blocks = []
for xb in x_blocks:
    blocks.extend(split_block_by_y_gap(xb))

b = blocks[4]

# Rotation deg calculation from columns
block_angles = []
for col in b["columns"]:
    if len(col) >= 2:
        cxs = [(p["bbox"][0] + p["bbox"][2]) / 2.0 for p in col]
        cys = [(p["bbox"][1] + p["bbox"][3]) / 2.0 for p in col]
        slope, _ = np.polyfit(cys, cxs, 1)
        block_angles.append(math.atan(slope))
rotation_rad = np.median(block_angles) if block_angles else 0.0
rotation_deg = np.degrees(rotation_rad)

# ROI coordinates
bx_min = min(p["bbox"][0] for p in b["panels"])
bx_max = max(p["bbox"][2] for p in b["panels"])
by_min = min(p["bbox"][1] for p in b["panels"])
by_max = max(p["bbox"][3] for p in b["panels"])

margin = 60
roi_x1 = max(0, int(bx_min - margin))
roi_y1 = max(0, int(by_min - margin))
roi_x2 = int(bx_max + margin)
roi_y2 = int(by_max + margin)
roi_w = roi_x2 - roi_x1
roi_h = roi_y2 - roi_y1

center_x = roi_w / 2.0
center_y = roi_h / 2.0
M = cv2.getRotationMatrix2D((center_x, center_y), -rotation_deg, 1.0)

def transform_orig_to_local(x, y):
    x_rel = float(x) - roi_x1
    y_rel = float(y) - roi_y1
    x_local = M[0, 0] * x_rel + M[0, 1] * y_rel + M[0, 2]
    y_local = M[1, 0] * x_rel + M[1, 1] * y_rel + M[1, 2]
    return float(x_local), float(y_local)

# Transform panels to rotated local coordinates
b_local = {"columns": [], "panels": []}
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

import edge_grid_full_image_test as eg
eg.IMAGE_STEM = "DJI_0845_R"

N_c = len(b_local["columns"])
N_r = max(len(col) for col in b_local["columns"])

angles = []
for col in b_local["columns"]:
    for p in col:
        TL, TR, BR, BL = eg.sort_panel_corners(p["refined_polygon"])
        angles.append(math.atan2(TR[1] - TL[1], TR[0] - TL[0]))
        angles.append(math.atan2(BR[1] - BL[1], BR[0] - BL[0]))
angles_deg = [np.degrees(a) for a in angles]
med_angle = np.median(angles_deg)
clean_angles = [a for a in angles_deg if abs(a - med_angle) <= 8.0]
theta_local_deg = np.mean(clean_angles) if clean_angles else med_angle
theta_local_rad = np.radians(theta_local_deg)

all_pts = []
for col in b_local["columns"]:
    for p in col:
        all_pts.extend(p["refined_polygon"])
xs = [pt[0] for pt in all_pts]
ys = [pt[1] for pt in all_pts]
x_c = np.mean(xs)
y_c = np.mean(ys)

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

local_corners = {}
for col_idx, col in enumerate(b_local["columns"]):
    local_corners[col_idx] = {}
    for row_idx, p in enumerate(col):
        poly_init = eg.get_local_yolo_polygon_prior(p["orig_ref"], transform_orig_to_local)
        TL, TR, BR, BL = eg.sort_panel_corners(poly_init)
        local_corners[col_idx][row_idx] = (
            project_to_local(TL),
            project_to_local(TR),
            project_to_local(BR),
            project_to_local(BL)
        )

H_lines = {}
for r in range(N_r + 1):
    if r == 0:
        v_vals = [local_corners[c][0][0][1] for c in range(N_c)] + [local_corners[c][0][1][1] for c in range(N_c)]
    elif r == N_r:
        v_vals = [local_corners[c][N_r-1][2][1] for c in range(N_c)] + [local_corners[c][N_r-1][3][1] for c in range(N_c)]
    else:
        v_vals = ([local_corners[c][r-1][2][1] for c in range(N_c)] + [local_corners[c][r-1][3][1] for c in range(N_c)] +
                  [local_corners[c][r][0][1] for c in range(N_c)] + [local_corners[c][r][1][1] for c in range(N_c)])
    v_init = float(np.median(v_vals))
    H_lines[r] = {'m': 0.0, 'c': v_init, 'v_init': v_init, 'source': 'yolo'}

V_lines = {}
for c in range(N_c + 1):
    if c == 0:
        u_vals = [local_corners[0][r][0][0] for r in range(len(b_local["columns"][0]))] + [local_corners[0][r][3][0] for r in range(len(b_local["columns"][0]))]
    elif c == N_c:
        u_vals = [local_corners[N_c-1][r][1][0] for r in range(len(b_local["columns"][N_c-1]))] + [local_corners[N_c-1][r][2][0] for r in range(len(b_local["columns"][N_c-1]))]
    else:
        u_vals = ([local_corners[c-1][r][1][0] for r in range(len(b_local["columns"][c-1]))] + [local_corners[c-1][r][2][0] for r in range(len(b_local["columns"][c-1]))] +
                  [local_corners[c][r][0][0] for r in range(len(b_local["columns"][c]))] + [local_corners[c][r][3][0] for r in range(len(b_local["columns"][c]))])
    u_init = float(np.median(u_vals))
    V_lines[c] = {'m': 0.0, 'c': u_init, 'u_init': u_init, 'source': 'yolo'}

def build_proposals(h_lines, v_lines):
    props = {}
    for col_idx in range(N_c):
        for row_idx in range(len(b_local["columns"][col_idx])):
            tl = eg.intersect_lines_uv(h_lines[row_idx]['m'], h_lines[row_idx]['c'], v_lines[col_idx]['m'], v_lines[col_idx]['c'])
            tr = eg.intersect_lines_uv(h_lines[row_idx]['m'], h_lines[row_idx]['c'], v_lines[col_idx+1]['m'], v_lines[col_idx+1]['c'])
            br = eg.intersect_lines_uv(h_lines[row_idx+1]['m'], h_lines[row_idx+1]['c'], v_lines[col_idx+1]['m'], v_lines[col_idx+1]['c'])
            bl = eg.intersect_lines_uv(h_lines[row_idx+1]['m'], h_lines[row_idx+1]['c'], v_lines[col_idx]['m'], v_lines[col_idx]['c'])
            props[(col_idx, row_idx)] = [
                project_to_rotated(tl[0], tl[1]),
                project_to_rotated(tr[0], tr[1]),
                project_to_rotated(br[0], br[1]),
                project_to_rotated(bl[0], bl[1])
            ]
    return props

proposals = build_proposals(H_lines, V_lines)

# Inspect Col 0 Row 0
p = b_local["columns"][0][0]
poly_init = eg.get_local_yolo_polygon_prior(p["orig_ref"], transform_orig_to_local)
poly_proposal = proposals[(0, 0)]

print("poly_init corners:")
for idx, pt in enumerate(poly_init):
    print(f"  {idx}: {pt}")
    
print("poly_proposal corners:")
for idx, pt in enumerate(poly_proposal):
    print(f"  {idx}: {pt}")

cx_prop = np.mean([pt[0] for pt in poly_proposal])
cy_prop = np.mean([pt[1] for pt in poly_proposal])
cx_yolo = np.mean([pt[0] for pt in poly_init])
cy_yolo = np.mean([pt[1] for pt in poly_init])
print(f"Prop center: ({cx_prop:.3f}, {cy_prop:.3f})")
print(f"YOLO center: ({cx_yolo:.3f}, {cy_yolo:.3f})")
print(f"Shift: {math.sqrt((cx_prop-cx_yolo)**2 + (cy_prop-cy_yolo)**2):.3f}")
