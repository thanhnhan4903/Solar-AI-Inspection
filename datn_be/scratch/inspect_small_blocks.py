import json
from pathlib import Path
import numpy as np

# Load panels
panels = []
with open("data/results/debug_logs/DJI_0845_R_panel_refine.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            panels.append(json.loads(line))

# group into blocks
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

def sort_panel_corners(poly):
    sorted_by_y = sorted(poly, key=lambda p: p[1])
    top_two = sorted(sorted_by_y[:2], key=lambda p: p[0])
    bottom_two = sorted(sorted_by_y[2:], key=lambda p: p[0])
    return top_two[0], top_two[1], bottom_two[1], bottom_two[0]

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

for idx, b in enumerate(blocks):
    row_count = max(len(col) for col in b["columns"])
    is_small = (row_count < 4 or len(b["panels"]) <= 4)
    if is_small:
        print(f"Block {idx+1}: layout = {len(b['columns'])} cols, panels = {len(b['panels'])}")
        for c_idx, col in enumerate(b["columns"]):
            print(f"  Col {c_idx}: {len(col)} panels")
            for r_idx, p in enumerate(col):
                print(f"    Row {r_idx}: bbox={p['bbox']}")
