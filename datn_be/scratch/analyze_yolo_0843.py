# Temporary script to analyse YOLO-original panels in DJI_0843_R final polygon trace
# Usage: python scratch/analyze_yolo_0843.py

import json
import pathlib
from collections import defaultdict

# Path to the final polygon trace log
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
TRACE_PATH = PROJECT_ROOT / "data" / "results" / "debug_logs" / "DJI_0843_R_final_polygon_trace.jsonl"

# Containers for aggregation
total = 0
by_block = defaultdict(list)   # block_id -> list of raw_idx
by_string = defaultdict(list)  # string_id -> list of raw_idx
area_bins = defaultdict(list)  # bin label -> list of raw_idx
region_bins = defaultdict(list)  # region label -> list of raw_idx
examples = []
centers = []  # store (x, y) for region calculation

# Helper for area binning
def area_bin(area):
    if area < 1200:
        return "<1200"
    if area < 1600:
        return "1200-1600"
    if area < 2200:
        return "1600-2200"
    return ">2200"

# Read and filter records
with TRACE_PATH.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        # Ensure required fields exist
        if (
            rec.get("final_polygon_source") == "yolo_original"
            and rec.get("final_polygon_stage") == "yolo_refinement"
            and rec.get("final_polygon_reason") == "yolo_detection_contour"
        ):
            total += 1
            raw_idx = rec.get("raw_idx") or rec.get("panel_idx")
            block_id = rec.get("block_id")
            string_id = rec.get("string_id")
            area = rec.get("area")
            bbox = rec.get("bbox")
            # compute centre
            if bbox and len(bbox) == 4:
                cx = (bbox[0] + bbox[2]) / 2.0
                cy = (bbox[1] + bbox[3]) / 2.0
                centers.append((cx, cy))
                centre = [cx, cy]
            else:
                centre = [None, None]

            # groupings
            if block_id is not None:
                by_block[block_id].append(raw_idx)
            if string_id is not None:
                by_string[string_id].append(raw_idx)
            area_bins[area_bin(area)].append(raw_idx)
            # store example
            if len(examples) < 20:
                examples.append({
                    "raw_idx": raw_idx,
                    "bbox": bbox,
                    "area": area,
                    "center": centre,
                    "block_id": block_id,
                    "string_id": string_id,
                })

# Determine image regions (left/center/right, top/middle/bottom) using the spread of centre coordinates
region_counts = defaultdict(int)
if centers:
    xs, ys = zip(*centers)
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    x_thr1 = min_x + (max_x - min_x) / 3.0
    x_thr2 = min_x + 2 * (max_x - min_x) / 3.0
    y_thr1 = min_y + (max_y - min_y) / 3.0
    y_thr2 = min_y + 2 * (max_y - min_y) / 3.0
    for cx, cy in centers:
        # horizontal region
        if cx < x_thr1:
            h = "left"
        elif cx < x_thr2:
            h = "center"
        else:
            h = "right"
        # vertical region
        if cy < y_thr1:
            v = "top"
        elif cy < y_thr2:
            v = "middle"
        else:
            v = "bottom"
        region = f"{v}_{h}"  # e.g., top_left
        region_counts[region] += 1
        region_bins[region].append((cx, cy))

# Output
print(f"total yolo_original panels (stage=yolo_refinement, reason=yolo_detection_contour): {total}\n")

print("Group by block_id (count):")
for blk, ids in sorted(by_block.items()):
    print(f"  block {blk}: {len(ids)}")
print()

print("Group by string_id (count):")
for sid, ids in sorted(by_string.items()):
    print(f"  string {sid}: {len(ids)}")
print()

print("Group by area bins (count):")
for blabel, ids in area_bins.items():
    print(f"  {blabel}: {len(ids)}")
print()

print("Group by image region (count):")
for region, cnt in sorted(region_counts.items()):
    print(f"  {region}: {cnt}")
print()

print("Raw indices per block_id (full list):")
for blk, ids in sorted(by_block.items()):
    print(f"  block {blk}: {ids}")
print()

print("First 20 examples:")
for ex in examples:
    print(ex)
