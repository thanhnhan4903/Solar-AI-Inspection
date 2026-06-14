import os
import json
import logging
from app.services.pv_panel_snapper import snap_panels_with_v61

# Configure simple logging to stdout
logging.basicConfig(level=logging.INFO)

# Load yolo panels from the log file
yolo_panels = []
log_path = "data/results/debug_logs/DJI_0959_panel_refine.jsonl"
with open(log_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            yolo_panels.append(json.loads(line))

print(f"Loaded {len(yolo_panels)} YOLO panels from log.")

image_path = "data/precalib/DJI_0959.JPG"
output_dir = "data/results"

print("Running snap_panels_with_v61...")
snapped = snap_panels_with_v61(
    image_path=image_path,
    yolo_panels=yolo_panels,
    output_dir=output_dir,
    debug=True
)

print(f"\n--- SNAPPER RESULTS ---")
print("Number of snapped panels returned:", len(snapped))
if snapped:
    print("Geometry source of first panel:", snapped[0].get("geometry_source"))
    print("Outer polygon of first panel:", snapped[0].get("outer_polygon"))
    print("Inner polygon of first panel:", snapped[0].get("inner_polygon"))
    print("Calc polygon of first panel:", snapped[0].get("calc_polygon"))
    print("Bbox of first panel:", snapped[0].get("bbox"))
    print("Center of first panel:", snapped[0].get("center"))
    print("Area of first panel:", snapped[0].get("area"))
    print("Outer area of first panel:", snapped[0].get("outer_area"))
    print("Inner area of first panel:", snapped[0].get("inner_area"))
