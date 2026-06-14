import sys
from pathlib import Path
from scratch.pv_fullsnap_universal_v62_backend_ready import process_image_for_backend

img_path = "data/precalib/DJI_0959.JPG"
log_path = "data/results/debug_logs/DJI_0959_panel_refine.jsonl"
out_dir = "data/results"

print("Calling process_image_for_backend...")
res = process_image_for_backend(
    image_path=img_path,
    panel_log_path=log_path,
    output_dir=out_dir,
    debug=True
)

print("\n--- RESULTS ---")
print("ok:", res.get("ok"))
print("stem:", res.get("stem"))
print("route:", res.get("route"))
print("engine:", res.get("engine"))
print("error:", res.get("error"))
panels = res.get("panels", [])
print("Number of panels:", len(panels))
print("Summary:", res.get("summary"))

for i, p in enumerate(panels):
    print(f"\nPanel {i}:")
    print("  valid:", p.get("valid"))
    print("  source:", p.get("source"))
    print("  engine:", p.get("engine"))
    print("  outer_area:", p.get("outer_area"))
    print("  inner_area:", p.get("inner_area"))
    print("  outer_polygon:", p.get("outer_polygon"))
    print("  inner_polygon:", p.get("inner_polygon"))
    print("  calc_polygon:", p.get("calc_polygon"))
