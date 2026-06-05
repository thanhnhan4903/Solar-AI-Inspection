# Temporary script to check whether two_col_block_lattice mutates final polygon
# Usage: python check_two_col_mutation.py <image_path>
# It runs the existing pipeline (process_yolo_predictions) up to string lattice,
# records a hash of each panel's polygon, then runs the two‑column refinement
# and reports changes.

import sys
import os
import json
import hashlib
from pathlib import Path

# Add project root to PYTHONPATH
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from app.services.ai_engine import AIEngine, PANEL_CONF_THRESHOLD, DEFECT_CONF_THRESHOLD
from app.services.panel_processor import process_yolo_predictions


def polygon_hash(poly):
    """Return a deterministic hash string for a polygon (list of [x, y] points)."""
    # Convert to a canonical string representation
    s = json.dumps(poly, sort_keys=True)
    return hashlib.sha256(s.encode()).hexdigest()[:12]


def run_pipeline(image_path):
    engine = AIEngine("weights/best.pt")
    result = engine.model.predict(source=image_path, conf=min(PANEL_CONF_THRESHOLD, DEFECT_CONF_THRESHOLD), save=False, verbose=False)[0]
    detections = process_yolo_predictions(
        result=result,
        orig_img=result.orig_img,
        panel_conf=PANEL_CONF_THRESHOLD,
        defect_conf=DEFECT_CONF_THRESHOLD,
        panel_class_name="panel",
    )
    panels = [d for d in detections if d["category"] == "panel"]
    # panels now have polygon after string lattice (string_first_polygon may exist)
    return panels, result


def main():
    if len(sys.argv) != 2:
        print("Usage: python check_two_col_mutation.py <image_path>")
        sys.exit(1)
    image_path = sys.argv[1]
    # Step 1: run up to string lattice
    panels_before, result = run_pipeline(image_path)
    # Record hash and source before two‑col
    before_info = {}
    for p in panels_before:
        raw_idx = p.get("raw_idx", p.get("panel_idx", -1))
        before_info[raw_idx] = {
            "hash": polygon_hash(p["polygon"]),
            "source": p.get("final_polygon_source", "unknown"),
        }
    # Step 2: run two‑column refinement (the same function is called inside process_yolo_predictions)
    # process_yolo_predictions already invoked it, so we need to invoke it again manually to see any change.
    # We'll call refine_panels_by_two_column_block_lattice directly.
    from app.services.panel_processor import refine_panels_by_two_column_block_lattice
    panels_after = refine_panels_by_two_column_block_lattice(panels_before, result.orig_img.shape, image_path)
    # Record after info
    after_info = {}
    for p in panels_after:
        raw_idx = p.get("raw_idx", p.get("panel_idx", -1))
        after_info[raw_idx] = {
            "hash": polygon_hash(p["polygon"]),
            "source": p.get("final_polygon_source", p.get("two_col_decision", "unknown")),
            "two_col_decision": p.get("two_col_decision", "unknown"),
        }
    # Compare
    changed = []
    for idx, info_before in before_info.items():
        info_after = after_info.get(idx)
        if not info_after:
            continue
        if info_before["hash"] != info_after["hash"]:
            changed.append({
                "raw_idx": idx,
                "before_source": info_before["source"],
                "after_source": info_after["source"],
                "two_col_decision": info_after["two_col_decision"],
            })
    # Output summary
    print(f"changed_count: {len(changed)}")
    for entry in changed[:10]:
        print(entry)
    # Determine if final_polygon_source ever changed
    source_changed = any(entry["before_source"] != entry["after_source"] for entry in changed)
    print(f"final_polygon_source changed: {source_changed}")
    # Decision
    if any(entry["after_source"] != "yolo_original" for entry in changed):
        print("Conclusion: A) two_col mutates final polygon")
    else:
        print("Conclusion: B) two_col diagnostic-only, does not mutate final polygon")

if __name__ == "__main__":
    main()
