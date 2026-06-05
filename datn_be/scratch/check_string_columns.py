# scratch/check_string_columns.py
# Diagnostic script for DJI_0843_R final polygon trace (no record_type filter)
# Requirements:
#   1. Read data/results/debug_logs/DJI_0843_R_final_polygon_trace.jsonl
#   2. Parse every JSON line.
#   3. Identify a panel record by presence of keys: "raw_idx", "final_polygon_source", "bbox".
#   4. Print the first 5 panel records with selected fields.
#   5. Count panels per block_id (normalize to str).
#   6. Count panels where final_polygon_source == "yolo_original",
#      final_polygon_stage == "yolo_refinement",
#      final_polygon_reason == "yolo_detection_contour" per block_id.
#   7. For blocks 0, 1, 5 print:
#        - total panels
#        - count of panels matching the three criteria above
#        - panels with string_id == null
#        - first 10 raw_idx values
#   8. Do not read string_lattice_refine.jsonl.
#   9. Do not make any conclusions about n_cols.
#   10. Run once and exit.

import json
import pathlib
from collections import defaultdict

# Resolve paths
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
FINAL_LOG = PROJECT_ROOT / "data" / "results" / "debug_logs" / "DJI_0843_R_final_polygon_trace.jsonl"

print("Reading final polygon trace from:", FINAL_LOG)

# Helper to safely get a field
def safe_get(rec, key, default=None):
    return rec.get(key, default)

# ---------------------------------------------------------------------
# Parse all lines, collect panel-like records
panel_records = []
with FINAL_LOG.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        # Identify panel record by required keys
        if all(k in rec for k in ("raw_idx", "final_polygon_source", "bbox")):
            panel_records.append(rec)

# ---------------------------------------------------------------------
# Print first 5 panel records with selected fields
print("\nFirst 5 panel records (selected fields):")
for rec in panel_records[:5]:
    display = {
        "raw_idx": safe_get(rec, "raw_idx"),
        "final_polygon_source": safe_get(rec, "final_polygon_source"),
        "final_polygon_stage": safe_get(rec, "final_polygon_stage"),
        "final_polygon_reason": safe_get(rec, "final_polygon_reason"),
        "block_id": safe_get(rec, "block_id"),
        "string_id": safe_get(rec, "string_id"),
    }
    print(display)

# ---------------------------------------------------------------------
# Aggregate statistics per block (normalize block_id to string)
stats = defaultdict(lambda: {
    "total": 0,
    "yolo_match": 0,   # matches all three criteria
    "string_id_null": 0,
    "raw_idxs": [],
})

for rec in panel_records:
    block_raw = rec.get("block_id")
    block_key = str(block_raw) if block_raw is not None else "None"
    s = stats[block_key]
    s["total"] += 1
    # Check three YOLO criteria simultaneously
    if (
        rec.get("final_polygon_source") == "yolo_original"
        and rec.get("final_polygon_stage") == "yolo_refinement"
        and rec.get("final_polygon_reason") == "yolo_detection_contour"
    ):
        s["yolo_match"] += 1
    # Count null string_id
    if rec.get("string_id") is None:
        s["string_id_null"] += 1
    # Store raw_idx (fallback to panel_idx if missing)
    raw_idx = rec.get("raw_idx") if rec.get("raw_idx") is not None else rec.get("panel_idx")
    s["raw_idxs"].append(raw_idx)

# ---------------------------------------------------------------------
# Report for blocks 0, 1, 5
print("\n=== Summary for blocks 0, 1, 5 ===")
for blk in ["0", "1", "5"]:
    s = stats.get(blk, {"total": 0, "yolo_match": 0, "string_id_null": 0, "raw_idxs": []})
    print(f"Block {blk}:")
    print(f"  total panels               : {s['total']}")
    print(f"  YOLO panels (source+stage+reason) count : {s['yolo_match']}")
    print(f"  panels with string_id null : {s['string_id_null']}")
    print(f"  first 10 raw_idx values    : {s['raw_idxs'][:10]}")

print("\nDone.")
