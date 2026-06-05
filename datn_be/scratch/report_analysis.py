import os
import json
import glob
import re
from collections import Counter, defaultdict

# Directory containing debug jsonl files
DEBUG_DIR = r"c:/Solar_Inspection_Project/datn_be/data/results/debug_logs"

# Patterns for files
final_pattern = "*_final_polygon_trace.jsonl"
string_pattern = "*_string_lattice_refine.jsonl"

# Find all matching files
final_files = glob.glob(os.path.join(DEBUG_DIR, final_pattern))
string_files = glob.glob(os.path.join(DEBUG_DIR, string_pattern))

# Prefer final files; if none for a prefix, use string files
files_by_prefix = {}
for f in final_files:
    # prefix is filename without suffix and extension
    name = os.path.basename(f)
    prefix = re.sub(r"_final_polygon_trace\.jsonl$", "", name)
    files_by_prefix[prefix] = f

for f in string_files:
    name = os.path.basename(f)
    prefix = re.sub(r"_string_lattice_refine\.jsonl$", "", name)
    if prefix not in files_by_prefix:
        files_by_prefix[prefix] = f

# If no distinction of "new test images", we just process all prefixes
report = []

for prefix, filepath in files_by_prefix.items():
    total_records = 0
    final_polygon_source_counter = Counter()
    geometry_gate_accept_counter = Counter()
    geometry_gate_reason_counter = Counter()
    final_polygon_reason_counter = Counter()
    string_lattice_middle_cnt = 0
    yolo_original_cnt = 0
    near_edge_counter = Counter()
    first_30_matches = []
    # Read file line by line
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            total_records += 1
            # counters
            fps = record.get("final_polygon_source")
            if fps:
                final_polygon_source_counter[fps] += 1
                if fps == "string_lattice_middle":
                    string_lattice_middle_cnt += 1
                if fps == "yolo_original":
                    yolo_original_cnt += 1
            # geometry_gate_accept
            gga = record.get("geometry_gate_accept")
            if gga is True:
                geometry_gate_accept_counter["True"] += 1
            elif gga is False:
                geometry_gate_accept_counter["False"] += 1
            else:
                geometry_gate_accept_counter["missing"] += 1
            # geometry_gate_reason
            ggr = record.get("geometry_gate_reason")
            if ggr:
                geometry_gate_reason_counter[ggr] += 1
            # final_polygon_reason starting with geometry_gate_rejected
            fpr = record.get("final_polygon_reason")
            if isinstance(fpr, str) and fpr.startswith("geometry_gate_rejected"):
                final_polygon_reason_counter[fpr] += 1
            # near_edge stats
            ne = record.get("near_edge")
            if ne is not None:
                near_edge_counter[str(ne)] += 1
            # collect first 30 matching records
            match_cond = (
                (gga is False) or
                (isinstance(fpr, str) and fpr.startswith("geometry_gate_rejected")) or
                (fps == "string_lattice_middle")
            )
            if match_cond and len(first_30_matches) < 30:
                # Gather required fields, default to None if missing
                first_30_matches.append({
                    "image_name": record.get("image_name"),
                    "raw_idx": record.get("raw_idx"),
                    "block_id": record.get("block_id"),
                    "bbox": record.get("bbox"),
                    "polygon": record.get("polygon"),
                    "final_polygon_source": fps,
                    "final_polygon_stage": record.get("final_polygon_stage"),
                    "final_polygon_reason": fpr,
                    "geometry_gate_accept": gga,
                    "geometry_gate_reason": ggr,
                    "iou": record.get("iou"),
                    "center_shift": record.get("center_shift"),
                    "area_ratio": record.get("area_ratio"),
                    "max_overlap": record.get("max_overlap"),
                    "near_edge": ne,
                })
    report.append({
        "image_prefix": prefix,
        "file_path": filepath,
        "total_records": total_records,
        "final_polygon_source_counts": dict(final_polygon_source_counter),
        "geometry_gate_accept_counts": dict(geometry_gate_accept_counter),
        "geometry_gate_reason_counts": dict(geometry_gate_reason_counter),
        "final_polygon_reason_counts": dict(final_polygon_reason_counter),
        "string_lattice_middle_count": string_lattice_middle_cnt,
        "yolo_original_count": yolo_original_cnt,
        "near_edge_counts": dict(near_edge_counter),
        "first_30_matches": first_30_matches,
    })

# Print report as pretty JSON
print(json.dumps(report, ensure_ascii=False, indent=2))
