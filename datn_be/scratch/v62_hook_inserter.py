"""
v62_hook_inserter.py
Script để chèn V62 backend hooks vào file pv_fullsnap_universal_v62_backend_ready.py
Chạy 1 lần, không phá thuật toán, chỉ thêm code collect kết quả.
"""
import re

filepath = r"c:\Solar_Inspection_Project\datn_be\scratch\pv_fullsnap_universal_v62_backend_ready.py"

with open(filepath, "r", encoding="utf-8") as f:
    content = f.read()

# Verify _BACKEND_RESULT_PANELS already exists
assert "_BACKEND_RESULT_PANELS" in content, "Global not found - run previous step first!"

# =====================================================================
# HOOK TEXT - placed before "return valid_count" in each engine
# Must be indented with 16 spaces (inside nested function inside try block)
# =====================================================================
v62_backend_hook = """
                # V62 Backend hook: collect finalized panel polygons for API consumers.
                # Runs AFTER all V60 filters. Appends to module-level _BACKEND_RESULT_PANELS.
                # Does NOT change how polygons are built - only reads the results.
                global _BACKEND_RESULT_PANELS
                for _be_e in panel_calc_entries:
                    _be_outer = _be_e.get("outer_polygon")
                    _be_inner = _be_e.get("inner_polygon")
                    if _be_outer is None:
                        continue
                    try:
                        _be_outer_list = _be_outer.tolist() if hasattr(_be_outer, "tolist") else list(_be_outer)
                        _be_inner_list = _be_inner.tolist() if (_be_inner is not None and hasattr(_be_inner, "tolist")) else (_be_outer_list if _be_inner is None else list(_be_inner))
                    except Exception:
                        continue
                    _BACKEND_RESULT_PANELS.append({
                        "outer_polygon": _be_outer_list,
                        "inner_polygon": _be_inner_list,
                        "outer_area": float(_be_e.get("outer_area", 0.0)),
                        "inner_area": float(_be_e.get("calc_area", 0.0)),
                        "valid": bool(_be_e.get("valid", False)),
                        "source": str(_be_e.get("src_name", _be_e.get("source", "line_snap"))),
                        "filter_reason": str(_be_e.get("filter_reason", "")),
                    })
"""

# =====================================================================
# Find the two UNIQUE markers for return valid_count
# In v34: inside _run_small_core_v34_for_one_image (before v42 engine at line ~8266)
# In v42: inside _run_large_local_v42_for_one_image (after line ~11800)
# Both have exactly: "                return valid_count\n\n            def refine_single_edge_local"
# =====================================================================
# The unique anchor after the v34 panel_calc loop
v34_anchor = '                    writer.writerows(rows_csv)\n\n                return valid_count\n\n            def refine_single_edge_local(rotated_gray, p1, p2, is_horizontal, win, cov_threshold, limit, res_threshold=2.5, angle_threshold=4.0):'

# The unique anchor after the v42 panel_calc loop - same pattern but in v42 body
v42_anchor = '                    writer.writerows(rows_csv)\n\n                return valid_count\n\n            def refine_single_edge_local(rotated_gray, p1, p2, is_horizontal, win, cov_threshold, limit, res_threshold=2.5, angle_threshold=4.0):'

count = content.count(v34_anchor)
print(f"Occurrences of anchor: {count}")

if count != 2:
    print("ERROR: expected exactly 2 occurrences (one per engine)!")
    exit(1)

# Insert the hook before EACH "return valid_count" in these anchor sequences
# Strategy: replace first occurrence for v34, second for v42
v34_replacement = (
    '                    writer.writerows(rows_csv)\n'
    + v62_backend_hook
    + '\n                return valid_count\n\n'
    + '            def refine_single_edge_local(rotated_gray, p1, p2, is_horizontal, win, cov_threshold, limit, res_threshold=2.5, angle_threshold=4.0):'
)

# Replace first occurrence (v34 engine)
content = content.replace(v34_anchor, v34_replacement, 1)
count_after = content.count(v34_anchor)
print(f"After 1st replace, remaining occurrences: {count_after}")

# Replace second occurrence (v42 engine) - the remaining one
content = content.replace(v34_anchor, v34_replacement, 1)
count_after2 = content.count(v34_anchor)
print(f"After 2nd replace, remaining occurrences: {count_after2}")

assert count_after == 1, "First replacement failed!"
assert count_after2 == 0, "Second replacement failed!"
assert content.count("_BACKEND_RESULT_PANELS.append") == 2, "Expected 2 hook insertions!"

print("Both hooks inserted successfully!")
print(f"Total _BACKEND_RESULT_PANELS.append calls: {content.count('_BACKEND_RESULT_PANELS.append')}")

# =====================================================================
# Add process_image_for_backend() function BEFORE main()
# =====================================================================
process_image_fn = '''

# ============================================================
# V62 Backend-ready entrypoint
# ============================================================
def process_image_for_backend(
    image_path: str,
    panel_log_path: str,
    output_dir: str,
    debug: bool = False,
    log_level: str = "quiet",
) -> Dict[str, Any]:
    """
    V62 API entrypoint: run the full line-snap pipeline for one image
    and return structured panel polygon data.

    Args:
        image_path:      Absolute path to the thermal image (JPG/PNG).
        panel_log_path:  Absolute path to the YOLO panel detection log
                         (*_panel_refine.jsonl or .json).
        output_dir:      Directory to write debug images into.
        debug:           If True, write debug images (line_snap, calc_inner).
        log_level:       "quiet" (only errors/summary), "normal", or "debug".

    Returns:
        {
            "ok": bool,
            "stem": str,
            "route": "core" | "local",
            "engine": "small_core_v34_full" | "large_local_v42_full",
            "line_snap_path": str,
            "calc_inner_polygon_path": str,
            "panels": [
                {
                    "outer_polygon": [[x,y], ...],   # 4-point, from line-snap
                    "inner_polygon": [[x,y], ...],   # 4-point, 3px inset
                    "outer_area": float,
                    "inner_area": float,
                    "valid": bool,
                    "source": str,
                    "filter_reason": str,
                }
            ],
            "summary": {
                "n_total": int,
                "n_valid": int,
                "n_rejected": int,
            },
            "error": str | None,
        }
    """
    import logging as _logging
    _logger = _logging.getLogger("solar_ai")

    global _BACKEND_RESULT_PANELS, IMAGE_STEM, CURRENT_IMAGE_PATH, CURRENT_PANELS_PATH
    global OUTPUT_DEBUG_DIR, DRAW_INNER_PANEL_DEBUG, GLOBAL_LOWER_SUPPORT_DEBUG_IMG

    # Clear the global result collector before this run
    _BACKEND_RESULT_PANELS = []

    image_path_obj = Path(image_path)
    panel_log_obj = Path(panel_log_path)
    out_dir_obj = Path(output_dir)

    if not image_path_obj.exists():
        return {"ok": False, "error": f"Image not found: {image_path}", "panels": [], "summary": {"n_total": 0, "n_valid": 0, "n_rejected": 0}}

    if not panel_log_obj.exists():
        return {"ok": False, "error": f"Panel log not found: {panel_log_path}", "panels": [], "summary": {"n_total": 0, "n_valid": 0, "n_rejected": 0}}

    stem = image_path_obj.stem
    out_dir_obj.mkdir(parents=True, exist_ok=True)

    # Configure debug image writing
    _save_debug = debug
    _orig_draw_inner = DRAW_INNER_PANEL_DEBUG
    if not _save_debug:
        DRAW_INNER_PANEL_DEBUG = False

    # Compute route metrics
    try:
        metrics = _route_metrics(image_path_obj, panel_log_obj, top_frac=REPRESENTATIVE_TOP_AREA_RATIO)
    except Exception as e:
        return {"ok": False, "error": f"route_metrics failed: {e}", "panels": [], "summary": {"n_total": 0, "n_valid": 0, "n_rejected": 0}}

    if metrics["rep_area_frac"] > ROUTE_PANEL_AREA_FRAC_THRESHOLD:
        route = "local"
        engine_name = "large_local_v42_full"
    else:
        route = "core"
        engine_name = "small_core_v34_full"

    _logger.info(f"[V62_ROUTE] stem={stem} route={route} rep_area_frac={metrics['rep_area_frac']:.5f}")

    # Run the chosen engine
    try:
        if route == "local":
            rc = _run_large_local_v42_for_one_image(image_path_obj, panel_log_obj.parent, out_dir_obj)
        else:
            rc = _run_small_core_v34_for_one_image(image_path_obj, panel_log_obj.parent, out_dir_obj)
    except Exception as e:
        # Restore global state
        DRAW_INNER_PANEL_DEBUG = _orig_draw_inner
        return {"ok": False, "error": f"Engine failed: {e}", "panels": [], "summary": {"n_total": 0, "n_valid": 0, "n_rejected": 0}}

    # Restore global state
    DRAW_INNER_PANEL_DEBUG = _orig_draw_inner

    # Collect results from the global hook
    panels = list(_BACKEND_RESULT_PANELS)
    n_total = len(panels)
    n_valid = sum(1 for p in panels if p.get("valid", False))
    n_rejected = n_total - n_valid

    # Tag engine source onto each panel
    for p in panels:
        p["engine"] = engine_name

    line_snap_path = str(out_dir_obj / f"debug_{stem}_line_snap.JPG")
    calc_inner_path = str(out_dir_obj / f"debug_{stem}_calc_inner_polygon.JPG")

    _logger.info(
        f"[V62_DONE] stem={stem} engine={engine_name} rc={rc} "
        f"n_total={n_total} n_valid={n_valid} n_rejected={n_rejected}"
    )

    return {
        "ok": True,
        "stem": stem,
        "route": route,
        "engine": engine_name,
        "line_snap_path": line_snap_path,
        "calc_inner_polygon_path": calc_inner_path,
        "panels": panels,
        "summary": {
            "n_total": n_total,
            "n_valid": n_valid,
            "n_rejected": n_rejected,
        },
        "error": None,
    }

'''

# Insert process_image_for_backend before def main()
main_anchor = "\ndef main() -> int:\n"
assert main_anchor in content, "Could not find 'def main()' anchor!"
content = content.replace(main_anchor, process_image_fn + "\ndef main() -> int:\n", 1)
print("process_image_for_backend() added before main()")

with open(filepath, "w", encoding="utf-8") as f:
    f.write(content)
print("File written successfully!")
print(f"Total lines: {content.count(chr(10))}")
