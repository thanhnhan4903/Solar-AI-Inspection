import sys
import os
import json
import numpy as np
import cv2
sys.path.insert(0, '.')

from app.services.ai_engine import AIEngine, PANEL_CONF_THRESHOLD, DEFECT_CONF_THRESHOLD
from app.services.panel_processor import process_yolo_predictions

engine = AIEngine("weights/best.pt")

test_images = [
    "data/precalib/DJI_0843_R.JPG",
    "data/precalib/DJI_0845_R.JPG",
    "data/precalib/DJI_0087_R.JPG",
    "data/precalib/DJI_0029_R.JPG"
]

for img_path in test_images:
    print(f"\n==================================================")
    print(f"RUNNING: {img_path}")
    print(f"==================================================")

    global_conf = min(PANEL_CONF_THRESHOLD, DEFECT_CONF_THRESHOLD)
    results = engine.model.predict(source=img_path, conf=global_conf, save=False, verbose=False)
    result = results[0]

    detections = process_yolo_predictions(
        result=result,
        orig_img=result.orig_img,
        panel_conf=PANEL_CONF_THRESHOLD,
        defect_conf=DEFECT_CONF_THRESHOLD,
        panel_class_name="panel"
    )

    panels = [d for d in detections if d["category"] == "panel"]
    defects = [d for d in detections if d["category"] == "defect"]

    image_name = os.path.splitext(os.path.basename(img_path))[0]
    string_log_path = f"data/results/debug_logs/{image_name}_string_lattice_refine.jsonl"
    panels_log = []
    strings_log = []

    print(f"\nDetections summary:")
    print(f"  - Total panels in final: {len(panels)}")
    print(f"  - Total defects in final: {len(defects)}")

    if os.path.exists(string_log_path):
        print(f"\nString Lattice + Snap Refinement Log ({string_log_path}):")

        with open(string_log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                rt = data.get("record_type")
                if rt == "panel":
                    panels_log.append(data)
                elif rt == "string":
                    strings_log.append(data)

        # --- String-level stats ---
        print(f"  - Block count: {len(strings_log)}")
        for s in strings_log:
            b_id = s.get('block_id', s.get('string_id'))
            b_panels = [p for p in panels_log if p.get('block_id') == b_id]
            m_panels = [p for p in b_panels if p.get('role') == 'middle']
            ep_panels = [p for p in b_panels if p.get('role') in ('endpoint_start', 'endpoint_end', 'endpoint')]
            
            middle_count = len(m_panels)
            endpoint_count = len(ep_panels)
            
            failed_middle_count = sum(1 for p in m_panels if p.get('final_geometry_source') != 'string_lattice_middle')
            failed_endpoint_count = sum(1 for p in ep_panels if p.get('final_geometry_source') != 'string_lattice_endpoint')
            
            covs = [p.get('original_coverage_ratio', 1.0) for p in m_panels]
            med_cov = float(np.median(covs)) if covs else 1.0
            min_cov = float(np.min(covs)) if covs else 1.0
            
            cuts = [p.get('max_inward_cut_ratio', 0.0) for p in m_panels]
            max_cut = float(np.max(cuts)) if cuts else 0.0
            
            ovs = [p.get('max_overlap', 0.0) for p in m_panels]
            max_ov = float(np.max(ovs)) if ovs else 0.0
            
            fallback_reason = s.get('fallback_reason', '')
            print(f"    * block_id={b_id}, decision={s['decision']}, n_panels={s['n_panels']}, n_cols={s['n_cols']}, n_rows={s['n_rows']}, middle_count={middle_count}, endpoint_count={endpoint_count}, middle_pass_ratio={s.get('middle_pass_ratio', 0.0):.4f}, failed_middle_count={failed_middle_count}, failed_endpoint_count={failed_endpoint_count}, median_middle_original_coverage={med_cov:.4f}, min_middle_original_coverage={min_cov:.4f}, max_middle_inward_cut_ratio={max_cut:.4f}, max_middle_overlap={max_ov:.4f}, fallback_reason={fallback_reason}")
            tc_count = s.get('trusted_core_count', 0)
            ax_src = s.get('axis_source', 'n/a')
            tc_p_cv = s.get('trusted_pitch_cv', 0.0)
            tc_ang_d = s.get('trusted_angle_delta_deg', 0.0)
            print(f"      [TRUSTED CORE]: tc_count={tc_count}, axis_source={ax_src}, trusted_pitch_cv={tc_p_cv:.4f}, trusted_angle_delta={tc_ang_d:.2f}")
            if s.get('decision') == 'fallback_string':
                valid_m_count = sum(1 for p in m_panels if p.get('raw_fail_reason', '') == '')
                failed_m_by_reason = {}
                for p in m_panels:
                    reason = p.get('raw_fail_reason', '')
                    if reason != '':
                        failed_m_by_reason[reason] = failed_m_by_reason.get(reason, 0) + 1
                
                print(f"      [DIAGNOSTIC] fallback_string block details:")
                print(f"        - block_id: {b_id}, n: {s.get('n_panels')}, cols: {s.get('n_cols')}, rows: {s.get('n_rows')}")
                print(f"        - fallback_reason: {fallback_reason}")
                print(f"        - middle_pass_ratio: {s.get('middle_pass_ratio', 0.0):.4f}")
                print(f"        - median_middle_original_coverage: {med_cov:.4f}")
                print(f"        - max_middle_inward_cut_ratio: {s.get('max_middle_inward_cut_ratio', max_cut):.4f}")
                print(f"        - max_middle_overlap: {max_ov:.4f}")
                print(f"        - pitch_cv: {s.get('pitch_cv', 0.0):.4f}")
                print(f"        - rail_width_cv: {s.get('rail_width_cv', 0.0):.4f}")
                print(f"        - divider_cross_count_middle: {s.get('divider_cross_count_middle', 0)}")
                print(f"        - outside_image_count_middle: {s.get('outside_image_count_middle', 0)}")
                print(f"        - number of individually valid middle panels: {valid_m_count}")
                print(f"        - number of failed middle panels grouped by fail_reason: {failed_m_by_reason}")

        # --- Panel-level stats ---
        use_panel_count = sum(1 for p in panels_log if p["decision"] == "use_string_lattice")
        fallback_panel_count = sum(1 for p in panels_log if p["decision"] == "fallback_original")
        reject_partial_count = sum(1 for p in panels_log if p["decision"] == "reject_partial")
        keep_orig_not_improved_count = sum(1 for p in panels_log if p["decision"] == "keep_original_not_improved")
        snap_used_count = sum(1 for p in panels_log if p.get("snap_used", False))

        ious_before = [p.get("iou_before_snap", p["iou_with_original"]) for p in panels_log if "iou_with_original" in p]
        ious_after = [p.get("iou_after_snap", p["iou_with_original"]) for p in panels_log if p.get("snap_used", False)]
        shifts_after = [p.get("center_shift_after_snap", 0.0) for p in panels_log if p.get("snap_used", False)]
        med_iou_before = float(np.median(ious_before)) if ious_before else 0.0
        med_iou_after = float(np.median(ious_after)) if ious_after else med_iou_before

        edge_ratios = [p["edge_score_ratio"] for p in panels_log if p.get("edge_score_ratio", 0) > 0]
        med_edge = float(np.median(edge_ratios)) if edge_ratios else 1.0

        print(f"  - Panels processed: {len(panels_log)}")
        print(f"    * use_string_lattice: {use_panel_count}")
        print(f"    * fallback_original:  {fallback_panel_count}")
        print(f"    * reject_partial:     {reject_partial_count}")
        print(f"    * keep_original_not_improved: {keep_orig_not_improved_count}")
        print(f"    * snap_used:          {snap_used_count}")
        print(f"    * median_iou_before:  {med_iou_before:.4f}")
        print(f"    * median_iou_after:   {med_iou_after:.4f} (snap panels only)")
        print(f"    * median_edge_ratio:  {med_edge:.4f} (log only)")

        # Snap decision distribution
        snap_decisions = {}
        for p in panels_log:
            sd = p.get("snap_decision", "n/a")
            snap_decisions[sd] = snap_decisions.get(sd, 0) + 1
        print(f"    * snap_decisions: {snap_decisions}")

        # Fallback reasons
        reasons_dist = {}
        for p in panels_log:
            r = p.get("fallback_reason", "ok")
            reasons_dist[r] = reasons_dist.get(r, 0) + 1
        print(f"    * Fallback reasons: {reasons_dist}")

        # Max overlap
        max_overlap_val = max([p.get("max_overlap", 0) for p in panels_log], default=0.0)
        print(f"    * Max overlap: {max_overlap_val:.4f}")

        # Violation checks
        violations_area = []
        violations_outside = []
        violations_overlap = []
        for p in panels_log:
            if p["decision"] == "use_string_lattice":
                ar = p.get("area_ratio", 1.0)
                if not (0.65 - 1e-5 <= ar <= 1.35 + 1e-5):
                    violations_area.append((p.get("string_id"), p.get("panel_idx"), ar))
                if p.get("fallback_reason") in ("outside_image",):
                    violations_outside.append((p.get("string_id"), p.get("panel_idx")))
                if p.get("max_overlap", 0) > 0.05 + 1e-5:
                    violations_overlap.append((p.get("string_id"), p.get("panel_idx"), p["max_overlap"]))

        print(f"    * Area limits [0.65, 1.35] use_string_lattice: {'PASS' if not violations_area else f'FAIL {violations_area}'}")
        print(f"    * Outside image check: {'PASS' if not violations_outside else f'FAIL {violations_outside}'}")
        print(f"    * Max overlap <= 0.05: {'PASS' if not violations_overlap else f'WARNING {violations_overlap}'}")

        debug_img = f"data/results/debug/debug_{image_name}_string_lattice_refine.JPG"
        print(f"  - Debug image: {debug_img} {'EXISTS' if os.path.exists(debug_img) else 'MISSING'}")
    else:
        print(f"\n  - No string lattice log found at {string_log_path}")
        if "0987" in image_name.upper() or "0995" in image_name.upper():
            print(f"    → BYPASS expected for large-panel image: CONFIRMED")

    two_col_log_path = f"data/results/debug_logs/{image_name}_two_col_lattice_refine.jsonl"
    if os.path.exists(two_col_log_path):
        print(f"\nTwo-column Block Lattice Refinement Log ({two_col_log_path}):")

        panels_2c = []
        blocks_2c = []
        with open(two_col_log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                rt = data.get("record_type")
                if rt == "panel":
                    panels_2c.append(data)
                elif rt == "two_col_block":
                    blocks_2c.append(data)

        total_blocks = len(blocks_2c)
        use_blocks = sum(1 for b in blocks_2c if b["decision"] == "use_two_col_lattice")
        fallback_blocks = sum(1 for b in blocks_2c if b["decision"] == "fallback_block")

        use_panels = sum(1 for p in panels_2c if p["decision"] == "use_two_col_lattice")
        fallback_string_panels = sum(1 for p in panels_2c if p["decision"] == "fallback_string")
        fallback_orig_panels = sum(1 for p in panels_2c if p["decision"] == "fallback_original")
        keep_orig_not_improved_panels = sum(1 for p in panels_2c if p["decision"] == "keep_original_not_improved")

        ious = [p["iou_with_original"] for p in panels_2c if "iou_with_original" in p]
        median_iou_2c = float(np.median(ious)) if ious else 0.0

        overlaps = [p["max_overlap"] for p in panels_2c if "max_overlap" in p]
        max_overlap_2c = float(np.max(overlaps)) if overlaps else 0.0

        reasons_dist = {}
        for p in panels_2c:
            r = p.get("reason", "ok")
            reasons_dist[r] = reasons_dist.get(r, 0) + 1

        print(f"  - Total two_col_blocks: {total_blocks}")
        print(f"    * use_two_col_lattice blocks: {use_blocks}")
        print(f"    * fallback_block blocks: {fallback_blocks}")
        print(f"  - Panels processed in blocks: {len(panels_2c)}")
        print(f"    * panels use_two_col_lattice: {use_panels}")
        print(f"    * panels fallback_string:     {fallback_string_panels}")
        print(f"    * panels fallback_original:   {fallback_orig_panels}")
        print(f"    * panels keep_original_not_improved: {keep_orig_not_improved_panels}")
        print(f"    * median_iou:                 {median_iou_2c:.4f}")
        print(f"    * max_overlap:                {max_overlap_2c:.4f}")
        print(f"    * top reasons: {reasons_dist}")

        violations_area_2c = []
        violations_outside_2c = []
        violations_overlap_2c = []
        for p in panels_2c:
            if p["decision"] == "use_two_col_lattice":
                ar = p.get("area_ratio", 1.0)
                if not (0.70 - 1e-5 <= ar <= 1.30 + 1e-5):
                    violations_area_2c.append((p.get("block_id"), p.get("panel_idx"), ar))
                if p.get("reason") in ("outside_image",):
                    violations_outside_2c.append((p.get("block_id"), p.get("panel_idx")))
                if p.get("max_overlap", 0) > 0.04 + 1e-5:
                    violations_overlap_2c.append((p.get("block_id"), p.get("panel_idx"), p["max_overlap"]))

        print(f"    * Area limits [0.70, 1.30] use_two_col_lattice: {'PASS' if not violations_area_2c else f'FAIL {violations_area_2c}'}")
        print(f"    * Outside image check: {'PASS' if not violations_outside_2c else f'FAIL {violations_outside_2c}'}")
        print(f"    * Max overlap <= 0.04: {'PASS' if not violations_overlap_2c else f'WARNING {violations_overlap_2c}'}")

        debug_img_2c = f"data/results/debug/debug_{image_name}_two_col_lattice_refine.JPG"
        print(f"  - Debug image: {debug_img_2c} {'EXISTS' if os.path.exists(debug_img_2c) else 'MISSING'}")
    else:
        print(f"\n  - No two-column block lattice log found at {two_col_log_path}")
        if "0987" in image_name.upper() or "0995" in image_name.upper():
            print(f"    → BYPASS expected for large-panel image: CONFIRMED")

    # Outer Boundary Guard stats
    outer_panels = []
    if os.path.exists(string_log_path):
        with open(string_log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                if data.get("record_type") == "panel" and data.get("is_outer_panel"):
                    outer_panels.append(data)
    if os.path.exists(two_col_log_path):
        block_outer = {}
        with open(two_col_log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                if data.get("record_type") == "panel" and data.get("is_outer_panel"):
                    block_outer[data["panel_idx"]] = data
        combined_outer = {}
        for op in outer_panels:
            combined_outer[op["panel_idx"]] = op
        for idx, op in block_outer.items():
            combined_outer[idx] = op
        outer_panels = list(combined_outer.values())

    if outer_panels:
        print(f"\nOuter Boundary Guard Statistics:")
        print(f"  - Number of outer panels: {len(outer_panels)}")
        
        decisions = {}
        scores = []
        area_ratios = []
        center_shifts = []
        overlaps = []
        reasons = {}
        
        for op in outer_panels:
            dec = op.get("outer_guard_decision", "n/a")
            dec_final = op.get("decision")
            if dec_final == "fallback_outer_panel":
                dec = "fallback_outer_panel"
            elif dec == "n/a":
                dec = op.get("snap_decision", "keep_lattice")
                
            decisions[dec] = decisions.get(dec, 0) + 1
            
            if "outer_divider_score" in op:
                scores.append(op["outer_divider_score"])
            if "area_ratio" in op:
                area_ratios.append(op["area_ratio"])
            if "center_shift" in op:
                center_shifts.append(op["center_shift"])
            elif "center_shift_ratio" in op:
                center_shifts.append(op["center_shift_ratio"])
            if "max_overlap" in op:
                overlaps.append(op["max_overlap"])
                
            r = op.get("outer_guard_reason", "ok")
            if r is None or r == "":
                r = "ok"
            reasons[r] = reasons.get(r, 0) + 1

        use_outer_snap = decisions.get("use_outer_snap", 0)
        keep_lattice = decisions.get("keep_lattice", 0) + decisions.get("use_snapped", 0) + decisions.get("use_lattice", 0)
        fallback_outer_panel = decisions.get("fallback_outer_panel", 0)
        
        med_score = float(np.median(scores)) if scores else 0.0
        max_area = float(np.max(area_ratios)) if area_ratios else 0.0
        max_shift = float(np.max(center_shifts)) if center_shifts else 0.0
        max_overlap = float(np.max(overlaps)) if overlaps else 0.0
        
        print(f"    * use_outer_snap: {use_outer_snap}")
        print(f"    * keep_lattice: {keep_lattice}")
        print(f"    * fallback_outer_panel: {fallback_outer_panel}")
        print(f"    * median outer_divider_score: {med_score:.4f}")
        print(f"    * max outer panel area ratio: {max_area:.4f}")
        print(f"    * max outer panel center shift: {max_shift:.4f}")
        print(f"    * max overlap: {max_overlap:.4f}")
        print(f"    * top outer_guard_reason: {reasons}")
        
        outer_guard_img = f"data/results/debug/debug_{image_name}_outer_guard.JPG"
        print(f"  - Outer Guard debug image: {outer_guard_img} {'EXISTS' if os.path.exists(outer_guard_img) else 'MISSING'}")
    else:
        if "0987" in image_name.upper() or "0995" in image_name.upper():
            print(f"\n  - Outer Boundary Guard: BYPASS CONFIRMED")

    # Endpoint Guard stats
    endpoint_panels = []
    if os.path.exists(string_log_path):
        with open(string_log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                if data.get("record_type") == "panel" and (data.get("is_endpoint_panel") or data.get("is_endpoint")):
                    endpoint_panels.append(data)
    if os.path.exists(two_col_log_path):
        block_endpoint = {}
        with open(two_col_log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                if data.get("record_type") == "panel" and (data.get("is_endpoint_panel") or data.get("is_endpoint")):
                    block_endpoint[data["panel_idx"]] = data
        combined_endpoint = {}
        for ep in endpoint_panels:
            combined_endpoint[ep["panel_idx"]] = ep
        for idx, ep in block_endpoint.items():
            combined_endpoint[idx] = ep
        endpoint_panels = list(combined_endpoint.values())

    if endpoint_panels:
        print(f"\nEndpoint Guard Statistics:")
        print(f"  - Total endpoint panels: {len(endpoint_panels)}")
        
        decisions = {}
        area_ratios = []
        center_shifts = []
        edge_ratios = []
        overlaps = []
        reasons = {}
        length_ratios = []
        outer_scores = []
        source_counts = {
            "yolo_projection": 0,
            "pitch_extrapolation": 0,
            "edge_scan": 0,
            "consensus": 0,
            "middle_locked_pitch": 0,
            "fallback_original": 0
        }
        
        endpoint_pass_count = 0
        endpoint_fallback_count = 0
        
        for ep in endpoint_panels:
            dec = ep.get("endpoint_decision", "n/a")
            decisions[dec] = decisions.get(dec, 0) + 1
            
            if dec in ("use_lattice_endpoint", "use_middle_locked_endpoint"):
                endpoint_pass_count += 1
            elif dec == "fallback_endpoint_original":
                endpoint_fallback_count += 1
                
            if "candidate_lattice_area_ratio" in ep:
                area_ratios.append(ep["candidate_lattice_area_ratio"])
            if "candidate_lattice_center_shift" in ep:
                center_shifts.append(ep["candidate_lattice_center_shift"])
            if "edge_score_ratio" in ep:
                edge_ratios.append(ep["edge_score_ratio"])
            if "max_overlap" in ep:
                overlaps.append(ep["max_overlap"])
            if "endpoint_length_ratio" in ep:
                length_ratios.append(ep["endpoint_length_ratio"])
            if "endpoint_outer_score" in ep:
                outer_scores.append(ep["endpoint_outer_score"])
                
            src = ep.get("endpoint_outer_source", "fallback_original")
            if src in source_counts:
                source_counts[src] += 1
                
            r = ep.get("endpoint_fallback_reason", ep.get("endpoint_reason", "ok"))
            if not r:
                r = "ok"
            reasons[r] = reasons.get(r, 0) + 1
            
        print(f"    * endpoint_pass_count: {endpoint_pass_count}")
        print(f"    * endpoint_fallback_count: {endpoint_fallback_count}")
        print(f"    * endpoint_source_counts: {source_counts}")
        print(f"    * median endpoint_length_ratio: {float(np.median(length_ratios)) if length_ratios else 0.0:.4f}")
        print(f"    * max endpoint_length_ratio: {float(np.max(length_ratios)) if length_ratios else 0.0:.4f}")
        print(f"    * median endpoint_outer_score: {float(np.median(outer_scores)) if outer_scores else 0.0:.4f}")
        print(f"    * fallback reasons: {reasons}")
        
        # String/Block count & pass ratios
        s_count = 0
        b_count = 0
        middle_pass_ratios = []
        
        if os.path.exists(string_log_path):
            with open(string_log_path, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line.strip())
                    if data.get("record_type") == "string":
                        s_count += 1
                        if "middle_pass_ratio" in data:
                            middle_pass_ratios.append(data["middle_pass_ratio"])
                            
        if os.path.exists(two_col_log_path):
            with open(two_col_log_path, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line.strip())
                    if data.get("record_type") == "two_col_block":
                        b_count += 1
                        if "middle_pass_ratio" in data:
                            middle_pass_ratios.append(data["middle_pass_ratio"])
                            
        print(f"    * String count: {s_count}")
        print(f"    * Block count: {b_count}")
        if middle_pass_ratios:
            print(f"    * median middle_pass_ratio: {float(np.median(middle_pass_ratios)):.4f}")
        
        ep_img = f"data/results/debug/debug_{image_name}_endpoint_hybrid.JPG"
        print(f"  - Endpoint Guard hybrid debug image: {ep_img} {'EXISTS' if os.path.exists(ep_img) else 'MISSING'}")
        ml_img = f"data/results/debug/debug_{image_name}_middle_locked_endpoint.JPG"
        print(f"  - Middle-locked endpoint debug image: {ml_img} {'EXISTS' if os.path.exists(ml_img) else 'MISSING'}")
    else:
        if "0987" in image_name.upper() or "0995" in image_name.upper():
            print(f"\n  - Endpoint Guard: BYPASS CONFIRMED")

    # True Block Lattice stats
    tbl_log_path = f"data/results/debug_logs/{image_name}_true_block_lattice.jsonl"
    if os.path.exists(tbl_log_path):
        print(f"\nTrue Block Lattice Log ({tbl_log_path}):")
        tbl_blocks = []
        tbl_panels = []
        with open(tbl_log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                if data.get("record_type") == "block":
                    tbl_blocks.append(data)
                elif data.get("record_type") == "panel":
                    tbl_panels.append(data)

        n_blocks = len(tbl_blocks)
        n_use = sum(1 for b in tbl_blocks if b["decision"] == "use_true_block_lattice")
        n_fallback_baseline = sum(1 for b in tbl_blocks if b["decision"] == "fallback_baseline")
        n_fallback_block = sum(1 for b in tbl_blocks if b["decision"] == "fallback_block")
        print(f"  - Blocks: {n_blocks} total | use={n_use} | fallback_baseline={n_fallback_baseline} | fallback_block={n_fallback_block}")
        for b in tbl_blocks:
            print(f"    * block={b['block_id']} n={b['n_panels']} cols={b['n_cols']} "
                  f"angle={b['angle_deg']:.1f} pitch={b['pitch']:.1f} "
                  f"mid_pass={b['middle_pass_ratio']:.2f} decision={b['decision']} reason={b.get('fallback_reason', 'n/a')}")

        n_panels_tbl = len(tbl_panels)
        n_panels_use = sum(1 for p in tbl_panels if p["decision"] == "use_true_block_lattice")
        n_panels_fallback_baseline = sum(1 for p in tbl_panels if p["decision"] == "fallback_baseline")
        n_panels_fallback_other = sum(1 for p in tbl_panels if p["decision"] not in ("use_true_block_lattice", "fallback_baseline"))
        reasons_tbl = {}
        for p in tbl_panels:
            r = p.get("fallback_reason", "ok")
            reasons_tbl[r] = reasons_tbl.get(r, 0) + 1
        for b in tbl_blocks:
            if b["decision"] != "use_true_block_lattice":
                r = b.get("fallback_reason", "ok")
                reasons_tbl[r] = reasons_tbl.get(r, 0) + 1

        expansions = []
        for b in tbl_blocks:
            expansions.extend([abs(b.get("rail_expansion_low", 0.0)), abs(b.get("rail_expansion_high", 0.0))])
        med_exp = float(np.median(expansions)) if expansions else 0.0
        max_exp = float(np.max(expansions)) if expansions else 0.0

        center_shifts = [p.get("center_shift", 0.0) for p in tbl_panels if "center_shift" in p]
        med_shift = float(np.median(center_shifts)) if center_shifts else 0.0
        max_shift = float(np.max(center_shifts)) if center_shifts else 0.0

        print(f"  - Panels: {n_panels_tbl} total | use_true_block_lattice={n_panels_use} | fallback_baseline={n_panels_fallback_baseline} | fallback_other={n_panels_fallback_other}")
        print(f"  - Top fallback reasons: {dict(sorted(reasons_tbl.items(), key=lambda x: -x[1])[:8])}")
        print(f"  - Median rail_expansion: {med_exp:.2f} px | Max rail_expansion: {max_exp:.2f} px")
        print(f"  - Median center_shift: {med_shift:.4f} | Max center_shift: {max_shift:.4f}")

        tbl_dbg = f"data/results/debug/debug_{image_name}_true_block_lattice.JPG"
        tbl_dbg_compare = f"data/results/debug/debug_{image_name}_true_block_lattice_compare.JPG"
        print(f"  - Debug image: {tbl_dbg} {'EXISTS' if os.path.exists(tbl_dbg) else 'MISSING'}")
        print(f"  - Compare image: {tbl_dbg_compare} {'EXISTS' if os.path.exists(tbl_dbg_compare) else 'MISSING'}")
    else:
        if "0987" in image_name.upper() or "0995" in image_name.upper():
            print(f"\n  - True Block Lattice: BYPASS CONFIRMED")
        else:
            print(f"\n  - True Block Lattice: no log found at {tbl_log_path}")

    # Task 10 print metrics & assertions
    print(f"\n[STRATEGIC_METRICS] img={image_name}:")
    
    block_count = len(strings_log)
    passed_blocks = sum(1 for s in strings_log if s["decision"] == "use_string_lattice")
    fallback_blocks = sum(1 for s in strings_log if s["decision"] != "use_string_lattice")
    
    string_lattice_middle_count = sum(1 for p in panels_log if p.get("final_geometry_source") == "string_lattice_middle")
    string_lattice_endpoint_count = sum(1 for p in panels_log if p.get("final_geometry_source") == "string_lattice_endpoint")
    endpoint_original_fallback_count = sum(1 for p in panels_log if p.get("final_geometry_source") == "endpoint_original_fallback")
    block_fallback_original_count = sum(1 for p in panels_log if p.get("final_geometry_source") == "block_fallback_original")
    
    endpoint_panels = [p for p in panels_log if p.get("role") in ("endpoint_start", "endpoint_end")]
    failed_endpoint_count = sum(1 for p in endpoint_panels if p.get("final_geometry_source") in ("endpoint_original_fallback", "block_fallback_original"))
    
    middle_panels = [p for p in panels_log if p.get("role") == "middle"]
    failed_middle_count = sum(1 for p in middle_panels if p.get("final_geometry_source") in ("yolo_original", "block_fallback_original"))
    
    divider_cross_count = sum(1 for p in panels_log if p.get("is_crossed", False))
    
    coverages = [p.get("original_coverage_ratio", 1.0) for p in middle_panels]
    median_middle_original_coverage = float(np.median(coverages)) if coverages else 1.0
    min_middle_original_coverage = float(np.min(coverages)) if coverages else 1.0
    
    max_inward_cut_ratio = float(np.max([p.get("max_inward_cut_ratio", 0.0) for p in panels_log])) if panels_log else 0.0
    outside_image_count_log = sum(1 for p in panels_log if p.get("outside_image", False))
    
    print(f"  - block count: {block_count}")
    print(f"  - passed blocks: {passed_blocks} | fallback blocks: {fallback_blocks}")
    print(f"  - string_lattice_middle count: {string_lattice_middle_count}")
    print(f"  - string_lattice_endpoint count: {string_lattice_endpoint_count}")
    print(f"  - endpoint_original_fallback count: {endpoint_original_fallback_count}")
    print(f"  - block_fallback_original count: {block_fallback_original_count}")
    print(f"  - failed endpoint count: {failed_endpoint_count}")
    print(f"  - failed middle count: {failed_middle_count}")
    print(f"  - divider_cross_count: {divider_cross_count}")
    print(f"  - median_middle_original_coverage: {median_middle_original_coverage:.4f}")
    print(f"  - min_middle_original_coverage: {min_middle_original_coverage:.4f}")
    print(f"  - max_inward_cut_ratio: {max_inward_cut_ratio:.4f}")
    print(f"  - outside_image_count (log): {outside_image_count_log}")

    string_panels_used = sum(1 for p in panels if p.get("final_polygon_source") in ("string_lattice_middle", "string_lattice_endpoint", "middle_locked_endpoint"))
    ep_repaired = sum(1 for p in panels if p.get("endpoint_repair_used") == True)
    ep_fallback = sum(1 for p in panels if p.get("is_endpoint_panel") == True and p.get("endpoint_repair_used") == False)
    two_col_app = sum(1 for p in panels if p.get("two_col_applied") == True)
    tbl_app = sum(1 for p in panels if p.get("tbl_applied") == True)
    
    # Compute max overlap
    max_overlap_val = 0.0
    for i, p1 in enumerate(panels):
        poly1 = np.array(p1["polygon"], dtype=np.float32)
        if len(poly1) < 3:
            continue
        h1 = cv2.convexHull(poly1.reshape(-1, 1, 2)).astype(np.float32)
        for j, p2 in enumerate(panels):
            if i == j:
                continue
            poly2 = np.array(p2["polygon"], dtype=np.float32)
            if len(poly2) < 3:
                continue
            h2 = cv2.convexHull(poly2.reshape(-1, 1, 2)).astype(np.float32)
            ai, _ = cv2.intersectConvexConvex(h1, h2)
            if ai > 0:
                min_a = min(cv2.contourArea(h1), cv2.contourArea(h2))
                overlap = ai / max(min_a, 1e-3)
                max_overlap_val = max(max_overlap_val, overlap)
                
    # Compute outside image count
    outside_image_count = 0
    h_img, w_img = result.orig_img.shape[:2]
    for p in panels:
        poly = p["polygon"]
        outside = any(pt[0] < 0 or pt[0] > w_img or pt[1] < 0 or pt[1] > h_img for pt in poly)
        if outside:
            outside_image_count += 1
            
    print(f"  - string_lattice panels used: {string_panels_used}")
    print(f"  - endpoint repaired count: {ep_repaired}")
    print(f"  - endpoint fallback original count: {ep_fallback}")
    print(f"  - two_col applied count: {two_col_app}")
    print(f"  - TBL applied count: {tbl_app}")
    
    geom_sources = {}
    for p in panels:
        src = p.get("final_polygon_source", "yolo_original")
        geom_sources[src] = geom_sources.get(src, 0) + 1
    print(f"  - final_geometry_source counts:")
    for src in ("string_lattice_middle", "string_lattice_endpoint", "endpoint_original_fallback", "block_fallback_original", "yolo_original"):
        print(f"    * {src}: {geom_sources.get(src, 0)}")
    
    # Print blocks by decision
    print("  - blocks by decision:")
    for decision in ("use_string_lattice", "recoverable_string", "fallback_string"):
        dec_blocks = [s for s in strings_log if s.get("decision") == decision]
        print(f"    * {decision}: {[s.get('block_id') for s in dec_blocks]}")
        
    print(f"  - max overlap cuoi cung: {max_overlap_val:.4f}")
    print(f"  - outside image count: {outside_image_count}")
    
    # Assert checks
    for p in panels:
        src = p.get("final_polygon_source")
        if src == "true_block_lattice":
            sys.exit(f"FAIL: final panel raw_idx={p.get('raw_idx')} has final_geometry_source == 'true_block_lattice'")
        if src in ("two_col_block_lattice", "two_col_block_middle", "two_col_block_endpoint"):
            sys.exit(f"FAIL: final panel raw_idx={p.get('raw_idx')} has final_geometry_source == 'two_col_block_lattice'")
            
    if outside_image_count > 0:
        sys.exit(f"FAIL: {outside_image_count} panels are outside the image bounds")
        
    for p in panels_log:
        src = p.get("final_geometry_source")
        cov = p.get("original_coverage_ratio", 1.0)
        cut = p.get("max_inward_cut_ratio", 0.0)
        crossed = p.get("is_crossed", False)
        
        if src == "string_lattice_middle" and cov < 0.85:
            sys.exit(f"FAIL: passed middle panel has original_coverage_ratio {cov} < 0.85")
        if src == "string_lattice_endpoint" and cov < 0.78:
            sys.exit(f"FAIL: passed endpoint panel has original_coverage_ratio {cov} < 0.78")
        if src == "string_lattice_middle" and cut > 0.20:
            sys.exit(f"FAIL: passed middle panel has max_inward_cut_ratio {cut} > 0.20")
        if src in ("string_lattice_middle", "string_lattice_endpoint") and crossed:
            sys.exit(f"FAIL: accepted lattice panel has is_crossed == True")

    if image_name in ("DJI_0987", "DJI_0995"):
        # Assert they are fully bypassed (no string lattice panels used)
        for p in panels:
            src = p.get("final_polygon_source", "yolo_original")
            if src != "yolo_original":
                sys.exit(f"FAIL: {image_name} has non-YOLO geometry source {src}")
                
    if image_name in ("DJI_0843_R", "DJI_0845_R"):
        num_passed_blocks = sum(1 for s in strings_log if s["decision"] == "use_string_lattice")
        if num_passed_blocks == 0:
            sys.exit(f"FAIL: all blocks fallback unexpectedly on {image_name}")
