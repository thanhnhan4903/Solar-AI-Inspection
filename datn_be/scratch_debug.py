import os
import py_compile

target_path = "app/services/panel_processor.py"

new_fit_code = """def fit_string_parallel_lines(
    string_info: Dict[str, Any],
    image_shape: Tuple[int, int, int],
    image: Optional[np.ndarray] = None,
    median_panel_area: float = 1.0
) -> Dict[str, Any]:
    \"\"\"
    Fit 2 parallel rails + dividers for a single string.
    Endpoint Isolation Architecture:
      - Rails fit from MIDDLE panels only (not endpoints).
      - u0 / pitch / dividers fit from MIDDLE panels only.
      - Endpoint candidates built from Middle-Locked reconstruction.
    Returns candidate polygons + validation results for each panel.
    \"\"\"
    img_h, img_w = image_shape[:2]
    panels = string_info["panels"]
    trusted_mask = string_info["trusted_mask"]
    u_axis = string_info["u_axis"]
    v_axis = string_info["v_axis"]
    pitch = string_info["pitch"]
    origin = string_info["origin"]
    n_trusted = string_info["n_trusted"]
    local_angle_deg = string_info.get("angle_deg", 0.0)
    n = len(panels)

    # Sort panels and trusted_mask by u_axis projection to ensure strict order
    centers_u = [float(np.dot(np.array(p["center"]) - origin, u_axis)) for p in panels]
    sorted_idx = np.argsort(centers_u)
    panels = [panels[idx] for idx in sorted_idx]
    trusted_mask = [trusted_mask[idx] for idx in sorted_idx]

    # Return failure if n < 4
    if n < 4:
        return {"success": False, "reason": "string_too_short"}

    # --- Step 1: Classify panel roles ---
    # Initialize all as middle
    for p in panels:
        p["panel_role"] = "middle"
        p["is_endpoint"] = False
        p["endpoint_side"] = None
        p["is_endpoint_panel"] = False

    # Mark endpoints
    panels[0]["panel_role"] = "endpoint_start"
    panels[0]["is_endpoint"] = True
    panels[0]["endpoint_side"] = "start"
    panels[0]["is_endpoint_panel"] = True

    panels[n-1]["panel_role"] = "endpoint_end"
    panels[n-1]["is_endpoint"] = True
    panels[n-1]["endpoint_side"] = "end"
    panels[n-1]["is_endpoint_panel"] = True

    middle_indices = [k for k in range(n) if panels[k]["panel_role"] == "middle"]
    middle_trusted_indices = [k for k in middle_indices if trusted_mask[k]]
    middle_panels = [panels[k] for k in middle_indices]

    # Check minimum middle panels
    min_middle_needed = 2 if n == 4 else 3
    if len(middle_indices) < min_middle_needed:
        return {"success": False, "reason": f"too_few_middle_panels_{len(middle_indices)}"}

    # --- Step 1b: Re-fit u_axis, v_axis using PCA on middle panel centers ---
    if len(middle_panels) >= 2:
        centers = np.array([p["center"] for p in middle_panels], dtype=np.float32)
        mean_center = np.mean(centers, axis=0)
        origin = mean_center  # Shift origin to mean center of middle panels for stability
        centered = centers - mean_center
        cov = np.cov(centered.T)
        if cov.ndim == 2:
            eigvals, eigvecs = np.linalg.eigh(cov)
            u_pca = eigvecs[:, np.argmax(eigvals)].astype(np.float32)
            u_pca /= (np.linalg.norm(u_pca) + 1e-8)
            if np.dot(u_pca, u_axis) < 0:
                u_pca = -u_pca
            u_axis = u_pca
            v_axis = np.array([-u_axis[1], u_axis[0]], dtype=np.float32)
            if np.dot(v_axis, string_info["v_axis"]) < 0:
                v_axis = -v_axis
            local_angle_deg = float(np.degrees(np.arctan2(u_axis[1], u_axis[0])))

    # --- Step 2: Fit rails from middle panels ---
    if len(middle_trusted_indices) >= 2:
        fit_panels = [panels[k] for k in middle_trusted_indices]
        rail_fit_source = "middle_trusted_panels"
    else:
        fit_panels = middle_panels
        rail_fit_source = "middle_all_panels"

    all_v_proj = []
    for p in fit_panels:
        pts = np.array(p["polygon"], dtype=np.float32)
        for pt in pts:
            all_v_proj.append(float(np.dot(pt - origin, v_axis)))

    if len(all_v_proj) < 4:
        return {"success": False, "reason": "too_few_points_for_rail"}

    if len(middle_indices) >= 5:
        v_low = float(np.percentile(all_v_proj, 3.0))
        v_high = float(np.percentile(all_v_proj, 97.0))
    else:
        # Symmetrical distribution based on median width & center
        widths = []
        centers_v = []
        for p in fit_panels:
            proj_v = [np.dot(pt - origin, v_axis) for pt in p["polygon"]]
            widths.append(max(proj_v) - min(proj_v))
            centers_v.append((max(proj_v) + min(proj_v)) / 2.0)
        median_width = float(np.median(widths)) if widths else 1.0
        median_center_v = float(np.median(centers_v)) if centers_v else 0.0
        v_low = median_center_v - 0.5 * median_width
        v_high = median_center_v + 0.5 * median_width

    if v_high <= v_low:
        return {"success": False, "reason": "rail_degenerate"}

    # --- Step 3: Fit pitch from middle panels ---
    if len(middle_trusted_indices) >= 2:
        u_for_pitch = sorted([
            float(np.dot(np.array(panels[k]["center"]) - origin, u_axis))
            for k in middle_trusted_indices
        ])
    else:
        u_for_pitch = sorted([
            float(np.dot(np.array(panels[k]["center"]) - origin, u_axis))
            for k in middle_indices
        ])

    if len(u_for_pitch) < 2:
        pitch_fit = pitch
    else:
        t_gaps = [u_for_pitch[k + 1] - u_for_pitch[k] for k in range(len(u_for_pitch) - 1)
                  if u_for_pitch[k + 1] > u_for_pitch[k]]
        if not t_gaps:
            pitch_fit = pitch
        else:
            raw_median = float(np.median(t_gaps))
            filtered_gaps = [g for g in t_gaps if 0.5 * raw_median <= g <= 1.5 * raw_median]
            pitch_fit = float(np.median(filtered_gaps)) if filtered_gaps else raw_median

    if pitch_fit <= 0:
        pitch_fit = pitch

    # u0: first middle panel center projection
    panel_u_proj = [float(np.dot(np.array(p["center"]) - origin, u_axis)) for p in panels]
    u0 = panel_u_proj[0]

    # Assign integer grid positions
    positions = []
    for u in panel_u_proj:
        pos_int = int(round((u - u0) / max(pitch_fit, 1e-3)))
        positions.append(pos_int)

    # Median middle panel area
    middle_areas = []
    for mid_idx in middle_indices:
        m_poly = np.array(panels[mid_idx]["polygon"], dtype=np.float32)
        middle_areas.append(float(cv2.contourArea(m_poly.reshape(-1, 1, 2))))
    median_middle_area = float(np.median(middle_areas)) if middle_areas else median_panel_area

    # --- Step 4: Build candidate polygons & validate ---
    panel_details = []

    for k, (p, pos) in enumerate(zip(panels, positions)):
        panel_role = p.get("panel_role", "middle")
        is_endpoint = p.get("is_endpoint", False)
        endpoint_side = p.get("endpoint_side", None)
        used_for_lattice_fit = (k in middle_trusted_indices) and not is_endpoint

        orig_poly = p.get("original_yolo_polygon") or p["polygon"]
        orig_area = p.get("area", 0.0)
        orig_pts = np.array(orig_poly, dtype=np.float32)
        orig_center = np.array(p["center"], dtype=np.float32)

        endpoint_extrapolate_ratio = 0.0
        candidate_polygon_source = "lattice_full"

        endpoint_yolo_outer_u = 0.0
        endpoint_pitch_outer_u = 0.0
        endpoint_edge_outer_u = 0.0
        endpoint_consensus_outer_u = 0.0
        endpoint_final_outer_u = 0.0
        endpoint_outer_source = "n/a"
        endpoint_outer_score = 0.0
        endpoint_candidates = []

        is_yolo_endpoint_small = False
        if is_endpoint:
            is_yolo_endpoint_small = (orig_area < 0.65 * median_middle_area)

        # Border proximity check
        touches_border = False
        for pt in orig_poly:
            if pt[0] <= 3 or pt[0] >= img_w - 3 or pt[1] <= 3 or pt[1] >= img_h - 3:
                touches_border = True
                break

        if is_endpoint:
            # Endpoint logic
            if endpoint_side == "start":
                if positions[1] - positions[0] == 1:
                    inner_divider_u = (panel_u_proj[0] + panel_u_proj[1]) / 2.0
                else:
                    inner_divider_u = panel_u_proj[0] + 0.5 * pitch_fit
                outer_u_candidate = inner_divider_u - pitch_fit
                yolo_u_values = [np.dot(pt - origin, u_axis) for pt in orig_poly]
                yolo_outer_u = min(yolo_u_values)
                max_shift = 0.35 * pitch_fit
                shift = outer_u_candidate - yolo_outer_u
                outer_u = yolo_outer_u + np.clip(shift, -max_shift, max_shift)
            else: # "end"
                if positions[n-1] - positions[n-2] == 1:
                    inner_divider_u = (panel_u_proj[n-2] + panel_u_proj[n-1]) / 2.0
                else:
                    inner_divider_u = panel_u_proj[n-1] - 0.5 * pitch_fit
                outer_u_candidate = inner_divider_u + pitch_fit
                yolo_u_values = [np.dot(pt - origin, u_axis) for pt in orig_poly]
                yolo_outer_u = max(yolo_u_values)
                max_shift = 0.35 * pitch_fit
                shift = outer_u_candidate - yolo_outer_u
                outer_u = yolo_outer_u + np.clip(shift, -max_shift, max_shift)

            endpoint_final_outer_u = outer_u
            endpoint_pitch_outer_u = outer_u
            endpoint_extrapolate_ratio = abs(outer_u - inner_divider_u) / max(pitch_fit, 1e-3)

            if touches_border:
                candidate_polygon = orig_poly
                candidate_polygon_source = "endpoint_touch_image_border_keep_yolo"
                pass_val = False
                fail_reason = "endpoint_touch_image_border_keep_yolo"
            else:
                pt1 = origin + inner_divider_u * u_axis + v_low * v_axis
                pt2 = origin + inner_divider_u * u_axis + v_high * v_axis
                pt3 = origin + outer_u * u_axis + v_high * v_axis
                pt4 = origin + outer_u * u_axis + v_low * v_axis
                candidate_polygon = _sort_corners(np.array([pt1, pt2, pt3, pt4], dtype=np.float32))
                candidate_polygon_source = "middle_locked_endpoint"
                
                # Validation checks for endpoint candidate
                cand_pts = np.array(candidate_polygon, dtype=np.float32)
                candidate_area = float(cv2.contourArea(cand_pts.reshape(-1, 1, 2)))
                cand_center = np.mean(cand_pts, axis=0)
                iou_with_original = _compute_mask_iou_cv(cand_pts, orig_pts, (img_h, img_w))
                
                orig_diag = float(np.linalg.norm(
                    [p.get("bbox", p.get("box", [0, 0, 0, 0]))[2] - p.get("bbox", p.get("box", [0, 0, 0, 0]))[0],
                     p.get("bbox", p.get("box", [0, 0, 0, 0]))[3] - p.get("bbox", p.get("box", [0, 0, 0, 0]))[1]]
                ))
                center_shift = float(np.linalg.norm(cand_center - orig_center))
                center_shift_ratio = center_shift / max(orig_diag, 1.0)
                area_ratio = candidate_area / max(orig_area, 1.0)

                outside_image = False
                for pt in candidate_polygon:
                    if pt[0] < -2.0 or pt[0] > img_w + 2.0 or pt[1] < -2.0 or pt[1] > img_h + 2.0:
                        outside_image = True
                        break

                neighbor_idx = 1 if endpoint_side == "start" else n - 2
                overlap_val = 0.0
                if 0 <= neighbor_idx < n:
                    if neighbor_idx == 0 or neighbor_idx == n - 1:
                        n_u_l = panel_u_proj[neighbor_idx] - 0.5 * pitch_fit
                        n_u_r = panel_u_proj[neighbor_idx] + 0.5 * pitch_fit
                    else:
                        if positions[neighbor_idx] - positions[neighbor_idx-1] == 1:
                            n_u_l = (panel_u_proj[neighbor_idx-1] + panel_u_proj[neighbor_idx]) / 2.0
                        else:
                            n_u_l = panel_u_proj[neighbor_idx] - 0.5 * pitch_fit
                        
                        if positions[neighbor_idx+1] - positions[neighbor_idx] == 1:
                            n_u_r = (panel_u_proj[neighbor_idx] + panel_u_proj[neighbor_idx+1]) / 2.0
                        else:
                            n_u_r = panel_u_proj[neighbor_idx] + 0.5 * pitch_fit
                    
                    neighbor_cand_poly = np.array([
                        origin + n_u_l * u_axis + v_low * v_axis,
                        origin + n_u_r * u_axis + v_low * v_axis,
                        origin + n_u_r * u_axis + v_high * v_axis,
                        origin + n_u_l * u_axis + v_high * v_axis,
                    ], dtype=np.float32)
                    overlap_val = compute_polygon_overlap_ratio(candidate_polygon, [[int(round(pt[0])), int(round(pt[1]))] for pt in neighbor_cand_poly], (img_h, img_w))

                pass_val = True
                fail_reason = "ok"

                if not (0.70 <= area_ratio <= 1.30):
                    pass_val = False
                    fail_reason = f"endpoint_area_ratio_out_{area_ratio:.2f}"
                elif center_shift_ratio > 0.35:
                    pass_val = False
                    fail_reason = f"endpoint_center_shift_{center_shift_ratio:.2f}"
                elif iou_with_original < 0.40:
                    pass_val = False
                    fail_reason = f"endpoint_iou_low_{iou_with_original:.2f}"
                elif overlap_val > 0.08:
                    pass_val = False
                    fail_reason = f"endpoint_overlap_too_high_{overlap_val:.3f}"
                elif outside_image:
                    pass_val = False
                    fail_reason = "outside_image_bounds"

        else:
            # Middle panel boundaries
            if k > 0:
                if positions[k] - positions[k-1] == 1:
                    u_left = (panel_u_proj[k-1] + panel_u_proj[k]) / 2.0
                else:
                    u_left = panel_u_proj[k] - 0.5 * pitch_fit
            else:
                u_left = panel_u_proj[k] - 0.5 * pitch_fit

            if k < n - 1:
                if positions[k+1] - positions[k] == 1:
                    u_right = (panel_u_proj[k] + panel_u_proj[k+1]) / 2.0
                else:
                    u_right = panel_u_proj[k] + 0.5 * pitch_fit
            else:
                u_right = panel_u_proj[k] + 0.5 * pitch_fit

            candidate_polygon = [
                [float(pt[0]), float(pt[1])] for pt in [
                    origin + u_left * u_axis + v_low * v_axis,
                    origin + u_right * u_axis + v_low * v_axis,
                    origin + u_right * u_axis + v_high * v_axis,
                    origin + u_left * u_axis + v_high * v_axis,
                ]
            ]
            candidate_polygon = _sort_corners(np.array(candidate_polygon, dtype=np.float32))
            candidate_polygon_source = "lattice_full"

            # Validate middle panel candidate
            cand_pts = np.array(candidate_polygon, dtype=np.float32)
            candidate_area = float(cv2.contourArea(cand_pts.reshape(-1, 1, 2)))
            cand_center = np.mean(cand_pts, axis=0)
            iou_with_original = _compute_mask_iou_cv(cand_pts, orig_pts, (img_h, img_w))

            orig_diag = float(np.linalg.norm(
                [p.get("bbox", p.get("box", [0, 0, 0, 0]))[2] - p.get("bbox", p.get("box", [0, 0, 0, 0]))[0],
                 p.get("bbox", p.get("box", [0, 0, 0, 0]))[3] - p.get("bbox", p.get("box", [0, 0, 0, 0]))[1]]
            ))
            center_shift = float(np.linalg.norm(cand_center - orig_center))
            center_shift_ratio = center_shift / max(orig_diag, 1.0)
            area_ratio = candidate_area / max(orig_area, 1.0)

            cand_u_proj = [float(np.dot(pt - origin, u_axis)) for pt in cand_pts]
            cand_v_proj = [float(np.dot(pt - origin, v_axis)) for pt in cand_pts]
            cand_w = max(cand_u_proj) - min(cand_u_proj)
            cand_h = max(cand_v_proj) - min(cand_v_proj)

            orig_u_proj = [float(np.dot(pt - origin, u_axis)) for pt in orig_pts]
            orig_v_proj = [float(np.dot(pt - origin, v_axis)) for pt in orig_pts]
            orig_w = max(orig_u_proj) - min(orig_u_proj) if orig_u_proj else 1.0
            orig_h = max(orig_v_proj) - min(orig_v_proj) if orig_v_proj else 1.0

            width_ratio = cand_w / max(orig_w, 1e-3)
            height_ratio = cand_h / max(orig_h, 1e-3)

            out_of_bounds = any(
                pt[0] < -STRING_PANEL_OUTSIDE_TOL_PX or pt[0] > img_w + STRING_PANEL_OUTSIDE_TOL_PX or
                pt[1] < -STRING_PANEL_OUTSIDE_TOL_PX or pt[1] > img_h + STRING_PANEL_OUTSIDE_TOL_PX
                for pt in candidate_polygon
            )

            pass_val = True
            fail_reason = "ok"

            if out_of_bounds:
                pass_val = False
                fail_reason = "outside_image"
            elif not (STRING_PANEL_AREA_RATIO_MIN <= area_ratio <= STRING_PANEL_AREA_RATIO_MAX):
                pass_val = False
                fail_reason = "area_ratio_out"
            elif center_shift_ratio > STRING_PANEL_CENTER_SHIFT_MAX:
                pass_val = False
                fail_reason = "center_shift_too_large"
            elif iou_with_original < STRING_PANEL_IOU_MIN:
                pass_val = False
                fail_reason = "iou_too_low"
            elif not (STRING_PANEL_WIDTH_RATIO_MIN <= width_ratio <= STRING_PANEL_WIDTH_RATIO_MAX):
                pass_val = False
                fail_reason = "width_ratio_out"
            elif not (STRING_PANEL_HEIGHT_RATIO_MIN <= height_ratio <= STRING_PANEL_HEIGHT_RATIO_MAX):
                pass_val = False
                fail_reason = "height_ratio_out"

        # Edge score (diagnostic only, no hard gate)
        edge_score_orig = score_panel_boundary_alignment(image, orig_poly) if image is not None else 0.0
        edge_score_cand = score_panel_boundary_alignment(image, candidate_polygon) if image is not None else 0.0
        edge_score_ratio = edge_score_cand / max(edge_score_orig, 1e-6)

        panel_details.append({
            "p_obj": p,
            "panel_idx": p.get("raw_idx", k),
            "trusted": trusted_mask[k],
            "u_pos": pos,
            "is_partial": False,
            "panel_role": panel_role,
            "is_endpoint": is_endpoint,
            "endpoint_side": endpoint_side,
            "used_for_lattice_fit": used_for_lattice_fit,
            "local_angle_deg": local_angle_deg,
            "endpoint_extrapolate_ratio": endpoint_extrapolate_ratio,
            "candidate_polygon_source": candidate_polygon_source,
            "candidate_polygon": candidate_polygon,
            "original_polygon": orig_poly,
            "iou_with_original": iou_with_original,
            "center_shift_ratio": center_shift_ratio,
            "area_ratio": area_ratio,
            "width_ratio": width_ratio if not is_endpoint else 1.0,
            "height_ratio": height_ratio if not is_endpoint else 1.0,
            "candidate_area": candidate_area,
            "original_area": orig_area,
            "out_of_bounds": touches_border if is_endpoint else out_of_bounds,
            "pass_individual": pass_val,
            "fail_reason": fail_reason,
            "edge_score_original": edge_score_orig,
            "edge_score_candidate": edge_score_cand,
            "edge_score_ratio": edge_score_ratio,
            # logging fields
            "endpoint_yolo_outer_u": endpoint_yolo_outer_u,
            "endpoint_pitch_outer_u": endpoint_pitch_outer_u,
            "endpoint_edge_outer_u": endpoint_edge_outer_u,
            "endpoint_consensus_outer_u": endpoint_consensus_outer_u,
            "endpoint_final_outer_u": endpoint_final_outer_u,
            "endpoint_outer_source": endpoint_outer_source,
            "endpoint_outer_score": endpoint_outer_score,
            "endpoint_length_ratio": endpoint_extrapolate_ratio,
            "endpoint_candidates": endpoint_candidates,
            "endpoint_decision": "use_middle_locked_endpoint" if (pass_val and is_endpoint) else "fallback_original",
            "endpoint_fallback_reason": fail_reason if not pass_val else "ok",
            "middle_lattice_unchanged": True,
            "angle_delta_to_block": 0.0,
            # locked pitch strategy extra fields
            "endpoint_strategy": "middle_locked_pitch",
            "used_for_middle_fit": used_for_lattice_fit,
            "middle_lattice_quality_good": len(middle_trusted_indices) >= MIN_MIDDLE_PANELS_FOR_LATTICE,
            "inner_u": float(inner_divider_u) if is_endpoint else 0.0,
            "outer_u": float(endpoint_final_outer_u) if is_endpoint else 0.0,
            "pitch": float(pitch_fit),
            "endpoint_pitch_ratio": float(endpoint_extrapolate_ratio) if is_endpoint else 0.0,
            "endpoint_iou_with_yolo": float(iou_with_original) if is_endpoint else 0.0,
            "endpoint_center_shift_ratio": float(center_shift_ratio) if is_endpoint else 0.0,
            "endpoint_overlap_with_neighbor": float(overlap_val) if (is_endpoint and not touches_border) else 0.0,
            "yolo_endpoint_area": float(orig_area) if is_endpoint else 0.0,
            "median_middle_area": float(median_middle_area),
            "is_yolo_endpoint_small": bool(is_yolo_endpoint_small) if is_endpoint else False,
            "local_pitch": float(pitch_fit),
            "rail_low": float(v_low),
            "rail_high": float(v_high),
        })

    # --- Pairwise overlap check among passing candidates ---
    for i in range(len(panel_details)):
        if not panel_details[i]["pass_individual"]:
            continue
        for j in range(i + 1, len(panel_details)):
            if not panel_details[j]["pass_individual"]:
                continue
            pi = np.array(panel_details[i]["candidate_polygon"], dtype=np.float32)
            pj = np.array(panel_details[j]["candidate_polygon"], dtype=np.float32)
            pi_h = cv2.convexHull(pi.reshape(-1, 1, 2)).astype(np.float32)
            pj_h = cv2.convexHull(pj.reshape(-1, 1, 2)).astype(np.float32)
            area_inter, _ = cv2.intersectConvexConvex(pi_h, pj_h)
            if area_inter > 0:
                min_a = min(cv2.contourArea(pi_h), cv2.contourArea(pj_h))
                overlap_r = area_inter / max(min_a, 1e-3)
                if overlap_r > STRING_PANEL_MAX_OVERLAP:
                    if panel_details[i]["iou_with_original"] < panel_details[j]["iou_with_original"]:
                        panel_details[i]["pass_individual"] = False
                        panel_details[i]["fail_reason"] = "overlap_too_high"
                    else:
                        panel_details[j]["pass_individual"] = False
                        panel_details[j]["fail_reason"] = "overlap_too_high"

    # Max overlap among final candidates
    max_overlap = 0.0
    passing = [d for d in panel_details if d["pass_individual"]]
    for i in range(len(passing)):
        for j in range(i + 1, len(passing)):
            pi = np.array(passing[i]["candidate_polygon"], dtype=np.float32)
            pj = np.array(passing[j]["candidate_polygon"], dtype=np.float32)
            pi_h = cv2.convexHull(pi.reshape(-1, 1, 2)).astype(np.float32)
            pj_h = cv2.convexHull(pj.reshape(-1, 1, 2)).astype(np.float32)
            area_inter, _ = cv2.intersectConvexConvex(pi_h, pj_h)
            if area_inter > 0:
                min_a = min(cv2.contourArea(pi_h), cv2.contourArea(pj_h))
                r = area_inter / max(min_a, 1e-3)
                max_overlap = max(max_overlap, r)

    pass_count_middle = sum(1 for d in panel_details if d["pass_individual"] and not d["is_endpoint"])
    pass_count_endpoint = sum(1 for d in panel_details if d["pass_individual"] and d["is_endpoint"])
    n_endpoints = sum(1 for d in panel_details if d["is_endpoint"])
    n_middle = n - n_endpoints
    pass_count = len(passing)
    pass_ratio = pass_count_middle / max(n_middle, 1) if n_middle > 0 else pass_count / n
    ious = [d["iou_with_original"] for d in panel_details]
    median_iou = float(np.median(ious)) if ious else 0.0

    return {
        "success": True,
        "v_low": v_low,
        "v_high": v_high,
        "pitch": pitch_fit,
        "u0": u0,
        "u_axis": u_axis,
        "v_axis": v_axis,
        "origin": origin,
        "local_angle_deg": local_angle_deg,
        "pass_count": pass_count,
        "pass_count_middle": pass_count_middle,
        "pass_count_endpoint": pass_count_endpoint,
        "n_middle": n_middle,
        "n_endpoints": n_endpoints,
        "pass_ratio": pass_ratio,
        "median_iou": median_iou,
        "max_overlap": max_overlap,
        "panel_details": panel_details,
        "rail_fit_source": rail_fit_source,
        "median_middle_area": median_middle_area,
    }
"""

with open(target_path, "r", encoding="utf-8") as f:
    target_lines = f.readlines()

# Find markers in target
start_idx = None
end_idx = None

for idx, line in enumerate(target_lines):
    if "def fit_string_parallel_lines(" in line:
        start_idx = idx
    if "def synchronize_strings(" in line:
        end_idx = idx
        break

if start_idx is None or end_idx is None:
    print(f"Error: Markers not found. start_idx={start_idx}, end_idx={end_idx}")
    exit(1)

print(f"Replacing fit_string_parallel_lines function lines {start_idx+1} to {end_idx}")

new_lines = target_lines[:start_idx] + [new_fit_code + "\n\n"] + target_lines[end_idx:]

temp_path = "app/services/panel_processor_temp.py"
with open(temp_path, "w", encoding="utf-8", newline="\n") as f:
    f.writelines(new_lines)

# Compile check
try:
    py_compile.compile(temp_path, doraise=True)
    print("Compilation check passed!")
    with open(target_path, "w", encoding="utf-8", newline="\n") as f:
        f.writelines(new_lines)
    print("Success! Overwrote panel_processor.py")
    if os.path.exists(temp_path):
        os.remove(temp_path)
except Exception as e:
    print(f"Compilation check failed on temp file: {e}")
    if os.path.exists(temp_path):
        os.remove(temp_path)
