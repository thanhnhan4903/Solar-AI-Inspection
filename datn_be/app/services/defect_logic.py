# app/services/defect_logic.py
# Chứa các hàm xử lý logic lỗi (defect) trong hệ thống kiểm tra tấm pin PV:
#   - Gán defect vào panel bằng intersection area (không dùng centroid)
#   - Phân loại mức độ hư hỏng (severity) theo area_ratio_percent
#   - Sinh recommendation từ severity
import uuid
import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.validation import make_valid
from app.services.panel_geometry import localize_defect_inside_panel

# ─────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────

DEFECT_PANEL_OVERLAP_THRESHOLD = 0.20
DEFECT_PANEL_BUFFER_PX = 5
DEFECT_PANEL_BUFFER_MAX_DISTANCE_PX = 6
DEFECT_PANEL_BUFFER_MIN_OVERLAP = 0.05

# Bật True để in log chi tiết từng bước assign.
_ASSIGN_DEBUG = False


def _simplify_coords(points: List[List[float]], max_vertices: int = 6) -> List[List[float]]:
    """Rút gọn một danh sách tọa độ đa giác về tối đa max_vertices đỉnh (thường là 6)."""
    if len(points) <= max_vertices:
        return points
        
    pts = np.array(points, dtype=np.float32)
    arc_len = cv2.arcLength(pts, True)
    
    # Tăng dần epsilon để nén đa giác
    for factor in [0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05]:
        epsilon = factor * arc_len
        approx = cv2.approxPolyDP(pts, epsilon, True)
        approx = approx.reshape(-1, 2)
        if len(approx) <= max_vertices:
            if len(approx) >= 3:
                return approx.tolist()
            break
            
    return points


def assign_defects_to_panels(
    panels: List[Dict[str, Any]],
    defects: List[Dict[str, Any]],
    panel_power: float = 600.0,
    image_path: Optional[str] = None
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Gán mỗi defect vào panel theo thứ tự ưu tiên hình học nghiêm ngặt:
    
    1. centroid_inside == True với đúng 1 panel -> Chọn panel đó.
       Reason: centroid_inside_single
    2. centroid_inside == True với nhiều panel -> Chọn panel có overlap_defect_ratio lớn nhất.
       Reason: centroid_inside_multi_overlap
    3. centroid_inside_buffer == True -> Chọn panel buffer gần nhất với:
       - distance_to_panel <= 6
       - overlap_defect_ratio >= 0.05
       Reason: centroid_inside_buffer
    4. overlap_defect_ratio >= DEFECT_PANEL_OVERLAP_THRESHOLD (0.20) -> Chọn panel có overlap_defect_ratio lớn nhất.
       Reason: overlap_threshold_passed
    5. Ngược lại -> UNASSIGNED (Không dùng nearest-panel fallback).
       Reason: none
    """
    import os
    import json

    # Khởi tạo trường defects cho mỗi panel
    for p in panels:
        p["defects"] = []

    # Build shapely geometry cho từng panel
    panel_geoms: List[Optional[ShapelyPolygon]] = []
    for p in panels:
        poly_pts = p.get("polygon", [])
        geom = _build_shapely_polygon(poly_pts)
        panel_geoms.append(geom)

    unassigned_defects: List[Dict[str, Any]] = []

    # On-the-fly dry-run of row-col local_id assignment for logging
    import copy
    from app.services.panel_geometry import assign_row_col_ids
    try:
        temp_panels = copy.deepcopy(panels)
        temp_panels = assign_row_col_ids(temp_panels)
        temp_id_map = {p.get("id"): p.get("local_id") for p in temp_panels if p.get("id")}
    except Exception:
        temp_id_map = {}

    import logging as _log
    logger = _log.getLogger("solar_ai")

    image_name = "unknown_image"
    if image_path:
        image_name = os.path.splitext(os.path.basename(image_path))[0]

    debug_records = []

    for d in defects:
        d_poly_pts = d.get("polygon", [])
        d_geom = _build_shapely_polygon(d_poly_pts)
        d_class = d.get("class_name", "?")
        d_center = d.get("center", [0, 0])
        d_conf = d.get("confidence", 0.0)
        d_bbox = d.get("bbox") or d.get("box", [])
        d_raw_area = d.get("area", 0.0)
        raw_idx = d.get("raw_idx", -1)

        logger.info(
            f"[DEFECT_RAW] idx={raw_idx} class={d_class} conf={d_conf:.4f} "
            f"raw_area={d_raw_area:.1f} bbox={[round(v, 1) for v in d_bbox]}"
        )

        if d_geom is None or d_geom.area == 0:
            unassigned_defects.append(d)
            rec = {
                "image_name": image_name,
                "defect_idx": raw_idx,
                "defect_class": d_class,
                "defect_conf": round(d_conf, 4),
                "defect_polygon": d_poly_pts,
                "defect_bbox": [round(v, 1) for v in d_bbox],
                "defect_area": 0.0,
                "defect_centroid": [0.0, 0.0],
                "candidate_panels": [],
                "selected_panel": "UNASSIGNED",
                "selected_reason": "none",
                "final_overlap_ratio": 0.0,
                "clipped_area": 0.0,
                "area_ratio_percent": 0.0
            }
            debug_records.append(rec)
            continue

        d_area = d_geom.area
        d_centroid = d_geom.centroid

        candidate_panels_log = []
        candidate_metrics = []

        for idx, (p, p_geom) in enumerate(zip(panels, panel_geoms)):
            if p_geom is None or not p_geom.is_valid:
                continue

            # 1. Tính toán diện tích giao cắt và tỷ lệ overlap
            intersection_area = 0.0
            try:
                inter = p_geom.intersection(d_geom)
                if not inter.is_empty:
                    intersection_area = inter.area
            except Exception:
                pass

            overlap_defect_ratio = intersection_area / d_area if d_area > 0 else 0.0
            overlap_panel_ratio = intersection_area / p_geom.area if p_geom.area > 0 else 0.0

            # 2. Kiểm tra centroid nằm trong panel gốc
            centroid_inside = False
            try:
                centroid_inside = p_geom.contains(d_centroid)
            except Exception:
                pass

            # 3. Kiểm tra centroid nằm trong panel buffer (5px)
            centroid_inside_buffer = False
            try:
                buffered = p_geom.buffer(DEFECT_PANEL_BUFFER_PX)
                centroid_inside_buffer = buffered.contains(d_centroid)
            except Exception:
                pass

            # 4. Khoảng cách Euclidean từ centroid tới panel polygon
            distance_to_panel = 9999.0
            try:
                distance_to_panel = float(d_centroid.distance(p_geom))
            except Exception:
                pass

            # Khoảng cách tới tâm panel làm tie-breaker
            panel_center = p.get("center", [0.0, 0.0])
            px_c, py_c = panel_center[0], panel_center[1]
            dist_to_center = float(np.sqrt((d_centroid.x - px_c) ** 2 + (d_centroid.y - py_c) ** 2))

            p_bbox = p.get("bbox") or p.get("box", [0, 0, 0, 0])
            p_local_id = temp_id_map.get(p.get("id"), f"R00_C00")

            # Check buffer rule and reason
            passed_buffer_rule = False
            reject_reason = "none"
            if centroid_inside_buffer:
                distance_ok = distance_to_panel <= DEFECT_PANEL_BUFFER_MAX_DISTANCE_PX
                overlap_ok = overlap_defect_ratio >= DEFECT_PANEL_BUFFER_MIN_OVERLAP
                if distance_ok and overlap_ok:
                    passed_buffer_rule = True
                else:
                    if overlap_defect_ratio == 0.0:
                        reject_reason = "buffer_overlap_zero"
                    elif not overlap_ok:
                        reject_reason = "buffer_overlap_too_low"
                    elif not distance_ok:
                        reject_reason = "buffer_distance_too_far"

            # Lưu log ứng viên nếu có tương tác bất kỳ (overlap hoặc centroid)
            if intersection_area > 0 or centroid_inside or centroid_inside_buffer:
                candidate_panels_log.append({
                    "local_id": p_local_id,
                    "panel_polygon": p.get("polygon", []),
                    "panel_bbox": [round(v, 1) for v in p_bbox],
                    "panel_area": round(p_geom.area, 1),
                    "panel_center": [round(v, 2) for v in panel_center],
                    "intersection_area": round(intersection_area, 2),
                    "overlap_defect_ratio": round(overlap_defect_ratio, 5),
                    "overlap_panel_ratio": round(overlap_panel_ratio, 5),
                    "centroid_inside": bool(centroid_inside),
                    "centroid_inside_buffer": bool(centroid_inside_buffer),
                    "distance_to_panel": round(distance_to_panel, 2),
                    "buffer_min_overlap": DEFECT_PANEL_BUFFER_MIN_OVERLAP,
                    "passed_buffer_rule": bool(passed_buffer_rule),
                    "reject_reason": reject_reason
                })

            # Lưu metrics để tính toán ưu tiên
            candidate_metrics.append({
                "index": idx,
                "local_id": p_local_id,
                "centroid_inside": centroid_inside,
                "centroid_inside_buffer": centroid_inside_buffer,
                "distance_to_panel": distance_to_panel,
                "overlap_defect_ratio": overlap_defect_ratio,
                "intersection_area": intersection_area,
                "panel_center_dist": dist_to_center
            })

        best_panel_idx = -1
        best_overlap_ratio = 0.0
        best_intersection_area = 0.0
        selected_reason = "none"
        selected_panel_id = "UNASSIGNED"

        # ── THỰC THI THỨ TỰ ƯU TIÊN ──

        # Priority 1 & 2: Centroid inside panel gốc
        insiders = [c for c in candidate_metrics if c["centroid_inside"]]
        if insiders:
            if len(insiders) == 1:
                best_cand = insiders[0]
                selected_reason = "centroid_inside_single"
            else:
                # Nhiều panel chứa centroid: Chọn cái có overlap tốt nhất, tie-breaker là khoảng cách tới tâm panel
                best_cand = max(insiders, key=lambda c: (c["overlap_defect_ratio"], -c["panel_center_dist"]))
                selected_reason = "centroid_inside_multi_overlap"
            
            best_panel_idx = best_cand["index"]
            best_overlap_ratio = best_cand["overlap_defect_ratio"]
            best_intersection_area = best_cand["intersection_area"]
            selected_panel_id = best_cand["local_id"]

        else:
            # Priority 3: Centroid inside buffer (5px) và distance_to_panel <= 6px và overlap >= 0.05
            buffer_candidates = []
            for c in candidate_metrics:
                if c["centroid_inside_buffer"]:
                    distance_ok = c["distance_to_panel"] <= DEFECT_PANEL_BUFFER_MAX_DISTANCE_PX
                    overlap_ok = c["overlap_defect_ratio"] >= DEFECT_PANEL_BUFFER_MIN_OVERLAP
                    if distance_ok and overlap_ok:
                        buffer_candidates.append(c)
            
            if buffer_candidates:
                # Chọn panel buffer có khoảng cách tới defect nhỏ nhất
                best_cand = min(buffer_candidates, key=lambda c: (c["distance_to_panel"], -c["overlap_defect_ratio"]))
                selected_reason = "centroid_inside_buffer"
                
                best_panel_idx = best_cand["index"]
                best_overlap_ratio = best_cand["overlap_defect_ratio"]
                best_intersection_area = best_cand["intersection_area"]
                selected_panel_id = best_cand["local_id"]
            
            else:
                # Priority 4: Overlap ratio >= 0.20
                overlap_candidates = [
                    c for c in candidate_metrics
                    if c["overlap_defect_ratio"] >= DEFECT_PANEL_OVERLAP_THRESHOLD
                ]
                if overlap_candidates:
                    # Chọn panel có tỷ lệ overlap lớn nhất
                    best_cand = max(overlap_candidates, key=lambda c: c["overlap_defect_ratio"])
                    selected_reason = "overlap_threshold_passed"
                    
                    best_panel_idx = best_cand["index"]
                    best_overlap_ratio = best_cand["overlap_defect_ratio"]
                    best_intersection_area = best_cand["intersection_area"]
                    selected_panel_id = best_cand["local_id"]

        # Thực hiện gán defect nếu có panel hợp lệ được chọn
        clipped_poly = d.get("polygon", [])
        actual_intersection_area = best_intersection_area if best_intersection_area > 0 else d_area
        area_ratio_pct = 0.0

        if best_panel_idx != -1:
            best_panel = panels[best_panel_idx]
            p_geom_best = panel_geoms[best_panel_idx]
            panel_area = p_geom_best.area

            # Clip defect polygon bằng panel polygon (chỉ khi đã gán hợp lệ)
            if p_geom_best is not None and d_geom is not None:
                try:
                    inter = p_geom_best.intersection(d_geom)
                    if not inter.is_empty:
                        if inter.geom_type == "Polygon":
                            raw_coords = [list(pt) for pt in inter.exterior.coords]
                            clipped_poly = _simplify_coords(raw_coords, max_vertices=6)
                        elif inter.geom_type == "MultiPolygon":
                            largest_poly = max(inter.geoms, key=lambda g: g.area)
                            raw_coords = [list(pt) for pt in largest_poly.exterior.coords]
                            clipped_poly = _simplify_coords(raw_coords, max_vertices=6)

                        clipped_geom = _build_shapely_polygon(clipped_poly)
                        if clipped_geom is not None:
                            actual_intersection_area = clipped_geom.area
                        else:
                            actual_intersection_area = inter.area
                except Exception:
                    pass

            area_ratio_pct = (actual_intersection_area / panel_area * 100) if panel_area > 0 else 0.0

            # Tính vị trí lỗi trong panel
            panel_bbox = best_panel.get("bbox") or best_panel.get("box", [0, 0, 0, 0])
            panel_poly = best_panel.get("polygon", [])
            loc_info = localize_defect_inside_panel(d_center, panel_bbox, panel_poly)

            severity = classify_severity(d.get("class_name", ""), area_ratio_pct)
            recommendation = recommendation_from_severity(severity)

            defect_entry = {
                "class_name": d.get("class_name", ""),
                "confidence": d.get("confidence", 0.0),
                "bbox": d.get("bbox") or d.get("box", []),
                "polygon": clipped_poly,
                "area": round(d.get("area", 0.0), 2),
                "center": d.get("center", [0, 0]),
                "assigned_panel_id": best_panel.get("id", ""),
                "assign_method": selected_reason,
                "overlap_ratio": round(best_overlap_ratio, 4),
                "area_inside_panel": round(actual_intersection_area, 2),
                "area_ratio_percent": round(area_ratio_pct, 4),
                "relative_position": {
                    "u": loc_info["u"],
                    "v": loc_info["v"],
                    "u_long": loc_info.get("u_long", loc_info["u"]),
                    "v_short": loc_info.get("v_short", loc_info["v"]),
                },
                "location_in_panel": loc_info["location_in_panel"],
                "severity": severity,
                "recommendation": recommendation,
                "type": d.get("class_name", ""),
                "loss": round(area_ratio_pct, 2),
            }
            best_panel["defects"].append(defect_entry)

            logger.info(
                f"[DEFECT_ASSIGN] defect_idx={raw_idx} class={d_class} panel={selected_panel_id} "
                f"overlap={best_overlap_ratio:.4f} clipped_area={actual_intersection_area:.1f} "
                f"area_ratio_percent={area_ratio_pct:.4f} location={loc_info['location_in_panel']}"
            )
        else:
            unassigned_defects.append(d)
            logger.info(
                f"[DEFECT_ASSIGN] defect_idx={raw_idx} class={d_class} panel=UNASSIGNED overlap=0.0 "
                f"clipped_area=0.0 area_ratio_percent=0.0 location=none"
            )

            # Diagnostic log: Tìm 3 panel gần nhất theo khoảng cách centroid
            panel_distances = []
            for idx, p in enumerate(panels):
                panel_center = p.get("center", [0.0, 0.0])
                px_c, py_c = panel_center[0], panel_center[1]
                dist = float(np.sqrt((d_centroid.x - px_c) ** 2 + (d_centroid.y - py_c) ** 2))
                p_local_id = temp_id_map.get(p.get("id"), f"R00_C00")
                p_bbox = p.get("bbox") or p.get("box", [0, 0, 0, 0])
                p_conf = p.get("confidence", 0.0)
                panel_distances.append({
                    "local_id": p_local_id,
                    "bbox": [round(v, 1) for v in p_bbox],
                    "center": [round(v, 2) for v in panel_center],
                    "distance": round(dist, 2),
                    "panel_conf": round(p_conf, 4)
                })
            panel_distances.sort(key=lambda x: x["distance"])
            nearest_panels = panel_distances[:3]
            
            logger.info(
                f"[UNASSIGNED_NEAREST_PANEL] defect_idx={raw_idx} "
                f"defect_centroid={[round(d_centroid.x, 2), round(d_centroid.y, 2)]} "
                f"nearest_panels={nearest_panels}"
            )
            if image_name == "DJI_0843_R":
                print(f"[UNASSIGNED_NEAREST_PANEL]")
                print(f"defect_idx={raw_idx}")
                print(f"defect_centroid={[round(d_centroid.x, 2), round(d_centroid.y, 2)]}")
                print(f"nearest_panels={json.dumps(nearest_panels, ensure_ascii=False)}")

        # Build assignment log record
        rec = {
            "image_name": image_name,
            "defect_idx": raw_idx,
            "defect_class": d_class,
            "defect_conf": round(d_conf, 4),
            "defect_polygon": d_poly_pts,
            "defect_bbox": [round(v, 1) for v in d_bbox],
            "defect_area": round(d_area, 1),
            "defect_centroid": [round(d_centroid.x, 2), round(d_centroid.y, 2)],
            "candidate_panels": candidate_panels_log,
            "selected_panel": selected_panel_id,
            "selected_reason": selected_reason,
            "final_overlap_ratio": round(best_overlap_ratio, 5),
            "clipped_area": round(actual_intersection_area, 2) if best_panel_idx != -1 else 0.0,
            "area_ratio_percent": round(area_ratio_pct, 4) if best_panel_idx != -1 else 0.0
        }
        debug_records.append(rec)

    # ── Write JSONL if image_path is provided ──
    if image_path:
        try:
            debug_dir = "data/results/debug_logs"
            os.makedirs(debug_dir, exist_ok=True)
            out_path = os.path.join(debug_dir, f"{image_name}_defect_assign.jsonl")
            with open(out_path, "w", encoding="utf-8") as f:
                for rec in debug_records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"[DEFECT_DEBUG_FILE] saved {out_path}")
            logger.info(f"[DEFECT_DEBUG_FILE] saved {out_path}")
        except Exception as e:
            logger.warning(f"[DEFECT_DEBUG_FILE] WARN: could not write debug file: {e}")

    # Tổng hợp thống kê cho mỗi panel
    for p in panels:
        _summarize_panel(p, panel_power)

    return panels, unassigned_defects


def recalculate_panel_defects(panels: List[Dict[str, Any]], panel_power: float = 600.0) -> None:
    """
    Tính lại tỷ lệ lỗi và severity cho các tấm pin sau khi đã tinh chỉnh viền (polygon).
    Chỉ chạy trên các tấm có lỗi (status == 'faulty' hoặc có defects).
    """
    for p in panels:
        defects = p.get("defects", [])
        if not defects:
            continue
            
        poly_pts = p.get("polygon", [])
        p_geom = _build_shapely_polygon(poly_pts)
        if p_geom is None or p_geom.area == 0:
            continue
            
        panel_area = p_geom.area
        panel_bbox = p.get("bbox") or p.get("box", [0, 0, 0, 0])
        
        for d in defects:
            d_poly_pts = d.get("polygon", [])
            d_geom = _build_shapely_polygon(d_poly_pts)
            if d_geom is None or d_geom.area == 0:
                continue
                
            try:
                inter = p_geom.intersection(d_geom)
                if not inter.is_empty:
                    inter_area = inter.area
                else:
                    inter_area = d_geom.area
            except Exception:
                inter_area = d_geom.area
                
            # Cập nhật lại các chỉ số
            area_ratio_pct = (inter_area / panel_area * 100) if panel_area > 0 else 0.0
            d["area_inside_panel"] = round(inter_area, 2)
            d["area_ratio_percent"] = round(area_ratio_pct, 4)
            d["loss"] = round(area_ratio_pct, 2)
            
            # Cập nhật lại severity mới
            severity = classify_severity(d.get("class_name", ""), area_ratio_pct)
            recommendation = recommendation_from_severity(severity)
            d["severity"] = severity
            d["recommendation"] = recommendation
            
            # Cập nhật vị trí
            d_center = d.get("center", [0, 0])
            loc_info = localize_defect_inside_panel(d_center, panel_bbox, poly_pts)
            d["relative_position"] = {"u": loc_info["u"], "v": loc_info["v"], "u_long": loc_info.get("u_long", loc_info["u"]), "v_short": loc_info.get("v_short", loc_info["v"])}
            d["location_in_panel"] = loc_info["location_in_panel"]
            
        # Tổng hợp lại panel (severity, area ratio...)
        _summarize_panel(p, panel_power)


def _build_shapely_polygon(points: List) -> Optional[ShapelyPolygon]:
    """Tạo Shapely Polygon từ list điểm. Trả None nếu không hợp lệ."""
    if not points or len(points) < 3:
        return None
    try:
        geom = ShapelyPolygon(points)
        geom = geom.buffer(0)  # Sửa triệt để lỗi TopologyException / Ring edge missing
        if not geom.is_valid:
            geom = make_valid(geom)
        if geom.geom_type not in ("Polygon", "MultiPolygon") or geom.area == 0:
            return None
        return geom
    except Exception:
        return None


def _summarize_panel(panel: Dict[str, Any], PANEL_POWER_W: float = 600.0) -> None:
    """
    Tính tổng hợp cho panel sau khi đã gán defects:
    - Tính sản lượng hao hụt (chia tấm pin làm 3 phần, lỗi mỗi phần trừ 1/3 hiệu suất)
    - Crack trừ 100%
    - Shading cảnh báo 6 giờ
    """
    defects = panel.get("defects", [])

    if not defects:
        panel["status"] = "healthy"
        panel["total_defect_area_ratio_percent"] = 0.0
        panel["max_defect_area_ratio_percent"] = 0.0
        panel["worst_severity"] = "healthy"
        panel["recommendation"] = "Không cần xử lý"
        panel["main_defect_class"] = None
        panel["total_panel_loss"] = 0.0
        panel["power_loss_w"] = 0.0
        return

    # Tính toán tổn thất sản lượng theo yêu cầu (chia 3 tấm pin)
    bbox = panel.get("bbox") or panel.get("box", [0, 0, 0, 0])
    w = max(bbox[2] - bbox[0], 1)
    h = max(bbox[3] - bbox[1], 1)
    is_landscape = w >= h

    affected_parts = set()
    has_crack = False
    has_shading = False

    for d in defects:
        cls_name = d.get("class_name", "")
        if cls_name == "crack":
            has_crack = True
        elif cls_name == "shading":
            has_shading = True
        elif cls_name in ["hotspot_single_cell", "hotspot_multi_cell"]:
            pos = d.get("relative_position", {})
            v = pos.get("v", 0.0)
            part_idx = min(2, int(v * 3))  # Luôn chia 3 phần theo trục dọc (upper, middle, lower)
            affected_parts.add(part_idx)

    if has_crack:
        power_loss_w = PANEL_POWER_W
    else:
        power_loss_w = len(affected_parts) * (PANEL_POWER_W / 3.0)

    shading_warning = "Bị che khuất ~6 giờ" if has_shading else ""

    ratios = [d["area_ratio_percent"] for d in defects]
    total_ratio = sum(ratios)
    max_ratio = max(ratios)

    # Severity ranking
    severity_order = ["very_minor", "minor", "moderate", "severe", "replace"]
    severities = [d["severity"] for d in defects]
    worst = max(severities, key=lambda s: severity_order.index(s) if s in severity_order else -1)

    # Ghi đè severity dựa trên logic lỗi
    if has_crack:
        worst = "replace"
    elif len(affected_parts) > 0:
        worst = "severe"
    if has_shading and worst in ["healthy", "very_minor", "minor"]:
        worst = "moderate"

    recommendation = recommendation_from_severity(worst)
    if shading_warning:
        if recommendation != "Không cần xử lý":
            recommendation = f"{recommendation} | {shading_warning}"
        else:
            recommendation = shading_warning

    # Class phổ biến nhất
    from collections import Counter
    class_counts = Counter(d["class_name"] for d in defects)
    main_class = class_counts.most_common(1)[0][0] if class_counts else None

    panel["status"] = "faulty"
    panel["total_defect_area_ratio_percent"] = round(total_ratio, 4)
    panel["max_defect_area_ratio_percent"] = round(max_ratio, 4)
    panel["worst_severity"] = worst
    panel["recommendation"] = recommendation
    panel["main_defect_class"] = main_class
    # Lưu công suất hao hụt (W) vào total_panel_loss thay vì % diện tích
    panel["total_panel_loss"] = round(power_loss_w, 2)
    panel["power_loss_w"] = round(power_loss_w, 2)


# ─────────────────────────────────────────
# 2. PHÂN LOẠI SEVERITY THEO AREA RATIO
# ─────────────────────────────────────────

def classify_severity(defect_class: str, area_ratio_percent: float) -> str:
    """
    Phân loại mức độ hư hỏng dựa trên tỉ lệ diện tích lỗi so với diện tích panel.

    Không dùng nhiệt độ °C hay ΔT (không có radiometric metadata).
    Không hard-code 33.33% cho multi_cell.

    Ngưỡng:
        < 0.2%   → very_minor (bỏ qua hoặc theo dõi)
        0.2–1%   → minor (theo dõi)
        1–3%     → moderate (kiểm tra trực tiếp)
        3–8%     → severe (ưu tiên bảo trì)
        >= 8%    → replace (khuyến nghị thay thế)

    Lưu ý: hotspot_multi_cell ảnh hưởng nhiều cell nên tăng 1 bậc severity
    nếu đang ở mức minor trở lên. Đây là bù trừ kỹ thuật vì multi_cell
    gây ra bypass bypass diode và ảnh hưởng chuỗi pin. area_ratio vẫn là chỉ số chính.
    """
    if area_ratio_percent < 0.2:
        return "very_minor"
    elif area_ratio_percent < 1.0:
        base = "minor"
    elif area_ratio_percent < 3.0:
        base = "moderate"
    elif area_ratio_percent < 8.0:
        base = "severe"
    else:
        return "replace"

    # Tăng 1 bậc cho hotspot_multi_cell (ảnh hưởng nhiều cell, bypass diode)
    if "multi_cell" in defect_class.lower():
        upgrade = {"minor": "moderate", "moderate": "severe", "severe": "replace"}
        return upgrade.get(base, base)

    return base


# ─────────────────────────────────────────
# 3. RECOMMENDATION TỪ SEVERITY
# ─────────────────────────────────────────

def recommendation_from_severity(severity: str) -> str:
    """
    Sinh text khuyến nghị từ mức độ hư hỏng.
    """
    mapping = {
        "very_minor": "Bỏ qua / Theo dõi",
        "healthy":    "Không cần xử lý",
        "minor":      "Theo dõi",
        "moderate":   "Kiểm tra",
        "severe":     "Ưu tiên bảo trì",
        "replace":    "Khuyến nghị thay thế",
    }
    return mapping.get(severity, "Kiểm tra")
