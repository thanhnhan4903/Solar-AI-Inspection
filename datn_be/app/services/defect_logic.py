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
        
    # Tạm thời tắt tính năng nén polygon bằng approxPolyDP để trả về mask gốc của YOLO theo yêu cầu của user
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
        clipped_display_poly = d.get("display_polygon", [])
        clipped_analysis_poly = d.get("analysis_polygon", [])
        
        actual_intersection_area = best_intersection_area if best_intersection_area > 0 else d_area
        area_ratio_pct = 0.0

        if best_panel_idx != -1:
            best_panel = panels[best_panel_idx]
            p_geom_best = panel_geoms[best_panel_idx]
            panel_area = p_geom_best.area

            # Hard Clip defect polygon bằng panel polygon
            clipped_poly = d.get("polygon", [])
            clipped_display_poly = d.get("display_polygon", [])
            clipped_analysis_poly = d.get("analysis_polygon", [])
            
            clipped_to_panel = False
            clip_source = d.get("polygon_source", "yolo_segmentation")
            
            if p_geom_best is not None:
                # Helper function to clip a polygon list against panel
                def _clip_polygon_pts(pts_list, simplify_tol=32):
                    if not pts_list: return []
                    geom = _build_shapely_polygon(pts_list)
                    if geom is None: return pts_list
                    try:
                        inter = p_geom_best.intersection(geom)
                        if inter.is_empty: return []
                        if inter.geom_type == "Polygon":
                            raw = [list(pt) for pt in inter.exterior.coords]
                            return _simplify_coords(raw, max_vertices=simplify_tol)
                        elif inter.geom_type == "MultiPolygon":
                            largest = max(inter.geoms, key=lambda g: g.area)
                            raw = [list(pt) for pt in largest.exterior.coords]
                            return _simplify_coords(raw, max_vertices=simplify_tol)
                    except Exception:
                        pass
                    return pts_list

                # Clip each representation
                clipped_poly = _clip_polygon_pts(d.get("polygon", []), 32)
                clipped_display_poly = _clip_polygon_pts(d.get("display_polygon", []), 32)
                clipped_analysis_poly = _clip_polygon_pts(d.get("analysis_polygon", []), 16)
                
                clipped_geom = _build_shapely_polygon(clipped_analysis_poly)
                if clipped_geom is not None:
                    actual_intersection_area = clipped_geom.area
                else:
                    actual_intersection_area = 0.0
                    
                clipped_to_panel = True
                clip_source = "panel_intersection"
            else:
                clipped_geom = d_geom
                if clipped_geom is not None:
                    actual_intersection_area = clipped_geom.area
                else:
                    actual_intersection_area = 0.0

            # If area becomes too small after clip, skip assigning
            if actual_intersection_area < 5.0 and p_geom_best is not None:
                unassigned_defects.append(d)
                continue
                
            # Recalculate bbox from clipped_poly
            x_coords = [p[0] for p in clipped_poly]
            y_coords = [p[1] for p in clipped_poly]
            if x_coords and y_coords:
                new_bbox = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
            else:
                new_bbox = d.get("bbox") or d.get("box", [])

            area_ratio_pct = (actual_intersection_area / panel_area * 100) if panel_area > 0 else 0.0

            # Tính vị trí lỗi trong panel
            panel_bbox = best_panel.get("bbox") or best_panel.get("box", [0, 0, 0, 0])
            panel_poly = best_panel.get("polygon", [])
            loc_info = localize_defect_inside_panel(
                d_center, panel_bbox, panel_poly,
                defect_polygon=clipped_poly,
                defect_class=d.get("class_name")
            )

            severity = classify_severity(d.get("class_name", ""), area_ratio_pct)
            recommendation = recommendation_from_severity(severity, d.get("class_name", ""), area_ratio_pct)
            
            need_review_geometry = False
            if clip_source == "bbox_fallback" and area_ratio_pct > 40.0 and "single" in d.get("class_name", "").lower():
                need_review_geometry = True

            defect_entry = {
                "class_name": d.get("class_name", ""),
                "confidence": d.get("confidence", 0.0),
                "bbox": [round(v) for v in new_bbox],
                "polygon": clipped_poly,
                "display_polygon": clipped_display_poly,

                "yolo_polygon": d.get("yolo_polygon", []),
                "raw_yolo_polygon": d.get("raw_yolo_polygon", []),
                "analysis_polygon": clipped_analysis_poly,
                "polygon_source": clip_source,
                "display_polygon_source": clip_source,
                "analysis_polygon_source": clip_source,
                "clipped_to_panel": clipped_to_panel,
                "low_geometry_confidence": d.get("low_geometry_confidence", False),
                "need_review_geometry": need_review_geometry,
                "area": round(actual_intersection_area, 2),
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
        p["defects"] = merge_overlapping_panel_defects(p.get("defects", []), p)
        _summarize_panel(p, panel_power)

    return panels, unassigned_defects


def recalculate_panel_defects(panels: List[Dict[str, Any]], panel_power: float = 600.0) -> None:
    """
    Tính lại tỷ lệ lỗi và severity cho các tấm pin sau khi đã tinh chỉnh viền (polygon).
    Chỉ chạy trên các tấm có lỗi (status == 'faulty' hoặc có defects).
    """
    for p in panels:
        p["defects"] = merge_overlapping_panel_defects(p.get("defects", []), p)
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
            recommendation = recommendation_from_severity(severity, d.get("class_name", ""), area_ratio_pct)
            d["severity"] = severity
            d["recommendation"] = recommendation
            
            # Cập nhật vị trí
            d_center = d.get("center", [0, 0])
            loc_info = localize_defect_inside_panel(
                d_center, panel_bbox, poly_pts,
                defect_polygon=d_poly_pts,
                defect_class=d.get("class_name")
            )
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


def defect_to_shapely(d: Dict[str, Any]) -> Optional[ShapelyPolygon]:
    poly_pts = (
        d.get("display_polygon")
        or d.get("polygon")
        or d.get("analysis_polygon")
        or []
    )
    return _build_shapely_polygon(poly_pts)


def defect_class_group(class_name: str) -> str:
    name = str(class_name or "").lower()
    if "hotspot" in name or "hot" in name or "abnormal" in name or "fault" in name:
        return "hotspot"
    if "crack" in name or "nut" in name:
        return "crack"
    if "shading" in name or "shadow" in name or "soil" in name or "dirt" in name:
        return "shading"
    return name


def polygon_merge_metrics(poly_a, poly_b):
    intersection_area = poly_a.intersection(poly_b).area if not poly_a.intersection(poly_b).is_empty else 0.0
    union_area = poly_a.union(poly_b).area if not poly_a.union(poly_b).is_empty else 1.0
    min_area = max(1e-6, min(poly_a.area, poly_b.area))

    iou = intersection_area / max(1e-6, union_area)
    overlap_min_ratio = intersection_area / min_area
    geom_distance = poly_a.distance(poly_b)
    centroid_distance = poly_a.centroid.distance(poly_b.centroid)

    return iou, overlap_min_ratio, geom_distance, centroid_distance


def should_merge(d_a, d_b):
    group_a = defect_class_group(d_a.get("class_name", ""))
    group_b = defect_class_group(d_b.get("class_name", ""))
    
    poly_a = defect_to_shapely(d_a)
    poly_b = defect_to_shapely(d_b)
    
    if poly_a is None or poly_b is None:
        return False
        
    iou, overlap_min_ratio, geom_distance, centroid_distance = polygon_merge_metrics(poly_a, poly_b)
    
    # Yêu cầu 10 — Không làm mất lỗi riêng biệt
    if centroid_distance > 35:
        if overlap_min_ratio < 0.85:
            return False
            
    if overlap_min_ratio < 0.85:
        if geom_distance > 12 or overlap_min_ratio < 0.25 or centroid_distance > 35:
            return False
            
    # Yêu cầu 5 — Không merge quá mạnh với class khác nhóm
    if group_a != group_b:
        if overlap_min_ratio >= 0.90 and centroid_distance <= 15:
            return True
        return False
        
    # Yêu cầu 4 — Merge theo buffer/proximity cho hotspot fragment
    if group_a == "hotspot":
        if overlap_min_ratio >= 0.50:
            return True
        if iou >= 0.30 and centroid_distance <= 25:
            return True
        if geom_distance <= 8 and centroid_distance <= 28:
            return True
        try:
            if poly_a.buffer(5).intersects(poly_b.buffer(5)) and centroid_distance <= 30:
                return True
        except Exception:
            pass
    else:
        if overlap_min_ratio >= 0.65 and centroid_distance <= 20:
            return True
            
    return False


def create_merged_defect(group: List[Dict[str, Any]], panel: Dict[str, Any] = None) -> Dict[str, Any]:
    from shapely.ops import unary_union
    
    best_conf_d = max(group, key=lambda d: d.get("confidence", 0.0))
    class_name = best_conf_d.get("class_name", "")
    
    confidence = max(d.get("confidence", 0.0) for d in group)
    
    severity_ranks = {"level_3_priority": 5, "level_2_inspection": 4, "level_1_monitoring": 3, "recheck_required": 2, "healthy": 1}
    severity = max(group, key=lambda d: severity_ranks.get(d.get("severity", "healthy").lower(), 1)).get("severity", "healthy")
    
    has_confirmed = any(d.get("thermal_validation_status") == "confirmed_by_relative_thermal" for d in group)
    if has_confirmed:
        thermal_validation_status = "confirmed_by_relative_thermal"
    else:
        thermal_validation_status = best_conf_d.get("thermal_validation_status")
        
    relative_thermal_delta = best_conf_d.get("relative_thermal_delta")
    area_ratio_inner = best_conf_d.get("area_ratio_inner")
    thermal_validation = best_conf_d.get("thermal_validation")
    
    geoms_display = []
    geoms_analysis = []
    
    for d in group:
        geom_d = defect_to_shapely(d)
        if geom_d is not None:
            geoms_display.append(geom_d)
            
        ap = d.get("analysis_polygon") or d.get("polygon") or []
        geom_a = _build_shapely_polygon(ap)
        if geom_a is not None:
            geoms_analysis.append(geom_a)
            
    merged_display_poly = []
    merged_analysis_poly = []
    
    group_group = defect_class_group(class_name)
    
    # Merge display polygons using Shapely union
    if geoms_display:
        try:
            union_geom_d = unary_union(geoms_display)
            if union_geom_d is not None and not union_geom_d.is_empty:
                if union_geom_d.geom_type == "Polygon":
                    raw_coords_d = [list(pt) for pt in union_geom_d.exterior.coords]
                    merged_display_poly = _simplify_coords(raw_coords_d, max_vertices=32)
                elif union_geom_d.geom_type == "MultiPolygon":
                    largest_poly_d = max(union_geom_d.geoms, key=lambda g: g.area)
                    raw_coords_d = [list(pt) for pt in largest_poly_d.exterior.coords]
                    merged_display_poly = _simplify_coords(raw_coords_d, max_vertices=32)
        except Exception:
            pass

    if not merged_display_poly:
        merged_display_poly = (
            best_conf_d.get("display_polygon")
            or best_conf_d.get("polygon")
            or []
        )
        
    yolo_polygon = best_conf_d.get("yolo_polygon") or []
    raw_yolo_polygon = best_conf_d.get("raw_yolo_polygon") or []
    display_polygon_source = "merged_union"
        
    if geoms_analysis:
        try:
            union_geom_a = unary_union(geoms_analysis)
                
            if union_geom_a is not None and not union_geom_a.is_empty:
                if union_geom_a.geom_type == "Polygon":
                    raw_coords_a = [list(pt) for pt in union_geom_a.exterior.coords]
                    merged_analysis_poly = _simplify_coords(raw_coords_a, max_vertices=16)
                elif union_geom_a.geom_type == "MultiPolygon":
                    largest_poly_a = max(union_geom_a.geoms, key=lambda g: g.area)
                    raw_coords_a = [list(pt) for pt in largest_poly_a.exterior.coords]
                    merged_analysis_poly = _simplify_coords(raw_coords_a, max_vertices=16)
        except Exception:
            pass
            
    if not merged_analysis_poly:
        merged_analysis_poly = _simplify_coords(merged_display_poly, max_vertices=16)
        
    polygon = merged_display_poly
    
    panel_area = None
    if panel:
        panel_poly = panel.get("polygon", [])
        p_geom = _build_shapely_polygon(panel_poly)
        if p_geom is not None:
            panel_area = p_geom.area
            
    d_geom = _build_shapely_polygon(polygon)
    if d_geom is not None:
        area = d_geom.area
    else:
        area = sum(d.get("area", 0.0) for d in group)
        
    if panel_area and panel_area > 0 and d_geom is not None:
        try:
            inter = p_geom.intersection(d_geom)
            area_inside_panel = inter.area if not inter.is_empty else area
            area_ratio_pct = (area_inside_panel / panel_area * 100)
        except Exception:
            area_inside_panel = area
            area_ratio_pct = max(d.get("area_ratio_percent", 0.0) for d in group)
    else:
        area_inside_panel = area
        area_ratio_pct = max(d.get("area_ratio_percent", 0.0) for d in group)
        
    area_ratio_pct = min(100.0, area_ratio_pct)
    
    severity = classify_severity(class_name, area_ratio_pct)
    recommendation = recommendation_from_severity(severity, class_name, area_ratio_pct)
    
    if d_geom is not None:
        c = d_geom.centroid
        center = [round(c.x, 2), round(c.y, 2)]
    else:
        centers = [d.get("center", [0, 0]) for d in group]
        center = [round(sum(c[0] for c in centers) / len(centers), 2), round(sum(c[1] for c in centers) / len(centers), 2)]
        
    loc_info = {
        "u": best_conf_d.get("relative_position", {}).get("u", 0.5),
        "v": best_conf_d.get("relative_position", {}).get("v", 0.5),
        "u_long": best_conf_d.get("relative_position", {}).get("u_long", 0.5),
        "v_short": best_conf_d.get("relative_position", {}).get("v_short", 0.5),
    }
    location_in_panel = best_conf_d.get("location_in_panel", "middle-center")
    
    if panel:
        panel_bbox = panel.get("bbox") or panel.get("box", [0, 0, 0, 0])
        panel_poly = panel.get("polygon", [])
        try:
            loc_info_new = localize_defect_inside_panel(
                center, panel_bbox, panel_poly,
                defect_polygon=polygon,
                defect_class=class_name
            )
            loc_info = {
                "u": loc_info_new["u"],
                "v": loc_info_new["v"],
                "u_long": loc_info_new.get("u_long", loc_info_new["u"]),
                "v_short": loc_info_new.get("v_short", loc_info_new["v"]),
            }
            location_in_panel = loc_info_new["location_in_panel"]
        except Exception:
            pass

    x_coords = []
    y_coords = []
    for d in group:
        bbox = d.get("bbox") or d.get("box")
        if bbox and len(bbox) == 4:
            x_coords.extend([bbox[0], bbox[2]])
            y_coords.extend([bbox[1], bbox[3]])
    if x_coords and y_coords:
        bbox = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
    else:
        bbox = best_conf_d.get("bbox") or best_conf_d.get("box", [])
        
    merged_defect = {
        "class_name": class_name,
        "confidence": round(confidence, 4),
        "bbox": [round(v) for v in bbox],
        "polygon": polygon,
        "display_polygon": merged_display_poly,

        "yolo_polygon": yolo_polygon,
        "raw_yolo_polygon": raw_yolo_polygon,
        "analysis_polygon": merged_analysis_poly,
        "polygon_source": best_conf_d.get("polygon_source", display_polygon_source),
        "display_polygon_source": display_polygon_source,
        "analysis_polygon_source": best_conf_d.get("analysis_polygon_source", "yolo_segmentation_simplified"),
        "area": round(area, 2),
        "center": center,
        "assigned_panel_id": best_conf_d.get("assigned_panel_id", ""),
        "assign_method": best_conf_d.get("assign_method", ""),
        "overlap_ratio": round(max(d.get("overlap_ratio", 0.0) for d in group), 4),
        "area_inside_panel": round(area_inside_panel, 2),
        "area_ratio_percent": round(area_ratio_pct, 4),
        "relative_position": loc_info,
        "location_in_panel": location_in_panel,
        "severity": severity,
        "recommendation": recommendation,
        "type": class_name,
        "loss": round(area_ratio_pct, 2),
        "merged_from_count": len(group),
        "merge_reason": "hotspot_fragment_proximity" if group_group == "hotspot" else "overlap_duplicate",
    }
    
    if best_conf_d.get("thermal_validation_status") is not None:
        merged_defect["thermal_validation_status"] = thermal_validation_status
    if relative_thermal_delta is not None:
        merged_defect["relative_thermal_delta"] = relative_thermal_delta
    if area_ratio_inner is not None:
        merged_defect["area_ratio_inner"] = area_ratio_inner
    if thermal_validation is not None:
        merged_defect["thermal_validation"] = thermal_validation
        
    return merged_defect


def merge_overlapping_panel_defects(panel_defects: List[Dict[str, Any]], panel: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    if not panel_defects:
        return []

    for d in panel_defects:
        if "duplicate_removed" not in d:
            d["duplicate_removed"] = False
        if "excluded_from_area" not in d:
            d["excluded_from_area"] = False
        d["overlap_resolved"] = True

    if len(panel_defects) < 2:
        return panel_defects
        
    def merge_one_pass(defects_list):
        n = len(defects_list)
        if n < 2:
            return defects_list
        parent = list(range(n))
        
        def find(i):
            if parent[i] == i:
                return i
            parent[i] = find(parent[i])
            return parent[i]
            
        def union(i, j):
            root_i = find(i)
            root_j = find(j)
            if root_i != root_j:
                parent[root_i] = root_j
                
        for i in range(n):
            if defects_list[i].get("duplicate_removed"): continue
            for j in range(i + 1, n):
                if defects_list[j].get("duplicate_removed"): continue
                if should_merge(defects_list[i], defects_list[j]):
                    union(i, j)
                    
        groups = {}
        for i in range(n):
            if defects_list[i].get("duplicate_removed"):
                continue
            root = find(i)
            if root not in groups:
                groups[root] = []
            groups[root].append(i)
            
        merged_list = []
        for root, indices in groups.items():
            if len(indices) == 1:
                merged_list.append(defects_list[indices[0]])
                continue
                
            group = [defects_list[idx] for idx in indices]
            merged = create_merged_defect(group, panel)
            merged["duplicate_removed"] = False
            merged["excluded_from_area"] = False
            merged["overlap_resolved"] = True
            merged_list.append(merged)
            
        return merged_list

    defects = panel_defects
    for _ in range(3):
        new_defects = merge_one_pass(defects)
        if len(new_defects) == len(defects):
            break
        defects = new_defects
        
    # Cross-class overlap resolution
    n = len(defects)
    for i in range(n):
        if defects[i].get("duplicate_removed"):
            continue
        poly_i = defect_to_shapely(defects[i])
        class_i = defects[i].get("class_name", "").lower()
        if not poly_i:
            continue

        for j in range(i + 1, n):
            if defects[j].get("duplicate_removed"):
                continue
            poly_j = defect_to_shapely(defects[j])
            class_j = defects[j].get("class_name", "").lower()
            if not poly_j:
                continue
                
            try:
                inter_area = poly_i.intersection(poly_j).area
            except Exception:
                continue

            if inter_area <= 0:
                continue

            containment_i = inter_area / max(1e-6, poly_i.area)
            containment_j = inter_area / max(1e-6, poly_j.area)

            # 1. hotspot_single_cell vs hotspot_multi_cell
            is_single_i = "single" in class_i
            is_multi_i = "multi" in class_i
            is_single_j = "single" in class_j
            is_multi_j = "multi" in class_j

            if is_single_i and is_multi_j:
                if containment_i > 0.6:
                    defects[i]["duplicate_removed"] = True
            elif is_multi_i and is_single_j:
                if containment_j > 0.6:
                    defects[j]["duplicate_removed"] = True

            # 2. shading vs hotspot/crack
            is_shading_i = "shading" in class_i or "soil" in class_i
            is_shading_j = "shading" in class_j or "soil" in class_j
            is_severe_i = "crack" in class_i or "hot" in class_i
            is_severe_j = "crack" in class_j or "hot" in class_j

            if is_shading_i and is_severe_j and containment_i > 0.2:
                defects[i]["excluded_from_area"] = True
            elif is_shading_j and is_severe_i and containment_j > 0.2:
                defects[j]["excluded_from_area"] = True

    # Geometry non-overlap cleanup
    from shapely.geometry import Polygon
    occupied_internal_geom = Polygon()
    
    # Priority rank: crack (1), hotspot_multi (2), hotspot_single (3), shading (4)
    def get_priority(d):
        c = d.get("class_name", "").lower()
        if "crack" in c: return 1
        if "multi" in c: return 2
        if "single" in c: return 3
        if "shading" in c or "soil" in c: return 4
        return 5
        
    defects.sort(key=lambda d: (get_priority(d), -d.get("confidence", 0.0), -d.get("area", 0.0)))
    
    for d in defects:
        if d.get("duplicate_removed"):
            continue
            
        geom = defect_to_shapely(d)
        if geom is None or geom.is_empty:
            d["duplicate_removed"] = True
            continue
            
        is_shading = get_priority(d) == 4
        
        try:
            clean_geom = geom.difference(occupied_internal_geom)
        except Exception:
            clean_geom = geom
            
        if clean_geom.is_empty or clean_geom.area < 5.0:
            d["duplicate_removed"] = True
            d["excluded_from_area"] = True
            continue
            
        # Update polygons
        if clean_geom.geom_type == "Polygon":
            raw = [list(pt) for pt in clean_geom.exterior.coords]
            clean_pts = _simplify_coords(raw, max_vertices=32)
        elif clean_geom.geom_type == "MultiPolygon":
            largest = max(clean_geom.geoms, key=lambda g: g.area)
            raw = [list(pt) for pt in largest.exterior.coords]
            clean_pts = _simplify_coords(raw, max_vertices=32)
        else:
            clean_pts = d.get("polygon", [])
            
        if clean_pts:
            d["polygon"] = clean_pts
            d["display_polygon"] = clean_pts
            d["analysis_polygon"] = clean_pts
            
            x_coords = [p[0] for p in clean_pts]
            y_coords = [p[1] for p in clean_pts]
            if x_coords and y_coords:
                d["bbox"] = [round(min(x_coords)), round(min(y_coords)), round(max(x_coords)), round(max(y_coords))]
                
        if not is_shading:
            try:
                occupied_internal_geom = occupied_internal_geom.union(clean_geom)
            except Exception:
                pass

    return defects


def get_substring_zone(defect: Dict[str, Any]) -> Optional[int]:
    pos = defect.get("relative_position", {})
    v = pos.get("v")
    if v is not None:
        try:
            v_val = float(v)
            if v_val < 1/3: return 0
            elif v_val < 2/3: return 1
            else: return 2
        except (ValueError, TypeError):
            pass
    loc = defect.get("location_in_panel", "")
    if "upper" in loc.lower() or "top" in loc.lower(): return 0
    if "mid" in loc.lower() or "center" in loc.lower(): return 1
    if "lower" in loc.lower() or "bottom" in loc.lower(): return 2
    return None

def check_panel_geometry_integrity(panel: Dict[str, Any]) -> None:
    panel_poly = panel.get("polygon", [])
    p_geom = _build_shapely_polygon(panel_poly)
    if not p_geom:
        return
        
    defects = [d for d in panel.get("defects", []) if not d.get("duplicate_removed") and not d.get("excluded_from_area")]
    n = len(defects)
    
    # Check outside
    for d in defects:
        d_geom = _build_shapely_polygon(d.get("display_polygon", []))
        if not d_geom:
            continue
        try:
            diff = d_geom.difference(p_geom)
            if diff.area > 5.0:
                logger.warning(f"[GEOMETRY_WARNING] Panel {panel.get('id')} - Defect {d.get('class_name')} extends outside panel by {diff.area:.1f} px^2")
        except Exception:
            pass
            
    # Check internal overlap
    for i in range(n):
        c_i = defects[i].get("class_name", "")
        if "shading" in c_i or "soil" in c_i:
            continue
        g_i = _build_shapely_polygon(defects[i].get("display_polygon", []))
        if not g_i: continue
        
        for j in range(i+1, n):
            c_j = defects[j].get("class_name", "")
            if "shading" in c_j or "soil" in c_j:
                continue
            g_j = _build_shapely_polygon(defects[j].get("display_polygon", []))
            if not g_j: continue
            
            try:
                inter = g_i.intersection(g_j)
                if inter.area > 5.0:
                    logger.warning(f"[GEOMETRY_WARNING] Panel {panel.get('id')} - Internal defects {c_i} and {c_j} overlap by {inter.area:.1f} px^2")
            except Exception:
                pass

def _summarize_panel(panel: Dict[str, Any], PANEL_POWER_W: float = 600.0) -> None:
    # Filter out duplicate_removed defects
    defects = [d for d in panel.get("defects", []) if not d.get("duplicate_removed")]
    panel["defects"] = defects  # Replace the original list

    if not defects:
        panel["status"] = "healthy"
        panel["total_defect_area_ratio_percent"] = 0.0
        panel["max_defect_area_ratio_percent"] = 0.0
        panel["worst_severity"] = "healthy"
        panel["recommendation"] = "Không cần xử lý."
        panel["main_defect_class"] = None
        panel["total_panel_loss"] = 0.0
        panel["power_loss_w"] = 0.0
        return

    from shapely.ops import unary_union

    affected_parts = set()
    has_crack = False
    has_shading = False
    single_cells = []
    
    valid_geoms_for_area = []

    for d in defects:
        cls_name = d.get("class_name", "")
        if cls_name == "crack":
            has_crack = True
        elif "shading" in cls_name or "soil" in cls_name:
            has_shading = True
        elif cls_name == "hotspot_single_cell":
            single_cells.append(d)
            
        if cls_name in ["hotspot_single_cell", "hotspot_multi_cell"]:
            z = get_substring_zone(d)
            if z is not None:
                affected_parts.add(z)
            else:
                affected_parts.add(f"unknown_{len(affected_parts)}")
                
        # Only add to area union if not excluded
        if not d.get("excluded_from_area"):
            geom = defect_to_shapely(d)
            if geom is not None and not geom.is_empty:
                valid_geoms_for_area.append(geom)

    if len(single_cells) >= 2:
        zones = set()
        zone_counts = {}
        for d in single_cells:
            z = get_substring_zone(d)
            if z is not None:
                zones.add(z)
                zone_counts[z] = zone_counts.get(z, 0) + 1
        
        if len(zones) >= 2:
            for d in single_cells:
                d["severity"] = "level_3_priority"
                d["recommendation"] = recommendation_from_severity("level_3_priority", "hotspot_single_cell", d.get("area_ratio_percent", 0))
        elif any(c >= 2 for c in zone_counts.values()):
            for d in single_cells:
                if d.get("severity") == "level_1_monitoring":
                    d["severity"] = "level_2_inspection"
                    d["recommendation"] = recommendation_from_severity("level_2_inspection", "hotspot_single_cell", d.get("area_ratio_percent", 0))

    if has_crack:
        power_loss_w = PANEL_POWER_W
    elif has_shading and not affected_parts:
        power_loss_w = 0.0
    else:
        num_affected = min(3, len(affected_parts))
        power_loss_w = num_affected * (PANEL_POWER_W / 3.0)
        
    if power_loss_w > PANEL_POWER_W: power_loss_w = PANEL_POWER_W
    if power_loss_w < 0: power_loss_w = 0.0

    severity_order = ["healthy", "recheck_required", "level_1_monitoring", "level_2_inspection", "level_3_priority"]
    severities = [d.get("severity", "healthy") for d in defects]
    worst = "healthy"
    for s in severities:
        if s in severity_order:
            if severity_order.index(s) > severity_order.index(worst):
                worst = s
                
    recommendation = recommendation_from_severity(worst)

    from collections import Counter
    class_counts = Counter(d.get("class_name", "unknown") for d in defects)
    main_class = class_counts.most_common(1)[0][0] if class_counts else None

    # Calculate union area
    panel_area = 1.0
    panel_geom = _build_shapely_polygon(panel.get("polygon", []))
    if panel_geom is not None and panel_geom.area > 0:
        panel_area = panel_geom.area
        
    total_area_ratio_percent = 0.0
    if valid_geoms_for_area and panel_geom:
        try:
            union_geom = unary_union(valid_geoms_for_area)
            inter = panel_geom.intersection(union_geom)
            total_area_ratio_percent = (inter.area / panel_area) * 100
        except Exception:
            # Fallback to sum if union fails
            total_area_ratio_percent = sum(d.get("area_ratio_percent", 0.0) for d in defects if not d.get("excluded_from_area"))
    elif not panel_geom:
        total_area_ratio_percent = sum(d.get("area_ratio_percent", 0.0) for d in defects if not d.get("excluded_from_area"))

    total_area_ratio_percent = min(100.0, total_area_ratio_percent)

    # Quy tắc nâng mức hao hụt dựa trên tổng diện tích lỗi
    if total_area_ratio_percent >= 66.66:
        power_loss_w = max(power_loss_w, PANEL_POWER_W)
    elif total_area_ratio_percent >= 33.33:
        power_loss_w = max(power_loss_w, 2 * (PANEL_POWER_W / 3.0))

    panel["status"] = "faulty"
    ratios = [d.get("area_ratio_percent", 0.0) for d in defects]
    panel["total_defect_area_ratio_percent"] = round(total_area_ratio_percent, 4)
    panel["max_defect_area_ratio_percent"] = round(max(ratios), 4) if ratios else 0.0
    panel["worst_severity"] = worst
    panel["recommendation"] = recommendation
    panel["main_defect_class"] = main_class
    panel["total_panel_loss"] = round(power_loss_w, 2)
    panel["power_loss_w"] = round(power_loss_w, 2)
    
    check_panel_geometry_integrity(panel)

# ─────────────────────────────────────────
# 2. PHÂN LOẠI SEVERITY THEO AREA RATIO
# ─────────────────────────────────────────

def classify_severity(defect_class: str, area_ratio_percent: float) -> str:
    cls = defect_class.lower()
    if cls == "crack":
        return "level_3_priority"
    if "shading" in cls:
        return "recheck_required"
    if cls == "hotspot_multi_cell":
        if area_ratio_percent >= 33.33:
            return "level_3_priority"
        return "level_2_inspection"
    if cls == "hotspot_single_cell":
        return "level_1_monitoring"
    return "level_1_monitoring"

# ─────────────────────────────────────────
# 3. RECOMMENDATION TỪ SEVERITY
# ─────────────────────────────────────────

def recommendation_from_severity(severity: str, defect_class: str = "", area_ratio_percent: float = 0.0) -> str:
    cls = defect_class.lower()
    if severity == "level_2_inspection" and cls == "hotspot_multi_cell" and area_ratio_percent < 16.67:
        return "Cần kiểm tra lại mask/class và vùng ảnh hưởng."
        
    if cls == "crack":
        return "Ưu tiên kiểm tra hiện trường, đo bổ sung nếu có và xem xét cách ly hoặc thay thế panel."
    if "shading" in cls:
        return "Cần chụp lại vào thời điểm khác khi không còn bóng che."

    mapping = {
        "healthy": "Không cần xử lý.",
        "level_1_monitoring": "Theo dõi ở lần kiểm tra định kỳ tiếp theo.",
        "level_2_inspection": "Đưa vào danh sách O&M kiểm tra hiện trường; chụp gần hơn hoặc đo bổ sung nếu cần.",
        "level_3_priority": "Ưu tiên kiểm tra và xử lý; đo I–V/EL nếu có; xem xét cách ly hoặc thay thế panel.",
        "recheck_required": "Cần chụp lại vào thời điểm khác để loại trừ ảnh hưởng che bóng."
    }
    return mapping.get(severity, "Không cần xử lý.")
