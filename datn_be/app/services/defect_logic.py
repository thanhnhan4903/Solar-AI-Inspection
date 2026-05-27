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

# Threshold overlap tối thiểu để gán defect vào panel.
# Dùng 0.3 (30%) để không bỏ sót lỗi ở mép panel.
DEFECT_PANEL_OVERLAP_THRESHOLD = 0.001


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
    defects: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Gán mỗi defect vào panel có overlap area lớn nhất.

    Công thức:
        overlap_ratio = intersection_area / defect_area

    Defect được gán vào panel nếu overlap_ratio >= DEFECT_PANEL_OVERLAP_THRESHOLD.
    Nếu không gán được (false positive hoặc nằm ngoài panel) → unassigned_defects.

    Args:
        panels: List panel detections (đã có polygon, bbox, area)
        defects: List defect detections (đã có polygon, bbox, area)

    Returns:
        (panels_with_defects, unassigned_defects)
        - panels_with_defects: mỗi panel có thêm field 'defects', 'status',
          'total_defect_area_ratio_percent', 'max_defect_area_ratio_percent',
          'worst_severity', 'recommendation'
        - unassigned_defects: list defect không gán được vào panel nào
    """
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

    for d in defects:
        d_poly_pts = d.get("polygon", [])
        d_geom = _build_shapely_polygon(d_poly_pts)

        if d_geom is None or d_geom.area == 0:
            unassigned_defects.append(d)
            continue

        d_area = d_geom.area

        # Tìm panel có overlap_ratio lớn nhất
        best_panel_idx = -1
        best_overlap_ratio = 0.0
        best_intersection_area = 0.0

        for idx, (p, p_geom) in enumerate(zip(panels, panel_geoms)):
            if p_geom is None or not p_geom.is_valid:
                continue
            try:
                inter = p_geom.intersection(d_geom)
                inter_area = inter.area
            except Exception:
                continue

            if inter_area <= 0:
                continue

            overlap_ratio = inter_area / d_area
            if overlap_ratio > best_overlap_ratio:
                best_overlap_ratio = overlap_ratio
                best_panel_idx = idx
                best_intersection_area = inter_area

        if best_panel_idx == -1 or best_overlap_ratio < DEFECT_PANEL_OVERLAP_THRESHOLD:
            # Fallback: Nếu không gán được qua overlap (do ở ngoài hoặc mép quá nhỏ),
            # tìm panel có khoảng cách tâm (centroid) gần nhất để gán cưỡng bức (assigned hết đi).
            # Dùng khoảng cách tâm có trọng số aspect ratio = 1.5 (giúp bù đắp đặc tính nằm ngang của tấm pin
            # mà không làm lệch cột khi lỗi nằm ở rìa cột dọc).
            if len(panels) > 0:
                min_dist = float('inf')
                closest_panel_idx = -1
                d_center = d.get("center", [0, 0])
                for idx, p in enumerate(panels):
                    p_center = p.get("center", [0, 0])
                    dx = (d_center[0] - p_center[0]) / 1.5
                    dy = d_center[1] - p_center[1]
                    dist = (dx**2 + dy**2)**0.5
                    if dist < min_dist:
                        min_dist = dist
                        closest_panel_idx = idx

                if closest_panel_idx != -1:
                    best_panel_idx = closest_panel_idx
                    best_overlap_ratio = max(best_overlap_ratio, DEFECT_PANEL_OVERLAP_THRESHOLD)
                    best_intersection_area = d_area * best_overlap_ratio

        if best_panel_idx == -1:
            unassigned_defects.append(d)
            continue

        # Gán defect vào panel tốt nhất
        best_panel = panels[best_panel_idx]
        panel_area = panel_geoms[best_panel_idx].area

        # ── ÉP LỖI NẰM TRONG PANEL ──
        # Tính đa giác giao giữa defect và panel để "ép" lỗi nằm trọn vẹn bên trong tấm pin
        clipped_poly = d.get("polygon", [])
        actual_intersection_area = d_area  # Mặc định dùng toàn bộ diện tích defect nếu không giao nhau (fallback)

        if panel_geoms[best_panel_idx] is not None and d_geom is not None:
            try:
                inter = panel_geoms[best_panel_idx].intersection(d_geom)
                if not inter.is_empty:
                    if inter.geom_type == "Polygon":
                        raw_coords = [list(pt) for pt in inter.exterior.coords]
                        clipped_poly = _simplify_coords(raw_coords, max_vertices=6)
                    elif inter.geom_type == "MultiPolygon":
                        largest_poly = max(inter.geoms, key=lambda p: p.area)
                        raw_coords = [list(pt) for pt in largest_poly.exterior.coords]
                        clipped_poly = _simplify_coords(raw_coords, max_vertices=6)
                    
                    # Tính toán diện tích thực của đa giác sau khi đã ép vào tấm pin
                    clipped_geom = _build_shapely_polygon(clipped_poly)
                    if clipped_geom is not None:
                        actual_intersection_area = clipped_geom.area
                    else:
                        actual_intersection_area = inter.area
            except Exception:
                pass

        # Tính tỷ lệ phần trăm diện tích lỗi so với diện tích tấm pin đã được ép lại
        area_ratio_pct = (actual_intersection_area / panel_area * 100) if panel_area > 0 else 0.0

        # Tính vị trí lỗi trong panel
        d_center = d.get("center", [0, 0])
        panel_bbox = best_panel.get("bbox") or best_panel.get("box", [0, 0, 0, 0])
        panel_poly = best_panel.get("polygon", [])
        loc_info = localize_defect_inside_panel(d_center, panel_bbox, panel_poly)

        severity = classify_severity(d.get("class_name", ""), area_ratio_pct)
        recommendation = recommendation_from_severity(severity)

        defect_entry = {
            # Thông tin detection gốc
            "class_name": d.get("class_name", ""),
            "confidence": d.get("confidence", 0.0),
            "bbox": d.get("bbox") or d.get("box", []),
            "polygon": clipped_poly,
            "area": round(d.get("area", 0.0), 2),
            "center": d.get("center", [0, 0]),
            # Thông tin assignment
            "assigned_panel_id": best_panel.get("id", ""),
            "overlap_ratio": round(best_overlap_ratio, 4),
            "area_inside_panel": round(actual_intersection_area, 2),
            "area_ratio_percent": round(area_ratio_pct, 4),
            # Vị trí trong panel
            "relative_position": {"u": loc_info["u"], "v": loc_info["v"], "u_long": loc_info.get("u_long", loc_info["u"]), "v_short": loc_info.get("v_short", loc_info["v"])},
            "location_in_panel": loc_info["location_in_panel"],
            # Phân loại
            "severity": severity,
            "recommendation": recommendation,
            # Backward compat
            "type": d.get("class_name", ""),
            "loss": round(area_ratio_pct, 2),
        }
        best_panel["defects"].append(defect_entry)

    # Tổng hợp thống kê cho mỗi panel
    for p in panels:
        _summarize_panel(p)

    return panels, unassigned_defects


def recalculate_panel_defects(panels: List[Dict[str, Any]]) -> None:
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
        _summarize_panel(p)


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


def _summarize_panel(panel: Dict[str, Any]) -> None:
    """
    Tính tổng hợp cho panel sau khi đã gán defects:
    - Tính sản lượng hao hụt (chia tấm pin làm 3 phần, lỗi mỗi phần trừ 1/3 hiệu suất)
    - Crack trừ 100%
    - Shading cảnh báo 6 giờ
    """
    defects = panel.get("defects", [])
    PANEL_POWER_W = 600.0

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
