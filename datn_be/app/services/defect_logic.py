# app/services/defect_logic.py
# Chứa các hàm xử lý logic lỗi (defect) trong hệ thống kiểm tra tấm pin PV:
#   - Gán defect vào panel bằng intersection area (không dùng centroid)
#   - Phân loại mức độ hư hỏng (severity) theo area_ratio_percent
#   - Sinh recommendation từ severity
import uuid
from typing import List, Dict, Any, Optional, Tuple
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.validation import make_valid
from app.services.panel_geometry import localize_defect_inside_panel

# ─────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────

# Threshold overlap tối thiểu để gán defect vào panel.
# Dùng 0.3 (30%) để không bỏ sót lỗi ở mép panel.
DEFECT_PANEL_OVERLAP_THRESHOLD = 0.3


# ─────────────────────────────────────────
# 1. GÁN DEFECT VÀO PANEL (OVERLAP AREA)
# ─────────────────────────────────────────

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
            # Không gán được (defect ở ngoài hoặc ở mép <30%)
            unassigned_defects.append(d)
            continue

        # Gán defect vào panel tốt nhất
        best_panel = panels[best_panel_idx]
        panel_area = panel_geoms[best_panel_idx].area
        area_ratio_pct = (best_intersection_area / panel_area * 100) if panel_area > 0 else 0.0

        # Tính vị trí lỗi trong panel
        d_center = d.get("center", [0, 0])
        panel_bbox = best_panel.get("bbox") or best_panel.get("box", [0, 0, 0, 0])
        loc_info = localize_defect_inside_panel(d_center, panel_bbox)

        severity = classify_severity(d.get("class_name", ""), area_ratio_pct)
        recommendation = recommendation_from_severity(severity)

        defect_entry = {
            # Thông tin detection gốc
            "class_name": d.get("class_name", ""),
            "confidence": d.get("confidence", 0.0),
            "bbox": d.get("bbox") or d.get("box", []),
            "polygon": d.get("polygon", []),
            "area": round(d.get("area", 0.0), 2),
            "center": d.get("center", [0, 0]),
            # Thông tin assignment
            "assigned_panel_id": best_panel.get("id", ""),
            "overlap_ratio": round(best_overlap_ratio, 4),
            "area_inside_panel": round(best_intersection_area, 2),
            "area_ratio_percent": round(area_ratio_pct, 4),
            # Vị trí trong panel
            "relative_position": {"u": loc_info["u"], "v": loc_info["v"]},
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


def _build_shapely_polygon(points: List) -> Optional[ShapelyPolygon]:
    """Tạo Shapely Polygon từ list điểm. Trả None nếu không hợp lệ."""
    if not points or len(points) < 3:
        return None
    try:
        geom = ShapelyPolygon(points)
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
    - total_defect_area_ratio_percent
    - max_defect_area_ratio_percent
    - worst_severity
    - recommendation
    - status: 'faulty' hoặc 'healthy'
    """
    defects = panel.get("defects", [])
    if not defects:
        panel["status"] = "healthy"
        panel["total_defect_area_ratio_percent"] = 0.0
        panel["max_defect_area_ratio_percent"] = 0.0
        panel["worst_severity"] = "healthy"
        panel["recommendation"] = "No action"
        panel["main_defect_class"] = None
        # Backward compat
        panel["total_panel_loss"] = 0.0
        return

    ratios = [d["area_ratio_percent"] for d in defects]
    total_ratio = sum(ratios)
    max_ratio = max(ratios)

    # Severity ranking
    severity_order = ["very_minor", "minor", "moderate", "severe", "replace"]
    severities = [d["severity"] for d in defects]
    worst = max(severities, key=lambda s: severity_order.index(s) if s in severity_order else -1)

    # Class phổ biến nhất
    from collections import Counter
    class_counts = Counter(d["class_name"] for d in defects)
    main_class = class_counts.most_common(1)[0][0] if class_counts else None

    panel["status"] = "faulty"
    panel["total_defect_area_ratio_percent"] = round(total_ratio, 4)
    panel["max_defect_area_ratio_percent"] = round(max_ratio, 4)
    panel["worst_severity"] = worst
    panel["recommendation"] = recommendation_from_severity(worst)
    panel["main_defect_class"] = main_class
    # Backward compat: giữ total_panel_loss để không break frontend cũ
    panel["total_panel_loss"] = round(total_ratio, 2)


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
        "very_minor": "No action / Monitor",
        "healthy":    "No action",
        "minor":      "Monitor",
        "moderate":   "Inspect",
        "severe":     "Prioritize maintenance",
        "replace":    "Recommend replacement",
    }
    return mapping.get(severity, "Inspect")
