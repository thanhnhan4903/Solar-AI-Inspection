# app/services/defect_thermal_validator.py
#
# CIE L*a*b* Relative Thermal Delta Validation
# Tên tiếng Việt: Kiểm chứng lỗi bằng chênh lệch nhiệt tương đối CIE L*
#
# Phương pháp:
#   relative_thermal_delta = mean(H_defect) - mean(H_panel_background)
#   H(x,y) = L_norm(x,y)
#   L_norm lấy từ kênh L* của không gian màu CIE L*a*b*, chuẩn hóa về 0–1.
#
# LƯU Ý QUAN TRỌNG:
#   Ảnh đầu vào là ảnh màu giả nhiệt (pseudo-color thermal), KHÔNG có metadata
#   radiometric, KHÔNG biết nhiệt độ tuyệt đối (°C).
#   Module này dùng L* CIE Lab làm proxy cho độ sáng/nhiệt tương đối.
#   Đây KHÔNG phải phép đo nhiệt độ bức xạ tuyệt đối.
#
#   THERMAL_DELTA_MIN là ngưỡng nội bộ tạm thời của team,
#   KHÔNG phải tiêu chuẩn quốc tế (IEC, SEMI, ISO).

import os
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("solar_ai")

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════

THERMAL_VALIDATOR_ENABLED = True

# Ngưỡng delta tối thiểu để coi là có tương phản nhiệt thật sự.
# Đây là ngưỡng nội bộ tạm thời — KHÔNG phải tiêu chuẩn quốc tế.
THERMAL_DELTA_MIN = 0.05

# Pixel coverage thresholds
MIN_DEFECT_PIXELS  = 10   # Dưới ngưỡng này → insufficient_pixels → not_run
MIN_PANEL_PIXELS   = 50   # Panel mask quá nhỏ → skip

# Số pixel tối thiểu trong panel để dùng normalization theo panel
MIN_PANEL_PIXELS_FOR_LOCAL_NORM = 50

# Kernel dilate để loại bỏ vùng biên defect khỏi background
DILATE_KERNEL_SIZE = 3

# Debug
SAVE_THERMAL_VALIDATION_DEBUG_JSONL = True


# ═══════════════════════════════════════════════════════════════
# HELPERS — Polygon → mask
# ═══════════════════════════════════════════════════════════════

def _polygon_to_mask(polygon, h: int, w: int) -> np.ndarray:
    """
    Chuyển polygon [[x,y],...] hoặc [(x,y),...] thành binary mask uint8 (0/255).
    Trả về mask rỗng nếu polygon không hợp lệ.
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    if not polygon or len(polygon) < 3:
        return mask
    try:
        pts = []
        for pt in polygon:
            if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                pts.append([int(round(float(pt[0]))), int(round(float(pt[1])))])
        if len(pts) < 3:
            return mask
        pts_arr = np.array(pts, dtype=np.int32)
        cv2.fillPoly(mask, [pts_arr], 255)
    except Exception:
        pass
    return mask


def _bbox_to_mask(bbox, h: int, w: int) -> np.ndarray:
    """Tạo mask từ bbox [x1,y1,x2,y2]."""
    mask = np.zeros((h, w), dtype=np.uint8)
    if not bbox or len(bbox) < 4:
        return mask
    try:
        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox[:4]]
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(w, x2); y2 = min(h, y2)
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 255
    except Exception:
        pass
    return mask


# ═══════════════════════════════════════════════════════════════
# CORE — CIE L*a*b* L_norm computation
# ═══════════════════════════════════════════════════════════════

def _compute_l_norm(image_bgr: np.ndarray, panel_mask: np.ndarray) -> np.ndarray:
    """
    Tính L_norm từ kênh L* của CIE L*a*b*.

    Quy trình:
      1. BGR → CIE L*a*b*  (cv2.COLOR_BGR2LAB)
      2. Lấy kênh L (range 0–255 trong OpenCV)
      3. Chuẩn hóa panel-local bằng percentile 2% và 98% trong panel_mask
         nếu panel đủ lớn và dải đủ rộng.
      4. Fallback: L_norm = L / 255.0

    Returns:
        L_norm: float32 array cùng shape H×W, giá trị trong [0,1].
    """
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0].astype(np.float32)   # 0–255 trong OpenCV

    # Thử chuẩn hóa local theo panel
    panel_pixels_mask = (panel_mask > 0)
    n_panel_px = int(panel_pixels_mask.sum())

    if n_panel_px >= MIN_PANEL_PIXELS_FOR_LOCAL_NORM:
        L_panel = L[panel_pixels_mask]
        p2  = float(np.percentile(L_panel, 2.0))
        p98 = float(np.percentile(L_panel, 98.0))
        denom = p98 - p2

        if denom > 1e-3:
            L_norm = np.clip((L - p2) / denom, 0.0, 1.0)
            return L_norm.astype(np.float32)

    # Fallback
    L_norm = (L / 255.0).astype(np.float32)
    return L_norm


# ═══════════════════════════════════════════════════════════════
# DISPLAY — Làm mượt polygon dùng riêng cho frontend
# ═══════════════════════════════════════════════════════════════




# ═══════════════════════════════════════════════════════════════
# CORE — Tính relative_thermal_delta cho 1 defect
# ═══════════════════════════════════════════════════════════════

def _compute_relative_thermal_delta(
    L_norm: np.ndarray,
    defect_mask: np.ndarray,
    panel_mask: np.ndarray,
    class_name: str,
) -> Tuple[Optional[float], Optional[float], Optional[float], str]:
    """
    Tính relative_thermal_delta = mean(L_norm[defect_mask]) - mean(L_norm[background]).

    Background = panel_mask trừ đi dilated_defect_mask để loại vùng biên.

    Returns:
        (relative_thermal_delta, defect_l_mean, background_l_mean, status_or_error)
    """
    h, w = L_norm.shape

    # Defect mask boolean
    defect_bool = (defect_mask > 0)
    n_defect = int(defect_bool.sum())
    if n_defect < MIN_DEFECT_PIXELS:
        return None, None, None, "not_run"

    # Dilate defect mask để loại biên khi tính background
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (DILATE_KERNEL_SIZE, DILATE_KERNEL_SIZE)
    )
    dilated = cv2.dilate(defect_mask, kernel, iterations=1)
    dilated_bool = (dilated > 0)

    # Background = panel − dilated_defect
    panel_bool = (panel_mask > 0)
    background_bool = panel_bool & (~dilated_bool)
    n_bg = int(background_bool.sum())
    if n_bg < MIN_DEFECT_PIXELS:
        return None, None, None, "not_run"

    bg_pixels = L_norm[background_bool]
    background_mean = float(bg_pixels.mean())

    defect_mean = float(L_norm[defect_bool].mean())
    delta = defect_mean - background_mean

    return round(delta, 6), round(defect_mean, 6), round(background_mean, 6), "ok"


# ═══════════════════════════════════════════════════════════════
# CORE — Phân loại status từ delta và class_name
# ═══════════════════════════════════════════════════════════════

def _classify_thermal_status(delta: float, class_name: str) -> Tuple[str, str]:
    """
    Gán thermal_validation_status và interpretation dựa trên delta và loại lỗi.

    Returns:
        (thermal_validation_status, interpretation)
    """
    cls = (class_name or "").lower()

    is_hotspot = (
        "hotspot" in cls
        or "single" in cls
        or "multi" in cls
        or "hot" in cls
    )
    is_shading = (
        "shad" in cls
        or "shadow" in cls
        or "soil" in cls
        or "dirt" in cls
    )
    is_crack = "crack" in cls or "nut" in cls

    # Interpretation text
    if delta > THERMAL_DELTA_MIN:
        interpretation = "Vùng lỗi sáng/nóng hơn nền panel"
    elif abs(delta) <= THERMAL_DELTA_MIN:
        interpretation = "Tương phản nhiệt tương đối chưa rõ"
    else:
        interpretation = "Vùng lỗi tối/lạnh hơn nền panel"

    # Status theo loại lỗi
    if is_crack:
        # Crack không dùng delta làm bằng chứng chính
        status = "shape_based_detection_review"

    elif is_shading:
        if delta < -THERMAL_DELTA_MIN:
            status = "cooler_than_panel_background_recheck"
        else:
            status = "shading_need_recheck"

    elif is_hotspot:
        if delta > THERMAL_DELTA_MIN:
            status = "hotter_than_panel_background"
        elif abs(delta) <= THERMAL_DELTA_MIN:
            status = "weak_relative_contrast"
        else:
            status = "thermal_mismatch_need_review"

    else:
        # Lớp không xác định → áp dụng logic hotspot mặc định
        if delta > THERMAL_DELTA_MIN:
            status = "hotter_than_panel_background"
        elif abs(delta) <= THERMAL_DELTA_MIN:
            status = "weak_relative_contrast"
        else:
            status = "thermal_mismatch_need_review"

    return status, interpretation


# ═══════════════════════════════════════════════════════════════
# HELPERS — Tag not_run
# ═══════════════════════════════════════════════════════════════

def _tag_defect_not_run(defect: Dict[str, Any], reason: str = "not_run") -> None:
    """Gắn thermal fields mặc định khi không thể validate."""
    defect["relative_thermal_delta"]    = None
    defect["thermal_validation_status"] = "not_run"
    defect["area_ratio_inner"]          = None
    defect["thermal_validation"] = {
        "enabled": True,
        "method": "lab_l_relative_thermal_delta",
        "source": "CIE_Lab_L_channel",
        "relative_thermal_delta": None,
        "defect_l_mean": None,
        "background_l_mean": None,
        "interpretation": f"Không chạy được validation: {reason}",
    }


def _tag_panels_not_run(panels: List[Dict[str, Any]]) -> None:
    """Tag tất cả defects là not_run khi không đọc được ảnh."""
    for panel in panels:
        for defect in panel.get("defects", []):
            _tag_defect_not_run(defect, reason="image_load_failed")


# ═══════════════════════════════════════════════════════════════
# HELPERS — Debug JSONL
# ═══════════════════════════════════════════════════════════════

def _write_debug_jsonl(
    records: List[Dict],
    debug_dir: Optional[str],
    stem: str,
) -> None:
    """Ghi debug records ra file JSONL."""
    if not debug_dir:
        debug_dir = "data/results/debug_logs"
    try:
        os.makedirs(debug_dir, exist_ok=True)
        out_path = os.path.join(debug_dir, f"{stem}_defect_thermal_validation.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        logger.info(
            f"[ThermalValidator] Debug JSONL saved: {out_path} ({len(records)} records)"
        )
    except Exception as e:
        logger.warning(f"[ThermalValidator] Không thể ghi debug JSONL: {e}")


# ═══════════════════════════════════════════════════════════════
# MAIN PUBLIC API
# ═══════════════════════════════════════════════════════════════

def validate_defects_by_relative_thermal_contrast(
    image_path: str,
    panels: List[Dict[str, Any]],
    config: Optional[Dict] = None,
    debug_dir: Optional[str] = None,
    image_stem: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Kiểm chứng lỗi YOLO bằng chênh lệch nhiệt tương đối CIE L*.

    Công thức:
        relative_thermal_delta = mean(H_defect) - mean(H_panel_background)
        H(x,y) = L_norm(x,y)     # kênh L* CIE Lab, chuẩn hóa 0–1

    Args:
        image_path:  Đường dẫn ảnh nhiệt BGR (data/precalib/<filename>).
        panels:      List panel dict đã được gán defects.
        config:      Reserved — không dùng trong phiên bản này.
        debug_dir:   Thư mục ghi debug JSONL.
        image_stem:  Tên file gốc không extension.

    Returns:
        panels: Danh sách panel với các field thermal đã được gắn vào từng defect:
            - relative_thermal_delta
            - thermal_validation_status
            - area_ratio_inner
            - thermal_validation (object)

    Notes:
        - Không xóa defect.
        - Không tự set review_status.
        - Không dùng nhiệt độ tuyệt đối.
    """
    if not THERMAL_VALIDATOR_ENABLED:
        return panels

    # ── 1. Đọc ảnh BGR ──
    img_bgr = None
    try:
        img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    except Exception as e:
        logger.warning(f"[ThermalValidator] Không đọc được ảnh {image_path}: {e}")

    if img_bgr is None or img_bgr.size == 0:
        logger.warning(
            f"[ThermalValidator] Ảnh không hợp lệ hoặc không tồn tại: {image_path}. "
            f"Bỏ qua validation."
        )
        _tag_panels_not_run(panels)
        return panels

    h, w = img_bgr.shape[:2]
    debug_records = []
    stem = image_stem or "unknown"

    # ── 2. Xử lý từng panel ──
    for panel in panels:
        local_id = panel.get("local_id", "?")
        defects  = panel.get("defects", [])

        # Tạo panel_mask từ outer_polygon / polygon / bbox
        panel_poly = (
            panel.get("outer_polygon")
            or panel.get("polygon")
        )
        if panel_poly and len(panel_poly) >= 3:
            panel_mask = _polygon_to_mask(panel_poly, h, w)
        else:
            bbox = panel.get("bbox") or panel.get("box", [])
            panel_mask = _bbox_to_mask(bbox, h, w)

        n_panel_px = int((panel_mask > 0).sum())
        if n_panel_px < MIN_PANEL_PIXELS:
            # Panel mask quá nhỏ
            for defect in defects:
                _tag_defect_not_run(defect, reason="panel_mask_too_small")
            continue

        # ── 3. Tính L_norm dựa trên panel_mask ──
        try:
            L_norm = _compute_l_norm(img_bgr, panel_mask)
        except Exception as e:
            logger.debug(
                f"[ThermalValidator] Panel {local_id}: lỗi khi tính L_norm: {e}"
            )
            for defect in defects:
                _tag_defect_not_run(defect, reason=f"l_norm_error:{type(e).__name__}")
            continue

        # ── 4. Validate từng defect ──
        for d_idx, defect in enumerate(defects):
            class_name = defect.get("class_name", "unknown")

            # IMPORTANT — L* is used only as a relative hotness map.
            # It must NOT be used to reshape defect polygons because
            # PV thermal images contain cell/string brightness bands that
            # cause L*-thresholded regions to follow cell/string patterns
            # rather than actual defect boundaries (e.g. hotspot_multi_cell
            # would collapse to a thin horizontal stripe).
            #
            # Defect mask for thermal validation always follows:
            #   analysis_polygon (fallback)
            #   → yolo_polygon (raw YOLO fallback)
            #   → polygon (legacy fallback)
            defect_poly = (
                defect.get("analysis_polygon")
                or defect.get("display_polygon")
                or defect.get("yolo_polygon")
                or defect.get("polygon")
            )
            if defect_poly and len(defect_poly) >= 3:
                defect_mask = _polygon_to_mask(defect_poly, h, w)
            else:
                d_bbox = defect.get("bbox") or defect.get("box", [])
                defect_mask = _bbox_to_mask(d_bbox, h, w)

            # Tính area_ratio_inner
            n_defect_px = int((defect_mask > 0).sum())
            area_ratio_inner = (
                round(n_defect_px / n_panel_px, 6) if n_panel_px > 0 else 0.0
            )

            # Tính delta
            try:
                delta, defect_l_mean, bg_l_mean, run_status = _compute_relative_thermal_delta(
                    L_norm, defect_mask, panel_mask, class_name
                )
            except Exception as e:
                logger.debug(
                    f"[ThermalValidator] Panel {local_id} defect {d_idx} "
                    f"({class_name}): lỗi compute delta: {e}"
                )
                _tag_defect_not_run(defect, reason=f"delta_error:{type(e).__name__}")
                defect["area_ratio_inner"] = round(area_ratio_inner, 6)
                continue

            if run_status == "not_run":
                _tag_defect_not_run(defect, reason="insufficient_pixels")
                defect["area_ratio_inner"] = round(area_ratio_inner, 6)
                continue

            # Phân loại status
            tv_status, interpretation = _classify_thermal_status(delta, class_name)

            # Gắn fields vào defect
            defect["relative_thermal_delta"]    = round(delta, 6)
            defect["thermal_validation_status"] = tv_status
            defect["yolo_area_ratio"]           = round(area_ratio_inner, 6) # Lưu giữ YOLO area ratio
            defect["area_ratio_inner"] = round(area_ratio_inner, 6)

            defect["thermal_validation"] = {
                "enabled":                True,
                "method":                 "lab_l_relative_thermal_delta",
                "source":                 "CIE_Lab_L_channel",
                "relative_thermal_delta": round(delta, 6),
                "defect_l_mean":          round(defect_l_mean, 6),
                "background_l_mean":      round(bg_l_mean, 6),
                "interpretation":         interpretation,
            }

            # Debug record
            if SAVE_THERMAL_VALIDATION_DEBUG_JSONL:
                debug_records.append({
                    "image":                   stem,
                    "local_id":                local_id,
                    "defect_idx":              d_idx,
                    "class_name":              class_name,
                    "thermal_validation_status": tv_status,
                    "relative_thermal_delta":  round(delta, 6),
                    "defect_l_mean":           round(defect_l_mean, 6),
                    "background_l_mean":       round(bg_l_mean, 6),
                    "area_ratio_inner":        round(area_ratio_inner, 6),
                    "interpretation":          interpretation,
                    "n_defect_px":             n_defect_px,
                    "n_panel_px":              n_panel_px,
                })

            logger.debug(
                f"[ThermalValidator] {local_id} d{d_idx} {class_name} "
                f"→ {tv_status}  delta={delta:.4f}  "
                f"defect_L={defect_l_mean:.4f}  bg_L={bg_l_mean:.4f}"
            )

    # ── 5. Ghi debug JSONL ──
    if SAVE_THERMAL_VALIDATION_DEBUG_JSONL and debug_records:
        _write_debug_jsonl(debug_records, debug_dir, stem)

    return panels
