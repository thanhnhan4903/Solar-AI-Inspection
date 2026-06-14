# app/services/defect_thermal_validator.py
#
# Relative Thermal Defect Validation
# Tên tiếng Việt: Kiểm chứng lỗi bằng tương quan nhiệt tương đối
#
# Phương pháp: Relative Spatial-Thermal Contrast Analysis
# (Phân tích Tương phản Không gian - Nhiệt độ Tương đối)
#
# LƯU Ý QUAN TRỌNG:
# Ảnh đầu vào là ảnh màu giả nhiệt (pseudo-color thermal), KHÔNG có metadata
# radiometric, KHÔNG biết nhiệt độ tuyệt đối (°C).
# Module này chỉ dùng chỉ báo tương phản tương đối (relative normalized intensity)
# như một proxy cho bất thường nhiệt, KHÔNG thay thế phép đo nhiệt độ bức xạ tuyệt đối.
#
# Relative normalized intensity is used as a non-radiometric proxy for
# thermal anomaly severity.

import os
import json
import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("solar_ai")

# ═══════════════════════════════════════════════════════════════
# CONFIG — Tất cả ngưỡng gom vào đây, không hardcode rải rác
# ═══════════════════════════════════════════════════════════════

THERMAL_VALIDATOR_ENABLED = True

# Normalization
ROBUST_MIN_PERCENTILE = 2.0
ROBUST_MAX_PERCENTILE = 98.0
USE_PANEL_LOCAL_NORMALIZATION = True  # Normalize từng panel theo chính nó nếu True

# Area ratio definitions (tỷ lệ diện tích defect / diện tích inner panel)
SINGLE_CELL_IDEAL_AREA_RATIO = (1.0 / 100.0, 1.0 / 60.0)  # ~1.0% – 1.67%
SINGLE_CELL_SOFT_AREA_RATIO  = (0.005, 0.030)               # 0.5% – 3.0%

MULTI_CELL_IDEAL_AREA_RATIO  = (0.08, 0.40)
MULTI_CELL_SOFT_AREA_RATIO   = (0.04, 0.55)

CRACK_BRIGHT_AREA_MIN_RATIO    = 0.65   # ≥65% panel sáng → crack candidate
CRACK_BRIGHT_AREA_TARGET_RATIO = 0.80   # Target

SHADING_DARK_AREA_MIN_RATIO    = 0.08   # ≥8% panel tối hơn background
SHADING_FULL_PANEL_DARK_RATIO  = 0.55   # Full-panel shading threshold

# Contrast thresholds (không gian normalized 0..1)
HOT_MEAN_DELTA_MIN     = 0.10   # Minimum hot_delta để gọi là "nóng hơn"
HOT_MEAN_DELTA_STRONG  = 0.20   # Mạnh
DARK_MEAN_DELTA_MIN    = 0.08   # Minimum dark_delta để gọi là "tối hơn"
DARK_MEAN_DELTA_STRONG = 0.16

# Blue chrominance suppression (dùng vì ảnh thermal IronBow: vùng nóng thiếu blue)
BLUE_SUPPRESSION_MIN    = 0.06
BLUE_SUPPRESSION_STRONG = 0.12

# TNI (Thermal Non-uniformity Index) — std của intensity trong vùng
TNI_LOW    = 0.06
TNI_MEDIUM = 0.12
TNI_HIGH   = 0.20

# Pixel coverage thresholds
MIN_DEFECT_PIXELS  = 10   # Dưới ngưỡng này → insufficient_pixels
MIN_PANEL_PIXELS   = 50   # Panel mask quá nhỏ → skip

# Validation score thresholds
VALIDATION_SCORE_CONFIRMED      = 0.60
VALIDATION_SCORE_NEEDS_REVIEW   = 0.38
VALIDATION_SCORE_FALSE_POSITIVE = 0.22

# Debug
SAVE_THERMAL_VALIDATION_DEBUG_JSONL  = True
SAVE_THERMAL_VALIDATION_DEBUG_IMAGE  = False  # Chưa cần ở giai đoạn đầu

# Optional: phát hiện candidate thiếu YOLO (mặc định False để không phá pipeline)
ENABLE_THERMAL_CANDIDATE_SCAN = False


# ═══════════════════════════════════════════════════════════════
# HELPERS — Image I/O & normalization
# ═══════════════════════════════════════════════════════════════

def _robust_normalize(
    arr: np.ndarray,
    p_low: float = ROBUST_MIN_PERCENTILE,
    p_high: float = ROBUST_MAX_PERCENTILE,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Robust min-max normalization về [0, 1] dùng percentile để tránh outlier.
    Nếu mask không None, chỉ lấy percentile từ vùng mask.
    """
    arr = arr.astype(np.float32)
    if mask is not None and mask.any():
        vals = arr[mask > 0]
    else:
        vals = arr.ravel()

    if len(vals) < 2:
        return np.zeros_like(arr)

    lo = float(np.percentile(vals, p_low))
    hi = float(np.percentile(vals, p_high))

    if hi - lo < 1e-6:
        return np.zeros_like(arr)

    out = (arr - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)


def _polygon_to_mask(
    shape: Tuple[int, int],
    polygon: List,
) -> Optional[np.ndarray]:
    """
    Tạo binary mask từ polygon [[x,y], ...].
    Trả None nếu polygon không hợp lệ.
    """
    if not polygon or len(polygon) < 3:
        return None
    h, w = shape[:2]
    pts = np.array([[int(round(x)), int(round(y))] for x, y in polygon], dtype=np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def _bbox_to_mask(
    shape: Tuple[int, int],
    bbox: List,
) -> Optional[np.ndarray]:
    """
    Tạo binary mask từ bbox [x1, y1, x2, y2].
    Trả None nếu bbox không hợp lệ.
    """
    if not bbox or len(bbox) < 4:
        return None
    h, w = shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in bbox[:4]]
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h))
    if x2 <= x1 or y2 <= y1:
        return None
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    return mask


def _safe_mean(arr: np.ndarray, mask: np.ndarray) -> float:
    """Mean của arr trong vùng mask, an toàn với mask rỗng."""
    vals = arr[mask > 0]
    return float(np.mean(vals)) if len(vals) > 0 else 0.0


def _safe_std(arr: np.ndarray, mask: np.ndarray) -> float:
    vals = arr[mask > 0]
    return float(np.std(vals)) if len(vals) > 0 else 0.0


def _safe_percentile(arr: np.ndarray, mask: np.ndarray, q: float) -> float:
    vals = arr[mask > 0]
    return float(np.percentile(vals, q)) if len(vals) > 0 else 0.0


def _mask_pixel_count(mask: np.ndarray) -> int:
    return int(np.count_nonzero(mask))


# ═══════════════════════════════════════════════════════════════
# BUILD IMAGE MAPS
# ═══════════════════════════════════════════════════════════════

def _build_thermal_maps(img_bgr: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Từ ảnh BGR tạo các map cần thiết cho phân tích nhiệt tương đối.
    Tất cả map đều trong range [0, 1] (chưa normalize theo ROI cụ thể).

    Trả về dict:
        gray_norm     — luminance từ grayscale
        value_norm    — HSV Value channel
        lab_l_norm    — LAB L* channel
        b_raw / g_raw / r_raw  — float32 kênh màu [0..255]
        blue_ratio    — b / (r+g+b+eps) per pixel
        hot_score     — weighted combination (proxy cho nhiệt)
        dark_score    — 1 - hot_score (proxy cho vùng lạnh/tối)
        rg_mean       — (r+g)/2, proxy cho yellow-hot trong IronBow/Jet
    """
    h, w = img_bgr.shape[:2]

    b_raw = img_bgr[:, :, 0].astype(np.float32)
    g_raw = img_bgr[:, :, 1].astype(np.float32)
    r_raw = img_bgr[:, :, 2].astype(np.float32)

    # Grayscale luminance
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_norm = _robust_normalize(gray)

    # HSV Value
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2].astype(np.float32)
    value_norm = _robust_normalize(value)

    # LAB L* channel (perceptual lightness)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    lab_l = lab[:, :, 0].astype(np.float32)
    lab_l_norm = _robust_normalize(lab_l)

    # Blue ratio (vùng nóng trong IronBow ít blue hơn)
    rgb_sum = r_raw + g_raw + b_raw + 1e-6
    blue_ratio = b_raw / rgb_sum  # [0..1]

    # Yellow-hot: R+G dominance trừ Blue
    rg_mean = (r_raw + g_raw) / 2.0
    yellow_hot_raw = rg_mean - b_raw
    yellow_hot_norm = _robust_normalize(yellow_hot_raw)

    # Blue suppression: ngược với blue_ratio
    blue_supp_raw = 1.0 - blue_ratio
    blue_supp_norm = _robust_normalize(blue_supp_raw)

    # Brightness: trung bình giữa Lab L* và HSV Value (ổn định hơn từng cái riêng)
    brightness_norm = 0.5 * lab_l_norm + 0.5 * value_norm

    # Hot score tổng hợp
    hot_score = (
        0.45 * brightness_norm
        + 0.35 * yellow_hot_norm
        + 0.20 * blue_supp_norm
    )
    hot_score = np.clip(hot_score, 0.0, 1.0)

    dark_score = 1.0 - hot_score

    return {
        "gray_norm": gray_norm,
        "value_norm": value_norm,
        "lab_l_norm": lab_l_norm,
        "b_raw": b_raw,
        "g_raw": g_raw,
        "r_raw": r_raw,
        "blue_ratio": blue_ratio,
        "rg_mean": rg_mean,
        "yellow_hot_norm": yellow_hot_norm,
        "blue_supp_norm": blue_supp_norm,
        "brightness_norm": brightness_norm,
        "hot_score": hot_score,
        "dark_score": dark_score,
    }


# ═══════════════════════════════════════════════════════════════
# PANEL-LEVEL FEATURES
# ═══════════════════════════════════════════════════════════════

def _compute_panel_features(
    maps: Dict[str, np.ndarray],
    panel: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Tính các feature nhiệt cho toàn bộ tấm pin.
    Ưu tiên dùng inner_polygon, fallback outer_polygon, fallback bbox.
    """
    shape = maps["hot_score"].shape

    # Chọn polygon region cho panel
    inner_poly = panel.get("inner_polygon") or panel.get("calc_polygon") or []
    outer_poly = panel.get("outer_polygon") or panel.get("polygon") or []
    bbox = panel.get("bbox") or panel.get("box") or []

    mask = None
    poly_used = "none"

    if inner_poly and len(inner_poly) >= 3:
        mask = _polygon_to_mask(shape, inner_poly)
        poly_used = "inner"
    if mask is None or _mask_pixel_count(mask) < MIN_PANEL_PIXELS:
        if outer_poly and len(outer_poly) >= 3:
            mask = _polygon_to_mask(shape, outer_poly)
            poly_used = "outer"
    if mask is None or _mask_pixel_count(mask) < MIN_PANEL_PIXELS:
        mask = _bbox_to_mask(shape, bbox)
        poly_used = "bbox"
    if mask is None:
        return {"valid": False, "poly_used": "none", "area_px": 0}

    n_px = _mask_pixel_count(mask)
    if n_px < MIN_PANEL_PIXELS:
        return {"valid": False, "poly_used": poly_used, "area_px": n_px}

    hot = maps["hot_score"]
    brightness = maps["brightness_norm"]
    blue_ratio = maps["blue_ratio"]

    panel_mean_hot  = _safe_mean(hot, mask)
    panel_std_hot   = _safe_std(hot, mask)
    panel_p90_hot   = _safe_percentile(hot, mask, 90)
    panel_p95_hot   = _safe_percentile(hot, mask, 95)
    panel_mean_brightness = _safe_mean(brightness, mask)
    panel_mean_blue = _safe_mean(blue_ratio, mask)
    tni = panel_std_hot  # TNI = std của hot_score trong panel

    return {
        "valid": True,
        "poly_used": poly_used,
        "area_px": n_px,
        "panel_mean_hot": panel_mean_hot,
        "panel_std_hot": panel_std_hot,
        "panel_p90_hot": panel_p90_hot,
        "panel_p95_hot": panel_p95_hot,
        "panel_mean_brightness": panel_mean_brightness,
        "panel_mean_blue": panel_mean_blue,
        "panel_tni": tni,
        "mask": mask,   # giữ lại để dùng trong defect features
    }


# ═══════════════════════════════════════════════════════════════
# DEFECT-LEVEL FEATURES
# ═══════════════════════════════════════════════════════════════

def _get_defect_mask(
    shape: Tuple[int, int],
    defect: Dict[str, Any],
    panel_mask: np.ndarray,
) -> Tuple[Optional[np.ndarray], str]:
    """
    Tạo defect mask clip vào panel_mask.
    Ưu tiên polygon, fallback bbox.
    Trả (clipped_mask, source_used).
    """
    d_poly = defect.get("polygon") or []
    d_bbox = defect.get("bbox") or defect.get("box") or []

    raw_mask = None
    src = "none"

    if d_poly and len(d_poly) >= 3:
        raw_mask = _polygon_to_mask(shape, d_poly)
        src = "polygon"
    if raw_mask is None or _mask_pixel_count(raw_mask) < MIN_DEFECT_PIXELS:
        raw_mask = _bbox_to_mask(shape, d_bbox)
        src = "bbox"

    if raw_mask is None:
        return None, "none"

    # Clip vào panel mask
    clipped = cv2.bitwise_and(raw_mask, panel_mask)
    return clipped, src


def _compute_defect_features(
    maps: Dict[str, np.ndarray],
    defect: Dict[str, Any],
    panel_feat: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Tính các feature nhiệt cho vùng defect và so sánh với background panel.
    """
    shape = maps["hot_score"].shape
    panel_mask = panel_feat.get("mask")
    if panel_mask is None:
        return {"valid": False, "reason": "no_panel_mask"}

    defect_mask, mask_src = _get_defect_mask(shape, defect, panel_mask)
    if defect_mask is None:
        return {"valid": False, "reason": "no_defect_mask"}

    n_defect_px = _mask_pixel_count(defect_mask)
    if n_defect_px < MIN_DEFECT_PIXELS:
        return {"valid": False, "reason": "insufficient_pixels", "pixel_count": n_defect_px}

    hot  = maps["hot_score"]
    dark = maps["dark_score"]
    brightness = maps["brightness_norm"]
    blue_ratio = maps["blue_ratio"]

    # Defect features
    d_mean_hot   = _safe_mean(hot, defect_mask)
    d_median_hot = float(np.median(hot[defect_mask > 0]))
    d_max_hot    = float(np.max(hot[defect_mask > 0]))
    d_p90_hot    = _safe_percentile(hot, defect_mask, 90)
    d_mean_bright = _safe_mean(brightness, defect_mask)
    d_mean_blue  = _safe_mean(blue_ratio, defect_mask)
    d_std_hot    = _safe_std(hot, defect_mask)

    # Background: panel inner trừ defect (dilated để tránh border effects)
    kernel = np.ones((5, 5), np.uint8)
    defect_dilated = cv2.dilate(defect_mask, kernel, iterations=2)
    background_mask = cv2.bitwise_and(panel_mask, cv2.bitwise_not(defect_dilated))

    n_bg_px = _mask_pixel_count(background_mask)
    if n_bg_px >= 20:
        bg_mean_hot    = _safe_mean(hot, background_mask)
        bg_mean_bright = _safe_mean(brightness, background_mask)
        bg_mean_blue   = _safe_mean(blue_ratio, background_mask)
    else:
        # Fallback: dùng toàn bộ panel
        bg_mean_hot    = panel_feat.get("panel_mean_hot", 0.5)
        bg_mean_bright = panel_feat.get("panel_mean_brightness", 0.5)
        bg_mean_blue   = panel_feat.get("panel_mean_blue", 0.3)

    hot_delta  = d_mean_hot    - bg_mean_hot      # dương → defect nóng hơn
    dark_delta = bg_mean_bright - d_mean_bright    # dương → defect tối hơn
    blue_drop  = bg_mean_blue  - d_mean_blue       # dương → defect ít blue (tức nóng)
    tni_local  = d_std_hot

    # Area ratio (vs inner/outer panel pixel count)
    panel_area_px = panel_feat.get("area_px", 1)
    area_ratio = n_defect_px / max(panel_area_px, 1)

    # Aspect ratio của defect bbox
    d_bbox = defect.get("bbox") or defect.get("box") or []
    if len(d_bbox) >= 4:
        bw = abs(d_bbox[2] - d_bbox[0])
        bh = abs(d_bbox[3] - d_bbox[1])
        aspect_ratio = bw / max(bh, 1.0)
    else:
        aspect_ratio = 1.0

    # Bright area ratio: tỷ lệ pixel panel sáng hơn ngưỡng (dùng cho crack)
    panel_hot_p90 = panel_feat.get("panel_p90_hot", 0.5)
    bright_px = np.count_nonzero((hot > panel_hot_p90) & (panel_mask > 0))
    bright_area_ratio = bright_px / max(panel_area_px, 1)

    return {
        "valid": True,
        "mask_source": mask_src,
        "pixel_count": n_defect_px,
        "bg_pixel_count": n_bg_px,
        "d_mean_hot": d_mean_hot,
        "d_median_hot": d_median_hot,
        "d_max_hot": d_max_hot,
        "d_p90_hot": d_p90_hot,
        "d_mean_bright": d_mean_bright,
        "d_mean_blue": d_mean_blue,
        "d_std_hot": d_std_hot,
        "bg_mean_hot": bg_mean_hot,
        "bg_mean_bright": bg_mean_bright,
        "bg_mean_blue": bg_mean_blue,
        "hot_delta": hot_delta,
        "dark_delta": dark_delta,
        "blue_drop": blue_drop,
        "tni_local": tni_local,
        "area_ratio": area_ratio,
        "aspect_ratio": aspect_ratio,
        "bright_area_ratio": bright_area_ratio,
    }


# ═══════════════════════════════════════════════════════════════
# RULE-BASED CLASSIFICATION
# ═══════════════════════════════════════════════════════════════

def _classify_hotspot_single_cell(feat: Dict) -> Tuple[float, Dict]:
    """
    Rule cho hotspot_single_cell:
    - Area ratio nhỏ (một cell): SINGLE_CELL_SOFT_AREA_RATIO
    - hot_delta đủ mạnh
    - blue_drop có thể hỗ trợ
    Trả (score, evidence_dict)
    """
    score = 0.0
    ev = {}

    ar = feat["area_ratio"]
    lo, hi = SINGLE_CELL_SOFT_AREA_RATIO
    if lo <= ar <= hi:
        ar_score = 0.30
    elif ar < lo:
        # Quá nhỏ — giảm điểm
        ratio_below = ar / lo if lo > 0 else 0
        ar_score = 0.30 * ratio_below
    else:
        # Quá lớn — penalty
        excess = (ar - hi) / hi if hi > 0 else 0
        ar_score = max(0.0, 0.30 - 0.30 * min(excess, 1.0))
    ev["area_ratio_score"] = round(ar_score, 3)
    ev["area_ratio"] = round(ar, 4)
    score += ar_score

    # Hot delta
    hd = feat["hot_delta"]
    if hd >= HOT_MEAN_DELTA_STRONG:
        hd_score = 0.40
    elif hd >= HOT_MEAN_DELTA_MIN:
        hd_score = 0.40 * (hd - HOT_MEAN_DELTA_MIN) / (HOT_MEAN_DELTA_STRONG - HOT_MEAN_DELTA_MIN + 1e-6)
        hd_score += 0.10
    else:
        hd_score = max(0.0, 0.10 * (hd / HOT_MEAN_DELTA_MIN))
    ev["hot_delta_score"] = round(hd_score, 3)
    ev["hot_delta"] = round(hd, 4)
    score += hd_score

    # Blue drop / blue suppression
    bd = feat["blue_drop"]
    if bd >= BLUE_SUPPRESSION_STRONG:
        bd_score = 0.20
    elif bd >= BLUE_SUPPRESSION_MIN:
        bd_score = 0.20 * (bd - BLUE_SUPPRESSION_MIN) / (BLUE_SUPPRESSION_STRONG - BLUE_SUPPRESSION_MIN + 1e-6)
        bd_score += 0.05
    else:
        bd_score = max(0.0, 0.05 * (bd / BLUE_SUPPRESSION_MIN if BLUE_SUPPRESSION_MIN > 0 else 0))
    ev["blue_drop_score"] = round(bd_score, 3)
    ev["blue_drop"] = round(bd, 4)
    score += bd_score

    # Aspect ratio penalty (hotspot cell thường rectangular, không mảnh)
    asp = feat["aspect_ratio"]
    if 0.3 <= asp <= 3.0:
        ev["aspect_ok"] = True
    else:
        score -= 0.05
        ev["aspect_ok"] = False

    # TNI bonus
    tni = feat["tni_local"]
    if tni >= TNI_MEDIUM:
        score += 0.05
        ev["tni_bonus"] = True

    ev["total_score"] = round(score, 3)
    return float(np.clip(score, 0.0, 1.0)), ev


def _classify_hotspot_multi_cell(feat: Dict) -> Tuple[float, Dict]:
    """
    Rule cho hotspot_multi_cell:
    - Area ratio lớn hơn single (~4%–50%)
    - hot_delta đủ mạnh
    - Có thể có nhiều component nóng liên tục (proxy qua d_p90_hot cao)
    """
    score = 0.0
    ev = {}

    ar = feat["area_ratio"]
    lo, hi = MULTI_CELL_SOFT_AREA_RATIO
    if lo <= ar <= hi:
        # Gần ideal → điểm cao hơn nếu trong ideal range
        lo_i, hi_i = MULTI_CELL_IDEAL_AREA_RATIO
        if lo_i <= ar <= hi_i:
            ar_score = 0.30
        else:
            ar_score = 0.22
    elif ar < lo:
        ratio_below = ar / lo if lo > 0 else 0
        ar_score = 0.30 * ratio_below
    else:
        excess = (ar - hi) / hi if hi > 0 else 0
        ar_score = max(0.0, 0.30 - 0.25 * min(excess, 1.0))
    ev["area_ratio_score"] = round(ar_score, 3)
    ev["area_ratio"] = round(ar, 4)
    score += ar_score

    # Hot delta
    hd = feat["hot_delta"]
    if hd >= HOT_MEAN_DELTA_STRONG:
        hd_score = 0.40
    elif hd >= HOT_MEAN_DELTA_MIN:
        hd_score = 0.25 + 0.15 * (hd - HOT_MEAN_DELTA_MIN) / (HOT_MEAN_DELTA_STRONG - HOT_MEAN_DELTA_MIN + 1e-6)
    else:
        hd_score = max(0.0, 0.12 * (hd / HOT_MEAN_DELTA_MIN if HOT_MEAN_DELTA_MIN > 0 else 0))
    ev["hot_delta_score"] = round(hd_score, 3)
    ev["hot_delta"] = round(hd, 4)
    score += hd_score

    # P90 hot score (multi_cell nên có vùng rất sáng)
    p90 = feat["d_p90_hot"]
    bg_hot = feat["bg_mean_hot"]
    p90_delta = p90 - bg_hot
    if p90_delta >= HOT_MEAN_DELTA_STRONG:
        p90_score = 0.20
    elif p90_delta >= HOT_MEAN_DELTA_MIN:
        p90_score = 0.10
    else:
        p90_score = 0.0
    ev["p90_delta_score"] = round(p90_score, 3)
    score += p90_score

    # Blue drop
    bd = feat["blue_drop"]
    if bd >= BLUE_SUPPRESSION_MIN:
        bd_score = min(0.10, 0.10 * bd / BLUE_SUPPRESSION_STRONG)
        score += bd_score
        ev["blue_drop_score"] = round(bd_score, 3)

    ev["total_score"] = round(score, 3)
    return float(np.clip(score, 0.0, 1.0)), ev


def _classify_crack(feat: Dict, panel_feat: Dict) -> Tuple[float, Dict]:
    """
    Rule cho crack (theo định nghĩa project: phần lớn panel sáng lên ~80%).
    Dùng bright_area_ratio trong panel và TNI panel.
    """
    score = 0.0
    ev = {}

    bar = feat["bright_area_ratio"]
    if bar >= CRACK_BRIGHT_AREA_TARGET_RATIO:
        bar_score = 0.45
    elif bar >= CRACK_BRIGHT_AREA_MIN_RATIO:
        bar_score = 0.30 + 0.15 * (bar - CRACK_BRIGHT_AREA_MIN_RATIO) / (
            CRACK_BRIGHT_AREA_TARGET_RATIO - CRACK_BRIGHT_AREA_MIN_RATIO + 1e-6
        )
    else:
        bar_score = max(0.0, 0.30 * (bar / CRACK_BRIGHT_AREA_MIN_RATIO))
    ev["bright_area_ratio"] = round(bar, 4)
    ev["bright_area_ratio_score"] = round(bar_score, 3)
    score += bar_score

    # Panel TNI (crack gây non-uniformity toàn panel)
    p_tni = panel_feat.get("panel_tni", 0.0)
    if p_tni >= TNI_HIGH:
        tni_score = 0.30
    elif p_tni >= TNI_MEDIUM:
        tni_score = 0.20
    elif p_tni >= TNI_LOW:
        tni_score = 0.10
    else:
        tni_score = 0.0
    ev["panel_tni"] = round(p_tni, 4)
    ev["panel_tni_score"] = round(tni_score, 3)
    score += tni_score

    # Hot delta / brightness
    hd = feat["hot_delta"]
    if hd >= HOT_MEAN_DELTA_MIN:
        score += 0.15
        ev["hot_ok"] = True
    else:
        ev["hot_ok"] = False

    ev["total_score"] = round(score, 3)
    return float(np.clip(score, 0.0, 1.0)), ev


def _classify_shading(feat: Dict) -> Tuple[float, Dict]:
    """
    Rule cho shading/soiling:
    - dark_delta đủ mạnh (vùng defect tối hơn nền)
    - hot_delta KHÔNG cao (nếu hot thì không phải shading)
    - Area có thể nhỏ hoặc lớn
    """
    score = 0.0
    ev = {}

    dd = feat["dark_delta"]
    if dd >= DARK_MEAN_DELTA_STRONG:
        dd_score = 0.45
    elif dd >= DARK_MEAN_DELTA_MIN:
        dd_score = 0.20 + 0.25 * (dd - DARK_MEAN_DELTA_MIN) / (
            DARK_MEAN_DELTA_STRONG - DARK_MEAN_DELTA_MIN + 1e-6
        )
    else:
        dd_score = max(0.0, 0.10 * (dd / DARK_MEAN_DELTA_MIN if DARK_MEAN_DELTA_MIN > 0 else 0))
    ev["dark_delta"] = round(dd, 4)
    ev["dark_delta_score"] = round(dd_score, 3)
    score += dd_score

    # Penalize nếu hot_delta cao (hot và dark cùng lúc là mâu thuẫn)
    hd = feat["hot_delta"]
    if hd >= HOT_MEAN_DELTA_STRONG:
        penalty = -0.20
        ev["hot_penalty"] = True
    elif hd >= HOT_MEAN_DELTA_MIN:
        penalty = -0.10
        ev["hot_penalty_mild"] = True
    else:
        penalty = 0.0
    score += penalty

    # Area size: shading có thể rất lớn
    ar = feat["area_ratio"]
    if ar >= SHADING_FULL_PANEL_DARK_RATIO:
        score += 0.25
        ev["full_panel_shading"] = True
    elif ar >= SHADING_DARK_AREA_MIN_RATIO:
        score += 0.15
        ev["partial_shading"] = True

    # TNI bonus (shading có thể làm panel không đều)
    tni = feat["tni_local"]
    if tni >= TNI_MEDIUM:
        score += 0.05

    ev["total_score"] = round(score, 3)
    return float(np.clip(score, 0.0, 1.0)), ev


def classify_defect_by_relative_rules(
    defect_feat: Dict[str, Any],
    panel_feat: Dict[str, Any],
    yolo_class: str,
) -> Dict[str, Any]:
    """
    Apply class-specific rules và trả về validation result.

    Returns:
        {
            "rule_class": str,
            "validation_status": str,
            "validation_score": float,
            "severity_by_relative_contrast": str,
            "evidence": dict,
            "suggested_review_status": str,
        }
    """
    if not defect_feat.get("valid", False):
        reason = defect_feat.get("reason", "unknown")
        status = "insufficient_pixels" if reason in ("insufficient_pixels", "no_defect_mask") else "validation_error"
        return {
            "rule_class": yolo_class,
            "validation_status": status,
            "validation_score": 0.0,
            "severity_by_relative_contrast": "low",
            "evidence": {"reason": reason},
            "suggested_review_status": "needs_review",
        }

    cls = yolo_class.lower()

    # Chạy tất cả classifiers để lấy score
    single_score, single_ev = _classify_hotspot_single_cell(defect_feat)
    multi_score,  multi_ev  = _classify_hotspot_multi_cell(defect_feat)
    crack_score,  crack_ev  = _classify_crack(defect_feat, panel_feat)
    shade_score,  shade_ev  = _classify_shading(defect_feat)

    # Bảng score theo YOLO class
    class_score_map = {
        "hotspot_single_cell": single_score,
        "hotspot_multi_cell":  multi_score,
        "crack":               crack_score,
        "shading":             shade_score,
        "soiling":             shade_score,  # soiling dùng cùng rule shading
    }

    # Score của class YOLO gốc
    yolo_base = "hotspot_single_cell"
    for key in class_score_map:
        if key in cls:
            yolo_base = key
            break
    yolo_score = class_score_map.get(yolo_base, 0.3)

    # Rule class: chọn class có score cao nhất
    best_rule_class = max(class_score_map, key=lambda k: class_score_map[k])
    best_rule_score = class_score_map[best_rule_class]

    # Xử lý evidence theo class YOLO
    if "single_cell" in cls:
        evidence = single_ev
    elif "multi_cell" in cls:
        evidence = multi_ev
    elif "crack" in cls:
        evidence = crack_ev
    elif "shading" in cls or "soiling" in cls:
        evidence = shade_ev
    else:
        evidence = {}

    evidence["yolo_score"] = round(yolo_score, 3)
    evidence["best_rule_class"] = best_rule_class
    evidence["best_rule_score"] = round(best_rule_score, 3)

    # Class mismatch nếu rule class khác YOLO class
    class_mismatch = (best_rule_class != yolo_base) and (best_rule_score > yolo_score + 0.20)
    evidence["class_mismatch"] = class_mismatch
    rule_class = yolo_base  # Giữ YOLO class làm rule class mặc định, chỉ log mismatch

    # Xác định validation_status theo score của YOLO class
    score = yolo_score
    if score >= VALIDATION_SCORE_CONFIRMED:
        validation_status = "confirmed_by_relative_thermal"
        suggested_review = "confirmed_defect"
    elif score >= VALIDATION_SCORE_NEEDS_REVIEW:
        if class_mismatch:
            validation_status = "class_mismatch"
        else:
            validation_status = "needs_review"
        suggested_review = "needs_review"
    elif score >= VALIDATION_SCORE_FALSE_POSITIVE:
        validation_status = "needs_review"
        suggested_review = "needs_review"
    else:
        validation_status = "suspect_false_positive"
        suggested_review = "needs_review"  # Review thủ công vẫn là quyết định cuối

    # Severity by relative contrast
    hot_delta = defect_feat.get("hot_delta", 0.0)
    dark_delta = defect_feat.get("dark_delta", 0.0)
    tni = defect_feat.get("tni_local", 0.0)

    if "shading" in cls or "soiling" in cls:
        main_delta = dark_delta
    else:
        main_delta = hot_delta

    if tni >= TNI_HIGH or main_delta >= HOT_MEAN_DELTA_STRONG:
        severity = "high"
    elif tni >= TNI_MEDIUM or main_delta >= HOT_MEAN_DELTA_MIN:
        severity = "medium"
    else:
        severity = "low"

    return {
        "rule_class": rule_class,
        "final_class_suggestion": best_rule_class if class_mismatch else yolo_base,
        "validation_status": validation_status,
        "validation_score": round(score, 4),
        "severity_by_relative_contrast": severity,
        "evidence": evidence,
        "suggested_review_status": suggested_review,
    }


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
    Kiểm chứng lỗi YOLO bằng tương quan nhiệt tương đối.

    Args:
        image_path:  Đường dẫn tới ảnh nhiệt (data/precalib/<filename>).
        panels:      List panel dict đã được gán defects (từ assign_defects_to_panels).
        config:      Override config dict (tuỳ chọn).
        debug_dir:   Thư mục để ghi debug JSONL.
        image_stem:  Tên gốc file ảnh (không extension), dùng cho debug file.

    Returns:
        panels: List panel dict đã được gắn thêm thermal_validation fields.

    Notes:
        - Không xóa defect. Không tự set review_status.
        - Thêm: thermal_validation_status, thermal_validation_score,
                 rule_class, final_class_suggestion,
                 relative_hot_delta, relative_dark_delta, blue_suppression,
                 tni, area_ratio_inner, severity_by_relative_contrast,
                 suggested_review_status.
        - Thêm thermal_panel_features vào mỗi panel.
    """
    if not THERMAL_VALIDATOR_ENABLED:
        return panels

    # ── 1. Đọc ảnh ──
    img_bgr = None
    try:
        img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    except Exception as e:
        logger.warning(f"[ThermalValidator] Không đọc được ảnh {image_path}: {e}")

    if img_bgr is None or img_bgr.size == 0:
        logger.warning(
            f"[ThermalValidator] Ảnh không hợp lệ hoặc không tồn tại: {image_path}. "
            f"Bỏ qua validation, giữ nguyên defects."
        )
        _tag_panels_not_run(panels)
        return panels

    # ── 2. Tạo các map ──
    try:
        maps = _build_thermal_maps(img_bgr)
    except Exception as e:
        logger.warning(f"[ThermalValidator] Lỗi khi tạo thermal maps: {e}")
        _tag_panels_not_run(panels)
        return panels

    # ── 3. Tính panel features và baseline ──
    panel_features_list = []
    for panel in panels:
        try:
            pf = _compute_panel_features(maps, panel)
        except Exception as e:
            logger.debug(f"[ThermalValidator] Panel {panel.get('local_id','?')} features error: {e}")
            pf = {"valid": False}
        panel_features_list.append(pf)

    # Tính baseline: median mean_hot của các panel hợp lệ
    valid_means = [
        pf["panel_mean_hot"]
        for pf in panel_features_list
        if pf.get("valid") and "panel_mean_hot" in pf
    ]
    baseline_mean_hot = float(np.median(valid_means)) if valid_means else 0.5

    debug_records = []

    # ── 4. Gắn features và validate từng defect ──
    stem = image_stem or "unknown"

    for panel, pf in zip(panels, panel_features_list):
        local_id = panel.get("local_id", "?")
        defects = panel.get("defects", [])

        # Gắn panel-level thermal features
        if pf.get("valid"):
            panel_mean_shift = pf["panel_mean_hot"] - baseline_mean_hot
            panel["thermal_panel_features"] = {
                "panel_mean_hot": round(pf["panel_mean_hot"], 4),
                "panel_std_hot": round(pf["panel_std_hot"], 4),
                "panel_tni": round(pf["panel_tni"], 4),
                "panel_mean_shift_from_baseline": round(panel_mean_shift, 4),
                "panel_p90_hot": round(pf["panel_p90_hot"], 4),
                "panel_p95_hot": round(pf["panel_p95_hot"], 4),
                "panel_area_px": pf["area_px"],
                "poly_used": pf["poly_used"],
                "baseline_mean_hot": round(baseline_mean_hot, 4),
            }
        else:
            panel["thermal_panel_features"] = {"valid": False}

        # Validate từng defect
        for d_idx, defect in enumerate(defects):
            yolo_class = defect.get("class_name", "unknown")

            if not pf.get("valid"):
                _tag_defect_not_run(defect, reason="panel_features_invalid")
                continue

            try:
                df = _compute_defect_features(maps, defect, pf)
                result = classify_defect_by_relative_rules(df, pf, yolo_class)
            except Exception as e:
                logger.debug(
                    f"[ThermalValidator] Panel {local_id} defect {d_idx} ({yolo_class}) error: {e}"
                )
                _tag_defect_not_run(defect, reason=f"exception: {type(e).__name__}")
                continue

            # Gắn vào defect (không phá field cũ)
            defect["thermal_validation_status"] = result["validation_status"]
            defect["thermal_validation_score"]  = result["validation_score"]
            defect["rule_class"]                = result["rule_class"]
            defect["final_class_suggestion"]    = result["final_class_suggestion"]
            defect["relative_hot_delta"]        = round(df.get("hot_delta", 0.0), 4)
            defect["relative_dark_delta"]       = round(df.get("dark_delta", 0.0), 4)
            defect["blue_suppression"]          = round(df.get("blue_drop", 0.0), 4)
            defect["tni"]                       = round(df.get("tni_local", 0.0), 4)
            defect["area_ratio_inner"]          = round(df.get("area_ratio", 0.0), 4)
            defect["severity_by_relative_contrast"] = result["severity_by_relative_contrast"]
            defect["suggested_review_status"]   = result["suggested_review_status"]
            defect["thermal_validation"] = {
                "enabled": True,
                "method": "relative_spatial_thermal_contrast",
                "yolo_class": yolo_class,
                "rule_class": result["rule_class"],
                "final_class_suggestion": result["final_class_suggestion"],
                "validation_status": result["validation_status"],
                "validation_score": result["validation_score"],
                "severity_by_tni": result["severity_by_relative_contrast"],
                "evidence": result["evidence"],
            }

            # Debug record
            if SAVE_THERMAL_VALIDATION_DEBUG_JSONL:
                debug_records.append({
                    "image": f"{stem}",
                    "local_id": local_id,
                    "defect_idx": d_idx,
                    "yolo_class": yolo_class,
                    "rule_class": result["rule_class"],
                    "final_class_suggestion": result.get("final_class_suggestion", yolo_class),
                    "validation_status": result["validation_status"],
                    "score": result["validation_score"],
                    "hot_delta": round(df.get("hot_delta", 0.0), 4),
                    "dark_delta": round(df.get("dark_delta", 0.0), 4),
                    "blue_suppression": round(df.get("blue_drop", 0.0), 4),
                    "area_ratio_inner": round(df.get("area_ratio", 0.0), 4),
                    "tni": round(df.get("tni_local", 0.0), 4),
                    "severity": result["severity_by_relative_contrast"],
                    "suggested_review": result["suggested_review_status"],
                    "bright_area_ratio": round(df.get("bright_area_ratio", 0.0), 4),
                    "pixel_count": df.get("pixel_count", 0),
                    "mask_source": df.get("mask_source", "none"),
                    "panel_mean_shift": round(
                        panel.get("thermal_panel_features", {}).get("panel_mean_shift_from_baseline", 0.0), 4
                    ),
                })

            logger.debug(
                f"[ThermalValidator] {local_id} d{d_idx} {yolo_class} "
                f"→ {result['validation_status']} score={result['validation_score']:.3f} "
                f"hot_delta={df.get('hot_delta', 0.0):.3f} ar={df.get('area_ratio', 0.0):.4f}"
            )

    # ── 5. Ghi debug JSONL ──
    if SAVE_THERMAL_VALIDATION_DEBUG_JSONL and debug_records:
        _write_debug_jsonl(debug_records, debug_dir, stem)

    return panels


# ═══════════════════════════════════════════════════════════════
# HELPERS — Tagging & Debug
# ═══════════════════════════════════════════════════════════════

def _tag_defect_not_run(defect: Dict[str, Any], reason: str = "not_run") -> None:
    """Gắn thermal_validation_status = not_run cho defect khi không thể validate."""
    defect["thermal_validation_status"]  = "not_run"
    defect["thermal_validation_score"]   = None
    defect["rule_class"]                 = defect.get("class_name", "unknown")
    defect["final_class_suggestion"]     = defect.get("class_name", "unknown")
    defect["relative_hot_delta"]         = None
    defect["relative_dark_delta"]        = None
    defect["blue_suppression"]           = None
    defect["tni"]                        = None
    defect["area_ratio_inner"]           = None
    defect["severity_by_relative_contrast"] = None
    defect["suggested_review_status"]    = "needs_review"
    defect["thermal_validation"]         = {
        "enabled": True,
        "method": "relative_spatial_thermal_contrast",
        "validation_status": "not_run",
        "reason": reason,
    }


def _tag_panels_not_run(panels: List[Dict[str, Any]]) -> None:
    """Tag tất cả defects là not_run khi validator không chạy được."""
    for panel in panels:
        panel["thermal_panel_features"] = {"valid": False, "reason": "image_load_failed"}
        for defect in panel.get("defects", []):
            _tag_defect_not_run(defect, reason="image_load_failed")


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
        logger.info(f"[ThermalValidator] Debug JSONL saved: {out_path} ({len(records)} records)")
    except Exception as e:
        logger.warning(f"[ThermalValidator] Không thể ghi debug JSONL: {e}")
