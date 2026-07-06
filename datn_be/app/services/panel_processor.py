"""
app/services/panel_processor.py
================================
Clean minimal pipeline for solar panel detection.

Pipeline:
  1. Receive YOLO segmentation result.
  2. Split detections into panels and defects.
  3. Refine panel contour safely using YOLO bbox + mask guard.
  4. Small panels use bbox directly when confidence is sufficient.
  5. Simplify defect polygons.
  6. Call assign_defects_to_panels (defect_logic.py).
  7. Return final detection list compatible with frontend schema.

Optional (all False by default):
  - ENABLE_EDGE_GRID_SEGMENTED_REFINEMENT  : edge-grid 4-segment debug prototype.

Legacy: panel_processor_legacy_overgrown.py
"""

import cv2
import logging
import numpy as np
import os
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional

from app.services.panel_geometry import get_polygon_features

logger = logging.getLogger("solar_ai")

# ===========================================================================
# CONFIG FLAGS  (all False by default — safe pipeline)
# ===========================================================================

# ROI-based Canny refinement — experimental, not used in production.
ENABLE_ROI_REFINEMENT: bool = False

# Area-based panel filtering (remove panels far from median area).
ENABLE_PANEL_SIZE_FILTER: bool = False

# Geometry normalization (median width/height/angle normalize).
ENABLE_PANEL_GEOMETRY_NORMALIZATION: bool = False

# Filter small panels that look like defects.
ENABLE_DEFECT_LIKE_PANEL_FILTER: bool = False

# Edge-grid segmented-4 boundary refinement — debug prototype only.
ENABLE_EDGE_GRID_SEGMENTED_REFINEMENT: bool = False


# ===========================================================================
# POLYGON GUARD THRESHOLDS
# ===========================================================================

# --- Strict (normal / large panels) ---
PANEL_POLY_MIN_BBOX_AREA_RATIO: float = 0.70
PANEL_POLY_MIN_BBOX_WIDTH_RATIO: float = 0.75
PANEL_POLY_MIN_BBOX_HEIGHT_RATIO: float = 0.75
PANEL_POLY_MAX_CENTER_SHIFT_RATIO: float = 0.35
PANEL_POLY_MIN_ASPECT_RATIO: float = 1.2
PANEL_POLY_MAX_ASPECT_RATIO: float = 8.0

# --- Relaxed (small panels, bbox_area <= SMALL_PANEL_BBOX_AREA_PX) ---
SMALL_PANEL_BBOX_AREA_PX: float = 2500.0
PANEL_POLY_SMALL_MIN_BBOX_AREA_RATIO: float = 0.30
PANEL_POLY_SMALL_MIN_BBOX_WIDTH_RATIO: float = 0.55
PANEL_POLY_SMALL_MIN_BBOX_HEIGHT_RATIO: float = 0.55
PANEL_POLY_SMALL_MAX_CENTER_SHIFT_RATIO: float = 0.40
PANEL_POLY_SMALL_MIN_ASPECT_RATIO: float = 1.0

# --- Small panel: force bbox when conf >= threshold ---
SMALL_PANEL_FORCE_BBOX: bool = True
SMALL_PANEL_USE_BBOX_CONF: float = 0.70


# ===========================================================================
# SECTION 1 — GEOMETRY HELPERS
# ===========================================================================

def _bbox_to_poly(bbox: List[float]) -> List[List[int]]:
    """Convert [x1,y1,x2,y2] bbox to 4-point clockwise polygon."""
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def _sort_corners(box: np.ndarray) -> List[List[int]]:
    """Sort 4 corner points: Top-Left, Top-Right, Bottom-Right, Bottom-Left."""
    cx, cy = np.mean(box, axis=0)
    angles = np.arctan2(box[:, 1] - cy, box[:, 0] - cx)
    sorted_box = box[np.argsort(angles)]
    min_x, min_y = np.min(box, axis=0)
    distances = np.linalg.norm(sorted_box - [min_x, min_y], axis=1)
    tl_idx = np.argmin(distances)
    sorted_box = np.roll(sorted_box, -tl_idx, axis=0)
    return [[int(round(pt[0])), int(round(pt[1]))] for pt in sorted_box]


def _check_geometry_quality(
    pts: np.ndarray, bbox: List[float], img_shape: Tuple[int, int]
) -> bool:
    """
    Basic 4-point polygon quality check:
    - area >= 75% of bbox area
    - aspect ratio in [0.8, 6.0]
    - opposite sides within 10% of each other
    - all points within ±50 px of image bounds
    """
    if len(pts) != 4:
        return False
    h_img, w_img = img_shape[:2]

    area_quad = float(cv2.contourArea(pts.reshape(-1, 1, 2)))
    w_box = bbox[2] - bbox[0]
    h_box = bbox[3] - bbox[1]
    area_box = w_box * h_box
    if area_box <= 0:
        return False
    if area_quad < 0.75 * area_box:
        return False

    rect = cv2.minAreaRect(pts)
    (_, _), (w, h), _ = rect
    w = max(w, 1.0)
    h = max(h, 1.0)
    aspect = max(w, h) / min(w, h)
    if aspect < 0.8 or aspect > 6.0:
        return False

    try:
        sorted_pts = _sort_corners(pts)
        tl, tr, br, bl = [np.array(pt, dtype=np.float32) for pt in sorted_pts]
        len_top    = np.linalg.norm(tr - tl)
        len_bottom = np.linalg.norm(br - bl)
        len_left   = np.linalg.norm(bl - tl)
        len_right  = np.linalg.norm(br - tr)
        if 0 in (len_top, len_bottom, len_left, len_right):
            return False
        if (min(len_top, len_bottom) / max(len_top, len_bottom) < 0.90 or
                min(len_left, len_right) / max(len_left, len_right) < 0.90):
            return False
    except Exception:
        return False

    for pt in pts:
        x, y = pt
        if x < -50 or x > w_img + 50 or y < -50 or y > h_img + 50:
            return False

    return True


# ===========================================================================
# SECTION 2 — PANEL POLYGON GUARD
# ===========================================================================

def _validate_panel_polygon_against_bbox(
    poly,
    bbox: List[float],
    method: str = "unknown",
) -> Tuple[bool, str]:
    """
    Validate polygon against the YOLO bbox it came from.

    Uses strict thresholds for normal panels and relaxed thresholds for small
    panels (bbox_area <= SMALL_PANEL_BBOX_AREA_PX).

    Guard criteria (strict):
        area_ratio       >= 0.70
        width_ratio      >= 0.75
        height_ratio     >= 0.75
        center_shift_ratio <= 0.35
        aspect_ratio     in [1.2, 8.0]
    """
    try:
        pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
        if len(pts) < 3:
            return False, "too_few_points"

        bbox_w = max(bbox[2] - bbox[0], 1.0)
        bbox_h = max(bbox[3] - bbox[1], 1.0)
        bbox_area = bbox_w * bbox_h

        if bbox_area <= SMALL_PANEL_BBOX_AREA_PX:
            threshold_set    = "small"
            min_area_ratio   = PANEL_POLY_SMALL_MIN_BBOX_AREA_RATIO
            min_width_ratio  = PANEL_POLY_SMALL_MIN_BBOX_WIDTH_RATIO
            min_height_ratio = PANEL_POLY_SMALL_MIN_BBOX_HEIGHT_RATIO
            max_center_shift = PANEL_POLY_SMALL_MAX_CENTER_SHIFT_RATIO
            min_aspect       = PANEL_POLY_SMALL_MIN_ASPECT_RATIO
        else:
            threshold_set    = "strict"
            min_area_ratio   = PANEL_POLY_MIN_BBOX_AREA_RATIO
            min_width_ratio  = PANEL_POLY_MIN_BBOX_WIDTH_RATIO
            min_height_ratio = PANEL_POLY_MIN_BBOX_HEIGHT_RATIO
            max_center_shift = PANEL_POLY_MAX_CENTER_SHIFT_RATIO
            min_aspect       = PANEL_POLY_MIN_ASPECT_RATIO

        max_aspect = PANEL_POLY_MAX_ASPECT_RATIO

        poly_area = float(cv2.contourArea(pts.reshape(-1, 1, 2)))
        xc, yc, w, h = cv2.boundingRect(pts.astype(np.int32).reshape(-1, 1, 2))
        poly_w = float(w)
        poly_h = float(h)

        area_ratio   = poly_area / bbox_area if bbox_area > 0 else 0.0
        width_ratio  = poly_w / bbox_w
        height_ratio = poly_h / bbox_h

        poly_cx = float(np.mean(pts[:, 0]))
        poly_cy = float(np.mean(pts[:, 1]))
        bbox_cx = (bbox[0] + bbox[2]) / 2.0
        bbox_cy = (bbox[1] + bbox[3]) / 2.0
        dist = float(np.sqrt((poly_cx - bbox_cx) ** 2 + (poly_cy - bbox_cy) ** 2))
        center_shift = dist / max(bbox_w, bbox_h)

        aspect_ratio = max(poly_w, poly_h) / max(min(poly_w, poly_h), 1e-6)

        decision = "pass"
        fail_reason = "ok"

        if area_ratio < min_area_ratio:
            decision, fail_reason = "reject", f"area_ratio_low({area_ratio:.3f}<{min_area_ratio})"
        elif width_ratio < min_width_ratio:
            decision, fail_reason = "reject", f"width_ratio_low({width_ratio:.3f}<{min_width_ratio})"
        elif height_ratio < min_height_ratio:
            decision, fail_reason = "reject", f"height_ratio_low({height_ratio:.3f}<{min_height_ratio})"
        elif center_shift > max_center_shift:
            decision, fail_reason = "reject", f"center_shift_high({center_shift:.3f}>{max_center_shift})"
        elif aspect_ratio < min_aspect or aspect_ratio > max_aspect:
            decision, fail_reason = "reject", f"aspect_invalid({aspect_ratio:.2f})"

        logger.info(
            f"[PANEL_REFINE_GUARD] method={method} "
            f"bbox_area={bbox_area:.1f} area_ratio={area_ratio:.3f} "
            f"width_ratio={width_ratio:.3f} height_ratio={height_ratio:.3f} "
            f"center_shift={center_shift:.3f} threshold_set={threshold_set} "
            f"decision={decision} reason={fail_reason}"
        )

        if decision == "reject":
            return False, fail_reason
        return True, "ok"

    except Exception as e:
        return False, f"exception:{str(e)}"


# ===========================================================================
# SECTION 3 — PANEL CONTOUR REFINEMENT
# ===========================================================================

def _log_refine_result(
    idx: int,
    conf: float,
    bbox: List[float],
    raw_area: float,
    refined_poly: List,
    method: str,
) -> None:
    """Log one-line summary of refine decision."""
    refined_area = 0.0
    try:
        ref_pts = np.array(refined_poly, dtype=np.float32)
        if len(ref_pts) >= 3:
            refined_area = float(cv2.contourArea(ref_pts.reshape(-1, 1, 2)))
    except Exception:
        pass
    bbox_area = max((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]), 1.0)
    logger.info(
        f"[PANEL_REFINE] idx={idx} conf={conf:.4f} method={method} "
        f"raw_area={raw_area:.1f} refined_area={refined_area:.1f} "
        f"bbox_area={bbox_area:.1f} n_pts={len(refined_poly)}"
    )


def refine_panel_contour(
    orig_img: np.ndarray,
    bbox: List[float],
    xy_polygon: np.ndarray,
    conf: float = 0.0,
    idx: int = -1,
    image_name: str = "unknown",
    debug_info: Optional[Dict[str, Any]] = None,
) -> List[List[int]]:
    """
    Refine panel contour from raw YOLO mask polygon.

    Flow:
        1. Build bbox_poly as final fallback.
        2. Small panel + conf >= 0.70 + bbox_area <= 2500 → return bbox_poly immediately.
        3. Validate raw mask against bbox (guard). Fail → bbox_poly.
        4. convexHull + approxPolyDP (4 points) + geometry check + guard.
        5. Fail → minAreaRect + guard.
        6. Fail → bbox_poly.
    """
    # Silence unused debug_info fields for compat
    if debug_info is not None:
        debug_info.update({
            "small_bbox_original": [],
            "snapped_bbox": [],
            "snap_left_shift": 0,
            "snap_right_shift": 0,
            "snap_top_shift": 0,
            "snap_bottom_shift": 0,
            "snap_score_left": 0.0,
            "snap_score_right": 0.0,
            "snap_score_top": 0.0,
            "snap_score_bottom": 0.0,
            "snap_decision": "n/a",
            "snap_reason": "legacy_disabled",
        })

    bbox_poly = _bbox_to_poly(bbox)

    # Compute raw mask area for logging
    raw_area = 0.0
    try:
        raw_pts = xy_polygon.astype(np.float32)
        if len(raw_pts) >= 3:
            raw_area = float(cv2.contourArea(raw_pts.reshape(-1, 1, 2)))
    except Exception:
        pass

    # Step 2: small panel force bbox
    bbox_area_px = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    if (SMALL_PANEL_FORCE_BBOX
            and bbox_area_px <= SMALL_PANEL_BBOX_AREA_PX
            and conf >= SMALL_PANEL_USE_BBOX_CONF):
        _log_refine_result(idx, conf, bbox, raw_area, bbox_poly, "small_bbox_forced")
        return bbox_poly

    # Step 3: raw mask guard
    if xy_polygon is None or len(xy_polygon) < 3:
        _log_refine_result(idx, conf, bbox, raw_area, bbox_poly, "bbox_fallback(no_mask)")
        return bbox_poly

    valid_ok, reason = _validate_panel_polygon_against_bbox(
        xy_polygon, bbox, method="raw_mask_guard"
    )
    if not valid_ok:
        _log_refine_result(idx, conf, bbox, raw_area, bbox_poly, f"bbox_fallback(raw_mask:{reason})")
        return bbox_poly

    # Steps 4-6: convexHull → approxPolyDP → minAreaRect → bbox
    refined_poly = bbox_poly
    method = "bbox_fallback"

    try:
        pts = xy_polygon.astype(np.float32)
        hull = cv2.convexHull(pts)

        # Hull guard
        valid_ok, reason = _validate_panel_polygon_against_bbox(
            hull, bbox, method="hull_guard"
        )
        if not valid_ok:
            _log_refine_result(idx, conf, bbox, raw_area, bbox_poly, f"bbox_fallback(hull:{reason})")
            return bbox_poly

        # Step 4: approxPolyDP → 4 points
        arc_len = cv2.arcLength(hull, True)
        found_4pt = False
        for factor in np.linspace(0.005, 0.15, 120):
            epsilon = factor * arc_len
            approx = cv2.approxPolyDP(hull, epsilon, True).reshape(-1, 2)
            if len(approx) == 4:
                if _check_geometry_quality(approx, bbox, orig_img.shape):
                    valid_ok, reason = _validate_panel_polygon_against_bbox(
                        approx, bbox, method="approxPolyDP"
                    )
                    if valid_ok:
                        refined_poly = _sort_corners(approx)
                        method = "convexHull+approxPolyDP"
                        found_4pt = True
                        break

        if not found_4pt:
            # Step 5: minAreaRect
            rect = cv2.minAreaRect(pts)
            box = cv2.boxPoints(rect)
            if len(box) == 4:
                valid_ok, reason = _validate_panel_polygon_against_bbox(
                    box, bbox, method="minAreaRect"
                )
                if valid_ok:
                    refined_poly = _sort_corners(box)
                    method = "minAreaRect"
                else:
                    # Step 6: bbox fallback
                    _log_refine_result(idx, conf, bbox, raw_area, bbox_poly,
                                       f"bbox_fallback(minAreaRect:{reason})")
                    return bbox_poly

    except Exception as e:
        logger.debug(f"[refine_panel_contour] idx={idx} exception: {e}")
        method = "bbox_fallback(exception)"
        refined_poly = bbox_poly

    _log_refine_result(idx, conf, bbox, raw_area, refined_poly, method)
    return refined_poly


# ===========================================================================
# SECTION 4 — DEFECT POLYGON SIMPLIFICATION
# ===========================================================================

def simplify_defect_polygon(
    xy_polygon: np.ndarray, max_vertices: int = 32
) -> List[List[float]]:
    """
    Simplify defect polygon to at most max_vertices vertices.
    Progressively increases epsilon until vertices <= max_vertices.
    Falls back to original if simplification collapses below 3 points.
    """
    if xy_polygon is None:
        return []
    try:
        poly_arr = np.array(xy_polygon, dtype=np.float32)
        if poly_arr is None or poly_arr.ndim == 0 or len(poly_arr) < 3:
            return []
        # Tạm thời tắt tính năng nén polygon bằng approxPolyDP để trả về mask gốc của YOLO theo yêu cầu của user
        return poly_arr.tolist()
    except Exception:
        if hasattr(xy_polygon, "tolist"):
            return xy_polygon.tolist()
        return list(xy_polygon) if xy_polygon is not None else []



# ===========================================================================
# SECTION 4b — YOLO MASK MORPHOLOGICAL SMOOTHING (NO L*, NO THERMAL)
# ===========================================================================

def mask_to_display_polygon(
    mask: np.ndarray,
    max_vertices: int = 128,
    fallback_polygon: Optional[List[List[float]]] = None,
) -> List[List[float]]:
    """
    Convert binary mask to display polygon via cv2.findContours.

    Rules:
      - Pick the largest contour.
      - Discard contours smaller than 1% of total mask area.
      - Apply very light approxPolyDP only if needed to stay under max_vertices.
      - Do NOT use Shapely union/buffer.
      - Do NOT force below 32 vertices for defects.
      - Fallback to fallback_polygon if no valid contour found.

    Args:
        mask:             Binary mask (uint8, 0/255), same coords as image.
        max_vertices:     Upper limit on output vertices (default 128).
        fallback_polygon: YOLO polygon to return if conversion fails.

    Returns:
        List of [x, y] float points.
    """
    if fallback_polygon is None:
        fallback_polygon = []

    if mask is None or mask.size == 0:
        return fallback_polygon

    try:
        mask_u8 = (mask > 0).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return fallback_polygon

        # Filter tiny contours (< 1% of total mask area)
        total_area = int((mask_u8 > 0).sum())
        min_area = max(4, total_area * 0.01)
        valid = [c for c in contours if cv2.contourArea(c) >= min_area]
        if not valid:
            valid = contours  # Keep all if filter is too strict

        largest = max(valid, key=cv2.contourArea)

        pts = largest.reshape(-1, 2)
        if len(pts) < 3:
            return fallback_polygon

        # Light approxPolyDP only if above max_vertices — keep shape faithful
        if len(pts) > max_vertices:
            arc = cv2.arcLength(largest, True)
            # Start very tight; only increase epsilon until under limit
            for factor in [0.001, 0.002, 0.005, 0.010, 0.020, 0.030]:
                approx = cv2.approxPolyDP(largest, factor * arc, True)
                if len(approx) <= max_vertices:
                    pts = approx.reshape(-1, 2)
                    break
            else:
                pts = largest.reshape(-1, 2)  # Give up, keep original

        return [[float(p[0]), float(p[1])] for p in pts]

    except Exception as e:
        logger.debug(f"[mask_to_display_polygon] exception: {e}")
        return fallback_polygon


def create_yolo_display_mask(
    yolo_polygon: List[List[float]],
    image_shape: Tuple[int, int],
    class_name: str,
) -> Tuple[np.ndarray, str, float]:
    """
    Apply safe morphological smoothing to the YOLO segmentation mask.

    Principles:
      - YOLO segmentation is the primary polygon source.
      - Do NOT use L* or any thermal data to reshape/threshold the mask.
      - Only light morphological operations are applied.
      - Result is always constrained back to original YOLO mask footprint.
      - If smoothing shrinks mask too much, fallback to YOLO raw mask.

    Args:
        yolo_polygon:  [[x,y], ...] YOLO segmentation polygon in pixel coords.
        image_shape:   (H, W) of the source image.
        class_name:    Defect class name (controls aggressiveness).

    Returns:
        (result_mask, source_tag, area_retention)
        - result_mask:    uint8 mask (0/255), H×W.
        - source_tag:     One of "yolo_segmentation", "yolo_segmentation_smoothed",
                          "yolo_segmentation_eroded_1px".
        - area_retention: float in [0, 1] — fraction of original YOLO mask area kept.
    """
    h, w = image_shape[:2]
    cls = (class_name or "").lower()

    # ── 1. Draw original YOLO mask ──────────────────────────────────────────
    yolo_mask = np.zeros((h, w), dtype=np.uint8)
    try:
        pts = np.array([[int(round(p[0])), int(round(p[1]))] for p in yolo_polygon], dtype=np.int32)
        if len(pts) >= 3:
            cv2.fillPoly(yolo_mask, [pts], 255)
    except Exception:
        pass

    area_raw = int((yolo_mask > 0).sum())
    if area_raw == 0:
        return yolo_mask, "yolo_segmentation", 1.0

    # ── 2. Crack: no morphological ops — return raw YOLO directly ───────────
    if "crack" in cls:
        return yolo_mask, "yolo_segmentation", 1.0

    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # ── 3. Morphological close → fill small gaps/holes ──────────────────────
    closed = cv2.morphologyEx(yolo_mask, cv2.MORPH_CLOSE, kernel3, iterations=1)

    # ── 4. Morphological open → remove tiny noise specks ────────────────────
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel3, iterations=1)

    # ── 5. Clamp back to original YOLO footprint (never expand outside) ─────
    clean_mask = cv2.bitwise_and(opened, yolo_mask)

    area_clean = int((clean_mask > 0).sum())
    retention_clean = area_clean / area_raw if area_raw > 0 else 0.0

    # ── 6. Erode 1px to trim slight over-segmentation ───────────────────────
    eroded_mask = cv2.erode(clean_mask, kernel3, iterations=1)
    area_eroded = int((eroded_mask > 0).sum())
    retention_eroded = area_eroded / area_raw if area_raw > 0 else 0.0

    # ── 7. Quality gate per class group ─────────────────────────────────────
    is_multi_cell  = "multi" in cls
    is_single_cell = "single" in cls or ("hotspot" in cls and not is_multi_cell)
    is_hotspot     = "hotspot" in cls or "hot" in cls
    is_shading     = "shad" in cls or "shadow" in cls or "soil" in cls or "dirt" in cls

    use_eroded = False
    use_clean  = False

    if is_multi_cell:
        # Multi-cell hotspot spans multiple cells — protect against shrinkage
        use_eroded = (retention_eroded >= 0.85)
    elif is_single_cell or is_hotspot:
        use_eroded = (retention_eroded >= 0.65)
    elif is_shading:
        # Shading covers large diffuse areas — do NOT erode
        use_eroded = False
        use_clean  = (retention_clean >= 0.80)
    else:
        # Unknown class: conservative
        use_eroded = (retention_eroded >= 0.70)

    # ── 8. Select result mask ────────────────────────────────────────────────
    if use_eroded and area_eroded >= 3:
        result_mask      = eroded_mask
        source_tag       = "yolo_segmentation_eroded_1px"
        area_retention   = round(retention_eroded, 4)
    elif (use_clean or (not use_eroded and not is_shading)) and area_clean >= 3 and retention_clean >= 0.60:
        result_mask      = clean_mask
        source_tag       = "yolo_segmentation_smoothed"
        area_retention   = round(retention_clean, 4)
    else:
        # Fallback to raw YOLO mask
        result_mask      = yolo_mask
        source_tag       = "yolo_segmentation"
        area_retention   = 1.0

    return result_mask, source_tag, area_retention


def calculate_bbox_iou(box_a: List[float], box_b: List[float]) -> float:
    """
    Calculate IoU between two bboxes [x1, y1, x2, y2].
    """
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def calculate_poly_iou(poly_a: List[List[float]], poly_b: List[List[float]], bbox_a: List[float], bbox_b: List[float]) -> float:
    """
    Calculate IoU between two polygons. Fallbacks to bbox IoU if shapely fails.
    """
    try:
        from shapely.geometry import Polygon as ShapelyPolygon
        p_a = ShapelyPolygon(poly_a)
        p_b = ShapelyPolygon(poly_b)
        if not p_a.is_valid:
            p_a = p_a.buffer(0)
        if not p_b.is_valid:
            p_b = p_b.buffer(0)
        if not p_a.is_empty and not p_b.is_empty:
            inter = p_a.intersection(p_b).area
            union = p_a.union(p_b).area
            if union > 0:
                return inter / union
    except Exception:
        pass
    return calculate_bbox_iou(bbox_a, bbox_b)


def calculate_centroid_distance(c1: List[float], c2: List[float]) -> float:
    return float(((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)**0.5)


def deduplicate_defects_by_polygon_iou(defects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not defects:
        return []
    
    sorted_defects = sorted(defects, key=lambda d: d.get("confidence", 0.0), reverse=True)
    kept = []
    
    for d in sorted_defects:
        is_duplicate = False
        poly_d = d.get("polygon", [])
        bbox_d = d.get("bbox", [0, 0, 0, 0])
        centroid_d = d.get("center", [0, 0])
        cls_d = d.get("class_name", "")
        
        for k in kept:
            poly_k = k.get("polygon", [])
            bbox_k = k.get("bbox", [0, 0, 0, 0])
            centroid_k = k.get("center", [0, 0])
            cls_k = k.get("class_name", "")
            
            iou = calculate_poly_iou(poly_d, poly_k, bbox_d, bbox_k)
            dist = calculate_centroid_distance(centroid_d, centroid_k)
            
            if cls_d == cls_k and iou >= 0.6:
                is_duplicate = True
                break
                
            if dist <= 6.0 and iou >= 0.45:
                is_duplicate = True
                break
                
        if not is_duplicate:
            kept.append(d)
            
    return sorted(kept, key=lambda d: d.get("raw_idx", 0))





# ===========================================================================
# SECTION 5 — DEBUG JSONL LOGGER
# ===========================================================================

def save_panel_refine_debug_jsonl(
    image_path: str,
    raw_panels: List[Dict[str, Any]],
) -> None:
    """
    Write per-panel refine debug JSONL to:
        data/results/debug_logs/<image_name>_panel_refine.jsonl

    Each record contains: bbox, raw mask ratio, method, decision, reason.
    """
    try:
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        debug_dir = "data/results/debug_logs"
        os.makedirs(debug_dir, exist_ok=True)
        out_path = os.path.join(debug_dir, f"{image_name}_panel_refine.jsonl")

        with open(out_path, "w", encoding="utf-8") as f:
            for p in raw_panels:
                box = p.get("box") or p.get("bbox", [0, 0, 0, 0])
                if len(box) < 4:
                    continue

                x1b, y1b, x2b, y2b = float(box[0]), float(box[1]), float(box[2]), float(box[3])
                bbox_w   = max(x2b - x1b, 1.0)
                bbox_h   = max(y2b - y1b, 1.0)
                bbox_area = bbox_w * bbox_h

                conf = float(p.get("raw_conf", p.get("confidence", 0.0)))
                idx  = int(p.get("raw_idx", -1))
                raw_yolo_poly = p.get("raw_yolo_poly", [])
                refined_poly  = p.get("polygon", [])

                # Raw mask metrics
                raw_mask_area = 0.0
                wm = hm = 0
                mask_cx = (x1b + x2b) / 2
                mask_cy = (y1b + y2b) / 2
                try:
                    raw_pts = np.array(raw_yolo_poly, dtype=np.float32)
                    if len(raw_pts) >= 3:
                        raw_mask_area = float(cv2.contourArea(raw_pts.reshape(-1, 1, 2)))
                        xm2, ym2, wm, hm = cv2.boundingRect(
                            raw_pts.astype(np.int32).reshape(-1, 1, 2)
                        )
                        mask_cx = float(np.mean(raw_pts[:, 0]))
                        mask_cy = float(np.mean(raw_pts[:, 1]))
                except Exception:
                    pass

                area_ratio   = raw_mask_area / bbox_area if bbox_area > 0 else 0.0
                width_ratio  = float(wm) / bbox_w if bbox_w > 0 else 0.0
                height_ratio = float(hm) / bbox_h if bbox_h > 0 else 0.0
                bbox_cx = (x1b + x2b) / 2
                bbox_cy = (y1b + y2b) / 2
                dist = float(np.sqrt((mask_cx - bbox_cx) ** 2 + (mask_cy - bbox_cy) ** 2))
                center_shift = dist / max(bbox_w, bbox_h) if max(bbox_w, bbox_h) > 0 else 0.0

                # Infer method / decision from polygon content
                x1i, y1i, x2i, y2i = (
                    int(round(x1b)), int(round(y1b)),
                    int(round(x2b)), int(round(y2b)),
                )
                expected_bbox = [[x1i, y1i], [x2i, y1i], [x2i, y2i], [x1i, y2i]]
                if refined_poly == expected_bbox:
                    if (SMALL_PANEL_FORCE_BBOX
                            and bbox_area <= SMALL_PANEL_BBOX_AREA_PX
                            and conf >= SMALL_PANEL_USE_BBOX_CONF):
                        method, decision, reason = "small_bbox", "pass", "force_bbox_small"
                    else:
                        method, decision, reason = "fallback_bbox", "fallback_bbox", "mask_guard_rejected"
                else:
                    method, decision, reason = "mask_pipeline", "pass", "mask_refined_ok"

                record = {
                    "image_name":    image_name,
                    "idx":           idx,
                    "local_id":      p.get("local_id", "N/A"),
                    "class_name":    "panel",
                    "conf":          round(conf, 4),
                    "bbox":          [round(v, 1) for v in box],
                    "bbox_area":     round(bbox_area, 1),
                    "raw_mask_area": round(raw_mask_area, 1),
                    "area_ratio":    round(area_ratio, 3),
                    "width_ratio":   round(width_ratio, 3),
                    "height_ratio":  round(height_ratio, 3),
                    "center_shift":  round(center_shift, 3),
                    "threshold_set": "small" if bbox_area <= SMALL_PANEL_BBOX_AREA_PX else "strict",
                    "method":        method,
                    "decision":      decision,
                    "refined_polygon": refined_poly,
                    "raw_yolo_poly": raw_yolo_poly,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.info(f"[PANEL_DEBUG_FILE] saved {out_path}")

    except Exception as e:
        logger.warning(f"[PANEL_DEBUG_FILE] WARN: {e}")


# ===========================================================================
# SECTION 6 — EDGE GRID SEGMENTED-4 (OPTIONAL DEBUG, FLAG=False)
# ===========================================================================

@dataclass
class EdgeGridConfig:
    roi_margin: int = 45
    vertical_mode: str = "segmented"
    n_segments: int = 4
    max_x_jump_px: float = 10.0
    max_angle_delta_deg: float = 8.0
    search_half_width: int = 10
    min_points_per_segment: int = 6
    max_dev_baseline_px: float = 12.0
    smooth_window: int = 3


def _intersect_h_v(
    h_y: float, v_pts: List[Tuple[float, float]]
) -> Optional[float]:
    """Intersect horizontal line y=h_y with piecewise vertical polyline v_pts."""
    if len(v_pts) < 2:
        return None
    for i in range(len(v_pts) - 1):
        y0, x0 = v_pts[i]
        y1, x1 = v_pts[i + 1]
        if y0 <= h_y <= y1 or y1 <= h_y <= y0:
            if abs(y1 - y0) < 1e-6:
                return (x0 + x1) / 2.0
            t = (h_y - y0) / (y1 - y0)
            return x0 + t * (x1 - x0)
    # Clamp to nearest endpoint
    if h_y <= v_pts[0][0]:
        return v_pts[0][1]
    return v_pts[-1][1]


def _detect_horizontal_gaps(
    roi_gray: np.ndarray,
    roi_y: int,
    img_h: int,
    n_segments: int,
) -> List[float]:
    """Detect horizontal gap Y positions (panel boundaries) within ROI."""
    h = roi_gray.shape[0]
    if h < 4:
        return []
    # Sobel-Y horizontal gradient
    sobel_y = cv2.Sobel(roi_gray, cv2.CV_32F, 0, 1, ksize=3)
    h_response = np.abs(sobel_y).mean(axis=1)
    # Smooth
    kernel_size = max(3, h // (n_segments * 4)) | 1
    if kernel_size > 1:
        h_response = np.convolve(h_response, np.ones(kernel_size) / kernel_size, mode="same")

    # Find local maxima as gap candidates
    gap_ys = []
    step = max(1, h // (n_segments + 1))
    for si in range(1, n_segments):
        lo = si * step - step // 2
        hi = si * step + step // 2
        lo, hi = max(0, lo), min(h - 1, hi)
        if lo >= hi:
            continue
        seg = h_response[lo:hi]
        local_max = int(np.argmax(seg)) + lo
        gap_ys.append(float(roi_y + local_max))
    return gap_ys


def _fit_segmented_vertical_boundary(
    roi_edges: np.ndarray,
    roi_x: int,
    roi_y: int,
    baseline_x: float,
    config: EdgeGridConfig,
) -> List[Tuple[float, float]]:
    """
    Fit a segmented vertical boundary polyline within the edge ROI.
    Returns list of (y_global, x_global) points.
    """
    h, w = roi_edges.shape[:2]
    if h < 8:
        return [(float(roi_y), baseline_x), (float(roi_y + h), baseline_x)]

    n = config.n_segments
    seg_h = h / n
    half_w = config.search_half_width
    baseline_local = baseline_x - roi_x

    pts_out: List[Tuple[float, float]] = []

    prev_angle = None
    prev_x = baseline_local

    for si in range(n):
        y0 = int(si * seg_h)
        y1 = min(int((si + 1) * seg_h), h)
        seg = roi_edges[y0:y1, :]

        # Candidate edge pixels within search band around current baseline
        x_lo = int(max(0, prev_x - half_w))
        x_hi = int(min(w, prev_x + half_w))
        seg_band = seg[:, x_lo:x_hi]
        edge_ys, edge_xs = np.where(seg_band > 0)

        fallback = False
        if len(edge_xs) < config.min_points_per_segment:
            fallback = True
        else:
            edge_xs_g = edge_xs + x_lo
            # Fit linear: x = a*y + b
            try:
                coeffs = np.polyfit(edge_ys, edge_xs_g, 1)
                slope  = coeffs[0]
                intercept = coeffs[1]
                angle_deg = float(np.degrees(np.arctan(slope)))

                # Jump constraint
                mid_y = (y0 + y1) / 2.0
                fit_x = slope * mid_y + intercept
                if abs(fit_x - prev_x) > config.max_x_jump_px:
                    fallback = True
                # Angle constraint
                elif prev_angle is not None and abs(angle_deg - prev_angle) > config.max_angle_delta_deg:
                    fallback = True
                # Baseline deviation constraint
                elif abs(fit_x - baseline_local) > config.max_dev_baseline_px:
                    fallback = True
                else:
                    prev_angle = angle_deg
                    prev_x = fit_x
                    x_top = slope * y0 + intercept
                    x_bot = slope * y1 + intercept
                    pts_out.append((float(roi_y + y0), float(roi_x + x_top)))
                    pts_out.append((float(roi_y + y1), float(roi_x + x_bot)))
                    continue
            except Exception:
                fallback = True

        if fallback:
            # Use baseline for this segment
            pts_out.append((float(roi_y + y0), baseline_x))
            pts_out.append((float(roi_y + y1), baseline_x))

    # Merge duplicate Y points (take midpoint X for same Y)
    if not pts_out:
        return [(float(roi_y), baseline_x), (float(roi_y + h), baseline_x)]

    # Smooth: moving average of X values over window
    ys = [p[0] for p in pts_out]
    xs = [p[1] for p in pts_out]
    win = config.smooth_window
    xs_smooth = []
    for i in range(len(xs)):
        lo2 = max(0, i - win // 2)
        hi2 = min(len(xs), i + win // 2 + 1)
        xs_smooth.append(float(np.mean(xs[lo2:hi2])))

    return list(zip(ys, xs_smooth))


def _compute_polygon_overlap_ratio(
    poly_a: List[List[int]],
    poly_b: List[List[int]],
    img_shape: Tuple[int, ...],
) -> float:
    """Overlap ratio = intersection_area / min(area_a, area_b)."""
    try:
        h, w = img_shape[:2]
        m_a = np.zeros((h, w), dtype=np.uint8)
        m_b = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(m_a, [np.array(poly_a, dtype=np.int32)], 1)
        cv2.fillPoly(m_b, [np.array(poly_b, dtype=np.int32)], 1)
        intersection = float(np.logical_and(m_a, m_b).sum())
        min_area = min(float(m_a.sum()), float(m_b.sum()))
        return intersection / min_area if min_area > 0 else 0.0
    except Exception:
        return 0.0


def propose_edge_grid_polygons_in_pipeline(
    orig_img: np.ndarray,
    panels: List[Dict[str, Any]],
    target_panel_indices_or_ids: List[int],
    config: EdgeGridConfig,
) -> Dict[str, Any]:
    """
    Run edge-grid segmented-4 boundary proposal for a target group of panels.
    Returns {"proposals": [...], "debug": {...}}.

    Each proposal: {"raw_idx": int, "polygon": [[x,y],...], "ok": bool,
                    "source": str, "metrics": {"iou": float, "center_shift": float, "area_ratio": float}}
    """
    target_set = set(target_panel_indices_or_ids)
    target_panels = [p for p in panels if p.get("raw_idx") in target_set]
    if not target_panels:
        return {"proposals": [], "debug": {}}

    img_h, img_w = orig_img.shape[:2]

    # Build ROI from target panel bboxes
    xs, ys = [], []
    for p in target_panels:
        bb = p.get("box") or p.get("bbox", [0, 0, img_w, img_h])
        xs.extend([bb[0], bb[2]])
        ys.extend([bb[1], bb[3]])

    margin = config.roi_margin
    roi_x1 = int(max(0, min(xs) - margin))
    roi_y1 = int(max(0, min(ys) - margin))
    roi_x2 = int(min(img_w, max(xs) + margin))
    roi_y2 = int(min(img_h, max(ys) + margin))

    roi = orig_img[roi_y1:roi_y2, roi_x1:roi_x2]
    if roi.size == 0:
        return {"proposals": [], "debug": {}}

    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
    sigma    = float(np.std(roi_gray))
    median_v = float(np.median(roi_gray))
    low_t    = max(10, int(median_v - 0.66 * sigma))
    high_t   = max(low_t + 20, int(median_v + 0.66 * sigma))
    roi_edges = cv2.Canny(cv2.GaussianBlur(roi_gray, (5, 5), 0), low_t, high_t)

    # Estimate left and right vertical boundaries from all target-panel bboxes
    all_x1s = [float((p.get("box") or p.get("bbox", [0, 0, img_w, img_h]))[0]) for p in target_panels]
    all_x2s = [float((p.get("box") or p.get("bbox", [0, 0, img_w, img_h]))[2]) for p in target_panels]
    baseline_left  = float(np.median(all_x1s))
    baseline_right = float(np.median(all_x2s))

    # Detect horizontal gaps
    gap_ys = _detect_horizontal_gaps(roi_gray, roi_y1, img_h, config.n_segments)

    # Fit left and right vertical boundaries
    left_vert  = _fit_segmented_vertical_boundary(
        roi_edges, roi_x1, roi_y1, baseline_left, config
    )
    right_vert = _fit_segmented_vertical_boundary(
        roi_edges, roi_x1, roi_y1, baseline_right, config
    )

    # Sort target panels by center Y
    target_sorted = sorted(target_panels, key=lambda p: p.get("center", [0, 0])[1])

    # Build horizontal lines (panel boundaries) from bboxes + gap detections
    h_lines = []
    for p in target_sorted:
        bb = p.get("box") or p.get("bbox", [0, 0, img_w, img_h])
        h_lines.append(float(bb[1]))
    if target_sorted:
        bb_last = target_sorted[-1].get("box") or target_sorted[-1].get("bbox", [0, 0, img_w, img_h])
        h_lines.append(float(bb_last[3]))
    h_lines = sorted(set(h_lines))

    # Build proposals
    proposals = []
    vb_fitted_count = [0, 0]  # [left_ok, right_ok]
    for si, p in enumerate(target_sorted):
        ridx = p.get("raw_idx", -1)
        bb = p.get("box") or p.get("bbox", [0, 0, img_w, img_h])
        top_y  = float(bb[1])
        bot_y  = float(bb[3])
        mid_y  = (top_y + bot_y) / 2.0

        xl_top = _intersect_h_v(top_y, left_vert)
        xl_bot = _intersect_h_v(bot_y, left_vert)
        xr_top = _intersect_h_v(top_y, right_vert)
        xr_bot = _intersect_h_v(bot_y, right_vert)

        if any(v is None for v in [xl_top, xl_bot, xr_top, xr_bot]):
            proposals.append({
                "raw_idx": ridx, "polygon": [], "ok": False,
                "source": "boundary_intersection_failed",
                "metrics": {"iou": 0.0, "center_shift": 9999.0, "area_ratio": 0.0},
            })
            continue

        # Clamp to image
        xl_top = max(0.0, min(float(img_w), xl_top))
        xl_bot = max(0.0, min(float(img_w), xl_bot))
        xr_top = max(0.0, min(float(img_w), xr_top))
        xr_bot = max(0.0, min(float(img_w), xr_bot))

        poly_proposed = [
            [xl_top, top_y], [xr_top, top_y],
            [xr_bot, bot_y], [xl_bot, bot_y],
        ]

        # Metrics vs original bbox polygon
        orig_poly = [
            [bb[0], bb[1]], [bb[2], bb[1]],
            [bb[2], bb[3]], [bb[0], bb[3]],
        ]
        try:
            m_orig = np.zeros((img_h, img_w), dtype=np.uint8)
            m_prop = np.zeros((img_h, img_w), dtype=np.uint8)
            cv2.fillPoly(m_orig, [np.array(orig_poly, dtype=np.int32)], 1)
            cv2.fillPoly(m_prop, [np.array([[int(round(x)), int(round(y))] for x, y in poly_proposed], dtype=np.int32)], 1)
            inter = float(np.logical_and(m_orig, m_prop).sum())
            union = float(np.logical_or(m_orig, m_prop).sum())
            iou = inter / union if union > 0 else 0.0
            area_orig = float(m_orig.sum())
            area_prop = float(m_prop.sum())
            area_ratio = area_prop / area_orig if area_orig > 0 else 1.0
        except Exception:
            iou, area_ratio = 0.0, 1.0

        # Center shift
        orig_cx = (bb[0] + bb[2]) / 2.0
        orig_cy = (bb[1] + bb[3]) / 2.0
        prop_cx = float(np.mean([pt[0] for pt in poly_proposed]))
        prop_cy = float(np.mean([pt[1] for pt in poly_proposed]))
        center_shift = float(np.sqrt((prop_cx - orig_cx) ** 2 + (prop_cy - orig_cy) ** 2))

        poly_int = [[int(round(x)), int(round(y))] for x, y in poly_proposed]
        proposals.append({
            "raw_idx": ridx,
            "polygon": poly_int,
            "ok": True,
            "source": "segmented_boundary_projection",
            "metrics": {
                "iou": round(iou, 4),
                "center_shift": round(center_shift, 4),
                "area_ratio": round(area_ratio, 4),
            },
        })

        # Track whether boundaries were non-trivially fitted
        if abs(xl_top - baseline_left) > 0.5 or abs(xl_bot - baseline_left) > 0.5:
            vb_fitted_count[0] += 1
        if abs(xr_top - baseline_right) > 0.5 or abs(xr_bot - baseline_right) > 0.5:
            vb_fitted_count[1] += 1

    debug_out = {
        "roi": [roi_x1, roi_y1, roi_x2, roi_y2],
        "left_boundary": left_vert,
        "right_boundary": right_vert,
        "horizontal_lines": h_lines,
        "vertical_boundaries_fitted": vb_fitted_count,
        "gap_ys": gap_ys,
    }
    return {"proposals": proposals, "debug": debug_out}


# ===========================================================================
# SECTION 7 — MAIN ENTRY POINT
# ===========================================================================

def process_yolo_predictions(
    result,
    orig_img: np.ndarray,
    panel_conf: float,
    defect_conf: float,
    panel_class_name: str = "panel",
) -> List[Dict[str, Any]]:
    """
    Extract panels and defects from a YOLO segmentation result.

    Pipeline:
      1. Iterate YOLO detections.
      2. Panel: apply conf threshold → refine_panel_contour → build panel dict.
      3. Defect: apply conf threshold → simplify_defect_polygon → build defect dict.
      4. Optional edge_grid_segmented4 pass (ENABLE_EDGE_GRID_SEGMENTED_REFINEMENT).
      5. Call assign_defects_to_panels.
      6. Write panel_refine debug JSONL.
      7. Return detections list (panels + unassigned defects).
    """

    if result.masks is None:
        return []

    masks_xy = result.masks.xy
    image_path = getattr(result, "path", "unknown_image.jpg")
    image_name = os.path.splitext(os.path.basename(image_path))[0]

    raw_panels: List[Dict] = []
    raw_defects: List[Dict] = []

    # ── PASS 1: Extract panels and defects ─────────────────────────────────
    for i in range(len(result.boxes)):
        class_id   = int(result.boxes.cls[i])
        class_name = result.names[class_id]
        confidence = float(result.boxes.conf[i])
        bbox       = result.boxes.xyxy[i].tolist()   # [x1,y1,x2,y2]
        xy_polygon = masks_xy[i]                      # np.ndarray (N,2)

        # Log raw YOLO output
        try:
            raw_pts   = xy_polygon.astype(np.float32)
            mask_area = float(cv2.contourArea(raw_pts.reshape(-1, 1, 2))) if len(raw_pts) >= 3 else 0.0
        except Exception:
            mask_area = 0.0
        bw = bbox[2] - bbox[0]
        bh = bbox[3] - bbox[1]
        logger.info(
            f"[RAW_YOLO] idx={i} class={class_name} conf={confidence:.4f} "
            f"bbox={[round(v, 1) for v in bbox]} "
            f"w={bw:.1f} h={bh:.1f} area={bw*bh:.1f} mask_area={mask_area:.1f} pts={len(xy_polygon)}"
        )

        if class_name.lower() == panel_class_name:
            # ── PANEL ───────────────────────────────────────────────────────
            if confidence < panel_conf or len(xy_polygon) < 3:
                continue

            debug_info: Dict[str, Any] = {}
            panel_poly = refine_panel_contour(
                orig_img=orig_img,
                bbox=bbox,
                xy_polygon=xy_polygon,
                conf=confidence,
                idx=i,
                image_name=image_name,
                debug_info=debug_info,
            )

            if len(panel_poly) != 4:
                logger.info(f"[PANEL_FILTER] Rejecting panel idx={i} due to polygon edges count != 4 ({len(panel_poly)})")
                continue

            features = get_polygon_features(panel_poly)

            panel_dict: Dict[str, Any] = {
                "class_name":   class_name,
                "confidence":   round(confidence, 4),
                "bbox":         [round(v) for v in features["bbox"]],
                "polygon":      panel_poly,
                "area":         features["area"],
                "center":       features["center"],
                "aspect_ratio": features["aspect_ratio"],
                "category":     "panel",
                "box":          [round(v) for v in bbox],
                # Debug / traceability
                "raw_yolo_poly":         xy_polygon.tolist() if xy_polygon is not None else [],
                "raw_idx":               i,
                "raw_conf":              confidence,
                "final_polygon_source":  "yolo_original",
                "final_polygon_stage":   "yolo_refinement",
                "final_polygon_reason":  "yolo_detection_contour",
            }
            panel_dict.update(debug_info)
            raw_panels.append(panel_dict)

        else:
            # ── DEFECT ──────────────────────────────────────────────────────
            if confidence < defect_conf:
                continue

            if xy_polygon is not None and len(xy_polygon) >= 3:
                yolo_polygon = xy_polygon.tolist()
                poly_source = "yolo_segmentation"
                # Analysis polygon: lightly simplified for geometry math
                analysis_poly = simplify_defect_polygon(
                    np.array(yolo_polygon, dtype=np.float32), max_vertices=64
                )
                if analysis_poly is None or len(analysis_poly) < 3:
                    analysis_poly = yolo_polygon
            else:
                # Fallback to bbox
                bx1, by1, bx2, by2 = bbox
                yolo_polygon = [[bx1, by1], [bx2, by1], [bx2, by2], [bx1, by2]]
                analysis_poly = yolo_polygon
                poly_source = "bbox_fallback"
                low_geometry_confidence = True

            features = get_polygon_features(yolo_polygon)

            raw_defects.append({
                "class_name":   class_name,
                "confidence":   round(confidence, 4),
                "bbox":         [round(v) for v in features["bbox"]],
                "low_geometry_confidence": locals().get("low_geometry_confidence", False),
                # ── Display fields (pure YOLO) ──
                "polygon":      yolo_polygon,
                "display_polygon": yolo_polygon,
                "yolo_polygon": yolo_polygon,
                "raw_yolo_polygon": yolo_polygon,
                # ── Analysis (for area/severity math) ──
                "analysis_polygon": analysis_poly,
                "polygon_source": poly_source,
                "display_polygon_source": poly_source,
                "analysis_polygon_source": poly_source,
                "area":         features["area"],
                "center":       features["center"],
                "category":     "defect",
                "box":          [round(v) for v in bbox],
                "raw_idx":      i,
                "polygon_point_count": len(yolo_polygon),
                "raw_polygon_point_count": len(yolo_polygon),
            })

    # ── OPTIONAL: Edge-Grid Segmented-4 Refinement ─────────────────────────
    if ENABLE_EDGE_GRID_SEGMENTED_REFINEMENT:
        # Identify candidate panels that still use yolo_original
        poor_panels = [
            p for p in raw_panels
            if (p.get("final_polygon_source") == "yolo_original"
                or "failed"         in p.get("final_polygon_reason", "")
                or "rejected"       in p.get("final_polygon_reason", "")
                or "area_ratio_out" in p.get("final_polygon_reason", "")
                or "large_col_dist" in p.get("final_polygon_reason", ""))
        ]

        # Initialize trace fields
        for p in raw_panels:
            p["edge_grid_attempted"]    = False
            p["edge_grid_accept"]       = False
            p["edge_grid_reject_reason"] = None
            p["edge_grid_area_ratio"]   = None
            p["edge_grid_center_shift"] = None
            p["edge_grid_overlap"]      = None
            p["edge_grid_vertical_mode"] = "segmented"
            p["edge_grid_n_segments"]    = 4

        # Cluster poor panels into columns (X-delta <= 35 px, Y-gap <= 120 px, size >= 4)
        poor_sorted = sorted(poor_panels, key=lambda p: p.get("center", [0, 0])[0])
        column_groups: List[List[Dict]] = []
        if poor_sorted:
            cur = [poor_sorted[0]]
            for p in poor_sorted[1:]:
                if abs(p.get("center", [0, 0])[0] - cur[-1].get("center", [0, 0])[0]) <= 35:
                    cur.append(p)
                else:
                    column_groups.append(cur)
                    cur = [p]
            column_groups.append(cur)

        valid_groups: List[List[Dict]] = []
        for grp in column_groups:
            grp_s = sorted(grp, key=lambda p: p.get("center", [0, 0])[1])
            cur_sub = [grp_s[0]]
            for p in grp_s[1:]:
                if abs(p.get("center", [0, 0])[1] - cur_sub[-1].get("center", [0, 0])[1]) <= 120.0:
                    cur_sub.append(p)
                else:
                    if len(cur_sub) >= 4:
                        valid_groups.append(cur_sub)
                    cur_sub = [p]
            if len(cur_sub) >= 4:
                valid_groups.append(cur_sub)

        config_eg = EdgeGridConfig(
            roi_margin=45, vertical_mode="segmented", n_segments=4,
            max_x_jump_px=10.0, max_angle_delta_deg=8.0, search_half_width=10,
            min_points_per_segment=6, max_dev_baseline_px=12.0, smooth_window=3,
        )

        for grp in valid_groups:
            target_raw_indices = [p["raw_idx"] for p in grp]
            for p in grp:
                p["edge_grid_attempted"] = True

            res = propose_edge_grid_polygons_in_pipeline(
                orig_img=orig_img,
                panels=raw_panels,
                target_panel_indices_or_ids=target_raw_indices,
                config=config_eg,
            )

            # Check if boundaries were properly fitted (not all baseline fallback)
            dbg = res.get("debug", {})
            fitted = dbg.get("vertical_boundaries_fitted", [])
            too_much_fallback = not fitted or (len(fitted) >= 2 and fitted[0] < 1 and fitted[1] < 1)

            for prop in res.get("proposals", []):
                ridx = prop["raw_idx"]
                panel_obj = next((p for p in grp if p["raw_idx"] == ridx), None)
                if panel_obj is None:
                    continue

                m = prop["metrics"]
                panel_obj["edge_grid_area_ratio"]   = m["area_ratio"]
                panel_obj["edge_grid_center_shift"] = m["center_shift"]

                if too_much_fallback:
                    panel_obj["edge_grid_accept"]       = False
                    panel_obj["edge_grid_reject_reason"] = "vertical_boundary_too_much_fallback"
                    continue

                if not prop["ok"]:
                    panel_obj["edge_grid_accept"]       = False
                    panel_obj["edge_grid_reject_reason"] = f"proposal_failed_{prop.get('source','unknown')}"
                    continue

                poly_proposed = prop["polygon"]
                if len(poly_proposed) != 4:
                    panel_obj["edge_grid_accept"]       = False
                    panel_obj["edge_grid_reject_reason"] = "not_quadrilateral"
                    continue

                poly_np = np.array(poly_proposed, dtype=np.float32)
                if not cv2.isContourConvex(poly_np.reshape(-1, 1, 2).astype(np.int32)):
                    panel_obj["edge_grid_accept"]       = False
                    panel_obj["edge_grid_reject_reason"] = "non_convex"
                    continue

                if not (0.75 <= m["area_ratio"] <= 1.35):
                    panel_obj["edge_grid_accept"]       = False
                    panel_obj["edge_grid_reject_reason"] = "area_ratio_out_of_bounds"
                    continue

                if m["center_shift"] > 25.0:
                    panel_obj["edge_grid_accept"]       = False
                    panel_obj["edge_grid_reject_reason"] = "large_center_shift"
                    continue

                # Overlap check
                max_ov = 0.0
                for other in raw_panels:
                    if other["raw_idx"] == ridx:
                        continue
                    ov = _compute_polygon_overlap_ratio(
                        poly_proposed, other.get("polygon", []), orig_img.shape
                    )
                    max_ov = max(max_ov, ov)
                panel_obj["edge_grid_overlap"] = max_ov

                if max_ov > 0.05:
                    panel_obj["edge_grid_accept"]       = False
                    panel_obj["edge_grid_reject_reason"] = "high_overlap"
                    continue

                # Accept
                panel_obj["edge_grid_accept"] = True
                panel_obj["edge_grid_reject_reason"] = None
                panel_obj["polygon"] = [[int(round(pt[0])), int(round(pt[1]))] for pt in poly_proposed]
                feats = get_polygon_features(panel_obj["polygon"])
                panel_obj["area"]         = feats["area"]
                panel_obj["center"]       = feats["center"]
                panel_obj["aspect_ratio"] = feats["aspect_ratio"]
                panel_obj["final_polygon_source"]  = "edge_grid_segmented4"
                panel_obj["final_polygon_stage"]   = "edge_grid_refinement"
                panel_obj["final_polygon_reason"]  = "segmented_boundary_projection"

        # Debug image
        try:
            debug_dir = "data/results/debug"
            os.makedirs(debug_dir, exist_ok=True)
            dbg_img = orig_img.copy()
            for p in raw_panels:
                poly = p.get("polygon", [])
                if len(poly) >= 3:
                    pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
                    color = (0, 255, 255) if p.get("final_polygon_source") == "edge_grid_segmented4" else (0, 255, 0)
                    cv2.polylines(dbg_img, [pts], True, color, 2, cv2.LINE_AA)
                cx, cy = p.get("center", [0, 0])
                cv2.putText(dbg_img, str(p.get("raw_idx")), (int(cx) - 8, int(cy) + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1, cv2.LINE_AA)
            dbg_path = os.path.join(debug_dir, f"debug_{image_name}_edge_grid_final.JPG")
            cv2.imwrite(dbg_path, dbg_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            print(f"[EDGE_GRID_DEBUG] Saved to: {dbg_path}")
        except Exception as e:
            logger.warning(f"[EDGE_GRID_DEBUG] Could not save debug image: {e}")

    # -- Write panel refine debug JSONL --
    save_panel_refine_debug_jsonl(image_path, raw_panels)

    # -- Split touching hotspots with watershed (DISABLED FOR DISPLAY PRESERVATION) --
    split_defects = raw_defects # split_touching_hotspots_with_watershed(orig_img, raw_defects)

    # -- Deduplicate defects --
    deduped_defects = deduplicate_defects_by_polygon_iou(split_defects)

    # NOTE: assign_defects_to_panels and assign_row_col_ids are called
    # by main.py after this function returns. Return raw lists.
    return raw_panels + deduped_defects


# ===========================================================================
# SECTION 8 — ANNOTATION DRAWING
# ===========================================================================

def draw_custom_annotation(
    image_bgr: np.ndarray,
    panels: List[Dict[str, Any]],
    unassigned_defects: Optional[List[Dict[str, Any]]] = None,
) -> np.ndarray:
    """
    Draw panel polygons and defects onto a copy of image_bgr.

    Panel outlines: blue (255, 0, 0)
    Defects: class-specific color with 40% alpha fill + cyan outline
    Unassigned defects: red thick outline labelled [LOST]

    Returns annotated BGR image (same resolution as input).
    """
    img = image_bgr.copy()

    DEFECT_COLORS = {
        "hotspot_single_cell": (0, 100, 255),
        "hotspot_multi_cell":  (0, 0, 255),
        "shading":             (200, 200, 0),
        "soiling":             (0, 165, 255),
        "crack":               (200, 0, 200),
    }
    DEFAULT_DEFECT_COLOR = (100, 100, 255)

    for p in panels:
        # V62: ưu tiên outer_polygon (từ line-snap), fallback về polygon cũ
        p_poly = p.get("outer_polygon") or p.get("polygon", [])
        if len(p_poly) >= 3:
            pts = np.array(p_poly, dtype=np.int32)
            cv2.polylines(img, [pts], True, (255, 0, 0), 3, cv2.LINE_AA)  # thickness=3px
            cx, cy = p.get("center", [0, 0])
            label = p.get("local_id", p.get("class_name", "panel"))
            cv2.putText(img, label, (int(cx) - 20, int(cy)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1, cv2.LINE_AA)

        for d in p.get("defects", []):
            d_poly = d.get("polygon", [])
            cls    = d.get("class_name", "")
            color  = DEFECT_COLORS.get(cls, DEFAULT_DEFECT_COLOR)
            if len(d_poly) >= 3:
                pts = np.array(d_poly, dtype=np.int32)
                overlay = img.copy()
                cv2.fillPoly(overlay, [pts], color=color)
                img = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)
                cv2.polylines(img, [pts], True, (0, 255, 255), 2, cv2.LINE_AA)
                dcx, dcy = d.get("center", [0, 0])
                info = f"{cls} {d.get('area_ratio_percent', 0):.1f}%"
                cv2.putText(img, info, (int(dcx) - 30, int(dcy)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)

    if unassigned_defects:
        for d in unassigned_defects:
            d_poly = d.get("polygon", [])
            cls    = d.get("class_name", "")
            if len(d_poly) >= 3:
                pts = np.array(d_poly, dtype=np.int32)
                cv2.polylines(img, [pts], True, (0, 0, 255), 3, cv2.LINE_AA)
                dcx, dcy = d.get("center", [0, 0])
                cv2.putText(img, f"[LOST] {cls}", (int(dcx) - 30, int(dcy)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)

    return img
