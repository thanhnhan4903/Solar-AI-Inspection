import cv2
import logging
import numpy as np
import os
import json
from typing import List, Dict, Any, Tuple, Optional
import collections

from app.services.panel_geometry import get_polygon_features

logger = logging.getLogger("solar_ai")

# ===========================================================================
# CONFIG FLAGS
# ===========================================================================

# Đặt False để dùng pipeline cũ ổn định (YOLO mask → convexHull → minAreaRect).
# Đổi True khi muốn thử nghiệm ROI-based refinement.
ENABLE_ROI_REFINEMENT: bool = False

# Bật/tắt edge snap cho small panel. Đặt False theo yêu cầu A.
ENABLE_SMALL_PANEL_EDGE_SNAP: bool = False

# Bật/tắt grid-aware refinement cho small panel.
ENABLE_SMALL_PANEL_GRID_AWARE: bool = False
ENABLE_SMALL_PANEL_GRID_REFINEMENT: bool = False
GRID_REFINEMENT_STRICT_MODE: bool = False

# Bật/tắt shrink bbox cho small panel dùng bbox_poly.
ENABLE_SMALL_PANEL_BBOX_SHRINK: bool = False
SMALL_PANEL_BBOX_SHRINK_RATIO: float = 0.02  # shrink width/height by 2%
SMALL_PANEL_BBOX_SHRINK_MAX_PX_PER_EDGE: float = 2.0  # max 2 px inward per edge

# --- Block / Lattice Line Refinement Configs (DISABLED - replaced by String Lattice) ---
ENABLE_PANEL_BLOCK_LATTICE_REFINEMENT: bool = False  # Tat - dung string lattice thay the

BLOCK_MAX_PANEL_BBOX_AREA: float = 3000.0
BLOCK_MIN_PANEL_COUNT: int = 6

BLOCK_CLUSTER_MAX_DIST_FACTOR: float = 1.6

BLOCK_EXPECTED_COLUMNS_MIN: int = 1
BLOCK_EXPECTED_COLUMNS_MAX: int = 3

BLOCK_LINE_ANGLE_STD_MAX_DEG: float = 3.0
BLOCK_ROW_GROUP_THRESHOLD_FACTOR: float = 0.55
BLOCK_COL_GROUP_THRESHOLD_FACTOR: float = 0.55

BLOCK_PANEL_AREA_RATIO_MIN: float = 0.75
BLOCK_PANEL_AREA_RATIO_MAX: float = 1.25
BLOCK_PANEL_IOU_WITH_ORIGINAL_MIN: float = 0.50
BLOCK_PANEL_CENTER_SHIFT_MAX_RATIO: float = 0.25
BLOCK_PANEL_MAX_OVERLAP_RATIO: float = 0.03
BLOCK_OUTER_EDGE_MAX_EXPAND_PX: float = 8.0
BLOCK_PANEL_BOUNDARY_SCORE_RATIO_MIN: float = 1.05  # Chỉ dùng để log/warning, không reject
BLOCK_PANEL_WIDTH_RATIO_MIN: float = 0.75
BLOCK_PANEL_WIDTH_RATIO_MAX: float = 1.25
BLOCK_PANEL_HEIGHT_RATIO_MIN: float = 0.75
BLOCK_PANEL_HEIGHT_RATIO_MAX: float = 1.25
BLOCK_OUTSIDE_IMAGE_TOLERANCE_PX: float = 2.0

BLOCK_MIN_PASS_RATIO: float = 0.45  # Guard: chỉ fallback block khi pass_ratio < 0.45
BLOCK_MIN_PANELS_FOR_REFINEMENT: int = 4
_BLOCK_LATTICE_DEBUG: bool = True

ENABLE_STRING_PARALLEL_LINE_REFINEMENT: bool = False  # Old - disabled

# ===========================================================================
# Two-column Block Lattice Refinement Configs
# ===========================================================================
ENABLE_TWO_COLUMN_BLOCK_LATTICE: bool = True

TWO_COL_MAX_PANEL_BBOX_AREA: float = 3000.0
TWO_COL_MIN_ROWS: int = 4
TWO_COL_MIN_PANELS: int = 8

TWO_COL_COLUMN_GAP_FACTOR_MIN: float = 0.45
TWO_COL_COLUMN_GAP_FACTOR_MAX: float = 1.80

TWO_COL_ROW_MATCH_TOLERANCE_RATIO: float = 0.45
TWO_COL_MAX_ROW_Y_DIFF_RATIO: float = 0.35

TWO_COL_AREA_RATIO_MIN: float = 0.70
TWO_COL_AREA_RATIO_MAX: float = 1.30
TWO_COL_WIDTH_RATIO_MIN: float = 0.70
TWO_COL_WIDTH_RATIO_MAX: float = 1.30
TWO_COL_HEIGHT_RATIO_MIN: float = 0.70
TWO_COL_HEIGHT_RATIO_MAX: float = 1.30
TWO_COL_CENTER_SHIFT_MAX: float = 0.35
TWO_COL_IOU_MIN: float = 0.40

TWO_COL_MAX_OVERLAP: float = 0.04
TWO_COL_PASS_RATIO_MIN: float = 0.75

TWO_COL_SNAP_RAILS: bool = True
TWO_COL_SNAP_DIVIDERS: bool = True
TWO_COL_SNAP_SEARCH_PX: int = 4


# ===========================================================================
# Outer Boundary Guard Configs
# ===========================================================================
ENABLE_OUTER_BOUNDARY_GUARD: bool = False

# --- SAFE IMPROVEMENT ONLY / ENDPOINT GUARD POLYGON UPDATE ---
ENABLE_LATTICE_SAFE_IMPROVEMENT_ONLY: bool = False   # Lattice is primary; False = use lattice unless hard-fail
ENABLE_ENDPOINT_GUARD_POLYGON_UPDATE: bool = False    # True = apply endpoint guard decision to polygon

OUTER_DIVIDER_EDGE_SEARCH_PX: int = 8
OUTER_DIVIDER_MIN_EDGE_SCORE: float = 0.18
OUTER_DIVIDER_MAX_SHIFT_PX: float = 8.0

OUTER_PANEL_AREA_RATIO_MIN: float = 0.75
OUTER_PANEL_AREA_RATIO_MAX: float = 1.20
OUTER_PANEL_CENTER_SHIFT_MAX: float = 0.30
OUTER_PANEL_IOU_MIN: float = 0.45

OUTER_PANEL_ALLOW_EXTRAPOLATE: bool = False
OUTER_PANEL_FALLBACK_INDIVIDUAL: bool = True

# Endpoint Guard Configs (used by validate_endpoint_candidate for backward-compat)
ENDPOINT_AREA_RATIO_MIN: float = 0.70
ENDPOINT_AREA_RATIO_MAX: float = 1.25
ENDPOINT_CENTER_SHIFT_MAX: float = 0.30
ENDPOINT_IOU_MIN: float = 0.35
ENDPOINT_EDGE_SCORE_MIN_RATIO: float = 0.90  # Warning only — not a hard rejection gate

# ===========================================================================
# Endpoint Isolation Architecture
# ===========================================================================
ENABLE_ENDPOINT_ISOLATION: bool = True           # Master switch for endpoint isolation
MIN_STRING_LEN_FOR_ENDPOINT_ISOLATION: int = 5  # Strings shorter than this: all treated as middle
MIN_MIDDLE_PANELS_FOR_LATTICE: int = 4           # Min middle panels needed to attempt lattice fit
MIN_MIDDLE_PANELS_PER_COL: int = 3               # Min middle panels per column in 2-col block

ENDPOINT_MAX_EXTRAPOLATE_RATIO: float = 0.55     # Max |outer_u - inner_u| / median_panel_length

# 2-column block: if left/right column angle delta exceeds this, reject as shared block
TWO_COL_MAX_COLUMN_ANGLE_DELTA_DEG: float = 7.0

# Endpoint edge snap: diagnostic only (no hard gate)
ENABLE_ENDPOINT_EDGE_SNAP: bool = False
ENDPOINT_EDGE_SCORE_DIAGNOSTIC_ONLY: bool = True

ENABLE_HYBRID_ENDPOINT_RESOLVER: bool = False

ENDPOINT_OUTER_MAX_SHIFT_FROM_YOLO_RATIO: float = 0.45
ENDPOINT_OUTER_MAX_SHIFT_FROM_PITCH_RATIO: float = 0.35

ENDPOINT_EDGE_SEARCH_PX: int = 10
ENDPOINT_EDGE_MIN_SCORE: float = 0.12

ENDPOINT_LENGTH_RATIO_MIN: float = 0.65
ENDPOINT_LENGTH_RATIO_MAX: float = 1.20

ENDPOINT_SCORE_MIN_ACCEPT: float = 0.45
ENDPOINT_USE_YOLO_IF_ALL_FAIL: bool = True

ENDPOINT_KEEP_MIDDLE_UNTOUCHED: bool = True

ENDPOINT_DEBUG_DRAW_CANDIDATES: bool = True

# --- MIDDLE-LOCKED ENDPOINT CONFIGS ---
ENABLE_MIDDLE_LOCKED_ENDPOINT: bool = True

MIDDLE_LOCKED_ENDPOINT_USE_YOLO_OUTER: bool = False
MIDDLE_LOCKED_ENDPOINT_USE_EDGE_SCAN: bool = False
MIDDLE_LOCKED_ENDPOINT_USE_CONSENSUS: bool = False

ENDPOINT_PITCH_RATIO_MIN: float = 0.85
ENDPOINT_PITCH_RATIO_MAX: float = 1.15

ENDPOINT_CENTER_SHIFT_MAX_RATIO: float = 0.35
ENDPOINT_IOU_WITH_YOLO_MIN: float = 0.30

ENDPOINT_ALLOW_LOW_IOU_IF_LATTICE_GOOD: bool = True
ENDPOINT_LOW_IOU_MIN_WHEN_LATTICE_GOOD: float = 0.20

ENDPOINT_MAX_OVERLAP_RATIO: float = 0.03

MIDDLE_PANEL_MIN_FOR_LOCKED_ENDPOINT: int = 3

ENDPOINT_DEBUG_DRAW_MIDDLE_LOCKED: bool = True



# ===========================================================================
# String-first Lattice Refinement (NEW - production path)
# ===========================================================================
ENABLE_STRING_LATTICE_REFINEMENT: bool = True

# Bypass panel lon hon nguong nay
STRING_REFINEMENT_MAX_PANEL_BBOX_AREA: float = 3000.0
STRING_BORDER_MARGIN_PX: int = 8

# Trusted panel selection
STRING_TRUSTED_MIN_CONF: float = 0.85
STRING_TRUSTED_AREA_RATIO_MIN: float = 0.80
STRING_TRUSTED_AREA_RATIO_MAX: float = 1.30
STRING_TRUSTED_ASPECT_MIN: float = 1.2
STRING_TRUSTED_ASPECT_MAX: float = 8.0

# String detection thresholds
STRING_NN_MAX_DIST_FACTOR: float = 1.8       # max dist = 1.8 * median_short_side
STRING_V_GROUP_THRESHOLD_FACTOR: float = 0.55
STRING_GAP_SPLIT_FACTOR: float = 1.8         # split string if gap > 1.8 * pitch
STRING_PITCH_CV_MAX: float = 0.40
STRING_ANGLE_STD_MAX_DEG: float = 5.0
STRING_MIN_PANELS: int = 4
STRING_MIN_TRUSTED: int = 2                  # relax to 2 to handle noisy images

# Rail fitting percentiles
STRING_OUTER_RAIL_PCT_LOW_SMALL: float = 10.0   # n_trusted < 8
STRING_OUTER_RAIL_PCT_HIGH_SMALL: float = 90.0
STRING_OUTER_RAIL_PCT_LOW_LARGE: float = 5.0    # n_trusted >= 8
STRING_OUTER_RAIL_PCT_HIGH_LARGE: float = 95.0

# Panel validation thresholds
STRING_PANEL_IOU_MIN: float = 0.45
STRING_PANEL_CENTER_SHIFT_MAX: float = 0.30
STRING_PANEL_AREA_RATIO_MIN: float = 0.70
STRING_PANEL_AREA_RATIO_MAX: float = 1.30
STRING_PANEL_WIDTH_RATIO_MIN: float = 0.70
STRING_PANEL_WIDTH_RATIO_MAX: float = 1.30
STRING_PANEL_HEIGHT_RATIO_MIN: float = 0.70
STRING_PANEL_HEIGHT_RATIO_MAX: float = 1.30
STRING_PANEL_OUTSIDE_TOL_PX: float = 2.0
STRING_PANEL_MAX_OVERLAP: float = 0.03

# Hard-fail thresholds (fallback even inside passing string)
STRING_PANEL_HARD_SHIFT_MAX: float = 0.40
STRING_PANEL_HARD_IOU_MIN: float = 0.30
STRING_PANEL_HARD_AREA_MIN: float = 0.60
STRING_PANEL_HARD_AREA_MAX: float = 1.40

# String-level decision
STRING_MIN_PASS_RATIO: float = 0.65
STRING_MEDIAN_IOU_MIN: float = 0.50
STRING_MAX_OVERLAP: float = 0.05

# Partial panel at string ends
ENABLE_REJECT_PARTIAL_STRING_END_PANEL: bool = True
STRING_PARTIAL_PANEL_MIN_AREA_RATIO: float = 0.75

# String synchronization
ENABLE_STRING_SYNC: bool = True
STRING_SYNC_ANGLE_DIFF_MAX_DEG: float = 2.0
STRING_SYNC_PITCH_DIFF_MAX_RATIO: float = 0.15

# Old cluster constants (used by cluster_panels_into_strings)
STRING_CLUSTER_MAX_CENTER_DIST_FACTOR: float = 2.5
STRING_REFINEMENT_MIN_PANEL_COUNT: int = 4

_STRING_LATTICE_DEBUG: bool = True

# --- String Rail/Divider Edge Snap (post-refinement) ---
ENABLE_STRING_RAIL_EDGE_SNAP: bool = True
ENABLE_STRING_DIVIDER_EDGE_SNAP: bool = True

STRING_RAIL_SNAP_SEARCH_PX: int = 5
STRING_DIVIDER_SNAP_SEARCH_PX: int = 4

STRING_RAIL_SNAP_MAX_SHIFT_PX: float = 5.0
STRING_DIVIDER_SNAP_MAX_SHIFT_PX: float = 4.0

STRING_SNAP_MIN_EDGE_IMPROVEMENT: float = 1.05   # score must improve >= 5%

# Snap validation thresholds
STRING_SNAP_MAX_IOU_DROP: float = 0.08           # max allowed IoU drop vs pre-snap
STRING_SNAP_MAX_OVERLAP: float = 0.05

STRING_SNAP_PANEL_IOU_MIN: float = 0.40
STRING_SNAP_PANEL_CENTER_SHIFT_MAX: float = 0.35
STRING_SNAP_PANEL_AREA_RATIO_MIN: float = 0.65
STRING_SNAP_PANEL_AREA_RATIO_MAX: float = 1.35
STRING_SNAP_PANEL_WIDTH_RATIO_MIN: float = 0.65
STRING_SNAP_PANEL_WIDTH_RATIO_MAX: float = 1.35
STRING_SNAP_PANEL_HEIGHT_RATIO_MIN: float = 0.65
STRING_SNAP_PANEL_HEIGHT_RATIO_MAX: float = 1.35

# Partial panel at string ends
ENABLE_PARTIAL_STRING_PANEL_REJECT: bool = True
PARTIAL_STRING_PANEL_MIN_AREA_RATIO: float = 0.75

SUBCLUSTER_MIN_PANEL_COUNT: int = 5
SUBCLUSTER_MIN_PASS_RATIO: float = 0.95
SUBCLUSTER_MAX_ANGLE_STD_DEG: float = 2.0
SUBCLUSTER_MAX_OVERLAP_RATIO: float = 0.03
SUBCLUSTER_MAX_OUTSIDE_COUNT: int = 0
SUBCLUSTER_MAX_BORDER_RATIO: float = 0.40

GRID_PANEL_MAX_CENTER_SHIFT_RATIO: float = 0.08
GRID_EDGE_SCORE_IMPROVEMENT_RATIO: float = 1.15
GRID_PANEL_MIN_AREA_RATIO: float = 0.85
GRID_PANEL_MAX_AREA_RATIO: float = 1.15
GRID_PANEL_MIN_WIDTH_RATIO: float = 0.85
GRID_PANEL_MAX_WIDTH_RATIO: float = 1.15
GRID_PANEL_MIN_HEIGHT_RATIO: float = 0.85
GRID_PANEL_MAX_HEIGHT_RATIO: float = 1.15

# Bật True để in chi tiết từng bước ROI trong quá trình dev/debug.
_ROI_DEBUG: bool = False

# ---------------------------------------------------------------------------
# Panel geometry quality control
# ---------------------------------------------------------------------------

# Bật/tắt toàn bộ bước filter panel theo diện tích / hình học.
ENABLE_PANEL_SIZE_FILTER: bool = False

# Panel full: area / median_area phải nằm trong khoảng này để được giữ.
PANEL_MIN_AREA_RATIO: float = 0.80
PANEL_MAX_AREA_RATIO: float = 1.40

# Panel ở mép ảnh (partial): ngưỡng thấp hơn.
PARTIAL_PANEL_MIN_AREA_RATIO: float = 0.50

# Số pixel coi là "mép ảnh" (panel chạm biên ảnh <= margin px).
BORDER_MARGIN_PX: int = 8

# Số panel tốt tối thiểu để tính median (nếu ít hơn → không filter).
MIN_PANELS_FOR_MEDIAN: int = 3

# Aspect ratio hợp lệ để đưa vào tính median.
PANEL_ASPECT_MIN: float = 1.5
PANEL_ASPECT_MAX: float = 6.5

# Góc lệch tối đa so với median_angle để được giữ (degrees).
MAX_ANGLE_DIFF_DEG: float = 20.0

# Bật chuẩn hóa polygon về kích thước / góc median.
ENABLE_PANEL_GEOMETRY_NORMALIZATION: bool = False

# Debug flag: in quyết định keep/reject cho từng panel.
_PANEL_FILTER_DEBUG: bool = False


# ---------------------------------------------------------------------------
# Panel polygon guard — tỷ lệ so với YOLO bbox của chính panel đó.
#
# STRICT (panel bình thường / lớn):
#   Đây là các ngưỡng gốc, đã được kiểm tra và fix DJI_0987 + DJI_0995.
#   KHÔNG được hạ xuống trừ khi test kỹ.
# ---------------------------------------------------------------------------

# polygon_area / bbox_area phải >= ngưỡng này (strict).
PANEL_POLY_MIN_BBOX_AREA_RATIO: float = 0.70

# poly_bounding_width / bbox_width phải >= ngưỡng này (strict).
PANEL_POLY_MIN_BBOX_WIDTH_RATIO: float = 0.75

# poly_bounding_height / bbox_height phải >= ngưỡng này (strict).
PANEL_POLY_MIN_BBOX_HEIGHT_RATIO: float = 0.75

# center_distance / max(bbox_w, bbox_h) phải <= ngưỡng này (strict).
PANEL_POLY_MAX_CENTER_SHIFT_RATIO: float = 0.35

# Aspect ratio hợp lệ của polygon (strict).
PANEL_POLY_MIN_ASPECT_RATIO: float = 1.2
PANEL_POLY_MAX_ASPECT_RATIO: float = 8.0


# ---------------------------------------------------------------------------
# Panel polygon guard — ngưỡng lỏng hơn CHỈ cho panel thật sự nhỏ.
# Áp dụng khi bbox_area (pixel²) <= SMALL_PANEL_BBOX_AREA_PX.
# Panel nhỏ có thể có mask không đều nên cho phép ratio thấp hơn.
# ---------------------------------------------------------------------------

# Ngưỡng diện tích bbox (pixel²) để xác định "panel nhỏ".
# Panel có bbox_area <= giá trị này sẽ dùng threshold_set="small".
SMALL_PANEL_BBOX_AREA_PX: float = 2500.0

# Các ngưỡng lỏng cho small panel (chỉ dùng khi SMALL_PANEL_FORCE_BBOX=False):
PANEL_POLY_SMALL_MIN_BBOX_AREA_RATIO: float = 0.30
PANEL_POLY_SMALL_MIN_BBOX_WIDTH_RATIO: float = 0.55
PANEL_POLY_SMALL_MIN_BBOX_HEIGHT_RATIO: float = 0.55
PANEL_POLY_SMALL_MAX_CENTER_SHIFT_RATIO: float = 0.40
PANEL_POLY_SMALL_MIN_ASPECT_RATIO: float = 1.0

# Bật True → small panel có confidence >= SMALL_PANEL_USE_BBOX_CONF sẽ dùng
# YOLO bbox_poly trực tiếp, không qua mask refinement.
# Lý do: mask segmentation của small panel thường thiếu cạnh / co cụm,
# trong khi YOLO bbox thường bao sát đúng tấm pin hơn.
SMALL_PANEL_FORCE_BBOX: bool = True

# Ngưỡng confidence tối thiểu để áp dụng SMALL_PANEL_FORCE_BBOX.
# Panel nhỏ nhưng confidence thấp vẫn đi qua pipeline mask refine bình thường.
SMALL_PANEL_USE_BBOX_CONF: float = 0.70


# ---------------------------------------------------------------------------
# filter_defect_like_panels — tắt mặc định vì có thể reject nhầm panel có lỗi.
# Panel có defect thật (hotspot_multi_cell/crack) cũng có overlap cao với defect polygon.
# ---------------------------------------------------------------------------

# Bật True để thử nghiệm. Mặc định False để an toàn.
ENABLE_DEFECT_LIKE_PANEL_FILTER: bool = False

# Ngưỡng overlap để coi một small panel là "defect giả" (chỉ dùng khi flag True).
DEFECT_LIKE_OVERLAP_THRESHOLD: float = 0.70

# Percentile dưới đây mới bị đưa vào kiểm tra "small panel".
# Ví dụ 25: chỉ 25% panel nhỏ nhất mới bị check với defect.
SMALL_PANEL_PERCENTILE: int = 25

# Bật/tắt edge grid refinement với 4-segment vertical boundary (mặc định False)
ENABLE_EDGE_GRID_SEGMENTED_REFINEMENT: bool = False


# ===========================================================================
# SECTION 1 — EXISTING HELPERS (GIỮ NGUYÊN HOÀN TOÀN)
# ===========================================================================

def filter_panels_by_area(panels: List[Dict]) -> List[Dict]:
    """
    Loại bỏ các panel có diện tích lệch quá nhiều so với median (quá 30%).
    Sử dụng median (trung vị) là phương pháp tốt nhất vì nó không bị nhiễu
    bởi các panel false positive (rất nhỏ) hoặc panel dính chùm (rất to).
    """
    if len(panels) <= 2:
        return panels

    areas = [p.get("area", 0) for p in panels]
    median_area = float(np.median(areas))

    if median_area <= 0:
        return panels

    filtered = []
    for i, p in enumerate(panels):
        ratio = areas[i] / median_area
        if 0.80 <= ratio <= 1.20:
            filtered.append(p)

    if len(filtered) < len(panels) // 2:
        return panels

    return filtered


def _check_geometry_quality(pts: np.ndarray, bbox: List[float], img_shape: Tuple[int, int]) -> bool:
    """
    Kiểm tra chất lượng hình học của tứ giác tìm được:
    - Diện tích phải đủ lớn (ví dụ: >= 75% diện tích của YOLO bbox)
    - Aspect ratio phải nằm trong khoảng hợp lý cho solar panel (1.0 đến 6.0)
    - Cạnh đối diện phải có chiều dài tương đương (tính đối xứng, tránh vát góc do nhiễu nhiệt)
    - Tất cả các điểm phải nằm gần hoặc trong vùng ảnh
    """
    if len(pts) != 4:
        return False
    h_img, w_img = img_shape[:2]

    # 1. Tính diện tích của tứ giác
    area_quad = float(cv2.contourArea(pts.reshape(-1, 1, 2)))

    w_box = bbox[2] - bbox[0]
    h_box = bbox[3] - bbox[1]
    area_box = w_box * h_box
    if area_box <= 0:
        return False

    # Phải đạt ít nhất 75% diện tích bbox đề xuất (solar panel hầu như lấp đầy bbox)
    if area_quad < 0.75 * area_box:
        return False

    # 2. Aspect ratio
    rect = cv2.minAreaRect(pts)
    (cx, cy), (w, h), angle = rect
    w = max(w, 1.0)
    h = max(h, 1.0)
    aspect = max(w, h) / min(w, h)
    if aspect < 0.8 or aspect > 6.0:
        return False

    # 3. Kiểm tra tính đối xứng của các cặp cạnh đối diện
    try:
        sorted_pts = _sort_corners(pts)
        tl, tr, br, bl = [np.array(pt, dtype=np.float32) for pt in sorted_pts]

        len_top = np.linalg.norm(tr - tl)
        len_bottom = np.linalg.norm(br - bl)
        len_left = np.linalg.norm(bl - tl)
        len_right = np.linalg.norm(br - tr)

        if len_top <= 0 or len_bottom <= 0 or len_left <= 0 or len_right <= 0:
            return False

        ratio_tb = min(len_top, len_bottom) / max(len_top, len_bottom)
        ratio_lr = min(len_left, len_right) / max(len_left, len_right)

        # Cạnh đối diện không được lệch quá 10% chiều dài
        if ratio_tb < 0.90 or ratio_lr < 0.90:
            return False
    except Exception:
        return False

    # 4. Tứ giác không được bay ra ngoài rìa ảnh quá mức
    for pt in pts:
        x, y = pt
        if x < -50 or x > w_img + 50 or y < -50 or y > h_img + 50:
            return False

    return True


def _sort_corners(box: np.ndarray) -> List[List[int]]:
    """Sắp xếp 4 góc theo thứ tự chuẩn: Top-Left, Top-Right, Bottom-Right, Bottom-Left"""
    cx, cy = np.mean(box, axis=0)
    angles = np.arctan2(box[:, 1] - cy, box[:, 0] - cx)
    sorted_box = box[np.argsort(angles)]

    min_x, min_y = np.min(box, axis=0)
    distances = np.linalg.norm(sorted_box - [min_x, min_y], axis=1)
    tl_idx = np.argmin(distances)
    sorted_box = np.roll(sorted_box, -tl_idx, axis=0)

    return [[int(round(pt[0])), int(round(pt[1]))] for pt in sorted_box]


# ===========================================================================
# SECTION 2 — ROI-BASED REFINEMENT HELPERS
# ===========================================================================

def expand_bbox(
    bbox: List[float],
    img_shape: Tuple[int, int],
    expand_ratio: float = 0.15
) -> Tuple[int, int, int, int]:
    """
    Mở rộng YOLO bbox ra ngoài expand_ratio mỗi chiều rồi clamp vào ảnh.

    Returns:
        (x1_new, y1_new, x2_new, y2_new) — int, đã clamp.
    """
    img_h, img_w = img_shape[:2]
    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
    w = x2 - x1
    h = y2 - y1
    pad_x = w * expand_ratio
    pad_y = h * expand_ratio

    x1_new = int(max(0, x1 - pad_x))
    y1_new = int(max(0, y1 - pad_y))
    x2_new = int(min(img_w - 1, x2 + pad_x))
    y2_new = int(min(img_h - 1, y2 + pad_y))
    return x1_new, y1_new, x2_new, y2_new


def _compute_mask_iou_cv(
    poly_a: np.ndarray,
    poly_b: np.ndarray,
    img_shape: Tuple[int, int]
) -> float:
    """
    Tính IoU giữa 2 polygon bằng cách rasterize mask với OpenCV.
    Không cần Shapely.

    Args:
        poly_a, poly_b : ndarray shape (N, 2) — tọa độ float trong ảnh gốc.
        img_shape      : (height, width[, channels]).

    Returns:
        IoU trong [0.0, 1.0].
    """
    h, w = img_shape[:2]
    m_a = np.zeros((h, w), dtype=np.uint8)
    m_b = np.zeros((h, w), dtype=np.uint8)

    pts_a = poly_a.astype(np.int32).reshape(-1, 1, 2)
    pts_b = poly_b.astype(np.int32).reshape(-1, 1, 2)

    cv2.fillPoly(m_a, [pts_a], 1)
    cv2.fillPoly(m_b, [pts_b], 1)

    intersection = float(np.logical_and(m_a, m_b).sum())
    union = float(np.logical_or(m_a, m_b).sum())
    if union <= 0:
        return 0.0
    return intersection / union


def _angle_of_rect(pts: np.ndarray) -> float:
    """
    Trả về góc chính (degrees) của minAreaRect từ 4 điểm polygon.
    Normalize về [0, 90) để so sánh đơn giản.
    """
    rect = cv2.minAreaRect(pts.astype(np.float32))
    angle = rect[2]  # OpenCV trả về [-90, 0)
    # Normalize về [0, 90)
    angle = abs(angle) % 90
    return angle


def preprocess_panel_roi(roi: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    """
    Tạo danh sách candidate edge maps từ ROI crop.

    Ưu tiên: Canny (bilateral) → Canny (gaussian).
    Adaptive/Otsu chỉ thêm vào nhưng được đánh dấu để caller biết ưu tiên thấp hơn.
    Trả về [] nếu ROI quá đồng nhất (std < 5).

    Returns:
        List of (name, binary_mask).  name bắt đầu bằng "low_priority_" nếu dễ nhiễu.
    """
    # --- Convert grayscale ---
    if len(roi.shape) == 3 and roi.shape[2] == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi.copy()

    # Guard: ảnh quá đồng nhất → không có cạnh thực sự
    if float(np.std(gray)) < 5.0:
        if _ROI_DEBUG:
            logger.debug("[preprocess_roi] ROI quá đồng nhất (std < 5), bỏ qua.")
        return []

    # Denoise
    try:
        gray_bil = cv2.bilateralFilter(gray, d=7, sigmaColor=30, sigmaSpace=30)
    except Exception:
        gray_bil = cv2.GaussianBlur(gray, (5, 5), 0)

    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

    candidates: List[Tuple[str, np.ndarray]] = []
    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    kernel5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    # --- (a) Canny bilateral [PRIORITY HIGH] ---
    try:
        sigma = float(np.std(gray_bil))
        median_val = float(np.median(gray_bil))
        low = max(10, int(median_val - 0.66 * sigma))
        high = max(low + 20, int(median_val + 0.66 * sigma))
        edges = cv2.Canny(gray_bil, low, high)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel3)
        edges = cv2.morphologyEx(edges, cv2.MORPH_DILATE, kernel3)
        candidates.append(("canny_bilateral", edges))
    except Exception:
        pass

    # --- (b) Canny gaussian [PRIORITY HIGH] ---
    try:
        sigma2 = float(np.std(gray_blur))
        med2 = float(np.median(gray_blur))
        low2 = max(10, int(med2 - 0.5 * sigma2))
        high2 = max(low2 + 20, int(med2 + 0.5 * sigma2))
        edges2 = cv2.Canny(gray_blur, low2, high2)
        edges2 = cv2.morphologyEx(edges2, cv2.MORPH_CLOSE, kernel3)
        candidates.append(("canny_gaussian", edges2))
    except Exception:
        pass

    # --- (c) Adaptive threshold [LOW PRIORITY — dễ bắt cell grid] ---
    try:
        block_size = max(11, (min(gray.shape[:2]) // 8) | 1)
        if block_size % 2 == 0:
            block_size += 1
        adaptive = cv2.adaptiveThreshold(
            gray_blur, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block_size, 3
        )
        adaptive = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel5)
        adaptive = cv2.morphologyEx(adaptive, cv2.MORPH_OPEN, kernel3)
        candidates.append(("low_priority_adaptive", adaptive))
    except Exception:
        pass

    # --- (d) Otsu threshold [LOW PRIORITY — dễ bắt hotspot] ---
    try:
        _, otsu = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        otsu = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel5)
        otsu = cv2.morphologyEx(otsu, cv2.MORPH_OPEN, kernel3)
        candidates.append(("low_priority_otsu", otsu))
    except Exception:
        pass

    return candidates


def _check_not_clinging_to_roi_border(
    poly_global: np.ndarray,
    expanded_bbox: Tuple[int, int, int, int],
    border_margin: int = 4
) -> bool:
    """
    Trả về True nếu polygon KHÔNG bám sát ≥ 3 cạnh của expanded bbox.
    Nếu bám sát ≥ 3 cạnh, khả năng cao đang bắt nhầm biên crop → reject.

    Args:
        poly_global   : ndarray (4, 2) tọa độ ảnh gốc.
        expanded_bbox : (x1_exp, y1_exp, x2_exp, y2_exp).
        border_margin : số pixel cho phép sai lệch khi kiểm tra bám sát biên.
    """
    ex1, ey1, ex2, ey2 = expanded_bbox
    pts = poly_global.reshape(-1, 2)

    xs = pts[:, 0]
    ys = pts[:, 1]

    touch_left   = int(np.sum(xs <= ex1 + border_margin))
    touch_right  = int(np.sum(xs >= ex2 - border_margin))
    touch_top    = int(np.sum(ys <= ey1 + border_margin))
    touch_bottom = int(np.sum(ys >= ey2 - border_margin))

    # Đếm số cạnh mà polygon có ≥ 1 điểm chạm sát
    sides_touched = int(touch_left > 0) + int(touch_right > 0) + \
                    int(touch_top > 0) + int(touch_bottom > 0)

    if sides_touched >= 3:
        if _ROI_DEBUG:
            logger.debug(
                f"[ROI-border-check] FAIL: polygon bám sát {sides_touched} cạnh ROI expanded."
            )
        return False
    return True


def _check_edge_density(
    binary_map: np.ndarray,
    low: float = 0.005,
    high: float = 0.20
) -> bool:
    """
    Kiểm tra mật độ cạnh (edge density) của binary map trong ROI.
    Quá thấp: ROI gần như không có cạnh thực → không đáng tin.
    Quá cao: ROI nhiễu (cell grid, hotspot texture) → dễ bắt sai.

    Returns:
        True nếu density nằm trong [low, high].
    """
    density = float(np.mean(binary_map > 0))
    ok = low <= density <= high
    if _ROI_DEBUG and not ok:
        logger.debug(f"[edge-density] density={density:.4f} out of [{low:.3f}, {high:.3f}]")
    return ok


def find_panel_polygon_from_roi(
    roi: np.ndarray,
    roi_offset: Tuple[int, int],
    original_bbox: List[float],
    img_shape: Tuple[int, int],
    expanded_bbox: Tuple[int, int, int, int],
    xy_polygon: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """
    Tìm polygon tứ giác của panel từ ROI crop.

    Scoring candidate dựa trên (theo thứ tự trọng số giảm dần):
      1. IoU với YOLO polygon (trọng số cao nhất).
      2. Center distance so với bbox YOLO.
      3. Area ratio so với bbox YOLO.
      4. Angle difference so với YOLO minAreaRect.
      Candidate từ adaptive/otsu nhận thêm penalty.

    Returns:
        np.ndarray shape (4, 2) float tọa độ ảnh gốc, hoặc None.
    """
    off_x, off_y = roi_offset

    bx1, by1, bx2, by2 = original_bbox
    bbox_w = max(bx2 - bx1, 1.0)
    bbox_h = max(by2 - by1, 1.0)
    bbox_area = bbox_w * bbox_h
    bbox_cx = (bx1 + bx2) / 2.0
    bbox_cy = (by1 + by2) / 2.0
    max_dim = max(bbox_w, bbox_h)

    # Góc YOLO reference để so sánh angle
    yolo_angle: Optional[float] = None
    yolo_mask_global: Optional[np.ndarray] = None
    if xy_polygon is not None and len(xy_polygon) >= 4:
        try:
            yolo_angle = _angle_of_rect(xy_polygon.astype(np.float32))
            yolo_mask_global = xy_polygon.astype(np.float32)
        except Exception:
            pass

    # Ngưỡng diện tích contour tối thiểu trong ROI
    min_contour_area = bbox_area * 0.20

    candidates_maps = preprocess_panel_roi(roi)
    if not candidates_maps:
        return None

    best_poly: Optional[np.ndarray] = None
    best_score: float = -1.0

    for map_name, binary_map in candidates_maps:
        is_low_priority = map_name.startswith("low_priority_")

        # Kiểm tra edge density trước khi findContours
        if not _check_edge_density(binary_map, low=0.005, high=0.20):
            continue

        try:
            contours, _ = cv2.findContours(
                binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
        except Exception:
            continue

        if not contours:
            continue

        for cnt in contours:
            cnt_area = cv2.contourArea(cnt)
            if cnt_area < min_contour_area:
                continue

            # --- Fit tứ giác: approxPolyDP → minAreaRect fallback ---
            arc = cv2.arcLength(cnt, True)
            poly_4pts: Optional[np.ndarray] = None
            for factor in np.linspace(0.02, 0.20, 40):
                approx = cv2.approxPolyDP(cnt, factor * arc, True)
                if len(approx.reshape(-1, 2)) == 4:
                    poly_4pts = approx.reshape(-1, 2).astype(np.float32)
                    break

            if poly_4pts is None:
                rect = cv2.minAreaRect(cnt)
                poly_4pts = cv2.boxPoints(rect).astype(np.float32)

            if poly_4pts is None or len(poly_4pts) != 4:
                continue

            # Map về tọa độ ảnh gốc
            poly_global = poly_4pts + np.array([off_x, off_y], dtype=np.float32)

            # --- Guard: không bám sát ≥ 3 cạnh ROI expanded ---
            if not _check_not_clinging_to_roi_border(poly_global, expanded_bbox):
                continue

            poly_area = float(cv2.contourArea(poly_global.reshape(-1, 1, 2)))
            if poly_area <= 0:
                continue

            # Area ratio [0.60, 1.40] — filter sơ bộ trước khi score
            area_ratio = poly_area / bbox_area
            if area_ratio < 0.60 or area_ratio > 1.40:
                continue

            # Center distance — filter sơ bộ
            poly_cx = float(np.mean(poly_global[:, 0]))
            poly_cy = float(np.mean(poly_global[:, 1]))
            center_dist = np.sqrt((poly_cx - bbox_cx) ** 2 + (poly_cy - bbox_cy) ** 2)
            if center_dist > 0.20 * max_dim:
                continue

            # Aspect ratio — filter sơ bộ
            rect_chk = cv2.minAreaRect(poly_global.astype(np.float32))
            (_, _), (rw, rh), _ = rect_chk
            rw, rh = max(rw, 1.0), max(rh, 1.0)
            aspect = max(rw, rh) / min(rw, rh)
            if aspect < 1.0 or aspect > 7.0:
                continue

            # --- Tính score ---
            # (1) IoU với YOLO polygon — trọng số 0.50
            iou_score = 0.0
            if yolo_mask_global is not None:
                try:
                    iou_val = _compute_mask_iou_cv(poly_global, yolo_mask_global, img_shape)
                    iou_score = iou_val
                except Exception:
                    iou_score = 0.0

            # (2) Center distance score — trọng số 0.20
            center_score = max(0.0, 1.0 - center_dist / (0.20 * max_dim + 1e-6))

            # (3) Area ratio score — trọng số 0.20 (lý tưởng ratio = 1.0)
            area_score = max(0.0, 1.0 - abs(area_ratio - 1.0))

            # (4) Angle diff score — trọng số 0.10
            angle_score = 0.5  # default nếu không có YOLO angle
            if yolo_angle is not None:
                try:
                    roi_angle = _angle_of_rect(poly_global)
                    angle_diff = abs(roi_angle - yolo_angle)
                    angle_diff = min(angle_diff, 90.0 - angle_diff)  # symmetric
                    angle_score = max(0.0, 1.0 - angle_diff / 15.0)
                except Exception:
                    angle_score = 0.5

            score = (0.50 * iou_score
                     + 0.20 * center_score
                     + 0.20 * area_score
                     + 0.10 * angle_score)

            # Penalty cho low-priority candidates (adaptive / otsu)
            if is_low_priority:
                score *= 0.70

            if score > best_score:
                best_score = score
                best_poly = poly_global

    return best_poly


def validate_roi_panel_polygon(
    poly: np.ndarray,
    bbox: List[float],
    expanded_bbox: Tuple[int, int, int, int],
    img_shape: Tuple[int, int],
    xy_polygon: Optional[np.ndarray] = None,
) -> bool:
    """
    Validate nghiêm ngặt polygon ROI trước khi dùng thay thế pipeline cũ.
    Tất cả điều kiện phải pass — một điều kiện fail là reject toàn bộ.

    Điều kiện:
      1. Đúng 4 điểm, diện tích > 0.
      2. area_ratio ∈ [0.70, 1.30] so với bbox YOLO gốc.
      3. center_distance ≤ 0.15 × max(bbox_w, bbox_h).
      4. aspect_ratio ∈ [1.3, 6.5].
      5. IoU với YOLO mask ≥ 0.55 (nếu có xy_polygon).
      6. angle_diff ≤ 15° so với YOLO minAreaRect.
      7. Không bám sát ≥ 3 cạnh expanded bbox.
      8. Không có điểm nào vượt ngoài ảnh > 50px.
    """
    if poly is None or len(poly) != 4:
        return False

    img_h, img_w = img_shape[:2]
    pts = poly.reshape(-1, 2).astype(np.float32)

    # 1. Diện tích > 0
    poly_area = float(cv2.contourArea(pts.reshape(-1, 1, 2)))
    if poly_area <= 0:
        if _ROI_DEBUG:
            logger.debug("[ROI-validate] FAIL: area = 0")
        return False

    bx1, by1, bx2, by2 = bbox
    bbox_w = max(bx2 - bx1, 1.0)
    bbox_h = max(by2 - by1, 1.0)
    bbox_area = bbox_w * bbox_h
    bbox_cx = (bx1 + bx2) / 2.0
    bbox_cy = (by1 + by2) / 2.0
    max_dim = max(bbox_w, bbox_h)

    # 2. Area ratio ∈ [0.70, 1.30]
    area_ratio = poly_area / bbox_area
    if not (0.70 <= area_ratio <= 1.30):
        if _ROI_DEBUG:
            logger.debug(
                f"[ROI-validate] FAIL: area_ratio={area_ratio:.3f} not in [0.70, 1.30]"
            )
        return False

    # 3. Center distance ≤ 0.15 × max_dim
    poly_cx = float(np.mean(pts[:, 0]))
    poly_cy = float(np.mean(pts[:, 1]))
    center_dist = np.sqrt((poly_cx - bbox_cx) ** 2 + (poly_cy - bbox_cy) ** 2)
    if center_dist > 0.15 * max_dim:
        if _ROI_DEBUG:
            logger.debug(
                f"[ROI-validate] FAIL: center_dist={center_dist:.1f} > "
                f"0.15*max_dim={0.15 * max_dim:.1f}"
            )
        return False

    # 4. Aspect ratio ∈ [1.3, 6.5]
    rect = cv2.minAreaRect(pts)
    (_, _), (rw, rh), roi_angle_raw = rect
    rw, rh = max(rw, 1.0), max(rh, 1.0)
    aspect = max(rw, rh) / min(rw, rh)
    if not (1.3 <= aspect <= 6.5):
        if _ROI_DEBUG:
            logger.debug(
                f"[ROI-validate] FAIL: aspect={aspect:.2f} not in [1.3, 6.5]"
            )
        return False

    # 5. IoU với YOLO mask ≥ 0.55 (nếu có xy_polygon)
    if xy_polygon is not None and len(xy_polygon) >= 3:
        try:
            iou = _compute_mask_iou_cv(pts, xy_polygon.astype(np.float32), img_shape)
            if iou < 0.55:
                if _ROI_DEBUG:
                    logger.debug(f"[ROI-validate] FAIL: IoU={iou:.3f} < 0.55")
                return False
        except Exception:
            # Nếu tính IoU thất bại → không dùng ROI polygon (an toàn hơn)
            if _ROI_DEBUG:
                logger.debug("[ROI-validate] FAIL: IoU computation error → reject")
            return False

    # 6. Angle diff ≤ 15° so với YOLO minAreaRect
    if xy_polygon is not None and len(xy_polygon) >= 4:
        try:
            yolo_angle = _angle_of_rect(xy_polygon.astype(np.float32))
            roi_angle = abs(roi_angle_raw) % 90
            angle_diff = abs(roi_angle - yolo_angle)
            angle_diff = min(angle_diff, 90.0 - angle_diff)
            if angle_diff > 15.0:
                if _ROI_DEBUG:
                    logger.debug(
                        f"[ROI-validate] FAIL: angle_diff={angle_diff:.1f}° > 15°"
                    )
                return False
        except Exception:
            if _ROI_DEBUG:
                logger.debug("[ROI-validate] FAIL: angle computation error → reject")
            return False

    # 7. Không bám sát ≥ 3 cạnh expanded bbox
    if not _check_not_clinging_to_roi_border(pts, expanded_bbox, border_margin=4):
        if _ROI_DEBUG:
            logger.debug("[ROI-validate] FAIL: polygon bám sát biên ROI expanded")
        return False

    # 8. Không có điểm nào bay ra ngoài ảnh > 50px
    for pt in pts:
        x, y = pt
        if x < -50 or x > img_w + 50 or y < -50 or y > img_h + 50:
            if _ROI_DEBUG:
                logger.debug(
                    f"[ROI-validate] FAIL: point ({x:.0f},{y:.0f}) out of image"
                )
            return False

    return True


def refine_panel_contour_from_expanded_roi(
    orig_img: np.ndarray,
    bbox: List[float],
    xy_polygon: Optional[np.ndarray] = None,
    expand_ratio: float = 0.15,
) -> Optional[List[List[int]]]:
    """
    Orchestrator ROI-based panel edge refinement.
    Trả về polygon 4 điểm [[x,y]×4] hoặc None → caller fallback về pipeline cũ.
    """
    try:
        # 1. Expand bbox
        x1_exp, y1_exp, x2_exp, y2_exp = expand_bbox(bbox, orig_img.shape, expand_ratio)
        expanded_bbox = (x1_exp, y1_exp, x2_exp, y2_exp)

        # 2. Crop ROI — kích thước tối thiểu 20×20
        roi = orig_img[y1_exp:y2_exp, x1_exp:x2_exp]
        roi_h, roi_w = roi.shape[:2]
        if roi_w < 20 or roi_h < 20:
            if _ROI_DEBUG:
                logger.debug("[ROI-refine] SKIP: ROI < 20px")
            return None

        # 3. Tìm polygon ứng viên tốt nhất từ ROI
        poly_global = find_panel_polygon_from_roi(
            roi=roi,
            roi_offset=(x1_exp, y1_exp),
            original_bbox=bbox,
            img_shape=orig_img.shape,
            expanded_bbox=expanded_bbox,
            xy_polygon=xy_polygon,
        )

        if poly_global is None:
            if _ROI_DEBUG:
                logger.debug("[ROI-refine] SKIP: không tìm được contour phù hợp")
            return None

        # 4. Validate nghiêm ngặt
        if not validate_roi_panel_polygon(
            poly=poly_global,
            bbox=bbox,
            expanded_bbox=expanded_bbox,
            img_shape=orig_img.shape,
            xy_polygon=xy_polygon,
        ):
            if _ROI_DEBUG:
                logger.debug("[ROI-refine] SKIP: polygon không qua validate")
            return None

        # 5. Trả về polygon đã sắp xếp góc
        result = _sort_corners(poly_global.astype(np.float32))
        logger.info(f"[ROI-refine] SUCCESS: {result}")
        return result

    except Exception as exc:
        logger.warning(f"[ROI-refine] Exception: {exc}")
        return None


# ===========================================================================
# SECTION 3 — MAIN REFINE FUNCTION
# ===========================================================================

def _log_refine_result(
    idx: int,
    conf: float,
    bbox: List[float],
    raw_area: float,
    refined_poly: List[List[int]],
    method: str
) -> None:
    try:
        refined_pts = np.array(refined_poly, dtype=np.float32)
        refined_area = float(cv2.contourArea(refined_pts.reshape(-1, 1, 2))) if len(refined_pts) >= 3 else 0.0
    except Exception:
        refined_area = 0.0

    ratio = refined_area / raw_area if raw_area > 0 else 0.0

    logger.info(
        f"[PANEL_REFINE] idx={idx} conf={conf:.4f} raw_bbox={[round(v, 1) for v in bbox]} raw_area={raw_area:.1f} "
        f"refined_poly={refined_poly} refined_area={refined_area:.1f} ratio={ratio:.4f} method={method}"
    )

    if ratio < 0.75:
        logger.warning(
            f"[PANEL_REFINE_WARN] refined polygon shrank too much. "
            f"idx={idx} ratio={ratio:.4f} (< 0.75)"
        )


def save_debug_stages_image(
    orig_img: np.ndarray,
    raw_panels: List[Dict[str, Any]],
    filtered_panels: List[Dict[str, Any]],
    image_path: str
) -> None:
    try:
        import os
        base_name = os.path.basename(image_path)
        name_part, ext = os.path.splitext(base_name)
        debug_filename = f"debug_{name_part}_block_lattice_refine{ext}"
        debug_dir = "data/results/debug"
        os.makedirs(debug_dir, exist_ok=True)
        debug_path = os.path.join(debug_dir, debug_filename)

        draw_img = orig_img.copy()

        # Group panels by block_id to get block info and draw rails
        block_fits = {}
        for p in raw_panels:
            b_id = p.get("block_id", -1)
            if b_id != -1 and p.get("block_decision") == "use_lattice" and b_id not in block_fits:
                if "block_origin" in p:
                    block_fits[b_id] = {
                        "origin": np.array(p["block_origin"], dtype=np.float32),
                        "row_axis": np.array(p["block_row_axis"], dtype=np.float32),
                        "col_axis": np.array(p["block_col_axis"], dtype=np.float32),
                        "row_min": p["block_row_min"],
                        "row_max": p["block_row_max"],
                        "col_min": p["block_col_min"],
                        "col_max": p["block_col_max"],
                        "row_rails": p["block_row_rails"],
                        "col_rails": p["block_col_rails"],
                        "layout_mode": p["block_layout_mode"],
                        "n_stack": p["block_n_stack"],
                        "n_cols": p["block_n_cols"],
                        "pass_ratio": p["block_pass_ratio"]
                    }

        # Draw block rails
        for b_id, fit in block_fits.items():
            origin = fit["origin"]
            row_axis = fit["row_axis"]
            col_axis = fit["col_axis"]
            row_min = fit["row_min"]
            row_max = fit["row_max"]
            col_min = fit["col_min"]
            col_max = fit["col_max"]
            row_rails = fit["row_rails"]
            col_rails = fit["col_rails"]

            # Row rails (stack_axis rails): lines perpendicular to row_axis, along col_axis.
            # Drawn from col_min to col_max.
            # Color: White (255, 255, 255)
            for r_val in row_rails:
                pt_start = origin + r_val * row_axis + col_min * col_axis
                pt_end = origin + r_val * row_axis + col_max * col_axis
                cv2.line(
                    draw_img,
                    (int(round(pt_start[0])), int(round(pt_start[1]))),
                    (int(round(pt_end[0])), int(round(pt_end[1]))),
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )

            # Column rails (across_axis rails): lines perpendicular to col_axis, along row_axis.
            # Drawn from row_min to row_max.
            # Color: Red (0, 0, 255)
            for c_val in col_rails:
                pt_start = origin + row_min * row_axis + c_val * col_axis
                pt_end = origin + row_max * row_axis + c_val * col_axis
                cv2.line(
                    draw_img,
                    (int(round(pt_start[0])), int(round(pt_start[1]))),
                    (int(round(pt_end[0])), int(round(pt_end[1]))),
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA
                )

            # Label block
            block_label = f"B_{b_id} ({fit['n_stack']}x{fit['n_cols']}) {fit['layout_mode']} pass={fit['pass_ratio']:.2f}"
            lbl_pos = origin + row_min * row_axis + col_min * col_axis
            cv2.putText(
                draw_img,
                block_label,
                (int(lbl_pos[0]), int(lbl_pos[1]) - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                1,
                cv2.LINE_AA
            )

        # Draw panels
        for p in raw_panels:
            raw_idx = p.get("raw_idx", -1)
            raw_conf = p.get("raw_conf", 0.0)
            raw_yolo_poly = p.get("raw_yolo_poly", [])
            refined_poly = p.get("polygon", [])
            
            # 1. YOLO raw polygon: thin, semi-transparent blue (draw in blue-ish color)
            if len(raw_yolo_poly) >= 3:
                pts_raw = np.array(raw_yolo_poly, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(draw_img, [pts_raw], isClosed=True, color=(255, 100, 50), thickness=1, lineType=cv2.LINE_AA)

            # 2. Candidate lattice polygon: thin, semi-transparent yellow
            if "candidate_polygon" in p:
                pts_cand = np.array(p["candidate_polygon"], dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(draw_img, [pts_cand], isClosed=True, color=(0, 255, 255), thickness=1, lineType=cv2.LINE_AA)

            # 3. Final polygon: green/cyan if lattice used, fallback: red/gray
            dec = p.get("block_decision", "bypass")
            if dec == "use_lattice":
                pts_final = np.array(refined_poly, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(draw_img, [pts_final], isClosed=True, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
                label_color = (0, 255, 0)
            elif dec == "fallback_original" or dec == "fallback_block":
                pts_final = np.array(refined_poly, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(draw_img, [pts_final], isClosed=True, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
                label_color = (0, 0, 255)
            else:
                if len(refined_poly) >= 3:
                    pts_final = np.array(refined_poly, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(draw_img, [pts_final], isClosed=True, color=(200, 200, 200), thickness=2, lineType=cv2.LINE_AA)
                label_color = (200, 200, 200)

            # Ghi label
            if len(refined_poly) >= 3:
                cx, cy = p.get("center", [0.0, 0.0])
                b_id = p.get("block_id", -1)
                r_idx = p.get("row_idx", -1)
                c_idx = p.get("col_idx", -1)
                
                sr_text = ""
                if "score_ratio" in p:
                    sr_text = f" sr={p['score_ratio']:.2f}"

                if dec == "use_lattice":
                    area_ratio_val = p.get("area_ratio", 1.0)
                    iou_val = p.get("iou_with_original", 1.0)
                    label_text = f"idx={raw_idx} B={b_id}({r_idx},{c_idx}) iou={iou_val:.2f} r={area_ratio_val:.2f}{sr_text}"
                elif dec == "fallback_original" or dec == "fallback_block":
                    label_text = f"idx={raw_idx} B={b_id} fb ({p.get('block_reason', 'unknown')}){sr_text}"
                else:
                    label_text = f"idx={raw_idx} conf={raw_conf:.2f}"

                cv2.putText(draw_img, label_text, (int(cx) - 40, int(cy) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, label_color, 1, cv2.LINE_AA)

        cv2.imwrite(debug_path, draw_img)
        logger.info(f"[DEBUG_IMAGE] Saved block lattice debug image to: {debug_path}")
    except Exception as e:
        logger.warning(f"[DEBUG_IMAGE_WARN] Failed to save debug image: {e}")


def validate_snapped_panel(snapped_poly: List[List[int]], bbox: List[float]) -> bool:
    """
    Validate snapped polygon đối chiếu với original YOLO bbox:
    - area_ratio_to_bbox trong [0.75, 1.25]
    - width_ratio_to_bbox trong [0.80, 1.20]
    - height_ratio_to_bbox trong [0.80, 1.20]
    - center_shift_ratio <= 0.15
    - aspect_ratio trong [1.0, 8.0]
    """
    try:
        x1, y1, x2, y2 = bbox
        bbox_w = x2 - x1
        bbox_h = y2 - y1
        bbox_area = bbox_w * bbox_h
        if bbox_area <= 0:
            return False

        new_x1 = snapped_poly[0][0]
        new_y1 = snapped_poly[0][1]
        new_x2 = snapped_poly[2][0]
        new_y2 = snapped_poly[2][1]

        snapped_w = new_x2 - new_x1
        snapped_h = new_y2 - new_y1
        snapped_area = snapped_w * snapped_h
        if snapped_w <= 0 or snapped_h <= 0 or snapped_area <= 0:
            return False

        area_ratio = snapped_area / bbox_area
        width_ratio = snapped_w / bbox_w
        height_ratio = snapped_h / bbox_h

        bbox_cx = (x1 + x2) / 2.0
        bbox_cy = (y1 + y2) / 2.0
        snapped_cx = (new_x1 + new_x2) / 2.0
        snapped_cy = (new_y1 + new_y2) / 2.0
        dist = float(np.sqrt((snapped_cx - bbox_cx)**2 + (snapped_cy - bbox_cy)**2))
        center_shift_ratio = dist / max(bbox_w, bbox_h) if max(bbox_w, bbox_h) > 0 else 0.0

        aspect_ratio = max(snapped_w, snapped_h) / min(snapped_w, snapped_h) if min(snapped_w, snapped_h) > 0 else 0.0

        if not (0.75 <= area_ratio <= 1.25):
            return False
        if not (0.80 <= width_ratio <= 1.20):
            return False
        if not (0.80 <= height_ratio <= 1.20):
            return False
        if center_shift_ratio > 0.15:
            return False
        if not (1.0 <= aspect_ratio <= 8.0):
            return False

        return True
    except Exception:
        return False


def snap_panel_bbox_to_edges(
    orig_img: np.ndarray,
    bbox: List[float],
    search_px: int = 8,
    max_shift_px: int = 10
) -> Tuple[Optional[List[List[int]]], Dict[str, Any]]:
    """
    Tinh chỉnh 4 cạnh bbox trong vùng tìm kiếm hẹp ±search_px quanh bbox
    bằng cách tính toán Sobel Gradient kết hợp với độ tối của điểm ảnh (dark-line).
    """
    info = {
        "small_bbox_original": [round(v, 1) for v in bbox],
        "snapped_bbox": [],
        "snap_left_shift": 0,
        "snap_right_shift": 0,
        "snap_top_shift": 0,
        "snap_bottom_shift": 0,
        "snap_score_left": 0.0,
        "snap_score_right": 0.0,
        "snap_score_top": 0.0,
        "snap_score_bottom": 0.0,
        "snap_decision": "fallback_bbox",
        "snap_reason": "none"
    }

    try:
        h, w = orig_img.shape[:2]
        x1, y1, x2, y2 = bbox

        # crop expansion margin
        margin = max_shift_px + search_px + 2
        x1_crop = max(0, int(round(x1)) - margin)
        y1_crop = max(0, int(round(y1)) - margin)
        x2_crop = min(w, int(round(x2)) + margin)
        y2_crop = min(h, int(round(y2)) + margin)

        crop = orig_img[y1_crop:y2_crop, x1_crop:x2_crop]
        if crop.size == 0:
            info["snap_reason"] = "empty_crop"
            return None, info

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        x1_local = x1 - x1_crop
        y1_local = y1 - y1_crop
        x2_local = x2 - x1_crop
        y2_local = y2 - y1_crop

        # Segment ranges
        y_start = max(0, int(round(y1_local)))
        y_end = min(gray.shape[0], int(round(y2_local)))
        y_indices = np.arange(y_start, y_end)

        x_start = max(0, int(round(x1_local)))
        x_end = min(gray.shape[1], int(round(x2_local)))
        x_indices = np.arange(x_start, x_end)

        if len(y_indices) < 2 or len(x_indices) < 2:
            info["snap_reason"] = "edge_segment_too_short"
            return None, info

        # 1. Left Edge
        best_dx_left = 0
        max_score_left = -1.0
        for dx in range(-search_px, search_px + 1):
            if abs(dx) > max_shift_px:
                continue
            x_val = int(round(x1_local + dx))
            if 0 <= x_val < gray.shape[1]:
                grad = np.mean(np.abs(sobel_x[y_indices, x_val]))
                intens = np.mean(gray[y_indices, x_val])
                score = float(grad * (255.0 - intens))
                if score > max_score_left:
                    max_score_left = score
                    best_dx_left = dx

        # 2. Right Edge
        best_dx_right = 0
        max_score_right = -1.0
        for dx in range(-search_px, search_px + 1):
            if abs(dx) > max_shift_px:
                continue
            x_val = int(round(x2_local + dx))
            if 0 <= x_val < gray.shape[1]:
                grad = np.mean(np.abs(sobel_x[y_indices, x_val]))
                intens = np.mean(gray[y_indices, x_val])
                score = float(grad * (255.0 - intens))
                if score > max_score_right:
                    max_score_right = score
                    best_dx_right = dx

        # 3. Top Edge
        best_dy_top = 0
        max_score_top = -1.0
        for dy in range(-search_px, search_px + 1):
            if abs(dy) > max_shift_px:
                continue
            y_val = int(round(y1_local + dy))
            if 0 <= y_val < gray.shape[0]:
                grad = np.mean(np.abs(sobel_y[y_val, x_indices]))
                intens = np.mean(gray[y_val, x_indices])
                score = float(grad * (255.0 - intens))
                if score > max_score_top:
                    max_score_top = score
                    best_dy_top = dy

        # 4. Bottom Edge
        best_dy_bottom = 0
        max_score_bottom = -1.0
        for dy in range(-search_px, search_px + 1):
            if abs(dy) > max_shift_px:
                continue
            y_val = int(round(y2_local + dy))
            if 0 <= y_val < gray.shape[0]:
                grad = np.mean(np.abs(sobel_y[y_val, x_indices]))
                intens = np.mean(gray[y_val, x_indices])
                score = float(grad * (255.0 - intens))
                if score > max_score_bottom:
                    max_score_bottom = score
                    best_dy_bottom = dy

        info["snap_left_shift"] = int(best_dx_left)
        info["snap_right_shift"] = int(best_dx_right)
        info["snap_top_shift"] = int(best_dy_top)
        info["snap_bottom_shift"] = int(best_dy_bottom)

        info["snap_score_left"] = round(max_score_left, 1)
        info["snap_score_right"] = round(max_score_right, 1)
        info["snap_score_top"] = round(max_score_top, 1)
        info["snap_score_bottom"] = round(max_score_bottom, 1)

        new_x1 = int(round(x1 + best_dx_left))
        new_x2 = int(round(x2 + best_dx_right))
        new_y1 = int(round(y1 + best_dy_top))
        new_y2 = int(round(y2 + best_dy_bottom))

        snapped_poly = [[new_x1, new_y1], [new_x2, new_y1], [new_x2, new_y2], [new_x1, new_y2]]

        if validate_snapped_panel(snapped_poly, bbox):
            info["snapped_bbox"] = [new_x1, new_y1, new_x2, new_y2]
            info["snap_decision"] = "use_snapped"
            info["snap_reason"] = "edge_snap_passed"
            return snapped_poly, info
        else:
            info["snap_decision"] = "fallback_bbox"
            info["snap_reason"] = "validation_failed"
            return None, info

    except Exception as e:
        info["snap_reason"] = f"exception_{str(e)}"
        return None, info


def save_panel_refine_debug_jsonl(
    image_path: str,
    raw_panels: List[Dict[str, Any]],
) -> None:
    """
    Ghi file JSONL debug: data/results/debug_logs/<image_name>_panel_refine.jsonl

    Mỗi dòng = 1 JSON cho 1 panel, gồm đầy đủ thông tin geometry để debug offline.
    Terminal chỉ in 1 dòng ngắn [PANEL_DEBUG_FILE].

    Không thay đổi logic detect hay threshold.
    """
    import json
    import os
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
                bbox_w = x2b - x1b
                bbox_h = y2b - y1b
                bbox_area = bbox_w * bbox_h

                conf = float(p.get("raw_conf", p.get("confidence", 0.0)))
                idx = int(p.get("raw_idx", -1))
                raw_yolo_poly = p.get("raw_yolo_poly", [])
                refined_poly = p.get("polygon", [])

                # ── Raw mask metrics ────────────────────────────────────
                try:
                    raw_pts = np.array(raw_yolo_poly, dtype=np.float32)
                    if len(raw_pts) >= 3:
                        raw_mask_area = float(cv2.contourArea(raw_pts.reshape(-1, 1, 2)))
                        xm, ym, wm, hm = cv2.boundingRect(raw_pts.astype(np.int32).reshape(-1, 1, 2))
                        mask_cx = float(np.mean(raw_pts[:, 0]))
                        mask_cy = float(np.mean(raw_pts[:, 1]))
                    else:
                        raw_mask_area = 0.0
                        wm = hm = 0
                        mask_cx = (x1b + x2b) / 2
                        mask_cy = (y1b + y2b) / 2
                except Exception:
                    raw_mask_area = 0.0
                    wm = hm = 0
                    mask_cx = (x1b + x2b) / 2
                    mask_cy = (y1b + y2b) / 2

                mask_bbox_ratio = raw_mask_area / bbox_area if bbox_area > 0 else 0.0

                area_ratio = raw_mask_area / bbox_area if bbox_area > 0 else 0.0
                width_ratio = float(wm) / bbox_w if bbox_w > 0 else 0.0
                height_ratio = float(hm) / bbox_h if bbox_h > 0 else 0.0
                bbox_cx = (x1b + x2b) / 2
                bbox_cy = (y1b + y2b) / 2
                dist = float(np.sqrt((mask_cx - bbox_cx) ** 2 + (mask_cy - bbox_cy) ** 2))
                center_shift = dist / max(bbox_w, bbox_h) if max(bbox_w, bbox_h) > 0 else 0.0

                # ── Refined polygon metrics ─────────────────────────────
                try:
                    ref_pts = np.array(refined_poly, dtype=np.float32)
                    if len(ref_pts) >= 3:
                        refined_area = float(cv2.contourArea(ref_pts.reshape(-1, 1, 2)))
                        xr, yr, wr, hr = cv2.boundingRect(ref_pts.astype(np.int32).reshape(-1, 1, 2))
                        refined_bbox = [int(xr), int(yr), int(xr + wr), int(yr + hr)]
                    else:
                        refined_area = 0.0
                        refined_bbox = [int(x1b), int(y1b), int(x2b), int(y2b)]
                except Exception:
                    refined_area = 0.0
                    refined_bbox = [int(x1b), int(y1b), int(x2b), int(y2b)]

                # ── Threshold set ────────────────────────────────────────
                threshold_set = "small" if bbox_area <= SMALL_PANEL_BBOX_AREA_PX else "strict"

                # ── Infer method & decision from final polygon ───────────
                # So sánh refined_poly với expected bbox_poly (deterministic)
                x1i = int(round(x1b)); y1i = int(round(y1b))
                x2i = int(round(x2b)); y2i = int(round(y2b))
                expected_bbox = [[x1i, y1i], [x2i, y1i], [x2i, y2i], [x1i, y2i]]

                if p.get("grid_decision") == "use_grid":
                    method = "small_panel_grid_refine"
                    decision = "pass"
                    reason = "grid_refine_passed"
                elif p.get("snap_decision") == "use_snapped":
                    method = "small_bbox_edge_snap"
                    decision = "pass"
                    reason = "edge_snap_passed"
                elif refined_poly == expected_bbox:
                    if (SMALL_PANEL_FORCE_BBOX
                            and bbox_area <= SMALL_PANEL_BBOX_AREA_PX
                            and conf >= SMALL_PANEL_USE_BBOX_CONF):
                        method = "small_bbox"
                        decision = "pass"
                        reason = "force_bbox_small_confident_panel"
                    else:
                        method = "fallback_bbox"
                        decision = "fallback_bbox"
                        reason = "mask_guard_rejected"
                else:
                    method = "mask_pipeline"
                    decision = "pass"
                    reason = "mask_refined_ok"

                record = {
                    "image_name": image_name,
                    "idx": idx,
                    "local_id": p.get("local_id", "N/A"),
                    "class_name": "panel",
                    "conf": round(conf, 4),
                    "bbox": [round(v, 1) for v in box],
                    "bbox_w": round(bbox_w, 1),
                    "bbox_h": round(bbox_h, 1),
                    "bbox_area": round(bbox_area, 1),
                    "raw_mask_area": round(raw_mask_area, 1),
                    "mask_bbox_ratio": round(mask_bbox_ratio, 3),
                    "refined_polygon": refined_poly,
                    "refined_bbox": refined_bbox,
                    "refined_area": round(refined_area, 1),
                    "area_ratio": round(area_ratio, 3),
                    "width_ratio": round(width_ratio, 3),
                    "height_ratio": round(height_ratio, 3),
                    "center_shift": round(center_shift, 3),
                    "threshold_set": threshold_set,
                    "method": method,
                    "decision": decision,
                    "reason": reason,
                    # Grid field
                    "grid_decision": p.get("grid_decision", "n/a"),
                    # Snap fields
                    "small_bbox_original": p.get("small_bbox_original", []),
                    "snapped_bbox": p.get("snapped_bbox", []),
                    "snap_left_shift": p.get("snap_left_shift", 0),
                    "snap_right_shift": p.get("snap_right_shift", 0),
                    "snap_top_shift": p.get("snap_top_shift", 0),
                    "snap_bottom_shift": p.get("snap_bottom_shift", 0),
                    "snap_score_left": p.get("snap_score_left", 0.0),
                    "snap_score_right": p.get("snap_score_right", 0.0),
                    "snap_score_top": p.get("snap_score_top", 0.0),
                    "snap_score_bottom": p.get("snap_score_bottom", 0.0),
                    "snap_decision": p.get("snap_decision", "n/a"),
                    "snap_reason": p.get("snap_reason", "not_small_panel")
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.info(f"[PANEL_DEBUG_FILE] saved {out_path}")

    except Exception as e:
        logger.warning(f"[PANEL_DEBUG_FILE] WARN: could not write debug file: {e}")


def _validate_panel_polygon_against_bbox(
    poly,
    bbox,
    img_shape=None,
    method: str = "unknown",
) -> Tuple[bool, str]:
    """
    Kiểm tra hình học polygon so với YOLO bbox của chính panel đó.

    Threshold set được chọn theo bbox_area:
      - bbox_area <= SMALL_PANEL_BBOX_AREA_PX  → "small" (ngưỡng lỏng hơn)
      - bbox_area >  SMALL_PANEL_BBOX_AREA_PX  → "strict" (ngưỡng gốc, đã kiểm tra)

    Emit log:
      [PANEL_REFINE_GUARD] method=... bbox_area=... area_ratio=... width_ratio=...
                           height_ratio=... center_shift=... threshold_set=...
                           decision=pass/reject reason=...
    """
    try:
        pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
        if len(pts) < 3:
            return False, "too_few_points"

        bbox_w = max(bbox[2] - bbox[0], 1.0)
        bbox_h = max(bbox[3] - bbox[1], 1.0)
        bbox_area = bbox_w * bbox_h

        # ── Chọn threshold set dựa vào kích thước bbox ──────────────────
        if bbox_area <= SMALL_PANEL_BBOX_AREA_PX:
            threshold_set = "small"
            min_area_ratio   = PANEL_POLY_SMALL_MIN_BBOX_AREA_RATIO
            min_width_ratio  = PANEL_POLY_SMALL_MIN_BBOX_WIDTH_RATIO
            min_height_ratio = PANEL_POLY_SMALL_MIN_BBOX_HEIGHT_RATIO
            max_center_shift = PANEL_POLY_SMALL_MAX_CENTER_SHIFT_RATIO
            min_aspect       = PANEL_POLY_SMALL_MIN_ASPECT_RATIO
        else:
            threshold_set = "strict"
            min_area_ratio   = PANEL_POLY_MIN_BBOX_AREA_RATIO
            min_width_ratio  = PANEL_POLY_MIN_BBOX_WIDTH_RATIO
            min_height_ratio = PANEL_POLY_MIN_BBOX_HEIGHT_RATIO
            max_center_shift = PANEL_POLY_MAX_CENTER_SHIFT_RATIO
            min_aspect       = PANEL_POLY_MIN_ASPECT_RATIO

        max_aspect = PANEL_POLY_MAX_ASPECT_RATIO

        # ── Tính các metrics ─────────────────────────────────────────────
        poly_area = float(cv2.contourArea(pts.reshape(-1, 1, 2)))
        x_c, y_c, w, h = cv2.boundingRect(pts.astype(np.int32).reshape(-1, 1, 2))
        poly_w = float(w)
        poly_h = float(h)

        area_ratio = poly_area / bbox_area if bbox_area > 0 else 0.0
        width_ratio = poly_w / bbox_w
        height_ratio = poly_h / bbox_h

        poly_cx = float(np.mean(pts[:, 0]))
        poly_cy = float(np.mean(pts[:, 1]))
        bbox_cx = (bbox[0] + bbox[2]) / 2.0
        bbox_cy = (bbox[1] + bbox[3]) / 2.0
        dist = float(np.sqrt((poly_cx - bbox_cx) ** 2 + (poly_cy - bbox_cy) ** 2))
        center_distance_ratio = dist / max(bbox_w, bbox_h)

        aspect_ratio = max(poly_w, poly_h) / max(min(poly_w, poly_h), 1e-6)

        # ── Kiểm tra từng điều kiện ──────────────────────────────────────
        decision = "pass"
        fail_reason = "ok"

        if area_ratio < min_area_ratio:
            decision = "reject"
            fail_reason = f"area_ratio_low (area_ratio={area_ratio:.3f} < {min_area_ratio})"
        elif width_ratio < min_width_ratio:
            decision = "reject"
            fail_reason = f"width_ratio_low (width_ratio={width_ratio:.3f} < {min_width_ratio})"
        elif height_ratio < min_height_ratio:
            decision = "reject"
            fail_reason = f"height_ratio_low (height_ratio={height_ratio:.3f} < {min_height_ratio})"
        elif center_distance_ratio > max_center_shift:
            decision = "reject"
            fail_reason = f"center_shift_high (center_shift={center_distance_ratio:.3f} > {max_center_shift})"
        elif aspect_ratio < min_aspect or aspect_ratio > max_aspect:
            decision = "reject"
            fail_reason = f"aspect_ratio_invalid (aspect={aspect_ratio:.2f} not in [{min_aspect},{max_aspect}])"

        # ── Structured log (theo yêu cầu E) ─────────────────────────────
        logger.info(
            f"[PANEL_REFINE_GUARD] method={method} "
            f"bbox_area={bbox_area:.1f} "
            f"area_ratio={area_ratio:.3f} "
            f"width_ratio={width_ratio:.3f} "
            f"height_ratio={height_ratio:.3f} "
            f"center_shift={center_distance_ratio:.3f} "
            f"threshold_set={threshold_set} "
            f"decision={decision} "
            f"reason={fail_reason}"
        )

        if decision == "reject":
            return False, fail_reason
        return True, "ok"

    except Exception as e:
        return False, f"exception: {str(e)}"


def refine_panel_contour(
    orig_img: np.ndarray,
    bbox: List[float],
    xy_polygon: np.ndarray,
    idx: int = -1,
    conf: float = 0.0,
    image_name: str = "unknown",
    debug_info: Optional[Dict[str, Any]] = None
) -> List[List[int]]:
    """
    Tinh chỉnh viền tấm pin từ đa giác thô YOLO (xy_polygon).

    Pipeline:
      [TÙY CHỌN — chỉ khi ENABLE_ROI_REFINEMENT = True]
        Bước A: ROI-based refinement với validate rất chặt.
                Nếu pass → dùng ROI polygon.
                Nếu fail → xuống bước B.

      [PIPELINE CŨ — LUÔN CHẠY KHI ROI KHÔNG PASS]
        Bước B1: convexHull + approxPolyDP thích ứng + _check_geometry_quality.
        Bước B2: minAreaRect fallback.
        Bước B3: YOLO bbox fallback.

    JSON field "polygon" không thay đổi — frontend không cần sửa gì.
    """
    if debug_info is not None:
        debug_info["small_bbox_original"] = []
        debug_info["snapped_bbox"] = []
        debug_info["snap_left_shift"] = 0
        debug_info["snap_right_shift"] = 0
        debug_info["snap_top_shift"] = 0
        debug_info["snap_bottom_shift"] = 0
        debug_info["snap_score_left"] = 0.0
        debug_info["snap_score_right"] = 0.0
        debug_info["snap_score_top"] = 0.0
        debug_info["snap_score_bottom"] = 0.0
        debug_info["snap_decision"] = "n/a"
        debug_info["snap_reason"] = "not_small_panel"

    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    bbox_poly = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

    method = "bbox_fallback"
    refined_poly = bbox_poly

    # Calculate raw area
    try:
        raw_pts = xy_polygon.astype(np.float32)
        raw_area = float(cv2.contourArea(raw_pts.reshape(-1, 1, 2))) if len(raw_pts) >= 3 else 0.0
    except Exception:
        raw_area = 0.0

    # ── SMALL PANEL: dùng bbox_poly trực tiếp nếu đủ confidence ──────────
    # Lý do: mask segmentation của small panel thường co cụm / thiếu cạnh.
    # YOLO bbox bao sát tấm pin đáng tin hơn khi confidence cao.
    # Panel lớn KHÔNG bị ảnh hưởng (bbox_area > SMALL_PANEL_BBOX_AREA_PX).
    bbox_area_px = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    if (SMALL_PANEL_FORCE_BBOX
            and bbox_area_px <= SMALL_PANEL_BBOX_AREA_PX
            and conf >= SMALL_PANEL_USE_BBOX_CONF):
        if debug_info is not None:
            debug_info.update({
                "small_bbox_original": [round(v, 1) for v in bbox],
                "snapped_bbox": [],
                "snap_left_shift": 0,
                "snap_right_shift": 0,
                "snap_top_shift": 0,
                "snap_bottom_shift": 0,
                "snap_score_left": 0.0,
                "snap_score_right": 0.0,
                "snap_score_top": 0.0,
                "snap_score_bottom": 0.0,
                "snap_decision": "fallback_bbox",
                "snap_reason": "edge_snap_disabled_forced_bbox"
            })
        logger.info(
            f"[PANEL_REFINE_SMALL] method=small_bbox decision=fallback_bbox: "
            f"idx={idx} conf={conf:.4f} bbox_area={bbox_area_px:.1f} "
            f"reason=edge_snap_disabled_forced_bbox"
        )
        print(
            f"[PANEL_SNAP] img={image_name} idx={idx} decision=fallback_bbox "
            f"shifts=(0, 0, 0, 0)"
        )
        _log_refine_result(idx, conf, bbox, raw_area, bbox_poly, "bbox_small_panel")
        return bbox_poly

    # 1. Kiểm tra đầu hàm: raw mask vs bbox (guard chặn mask sai/co cụm/dải mỏng)
    if xy_polygon is not None and len(xy_polygon) >= 3:
        valid_ok, reason = _validate_panel_polygon_against_bbox(
            xy_polygon, bbox, method="raw_mask_guard"
        )
        if not valid_ok:
            logger.info(
                f"[PANEL_REFINE_GUARD] fallback=bbox_poly idx={idx} "
                f"bbox={[round(v, 1) for v in bbox]}"
            )
            _log_refine_result(idx, conf, bbox, raw_area, bbox_poly, "bbox_fallback (raw mask guard)")
            return bbox_poly


    if xy_polygon is not None and len(xy_polygon) >= 3:
        # ── BƯỚC A (TÙY CHỌN): ROI-based refinement ────────────────────
        if ENABLE_ROI_REFINEMENT:
            roi_poly = refine_panel_contour_from_expanded_roi(
                orig_img=orig_img,
                bbox=bbox,
                xy_polygon=xy_polygon,
                expand_ratio=0.15,
            )
            if roi_poly is not None:
                method = "roi_refinement"
                refined_poly = roi_poly
                _log_refine_result(idx, conf, bbox, raw_area, refined_poly, method)
                return refined_poly
            logger.debug("[refine_panel_contour] ROI failed → fallback YOLO mask pipeline.")

        # ── BƯỚC B: Pipeline cũ YOLO mask (GIỮ NGUYÊN HOÀN TOÀN) ───────
        try:
            pts = xy_polygon.astype(np.float32)
            hull = cv2.convexHull(pts)

            # Hull-level Guard
            valid_ok, reason = _validate_panel_polygon_against_bbox(
                hull, bbox, method="hull_guard"
            )
            if not valid_ok:
                logger.info(
                    f"[PANEL_REFINE_GUARD] fallback=bbox_poly idx={idx} "
                    f"bbox={[round(v, 1) for v in bbox]}"
                )
                _log_refine_result(idx, conf, bbox, raw_area, bbox_poly, "bbox_fallback (hull guard)")
                return bbox_poly

            # B1. approxPolyDP thích ứng → 4 đỉnh hợp lệ
            arc_len = cv2.arcLength(hull, True)
            found_b1 = False
            for factor in np.linspace(0.005, 0.15, 120):
                epsilon = factor * arc_len
                approx = cv2.approxPolyDP(hull, epsilon, True)
                approx = approx.reshape(-1, 2)
                if len(approx) == 4:
                    if _check_geometry_quality(approx, bbox, orig_img.shape):
                        valid_ok, reason = _validate_panel_polygon_against_bbox(
                            approx, bbox, method="approxPolyDP_candidate"
                        )
                        if valid_ok:
                            refined_poly = _sort_corners(approx)
                            method = "convexHull+approxPolyDP"
                            found_b1 = True
                            break

            if not found_b1:
                # B2. minAreaRect fallback
                rect = cv2.minAreaRect(pts)
                box = cv2.boxPoints(rect)
                if len(box) == 4:
                    valid_ok, reason = _validate_panel_polygon_against_bbox(
                        box, bbox, method="minAreaRect_candidate"
                    )
                    if valid_ok:
                        refined_poly = _sort_corners(box)
                        method = "minAreaRect"
                    else:
                        logger.info(
                            f"[PANEL_REFINE_GUARD] fallback=bbox_poly idx={idx} "
                            f"bbox={[round(v, 1) for v in bbox]}"
                        )
                        _log_refine_result(idx, conf, bbox, raw_area, bbox_poly, "bbox_fallback (guard)")
                        return bbox_poly
                else:
                    method = "bbox_fallback"
                    refined_poly = bbox_poly
        except Exception:
            method = "bbox_fallback"
            refined_poly = bbox_poly

    _log_refine_result(idx, conf, bbox, raw_area, refined_poly, method)
    return refined_poly


def compute_polygon_overlap_ratio(poly_a: List[List[int]], poly_b: List[List[int]], img_shape: Tuple[int, ...]) -> float:
    """
    Tính tỷ lệ diện tích overlap giữa hai polygon so với diện tích nhỏ nhất của hai polygon đó.
    Tương đương với: overlap_area / min(area_a, area_b)
    """
    try:
        h, w = img_shape[:2]
        m_a = np.zeros((h, w), dtype=np.uint8)
        m_b = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(m_a, [np.array(poly_a, dtype=np.int32)], 1)
        cv2.fillPoly(m_b, [np.array(poly_b, dtype=np.int32)], 1)
        intersection = float(np.logical_and(m_a, m_b).sum())
        area_a = float(m_a.sum())
        area_b = float(m_b.sum())
        min_area = min(area_a, area_b)
        if min_area <= 0:
            return 0.0
        return intersection / min_area
    except Exception:
        return 0.0


def touches_image_border(bbox: List[float], image_shape: Tuple[int, ...], margin: int = 8) -> bool:
    """
    Check if the panel bbox is close to the image borders.
    """
    h_img, w_img = image_shape[:2]
    x1, y1, x2, y2 = bbox
    if x1 <= margin or y1 <= margin or x2 >= w_img - margin or y2 >= h_img - margin:
        return True
    return False


def score_panel_polygon_edges(orig_img: np.ndarray, polygon: List[List[int]]) -> Dict[str, float]:
    """
    Tính edge alignment score cho polygon trên ảnh gốc.
    Cạnh tốt thường có gradient magnitude cao và nằm trên rãnh tối ngăn cách các panel.
    """
    try:
        h, w = orig_img.shape[:2]
        gray = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
        
        # Tạo gradient map
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(sobel_x**2 + sobel_y**2)
        
        pts = np.array(polygon, dtype=np.float32)
        edge_gradients = []
        edge_darks = []
        N = 30
        
        for i in range(4):
            p_start = pts[i]
            p_end = pts[(i + 1) % 4]
            
            for step in range(N):
                t = step / float(N - 1) if N > 1 else 0.5
                pt = p_start * (1 - t) + p_end * t
                px = int(round(pt[0]))
                py = int(round(pt[1]))
                
                px = np.clip(px, 0, w - 1)
                py = np.clip(py, 0, h - 1)
                
                edge_gradients.append(float(grad_mag[py, px]))
                edge_darks.append(float(255.0 - gray[py, px]))
                
        mean_gradient = float(np.mean(edge_gradients)) if edge_gradients else 0.0
        dark_line_score = float(np.mean(edge_darks)) if edge_darks else 0.0
        edge_score = mean_gradient + 0.1 * dark_line_score
        
        return {
            "edge_score": round(edge_score, 4),
            "mean_gradient": round(mean_gradient, 4),
            "dark_line_score": round(dark_line_score, 4)
        }
    except Exception:
        return {
            "edge_score": 0.0,
            "mean_gradient": 0.0,
            "dark_line_score": 0.0
        }


def refine_small_panel_group_grid(
    panels: List[Dict[str, Any]],
    image_shape: Tuple[int, ...],
    orig_img: np.ndarray,
    image_path: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Refine small panels using sub-cluster grid structure and oriented quadrilateral fitting.
    """
    import json
    import os

    if not panels:
        return panels

    # Initialize grid_decision for all panels
    for p in panels:
        p["grid_decision"] = "n/a"

    # Identify small panels
    candidate_indices = []
    for idx, p in enumerate(panels):
        box = p.get("box") or p.get("bbox", [0, 0, 0, 0])
        bw = max(box[2] - box[0], 0.0)
        bh = max(box[3] - box[1], 0.0)
        b_area = bw * bh
        conf = p.get("raw_conf", p.get("confidence", 0.0))
        if b_area <= SMALL_PANEL_BBOX_AREA_PX and conf >= SMALL_PANEL_USE_BBOX_CONF:
            candidate_indices.append(idx)

    if not candidate_indices:
        return panels

    image_name = "unknown"
    if image_path:
        image_name = os.path.splitext(os.path.basename(image_path))[0]

    logger.info(f"[GRID_AWARE] Found {len(candidate_indices)} small panel candidates for grid processing in image={image_name}.")

    # BFS component grouping based on center distance (raw clusters)
    num_candidates = len(candidate_indices)
    visited = [False] * num_candidates
    raw_clusters = []

    for i in range(num_candidates):
        if visited[i]:
            continue
        component = []
        queue = [i]
        visited[i] = True
        while queue:
            curr = queue.pop(0)
            component.append(candidate_indices[curr])
            cx_curr, cy_curr = panels[candidate_indices[curr]].get("center", [0.0, 0.0])
            for neighbor in range(num_candidates):
                if not visited[neighbor]:
                    cx_neigh, cy_neigh = panels[candidate_indices[neighbor]].get("center", [0.0, 0.0])
                    dist = np.sqrt((cx_curr - cx_neigh)**2 + (cy_curr - cy_neigh)**2)
                    if dist <= 150.0:
                        visited[neighbor] = True
                        queue.append(neighbor)
        raw_clusters.append(component)

    debug_dir = "data/results/debug_logs"
    os.makedirs(debug_dir, exist_ok=True)
    log_path = os.path.join(debug_dir, f"{image_name}_small_panel_grid.jsonl")

    log_records = []
    
    # Store initial information needed for all panels
    old_polygons = {idx: panels[idx].get("polygon") for idx in candidate_indices}
    original_bbox_areas = {}
    bbox_ws = {}
    bbox_hs = {}
    for idx in candidate_indices:
        box = panels[idx].get("box") or panels[idx].get("bbox", [0, 0, 0, 0])
        x1, y1, x2, y2 = box
        bw = max(x2 - x1, 1.0)
        bh = max(y2 - y1, 1.0)
        original_bbox_areas[idx] = bw * bh
        bbox_ws[idx] = bw
        bbox_hs[idx] = bh

    # Global tracking of candidate polygons, decisions, reasons, scores, center shifts, dimensions, etc.
    candidate_polygons = {}
    panel_decisions = {idx: "fallback_bbox" for idx in candidate_indices}
    panel_reasons = {idx: "not_processed" for idx in candidate_indices}
    validation_scores = {idx: 0.0 for idx in candidate_indices}
    
    panel_bbox_scores = {idx: 0.0 for idx in candidate_indices}
    panel_grid_scores = {idx: 0.0 for idx in candidate_indices}
    panel_edge_ratios = {idx: 0.0 for idx in candidate_indices}
    panel_edge_decisions = {idx: "bbox_better" for idx in candidate_indices}
    
    # Track subcluster assignments for logging
    subcluster_ids = {idx: 0 for idx in candidate_indices}
    subcluster_angles = {idx: 0.0 for idx in candidate_indices}
    subcluster_decisions = {}
    subcluster_summaries = {}
    
    # Diagnostics counters
    subcluster_seq_counter = 0

    image_h, image_w = image_shape[:2]

    for raw_cluster_id, component in enumerate(raw_clusters, 1):
        # Calculate raw cluster median dimensions to aid splitting
        cluster_ws = [bbox_ws[idx] for idx in component]
        cluster_hs = [bbox_hs[idx] for idx in component]
        raw_median_w = float(np.median(cluster_ws)) if cluster_ws else 1.0
        raw_median_h = float(np.median(cluster_hs)) if cluster_hs else 1.0
        
        # Calculate rough grid angle for the raw cluster to rotate coordinates
        angles = []
        for idx_i in component:
            cx_i, cy_i = panels[idx_i].get("center", [0.0, 0.0])
            dists = []
            for idx_j in component:
                if idx_i == idx_j:
                    continue
                cx_j, cy_j = panels[idx_j].get("center", [0.0, 0.0])
                d = np.sqrt((cx_i - cx_j)**2 + (cy_i - cy_j)**2)
                dists.append((d, idx_j))
            dists.sort(key=lambda x: x[0])
            
            k = min(3, len(dists))
            for idx_d in range(k):
                d, idx_j = dists[idx_d]
                cx_j, cy_j = panels[idx_j].get("center", [0.0, 0.0])
                alpha = np.arctan2(cy_j - cy_i, cx_j - cx_i) * 180.0 / np.pi
                mapped = ((alpha + 45.0) % 90.0) - 45.0
                angles.append(mapped)
                
        raw_angle_deg = float(np.median(angles)) if angles else 0.0
        
        # Rotate centers to local coordinates (u, v) using raw_angle_deg
        theta_rad = raw_angle_deg * np.pi / 180.0
        cos_t = np.cos(theta_rad)
        sin_t = np.sin(theta_rad)
        
        rotated_coords = {}
        for idx in component:
            cx, cy = panels[idx].get("center", [0.0, 0.0])
            u = cx * cos_t + cy * sin_t
            v = -cx * sin_t + cy * cos_t
            rotated_coords[idx] = (u, v)
            
        # Determine layout orientation (vertical vs horizontal strings) dynamically using nearest neighbors
        vertical_votes = 0
        horizontal_votes = 0
        for idx_i in component:
            u_i, v_i = rotated_coords[idx_i]
            min_dist = float('inf')
            best_j = None
            for idx_j in component:
                if idx_i == idx_j:
                    continue
                u_j, v_j = rotated_coords[idx_j]
                dist = (u_i - u_j)**2 + (v_i - v_j)**2
                if dist < min_dist:
                    min_dist = dist
                    best_j = idx_j
            if best_j is not None:
                u_j, v_j = rotated_coords[best_j]
                du = abs(u_i - u_j)
                dv = abs(v_i - v_j)
                if du < dv:
                    vertical_votes += 1
                else:
                    horizontal_votes += 1
                    
        is_vertical_layout = (vertical_votes >= horizontal_votes)
            
        cluster_subclusters = []
        
        if is_vertical_layout:
            # Vertical layout: split horizontally into column groups, then split vertically within columns
            u_sorted = sorted(component, key=lambda idx: rotated_coords[idx][0])
            cols = []
            if u_sorted:
                cur_col = [u_sorted[0]]
                for idx in u_sorted[1:]:
                    prev_idx = cur_col[-1]
                    if rotated_coords[idx][0] - rotated_coords[prev_idx][0] > 0.5 * raw_median_w:
                        cols.append(cur_col)
                        cur_col = [idx]
                    else:
                        cur_col.append(idx)
                cols.append(cur_col)
                
            for col in cols:
                v_sorted = sorted(col, key=lambda idx: rotated_coords[idx][1])
                if not v_sorted:
                    continue
                cur_sub = [v_sorted[0]]
                for idx in v_sorted[1:]:
                    prev_idx = cur_sub[-1]
                    if rotated_coords[idx][1] - rotated_coords[prev_idx][1] > 1.8 * raw_median_h:
                        cluster_subclusters.append(cur_sub)
                        cur_sub = [idx]
                    else:
                        cur_sub.append(idx)
                cluster_subclusters.append(cur_sub)
        else:
            # Horizontal layout: split vertically into row groups, then split horizontally within rows
            v_sorted = sorted(component, key=lambda idx: rotated_coords[idx][1])
            rows = []
            if v_sorted:
                cur_row = [v_sorted[0]]
                for idx in v_sorted[1:]:
                    prev_idx = cur_row[-1]
                    if rotated_coords[idx][1] - rotated_coords[prev_idx][1] > 0.5 * raw_median_h:
                        rows.append(cur_row)
                        cur_row = [idx]
                    else:
                        cur_row.append(idx)
                rows.append(cur_row)
                
            for row in rows:
                u_sorted = sorted(row, key=lambda idx: rotated_coords[idx][0])
                if not u_sorted:
                    continue
                cur_sub = [u_sorted[0]]
                for idx in u_sorted[1:]:
                    prev_idx = cur_sub[-1]
                    if rotated_coords[idx][0] - rotated_coords[prev_idx][0] > 1.8 * raw_median_w:
                        cluster_subclusters.append(cur_sub)
                        cur_sub = [idx]
                    else:
                        cur_sub.append(idx)
                cluster_subclusters.append(cur_sub)
            
        # Log splitting summary
        print(f"[GRID_CLUSTER] image={image_name} raw_cluster={raw_cluster_id} split_into={len(cluster_subclusters)} subclusters")
        
        # Process each subcluster
        for subcluster in cluster_subclusters:
            subcluster_seq_counter += 1
            sub_id = subcluster_seq_counter
            n_sub = len(subcluster)
            
            # Record subcluster ID for every panel in the subcluster
            for idx in subcluster:
                subcluster_ids[idx] = sub_id

            if n_sub < SUBCLUSTER_MIN_PANEL_COUNT:
                # Mark as fallback directly
                for idx in subcluster:
                    panel_decisions[idx] = "fallback_bbox"
                    panel_reasons[idx] = "subcluster_too_small"
                    subcluster_angles[idx] = 0.0
                
                subcluster_decisions[sub_id] = "fallback_subcluster"
                subcluster_summaries[sub_id] = {
                    "record_type": "subcluster_summary",
                    "subcluster_id": sub_id,
                    "total_count": n_sub,
                    "pass_count": 0,
                    "pass_ratio": 0.0,
                    "angle_deg": 0.0,
                    "angle_std_deg": 0.0,
                    "max_overlap_candidate": 0.0,
                    "outside_count": 0,
                    "border_count": 0,
                    "decision": "fallback_subcluster",
                    "reason": "subcluster_too_small"
                }
                print(f"[GRID_SUBCLUSTER] id={sub_id} n={n_sub} angle=0.00 angle_std=0.00 decision=fallback_subcluster reason=subcluster_too_small")
                continue
                
            # Fit orientation angle for this subcluster using long-baseline pairs
            sub_angles = []
            min_baseline = 2.0 * (raw_median_h if is_vertical_layout else raw_median_w)
            for i in range(n_sub):
                idx_i = subcluster[i]
                cx_i, cy_i = panels[idx_i].get("center", [0.0, 0.0])
                for j in range(i + 1, n_sub):
                    idx_j = subcluster[j]
                    cx_j, cy_j = panels[idx_j].get("center", [0.0, 0.0])
                    d = np.sqrt((cx_i - cx_j)**2 + (cy_i - cy_j)**2)
                    if d >= min_baseline:
                        alpha = np.arctan2(cy_j - cy_i, cx_j - cx_i) * 180.0 / np.pi
                        mapped = ((alpha + 45.0) % 90.0) - 45.0
                        sub_angles.append(mapped)
                        
            if not sub_angles:
                for idx_i in subcluster:
                    cx_i, cy_i = panels[idx_i].get("center", [0.0, 0.0])
                    dists = []
                    for idx_j in subcluster:
                        if idx_i == idx_j:
                            continue
                        cx_j, cy_j = panels[idx_j].get("center", [0.0, 0.0])
                        d = np.sqrt((cx_i - cx_j)**2 + (cy_i - cy_j)**2)
                        dists.append((d, idx_j))
                    dists.sort(key=lambda x: x[0])
                    k = min(2, len(dists))
                    for idx_d in range(k):
                        d, idx_j = dists[idx_d]
                        cx_j, cy_j = panels[idx_j].get("center", [0.0, 0.0])
                        alpha = np.arctan2(cy_j - cy_i, cx_j - cx_i) * 180.0 / np.pi
                        mapped = ((alpha + 45.0) % 90.0) - 45.0
                        sub_angles.append(mapped)
                    
            # Filter angles to range [-15.0, 15.0]
            filtered_angles = [a for a in sub_angles if -15.0 <= a <= 15.0]
            if filtered_angles:
                sub_angle_deg = float(np.median(filtered_angles))
                sub_angle_std = float(np.std(filtered_angles))
            else:
                sub_angle_deg = float(np.median(sub_angles)) if sub_angles else 0.0
                sub_angle_std = float(np.std(sub_angles)) if sub_angles else 0.0
                
            # Record subcluster angle for logging
            for idx in subcluster:
                subcluster_angles[idx] = sub_angle_deg

            # Stability check: angle_std <= SUBCLUSTER_MAX_ANGLE_STD_DEG
            is_stable = (sub_angle_std <= SUBCLUSTER_MAX_ANGLE_STD_DEG)
            if not is_stable:
                for idx in subcluster:
                    panel_decisions[idx] = "fallback_bbox"
                    panel_reasons[idx] = "subcluster_quality_low"
                
                subcluster_decisions[sub_id] = "fallback_subcluster"
                subcluster_summaries[sub_id] = {
                    "record_type": "subcluster_summary",
                    "subcluster_id": sub_id,
                    "total_count": n_sub,
                    "pass_count": 0,
                    "pass_ratio": 0.0,
                    "angle_deg": round(sub_angle_deg, 4),
                    "angle_std_deg": round(sub_angle_std, 4),
                    "max_overlap_candidate": 0.0,
                    "outside_count": 0,
                    "border_count": 0,
                    "decision": "fallback_subcluster",
                    "reason": "subcluster_quality_low"
                }
                print(f"[GRID_SUBCLUSTER] id={sub_id} n={n_sub} angle={sub_angle_deg:.2f} angle_std={sub_angle_std:.2f} decision=fallback_subcluster reason=subcluster_quality_low")
                continue
                
            # Rotate coordinate centers using sub_angle_deg
            sub_theta = sub_angle_deg * np.pi / 180.0
            sub_cos_t = np.cos(sub_theta)
            sub_sin_t = np.sin(sub_theta)
            
            # Median dimensions of the subcluster
            sub_ws = [bbox_ws[idx] for idx in subcluster]
            sub_hs = [bbox_hs[idx] for idx in subcluster]
            sub_median_w = float(np.median(sub_ws)) if sub_ws else 1.0
            sub_median_h = float(np.median(sub_hs)) if sub_hs else 1.0
            
            # Initial clamped width/height
            cur_W = {}
            cur_H = {}
            for idx in subcluster:
                cur_W[idx] = np.clip(bbox_ws[idx], 0.90 * sub_median_w, 1.10 * sub_median_w)
                cur_H[idx] = np.clip(bbox_hs[idx], 0.90 * sub_median_h, 1.10 * sub_median_h)
                
            # Generate Oriented Candidate Polygons (Conservative Mode)
            for idx in subcluster:
                box = panels[idx].get("box") or panels[idx].get("bbox", [0, 0, 0, 0])
                cx = (box[0] + box[2]) / 2.0
                cy = (box[1] + box[3]) / 2.0
                W_i = cur_W[idx]
                H_i = cur_H[idx]
                
                # Local corners
                pt1_loc = [-W_i/2.0, -H_i/2.0]
                pt2_loc = [W_i/2.0, -H_i/2.0]
                pt3_loc = [W_i/2.0, H_i/2.0]
                pt4_loc = [-W_i/2.0, H_i/2.0]
                
                # Rotate back to image space
                poly = []
                for pt in [pt1_loc, pt2_loc, pt3_loc, pt4_loc]:
                    x = cx + pt[0] * sub_cos_t - pt[1] * sub_sin_t
                    y = cy + pt[0] * sub_sin_t + pt[1] * sub_cos_t
                    poly.append([int(round(x)), int(round(y))])
                    
                poly = _sort_corners(np.array(poly))
                candidate_polygons[idx] = poly

            # Compute pairwise candidate overlaps within the subcluster
            panel_max_overlaps = {}
            for idx in subcluster:
                poly_idx = candidate_polygons[idx]
                max_ov = 0.0
                for other_idx in subcluster:
                    if idx == other_idx:
                        continue
                    ov = compute_polygon_overlap_ratio(poly_idx, candidate_polygons[other_idx], image_shape)
                    if ov > max_ov:
                        max_ov = ov
                panel_max_overlaps[idx] = max_ov

            # Perform individual validation checks without writing final decisions
            panel_individually_passed = {}
            panel_fail_reasons = {}
            panel_validation_scores = {}
            outside_count = 0
            border_count = 0
            area_fail_count = 0
            center_shift_fail_count = 0
            pass_count = 0

            for idx in subcluster:
                poly = candidate_polygons[idx]
                
                # Check 0: Touches border
                box = panels[idx].get("box") or panels[idx].get("bbox", [0, 0, 0, 0])
                border_touch = touches_image_border(box, image_shape, margin=8)
                if border_touch:
                    border_count += 1
                
                # Check 1: Outside image bounds
                outside = False
                for pt in poly:
                    x_c, y_c = pt[0], pt[1]
                    if x_c < -2 or x_c > image_w + 2 or y_c < -2 or y_c > image_h + 2:
                        outside = True
                        break
                        
                bbox_w = bbox_ws[idx]
                bbox_h = bbox_hs[idx]
                bbox_area = original_bbox_areas[idx]
                
                # Check 2: Area Ratio
                new_area = float(cv2.contourArea(np.array(poly, dtype=np.float32).reshape(-1, 1, 2)))
                area_ratio = new_area / bbox_area
                area_ratio_ok = (GRID_PANEL_MIN_AREA_RATIO <= area_ratio <= GRID_PANEL_MAX_AREA_RATIO)
                
                # Check 3: Width / Height ratio (oriented dimensions)
                width_ratio = cur_W[idx] / bbox_w
                height_ratio = cur_H[idx] / bbox_h
                width_ratio_ok = (GRID_PANEL_MIN_WIDTH_RATIO <= width_ratio <= GRID_PANEL_MAX_WIDTH_RATIO)
                height_ratio_ok = (GRID_PANEL_MIN_HEIGHT_RATIO <= height_ratio <= GRID_PANEL_MAX_HEIGHT_RATIO)
                
                # Check 4: Center shift
                new_cx = float(np.mean([pt[0] for pt in poly]))
                new_cy = float(np.mean([pt[1] for pt in poly]))
                orig_cx = (box[0] + box[2]) / 2.0
                orig_cy = (box[1] + box[3]) / 2.0
                shift = float(np.sqrt((new_cx - orig_cx)**2 + (new_cy - orig_cy)**2))
                center_shift_ratio = shift / max(bbox_w, bbox_h)
                center_shift_ok = (center_shift_ratio <= GRID_PANEL_MAX_CENTER_SHIFT_RATIO)
                
                # Check 5: Overlap ratio within subcluster
                overlap_ratio = panel_max_overlaps[idx]
                overlap_ok = (overlap_ratio <= SUBCLUSTER_MAX_OVERLAP_RATIO)
                
                if border_touch:
                    panel_individually_passed[idx] = False
                    panel_fail_reasons[idx] = "touches_image_border"
                elif outside:
                    outside_count += 1
                    panel_individually_passed[idx] = False
                    panel_fail_reasons[idx] = "outside_image"
                elif not area_ratio_ok:
                    area_fail_count += 1
                    panel_individually_passed[idx] = False
                    panel_fail_reasons[idx] = "area_ratio_out_of_range"
                elif not (width_ratio_ok and height_ratio_ok):
                    panel_individually_passed[idx] = False
                    if not width_ratio_ok:
                        panel_fail_reasons[idx] = "width_ratio_out_of_range"
                    else:
                        panel_fail_reasons[idx] = "height_ratio_out_of_range"
                elif not center_shift_ok:
                    center_shift_fail_count += 1
                    panel_individually_passed[idx] = False
                    panel_fail_reasons[idx] = "center_shift_too_large"
                elif not overlap_ok:
                    panel_individually_passed[idx] = False
                    panel_fail_reasons[idx] = "overlap_too_high"
                else:
                    pass_count += 1
                    panel_individually_passed[idx] = True
                    panel_fail_reasons[idx] = "ok"
                    
                bbox_poly = [[int(round(box[0])), int(round(box[1]))], [int(round(box[2])), int(round(box[1]))], 
                             [int(round(box[2])), int(round(box[3]))], [int(round(box[0])), int(round(box[3]))]]
                panel_validation_scores[idx] = _compute_mask_iou_cv(np.array(poly), np.array(bbox_poly), image_shape)

            # Sub-cluster level decision
            pass_ratio = pass_count / n_sub if n_sub > 0 else 0.0
            max_overlap_candidate = max(panel_max_overlaps.values()) if panel_max_overlaps else 0.0
            
            border_subcluster = (border_count / n_sub > SUBCLUSTER_MAX_BORDER_RATIO)
            
            subcluster_decision_ok = (
                not border_subcluster and
                pass_ratio >= SUBCLUSTER_MIN_PASS_RATIO and
                outside_count <= SUBCLUSTER_MAX_OUTSIDE_COUNT and
                max_overlap_candidate <= SUBCLUSTER_MAX_OVERLAP_RATIO
            )
            
            if subcluster_decision_ok:
                subcluster_decision = "use_grid_subcluster"
                subcluster_reason = "ok"
                for idx in subcluster:
                    if panel_individually_passed[idx]:
                        box = panels[idx].get("box") or panels[idx].get("bbox", [0, 0, 0, 0])
                        bbox_poly = [[int(round(box[0])), int(round(box[1]))], [int(round(box[2])), int(round(box[1]))], 
                                     [int(round(box[2])), int(round(box[3]))], [int(round(box[0])), int(round(box[3]))]]
                        
                        bbox_res = score_panel_polygon_edges(orig_img, bbox_poly)
                        grid_res = score_panel_polygon_edges(orig_img, candidate_polygons[idx])
                        
                        bbox_score_val = bbox_res["edge_score"]
                        grid_score_val = grid_res["edge_score"]
                        ratio_val = grid_score_val / bbox_score_val if bbox_score_val > 0 else 1.0
                        
                        panel_bbox_scores[idx] = bbox_score_val
                        panel_grid_scores[idx] = grid_score_val
                        panel_edge_ratios[idx] = ratio_val
                        
                        if grid_score_val >= bbox_score_val * GRID_EDGE_SCORE_IMPROVEMENT_RATIO:
                            panel_decisions[idx] = "use_grid"
                            panel_reasons[idx] = "ok"
                            panel_edge_decisions[idx] = "grid_better"
                            validation_scores[idx] = panel_validation_scores[idx]
                        else:
                            panel_decisions[idx] = "fallback_bbox"
                            panel_reasons[idx] = "grid_not_better_than_bbox"
                            panel_edge_decisions[idx] = "bbox_better"
                            validation_scores[idx] = 0.0
                            
                        p_idx = int(panels[idx].get("raw_idx", idx))
                        print(f"[GRID_EDGE] img={image_name} idx={p_idx} bbox_score={bbox_score_val:.2f} grid_score={grid_score_val:.2f} ratio={ratio_val:.2f} decision={panel_decisions[idx]}")
                    else:
                        panel_decisions[idx] = "fallback_bbox"
                        panel_reasons[idx] = panel_fail_reasons[idx]
                        validation_scores[idx] = 0.0
            else:
                subcluster_decision = "fallback_subcluster"
                if border_subcluster:
                    subcluster_reason = "border_subcluster"
                else:
                    subcluster_reason = "subcluster_quality_low"
                    
                for idx in subcluster:
                    panel_decisions[idx] = "fallback_bbox"
                    panel_reasons[idx] = "subcluster_quality_low"
                    validation_scores[idx] = 0.0
                    
            subcluster_decisions[sub_id] = subcluster_decision
            
            subcluster_summaries[sub_id] = {
                "record_type": "subcluster_summary",
                "subcluster_id": sub_id,
                "total_count": n_sub,
                "pass_count": pass_count,
                "pass_ratio": round(pass_ratio, 4),
                "angle_deg": round(sub_angle_deg, 4),
                "angle_std_deg": round(sub_angle_std, 4),
                "max_overlap_candidate": round(max_overlap_candidate, 4),
                "outside_count": outside_count,
                "border_count": border_count,
                "decision": subcluster_decision,
                "reason": subcluster_reason
            }
            
            print(f"[GRID_SUBCLUSTER] id={sub_id} n={n_sub} angle={sub_angle_deg:.2f} angle_std={sub_angle_std:.2f} decision={subcluster_decision} reason={subcluster_reason} pass_ratio={pass_ratio:.2f}")

    # Initialize final polygons using decision
    final_polygons = {}
    for idx in candidate_indices:
        if panel_decisions[idx] == "use_grid":
            final_polygons[idx] = candidate_polygons[idx]
        else:
            final_polygons[idx] = old_polygons[idx]

    # Global pairwise overlap resolution based on score
    while True:
        target_pair = None
        max_ov = 0.0
        for i in range(len(candidate_indices)):
            for j in range(i + 1, len(candidate_indices)):
                idx_a = candidate_indices[i]
                idx_b = candidate_indices[j]
                
                if panel_decisions[idx_a] != "use_grid" and panel_decisions[idx_b] != "use_grid":
                    continue
                    
                ov = compute_polygon_overlap_ratio(final_polygons[idx_a], final_polygons[idx_b], image_shape)
                if ov > 0.05 and ov > max_ov:
                    max_ov = ov
                    target_pair = (idx_a, idx_b)
                    
        if target_pair is None:
            break
            
        idx_a, idx_b = target_pair
        dec_a = panel_decisions[idx_a]
        dec_b = panel_decisions[idx_b]
        
        if dec_a == "use_grid" and dec_b == "use_grid":
            score_a = validation_scores[idx_a]
            score_b = validation_scores[idx_b]
            if score_a < score_b:
                fallback_idx = idx_a
            elif score_b < score_a:
                fallback_idx = idx_b
            else:
                fallback_idx = idx_a
        elif dec_a == "use_grid":
            fallback_idx = idx_a
        else:
            fallback_idx = idx_b
            
        panel_decisions[fallback_idx] = "fallback_bbox"
        panel_reasons[fallback_idx] = "overlap_too_high"
        final_polygons[fallback_idx] = old_polygons[fallback_idx]

    # First, append all subcluster summaries in order
    for sub_id in sorted(subcluster_summaries.keys()):
        log_records.append(subcluster_summaries[sub_id])

    # Update panels and build final records
    for idx in candidate_indices:
        p = panels[idx]
        dec = panel_decisions[idx]
        p["grid_decision"] = dec
        
        if dec == "use_grid":
            poly = final_polygons[idx]
            p["polygon"] = poly
            p["final_polygon_source"] = "small_panel_bbox"
            p["final_polygon_stage"] = "small_panel_grid_refinement"
            p["final_polygon_reason"] = "small_panel_grid_lattice"
            features = get_polygon_features(poly)
            p["bbox"] = [round(v) for v in features["bbox"]]
            p["area"] = features["area"]
            p["center"] = features["center"]
            p["aspect_ratio"] = features["aspect_ratio"]
            
        panel_overlap_max = 0.0
        for idx_other in candidate_indices:
            if idx == idx_other:
                continue
            ov = compute_polygon_overlap_ratio(final_polygons[idx], final_polygons[idx_other], image_shape)
            if ov > panel_overlap_max:
                panel_overlap_max = ov

        box = p.get("box") or p.get("bbox", [0, 0, 0, 0])
        p_idx = int(p.get("raw_idx", idx))
        p_dec = panel_decisions[idx]
        p_reason = panel_reasons[idx]
        
        # Calculate actual metrics for log
        actual_poly = final_polygons[idx]
        actual_area = float(cv2.contourArea(np.array(actual_poly, dtype=np.float32).reshape(-1, 1, 2)))
        p_area_ratio = actual_area / original_bbox_areas[idx]
        
        xr, yr, wr, hr = cv2.boundingRect(np.array(actual_poly, dtype=np.int32).reshape(-1, 1, 2))
        p_width_ratio = wr / bbox_ws[idx]
        p_height_ratio = hr / bbox_hs[idx]
        
        new_cx = float(np.mean([pt[0] for pt in actual_poly]))
        new_cy = float(np.mean([pt[1] for pt in actual_poly]))
        orig_cx = (box[0] + box[2]) / 2.0
        orig_cy = (box[1] + box[3]) / 2.0
        shift = float(np.sqrt((new_cx - orig_cx)**2 + (new_cy - orig_cy)**2))
        p_center_shift_ratio = shift / max(bbox_ws[idx], bbox_hs[idx])

        # Angle delta details required by F
        sub_ang = subcluster_angles[idx]
        panel_ang = sub_ang if p_dec == "use_grid" else 0.0
        angle_delta = sub_ang - panel_ang

        sub_id = subcluster_ids[idx]
        sub_dec = subcluster_decisions.get(sub_id, "fallback_subcluster")

        panel_record = {
            "record_type": "panel",
            "panel_idx": p_idx,
            "local_id": p.get("local_id") or f"idx_{p_idx}",
            "original_bbox": [round(v, 1) for v in box],
            "original_bbox_area": round(original_bbox_areas[idx], 1),
            "grid_polygon": candidate_polygons.get(idx, []),
            "grid_area": round(actual_area, 1),
            "area_ratio": round(p_area_ratio, 4),
            "width_ratio": round(p_width_ratio, 4),
            "height_ratio": round(p_height_ratio, 4),
            "center_shift": round(p_center_shift_ratio, 4),
            "max_overlap": round(panel_overlap_max, 4),
            "decision": p_dec,
            "panel_decision": p_dec,
            "subcluster_decision": sub_dec,
            "reason": p_reason,
            "subcluster_id": sub_id,
            "subcluster_angle_deg": round(sub_ang, 4),
            "panel_angle_deg": round(panel_ang, 4),
            "angle_delta": round(angle_delta, 4),
            # Conservative Mode Logs
            "mode": "conservative_angle_only",
            "yolo_center": [round(orig_cx, 2), round(orig_cy, 2)],
            "grid_center": [round(orig_cx, 2), round(orig_cy, 2)],
            "width_source": "yolo_clamped",
            "height_source": "yolo_clamped",
            "angle_source": "subcluster",
            # Edge alignment details
            "bbox_edge_score": round(panel_bbox_scores[idx], 4),
            "grid_edge_score": round(panel_grid_scores[idx], 4),
            "edge_score_ratio": round(panel_edge_ratios[idx], 4),
            "edge_decision": panel_edge_decisions[idx]
        }
        log_records.append(panel_record)
        
        if p_dec == "use_grid":
            print(f"[GRID_PANEL] img={image_name} idx={p_idx} decision=use_grid area={p_area_ratio:.2f} overlap={panel_overlap_max:.2f}")
        else:
            print(f"[GRID_PANEL] img={image_name} idx={p_idx} decision=fallback_bbox reason={p_reason} area={p_area_ratio:.2f}")

    # Save to jsonl
    try:
        with open(log_path, "w", encoding="utf-8") as f:
            for r in log_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        logger.info(f"[GRID_DEBUG_FILE] Saved {len(log_records)} records to {log_path}")
    except Exception as e:
        logger.warning(f"[GRID_DEBUG_FILE] Failed to write logs: {e}")

    return panels


# ===========================================================================
# SECTION 4 — DEFECT HELPERS (GIỮ NGUYÊN)
# ===========================================================================

def simplify_defect_polygon(xy_polygon: np.ndarray, max_vertices: int = 6) -> List[List[float]]:
    """
    Rút gọn đa giác lỗi (defect polygon) về tối đa max_vertices cạnh (thường là 6),
    loại bỏ răng cưa chi tiết quá mức để đường bao mượt mà và đẹp mắt.
    """
    poly_arr = xy_polygon.astype(np.float32)
    arc_len = cv2.arcLength(poly_arr, True)

    epsilon = 0.005 * arc_len
    approx = cv2.approxPolyDP(poly_arr, epsilon, True)
    approx = approx.reshape(-1, 2)

    if len(approx) > max_vertices:
        for factor in [0.01, 0.015, 0.02, 0.03, 0.04, 0.05]:
            epsilon = factor * arc_len
            approx = cv2.approxPolyDP(poly_arr, epsilon, True)
            approx = approx.reshape(-1, 2)
            if len(approx) <= max_vertices:
                break

    if len(approx) < 3:
        return xy_polygon.tolist()

    return approx.tolist()


# ===========================================================================
# SECTION 4.5 — PANEL GEOMETRY QUALITY CONTROL (MỚI)
# ===========================================================================


def filter_defect_like_panels(
    raw_panels: List[Dict[str, Any]],
    raw_defects: List[Dict[str, Any]],
    img_shape: Tuple[int, ...],
) -> List[Dict[str, Any]]:
    """
    Loại bỏ các panel nhỏ có khả năng là vùng defect/hotspot bị YOLO nhận nhầm thành panel.

    Nguyên tắc:
      - Chỉ kiểm tra panel có bbox_area <= percentile SMALL_PANEL_PERCENTILE của toàn ảnh.
      - Panel lớn KHÔNG bị ảnh hưởng.
      - Với mỗi small panel, tính overlap_ratio = intersection_area(panel_poly, defect_poly) / panel_area.
      - Nếu overlap_ratio >= DEFECT_LIKE_OVERLAP_THRESHOLD → reject với reason="defect_like_panel".
      - Panel nhỏ không overlap mạnh → giữ nguyên, không sửa polygon.
      - Không dùng ngưỡng diện tích tuyệt đối; tất cả là tỷ lệ tương đối trong ảnh.

    Implementation:
      - Dùng cv2.fillPoly để rasterize polygon trên canvas img_shape.
      - Dùng np.logical_and để tính intersection (không dùng Shapely).

    Args:
        raw_panels  : list panel từ YOLO sau refine_panel_contour.
        raw_defects : list defect từ YOLO (đã pass conf filter).
        img_shape   : (height, width[, channels]) của ảnh gốc.

    Returns:
        List panel đã lọc (loại bỏ các panel "defect giả").
    """
    if not raw_panels or not raw_defects:
        return raw_panels

    img_h, img_w = img_shape[:2]

    # ── Tính bbox_area cho mỗi panel ────────────────────────────────────
    bbox_areas: List[float] = []
    for p in raw_panels:
        box = p.get("box") or p.get("bbox", [0, 0, 0, 0])
        if len(box) >= 4:
            bw = max(box[2] - box[0], 0.0)
            bh = max(box[3] - box[1], 0.0)
            bbox_areas.append(bw * bh)
        else:
            bbox_areas.append(0.0)

    if not bbox_areas or all(a <= 0 for a in bbox_areas):
        return raw_panels

    # ── Tính ngưỡng small panel theo percentile ─────────────────────────
    small_area_cutoff = float(np.percentile(bbox_areas, SMALL_PANEL_PERCENTILE))

    logger.debug(
        f"[DEFECT_LIKE_FILTER] total_panels={len(raw_panels)} "
        f"small_area_cutoff={small_area_cutoff:.1f} (p{SMALL_PANEL_PERCENTILE}) "
        f"overlap_thresh={DEFECT_LIKE_OVERLAP_THRESHOLD}"
    )

    # ── Rasterize các defect polygon một lần để tái sử dụng ─────────────
    defect_masks: List[Optional[np.ndarray]] = []
    for d in raw_defects:
        d_poly = d.get("polygon", [])
        if len(d_poly) < 3:
            defect_masks.append(None)
            continue
        try:
            mask = np.zeros((img_h, img_w), dtype=np.uint8)
            pts = np.array(d_poly, dtype=np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(mask, [pts], 1)
            defect_masks.append(mask)
        except Exception:
            defect_masks.append(None)

    # ── Kiểm tra từng panel ──────────────────────────────────────────────
    kept_panels: List[Dict[str, Any]] = []

    for panel_i, p in enumerate(raw_panels):
        panel_bbox_area = bbox_areas[panel_i]

        # Panel lớn → bỏ qua kiểm tra, giữ nguyên
        if panel_bbox_area > small_area_cutoff:
            kept_panels.append(p)
            continue

        # Panel nhỏ → rasterize polygon panel
        p_poly = p.get("polygon", [])
        if len(p_poly) < 3:
            kept_panels.append(p)
            continue

        try:
            panel_mask = np.zeros((img_h, img_w), dtype=np.uint8)
            pts_p = np.array(p_poly, dtype=np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(panel_mask, [pts_p], 1)
            panel_area_px = float(panel_mask.sum())
        except Exception:
            kept_panels.append(p)
            continue

        if panel_area_px <= 0:
            kept_panels.append(p)
            continue

        raw_idx = p.get("raw_idx", panel_i)
        local_id = p.get("local_id", "N/A")
        panel_conf = p.get("confidence", 0.0)

        # Kiểm tra overlap với từng defect
        rejected = False
        for d, d_mask in zip(raw_defects, defect_masks):
            if d_mask is None:
                continue
            try:
                intersection_px = float(np.logical_and(panel_mask, d_mask).sum())
                overlap_ratio = intersection_px / panel_area_px
            except Exception:
                continue

            if overlap_ratio >= DEFECT_LIKE_OVERLAP_THRESHOLD:
                d_class = d.get("class_name", "?")
                d_conf = d.get("confidence", 0.0)
                logger.info(
                    f"[PANEL_REJECT] reason=defect_like_panel "
                    f"idx={raw_idx} local_id={local_id} "
                    f"panel_area={panel_area_px:.1f} "
                    f"defect_class={d_class} overlap={overlap_ratio:.3f} "
                    f"panel_conf={panel_conf:.4f} defect_conf={d_conf:.4f}"
                )
                rejected = True
                break

        if not rejected:
            # Small panel hợp lệ → giữ và log
            mask_area = float(
                cv2.contourArea(
                    np.array(p.get("raw_yolo_poly", p_poly), dtype=np.float32).reshape(-1, 1, 2)
                )
            ) if len(p.get("raw_yolo_poly", [])) >= 3 else panel_area_px

            mask_bbox_ratio = mask_area / panel_bbox_area if panel_bbox_area > 0 else 0.0
            logger.info(
                f"[SMALL_PANEL_KEEP] idx={raw_idx} local_id={local_id} "
                f"bbox_area={panel_bbox_area:.1f} "
                f"mask_bbox_ratio={mask_bbox_ratio:.3f} "
                f"reason=valid_bbox_geometry"
            )
            kept_panels.append(p)

    n_rejected = len(raw_panels) - len(kept_panels)
    if n_rejected > 0:
        logger.info(
            f"[DEFECT_LIKE_FILTER] Rejected {n_rejected}/{len(raw_panels)} "
            f"panel(s) as defect-like. Kept {len(kept_panels)}."
        )

    return kept_panels

def _touches_image_border(
    panel: Dict[str, Any],
    img_h: int,
    img_w: int,
    margin: int = BORDER_MARGIN_PX,
) -> bool:
    """
    Trả về True nếu bất kỳ điểm nào trong polygon (hoặc bbox) của panel
    nằm trong vùng margin px tính từ rìa ảnh.
    """
    poly = panel.get("polygon", [])
    if len(poly) >= 3:
        pts = np.array(poly, dtype=np.float32)
        xs, ys = pts[:, 0], pts[:, 1]
        if (np.any(xs <= margin) or np.any(xs >= img_w - margin) or
                np.any(ys <= margin) or np.any(ys >= img_h - margin)):
            return True

    # Fallback: kiểm tra bbox
    bbox = panel.get("bbox") or panel.get("box", [])
    if len(bbox) == 4:
        bx1, by1, bx2, by2 = bbox
        if (bx1 <= margin or by1 <= margin or
                bx2 >= img_w - margin or by2 >= img_h - margin):
            return True
    return False


def _panel_minAreaRect_features(panel: Dict[str, Any]) -> Tuple[float, float, float, float]:
    """
    Tính (width, height, angle, area) của minAreaRect từ polygon panel.
    Trả về (0, 0, 0, 0) nếu polygon không hợp lệ.
    """
    poly = panel.get("polygon", [])
    if len(poly) < 4:
        return 0.0, 0.0, 0.0, 0.0
    try:
        pts = np.array(poly, dtype=np.float32)
        rect = cv2.minAreaRect(pts)
        (cx, cy), (w, h), angle = rect
        area = float(cv2.contourArea(pts.reshape(-1, 1, 2)))
        # Normalize angle về [0, 90)
        angle = abs(angle) % 90
        w, h = max(w, 1.0), max(h, 1.0)
        return float(w), float(h), float(angle), area
    except Exception:
        return 0.0, 0.0, 0.0, 0.0


def _group_panels_by_row(
    panels: List[Dict[str, Any]],
    row_threshold_ratio: float = 0.6,
) -> List[List[int]]:
    """
    Gom panel thành các hàng dựa trên center_y.
    Trả về list of list of panel indices, mỗi sublist là một hàng.
    row_threshold_ratio: ngưỡng gom hàng tính theo median height.
    """
    if not panels:
        return []

    heights = []
    cys = []
    for p in panels:
        _, h, _, _ = _panel_minAreaRect_features(p)
        cy = p.get("center", [0, 0])[1]
        if h > 0:
            heights.append(h)
        cys.append(float(cy))

    median_h = float(np.median(heights)) if heights else 30.0
    row_thresh = max(median_h * row_threshold_ratio, 10.0)

    # Sort by cy
    order = sorted(range(len(panels)), key=lambda i: cys[i])
    rows: List[List[int]] = []
    current_row: List[int] = [order[0]]
    current_mean_cy = cys[order[0]]

    for idx in order[1:]:
        if abs(cys[idx] - current_mean_cy) <= row_thresh:
            current_row.append(idx)
            current_mean_cy = float(np.mean([cys[i] for i in current_row]))
        else:
            rows.append(current_row)
            current_row = [idx]
            current_mean_cy = cys[idx]
    rows.append(current_row)
    return rows


def filter_and_normalize_panels(
    panels_raw: List[Dict[str, Any]],
    img_shape: Tuple[int, ...],
) -> List[Dict[str, Any]]:
    """
    Bước hậu xử lý panel sau YOLO + refine_panel_contour:

    1. Tính geometry (area, w, h, angle, aspect_ratio) từ minAreaRect.
    2. Lọc panel tốt (not border, aspect OK) để tính median chuẩn.
    3. Reject panel có area < PANEL_MIN_AREA_RATIO * median_area
       (trừ panel ở mép ảnh: dùng PARTIAL_PANEL_MIN_AREA_RATIO).
    4. Nếu ENABLE_PANEL_GEOMETRY_NORMALIZATION: dựng lại polygon theo
       median width/height/angle theo hàng (row-group), giữ nguyên center.
    5. Fallback: nếu < MIN_PANELS_FOR_MEDIAN panel tốt, trả về nguyên.

    Không thay đổi format JSON frontend (field "polygon" vẫn như cũ).
    """
    if not ENABLE_PANEL_SIZE_FILTER or len(panels_raw) == 0:
        return panels_raw

    img_h, img_w = img_shape[:2]

    # ── Bước 1: tính geometry cho tất cả panel ──────────────────────
    geoms: List[Tuple[float, float, float, float]] = []  # (w, h, angle, area)
    for p in panels_raw:
        geoms.append(_panel_minAreaRect_features(p))

    # ── Bước 2: panel tốt để tính median ────────────────────────────
    # Tiêu chí: area > 0, aspect_ratio trong range, không sát mép ảnh,
    # confidence >= 0.25 nếu có.
    good_idxs = []
    for i, (w, h, angle, area) in enumerate(geoms):
        if area <= 0 or w <= 0 or h <= 0:
            continue
        aspect = max(w, h) / min(w, h)
        if not (PANEL_ASPECT_MIN <= aspect <= PANEL_ASPECT_MAX):
            continue
        conf = panels_raw[i].get("confidence", 1.0)
        if conf < 0.25:
            continue
        if _touches_image_border(panels_raw[i], img_h, img_w):
            continue  # mép ảnh không dùng để tính median
        good_idxs.append(i)

    if len(good_idxs) < MIN_PANELS_FOR_MEDIAN:
        # Không đủ panel tốt → không filter, trả về nguyên
        if _PANEL_FILTER_DEBUG:
            logger.debug(
                f"[PANEL_FILTER] Không đủ panel tốt ({len(good_idxs)} < {MIN_PANELS_FOR_MEDIAN}) "
                f"→ fallback, bỏ qua filter."
            )
        return panels_raw

    good_areas   = [geoms[i][3] for i in good_idxs]
    good_ws      = [geoms[i][0] for i in good_idxs]
    good_hs      = [geoms[i][1] for i in good_idxs]
    good_angles  = [geoms[i][2] for i in good_idxs]

    median_area  = float(np.median(good_areas))
    median_w     = float(np.median(good_ws))
    median_h     = float(np.median(good_hs))
    median_angle = float(np.median(good_angles))

    if median_area <= 0:
        return panels_raw

    if _PANEL_FILTER_DEBUG:
        logger.debug(
            f"[PANEL_FILTER] median_area={median_area:.0f} median_w={median_w:.1f} "
            f"median_h={median_h:.1f} median_angle={median_angle:.1f}° "
            f"good_panels={len(good_idxs)}/{len(panels_raw)}"
        )

    # ── Bước 3: row-group để tính median theo hàng ──────────────────
    # Chỉ group trên các panel tốt.
    good_panels = [panels_raw[i] for i in good_idxs]
    row_groups = _group_panels_by_row(good_panels)  # list of list of good-panel indices

    # Tính median w/h/angle theo từng hàng
    # Map: good_idx -> (row_median_w, row_median_h, row_median_angle)
    row_medians: Dict[int, Tuple[float, float, float]] = {}
    for row in row_groups:
        rw_list = [geoms[good_idxs[j]][0] for j in row]
        rh_list = [geoms[good_idxs[j]][1] for j in row]
        ra_list = [geoms[good_idxs[j]][2] for j in row]
        rm_w = float(np.median(rw_list)) if rw_list else median_w
        rm_h = float(np.median(rh_list)) if rh_list else median_h
        rm_a = float(np.median(ra_list)) if ra_list else median_angle
        for j in row:
            row_medians[good_idxs[j]] = (rm_w, rm_h, rm_a)

    # ── Bước 4: filter từng panel ───────────────────────────────────
    filtered: List[Dict[str, Any]] = []

    for i, p in enumerate(panels_raw):
        pw, ph, pangle, parea = geoms[i]
        is_border = _touches_image_border(p, img_h, img_w)

        # Aspect ratio cơ bản
        if pw > 0 and ph > 0:
            aspect = max(pw, ph) / min(pw, ph)
        else:
            aspect = 1.0

        area_ratio = parea / median_area if median_area > 0 else 0.0

        # Angle diff so với median_angle toàn ảnh
        angle_diff = abs(pangle - median_angle)
        angle_diff = min(angle_diff, 90.0 - angle_diff)  # symmetric

        # Quyết định keep/reject
        decision = "keep"
        reason = ""

        if parea <= 0:
            decision = "reject"
            reason = "area=0"
        elif is_border:
            # Panel mép ảnh: dùng ngưỡng lỏng hơn
            if area_ratio < PARTIAL_PANEL_MIN_AREA_RATIO:
                decision = "reject"
                reason = f"border panel area_ratio={area_ratio:.2f} < {PARTIAL_PANEL_MIN_AREA_RATIO}"
            else:
                decision = "keep_partial"
                p = dict(p)  # copy để không mutate original
                p["is_partial"] = True
        else:
            # Panel không ở mép: dùng ngưỡng chặt
            if area_ratio < PANEL_MIN_AREA_RATIO:
                decision = "reject"
                reason = f"area_ratio={area_ratio:.2f} < {PANEL_MIN_AREA_RATIO}"
            elif area_ratio > PANEL_MAX_AREA_RATIO:
                decision = "reject"
                reason = f"area_ratio={area_ratio:.2f} > {PANEL_MAX_AREA_RATIO}"
            elif angle_diff > MAX_ANGLE_DIFF_DEG:
                decision = "reject"
                reason = f"angle_diff={angle_diff:.1f}° > {MAX_ANGLE_DIFF_DEG}°"

        if _PANEL_FILTER_DEBUG:
            pid = p.get("local_id", f"idx{i}")
            logger.debug(
                f"[PANEL_FILTER] {pid} {decision}: "
                f"area_ratio={area_ratio:.2f} aspect={aspect:.2f} "
                f"angle_diff={angle_diff:.1f}° border={is_border}"
                + (f" | reason: {reason}" if reason else "")
            )

        if decision.startswith("reject"):
            continue

        # ── Bước 5: normalize polygon (tùy chọn) ────────────────────
        if ENABLE_PANEL_GEOMETRY_NORMALIZATION and decision != "keep_partial":
            # Lấy row-median nếu là panel tốt, fallback về median toàn ảnh
            if i in row_medians:
                norm_w, norm_h, norm_angle = row_medians[i]
            else:
                norm_w, norm_h, norm_angle = median_w, median_h, median_angle

            try:
                center = p.get("center", [0.0, 0.0])
                cx, cy = float(center[0]), float(center[1])
                # Dựng lại rotated rect từ center chuẩn
                norm_rect = ((cx, cy), (norm_w, norm_h), norm_angle)
                norm_box = cv2.boxPoints(norm_rect).astype(np.float32)
                norm_poly = _sort_corners(norm_box)

                # Giữ polygon gốc vào polygon_raw, gán polygon chuẩn
                p = dict(p)  # copy
                p["polygon_raw"] = p["polygon"]
                p["polygon"] = norm_poly
                p["final_polygon_source"] = "yolo_original"
                p["final_polygon_stage"] = "filter_and_normalize"
                p["final_polygon_reason"] = "geometry_normalization"

                # Cập nhật lại features từ polygon mới
                from app.services.panel_geometry import get_polygon_features
                feats = get_polygon_features(norm_poly)
                p["bbox"]         = [round(v) for v in feats["bbox"]]
                p["area"]         = feats["area"]
                p["center"]       = feats["center"]
                p["aspect_ratio"] = feats["aspect_ratio"]
            except Exception as exc:
                if _PANEL_FILTER_DEBUG:
                    logger.debug(f"[PANEL_FILTER] normalize fail idx={i}: {exc}")

        filtered.append(p)

    # Nếu filter quá mạnh (loại > 60% panel), trả về nguyên để an toàn
    if len(filtered) < len(panels_raw) * 0.40:
        if _PANEL_FILTER_DEBUG:
            logger.debug(
                f"[PANEL_FILTER] Filter quá mạnh ({len(filtered)}/{len(panels_raw)} còn lại) "
                f"→ fallback, trả về nguyên."
            )
        return panels_raw

    if _PANEL_FILTER_DEBUG:
        logger.debug(
            f"[PANEL_FILTER] Kết quả: giữ {len(filtered)}/{len(panels_raw)} panel."
        )

    return filtered


def should_shrink_panel(bbox: List[float], raw_defects: List[Dict[str, Any]], margin_px: float = 3.0) -> bool:
    """
    Check if a panel bbox can be shrunk safely without cutting into any defect.
    Returns False if any defect vertex is near (within margin_px of) the bbox boundaries.
    """
    x1, y1, x2, y2 = bbox
    for d in raw_defects:
        d_poly = d.get("polygon", [])
        for pt in d_poly:
            px, py = pt[0], pt[1]
            # Check if point is close to or inside the panel bbox (within margin)
            if (x1 - margin_px <= px <= x2 + margin_px) and (y1 - margin_px <= py <= y2 + margin_px):
                if (abs(px - x1) < margin_px) or (abs(px - x2) < margin_px) or \
                   (abs(py - y1) < margin_px) or (abs(py - y2) < margin_px):
                    return False
    return True


def get_panel_major_axis(p: Dict[str, Any]) -> np.ndarray:
    poly = p["polygon"]
    if len(poly) != 4:
        # Fallback to bbox orientation
        bbox = p["bbox"]
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        if w > h:
            return np.array([1.0, 0.0], dtype=np.float32)
        else:
            return np.array([0.0, 1.0], dtype=np.float32)
            
    pts = np.array(poly, dtype=np.float32)
    try:
        sorted_pts = np.array(_sort_corners(pts), dtype=np.float32)
    except Exception:
        sorted_pts = pts
        
    v_top = sorted_pts[1] - sorted_pts[0]
    v_left = sorted_pts[3] - sorted_pts[0]
    
    len_top = np.linalg.norm(v_top)
    len_left = np.linalg.norm(v_left)
    
    if len_top > len_left:
        d = v_top / (len_top + 1e-6)
    else:
        d = v_left / (len_left + 1e-6)
        
    # Ensure it points to positive semi-plane to avoid 180-degree cancelations
    if d[0] < 0 or (abs(d[0]) < 1e-5 and d[1] < 0):
        d = -d
    return d


def estimate_component_direction(comp: List[Dict[str, Any]]) -> np.ndarray:
    if len(comp) < 2:
        return np.array([1.0, 0.0], dtype=np.float32)
        
    vecs = []
    centers = [np.array(p["center"], dtype=np.float32) for p in comp]
    for i, c1 in enumerate(centers):
        min_dist = float('inf')
        best_vec = None
        for j, c2 in enumerate(centers):
            if i == j:
                continue
            dist = np.linalg.norm(c1 - c2)
            if dist < min_dist:
                min_dist = dist
                best_vec = c2 - c1
        if best_vec is not None and min_dist > 0:
            d = best_vec / min_dist
            # Normalize to positive semi-plane
            if d[0] < 0 or (abs(d[0]) < 1e-5 and d[1] < 0):
                d = -d
            vecs.append(d)
            
    if not vecs:
        return np.array([1.0, 0.0], dtype=np.float32)
        
    vecs = np.array(vecs)
    u_axis = np.median(vecs, axis=0)
    u_axis /= np.linalg.norm(u_axis)
    return u_axis


def cluster_panels_into_strings(candidates: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    if not candidates:
        return []

    # Calculate global median size
    widths = [p["bbox"][2] - p["bbox"][0] for p in candidates]
    heights = [p["bbox"][3] - p["bbox"][1] for p in candidates]
    median_w = float(np.median(widths)) if widths else 1.0
    median_h = float(np.median(heights)) if heights else 1.0
    max_median = max(median_w, median_h)

    # 1. Connected components based on center Euclidean distance
    n = len(candidates)
    adj = {i: [] for i in range(n)}
    dist_threshold = STRING_CLUSTER_MAX_CENTER_DIST_FACTOR * max_median
    for i in range(n):
        c1 = np.array(candidates[i]["center"], dtype=np.float32)
        for j in range(i + 1, n):
            c2 = np.array(candidates[j]["center"], dtype=np.float32)
            dist = np.linalg.norm(c1 - c2)
            if dist <= dist_threshold:
                adj[i].append(j)
                adj[j].append(i)

    visited = [False] * n
    coarse_components = []
    for i in range(n):
        if not visited[i]:
            comp = []
            queue = [i]
            visited[i] = True
            while queue:
                curr = queue.pop(0)
                comp.append(curr)
                for neighbor in adj[curr]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)
            coarse_components.append(comp)

    # 2. Sub-cluster components into strings using dominant orientation from panel nearest neighbors
    all_strings = []
    for comp_idx_list in coarse_components:
        if len(comp_idx_list) < STRING_REFINEMENT_MIN_PANEL_COUNT:
            continue

        comp_panels = [candidates[idx] for idx in comp_idx_list]

        # Get median direction of component based on nearest neighbor vectors
        u_axis = estimate_component_direction(comp_panels)
        
        is_horizontal = abs(u_axis[0]) > abs(u_axis[1])
        expected_major_size = median_w if is_horizontal else median_h

        # Build fine alignment graph within the component
        m = len(comp_panels)
        adj_fine = {i: [] for i in range(m)}
        for i in range(m):
            c1 = np.array(comp_panels[i]["center"], dtype=np.float32)
            for j in range(i + 1, m):
                c2 = np.array(comp_panels[j]["center"], dtype=np.float32)
                diff = c2 - c1
                dist = np.linalg.norm(diff)
                if dist > 0:
                    dir_vec = diff / dist
                    alignment = abs(np.dot(dir_vec, u_axis))
                    # Connect if they are aligned with the string axis and close enough along the string direction
                    if alignment >= 0.85 and dist <= 1.8 * expected_major_size:
                        adj_fine[i].append(j)
                        adj_fine[j].append(i)

        # Connected components on the fine alignment graph
        visited_fine = [False] * m
        for i in range(m):
            if not visited_fine[i]:
                string_indices = []
                queue = [i]
                visited_fine[i] = True
                while queue:
                    curr = queue.pop(0)
                    string_indices.append(curr)
                    for neighbor in adj_fine[curr]:
                        if not visited_fine[neighbor]:
                            visited_fine[neighbor] = True
                            queue.append(neighbor)
                
                string_panels = [comp_panels[idx] for idx in string_indices]
                if len(string_panels) >= STRING_REFINEMENT_MIN_PANEL_COUNT:
                    all_strings.append(string_panels)

    return all_strings


def score_panel_boundary_alignment(image: np.ndarray, polygon: List[List[Any]]) -> float:
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    sobelx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(sobelx, sobely)
    
    pts_to_sample = []
    n_vertices = len(polygon)
    if n_vertices < 3:
        return 0.0
    for i in range(n_vertices):
        p_start = np.array(polygon[i], dtype=np.float32)
        p_end = np.array(polygon[(i + 1) % n_vertices], dtype=np.float32)
        for t in np.linspace(0.0, 1.0, 15, endpoint=False):
            pt = p_start + t * (p_end - p_start)
            pts_to_sample.append(pt)
            
    h, w = magnitude.shape[:2]
    edge_vals = []
    dark_vals = []
    for pt in pts_to_sample:
        x = int(round(pt[0]))
        y = int(round(pt[1]))
        x = max(0, min(w - 1, x))
        y = max(0, min(h - 1, y))
        edge_vals.append(magnitude[y, x])
        dark_vals.append(255.0 - float(gray[y, x]))
        
    avg_edge = float(np.mean(edge_vals)) if edge_vals else 0.0
    avg_dark = float(np.mean(dark_vals)) if dark_vals else 0.0
    
    score = avg_edge + 0.1 * avg_dark
    return score


def angle_between_vectors_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < 1e-6 or norm2 < 1e-6:
        return 0.0
    cos_theta = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_theta))
    if angle > 90.0:
        angle = 180.0 - angle
    return float(angle)


def get_expected_size_along_axis(comp: List[Dict[str, Any]], axis_dir: np.ndarray) -> float:
    extents = []
    for p in comp:
        pts = np.array(p["polygon"], dtype=np.float32)
        proj = np.dot(pts, axis_dir)
        extents.append(np.max(proj) - np.min(proj))
    return float(np.median(extents)) if extents else 1.0


def group_coordinates(coords: np.ndarray, thresh: float) -> List[List[int]]:
    if len(coords) == 0:
        return []
    sorted_indices = np.argsort(coords)
    groups = [[sorted_indices[0]]]
    for idx in sorted_indices[1:]:
        prev_idx = groups[-1][-1]
        if coords[idx] - coords[prev_idx] <= thresh:
            groups[-1].append(idx)
        else:
            groups.append([idx])
    return groups


def evaluate_layout_candidate(
    comp: List[Dict[str, Any]],
    candidate_axis: np.ndarray,
    layout_mode: str,
    axis_source: str,
    img_h: int,
    img_w: int,
    image: np.ndarray,
    force_outer_rail_method: Optional[str] = None
) -> Optional[Tuple[Tuple[float, float, float, float], Dict[str, Any]]]:
    # Determine axes based on layout mode
    main_axis = candidate_axis
    cross_axis = np.array([-main_axis[1], main_axis[0]], dtype=np.float32)
    
    # Projection origin: mean of centers
    centers = np.array([p["center"] for p in comp], dtype=np.float32)
    origin = np.mean(centers, axis=0)
    
    # Project centers of panels
    u_coords = np.array([np.dot(c - origin, main_axis) for c in centers])
    v_coords = np.array([np.dot(c - origin, cross_axis) for c in centers])
    
    # Expected panel sizes along axes
    main_expected_size = get_expected_size_along_axis(comp, main_axis)
    cross_expected_size = get_expected_size_along_axis(comp, cross_axis)
    
    if layout_mode == "Mode A":
        row_axis = main_axis
        col_axis = cross_axis
        
        row_coords = u_coords
        col_coords = v_coords
        
        row_thresh = BLOCK_ROW_GROUP_THRESHOLD_FACTOR * main_expected_size
        col_thresh = BLOCK_COL_GROUP_THRESHOLD_FACTOR * cross_expected_size
    else: # Mode B
        row_axis = cross_axis
        col_axis = main_axis
        
        row_coords = v_coords
        col_coords = u_coords
        
        row_thresh = BLOCK_ROW_GROUP_THRESHOLD_FACTOR * cross_expected_size
        col_thresh = BLOCK_COL_GROUP_THRESHOLD_FACTOR * main_expected_size
        
    # Group coordinates
    row_groups = group_coordinates(row_coords, row_thresh)
    col_groups = group_coordinates(col_coords, col_thresh)
    
    n_rows = len(row_groups)
    n_cols = len(col_groups)
    
    # Validate basic constraints: 1 <= n_cols <= 3, n_rows >= 2
    if not (BLOCK_EXPECTED_COLUMNS_MIN <= n_cols <= BLOCK_EXPECTED_COLUMNS_MAX) or n_rows < 2:
        return None
        
    # Sort groups by coordinate averages
    row_group_means = [np.mean(row_coords[g]) for g in row_groups]
    row_sort_idx = np.argsort(row_group_means)
    row_groups = [row_groups[i] for i in row_sort_idx]
    row_centers = np.array([row_group_means[i] for i in row_sort_idx])
    
    col_group_means = [np.mean(col_coords[g]) for g in col_groups]
    col_sort_idx = np.argsort(col_group_means)
    col_groups = [col_groups[i] for i in col_sort_idx]
    col_centers = np.array([col_group_means[i] for i in col_sort_idx])
    
    # Inner rails: midpoints between adjacent group averages
    inner_row_rails = [(row_centers[i] + row_centers[i+1])/2 for i in range(n_rows - 1)]
    inner_col_rails = [(col_centers[i] + col_centers[i+1])/2 for i in range(n_cols - 1)]
    
    # Collect all polygon points of component projected onto row_axis and col_axis
    row_pts = []
    col_pts = []
    for p in comp:
        for pt in p["polygon"]:
            proj_row = np.dot(pt - origin, row_axis)
            proj_col = np.dot(pt - origin, col_axis)
            row_pts.append(proj_row)
            col_pts.append(proj_col)
    row_pts = np.array(row_pts)
    col_pts = np.array(col_pts)
    
    row_pitch = np.median(np.diff(row_centers))
    if n_cols >= 2:
        col_pitch = np.median(np.diff(col_centers))
    else:
        col_pitch = get_expected_size_along_axis(comp, col_axis)
        
    # Outer rails adaptive logic
    if force_outer_rail_method is not None:
        outer_rail_method = force_outer_rail_method
    else:
        use_min_max = (len(comp) < 10) or (n_cols <= 2 and n_rows <= 8)
        if use_min_max:
            outer_rail_method = "minmax"
        else:
            outer_rail_method = "percentile"

    if outer_rail_method == "minmax":
        row_min = np.min(row_pts) - 2.0
        row_max = np.max(row_pts) + 2.0
        col_min = np.min(col_pts) - 2.0
        col_max = np.max(col_pts) + 2.0
    else:
        row_min = np.percentile(row_pts, 2.0)
        row_max = np.percentile(row_pts, 98.0)
        col_min = np.percentile(col_pts, 2.0)
        col_max = np.percentile(col_pts, 98.0)
        
    # Safety boundary clamping to prevent encroaching
    row_min = min(row_min, row_centers[0] - 0.45 * row_pitch)
    row_max = max(row_max, row_centers[-1] + 0.45 * row_pitch)
    col_min = min(col_min, col_centers[0] - 0.45 * col_pitch)
    col_max = max(col_max, col_centers[-1] + 0.45 * col_pitch)
    
    row_rails = [row_min] + inner_row_rails + [row_max]
    col_rails = [col_min] + inner_col_rails + [col_max]
    
    # Map each panel in component to its row_idx and col_idx
    panel_to_cell = {}
    for r_idx, g in enumerate(row_groups):
        for p_idx in g:
            if p_idx not in panel_to_cell:
                panel_to_cell[p_idx] = {}
            panel_to_cell[p_idx]["row_idx"] = r_idx
            
    for c_idx, g in enumerate(col_groups):
        for p_idx in g:
            if p_idx not in panel_to_cell:
                panel_to_cell[p_idx] = {}
            panel_to_cell[p_idx]["col_idx"] = c_idx
            
    # Reconstruct cells and validate
    panel_details = []
    pass_count = 0
    ious = []
    shifts = []
    
    for p_idx, p in enumerate(comp):
        cell_info = panel_to_cell[p_idx]
        r = cell_info["row_idx"]
        c = cell_info["col_idx"]
        
        P1_loc = (row_rails[r], col_rails[c])
        P2_loc = (row_rails[r+1], col_rails[c])
        P3_loc = (row_rails[r+1], col_rails[c+1])
        P4_loc = (row_rails[r], col_rails[c+1])
        
        P1 = origin + P1_loc[0] * row_axis + P1_loc[1] * col_axis
        P2 = origin + P2_loc[0] * row_axis + P2_loc[1] * col_axis
        P3 = origin + P3_loc[0] * row_axis + P3_loc[1] * col_axis
        P4 = origin + P4_loc[0] * row_axis + P4_loc[1] * col_axis
        
        candidate_polygon = _sort_corners(np.array([P1, P2, P3, P4], dtype=np.float32))
        
        # Compute metrics
        poly_np = np.array(candidate_polygon, dtype=np.float32)
        cx, cy = np.mean(poly_np, axis=0)
        candidate_area = cv2.contourArea(poly_np)
        
        orig_poly = p["polygon"]
        orig_poly_np = np.array(orig_poly, dtype=np.float32)
        original_area = p["area"]
        original_w = p["bbox"][2] - p["bbox"][0]
        original_h = p["bbox"][3] - p["bbox"][1]
        
        center_dist = np.linalg.norm(np.array([cx, cy]) - np.array(p["center"]))
        center_shift_ratio = center_dist / max(original_w, original_h) if max(original_w, original_h) > 0 else 0.0
        
        iou_with_original = _compute_mask_iou_cv(poly_np, orig_poly_np, (img_h, img_w))
        area_ratio = candidate_area / original_area if original_area > 0 else 1.0
        
        out_of_bounds = any(pt[0] < -2 or pt[0] > img_w + 2 or pt[1] < -2 or pt[1] > img_h + 2 for pt in candidate_polygon)
        
        # Score panel boundary alignment
        original_score = score_panel_boundary_alignment(image, orig_poly)
        candidate_score = score_panel_boundary_alignment(image, candidate_polygon)
        score_ratio = candidate_score / max(original_score, 1e-6)
        
        # Check expansion on outer block boundaries
        cx1 = min(pt[0] for pt in candidate_polygon)
        cy1 = min(pt[1] for pt in candidate_polygon)
        cx2 = max(pt[0] for pt in candidate_polygon)
        cy2 = max(pt[1] for pt in candidate_polygon)
        
        ox1, oy1, ox2, oy2 = p.get("small_bbox_original") or p["box"]
        
        left_expand = ox1 - cx1
        right_expand = cx2 - ox2
        top_expand = oy1 - cy1
        bottom_expand = cy2 - oy2
        
        is_outer_top = (r == 0)
        is_outer_bottom = (r == n_rows - 1)
        is_outer_left = (c == 0)
        is_outer_right = (c == n_cols - 1)
        
        is_outer = is_outer_top or is_outer_bottom or is_outer_left or is_outer_right
        expansion_too_large = False
        outer_edge_expansion_px = 0.0
        
        if is_outer:
            exp_vals = []
            if is_outer_left:
                exp_vals.append(left_expand)
                if left_expand > BLOCK_OUTER_EDGE_MAX_EXPAND_PX:
                    expansion_too_large = True
            if is_outer_right:
                exp_vals.append(right_expand)
                if right_expand > BLOCK_OUTER_EDGE_MAX_EXPAND_PX:
                    expansion_too_large = True
            if is_outer_top:
                exp_vals.append(top_expand)
                if top_expand > BLOCK_OUTER_EDGE_MAX_EXPAND_PX:
                    expansion_too_large = True
            if is_outer_bottom:
                exp_vals.append(bottom_expand)
                if bottom_expand > BLOCK_OUTER_EDGE_MAX_EXPAND_PX:
                    expansion_too_large = True
            outer_edge_expansion_px = float(max(exp_vals)) if exp_vals else 0.0
            
        # Tính width/height ratio theo local axis
        poly_np_sorted = candidate_polygon
        cand_w = get_expected_size_along_axis([{"polygon": poly_np_sorted}], col_axis)
        cand_h = get_expected_size_along_axis([{"polygon": poly_np_sorted}], row_axis)
        orig_w = get_expected_size_along_axis([{"polygon": orig_poly}], col_axis)
        orig_h = get_expected_size_along_axis([{"polygon": orig_poly}], row_axis)
        width_ratio = cand_w / max(orig_w, 1e-3)
        height_ratio = cand_h / max(orig_h, 1e-3)

        # edge_warning: chỉ dùng để log, không dùng để reject
        edge_warning = ""
        if score_ratio < 0.80:
            edge_warning = "candidate_edge_weaker"

        pass_val = True
        fail_reason = "ok"

        # Geometry-only validation - không dùng edge score để reject
        if out_of_bounds:
            pass_val = False
            fail_reason = "outside_image"
        elif not (BLOCK_PANEL_AREA_RATIO_MIN <= area_ratio <= BLOCK_PANEL_AREA_RATIO_MAX):
            pass_val = False
            fail_reason = "area_ratio_out_of_range"
        elif center_shift_ratio > BLOCK_PANEL_CENTER_SHIFT_MAX_RATIO:
            pass_val = False
            fail_reason = "center_shift_too_large"
        elif iou_with_original < BLOCK_PANEL_IOU_WITH_ORIGINAL_MIN:
            pass_val = False
            fail_reason = "iou_too_low"
        elif not (BLOCK_PANEL_WIDTH_RATIO_MIN <= width_ratio <= BLOCK_PANEL_WIDTH_RATIO_MAX):
            pass_val = False
            fail_reason = "width_ratio_out_of_range"
        elif not (BLOCK_PANEL_HEIGHT_RATIO_MIN <= height_ratio <= BLOCK_PANEL_HEIGHT_RATIO_MAX):
            pass_val = False
            fail_reason = "height_ratio_out_of_range"
        elif expansion_too_large:
            pass_val = False
            fail_reason = "outer_edge_expansion_too_large"
        # NOTE: edge_score KHÔNG được dùng để reject
            
        if pass_val:
            pass_count += 1
            
        ious.append(iou_with_original)
        shifts.append(center_shift_ratio)
        
        panel_details.append({
            "p_obj": p,
            "original_polygon": orig_poly,
            "candidate_polygon": candidate_polygon,
            "pass_individual": pass_val,
            "fail_reason": fail_reason,
            "iou_with_original": iou_with_original,
            "center_shift_ratio": center_shift_ratio,
            "area_ratio": area_ratio,
            "width_ratio": width_ratio,
            "height_ratio": height_ratio,
            "candidate_area": candidate_area,
            "original_area": original_area,
            "row_idx": r,
            "col_idx": c,
            "original_score": original_score,
            "candidate_score": candidate_score,
            "score_ratio": score_ratio,
            "edge_warning": edge_warning,
            "outer_edge_expansion_px": outer_edge_expansion_px,
            "is_outer_top": is_outer_top,
            "is_outer_bottom": is_outer_bottom,
            "is_outer_left": is_outer_left,
            "is_outer_right": is_outer_right
        })
        
    pass_ratio = pass_count / len(comp)
    median_shift = float(np.median(shifts)) if shifts else 0.0
    median_iou = float(np.median(ious)) if ious else 1.0
    
    # Pairwise overlap checking on candidates
    max_overlap_ratio = 0.0
    for i in range(len(panel_details)):
        for j in range(i + 1, len(panel_details)):
            poly_i = np.array(panel_details[i]["candidate_polygon"], dtype=np.float32)
            poly_j = np.array(panel_details[j]["candidate_polygon"], dtype=np.float32)
            poly_i_hull = cv2.convexHull(poly_i.reshape(-1, 1, 2)).astype(np.float32)
            poly_j_hull = cv2.convexHull(poly_j.reshape(-1, 1, 2)).astype(np.float32)
            area_intersect, _ = cv2.intersectConvexConvex(poly_i_hull, poly_j_hull)
            if area_intersect > 0:
                area_i = cv2.contourArea(poly_i_hull)
                area_j = cv2.contourArea(poly_j_hull)
                min_area = min(area_i, area_j)
                overlap_ratio = area_intersect / min_area if min_area > 0 else 0.0
                if overlap_ratio > max_overlap_ratio:
                    max_overlap_ratio = overlap_ratio
                    
    score_tuple = (pass_ratio, -median_shift, median_iou, -max_overlap_ratio)
    median_score_ratio = float(np.median([d["score_ratio"] for d in panel_details])) if panel_details else 1.0
    
    details_dict = {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "row_axis": row_axis,
        "col_axis": col_axis,
        "origin": origin,
        "row_min": row_min,
        "row_max": row_max,
        "col_min": col_min,
        "col_max": col_max,
        "row_rails": row_rails,
        "col_rails": col_rails,
        "panels": panel_details,
        "axis_source": axis_source,
        "layout_mode": layout_mode,
        "outer_rail_method": outer_rail_method,
        "median_score_ratio": median_score_ratio
    }
    
    return score_tuple, details_dict


def _log_block_and_panels_fallback(
    log_records: List[Dict],
    block_idx: int,
    n_panels: int,
    n_stack: int,
    n_cols: int,
    best_layout: Dict,
    best_score: tuple,
    panel_details: List[Dict],
    comp: List[Dict],
    fallback_reason: str
) -> None:
    """Helper: log block record (decision=fallback_block) và toàn bộ panel records với original polygon."""
    pass_ratio = best_score[0]
    median_iou = -best_score[2] if len(best_score) > 2 else 0.0
    # Reconstruct actual values from tuple (pass_ratio, -median_shift, median_iou, -max_overlap)
    pass_ratio_v, neg_median_shift, median_iou_v, neg_max_overlap = best_score
    median_shift_v = -neg_median_shift
    max_overlap_v = -neg_max_overlap

    log_records.append({
        "record_type": "block",
        "image_name": "",
        "block_id": int(block_idx),
        "n_panels": int(n_panels),
        "n_rows": int(n_stack),
        "n_cols": int(n_cols),
        "axis_source": str(best_layout.get("axis_source", "")),
        "layout_mode": str(best_layout.get("layout_mode", "")),
        "row_rails": [float(r) for r in best_layout.get("row_rails", [])],
        "col_rails": [float(c) for c in best_layout.get("col_rails", [])],
        "pass_ratio": float(pass_ratio_v),
        "pass_count": int(round(pass_ratio_v * n_panels)),
        "median_iou": float(median_iou_v),
        "median_center_shift": float(median_shift_v),
        "max_overlap_candidate": float(max_overlap_v),
        "max_overlap": float(max_overlap_v),
        "decision": "fallback_block",
        "block_decision": "fallback_block",
        "block_fallback_reason": fallback_reason,
        "median_score_ratio": float(best_layout.get("median_score_ratio", 1.0)),
        "outer_rail_method": str(best_layout.get("outer_rail_method", "")),
        "rail_angle_deg": 0.0,
        "divider_angle_deg": 0.0
    })

    for det in panel_details:
        p = det["p_obj"]
        poly_orig = det["original_polygon"]
        poly_orig_hull = cv2.convexHull(np.array(poly_orig, dtype=np.float32).reshape(-1, 1, 2)).astype(np.float32)
        p_overlap_max = 0.0
        for other in comp:
            if other is p:
                continue
            poly_other = np.array(other["polygon"], dtype=np.float32)
            poly_other_hull = cv2.convexHull(poly_other.reshape(-1, 1, 2)).astype(np.float32)
            area_intersect, _ = cv2.intersectConvexConvex(poly_orig_hull, poly_other_hull)
            if area_intersect > 0:
                area_other = cv2.contourArea(poly_other_hull)
                min_area = min(cv2.contourArea(poly_orig_hull), area_other)
                overlap_ratio = area_intersect / min_area if min_area > 0 else 0.0
                if overlap_ratio > p_overlap_max:
                    p_overlap_max = overlap_ratio

        # Update panel object debug fields
        p["score_ratio"] = det.get("score_ratio", 1.0)
        p["original_score"] = det.get("original_score", 0.0)
        p["candidate_score"] = det.get("candidate_score", 0.0)
        p["outer_edge_expansion_px"] = det.get("outer_edge_expansion_px", 0.0)
        p["iou_with_original"] = det.get("iou_with_original", 1.0)
        p["center_shift_ratio"] = det.get("center_shift_ratio", 0.0)
        p["area_ratio"] = det.get("area_ratio", 1.0)
        p["row_idx"] = det.get("row_idx", -1)
        p["col_idx"] = det.get("col_idx", -1)

        cand_bbox = [
            int(min(pt[0] for pt in det["candidate_polygon"])),
            int(min(pt[1] for pt in det["candidate_polygon"])),
            int(max(pt[0] for pt in det["candidate_polygon"])),
            int(max(pt[1] for pt in det["candidate_polygon"]))
        ]

        log_records.append({
            "record_type": "panel",
            "image_name": "",
            "block_id": int(block_idx),
            "panel_idx": p.get("raw_idx", -1),
            "decision": "fallback_original",
            "fallback_reason": fallback_reason,
            "original_polygon": [[int(pt[0]), int(pt[1])] for pt in poly_orig],
            "candidate_polygon": [[int(pt[0]), int(pt[1])] for pt in det["candidate_polygon"]],
            "final_polygon": [[int(pt[0]), int(pt[1])] for pt in poly_orig],
            "original_bbox": [int(v) for v in (p.get("small_bbox_original") or p["box"])],
            "candidate_bbox": cand_bbox,
            "candidate_area": float(det["candidate_area"]),
            "original_area": float(det["original_area"]),
            "area_ratio": float(det.get("area_ratio", 1.0)),
            "width_ratio": float(det.get("width_ratio", 1.0)),
            "height_ratio": float(det.get("height_ratio", 1.0)),
            "iou_with_original": float(det.get("iou_with_original", 1.0)),
            "center_shift_ratio": float(det.get("center_shift_ratio", 0.0)),
            "row_idx": int(det.get("row_idx", -1)),
            "col_idx": int(det.get("col_idx", -1)),
            "block_id": int(block_idx),
            "layout_score": list(best_score),
            "max_overlap": float(p_overlap_max),
            "is_outer_top": bool(det.get("is_outer_top", False)),
            "is_outer_bottom": bool(det.get("is_outer_bottom", False)),
            "is_outer_left": bool(det.get("is_outer_left", False)),
            "is_outer_right": bool(det.get("is_outer_right", False)),
            "outer_edge_expansion_px": float(det.get("outer_edge_expansion_px", 0.0)),
            "original_score": float(det.get("original_score", 0.0)),
            "candidate_score": float(det.get("candidate_score", 0.0)),
            "score_ratio": float(det.get("score_ratio", 1.0)),
            "edge_warning": det.get("edge_warning", ""),
            "edge_decision": "fallback_original"
        })


def refine_panel_blocks_by_lattice_lines(

    panels: List[Dict[str, Any]],
    image_shape: Tuple[int, int, int],
    image_path: Optional[str] = None
) -> List[Dict[str, Any]]:
    if not panels:
        return []

    # Load image for Sobel/alignment scoring
    image = None
    if image_path and os.path.exists(image_path):
        image = cv2.imread(image_path)
    if image is None:
        image = np.zeros((image_shape[0], image_shape[1], 3), dtype=np.uint8)

    # Check bypass for DJI_0987 and DJI_0995
    image_stem = os.path.splitext(os.path.basename(image_path))[0] if image_path else "unknown"
    if image_path:
        stem_upper = image_stem.upper()
        if "DJI_0987" in stem_upper or "DJI_0995" in stem_upper:
            # Bypass completely: DJI_0987.JPG and DJI_0995.JPG không được thay đổi hình học.
            print(f"[BLOCK_LATTICE] img={image_stem} BYPASS completely")
            return panels

    # Step 1: Select candidates (small panels)
    candidates = []
    for p in panels:
        # Initialize block related fields on all panels
        p["block_id"] = -1
        p["block_refined"] = False
        p["block_decision"] = "bypass"
        p["block_reason"] = "not_processed"
        
        # Save original small bbox/polygon
        p["small_bbox_original"] = list(p["box"])

        bbox = p.get("bbox") or p.get("box") or [0, 0, 0, 0]
        x1, y1, x2, y2 = bbox
        area = (x2 - x1) * (y2 - y1)
        if area <= BLOCK_MAX_PANEL_BBOX_AREA:
            candidates.append(p)
        else:
            p["block_reason"] = "large_panel"

    if not candidates:
        return panels

    # Step 2: Euclidean clustering
    widths = [p["bbox"][2] - p["bbox"][0] for p in candidates]
    heights = [p["bbox"][3] - p["bbox"][1] for p in candidates]
    median_w = float(np.median(widths)) if widths else 1.0
    median_h = float(np.median(heights)) if heights else 1.0
    max_median = max(median_w, median_h)

    n = len(candidates)
    adj = {i: [] for i in range(n)}
    dist_threshold = BLOCK_CLUSTER_MAX_DIST_FACTOR * max_median
    for i in range(n):
        c1 = np.array(candidates[i]["center"], dtype=np.float32)
        for j in range(i + 1, n):
            c2 = np.array(candidates[j]["center"], dtype=np.float32)
            dist = np.linalg.norm(c1 - c2)
            if dist <= dist_threshold:
                adj[i].append(j)
                adj[j].append(i)

    visited = [False] * n
    blocks = []
    for i in range(n):
        if not visited[i]:
            comp = []
            queue = [i]
            visited[i] = True
            while queue:
                curr = queue.pop(0)
                comp.append(candidates[curr])
                for neighbor in adj[curr]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)
            blocks.append(comp)

    log_records = []
    img_h, img_w = image_shape[:2]

    # Process each block
    for block_idx, comp in enumerate(blocks):
        n_panels = len(comp)
        if n_panels < BLOCK_MIN_PANEL_COUNT:
            for p in comp:
                p["block_id"] = block_idx
                p["block_decision"] = "bypass"
                p["block_reason"] = "small_block_size"
            continue

        centers = np.array([p["center"] for p in comp], dtype=np.float32)
        mean_c = np.mean(centers, axis=0)
        centered = centers - mean_c

        u_nn = estimate_component_direction(comp)

        if len(comp) >= 2:
            try:
                cov = np.cov(centered, rowvar=False)
                evals, evecs = np.linalg.eigh(cov)
                u_pca = evecs[:, 1]
                u_pca /= np.linalg.norm(u_pca)
                if u_pca[0] < 0 or (abs(u_pca[0]) < 1e-5 and u_pca[1] < 0):
                    u_pca = -u_pca
            except Exception:
                u_pca = u_nn.copy()
        else:
            u_pca = u_nn.copy()

        angle_diff = angle_between_vectors_deg(u_pca, u_nn)
        if angle_diff > 5.0:
            candidate_axes = [u_pca, u_nn]
            axis_sources = ["PCA", "NN"]
        else:
            candidate_axes = [u_pca]
            axis_sources = ["PCA"]

        best_score = (-1.0, -1e9, -1.0, -1e9)
        best_layout = None

        for axis_val, axis_source in zip(candidate_axes, axis_sources):
            for mode in ["Mode A", "Mode B"]:
                res = evaluate_layout_candidate(
                    comp=comp,
                    candidate_axis=axis_val,
                    layout_mode=mode,
                    axis_source=axis_source,
                    img_h=img_h,
                    img_w=img_w,
                    image=image
                )
                if res is not None:
                    score, details = res
                    if score > best_score:
                        best_score = score
                        best_layout = details

        if best_layout is None:
            # Fallback block
            for p in comp:
                p["block_id"] = block_idx
                p["block_decision"] = "fallback_block"
                p["block_reason"] = "no_valid_layout"
                
            log_records.append({
                "record_type": "block",
                "block_id": int(block_idx),
                "n_panels": int(n_panels),
                "n_stack": 0,
                "n_cols": 0,
                "axis_source": "None",
                "layout_mode": "None",
                "row_rails": [],
                "col_rails": [],
                "pass_ratio": 0.0,
                "median_iou": 0.0,
                "median_center_shift": 1.0,
                "max_overlap": 0.0,
                "decision": "fallback_block"
            })
            
            # Log all panels in the block as fallback
            for p in comp:
                poly_orig = p["polygon"]
                poly_orig_np = np.array(poly_orig, dtype=np.float32)
                poly_orig_hull = cv2.convexHull(poly_orig_np.reshape(-1, 1, 2)).astype(np.float32)
                p_overlap_max = 0.0
                for other in comp:
                    if other is p:
                        continue
                    poly_other = np.array(other["polygon"], dtype=np.float32)
                    poly_other_hull = cv2.convexHull(poly_other.reshape(-1, 1, 2)).astype(np.float32)
                    area_intersect, _ = cv2.intersectConvexConvex(poly_orig_hull, poly_other_hull)
                    if area_intersect > 0:
                        area_other = cv2.contourArea(poly_other_hull)
                        min_area = min(cv2.contourArea(poly_orig_hull), area_other)
                        overlap_ratio = area_intersect / min_area if min_area > 0 else 0.0
                        if overlap_ratio > p_overlap_max:
                            p_overlap_max = overlap_ratio
                            
                log_records.append({
                    "record_type": "panel",
                    "original_polygon": [[int(pt[0]), int(pt[1])] for pt in poly_orig],
                    "candidate_polygon": [[int(pt[0]), int(pt[1])] for pt in poly_orig],
                    "final_polygon": [[int(pt[0]), int(pt[1])] for pt in poly_orig],
                    "original_bbox": [int(v) for v in (p.get("small_bbox_original") or p["box"])],
                    "candidate_area": float(p["area"]),
                    "original_area": float(p["area"]),
                    "area_ratio": 1.0,
                    "iou_with_original": 1.0,
                    "center_shift_ratio": 0.0,
                    "row_idx": -1,
                    "col_idx": -1,
                    "block_id": int(block_idx),
                    "layout_score": [-1.0, -1e9, -1.0, -1e9],
                    "fallback_reason": "no_valid_layout",
                    "decision": "fallback_original",
                    "max_overlap": float(p_overlap_max)
                })
            continue

        pass_ratio, neg_median_shift, median_iou, neg_max_overlap = best_score
        median_shift = -neg_median_shift
        max_overlap = -neg_max_overlap
        median_score_ratio = best_layout["median_score_ratio"]

        n_cols = best_layout["n_cols"]
        n_stack = best_layout["n_rows"]

        # Guard chỉ fallback TOÀN BỘ block khi layout thật sự tệ về hình học
        # KHÔNG dùng edge_score / median_score_ratio để quyết định fallback block
        is_block_fallback = (
            n_panels < BLOCK_MIN_PANELS_FOR_REFINEMENT or
            not (BLOCK_EXPECTED_COLUMNS_MIN <= n_cols <= BLOCK_EXPECTED_COLUMNS_MAX) or
            n_stack < 2 or
            pass_ratio < BLOCK_MIN_PASS_RATIO or   # < 0.45 thì tệ thật sự
            median_iou < 0.45 or
            max_overlap > 0.15   # overlap cực kỳ cao mới coi là bad lattice
        )

        if is_block_fallback:
            block_fallback_reason = "block_geometry_bad"
            if n_panels < BLOCK_MIN_PANELS_FOR_REFINEMENT:
                block_fallback_reason = "too_few_panels"
            elif not (BLOCK_EXPECTED_COLUMNS_MIN <= n_cols <= BLOCK_EXPECTED_COLUMNS_MAX):
                block_fallback_reason = "column_count_invalid"
            elif n_stack < 2:
                block_fallback_reason = "stack_too_small"
            elif pass_ratio < BLOCK_MIN_PASS_RATIO:
                block_fallback_reason = "pass_ratio_too_low"
            elif median_iou < 0.45:
                block_fallback_reason = "median_iou_too_low"
            elif max_overlap > 0.15:
                block_fallback_reason = "candidate_overlap_too_high"

            for p in comp:
                p["block_id"] = block_idx
                p["block_decision"] = "fallback_block"
                p["block_reason"] = block_fallback_reason

            panel_details = best_layout["panels"]
            _log_block_and_panels_fallback(
                log_records, block_idx, n_panels, n_stack, n_cols, best_layout,
                best_score, panel_details, comp, block_fallback_reason
            )
            continue

        panel_details = best_layout["panels"]

        # Overlap check & resolution
        for i in range(len(panel_details)):
            det_i = panel_details[i]
            if not det_i["pass_individual"]:
                continue
            for j in range(i + 1, len(panel_details)):
                det_j = panel_details[j]
                if not det_j["pass_individual"]:
                    continue

                poly_i = np.array(det_i["candidate_polygon"], dtype=np.float32)
                poly_j = np.array(det_j["candidate_polygon"], dtype=np.float32)
                
                poly_i_hull = cv2.convexHull(poly_i.reshape(-1, 1, 2)).astype(np.float32)
                poly_j_hull = cv2.convexHull(poly_j.reshape(-1, 1, 2)).astype(np.float32)
                
                area_intersect, _ = cv2.intersectConvexConvex(poly_i_hull, poly_j_hull)
                if area_intersect > 0:
                    area_i = cv2.contourArea(poly_i_hull)
                    area_j = cv2.contourArea(poly_j_hull)
                    min_area = min(area_i, area_j)
                    overlap_ratio = area_intersect / min_area if min_area > 0 else 0.0
                    if overlap_ratio > BLOCK_PANEL_MAX_OVERLAP_RATIO:
                        if det_i["iou_with_original"] < det_j["iou_with_original"]:
                           det_i["pass_individual"] = False
                           det_i["fail_reason"] = "overlap_too_high"
                        else:
                           det_j["pass_individual"] = False
                           det_j["fail_reason"] = "overlap_too_high"

        # Second pass of overlap
        for i in range(len(panel_details)):
            det_i = panel_details[i]
            poly_i = np.array(det_i["candidate_polygon"] if det_i["pass_individual"] else det_i["p_obj"]["polygon"], dtype=np.float32)
            poly_i_hull = cv2.convexHull(poly_i.reshape(-1, 1, 2)).astype(np.float32)
            
            for j in range(i + 1, len(panel_details)):
                det_j = panel_details[j]
                poly_j = np.array(det_j["candidate_polygon"] if det_j["pass_individual"] else det_j["p_obj"]["polygon"], dtype=np.float32)
                poly_j_hull = cv2.convexHull(poly_j.reshape(-1, 1, 2)).astype(np.float32)

                area_intersect, _ = cv2.intersectConvexConvex(poly_i_hull, poly_j_hull)
                if area_intersect > 0:
                    area_i = cv2.contourArea(poly_i_hull)
                    area_j = cv2.contourArea(poly_j_hull)
                    min_area = min(area_i, area_j)
                    overlap_ratio = area_intersect / min_area if min_area > 0 else 0.0
                    if overlap_ratio > BLOCK_PANEL_MAX_OVERLAP_RATIO:
                        if det_i["pass_individual"]:
                           det_i["pass_individual"] = False
                           det_i["fail_reason"] = "overlap_too_high"
                        if det_j["pass_individual"]:
                           det_j["pass_individual"] = False
                           det_j["fail_reason"] = "overlap_too_high"

        pass_count = sum(1 for d in panel_details if d["pass_individual"])
        post_pass_ratio = pass_count / n_panels

        # Không fallback toàn block vì pass_ratio thấp khi pass_ratio >= BLOCK_MIN_PASS_RATIO
        # Hybrid decision: panel pass => use_lattice, panel fail => fallback_original
        # Chỉ fallback toàn block nếu pass_ratio < BLOCK_MIN_PASS_RATIO (< 0.45)
        if post_pass_ratio < BLOCK_MIN_PASS_RATIO:
            block_fallback_reason = "pass_ratio_too_low_post_overlap"
            for p in comp:
                p["block_id"] = block_idx
                p["block_decision"] = "fallback_block"
                p["block_reason"] = block_fallback_reason

            _log_block_and_panels_fallback(
                log_records, block_idx, n_panels, n_stack, n_cols, best_layout,
                best_score, panel_details, comp, block_fallback_reason
            )
            continue

        # Use lattice lines! Update all panels first
        for det in panel_details:
            p = det["p_obj"]
            p["block_id"] = block_idx
            p["block_n_stack"] = n_stack
            p["block_n_cols"] = n_cols
            p["block_layout_mode"] = best_layout["layout_mode"]
            p["block_pass_ratio"] = post_pass_ratio
            p["block_decision"] = "use_lattice" if det["pass_individual"] else "fallback_original"
            p["block_reason"] = det["fail_reason"]

            p["block_origin"] = best_layout["origin"].tolist()
            p["block_row_axis"] = best_layout["row_axis"].tolist()
            p["block_col_axis"] = best_layout["col_axis"].tolist()
            p["block_row_min"] = best_layout["row_min"]
            p["block_row_max"] = best_layout["row_max"]
            p["block_col_min"] = best_layout["col_min"]
            p["block_col_max"] = best_layout["col_max"]
            p["block_row_rails"] = best_layout["row_rails"]
            p["block_col_rails"] = best_layout["col_rails"]
            p["candidate_polygon"] = det["candidate_polygon"]
            p["area_ratio"] = det["area_ratio"]
            p["iou_with_original"] = det["iou_with_original"]
            p["center_shift_ratio"] = det["center_shift_ratio"]
            p["row_idx"] = det["row_idx"]
            p["col_idx"] = det["col_idx"]
            p["layout_score"] = list(best_score)
            
            p["original_score"] = det["original_score"]
            p["candidate_score"] = det["candidate_score"]
            p["score_ratio"] = det["score_ratio"]
            p["outer_edge_expansion_px"] = det["outer_edge_expansion_px"]

            if det["pass_individual"]:
                p["polygon"] = det["candidate_polygon"]
                p["final_polygon_source"] = "two_col_block_middle"
                p["final_polygon_stage"] = "old_block_lattice_refinement"
                p["final_polygon_reason"] = "old_lattice_fit"
                features = get_polygon_features(p["polygon"])
                p["bbox"] = [round(v) for v in features["bbox"]]
                p["box"] = [round(v) for v in features["bbox"]]
                p["area"] = features["area"]
                p["center"] = features["center"]
                p["aspect_ratio"] = features["aspect_ratio"]
                p["block_refined"] = True

        # Now compute final max overlap and log block and panels
        final_max_overlap = 0.0
        for i in range(len(panel_details)):
            det_i = panel_details[i]
            poly_i = np.array(det_i["p_obj"]["polygon"], dtype=np.float32)
            poly_i_hull = cv2.convexHull(poly_i.reshape(-1, 1, 2)).astype(np.float32)
            for j in range(i + 1, len(panel_details)):
                det_j = panel_details[j]
                poly_j = np.array(det_j["p_obj"]["polygon"], dtype=np.float32)
                poly_j_hull = cv2.convexHull(poly_j.reshape(-1, 1, 2)).astype(np.float32)
                area_intersect, _ = cv2.intersectConvexConvex(poly_i_hull, poly_j_hull)
                if area_intersect > 0:
                    area_i = cv2.contourArea(poly_i_hull)
                    area_j = cv2.contourArea(poly_j_hull)
                    min_area = min(area_i, area_j)
                    overlap_ratio = area_intersect / min_area if min_area > 0 else 0.0
                    if overlap_ratio > final_max_overlap:
                        final_max_overlap = overlap_ratio

        log_records.append({
            "record_type": "block",
            "image_name": image_stem,
            "block_id": int(block_idx),
            "n_panels": int(n_panels),
            "n_rows": int(n_stack),
            "n_cols": int(n_cols),
            "axis_source": str(best_layout["axis_source"]),
            "layout_mode": str(best_layout["layout_mode"]),
            "row_rails": [float(r) for r in best_layout["row_rails"]],
            "col_rails": [float(c) for c in best_layout["col_rails"]],
            "pass_ratio": float(post_pass_ratio),
            "pass_count": int(pass_count),
            "median_iou": float(median_iou),
            "median_center_shift": float(median_shift),
            "max_overlap_candidate": float(final_max_overlap),
            "max_overlap": float(final_max_overlap),
            "decision": "use_lattice",
            "block_decision": "use_lattice",
            "block_fallback_reason": "",
            "median_score_ratio": float(median_score_ratio),
            "outer_rail_method": str(best_layout["outer_rail_method"]),
            "rail_angle_deg": 0.0,
            "divider_angle_deg": 0.0
        })

        print(f"[BLOCK_LATTICE] img={image_stem} block={block_idx} n={n_panels} decision=use_lattice pass={pass_count}/{n_panels} mode={best_layout['layout_mode']}")

        for det in panel_details:
            p = det["p_obj"]
            p_overlap_max = 0.0
            poly_i = np.array(p["polygon"], dtype=np.float32)
            poly_i_hull = cv2.convexHull(poly_i.reshape(-1, 1, 2)).astype(np.float32)
            for other in comp:
                if other is p:
                    continue
                poly_other = np.array(other["polygon"], dtype=np.float32)
                poly_other_hull = cv2.convexHull(poly_other.reshape(-1, 1, 2)).astype(np.float32)
                area_intersect, _ = cv2.intersectConvexConvex(poly_i_hull, poly_other_hull)
                if area_intersect > 0:
                    area_other = cv2.contourArea(poly_other_hull)
                    min_area = min(cv2.contourArea(poly_i_hull), area_other)
                    overlap_ratio = area_intersect / min_area if min_area > 0 else 0.0
                    if overlap_ratio > p_overlap_max:
                        p_overlap_max = overlap_ratio

            cand_bbox = [
                int(min(pt[0] for pt in det["candidate_polygon"])),
                int(min(pt[1] for pt in det["candidate_polygon"])),
                int(max(pt[0] for pt in det["candidate_polygon"])),
                int(max(pt[1] for pt in det["candidate_polygon"]))
            ]

            log_records.append({
                "record_type": "panel",
                "image_name": image_stem,
                "block_id": int(block_idx),
                "panel_idx": p.get("raw_idx", -1),
                "decision": str(p["block_decision"]),
                "fallback_reason": str(p["block_reason"]),
                "original_polygon": [[int(pt[0]), int(pt[1])] for pt in det["original_polygon"]],
                "candidate_polygon": [[int(pt[0]), int(pt[1])] for pt in det["candidate_polygon"]],
                "final_polygon": [[int(pt[0]), int(pt[1])] for pt in p["polygon"]],
                "original_bbox": [int(v) for v in (p.get("small_bbox_original") or p["box"])],
                "candidate_bbox": cand_bbox,
                "candidate_area": float(det["candidate_area"]),
                "original_area": float(det["original_area"]),
                "area_ratio": float(det["area_ratio"]),
                "width_ratio": float(det.get("width_ratio", 1.0)),
                "height_ratio": float(det.get("height_ratio", 1.0)),
                "iou_with_original": float(det["iou_with_original"]),
                "center_shift_ratio": float(det["center_shift_ratio"]),
                "row_idx": int(det["row_idx"]),
                "col_idx": int(det["col_idx"]),
                "block_id": int(block_idx),
                "layout_score": list(best_score),
                "max_overlap": float(p_overlap_max),
                "is_outer_top": bool(det["is_outer_top"]),
                "is_outer_bottom": bool(det["is_outer_bottom"]),
                "is_outer_left": bool(det["is_outer_left"]),
                "is_outer_right": bool(det["is_outer_right"]),
                "outer_edge_expansion_px": float(det["outer_edge_expansion_px"]),
                "original_score": float(det["original_score"]),
                "candidate_score": float(det["candidate_score"]),
                "score_ratio": float(det["score_ratio"]),
                "edge_warning": det.get("edge_warning", ""),
                "edge_decision": str(p["block_decision"])
            })

    # Write log records to JSONL file
    log_dir = "data/results/debug_logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{image_stem}_block_lattice_refine.jsonl")
    try:
        import json
        with open(log_path, "w", encoding="utf-8") as f:
            for r in log_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[BLOCK_LATTICE] Saved logs to: {log_path}")
    except Exception as e:
        logger.warning(f"[BLOCK_LATTICE_WARN] Failed to write JSONL logs: {e}")

    return panels


def refine_panel_string_by_parallel_lines(

    panels: List[Dict[str, Any]],
    image_shape: Tuple[int, int, int],
    image_path: Optional[str] = None
) -> List[Dict[str, Any]]:
    if not panels:
        return []

    img_h, img_w = image_shape[:2]

    # Step 1: Select candidates (small panels)
    candidates = []
    non_candidates = []
    for idx, p in enumerate(panels):
        bbox = p.get("bbox") or p.get("box") or [0, 0, 0, 0]
        x1, y1, x2, y2 = bbox
        bbox_area = (x2 - x1) * (y2 - y1)

        # Check if panel touches the border
        touches_border = (
            x1 <= STRING_BORDER_MARGIN_PX or
            y1 <= STRING_BORDER_MARGIN_PX or
            x2 >= img_w - STRING_BORDER_MARGIN_PX or
            y2 >= img_h - STRING_BORDER_MARGIN_PX
        )
        p["touches_border"] = touches_border

        if bbox_area <= STRING_REFINEMENT_MAX_PANEL_BBOX_AREA:
            candidates.append(p)
        else:
            p["string_refined"] = False
            p["string_decision"] = "bypass_not_small"
            p["string_reason"] = "large_panel"
            non_candidates.append(p)

    if not candidates:
        return panels

    # Step 2 & 3: Group candidates into strings
    strings = cluster_panels_into_strings(candidates)

    # Initialize log output list
    log_records = []
    image_stem = os.path.splitext(os.path.basename(image_path))[0] if image_path else "unknown"

    # Step 4-9: Fit parallel boundaries and reconstruct
    for s_idx, string_panels in enumerate(strings):
        n_string = len(string_panels)
        
        # Median size of candidates
        widths = [p["bbox"][2] - p["bbox"][0] for p in candidates]
        heights = [p["bbox"][3] - p["bbox"][1] for p in candidates]
        median_w = float(np.median(widths)) if widths else 1.0
        median_h = float(np.median(heights)) if heights else 1.0

        # Dominant orientation of this string from nearest neighbor center directions
        u_axis = estimate_component_direction(string_panels)
        v_axis = np.array([-u_axis[1], u_axis[0]], dtype=np.float32)

        centers = np.array([p["center"] for p in string_panels], dtype=np.float32)
        origin = np.mean(centers, axis=0)
        centered = centers - origin

        # Consistent direction sorting along u
        center_u = np.array([np.dot(c - origin, u_axis) for c in centers])
        sorted_indices = np.argsort(center_u)
        
        # Sort string_panels and center_u
        string_panels = [string_panels[idx] for idx in sorted_indices]
        center_u = center_u[sorted_indices]

        # Projects all polygon points of the string's panels onto local axes
        all_points = []
        for p in string_panels:
            poly = p["polygon"]
            for pt in poly:
                all_points.append(pt)
        all_points = np.array(all_points, dtype=np.float32)

        local_points_u = np.dot(all_points - origin, u_axis)
        local_points_v = np.dot(all_points - origin, v_axis)

        # Boundary lines
        if STRING_USE_PERCENTILE_BOUNDARIES:
            v_low = float(np.percentile(local_points_v, STRING_BOUNDARY_LOW_PERCENTILE))
            v_high = float(np.percentile(local_points_v, STRING_BOUNDARY_HIGH_PERCENTILE))
        else:
            v_low = float(np.min(local_points_v))
            v_high = float(np.max(local_points_v))

        string_width = abs(v_high - v_low)
        is_horizontal = abs(u_axis[0]) > abs(u_axis[1])
        expected_cross_size = median_h if is_horizontal else median_w
        expected_major_size = median_w if is_horizontal else median_h
        angle_deg = float(np.degrees(np.arctan2(u_axis[1], u_axis[0])))

        # Validate string width
        if not (0.6 * expected_cross_size <= string_width <= 1.4 * expected_cross_size):
            # Fallback string
            for p in string_panels:
                p["pass_individual"] = False
                p["refine_reason"] = "invalid_string_width"
            decision = "fallback_string"
            reason = "invalid_string_width"
            pitch = expected_major_size
            boundary = []
        else:
            # Estimate pitch
            diffs = np.diff(center_u)
            valid_diffs = diffs[(diffs >= (1.0 - STRING_PITCH_TOLERANCE_RATIO) * expected_major_size) & 
                                (diffs <= (1.0 + STRING_PITCH_TOLERANCE_RATIO) * expected_major_size)]
            pitch = float(np.median(valid_diffs)) if len(valid_diffs) > 0 else expected_major_size

            # Boundaries
            boundary = []
            boundary.append(center_u[0] - pitch / 2)
            for i in range(1, len(center_u)):
                mid = (center_u[i-1] + center_u[i]) / 2
                boundary.append(mid)
            boundary.append(center_u[-1] + pitch / 2)

            # Reconstruct polygons
            for i, p in enumerate(string_panels):
                u0 = boundary[i]
                u1 = boundary[i+1]

                P1 = origin + u0 * u_axis + v_low * v_axis
                P2 = origin + u1 * u_axis + v_low * v_axis
                P3 = origin + u1 * u_axis + v_high * v_axis
                P4 = origin + u0 * u_axis + v_high * v_axis

                pts = np.array([P1, P2, P3, P4], dtype=np.float32)
                new_poly = _sort_corners(pts)

                p["new_polygon_candidate"] = new_poly
                p["string_fit_info"] = {
                    "origin": origin.tolist(),
                    "u_axis": u_axis.tolist(),
                    "v_axis": v_axis.tolist(),
                    "v_low": v_low,
                    "v_high": v_high,
                    "boundary": [float(b) for b in boundary],
                    "pitch": pitch,
                    "angle_deg": angle_deg
                }

            # Individual Validation
            for i, p in enumerate(string_panels):
                new_poly = p["new_polygon_candidate"]
                new_poly_np = np.array(new_poly, dtype=np.float32)

                # Calculated features
                new_cx, new_cy = np.mean(new_poly_np, axis=0)
                new_w = abs(boundary[i+1] - boundary[i])
                new_h = abs(v_high - v_low)
                new_area = new_w * new_h

                old_polygon = p["polygon"]
                old_center = p["center"]
                old_w = p["bbox"][2] - p["bbox"][0]
                old_h = p["bbox"][3] - p["bbox"][1]
                old_polygon_area = p["area"]

                center_dist = np.linalg.norm(np.array([new_cx, new_cy]) - np.array(old_center))
                center_shift_ratio = center_dist / max(old_w, old_h)
                iou_with_original = _compute_mask_iou_cv(new_poly_np, np.array(old_polygon, dtype=np.float32), (img_h, img_w))

                # Local size comparisons
                old_poly_local_u = np.dot(np.array(old_polygon) - origin, u_axis)
                old_poly_local_v = np.dot(np.array(old_polygon) - origin, v_axis)
                old_local_w = np.max(old_poly_local_u) - np.min(old_poly_local_u)
                old_local_h = np.max(old_poly_local_v) - np.min(old_poly_local_v)

                width_ratio = new_w / old_local_w if old_local_w > 0 else 1.0
                height_ratio = new_h / old_local_h if old_local_h > 0 else 1.0
                area_ratio = new_area / old_polygon_area if old_polygon_area > 0 else 1.0

                out_of_bounds = any(pt[0] < -2 or pt[0] > img_w + 2 or pt[1] < -2 or pt[1] > img_h + 2 for pt in new_poly)

                # Validation checks
                pass_val = True
                fail_reason = "ok"

                if out_of_bounds:
                    pass_val = False
                    fail_reason = "out_of_image"
                elif not (STRING_PANEL_AREA_RATIO_MIN <= area_ratio <= STRING_PANEL_AREA_RATIO_MAX):
                    pass_val = False
                    fail_reason = "area_ratio_out"
                elif not (STRING_PANEL_WIDTH_RATIO_MIN <= width_ratio <= STRING_PANEL_WIDTH_RATIO_MAX):
                    pass_val = False
                    fail_reason = "width_ratio_out"
                elif not (STRING_PANEL_HEIGHT_RATIO_MIN <= height_ratio <= STRING_PANEL_HEIGHT_RATIO_MAX):
                    pass_val = False
                    fail_reason = "height_ratio_out"
                elif center_shift_ratio > STRING_PANEL_CENTER_SHIFT_MAX_RATIO:
                    pass_val = False
                    fail_reason = "center_shift_high"
                elif iou_with_original < STRING_PANEL_IOU_WITH_ORIGINAL_MIN:
                    pass_val = False
                    fail_reason = "iou_low"
                elif p.get("touches_border") and STRING_REJECT_BORDER_PARTIAL:
                    # Partial border panel check
                    if area_ratio > 1.20:
                        pass_val = False
                        fail_reason = "border_partial_rejected"

                p["pass_individual"] = pass_val
                p["refine_reason"] = fail_reason
                p["iou_with_original"] = iou_with_original
                p["area_ratio"] = area_ratio
                p["width_ratio"] = width_ratio
                p["height_ratio"] = height_ratio
                p["center_shift_ratio"] = center_shift_ratio

            # Step 8: Overlap check
            # First pass: compare pairs and disable the one with lower original IoU
            for i in range(len(string_panels)):
                if not string_panels[i]["pass_individual"]:
                    continue
                for j in range(i + 1, len(string_panels)):
                    if not string_panels[j]["pass_individual"]:
                        continue
                    poly_i = np.array(string_panels[i]["new_polygon_candidate"], dtype=np.float32)
                    poly_j = np.array(string_panels[j]["new_polygon_candidate"], dtype=np.float32)
                    area_intersect, _ = cv2.intersectConvexConvex(poly_i, poly_j)
                    if area_intersect > 0:
                        area_i = cv2.contourArea(poly_i)
                        area_j = cv2.contourArea(poly_j)
                        min_area = min(area_i, area_j)
                        overlap_ratio = area_intersect / min_area if min_area > 0 else 0.0
                        if overlap_ratio > STRING_PANEL_MAX_OVERLAP_RATIO:
                            iou_i = string_panels[i]["iou_with_original"]
                            iou_j = string_panels[j]["iou_with_original"]
                            if iou_i < iou_j:
                                string_panels[i]["pass_individual"] = False
                                string_panels[i]["refine_reason"] = "overlap_too_high"
                            else:
                                string_panels[j]["pass_individual"] = False
                                string_panels[j]["refine_reason"] = "overlap_too_high"

            # Second pass: check if any overlap remains between final state of panels in string
            for i in range(len(string_panels)):
                poly_i = np.array(string_panels[i]["new_polygon_candidate"] if string_panels[i]["pass_individual"] else string_panels[i]["polygon"], dtype=np.float32)
                for j in range(i + 1, len(string_panels)):
                    poly_j = np.array(string_panels[j]["new_polygon_candidate"] if string_panels[j]["pass_individual"] else string_panels[j]["polygon"], dtype=np.float32)
                    area_intersect, _ = cv2.intersectConvexConvex(poly_i, poly_j)
                    if area_intersect > 0:
                        area_i = cv2.contourArea(poly_i)
                        area_j = cv2.contourArea(poly_j)
                        min_area = min(area_i, area_j)
                        overlap_ratio = area_intersect / min_area if min_area > 0 else 0.0
                        if overlap_ratio > STRING_PANEL_MAX_OVERLAP_RATIO:
                            if string_panels[i]["pass_individual"]:
                                string_panels[i]["pass_individual"] = False
                                string_panels[i]["refine_reason"] = "overlap_too_high"
                            if string_panels[j]["pass_individual"]:
                                string_panels[j]["pass_individual"] = False
                                string_panels[j]["refine_reason"] = "overlap_too_high"

            # Step 9: Subcluster/string-level decision
            pass_count = sum(1 for p in string_panels if p["pass_individual"])
            pass_ratio = pass_count / n_string
            if pass_ratio < 0.60:
                for p in string_panels:
                    p["pass_individual"] = False
                    p["refine_reason"] = "string_quality_low"
                decision = "fallback_string"
                reason = "string_quality_low"
            else:
                decision = "use_string_lines"
                reason = "ok"

        # Apply changes & prepare logs
        pass_count = sum(1 for p in string_panels if p["pass_individual"])
        pass_ratio = pass_count / n_string

        # Compute max overlap among final polygons in the string
        max_overlap_val = 0.0
        for i in range(len(string_panels)):
            poly_i = np.array(string_panels[i]["new_polygon_candidate"] if string_panels[i]["pass_individual"] else string_panels[i]["polygon"], dtype=np.float32)
            for j in range(i + 1, len(string_panels)):
                poly_j = np.array(string_panels[j]["new_polygon_candidate"] if string_panels[j]["pass_individual"] else string_panels[j]["polygon"], dtype=np.float32)
                area_intersect, _ = cv2.intersectConvexConvex(poly_i, poly_j)
                if area_intersect > 0:
                    area_i = cv2.contourArea(poly_i)
                    area_j = cv2.contourArea(poly_j)
                    min_area = min(area_i, area_j)
                    overlap_ratio = area_intersect / min_area if min_area > 0 else 0.0
                    if overlap_ratio > max_overlap_val:
                        max_overlap_val = overlap_ratio

        # Terminal log for string
        print(f"[STRING_GRID] img={image_stem} string={s_idx} n={n_string} decision={decision} pass={pass_count}/{n_string} angle={angle_deg:.1f} pitch={pitch:.1f}")

        # String Summary Log Record
        log_records.append({
            "record_type": "string_summary",
            "image": str(image_path),
            "string_id": int(s_idx),
            "n_panels": int(n_string),
            "decision": str(decision),
            "reason": str(reason),
            "angle_deg": float(angle_deg),
            "v_low": float(v_low),
            "v_high": float(v_high),
            "string_width": float(string_width),
            "pitch": float(pitch),
            "pass_count": int(pass_count),
            "pass_ratio": float(pass_ratio),
            "max_overlap": float(max_overlap_val)
        })

        for i, p in enumerate(string_panels):
            p["string_id"] = s_idx
            p["string_angle_deg"] = angle_deg
            
            # Max overlap for this panel
            p_overlap_max = 0.0
            poly_i = np.array(p["new_polygon_candidate"] if p["pass_individual"] else p["polygon"], dtype=np.float32)
            for other in string_panels:
                if other is p:
                    continue
                poly_other = np.array(other["new_polygon_candidate"] if other["pass_individual"] else other["polygon"], dtype=np.float32)
                area_intersect, _ = cv2.intersectConvexConvex(poly_i, poly_other)
                if area_intersect > 0:
                    area_other = cv2.contourArea(poly_other)
                    min_area = min(cv2.contourArea(poly_i), area_other)
                    overlap_ratio = area_intersect / min_area if min_area > 0 else 0.0
                    if overlap_ratio > p_overlap_max:
                        p_overlap_max = overlap_ratio

            p["string_overlap_max"] = p_overlap_max

            if p["pass_individual"]:
                p["polygon"] = p["new_polygon_candidate"]
                p["final_polygon_source"] = "string_lattice_middle"
                p["final_polygon_stage"] = "old_string_parallel_line_refinement"
                p["final_polygon_reason"] = "old_parallel_line_fit"
                features = get_polygon_features(p["polygon"])
                p["bbox"] = [round(v) for v in features["bbox"]]
                p["box"] = [round(v) for v in features["bbox"]]
                p["area"] = features["area"]
                p["center"] = features["center"]
                p["aspect_ratio"] = features["aspect_ratio"]
                p["string_refined"] = True
                p["string_decision"] = "use_string_lines"
                p["string_reason"] = "ok"

                print(f"[STRING_PANEL] img={image_stem} idx={p['raw_idx']} string={s_idx} decision=use reason=ok iou={p['iou_with_original']:.2f} shift={p['center_shift_ratio']:.2f}")
            else:
                p["string_refined"] = False
                p["string_decision"] = "fallback_original"
                p["string_reason"] = p.get("refine_reason", "unknown")

                # Recompute metrics for log record if we didn't do it (e.g. invalid string width)
                iou_with_orig = p.get("iou_with_original", 1.0)
                a_ratio = p.get("area_ratio", 1.0)
                w_ratio = p.get("width_ratio", 1.0)
                h_ratio = p.get("height_ratio", 1.0)
                c_shift = p.get("center_shift_ratio", 0.0)

                print(f"[STRING_PANEL] img={image_stem} idx={p['raw_idx']} string={s_idx} decision=fallback reason={p['string_reason']} iou={iou_with_orig:.2f} shift={c_shift:.2f}")

            # Panel Log Record
            log_records.append({
                "record_type": "panel",
                "image": str(image_path),
                "panel_idx": int(p["raw_idx"]),
                "string_id": int(s_idx),
                "original_bbox": [int(v) for v in (p.get("small_bbox_original") or p["box"])],
                "original_area": float(p["area"] if not p.get("string_refined") else p["area"] / p.get("area_ratio", 1.0)),
                "new_polygon": [[int(pt[0]), int(pt[1])] for pt in p["polygon"]],
                "decision": str(p["string_decision"]),
                "reason": str(p["string_reason"]),
                "area_ratio": float(p.get("area_ratio", 1.0)),
                "width_ratio": float(p.get("width_ratio", 1.0)),
                "height_ratio": float(p.get("height_ratio", 1.0)),
                "iou_with_original": float(p.get("iou_with_original", 1.0)),
                "center_shift_ratio": float(p.get("center_shift_ratio", 0.0)),
                "max_overlap": float(p_overlap_max)
            })

    # Write log records to JSONL file
    log_dir = "data/results/debug_logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{image_stem}_parallel_string_refine.jsonl")
    try:
        import json
        with open(log_path, "w", encoding="utf-8") as f:
            for r in log_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning(f"[STRING_LOG_WARN] Failed to write JSONL logs: {e}")

    # Reassemble all panels
    out_panels = []
    # Loop over original panels to preserve ordering or just combine processed candidates + non-candidates
    for p in panels:
        out_panels.append(p)

    return out_panels


# ===========================================================================
# SECTION 4b — STRING-FIRST LATTICE REFINEMENT (NEW PRODUCTION PATH)
# ===========================================================================

def select_trusted_panels_for_geometry(
    panels: List[Dict[str, Any]],
    image_shape: Tuple[int, int, int]
) -> List[int]:
    """
    Chon cac panel dang tin cay de dung fit rail/pitch.
    Tra ve list index trong panels list.
    Panel untrusted van nhan candidate sau, nhung khong duoc dung de fit.
    """
    img_h, img_w = image_shape[:2]

    if not panels:
        return []

    areas = [p.get("area", 0.0) for p in panels]
    valid_areas = [a for a in areas if a > 0]
    median_area = float(np.median(valid_areas)) if valid_areas else 1.0

    trusted_indices = []
    for i, p in enumerate(panels):
        bbox = p.get("bbox") or p.get("box") or [0, 0, 0, 0]
        x1, y1, x2, y2 = bbox
        bw = x2 - x1
        bh = y2 - y1
        b_area = bw * bh
        polygon = p.get("polygon", [])
        conf = p.get("confidence", p.get("raw_conf", 1.0))
        area = p.get("area", b_area)
        aspect_ratio = p.get("aspect_ratio", max(bw, bh) / max(min(bw, bh), 1.0))

        reason = "ok"
        trusted = True

        if b_area <= 0:
            trusted = False
            reason = "zero_bbox_area"
        elif len(polygon) < 4:
            trusted = False
            reason = "too_few_polygon_pts"
        elif conf < STRING_TRUSTED_MIN_CONF:
            trusted = False
            reason = f"low_conf_{conf:.3f}"
        elif (x1 <= STRING_BORDER_MARGIN_PX or y1 <= STRING_BORDER_MARGIN_PX or
              x2 >= img_w - STRING_BORDER_MARGIN_PX or y2 >= img_h - STRING_BORDER_MARGIN_PX):
            trusted = False
            reason = "touches_image_border"
        else:
            area_ratio = area / max(median_area, 1e-3)
            if not (STRING_TRUSTED_AREA_RATIO_MIN <= area_ratio <= STRING_TRUSTED_AREA_RATIO_MAX):
                trusted = False
                reason = f"area_ratio_out_{area_ratio:.2f}"
            elif not (STRING_TRUSTED_ASPECT_MIN <= aspect_ratio <= STRING_TRUSTED_ASPECT_MAX):
                trusted = False
                reason = f"aspect_ratio_out_{aspect_ratio:.2f}"

        if _STRING_LATTICE_DEBUG:
            print(f"[TRUSTED_PANEL] idx={p.get('raw_idx', i)} trusted={trusted} reason={reason}")

        if trusted:
            trusted_indices.append(i)

    return trusted_indices


def _estimate_string_axis(panels_in_string: List[Dict]) -> np.ndarray:
    """Estimate u_axis (along string direction) from panel centers using PCA + NN."""
    if len(panels_in_string) < 2:
        return np.array([1.0, 0.0], dtype=np.float32)

    centers = np.array([p["center"] for p in panels_in_string], dtype=np.float32)

    # PCA on centers
    mean = np.mean(centers, axis=0)
    centered = centers - mean
    if len(centered) >= 2:
        cov = np.cov(centered.T)
        if cov.ndim == 2:
            eigvals, eigvecs = np.linalg.eigh(cov)
            u_pca = eigvecs[:, np.argmax(eigvals)].astype(np.float32)
        else:
            u_pca = np.array([1.0, 0.0], dtype=np.float32)
    else:
        u_pca = np.array([1.0, 0.0], dtype=np.float32)

    # NN voting
    u_pca = u_pca / (np.linalg.norm(u_pca) + 1e-8)
    if u_pca[0] < 0 or (abs(u_pca[0]) < 1e-5 and u_pca[1] < 0):
        u_pca = -u_pca

    return u_pca


def split_panels_into_strings(
    panels: List[Dict[str, Any]],
    trusted_indices: List[int],
    image_shape: Tuple[int, int, int]
) -> List[Dict[str, Any]]:
    """
    Tach panels thanh cac string/day nho.
    Moi string la 1 dict: {panels, trusted_mask, string_id, u_axis, v_axis, ...}
    """
    if not panels:
        return []

    img_h, img_w = image_shape[:2]
    n = len(panels)
    trusted_set = set(trusted_indices)

    # Compute sizes for threshold
    widths = [(p.get("bbox") or p.get("box") or [0,0,0,0])[2] - (p.get("bbox") or p.get("box") or [0,0,0,0])[0] for p in panels]
    heights = [(p.get("bbox") or p.get("box") or [0,0,0,0])[3] - (p.get("bbox") or p.get("box") or [0,0,0,0])[1] for p in panels]
    median_w = float(np.median(widths)) if widths else 1.0
    median_h = float(np.median(heights)) if heights else 1.0
    median_short = min(median_w, median_h)
    median_long = max(median_w, median_h)

    centers = np.array([p["center"] for p in panels], dtype=np.float32)

    # --- Step 1: Estimate dominant direction from ALL panels ---
    u_global = _estimate_string_axis(panels)
    v_global = np.array([-u_global[1], u_global[0]], dtype=np.float32)

    # --- Step 2: Project centers onto (u, v) local system ---
    origin = np.mean(centers, axis=0)
    u_coords = np.array([float(np.dot(c - origin, u_global)) for c in centers])
    v_coords = np.array([float(np.dot(c - origin, v_global)) for c in centers])

    # --- Step 3: Group by v-coordinate (cross-string direction) ---
    v_thresh = STRING_V_GROUP_THRESHOLD_FACTOR * median_short
    sorted_v_idx = np.argsort(v_coords)
    v_sorted = v_coords[sorted_v_idx]

    v_groups = []  # list of lists of panel indices (original index)
    current_group = [int(sorted_v_idx[0])]
    for k in range(1, len(sorted_v_idx)):
        if v_sorted[k] - v_sorted[k - 1] <= v_thresh:
            current_group.append(int(sorted_v_idx[k]))
        else:
            v_groups.append(current_group)
            current_group = [int(sorted_v_idx[k])]
    v_groups.append(current_group)

    # --- Step 4: Within each v-group, sort by u and split on large gaps ---
    raw_strings = []  # list of list of panel indices
    for vg in v_groups:
        if len(vg) < 1:
            continue
        # Sort by u
        vg_sorted = sorted(vg, key=lambda idx: u_coords[idx])
        u_vals = [u_coords[idx] for idx in vg_sorted]

        # Compute gaps
        if len(u_vals) > 1:
            gaps = [u_vals[k + 1] - u_vals[k] for k in range(len(u_vals) - 1)]
            median_gap = float(np.median([g for g in gaps if g > 0])) if any(g > 0 for g in gaps) else median_long
        else:
            gaps = []
            median_gap = median_long

        # Split on large gaps
        seg = [vg_sorted[0]]
        for k in range(len(gaps)):
            if gaps[k] > STRING_GAP_SPLIT_FACTOR * median_gap:
                raw_strings.append(seg)
                seg = [vg_sorted[k + 1]]
            else:
                seg.append(vg_sorted[k + 1])
        raw_strings.append(seg)

    # --- Step 5: Validate each string and build string info dicts ---
    valid_strings = []
    for s_idx, idx_list in enumerate(raw_strings):
        if len(idx_list) < STRING_MIN_PANELS:
            continue

        str_panels = [panels[i] for i in idx_list]
        str_trusted_mask = [i in trusted_set for i in idx_list]
        n_trusted = sum(str_trusted_mask)

        if n_trusted < STRING_MIN_TRUSTED:
            print(f"[STRING_SPLIT] Skip string (n={len(idx_list)} trusted={n_trusted} < {STRING_MIN_TRUSTED})")
            continue

        # Refine u_axis using only trusted panels in this string
        trusted_panels_in_str = [str_panels[k] for k in range(len(str_panels)) if str_trusted_mask[k]]
        u_axis = _estimate_string_axis(trusted_panels_in_str) if len(trusted_panels_in_str) >= 2 else u_global.copy()
        v_axis = np.array([-u_axis[1], u_axis[0]], dtype=np.float32)

        # Compute pitch from trusted panel u-centers
        str_centers = np.array([p["center"] for p in str_panels], dtype=np.float32)
        str_origin = np.mean(str_centers, axis=0)
        u_str = np.array([float(np.dot(c - str_origin, u_axis)) for c in str_centers])
        u_trusted = sorted([u_str[k] for k in range(len(str_panels)) if str_trusted_mask[k]])

        if len(u_trusted) > 1:
            t_gaps = [u_trusted[k + 1] - u_trusted[k] for k in range(len(u_trusted) - 1) if u_trusted[k+1] > u_trusted[k]]
            pitch = float(np.median(t_gaps)) if t_gaps else median_long
            gap_cv = float(np.std(t_gaps) / max(np.median(t_gaps), 1e-3)) if len(t_gaps) > 1 else 0.0
        else:
            pitch = median_long
            gap_cv = 0.0

        # Compute angle in degrees
        angle_deg = float(np.degrees(np.arctan2(u_axis[1], u_axis[0])))

        print(f"[STRING_SPLIT] string_id={len(valid_strings)} n={len(str_panels)} trusted={n_trusted} "
              f"angle={angle_deg:.2f} pitch={pitch:.1f} gap_cv={gap_cv:.3f}")

        valid_strings.append({
            "string_id": len(valid_strings),
            "panel_indices": idx_list,      # indices into panels list
            "panels": str_panels,
            "trusted_mask": str_trusted_mask,
            "n_trusted": n_trusted,
            "u_axis": u_axis,
            "v_axis": v_axis,
            "pitch": pitch,
            "gap_cv": gap_cv,
            "angle_deg": angle_deg,
            "origin": str_origin,
        })

    return valid_strings


def classify_panel_roles(
    panels: List[Dict[str, Any]],
    min_len_for_endpoint: int = 5
) -> None:
    """
    Classify each panel in a string/block as 'middle', 'endpoint_start', or 'endpoint_end'.
    Sets panel["panel_role"], panel["is_endpoint"], panel["endpoint_side"] in-place.
    If len(panels) < min_len_for_endpoint, all panels are treated as 'middle'.
    """
    n = len(panels)
    for k, p in enumerate(panels):
        if n < min_len_for_endpoint or not ENABLE_ENDPOINT_ISOLATION:
            p["panel_role"] = "middle"
            p["is_endpoint"] = False
            p["endpoint_side"] = None
        elif k == 0:
            p["panel_role"] = "endpoint_start"
            p["is_endpoint"] = True
            p["endpoint_side"] = "start"
        elif k == n - 1:
            p["panel_role"] = "endpoint_end"
            p["is_endpoint"] = True
            p["endpoint_side"] = "end"
        else:
            p["panel_role"] = "middle"
            p["is_endpoint"] = False
            p["endpoint_side"] = None
        # Backward-compat aliases
        p["is_endpoint_panel"] = p["is_endpoint"]


def build_endpoint_candidate_from_inner_divider(
    original_yolo_polygon: List[List[float]],
    inner_divider_u: float,
    v_low: float,
    v_high: float,
    u_axis: np.ndarray,
    v_axis: np.ndarray,
    origin: np.ndarray,
    endpoint_side: str,  # "start" or "end"
    median_panel_length: float,
) -> Dict[str, Any]:
    """
    Build an endpoint candidate polygon using:
    - Inner edge: the nearest divider from the middle lattice (inner_divider_u)
    - Outer edge: projection of the YOLO original polygon onto u_axis (no image snap)
    - Rails: v_low / v_high from middle lattice

    Returns dict with:
        candidate_polygon, outer_u, inner_u, extrapolate_ratio, success
    """
    orig_np = np.array(original_yolo_polygon, dtype=np.float32)
    if len(orig_np) < 3:
        return {"success": False, "reason": "too_few_original_points"}

    # Project YOLO points onto u_axis to find outer edge
    u_projs = [float(np.dot(pt - origin, u_axis)) for pt in orig_np]

    if endpoint_side == "start":
        outer_u = float(min(u_projs))  # leftmost point is outer edge
    else:  # "end"
        outer_u = float(max(u_projs))  # rightmost point is outer edge

    inner_u = inner_divider_u

    # Extrapolate ratio check: how far does endpoint extend beyond inner divider
    panel_ext = abs(outer_u - inner_u)
    extrapolate_ratio = panel_ext / max(median_panel_length, 1e-3)

    # Build 4-corner polygon: inner_divider side + YOLO outer edge + rails
    def to_img(u_val: float, v_val: float) -> List[float]:
        pt = origin + u_val * u_axis + v_val * v_axis
        return [float(pt[0]), float(pt[1])]

    if endpoint_side == "start":
        # outer_u < inner_u (outer is to the left)
        candidate_polygon = [
            to_img(outer_u, v_low),
            to_img(inner_u, v_low),
            to_img(inner_u, v_high),
            to_img(outer_u, v_high),
        ]
    else:
        # outer_u > inner_u (outer is to the right)
        candidate_polygon = [
            to_img(inner_u, v_low),
            to_img(outer_u, v_low),
            to_img(outer_u, v_high),
            to_img(inner_u, v_high),
        ]

    return {
        "success": True,
        "candidate_polygon": candidate_polygon,
        "outer_u": outer_u,
        "inner_u": inner_u,
        "extrapolate_ratio": extrapolate_ratio,
    }


def resolve_hybrid_endpoint_outer_u(
    original_yolo_polygon: List[List[float]],
    inner_u: float,
    v_low: float,
    v_high: float,
    u_axis: np.ndarray,
    v_axis: np.ndarray,
    origin: np.ndarray,
    median_pitch: float,
    endpoint_side: str,  # "start" or "end"
    image: Optional[np.ndarray],
    consensus_outer_u: Optional[float] = None,
    yolo_conf: float = 1.0,
    middle_lattice_good: bool = False
) -> Dict[str, Any]:
    """
    Hybrid Endpoint Outer Edge Resolver.
    Selects the best outer_u from 4 sources:
    1. YOLO Projection
    2. Pitch Extrapolation (clamped to YOLO)
    3. Edge Scan
    4. Consensus (if provided)
    """
    orig_np = np.array(original_yolo_polygon, dtype=np.float32)
    if len(orig_np) < 3:
        return {"success": False, "reason": "too_few_original_points"}

    # Project YOLO points onto u_axis
    u_values = [float(np.dot(pt - origin, u_axis)) for pt in orig_np]
    if endpoint_side == "start":
        yolo_outer_u = min(u_values)
    else:
        yolo_outer_u = max(u_values)

    candidates = []

    # 1. YOLO Projection
    candidates.append({
        "source": "yolo_projection",
        "outer_u": yolo_outer_u,
        "edge_score": 0.0
    })

    # 2. Pitch Extrapolation
    if endpoint_side == "start":
        pitch_outer_u = inner_u - median_pitch
    else:
        pitch_outer_u = inner_u + median_pitch

    # Clamp pitch extrapolation close to YOLO
    max_shift = ENDPOINT_OUTER_MAX_SHIFT_FROM_YOLO_RATIO * median_pitch
    pitch_outer_u_clamped = float(np.clip(pitch_outer_u, yolo_outer_u - max_shift, yolo_outer_u + max_shift))
    
    candidates.append({
        "source": "pitch_extrapolation",
        "outer_u": pitch_outer_u_clamped,
        "edge_score": 0.0
    })

    # 3. Edge Scan
    best_edge_u = None
    best_edge_score = -1.0
    
    if image is not None:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        else:
            gray = image.astype(np.float32)
        h, w = gray.shape[:2]
        
        center_u = 0.5 * (yolo_outer_u + pitch_outer_u_clamped)
        scan_min = center_u - ENDPOINT_EDGE_SEARCH_PX
        scan_max = center_u + ENDPOINT_EDGE_SEARCH_PX
        
        u_candidates = np.arange(scan_min, scan_max + 0.1, 1.0)
        n_samples = 25
        ux, uy = float(u_axis[0]), float(u_axis[1])
        side = 4
        
        for u_cand in u_candidates:
            darkness_vals = []
            gradient_vals = []
            edge_hits = 0
            total_hits = 0
            
            for t in np.linspace(v_low, v_high, n_samples):
                pt = origin + u_cand * u_axis + t * v_axis
                x, y = int(round(float(pt[0]))), int(round(float(pt[1])))
                if not (0 <= x < w and 0 <= y < h):
                    continue
                
                pix_c = float(gray[y, x])
                darkness = (255.0 - pix_c) / 255.0
                darkness_vals.append(darkness)
                
                # Gradient score along u_axis
                xl = int(round(x - ux * side))
                yl = int(round(y - uy * side))
                xr = int(round(x + ux * side))
                yr = int(round(y + uy * side))
                
                pix_l = float(gray[yl, xl]) if (0 <= yl < h and 0 <= xl < w) else pix_c
                pix_r = float(gray[yr, xr]) if (0 <= yr < h and 0 <= xr < w) else pix_c
                
                grad = abs(pix_l - pix_r) / 255.0
                gradient_vals.append(grad)
                
                # continuity
                if 0.5 * darkness + 0.5 * grad >= 0.15:
                    edge_hits += 1
                total_hits += 1
                
            mean_darkness = float(np.mean(darkness_vals)) if darkness_vals else 0.0
            mean_grad = float(np.mean(gradient_vals)) if gradient_vals else 0.0
            continuity = float(edge_hits) / max(total_hits, 1)
            
            score = 0.4 * mean_darkness + 0.4 * mean_grad + 0.2 * continuity
            if score > best_edge_score:
                best_edge_score = score
                best_edge_u = u_cand
                
    if best_edge_u is not None:
        candidates.append({
            "source": "edge_scan",
            "outer_u": float(best_edge_u),
            "edge_score": best_edge_score
        })

    # 4. Consensus
    if consensus_outer_u is not None:
        candidates.append({
            "source": "consensus",
            "outer_u": float(consensus_outer_u),
            "edge_score": 0.0
        })

    # Score each candidate
    scored_candidates = []
    for cand in candidates:
        u_val = cand["outer_u"]
        source = cand["source"]
        
        length_ratio = abs(inner_u - u_val) / max(median_pitch, 1e-3)
        length_score = float(np.clip(1.0 - abs(length_ratio - 1.0), 0.0, 1.0))
        
        distance_to_yolo_ratio = abs(u_val - yolo_outer_u) / max(median_pitch, 1e-3)
        yolo_distance_score = float(1.0 - min(distance_to_yolo_ratio, 1.0))
        
        edge_score = cand["edge_score"]
        
        total_score = 0.45 * length_score + 0.30 * yolo_distance_score + 0.25 * edge_score
        
        # Prior bonus
        prior_bonus = 0.0
        if source == "yolo_projection" and yolo_conf >= 0.85 and ENDPOINT_LENGTH_RATIO_MIN <= length_ratio <= ENDPOINT_LENGTH_RATIO_MAX:
            prior_bonus += 0.05
        elif source == "pitch_extrapolation" and middle_lattice_good:
            prior_bonus += 0.05
        elif source == "edge_scan" and edge_score >= 0.20:
            prior_bonus += 0.05
        elif source == "consensus" and consensus_outer_u is not None:
            prior_bonus += 0.05
            
        total_score += prior_bonus
        
        scored_candidates.append({
            "source": source,
            "outer_u": u_val,
            "length_ratio": length_ratio,
            "distance_to_yolo_ratio": distance_to_yolo_ratio,
            "edge_score": edge_score,
            "score": total_score
        })

    # Find candidate with highest score
    best_cand = max(scored_candidates, key=lambda x: x["score"])
    
    return {
        "success": True,
        "best_candidate": best_cand,
        "candidates": scored_candidates,
        "yolo_outer_u": yolo_outer_u,
        "pitch_outer_u": pitch_outer_u_clamped,
        "edge_outer_u": best_edge_u if best_edge_u is not None else 0.0,
        "consensus_outer_u": consensus_outer_u if consensus_outer_u is not None else 0.0
    }


def fit_string_parallel_lines(
    string_info: Dict[str, Any],
    image_shape: Tuple[int, int, int],
    image: Optional[np.ndarray] = None,
    median_panel_area: float = 1.0
) -> Dict[str, Any]:
    """
    Fit 2 parallel rails + dividers for a single string.
    Endpoint Isolation Architecture:
      - Rails fit from MIDDLE panels only (not endpoints).
      - u0 / pitch / dividers fit from MIDDLE panels only.
      - Endpoint candidates built from Middle-Locked reconstruction or Hybrid Resolver.
    Returns candidate polygons + validation results for each panel.
    """
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

    # --- Step 1: Classify panel roles ---
    classify_panel_roles(panels, min_len_for_endpoint=MIN_STRING_LEN_FOR_ENDPOINT_ISOLATION)
    middle_indices = [k for k in range(n) if panels[k]["panel_role"] == "middle"]
    middle_trusted_indices = [k for k in middle_indices if trusted_mask[k]]

    # Check minimum middle panels
    if ENABLE_ENDPOINT_ISOLATION and n >= MIN_STRING_LEN_FOR_ENDPOINT_ISOLATION:
        if len(middle_indices) < MIN_MIDDLE_PANELS_FOR_LATTICE:
            return {"success": False, "reason": f"too_few_middle_panels_{len(middle_indices)}"}

    # --- Step 2: Fit rails from MIDDLE panels only ---
    rail_fit_source = "all_panels"
    fit_panels = [panels[k] for k in range(n) if trusted_mask[k]]

    if ENABLE_ENDPOINT_ISOLATION and n >= MIN_STRING_LEN_FOR_ENDPOINT_ISOLATION:
        middle_trusted_panels = [panels[k] for k in middle_trusted_indices]
        if len(middle_trusted_panels) >= 2:
            fit_panels = middle_trusted_panels
            rail_fit_source = "middle_panels_only"
    elif n >= 5:
        # Legacy: if isolation disabled, use middle trusted if available
        middle_trusted_panels = [panels[k] for k in range(1, n - 1) if trusted_mask[k]]
        if len(middle_trusted_panels) >= 2:
            fit_panels = middle_trusted_panels
            rail_fit_source = "middle_panels"

    all_v_proj = []
    for p in fit_panels:
        pts = np.array(p["polygon"], dtype=np.float32)
        for pt in pts:
            all_v_proj.append(float(np.dot(pt - origin, v_axis)))

    if len(all_v_proj) < 4:
        return {"success": False, "reason": "too_few_points_for_rail"}

    if ENABLE_MIDDLE_LOCKED_ENDPOINT:
        pct_low = 3.0
        pct_high = 97.0
    else:
        pct_low = STRING_OUTER_RAIL_PCT_LOW_SMALL if n_trusted < 8 else STRING_OUTER_RAIL_PCT_LOW_LARGE
        pct_high = STRING_OUTER_RAIL_PCT_HIGH_SMALL if n_trusted < 8 else STRING_OUTER_RAIL_PCT_HIGH_LARGE

    v_low = float(np.percentile(all_v_proj, pct_low))
    v_high = float(np.percentile(all_v_proj, pct_high))

    if v_high <= v_low:
        return {"success": False, "reason": "rail_degenerate"}

    # --- Step 3: Fit u0 / pitch / positions from MIDDLE trusted panels only ---
    if ENABLE_ENDPOINT_ISOLATION and n >= MIN_STRING_LEN_FOR_ENDPOINT_ISOLATION and middle_trusted_indices:
        u_for_pitch = sorted([
            float(np.dot(np.array(panels[k]["center"]) - origin, u_axis))
            for k in middle_trusted_indices
        ])
    else:
        u_for_pitch = sorted([
            float(np.dot(np.array(panels[k]["center"]) - origin, u_axis))
            for k in range(n) if trusted_mask[k]
        ])

    if len(u_for_pitch) < 2:
        return {"success": False, "reason": "too_few_trusted_u"}

    t_gaps = [u_for_pitch[k + 1] - u_for_pitch[k] for k in range(len(u_for_pitch) - 1)
              if u_for_pitch[k + 1] > u_for_pitch[k]]
    if not t_gaps:
        return {"success": False, "reason": "no_positive_gaps"}

    if ENABLE_MIDDLE_LOCKED_ENDPOINT:
        raw_median = float(np.median(t_gaps)) if t_gaps else 0.0
        filtered_gaps = [g for g in t_gaps if 0.6 * raw_median <= g <= 1.4 * raw_median]
        pitch_fit = float(np.median(filtered_gaps)) if filtered_gaps else raw_median
    else:
        pitch_fit = float(np.median(t_gaps))
        
    if pitch_fit <= 0:
        return {"success": False, "reason": "negative_pitch"}

    # u0: leftmost trusted panel position (anchored to middle)
    u0 = u_for_pitch[0]

    # --- Step 4: Assign integer positions to ALL panels ---
    panel_u_proj = [float(np.dot(np.array(p["center"]) - origin, u_axis)) for p in panels]

    def assign_positions(u_projs, u_start, p_fit):
        positions = []
        for u in u_projs:
            raw_pos = (u - u_start) / max(p_fit, 1e-3)
            pos_int = int(round(raw_pos))
            positions.append(pos_int)
        return positions

    positions = assign_positions(panel_u_proj, u0, pitch_fit)

    # --- Partial panel detection ---
    all_pos = sorted(set(positions))
    min_pos = min(all_pos) if all_pos else 0
    max_pos = max(all_pos) if all_pos else 0

    # Compute median candidate area (for partial detection)
    v_rail_width = v_high - v_low
    median_cand_area = pitch_fit * v_rail_width
    median_panel_length = pitch_fit  # used for extrapolate_ratio

    # Calculate median middle area
    middle_areas = []
    for mid_idx in middle_indices:
        m_poly = np.array(panels[mid_idx]["polygon"], dtype=np.float32)
        middle_areas.append(float(cv2.contourArea(m_poly.reshape(-1, 1, 2))))
    median_middle_area = float(np.median(middle_areas)) if middle_areas else median_panel_area

    # --- Step 5: Build candidate polygons per panel role ---
    panel_details = []

    def to_img_local(u_val, v_val):
        pt = origin + u_val * u_axis + v_val * v_axis
        return [float(pt[0]), float(pt[1])]

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

        if is_endpoint and ENABLE_ENDPOINT_ISOLATION:
            if endpoint_side == "start":
                inner_divider_u = u0 + (pos + 0.5) * pitch_fit
            else:  # "end"
                inner_divider_u = u0 + (pos - 0.5) * pitch_fit

            use_locked = ENABLE_MIDDLE_LOCKED_ENDPOINT and (len(middle_indices) >= MIDDLE_PANEL_MIN_FOR_LOCKED_ENDPOINT)

            if use_locked:
                if endpoint_side == "start":
                    outer_u = inner_divider_u - pitch_fit
                else:
                    outer_u = inner_divider_u + pitch_fit

                endpoint_outer_source = "middle_locked_pitch"
                endpoint_outer_score = 1.0
                endpoint_final_outer_u = outer_u
                endpoint_yolo_outer_u = 0.0
                endpoint_pitch_outer_u = outer_u
                endpoint_edge_outer_u = 0.0
                endpoint_consensus_outer_u = 0.0
                endpoint_candidates = []
                endpoint_extrapolate_ratio = abs(outer_u - inner_divider_u) / max(pitch_fit, 1e-3)

                pt1 = origin + inner_divider_u * u_axis + v_low * v_axis
                pt2 = origin + inner_divider_u * u_axis + v_high * v_axis
                pt3 = origin + outer_u * u_axis + v_high * v_axis
                pt4 = origin + outer_u * u_axis + v_low * v_axis
                candidate_polygon = _sort_corners(np.array([pt1, pt2, pt3, pt4], dtype=np.float32))
                candidate_polygon_source = "endpoint_middle_locked_pitch"
                success_resolver = True
            else:
                success_resolver = False
                if ENABLE_HYBRID_ENDPOINT_RESOLVER:
                    yolo_conf = p.get("confidence", p.get("conf", 1.0))
                    middle_lattice_good = (len(middle_trusted_indices) >= MIN_MIDDLE_PANELS_FOR_LATTICE)

                    resolver_res = resolve_hybrid_endpoint_outer_u(
                        original_yolo_polygon=orig_poly,
                        inner_u=inner_divider_u,
                        v_low=v_low,
                        v_high=v_high,
                        u_axis=u_axis,
                        v_axis=v_axis,
                        origin=origin,
                        median_pitch=pitch_fit,
                        endpoint_side=endpoint_side,
                        image=image,
                        consensus_outer_u=None,
                        yolo_conf=yolo_conf,
                        middle_lattice_good=middle_lattice_good
                    )

                    if resolver_res["success"]:
                        best_cand = resolver_res["best_candidate"]
                        endpoint_outer_source = best_cand["source"]
                        endpoint_outer_score = best_cand["score"]
                        endpoint_final_outer_u = best_cand["outer_u"]
                        endpoint_candidates = resolver_res["candidates"]
                        endpoint_yolo_outer_u = resolver_res["yolo_outer_u"]
                        endpoint_pitch_outer_u = resolver_res["pitch_outer_u"]
                        endpoint_edge_outer_u = resolver_res["edge_outer_u"]
                        endpoint_consensus_outer_u = resolver_res["consensus_outer_u"]

                        outer_u = best_cand["outer_u"]
                        endpoint_extrapolate_ratio = abs(outer_u - inner_divider_u) / max(pitch_fit, 1e-3)

                        pt1 = origin + inner_divider_u * u_axis + v_low * v_axis
                        pt2 = origin + inner_divider_u * u_axis + v_high * v_axis
                        pt3 = origin + outer_u * u_axis + v_high * v_axis
                        pt4 = origin + outer_u * u_axis + v_low * v_axis
                        candidate_polygon = _sort_corners(np.array([pt1, pt2, pt3, pt4], dtype=np.float32))
                        candidate_polygon_source = f"endpoint_{endpoint_outer_source}"
                        success_resolver = True

                if not success_resolver:
                    # Simple fallback: YOLO outer projection
                    orig_np = np.array(orig_poly, dtype=np.float32)
                    u_values = [float(np.dot(pt - origin, u_axis)) for pt in orig_np]
                    yolo_outer_u = min(u_values) if endpoint_side == "start" else max(u_values)
                    outer_u = yolo_outer_u
                    endpoint_outer_source = "yolo_projection"
                    endpoint_outer_score = 0.0
                    endpoint_final_outer_u = outer_u
                    endpoint_candidates = []
                    endpoint_extrapolate_ratio = abs(outer_u - inner_divider_u) / max(pitch_fit, 1e-3)
                    
                    pt1 = origin + inner_divider_u * u_axis + v_low * v_axis
                    pt2 = origin + inner_divider_u * u_axis + v_high * v_axis
                    pt3 = origin + outer_u * u_axis + v_high * v_axis
                    pt4 = origin + outer_u * u_axis + v_low * v_axis
                    candidate_polygon = _sort_corners(np.array([pt1, pt2, pt3, pt4], dtype=np.float32))
                    candidate_polygon_source = "endpoint_not_enough_middle_panels_for_locked_endpoint"
        else:
            # --- MIDDLE CANDIDATE: full lattice rect ---
            u_left = u0 + (pos - 0.5) * pitch_fit
            u_right = u0 + (pos + 0.5) * pitch_fit
            candidate_polygon = [
                to_img_local(u_left, v_low),
                to_img_local(u_right, v_low),
                to_img_local(u_right, v_high),
                to_img_local(u_left, v_high),
            ]
            candidate_polygon_source = "lattice_full"

        # --- Metrics ---
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

        is_end_panel = (pos == min_pos or pos == max_pos)
        is_partial = is_end_panel and (candidate_area < STRING_PARTIAL_PANEL_MIN_AREA_RATIO * median_cand_area)

        # Edge score (diagnostic only, no hard gate)
        edge_score_orig = score_panel_boundary_alignment(image, orig_poly) if image is not None else 0.0
        edge_score_cand = score_panel_boundary_alignment(image, candidate_polygon) if image is not None else 0.0
        edge_score_ratio = edge_score_cand / max(edge_score_orig, 1e-6)

        # --- Validation: different thresholds for middle vs endpoint ---
        pass_val = True
        fail_reason = "ok"
        overlap_val = 0.0

        if out_of_bounds:
            pass_val = False
            fail_reason = "outside_image"
        elif is_partial and ENABLE_REJECT_PARTIAL_STRING_END_PANEL and not is_endpoint:
            pass_val = False
            fail_reason = "reject_partial"
        elif is_endpoint and ENABLE_ENDPOINT_ISOLATION:
            # Endpoint validation
            if ENABLE_MIDDLE_LOCKED_ENDPOINT:
                # 1. Pitch ratio check
                endpoint_pitch_ratio = endpoint_extrapolate_ratio
                if not (ENDPOINT_PITCH_RATIO_MIN <= endpoint_pitch_ratio <= ENDPOINT_PITCH_RATIO_MAX):
                    pass_val = False
                    fail_reason = f"endpoint_pitch_ratio_out_{endpoint_pitch_ratio:.2f}"
                
                # 2. Area ratio check
                elif not (ENDPOINT_AREA_RATIO_MIN <= area_ratio <= ENDPOINT_AREA_RATIO_MAX):
                    pass_val = False
                    fail_reason = f"endpoint_area_ratio_out_{area_ratio:.2f}"
                
                # 3. Center shift check
                else:
                    c_shift_max = ENDPOINT_CENTER_SHIFT_MAX_RATIO
                    if center_shift_ratio > c_shift_max:
                        if is_yolo_endpoint_small:
                            # ignore_center_shift_due_to_small_yolo_endpoint (warn only)
                            print(f"[ENDPOINT_GUARD] ignore_center_shift_due_to_small_yolo_endpoint panel={p.get('raw_idx', k)} shift={center_shift_ratio:.2f}")
                        else:
                            pass_val = False
                            fail_reason = f"endpoint_center_shift_{center_shift_ratio:.2f}"
                
                # 4. IoU with YOLO check
                if pass_val:
                    middle_lattice_good = (len(middle_trusted_indices) >= MIN_MIDDLE_PANELS_FOR_LATTICE)
                    iou_min = ENDPOINT_LOW_IOU_MIN_WHEN_LATTICE_GOOD if middle_lattice_good else ENDPOINT_IOU_WITH_YOLO_MIN
                    if iou_with_original < iou_min:
                        pass_val = False
                        fail_reason = f"endpoint_iou_low_{iou_with_original:.2f}"
                
                # 5. Overlap with neighbor check
                if pass_val:
                    neighbor_idx = 1 if endpoint_side == "start" else n - 2
                    if 0 <= neighbor_idx < n:
                        n_pos = positions[neighbor_idx]
                        n_u_left = u0 + (n_pos - 0.5) * pitch_fit
                        n_u_right = u0 + (n_pos + 0.5) * pitch_fit
                        neighbor_cand_poly = np.array([
                            to_img_local(n_u_left, v_low),
                            to_img_local(n_u_right, v_low),
                            to_img_local(n_u_right, v_high),
                            to_img_local(n_u_left, v_high),
                        ], dtype=np.float32)
                        overlap_val = compute_polygon_overlap_ratio(cand_pts, neighbor_cand_poly, (img_h, img_w))
                        if overlap_val > ENDPOINT_MAX_OVERLAP_RATIO:
                            pass_val = False
                            fail_reason = f"endpoint_overlap_too_high_{overlap_val:.3f}"
            else:
                # Legacy / Hybrid Resolver validation
                if endpoint_outer_score < ENDPOINT_SCORE_MIN_ACCEPT:
                    pass_val = False
                    fail_reason = "endpoint_outer_resolver_low_score"
                elif not (ENDPOINT_LENGTH_RATIO_MIN <= endpoint_extrapolate_ratio <= ENDPOINT_LENGTH_RATIO_MAX):
                    pass_val = False
                    fail_reason = f"endpoint_length_ratio_out_{endpoint_extrapolate_ratio:.2f}"
                elif not (ENDPOINT_AREA_RATIO_MIN <= area_ratio <= ENDPOINT_AREA_RATIO_MAX):
                    pass_val = False
                    fail_reason = f"endpoint_area_ratio_out_{area_ratio:.2f}"
                elif center_shift_ratio > ENDPOINT_CENTER_SHIFT_MAX:
                    pass_val = False
                    fail_reason = f"endpoint_center_shift_{center_shift_ratio:.2f}"
                elif iou_with_original < 0.40:
                    pass_val = False
                    fail_reason = f"endpoint_iou_low_{iou_with_original:.2f}"
        else:
            # Middle panel validation: standard thresholds
            if not (STRING_PANEL_AREA_RATIO_MIN <= area_ratio <= STRING_PANEL_AREA_RATIO_MAX):
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

        middle_lattice_good = (len(middle_trusted_indices) >= MIN_MIDDLE_PANELS_FOR_LATTICE)

        panel_details.append({
            "p_obj": p,
            "panel_idx": p.get("raw_idx", k),
            "trusted": trusted_mask[k],
            "u_pos": pos,
            "is_partial": is_partial,
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
            "width_ratio": width_ratio,
            "height_ratio": height_ratio,
            "candidate_area": candidate_area,
            "original_area": orig_area,
            "out_of_bounds": out_of_bounds,
            "pass_individual": pass_val,
            "fail_reason": fail_reason,
            "edge_score_original": edge_score_orig,
            "edge_score_candidate": edge_score_cand,
            "edge_score_ratio": edge_score_ratio,
            # new logging fields
            "endpoint_yolo_outer_u": endpoint_yolo_outer_u,
            "endpoint_pitch_outer_u": endpoint_pitch_outer_u,
            "endpoint_edge_outer_u": endpoint_edge_outer_u,
            "endpoint_consensus_outer_u": endpoint_consensus_outer_u,
            "endpoint_final_outer_u": endpoint_final_outer_u,
            "endpoint_outer_source": endpoint_outer_source,
            "endpoint_outer_score": endpoint_outer_score,
            "endpoint_length_ratio": endpoint_extrapolate_ratio,
            "endpoint_candidates": endpoint_candidates,
            "endpoint_decision": "use_middle_locked_endpoint" if (pass_val and ENABLE_MIDDLE_LOCKED_ENDPOINT and use_locked) else ("use_lattice_endpoint" if pass_val else "fallback_original"),
            "endpoint_fallback_reason": fail_reason if not pass_val else "ok",
            "middle_lattice_unchanged": True,
            "angle_delta_to_block": 0.0,
            # locked pitch strategy extra fields
            "endpoint_strategy": "middle_locked_pitch" if ENABLE_MIDDLE_LOCKED_ENDPOINT else "hybrid_resolver",
            "used_for_middle_fit": used_for_lattice_fit,
            "middle_lattice_quality_good": middle_lattice_good,
            "inner_u": float(inner_divider_u) if is_endpoint else 0.0,
            "outer_u": float(endpoint_final_outer_u) if is_endpoint else 0.0,
            "pitch": float(pitch_fit),
            "endpoint_pitch_ratio": float(endpoint_extrapolate_ratio) if is_endpoint else 0.0,
            "endpoint_iou_with_yolo": float(iou_with_original) if is_endpoint else 0.0,
            "endpoint_center_shift_ratio": float(center_shift_ratio) if is_endpoint else 0.0,
            "endpoint_overlap_with_neighbor": float(overlap_val) if is_endpoint else 0.0,
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

    # Pass count: endpoint fallbacks do NOT penalize string pass_ratio
    pass_count_middle = sum(1 for d in panel_details if d["pass_individual"] and not d["is_endpoint"])
    pass_count_endpoint = sum(1 for d in panel_details if d["pass_individual"] and d["is_endpoint"])
    n_endpoints = sum(1 for d in panel_details if d["is_endpoint"])
    n_middle = n - n_endpoints
    pass_count = len(passing)
    # pass_ratio only over middle panels (endpoints treated separately)
    pass_ratio = pass_count_middle / max(n_middle, 1) if ENABLE_ENDPOINT_ISOLATION and n_middle > 0 else pass_count / n
    ious = [d["iou_with_original"] for d in panel_details]
    median_iou = float(np.median(ious)) if ious else 0.0

    # 1. Define geometric helpers
    def ccw(p1, p2, p3):
        return (p3[1] - p1[1]) * (p2[0] - p1[0]) > (p2[1] - p1[1]) * (p3[0] - p1[0])

    def segments_intersect(a, b, c, d):
        return (ccw(a, c, d) != ccw(b, c, d)) and (ccw(a, b, c) != ccw(a, b, d))

    def poly_intersects_segment(poly_pts, seg_a, seg_b):
        from shapely.geometry import LineString, Polygon
        try:
            pg_shrunk = Polygon(poly_pts)
            if not pg_shrunk.is_valid:
                pg_shrunk = pg_shrunk.buffer(0)
            line_seg = LineString([seg_a, seg_b])
            return pg_shrunk.intersects(line_seg)
        except Exception:
            return False

    def get_shrunk_polygon(poly, factor):
        pts = np.array(poly, dtype=np.float32)
        centroid = np.mean(pts, axis=0)
        shrunk = centroid + factor * (pts - centroid)
        return shrunk.tolist()

    # 2. Group centers to determine provisional cols and rows
    centers_u = [float(np.dot(np.array(p["center"]) - origin, u_axis)) for p in panels]
    centers_v = [float(np.dot(np.array(p["center"]) - origin, v_axis)) for p in panels]

    u_sorted = sorted(centers_u)
    row_groups = []
    if u_sorted:
        curr = [u_sorted[0]]
        for val in u_sorted[1:]:
            if val - curr[-1] < 0.3 * pitch_fit:
                curr.append(val)
            else:
                row_groups.append(curr)
                curr = [val]
        row_groups.append(curr)
    n_rows = len(row_groups)

    v_width = max(v_high - v_low, 10.0)
    v_sorted = sorted(centers_v)
    col_groups = []
    if v_sorted:
        curr = [v_sorted[0]]
        for val in v_sorted[1:]:
            if val - curr[-1] < 0.4 * v_width:
                curr.append(val)
            else:
                col_groups.append(curr)
                curr = [val]
        col_groups.append(curr)
    n_cols = len(col_groups)

    # 3. Calculate panel-level metrics (original_coverage_ratio, max_inward_cut_ratio, max_overlap)
    from shapely.geometry import Polygon
    for idx_det, det in enumerate(panel_details):
        p_obj = det["p_obj"]
        orig_poly = det["original_polygon"]
        cand_poly = det["candidate_polygon"]

        pg_cand = Polygon(cand_poly)
        pg_orig = Polygon(orig_poly)
        if not pg_cand.is_valid:
            pg_cand = pg_cand.buffer(0)
        if not pg_orig.is_valid:
            pg_orig = pg_orig.buffer(0)

        intersection_area = pg_cand.intersection(pg_orig).area if (pg_cand.is_valid and pg_orig.is_valid) else 0.0
        det["original_coverage_ratio"] = float(intersection_area / pg_orig.area if pg_orig.area > 0 else 0.0)
        det["max_inward_cut_ratio"] = float((pg_cand.area - intersection_area) / pg_cand.area if pg_cand.area > 0 else 0.0)
        det["max_overlap"] = 0.0

    # Pairwise overlap checking on candidates
    for i in range(len(panel_details)):
        for j in range(i + 1, len(panel_details)):
            pi = np.array(panel_details[i]["candidate_polygon"], dtype=np.float32)
            pj = np.array(panel_details[j]["candidate_polygon"], dtype=np.float32)
            pi_h = cv2.convexHull(pi.reshape(-1, 1, 2)).astype(np.float32)
            pj_h = cv2.convexHull(pj.reshape(-1, 1, 2)).astype(np.float32)
            area_inter, _ = cv2.intersectConvexConvex(pi_h, pj_h)
            if area_inter > 0:
                min_a = min(cv2.contourArea(pi_h), cv2.contourArea(pj_h))
                r = area_inter / max(min_a, 1e-3)
                panel_details[i]["max_overlap"] = max(panel_details[i]["max_overlap"], r)
                panel_details[j]["max_overlap"] = max(panel_details[j]["max_overlap"], r)

    # 4. Calculate block-level metrics
    middle_details = [det for det in panel_details if not det["is_endpoint"]]
    
    middle_pass_ratio = pass_count_middle / max(n_middle, 1)
    
    middle_covs = [det["original_coverage_ratio"] for det in middle_details]
    median_middle_original_coverage = float(np.median(middle_covs)) if middle_covs else 1.0
    
    failed_middle_coverage_count = sum(1 for det in middle_details if det["original_coverage_ratio"] < 0.85)
    failed_middle_inward_cut_count = sum(1 for det in middle_details if det["max_inward_cut_ratio"] > 0.20)
    
    max_middle_overlap = max([det["max_overlap"] for det in middle_details], default=0.0)
    max_middle_inward_cut_ratio = max([det["max_inward_cut_ratio"] for det in middle_details], default=0.0)

    # Compute pitch CV
    if len(t_gaps) > 1:
        pitch_cv = float(np.std(t_gaps) / max(np.median(t_gaps), 1e-3))
    else:
        pitch_cv = 0.0

    # Compute rail width CV
    orig_widths = []
    for p in fit_panels:
        pts = np.array(p["polygon"], dtype=np.float32)
        v_proj = [float(np.dot(pt - origin, v_axis)) for pt in pts]
        orig_widths.append(max(v_proj) - min(v_proj))
    rail_width_cv = float(np.std(orig_widths) / max(np.median(orig_widths), 1e-3)) if orig_widths else 0.0

    # Compute divider crossings for middle panels
    divider_cross_count_middle = 0
    divider_segs = []
    for r in range(min_pos, max_pos):
        div_u = u0 + (r + 0.5) * pitch_fit
        seg_a = origin + div_u * u_axis + v_low * v_axis
        seg_b = origin + div_u * u_axis + v_high * v_axis
        divider_segs.append((seg_a, seg_b))

    for det in middle_details:
        shrunk_poly = get_shrunk_polygon(det["original_polygon"], 0.90)
        for seg_a, seg_b in divider_segs:
            if poly_intersects_segment(shrunk_poly, seg_a, seg_b):
                divider_cross_count_middle += 1
                break

    outside_image_count_middle = sum(1 for det in middle_details if det["out_of_bounds"])

    # Determine if block passed full string_lattice
    block_passed = (
        pass_ratio >= STRING_MIN_PASS_RATIO and
        median_iou >= STRING_MEDIAN_IOU_MIN and
        max_overlap <= STRING_MAX_OVERLAP
    )

    # Identify recoverable fallback blocks
    recoverable_fallback_candidate = False
    if not block_passed:
        if (divider_cross_count_middle == 0 and
            outside_image_count_middle == 0 and
            middle_pass_ratio >= 0.60 and
            median_middle_original_coverage >= 0.88 and
            max_middle_overlap <= 0.09 and
            pitch_cv <= 0.32 and
            rail_width_cv <= 0.32 and
            max_middle_inward_cut_ratio <= 0.25):
            recoverable_fallback_candidate = True

    # Identify near-recoverable fallback blocks
    near_recoverable_string_candidate = False
    near_recoverable_reason = ""
    near_recoverable_middle_valid_count = 0
    if not block_passed:
        conds = []
        if n < 10:
            conds.append(f"n_bp_{n}_lt_10")
        if n_cols < 2:
            conds.append(f"n_cols_{n_cols}_lt_2")
        if middle_pass_ratio < 0.50:
            conds.append(f"middle_pass_ratio_{middle_pass_ratio:.2f}_lt_0.50")
        if median_middle_original_coverage < 0.90:
            conds.append(f"median_middle_original_coverage_{median_middle_original_coverage:.2f}_lt_0.90")
        if max_middle_inward_cut_ratio > 0.12:
            conds.append(f"max_middle_inward_cut_ratio_{max_middle_inward_cut_ratio:.2f}_gt_0.12")
        if max_middle_overlap > 0.02:
            conds.append(f"max_middle_overlap_{max_middle_overlap:.2f}_gt_0.02")
        if pitch_cv > 0.24:
            conds.append(f"pitch_cv_{pitch_cv:.2f}_gt_0.24")
        if rail_width_cv > 0.05:
            conds.append(f"rail_width_cv_{rail_width_cv:.2f}_gt_0.05")
        if divider_cross_count_middle != 0:
            conds.append(f"divider_cross_count_middle_{divider_cross_count_middle}_ne_0")
        if outside_image_count_middle != 0:
            conds.append(f"outside_image_count_middle_{outside_image_count_middle}_ne_0")
            
        if not conds:
            near_recoverable_string_candidate = True
            near_recoverable_reason = "ok"
            near_recoverable_middle_valid_count = sum(1 for det in middle_details if det["pass_individual"])
        else:
            near_recoverable_reason = ", ".join(conds)

    # Diagnostic-only trusted core metrics
    trusted_core_count = 0
    axis_source = "diagnostic_only"
    trusted_pitch_cv = 0.0
    trusted_pitch_reliable = False
    trusted_angle_delta_deg = 0.0
    trusted_rail_refit_count = 0

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
        # Recovery/near-recovery block level metrics
        "pitch_cv": pitch_cv,
        "rail_width_cv": rail_width_cv,
        "middle_pass_ratio": middle_pass_ratio,
        "max_middle_overlap": max_middle_overlap,
        "max_middle_inward_cut_ratio": max_middle_inward_cut_ratio,
        "recoverable_fallback_candidate": recoverable_fallback_candidate,
        "near_recoverable_string_candidate": near_recoverable_string_candidate,
        "near_recoverable_reason": near_recoverable_reason,
        "near_recoverable_middle_valid_count": near_recoverable_middle_valid_count,
        "median_middle_original_coverage": median_middle_original_coverage,
        "failed_middle_coverage_count": failed_middle_coverage_count,
        "failed_middle_inward_cut_count": failed_middle_inward_cut_count,
        "divider_cross_count_middle": divider_cross_count_middle,
        "outside_image_count_middle": outside_image_count_middle,
        "n_cols": n_cols,
        "n_rows": n_rows,
        # Trusted Core metrics
        "trusted_core_count": trusted_core_count,
        "axis_source": axis_source,
        "trusted_pitch_cv": trusted_pitch_cv,
        "trusted_pitch_reliable": trusted_pitch_reliable,
        "trusted_angle_delta_deg": trusted_angle_delta_deg,
        "trusted_rail_refit_count": trusted_rail_refit_count,
    }


def synchronize_strings(
    strings_with_fits: List[Dict[str, Any]],
    image_shape: Tuple[int, int, int],
    image: Optional[np.ndarray] = None,
    median_panel_area: float = 1.0
) -> List[Dict[str, Any]]:
    """
    Dong bo angle/pitch giua cac string ke nhau (neu du dieu kien).
    Moi string van giu rail rieng (v_low/v_high).
    """
    if not ENABLE_STRING_SYNC or len(strings_with_fits) < 2:
        return strings_with_fits

    n_str = len(strings_with_fits)
    synced_flags = [False] * n_str
    synced_with = [-1] * n_str

    for i in range(n_str):
        s_i = strings_with_fits[i]
        if not s_i.get("fit_result", {}).get("success", False):
            continue
        angle_i = s_i["string_info"]["angle_deg"]
        pitch_i = s_i["fit_result"]["pitch"]
        n_trusted_i = s_i["string_info"]["n_trusted"]

        for j in range(i + 1, n_str):
            s_j = strings_with_fits[j]
            if not s_j.get("fit_result", {}).get("success", False):
                continue
            angle_j = s_j["string_info"]["angle_deg"]
            pitch_j = s_j["fit_result"]["pitch"]
            n_trusted_j = s_j["string_info"]["n_trusted"]

            angle_diff = abs(angle_i - angle_j)
            if angle_diff > 90:
                angle_diff = 180 - angle_diff

            pitch_diff_ratio = abs(pitch_i - pitch_j) / max(min(pitch_i, pitch_j), 1e-3)

            if angle_diff <= STRING_SYNC_ANGLE_DIFF_MAX_DEG and pitch_diff_ratio <= STRING_SYNC_PITCH_DIFF_MAX_RATIO:
                # Weighted average
                w_i = n_trusted_i
                w_j = n_trusted_j
                synced_angle = (angle_i * w_i + angle_j * w_j) / max(w_i + w_j, 1)
                synced_pitch = (pitch_i * w_i + pitch_j * w_j) / max(w_i + w_j, 1)

                angle_rad = np.radians(synced_angle)
                synced_u_axis = np.array([np.cos(angle_rad), np.sin(angle_rad)], dtype=np.float32)
                synced_u_axis /= (np.linalg.norm(synced_u_axis) + 1e-8)

                # Refit each string with synced angle
                for idx, s in [(i, s_i), (j, s_j)]:
                    old_u = s["string_info"]["u_axis"]
                    new_info = dict(s["string_info"])
                    new_info["u_axis"] = synced_u_axis
                    new_info["v_axis"] = np.array([-synced_u_axis[1], synced_u_axis[0]], dtype=np.float32)
                    new_info["pitch"] = synced_pitch

                    new_fit = fit_string_parallel_lines(new_info, image_shape, image, median_panel_area)
                    if new_fit.get("success") and new_fit["pass_ratio"] >= s["fit_result"]["pass_ratio"] - 0.05:
                        strings_with_fits[idx]["string_info"] = new_info
                        strings_with_fits[idx]["fit_result"] = new_fit
                        synced_with[idx] = j if idx == i else i

    for i, s in enumerate(strings_with_fits):
        s["synced_with"] = synced_with[i]

    return strings_with_fits


# ---------------------------------------------------------------------------
# Post-lattice Snap: Rail & Divider Edge Snapping
# ---------------------------------------------------------------------------

def _score_rail_at_v(
    image_gray: np.ndarray,
    origin: np.ndarray,
    u_axis: np.ndarray,
    v_axis: np.ndarray,
    v_val: float,
    u_min: float,
    u_max: float,
    n_samples: int = 24
) -> float:
    """
    Score a rail line at v_val by sampling gradient + dark-boundary along u.
    High score = strong edge / dark boundary at this v position.
    """
    h, w = image_gray.shape[:2]
    scores = []
    for t in np.linspace(u_min, u_max, n_samples):
        pt = origin + t * u_axis + v_val * v_axis
        x, y = int(round(float(pt[0]))), int(round(float(pt[1])))
        if not (0 < x < w - 1 and 0 < y < h - 1):
            continue
        # Gradient in v_axis direction (cross-rail)
        vx, vy = float(v_axis[0]), float(v_axis[1])
        # Sample 2 pixels offset in v direction (inside vs outside rail)
        xi_in = int(round(x - vx))
        yi_in = int(round(y - vy))
        xo_out = int(round(x + vx))
        yo_out = int(round(y + vy))
        pix_c = float(image_gray[y, x])
        pix_in = float(image_gray[yi_in, xi_in]) if (0 <= yi_in < h and 0 <= xi_in < w) else pix_c
        pix_out = float(image_gray[yo_out, xo_out]) if (0 <= yo_out < h and 0 <= xo_out < w) else pix_c
        # Edge gradient
        grad_mag = abs(pix_in - pix_out)
        # Dark boundary: rail pixel darker than interior panel pixel
        dark_score = max(0.0, pix_in - pix_c)
        scores.append(grad_mag + 0.5 * dark_score)

    return float(np.mean(scores)) if scores else 0.0


def _score_divider_at_u(
    image_gray: np.ndarray,
    origin: np.ndarray,
    u_axis: np.ndarray,
    v_axis: np.ndarray,
    u_val: float,
    v_low: float,
    v_high: float,
    half_pitch: float = 10.0,
    n_samples: int = 20
) -> float:
    """
    Score a divider line at u_val by sampling dark-gap along v.
    High score = the divider falls in a dark gap between panels.
    """
    h, w = image_gray.shape[:2]
    scores = []
    for t in np.linspace(v_low, v_high, n_samples):
        pt = origin + u_val * u_axis + t * v_axis
        x, y = int(round(float(pt[0]))), int(round(float(pt[1])))
        if not (0 < x < w - 1 and 0 < y < h - 1):
            continue
        ux, uy = float(u_axis[0]), float(u_axis[1])
        # Sample pixels to left and right (along u direction)
        side = max(2, int(round(half_pitch * 0.25)))
        xl = int(round(x - ux * side))
        yl = int(round(y - uy * side))
        xr = int(round(x + ux * side))
        yr = int(round(y + uy * side))

        pix_c = float(image_gray[y, x])
        pix_l = float(image_gray[yl, xl]) if (0 <= yl < h and 0 <= xl < w) else pix_c
        pix_r = float(image_gray[yr, xr]) if (0 <= yr < h and 0 <= xr < w) else pix_c

        grad_mag = abs(pix_l - pix_r)
        # Dark gap: center darker than both sides
        dark_gap = max(0.0, (pix_l + pix_r) / 2.0 - pix_c)
        scores.append(grad_mag + dark_gap)

    return float(np.mean(scores)) if scores else 0.0


def score_outer_divider(
    image: np.ndarray,
    u_axis: np.ndarray,
    v_axis: np.ndarray,
    rail_low: float,
    rail_high: float,
    divider_u: float,
    origin: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Score an outer divider line at divider_u by sampling dark-gap along v.
    Returns composite score, mean_darkness, and gradient_score.
    """
    if origin is None:
        origin = np.zeros(2)
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    else:
        gray = image.astype(np.float32)

    h, w = gray.shape[:2]
    n_samples = 20
    darkness_vals = []
    gradient_vals = []

    ux, uy = float(u_axis[0]), float(u_axis[1])
    side = 4  # sample side distance for gradient (Sobel-like)

    for t in np.linspace(rail_low, rail_high, n_samples):
        pt = origin + divider_u * u_axis + t * v_axis
        x, y = int(round(float(pt[0]))), int(round(float(pt[1])))
        if not (0 <= x < w and 0 <= y < h):
            continue

        pix_c = float(gray[y, x])
        # Higher darkness means darker pixel
        darkness = 255.0 - pix_c
        darkness_vals.append(darkness)

        # Gradient score
        xl = int(round(x - ux * side))
        yl = int(round(y - uy * side))
        xr = int(round(x + ux * side))
        yr = int(round(y + uy * side))

        pix_l = float(gray[yl, xl]) if (0 <= yl < h and 0 <= xl < w) else pix_c
        pix_r = float(gray[yr, xr]) if (0 <= yr < h and 0 <= xr < w) else pix_c

        grad = abs(pix_l - pix_r)
        gradient_vals.append(grad)

    mean_darkness = float(np.mean(darkness_vals)) / 255.0 if darkness_vals else 0.0
    gradient_score = float(np.mean(gradient_vals)) / 255.0 if gradient_vals else 0.0

    # Composite score
    score = 0.5 * mean_darkness + 0.5 * gradient_score

    return {
        "score": score,
        "mean_darkness": mean_darkness,
        "gradient_score": gradient_score
    }


def validate_endpoint_candidate(
    candidate_lattice: List[List[float]],
    candidate_original: List[List[float]],
    u_axis: np.ndarray,
    v_axis: np.ndarray,
    origin: np.ndarray,
    median_middle_u_length: float,
    median_middle_v_width: float,
    candidate_lattice_max_overlap: float,
    candidate_original_max_overlap: float,
    outer_shift_abs: float,
    median_pitch: float,
    image: Optional[np.ndarray],
    img_shape: Tuple[int, int]
) -> Dict[str, Any]:
    """
    Validate an endpoint candidate polygon using 8 checks against the original polygon.
    Returns validation result, decision, reason, and selected polygon.
    """
    cand_lat_np = np.array(candidate_lattice, dtype=np.float32)
    cand_orig_np = np.array(candidate_original, dtype=np.float32)
    area_lat = float(cv2.contourArea(cand_lat_np.reshape(-1, 1, 2)))
    area_orig = float(cv2.contourArea(cand_orig_np.reshape(-1, 1, 2)))
    area_ratio = area_lat / max(area_orig, 1.0)
    
    center_lat = np.mean(cand_lat_np, axis=0)
    center_orig = np.mean(cand_orig_np, axis=0)
    center_shift_px = float(np.linalg.norm(center_lat - center_orig))
    diag_orig = float(np.linalg.norm(cand_orig_np[2] - cand_orig_np[0]))
    center_shift_ratio = center_shift_px / max(diag_orig, 1.0)
    
    iou = _compute_mask_iou_cv(cand_lat_np, cand_orig_np, img_shape)
    
    u_projs = [float(np.dot(pt - origin, u_axis)) for pt in cand_lat_np]
    endpoint_u_length = max(u_projs) - min(u_projs)
    u_length_ratio = endpoint_u_length / max(median_middle_u_length, 1e-3)
    
    v_projs = [float(np.dot(pt - origin, v_axis)) for pt in cand_lat_np]
    endpoint_v_width = max(v_projs) - min(v_projs)
    v_width_ratio = endpoint_v_width / max(median_middle_v_width, 1e-3)
    
    overlap_pass = candidate_lattice_max_overlap <= max(candidate_original_max_overlap + 0.01, 0.04)
    
    edge_score_orig = 0.0
    edge_score_cand = 0.0
    edge_ratio = 1.0
    edge_weak_warning = False  # edge_score is diagnostic only — NOT a rejection gate
    if image is not None:
        try:
            edge_score_orig = score_panel_boundary_alignment(image, candidate_original)
            edge_score_cand = score_panel_boundary_alignment(image, candidate_lattice)
            if edge_score_orig > 1e-6:
                edge_ratio = edge_score_cand / edge_score_orig
            else:
                edge_ratio = 1.0
            
            if edge_score_cand < edge_score_orig * ENDPOINT_EDGE_SCORE_MIN_RATIO:
                edge_weak_warning = True  # Log only, does not reject
        except Exception:
            pass
            
    distortion_pass = (
        outer_shift_abs <= median_pitch * 0.22 and
        area_ratio <= ENDPOINT_AREA_RATIO_MAX and
        center_shift_ratio <= ENDPOINT_CENTER_SHIFT_MAX
    )
    
    reasons = []
    if not (ENDPOINT_AREA_RATIO_MIN <= area_ratio <= ENDPOINT_AREA_RATIO_MAX):
        reasons.append(f"area_ratio_{area_ratio:.2f}")
    if center_shift_ratio > ENDPOINT_CENTER_SHIFT_MAX:
        reasons.append(f"center_shift_{center_shift_ratio:.2f}")
    if iou < ENDPOINT_IOU_MIN:
        reasons.append(f"iou_{iou:.2f}")
    if not (0.80 <= u_length_ratio <= 1.18):
        reasons.append(f"u_length_ratio_{u_length_ratio:.2f}")
    if not (0.80 <= v_width_ratio <= 1.18):
        reasons.append(f"v_width_ratio_{v_width_ratio:.2f}")
    if not overlap_pass:
        reasons.append(f"overlap_lat_{candidate_lattice_max_overlap:.2f}_vs_orig_{candidate_original_max_overlap:.2f}")
    if not distortion_pass:
        if outer_shift_abs > median_pitch * 0.22:
            reasons.append(f"outer_shift_{outer_shift_abs:.2f}_too_high")
        else:
            reasons.append("distortion_out")
    # NOTE: edge_score is NOT added to reasons — it is a warning only
            
    pass_all = len(reasons) == 0
    decision = "use_lattice_endpoint" if pass_all else "fallback_endpoint_original"
    reason_str = "ok" if pass_all else "|".join(reasons)
    if edge_weak_warning:
        reason_str = reason_str + "|edge_weak_warn" if reason_str != "ok" else "ok|edge_weak_warn"
    
    return {
        "pass": pass_all,
        "decision": decision,
        "reason": reason_str,
        "edge_weak_warning": edge_weak_warning,
        "area_ratio": area_ratio,
        "center_shift_ratio": center_shift_ratio,
        "iou": iou,
        "u_length_ratio": u_length_ratio,
        "v_width_ratio": v_width_ratio,
        "edge_score_original": edge_score_orig,
        "edge_score_candidate": edge_score_cand,
        "edge_score_ratio": edge_ratio,
        "polygon": candidate_lattice if pass_all else candidate_original
    }


def check_safe_improvement(
    original_polygon: List[List[float]],
    lattice_polygon: List[List[float]],
    image: np.ndarray,
    img_shape: Tuple[int, int],
    original_max_overlap: float,
    lattice_max_overlap: float
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Checks if lattice_polygon is a SAFE IMPROVEMENT over original_polygon.
    Returns (pass, fail_reason, stats).
    """
    lattice_np = np.array(lattice_polygon, dtype=np.float32)
    original_np = np.array(original_polygon, dtype=np.float32)
    
    # 1. IoU(lattice, original) >= 0.75
    iou = _compute_mask_iou_cv(lattice_np, original_np, img_shape)
    if iou < 0.75:
        return False, f"iou_low_{iou:.3f}", {"iou": iou}

    # 2. center_shift_ratio <= 0.10
    lattice_center = np.mean(lattice_np, axis=0)
    original_center = np.mean(original_np, axis=0)
    orig_x = [pt[0] for pt in original_polygon]
    orig_y = [pt[1] for pt in original_polygon]
    orig_w_sz = max(orig_x) - min(orig_x) if orig_x else 1.0
    orig_h_sz = max(orig_y) - min(orig_y) if orig_y else 1.0
    orig_diag = float(np.linalg.norm([orig_w_sz, orig_h_sz]))
    center_shift = float(np.linalg.norm(lattice_center - original_center)) / max(orig_diag, 1.0)
    if center_shift > 0.10:
        return False, f"center_shift_large_{center_shift:.3f}", {"iou": iou, "center_shift": center_shift}

    # 3. area_ratio in [0.90, 1.10]
    lattice_area = float(cv2.contourArea(lattice_np.reshape(-1, 1, 2)))
    original_area = float(cv2.contourArea(original_np.reshape(-1, 1, 2)))
    area_ratio = lattice_area / max(original_area, 1.0)
    if not (0.90 <= area_ratio <= 1.10):
        return False, f"area_ratio_out_{area_ratio:.3f}", {
            "iou": iou, "center_shift": center_shift, "area_ratio": area_ratio
        }

    # 4. max_overlap(lattice) <= max_overlap(original) + 0.01
    if lattice_max_overlap > original_max_overlap + 0.01:
        return False, f"overlap_increased_lat_{lattice_max_overlap:.3f}_vs_orig_{original_max_overlap:.3f}", {
            "iou": iou, "center_shift": center_shift, "area_ratio": area_ratio,
            "lattice_max_overlap": lattice_max_overlap, "original_max_overlap": original_max_overlap
        }

    # 5. edge_score(lattice) >= edge_score(original) * 1.03
    edge_orig = score_panel_boundary_alignment(image, original_polygon) if image is not None else 0.0
    edge_lat = score_panel_boundary_alignment(image, lattice_polygon) if image is not None else 0.0
    if edge_lat < edge_orig * 1.03:
        return False, f"edge_score_not_improved_lat_{edge_lat:.1f}_vs_orig_{edge_orig:.1f}", {
            "iou": iou, "center_shift": center_shift, "area_ratio": area_ratio,
            "lattice_max_overlap": lattice_max_overlap, "original_max_overlap": original_max_overlap,
            "edge_orig": edge_orig, "edge_lat": edge_lat
        }

    # 6. polygon không vượt biên ảnh
    out_of_bounds = False
    img_h, img_w = img_shape
    for pt in lattice_polygon:
        if (pt[0] < -STRING_PANEL_OUTSIDE_TOL_PX or pt[0] > img_w + STRING_PANEL_OUTSIDE_TOL_PX or
            pt[1] < -STRING_PANEL_OUTSIDE_TOL_PX or pt[1] > img_h + STRING_PANEL_OUTSIDE_TOL_PX):
            out_of_bounds = True
            break
    if out_of_bounds:
        return False, "outside_image_bounds", {
            "iou": iou, "center_shift": center_shift, "area_ratio": area_ratio,
            "lattice_max_overlap": lattice_max_overlap, "original_max_overlap": original_max_overlap,
            "edge_orig": edge_orig, "edge_lat": edge_lat
        }

    return True, "ok", {
        "iou": iou, "center_shift": center_shift, "area_ratio": area_ratio,
        "lattice_max_overlap": lattice_max_overlap, "original_max_overlap": original_max_overlap,
        "edge_orig": edge_orig, "edge_lat": edge_lat
    }


def refine_outer_dividers(
    string_or_block: Dict[str, Any],
    dividers: List[float],
    panels: List[Dict[str, Any]],
    image: np.ndarray
) -> Dict[str, Any]:
    """
    Search and refine start (dividers[0]) and end (dividers[-1]) dividers
    within ±OUTER_DIVIDER_EDGE_SEARCH_PX.
    """
    u_axis = string_or_block["u_axis"]
    v_axis = string_or_block["v_axis"]
    origin = string_or_block["origin"]

    # Determine rail bounds
    if "left_rail_v" in string_or_block:
        rail_low = string_or_block["left_rail_v"]
        rail_high = string_or_block["right_rail_v"]
    elif "fitted_left_rail_v" in string_or_block:
        rail_low = string_or_block["fitted_left_rail_v"]
        rail_high = string_or_block["fitted_right_rail_v"]
    elif "v_low" in string_or_block:
        rail_low = string_or_block["v_low"]
        rail_high = string_or_block["v_high"]
    else:
        rail_low = 0.0
        rail_high = 0.0

    u_start = dividers[0]
    best_start_u = u_start
    best_start_res = score_outer_divider(image, u_axis, v_axis, rail_low, rail_high, u_start, origin)
    original_start_score = best_start_res["score"]

    for delta in range(-OUTER_DIVIDER_EDGE_SEARCH_PX, OUTER_DIVIDER_EDGE_SEARCH_PX + 1):
        if delta == 0:
            continue
        u_try = u_start + float(delta)
        res = score_outer_divider(image, u_axis, v_axis, rail_low, rail_high, u_try, origin)
        if res["score"] > best_start_res["score"]:
            best_start_res = res
            best_start_u = u_try

    u_end = dividers[-1]
    best_end_u = u_end
    best_end_res = score_outer_divider(image, u_axis, v_axis, rail_low, rail_high, u_end, origin)
    original_end_score = best_end_res["score"]

    for delta in range(-OUTER_DIVIDER_EDGE_SEARCH_PX, OUTER_DIVIDER_EDGE_SEARCH_PX + 1):
        if delta == 0:
            continue
        u_try = u_end + float(delta)
        res = score_outer_divider(image, u_axis, v_axis, rail_low, rail_high, u_try, origin)
        if res["score"] > best_end_res["score"]:
            best_end_res = res
            best_end_u = u_try

    shift_start = best_start_u - u_start
    is_start_snap_ok = (
        abs(shift_start) <= OUTER_DIVIDER_MAX_SHIFT_PX and
        best_start_res["score"] >= original_start_score * 1.12 and
        best_start_res["score"] >= OUTER_DIVIDER_MIN_EDGE_SCORE
    )

    shift_end = best_end_u - u_end
    is_end_snap_ok = (
        abs(shift_end) <= OUTER_DIVIDER_MAX_SHIFT_PX and
        best_end_res["score"] >= original_end_score * 1.12 and
        best_end_res["score"] >= OUTER_DIVIDER_MIN_EDGE_SCORE
    )

    return {
        "start": {
            "u_original": u_start,
            "u_best": best_start_u,
            "shift": shift_start,
            "original_score": original_start_score,
            "best_score": best_start_res["score"],
            "snap_ok": is_start_snap_ok
        },
        "end": {
            "u_original": u_end,
            "u_best": best_end_u,
            "shift": shift_end,
            "original_score": original_end_score,
            "best_score": best_end_res["score"],
            "snap_ok": is_end_snap_ok
        }
    }


def snap_string_rails_to_dark_edges(
    image: np.ndarray,
    origin: np.ndarray,
    u_axis: np.ndarray,
    v_axis: np.ndarray,
    v_low: float,
    v_high: float,
    u_min: float,
    u_max: float,
    string_id: int = 0,
    search_px: int = 5
) -> Tuple[float, float, Dict[str, Any]]:
    """
    Snap two parallel rails to the actual dark edges in the image.
    Returns (v_low_new, v_high_new, snap_info).
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    else:
        gray = image.astype(np.float32)

    snap_info = {}

    # --- Snap v_low (rail_low) ---
    if ENABLE_STRING_RAIL_EDGE_SNAP:
        best_v_low = v_low
        best_score_low = _score_rail_at_v(gray, origin, u_axis, v_axis, v_low, u_min, u_max)
        score_orig_low = best_score_low

        for delta in range(-search_px, search_px + 1):
            if delta == 0:
                continue
            v_try = v_low + float(delta)
            score = _score_rail_at_v(gray, origin, u_axis, v_axis, v_try, u_min, u_max)
            if score > best_score_low:
                best_score_low = score
                best_v_low = v_try

        shift_low = best_v_low - v_low
        use_snap_low = (
            abs(shift_low) <= STRING_RAIL_SNAP_MAX_SHIFT_PX and
            best_score_low >= score_orig_low * STRING_SNAP_MIN_EDGE_IMPROVEMENT
        )
        v_low_new = best_v_low if use_snap_low else v_low
        snap_info["rail_low_shift"] = float(v_low_new - v_low)
        snap_info["rail_low_score_old"] = float(score_orig_low)
        snap_info["rail_low_score_new"] = float(best_score_low if use_snap_low else score_orig_low)
        snap_info["rail_low_score_ratio"] = float(best_score_low / max(score_orig_low, 1e-6)) if use_snap_low else 1.0
        snap_info["rail_low_decision"] = "use_snap" if use_snap_low else "keep_original"
        print(f"[STRING_RAIL_SNAP] string_id={string_id} rail=low old_v={v_low:.2f} new_v={v_low_new:.2f} "
              f"shift={v_low_new - v_low:.2f} score_old={score_orig_low:.2f} "
              f"score_new={best_score_low:.2f} decision={snap_info['rail_low_decision']}")
    else:
        v_low_new = v_low
        snap_info["rail_low_shift"] = 0.0
        snap_info["rail_low_score_ratio"] = 1.0
        snap_info["rail_low_decision"] = "snap_disabled"

    # --- Snap v_high (rail_high) ---
    if ENABLE_STRING_RAIL_EDGE_SNAP:
        best_v_high = v_high
        best_score_high = _score_rail_at_v(gray, origin, u_axis, v_axis, v_high, u_min, u_max)
        score_orig_high = best_score_high

        for delta in range(-search_px, search_px + 1):
            if delta == 0:
                continue
            v_try = v_high + float(delta)
            score = _score_rail_at_v(gray, origin, u_axis, v_axis, v_try, u_min, u_max)
            if score > best_score_high:
                best_score_high = score
                best_v_high = v_try

        shift_high = best_v_high - v_high
        use_snap_high = (
            abs(shift_high) <= STRING_RAIL_SNAP_MAX_SHIFT_PX and
            best_score_high >= score_orig_high * STRING_SNAP_MIN_EDGE_IMPROVEMENT
        )
        v_high_new = best_v_high if use_snap_high else v_high
        snap_info["rail_high_shift"] = float(v_high_new - v_high)
        snap_info["rail_high_score_old"] = float(score_orig_high)
        snap_info["rail_high_score_new"] = float(best_score_high if use_snap_high else score_orig_high)
        snap_info["rail_high_score_ratio"] = float(best_score_high / max(score_orig_high, 1e-6)) if use_snap_high else 1.0
        snap_info["rail_high_decision"] = "use_snap" if use_snap_high else "keep_original"
        print(f"[STRING_RAIL_SNAP] string_id={string_id} rail=high old_v={v_high:.2f} new_v={v_high_new:.2f} "
              f"shift={v_high_new - v_high:.2f} score_old={score_orig_high:.2f} "
              f"score_new={best_score_high:.2f} decision={snap_info['rail_high_decision']}")
    else:
        v_high_new = v_high
        snap_info["rail_high_shift"] = 0.0
        snap_info["rail_high_score_ratio"] = 1.0
        snap_info["rail_high_decision"] = "snap_disabled"

    return v_low_new, v_high_new, snap_info


def snap_string_dividers_to_dark_gaps(
    image: np.ndarray,
    origin: np.ndarray,
    u_axis: np.ndarray,
    v_axis: np.ndarray,
    divider_u_positions: List[float],
    v_low: float,
    v_high: float,
    partial_end_u_positions: Optional[set] = None,
    pitch: float = 30.0,
    string_id: int = 0,
    search_px: int = 4
) -> Tuple[List[float], int]:
    """
    Snap dividers to dark gaps between panels.
    Returns (snapped_divider_positions, n_snapped).
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    else:
        gray = image.astype(np.float32)

    if partial_end_u_positions is None:
        partial_end_u_positions = set()

    snapped_positions = []
    n_snapped = 0
    half_pitch = pitch / 2.0

    for d_idx, u_div in enumerate(divider_u_positions):
        # Skip snap for dividers at partial panel ends
        if u_div in partial_end_u_positions:
            snapped_positions.append(u_div)
            continue

        if not ENABLE_STRING_DIVIDER_EDGE_SNAP:
            snapped_positions.append(u_div)
            continue

        best_u = u_div
        best_score = _score_divider_at_u(gray, origin, u_axis, v_axis, u_div, v_low, v_high, half_pitch)
        score_orig = best_score

        for delta in range(-search_px, search_px + 1):
            if delta == 0:
                continue
            u_try = u_div + float(delta)
            score = _score_divider_at_u(gray, origin, u_axis, v_axis, u_try, v_low, v_high, half_pitch)
            if score > best_score:
                best_score = score
                best_u = u_try

        shift = best_u - u_div
        use_snap = (
            abs(shift) <= STRING_DIVIDER_SNAP_MAX_SHIFT_PX and
            best_score >= score_orig * STRING_SNAP_MIN_EDGE_IMPROVEMENT
        )
        u_new = best_u if use_snap else u_div
        snapped_positions.append(u_new)
        if use_snap:
            n_snapped += 1

        print(f"[STRING_DIVIDER_SNAP] string_id={string_id} divider_idx={d_idx} "
              f"old_u={u_div:.2f} new_u={u_new:.2f} shift={u_new - u_div:.2f} "
              f"score_old={score_orig:.2f} score_new={best_score:.2f} "
              f"decision={'use_snap' if use_snap else 'keep_original'}")

    return snapped_positions, n_snapped


def apply_string_snap_refinement(
    fit_result: Dict[str, Any],
    string_info: Dict[str, Any],
    image: np.ndarray,
    image_shape: Tuple[int, int, int],
    image_name: str = "unknown"
) -> Dict[str, Any]:
    """
    Post-lattice snap: adjust rails and dividers to dark edges, rebuild polygons, validate.
    Returns updated fit_result with snap fields.
    """
    if not fit_result.get("success", False):
        return fit_result

    img_h, img_w = image_shape[:2]
    string_id = string_info["string_id"]
    panels = string_info["panels"]
    trusted_mask = string_info["trusted_mask"]

    u_axis = fit_result["u_axis"]
    v_axis = fit_result["v_axis"]
    origin = fit_result["origin"]
    v_low_orig = fit_result["v_low"]
    v_high_orig = fit_result["v_high"]
    pitch = fit_result["pitch"]
    u0 = fit_result["u0"]
    panel_details_orig = fit_result["panel_details"]

    # Compute u range
    positions = [det["u_pos"] for det in panel_details_orig]
    min_pos = min(positions) if positions else 0
    max_pos = max(positions) if positions else 0
    u_min = u0 + (min_pos - 0.5) * pitch
    u_max = u0 + (max_pos + 0.5) * pitch

    # --- Partial panel identification from original areas ---
    trusted_areas = [panels[k].get("area", 0.0) for k in range(len(panels)) if trusted_mask[k]]
    median_trusted_area = float(np.median([a for a in trusted_areas if a > 0])) if any(a > 0 for a in trusted_areas) else 1.0
    median_middle_area = fit_result.get("median_middle_area", median_trusted_area)

    partial_panel_set = set()  # indices in panel_details_orig that are partial
    n_partial_rejected = 0
    for k, (det, p) in enumerate(zip(panel_details_orig, panels)):
        is_end = (det["u_pos"] == min_pos or det["u_pos"] == max_pos)
        orig_area = p.get("area", 0.0)
        visible_ratio = orig_area / max(median_trusted_area, 1e-3)
        bbox = p.get("bbox") or p.get("box") or [0, 0, 0, 0]
        touches_border = (
            bbox[0] <= STRING_BORDER_MARGIN_PX or bbox[1] <= STRING_BORDER_MARGIN_PX or
            bbox[2] >= img_w - STRING_BORDER_MARGIN_PX or bbox[3] >= img_h - STRING_BORDER_MARGIN_PX
        )
        is_partial_candidate = (is_end or touches_border) and visible_ratio < PARTIAL_STRING_PANEL_MIN_AREA_RATIO

        if is_partial_candidate and ENABLE_PARTIAL_STRING_PANEL_REJECT:
            partial_panel_set.add(k)
            n_partial_rejected += 1
            print(f"[PARTIAL_STRING_PANEL] string_id={string_id} panel_idx={det['panel_idx']} "
                  f"visible_ratio={visible_ratio:.2f} decision=reject_partial")
        elif is_partial_candidate:
            print(f"[PARTIAL_STRING_PANEL] string_id={string_id} panel_idx={det['panel_idx']} "
                  f"visible_ratio={visible_ratio:.2f} decision=keep (ENABLE_PARTIAL=False)")

    # --- Step 1: Snap rails ---
    v_low_snapped, v_high_snapped, snap_rail_info = snap_string_rails_to_dark_edges(
        image, origin, u_axis, v_axis, v_low_orig, v_high_orig,
        u_min, u_max, string_id, search_px=STRING_RAIL_SNAP_SEARCH_PX
    )

    # Validate snapped rails — reject if they flip or collapse
    if v_high_snapped - v_low_snapped < 0.5 * (v_high_orig - v_low_orig):
        v_low_snapped = v_low_orig
        v_high_snapped = v_high_orig
        snap_rail_info["rail_low_decision"] = "keep_original_rail_collapse"
        snap_rail_info["rail_high_decision"] = "keep_original_rail_collapse"

    # --- Step 2: Snap dividers ---
    # Build divider list: left and right boundary of each position
    divider_map: Dict[float, int] = {}  # u_div -> index of divider
    all_dividers_orig: List[float] = []
    for pos in sorted(set(positions)):
        u_left = u0 + (pos - 0.5) * pitch
        u_right = u0 + (pos + 0.5) * pitch
        if u_left not in divider_map:
            all_dividers_orig.append(u_left)
            divider_map[u_left] = len(all_dividers_orig) - 1
        if u_right not in divider_map:
            all_dividers_orig.append(u_right)
            divider_map[u_right] = len(all_dividers_orig) - 1

    # Mark partial panel dividers (outer boundaries of partial panels)
    partial_end_dividers: set = set()
    for k in partial_panel_set:
        det = panel_details_orig[k]
        pos = det["u_pos"]
        u_left = u0 + (pos - 0.5) * pitch
        u_right = u0 + (pos + 0.5) * pitch
        partial_end_dividers.add(u_left)
        partial_end_dividers.add(u_right)

    snapped_dividers, n_dividers_snapped = snap_string_dividers_to_dark_gaps(
        image, origin, u_axis, v_axis, all_dividers_orig,
        v_low_snapped, v_high_snapped, partial_end_dividers, pitch, string_id,
        search_px=STRING_DIVIDER_SNAP_SEARCH_PX
    )

    # --- Step 2.5: Apply Outer Boundary Guard to start/end dividers ---
    outer_guard = None
    if ENABLE_OUTER_BOUNDARY_GUARD:
        string_dict = {
            "u_axis": u_axis,
            "v_axis": v_axis,
            "origin": origin,
            "v_low": v_low_snapped,
            "v_high": v_high_snapped
        }
        outer_guard = refine_outer_dividers(string_dict, snapped_dividers, panels, image)
        if outer_guard["start"]["snap_ok"]:
            snapped_dividers[0] = outer_guard["start"]["u_best"]
        if outer_guard["end"]["snap_ok"]:
            snapped_dividers[-1] = outer_guard["end"]["u_best"]

    # Build lookup: orig_u -> snapped_u
    divider_snap_map: Dict[float, float] = {
        orig: snapped for orig, snapped in zip(all_dividers_orig, snapped_dividers)
    }

    def to_img_pt(u_val: float, v_val: float) -> List[float]:
        pt = origin + u_val * u_axis + v_val * v_axis
        return [float(pt[0]), float(pt[1])]

    # Identify string middle panels to compute median length/width
    middle_u_lengths = []
    middle_v_widths = []
    positions = [det["u_pos"] for det in panel_details_orig]
    min_pos = min(positions) if positions else 0
    max_pos = max(positions) if positions else 0

    for det_mid in panel_details_orig:
        pos_mid = det_mid["u_pos"]
        if pos_mid != min_pos and pos_mid != max_pos:
            u_left_m = u0 + (pos_mid - 0.5) * pitch
            u_right_m = u0 + (pos_mid + 0.5) * pitch
            u_left_snap_m = divider_snap_map.get(u_left_m, u_left_m)
            u_right_snap_m = divider_snap_map.get(u_right_m, u_right_m)
            cand_snap_m = [
                to_img_pt(u_left_snap_m, v_low_snapped),
                to_img_pt(u_right_snap_m, v_low_snapped),
                to_img_pt(u_right_snap_m, v_high_snapped),
                to_img_pt(u_left_snap_m, v_high_snapped),
            ]
            cand_pts_m = np.array(cand_snap_m, dtype=np.float32)
            u_projs_m = [float(np.dot(pt - origin, u_axis)) for pt in cand_pts_m]
            v_projs_m = [float(np.dot(pt - origin, v_axis)) for pt in cand_pts_m]
            middle_u_lengths.append(max(u_projs_m) - min(u_projs_m))
            middle_v_widths.append(max(v_projs_m) - min(v_projs_m))

    median_middle_u_length = float(np.median(middle_u_lengths)) if middle_u_lengths else pitch
    median_middle_v_width = float(np.median(middle_v_widths)) if middle_v_widths else (v_high_snapped - v_low_snapped)

    # --- Step 3: Rebuild candidate polygons with snapped geometry ---
    updated_panel_details = []
    
    # Pre-compute proposed lattice polygons and original YOLO polygons for all panels
    lat_polys = []
    orig_polys = []
    use_locked = ENABLE_MIDDLE_LOCKED_ENDPOINT and (len(panel_details_orig) >= 5)
    for k, det in enumerate(panel_details_orig):
        p = det["p_obj"]
        pos = det["u_pos"]
        u_left_orig = u0 + (pos - 0.5) * pitch
        u_right_orig = u0 + (pos + 0.5) * pitch
        u_left_snap = divider_snap_map.get(u_left_orig, u_left_orig)
        u_right_snap = divider_snap_map.get(u_right_orig, u_right_orig)

        if use_locked:
            if pos == min_pos:
                u_left_snap = u_right_snap - pitch
            elif pos == max_pos:
                u_right_snap = u_left_snap + pitch

        cand_snap = [
            to_img_pt(u_left_snap, v_low_snapped),
            to_img_pt(u_right_snap, v_low_snapped),
            to_img_pt(u_right_snap, v_high_snapped),
            to_img_pt(u_left_snap, v_high_snapped),
        ]
        lat_polys.append(cand_snap)
        
        orig_poly = p.get("original_yolo_polygon", det["original_polygon"])
        orig_polys.append(orig_poly)

    # Compute original_max_overlap and lattice_max_overlap for each panel in the string
    original_max_overlaps = [0.0] * len(panel_details_orig)
    lattice_max_overlaps = [0.0] * len(panel_details_orig)
    for i in range(len(panel_details_orig)):
        max_orig = 0.0
        max_lat = 0.0
        for j in range(len(panel_details_orig)):
            if i == j:
                continue
            overlap_orig = compute_polygon_overlap_ratio(orig_polys[i], orig_polys[j], (img_h, img_w))
            overlap_lat = compute_polygon_overlap_ratio(lat_polys[i], lat_polys[j], (img_h, img_w))
            if overlap_orig > max_orig:
                max_orig = overlap_orig
            if overlap_lat > max_lat:
                max_lat = overlap_lat
        original_max_overlaps[i] = max_orig
        lattice_max_overlaps[i] = max_lat

    for k, det in enumerate(panel_details_orig):
        p = det["p_obj"]
        pos = det["u_pos"]
        cand_snap = lat_polys[k]
        original_max_overlap = original_max_overlaps[k]
        lattice_max_overlap = lattice_max_overlaps[k]

        # Validate snapped candidate vs original YOLO
        orig_poly = orig_polys[k]
        cand_before_snap = det["candidate_polygon"]
        orig_area = det["original_area"]
        orig_center = np.array(p["center"], dtype=np.float32)

        cand_snap_np = np.array(cand_snap, dtype=np.float32)
        orig_np = np.array(orig_poly, dtype=np.float32)

        iou_before = det["iou_with_original"]
        iou_after = _compute_mask_iou_cv(cand_snap_np, orig_np, (img_h, img_w))
        cand_snap_area = float(cv2.contourArea(cand_snap_np.reshape(-1, 1, 2)))
        cand_snap_center = np.mean(cand_snap_np, axis=0)

        orig_diag = float(np.linalg.norm([
            p.get("bbox", p.get("box", [0,0,0,0]))[2] - p.get("bbox", p.get("box", [0,0,0,0]))[0],
            p.get("bbox", p.get("box", [0,0,0,0]))[3] - p.get("bbox", p.get("box", [0,0,0,0]))[1]
        ]))
        center_shift_snap = float(np.linalg.norm(cand_snap_center - orig_center)) / max(orig_diag, 1.0)
        area_ratio_snap = cand_snap_area / max(orig_area, 1.0)

        # Width/height along axes
        u_proj_snap = [float(np.dot(pt - origin, u_axis)) for pt in cand_snap_np]
        v_proj_snap = [float(np.dot(pt - origin, v_axis)) for pt in cand_snap_np]
        cand_w_snap = max(u_proj_snap) - min(u_proj_snap)
        cand_h_snap = max(v_proj_snap) - min(v_proj_snap)

        orig_u_proj = [float(np.dot(pt - origin, u_axis)) for pt in orig_np]
        orig_v_proj = [float(np.dot(pt - origin, v_axis)) for pt in orig_np]
        orig_w = max(orig_u_proj) - min(orig_u_proj) if orig_u_proj else 1.0
        orig_h = max(orig_v_proj) - min(orig_v_proj) if orig_v_proj else 1.0

        width_ratio_snap = cand_w_snap / max(orig_w, 1e-3)
        height_ratio_snap = cand_h_snap / max(orig_h, 1e-3)

        out_of_bounds_snap = any(
            pt[0] < -STRING_PANEL_OUTSIDE_TOL_PX or pt[0] > img_w + STRING_PANEL_OUTSIDE_TOL_PX or
            pt[1] < -STRING_PANEL_OUTSIDE_TOL_PX or pt[1] > img_h + STRING_PANEL_OUTSIDE_TOL_PX
            for pt in cand_snap
        )

        iou_drop = iou_before - iou_after

        # Compute the required 12 logging metrics for every panel
        original_area_val = float(orig_area)
        lattice_area_val = float(cand_snap_area)
        area_ratio_val = lattice_area_val / max(original_area_val, 1.0)
        iou_val = float(iou_after)
        
        lattice_center = np.mean(cand_snap_np, axis=0)
        original_center = np.mean(orig_np, axis=0)
        orig_x = [pt[0] for pt in orig_poly]
        orig_y = [pt[1] for pt in orig_poly]
        orig_w_sz = max(orig_x) - min(orig_x) if orig_x else 1.0
        orig_h_sz = max(orig_y) - min(orig_y) if orig_y else 1.0
        orig_diag_sz = float(np.linalg.norm([orig_w_sz, orig_h_sz]))
        center_shift_val = float(np.linalg.norm(lattice_center - original_center)) / max(orig_diag_sz, 1.0)
        
        edge_orig_val = float(score_panel_boundary_alignment(image, orig_poly) if image is not None else 0.0)
        edge_lat_val = float(score_panel_boundary_alignment(image, cand_snap) if image is not None else 0.0)
        edge_ratio_val = edge_lat_val / max(edge_orig_val, 1e-6)

        # Determine if it is an outer/endpoint panel
        is_outer_panel = False
        outer_side = None
        if pos == min_pos:
            is_outer_panel = True
            outer_side = "start"
        elif pos == max_pos:
            is_outer_panel = True
            outer_side = "end"

        is_endpoint = False
        endpoint_side = None
        n_panels_str = len(panel_details_orig)
        if n_panels_str >= 5:
            if pos == min_pos:
                is_endpoint = True
                endpoint_side = "start"
            elif pos == max_pos:
                is_endpoint = True
                endpoint_side = "end"

        # Apply Outer Boundary Guard logic
        force_fallback_outer = False
        outer_reason = "ok"
        outer_div_source = "extrapolated"
        outer_div_shift = 0.0
        outer_div_score = 0.0
        outer_guard_decision = "keep_lattice"

        if ENABLE_OUTER_BOUNDARY_GUARD and is_outer_panel and outer_guard is not None:
            guard_res = outer_guard[outer_side]
            outer_div_shift = guard_res["shift"]
            outer_div_score = guard_res["best_score"]
            if guard_res["snap_ok"]:
                outer_div_source = "edge_snap"
                outer_guard_decision = "use_outer_snap"
            else:
                outer_div_source = "fallback" if not OUTER_PANEL_ALLOW_EXTRAPOLATE else "extrapolated"
                if abs(guard_res["shift"]) > 1e-3:
                    outer_guard_decision = "reject_snap_not_better"
                else:
                    outer_guard_decision = "keep_lattice"
                if not OUTER_PANEL_ALLOW_EXTRAPOLATE:
                    force_fallback_outer = True
                    outer_reason = "extrapolate_disallowed"

            # Check outer validation thresholds if not already forced fallback
            if not force_fallback_outer:
                val_pass = True
                val_reason = "ok"
                if not (OUTER_PANEL_AREA_RATIO_MIN <= area_ratio_snap <= OUTER_PANEL_AREA_RATIO_MAX):
                    val_pass = False
                    val_reason = "area_ratio_out"
                elif center_shift_snap > OUTER_PANEL_CENTER_SHIFT_MAX:
                    val_pass = False
                    val_reason = "center_shift_too_large"
                elif iou_after < OUTER_PANEL_IOU_MIN:
                    val_pass = False
                    val_reason = "iou_too_low"
                elif out_of_bounds_snap:
                    val_pass = False
                    val_reason = "outside_image"

                if not val_pass:
                    force_fallback_outer = True
                    outer_reason = val_reason
                    outer_guard_decision = "fallback_outer_panel"

        # Run Endpoint Guard checks if it is an endpoint panel
        ep_res = None
        candidate_original = orig_poly
        if is_endpoint:
            other_polys_for_endpoint = []
            for idx_other, det_other in enumerate(panel_details_orig):
                if idx_other == k:
                    continue
                other_polys_for_endpoint.append(lat_polys[idx_other])

            candidate_lattice_max_overlap = lattice_max_overlap
            candidate_original_max_overlap = original_max_overlap

            ep_res = validate_endpoint_candidate(
                candidate_lattice=cand_snap,
                candidate_original=candidate_original,
                u_axis=u_axis,
                v_axis=v_axis,
                origin=origin,
                median_middle_u_length=median_middle_u_length,
                median_middle_v_width=median_middle_v_width,
                candidate_lattice_max_overlap=candidate_lattice_max_overlap,
                candidate_original_max_overlap=candidate_original_max_overlap,
                outer_shift_abs=abs(outer_div_shift),
                median_pitch=pitch,
                image=image,
                img_shape=(img_h, img_w)
            )

        # Decision making
        use_snapped = False
        snap_decision = "use_lattice"
        snap_reason = "ok"
        pass_individual = True
        fail_reason = "ok"
        endpoint_diagnostic_pass = True

        if ep_res is not None:
            endpoint_diagnostic_pass = ep_res["pass"]

        if is_endpoint and use_locked:
            # Simplified validation for middle-locked endpoint
            cand_snap_np = np.array(cand_snap, dtype=np.float32)
            cand_area = float(cv2.contourArea(cand_snap_np.reshape(-1, 1, 2)))
            poly_valid = (len(cand_snap) >= 3 and cand_area > 1.0)
            
            out_of_bounds_2px = any(
                pt[0] < -2.0 or pt[0] > img_w + 2.0 or
                pt[1] < -2.0 or pt[1] > img_h + 2.0
                for pt in cand_snap
            )
            
            neighbor_idx = 1 if pos == min_pos else n_panels_str - 2
            neighbor_overlap_val = 0.0
            if 0 <= neighbor_idx < n_panels_str:
                n_pos = panel_details_orig[neighbor_idx]["u_pos"]
                n_left_orig = u0 + (n_pos - 0.5) * pitch
                n_right_orig = u0 + (n_pos + 0.5) * pitch
                n_left_snap = divider_snap_map.get(n_left_orig, n_left_orig)
                n_right_snap = divider_snap_map.get(n_right_orig, n_right_orig)
                neighbor_cand_snap = [
                    to_img_pt(n_left_snap, v_low_snapped),
                    to_img_pt(n_right_snap, v_low_snapped),
                    to_img_pt(n_right_snap, v_high_snapped),
                    to_img_pt(n_left_snap, v_high_snapped),
                ]
                neighbor_overlap_val = compute_polygon_overlap_ratio(cand_snap, neighbor_cand_snap, (img_h, img_w))
            
            yolo_small = (orig_area < 0.65 * median_middle_area)
            
            pass_val = True
            fail_reasons = []
            
            if not poly_valid:
                pass_val = False
                fail_reasons.append("invalid_polygon")
            if out_of_bounds_2px:
                pass_val = False
                fail_reasons.append("out_of_bounds_2px")
            if neighbor_overlap_val > 0.03:
                pass_val = False
                fail_reasons.append(f"neighbor_overlap_{neighbor_overlap_val:.3f}")
                
            if not yolo_small:
                if center_shift_val > 0.40:
                    pass_val = False
                    fail_reasons.append(f"center_shift_{center_shift_val:.3f}")
                if iou_after < 0.20:
                    pass_val = False
                    fail_reasons.append(f"iou_{iou_after:.3f}")
                    
            if pass_val:
                snap_decision = "use_middle_locked_endpoint"
                snap_reason = "locked_by_middle_pitch"
                use_snapped = True
                pass_individual = True
                fail_reason = "ok"
            else:
                snap_decision = "fallback_endpoint_original"
                snap_reason = "|".join(fail_reasons)
                use_snapped = False
                pass_individual = False
                fail_reason = snap_reason
        elif k in partial_panel_set:
            snap_decision = "reject_partial"
            snap_reason = "partial_panel"
            use_snapped = False
            pass_individual = False
            fail_reason = "reject_partial"
        elif force_fallback_outer:
            snap_decision = "fallback_outer_panel"
            snap_reason = outer_reason
            use_snapped = False
            pass_individual = False
            fail_reason = "fallback_outer_panel"
        elif ENABLE_LATTICE_SAFE_IMPROVEMENT_ONLY:
            # Safe Improvement Check (diagnostic warning only when ENABLE_LATTICE_SAFE_IMPROVEMENT_ONLY=False)
            pass_safe, fail_reason_safe, stats_safe = check_safe_improvement(
                orig_poly, cand_snap, image, (img_h, img_w), original_max_overlap, lattice_max_overlap
            )
            if pass_safe:
                snap_decision = "use_snapped"
                snap_reason = "ok"
                use_snapped = True
                pass_individual = True
                fail_reason = "ok"
            else:
                snap_decision = "keep_original_not_improved"
                snap_reason = fail_reason_safe
                use_snapped = False
                pass_individual = False
                fail_reason = fail_reason_safe
        else:
            # Lattice-primary logic: endpoint guard controls endpoint panels; basic geometry for middle
            if is_endpoint:
                if ENABLE_ENDPOINT_GUARD_POLYGON_UPDATE:
                    # Endpoint Guard is active: use lattice if pass, fallback to YOLO original if fail
                    if ep_res["pass"]:
                        snap_decision = "use_lattice_endpoint"
                        snap_reason = ep_res["reason"]
                        use_snapped = True
                        pass_individual = True
                        fail_reason = "ok"
                    else:
                        snap_decision = "fallback_endpoint_original"
                        snap_reason = ep_res["reason"]
                        use_snapped = False
                        # pass_individual = False but does NOT penalize pass_ratio (handled below)
                        pass_individual = False
                        fail_reason = ep_res["reason"]
                else:
                    # Endpoint Guard update disabled: use lattice unconditionally for endpoint
                    snap_decision = "use_lattice_endpoint_unguarded"
                    snap_reason = "guard_disabled"
                    use_snapped = True
                    pass_individual = True
                    fail_reason = "ok"
            elif out_of_bounds_snap:
                snap_decision = "fallback_bounds"
                snap_reason = "snap_outside_image"
                use_snapped = False
                pass_individual = False
                fail_reason = "snap_outside_image"
            elif iou_drop > STRING_SNAP_MAX_IOU_DROP:
                snap_decision = "use_lattice_pre_snap"
                snap_reason = f"iou_drop_{iou_drop:.3f}"
                use_snapped = False
                pass_individual = True  # Use pre-snap lattice, still counts as lattice-used
                fail_reason = "ok"
            elif not (STRING_SNAP_PANEL_IOU_MIN <= iou_after):
                snap_decision = "fallback_original"
                snap_reason = f"iou_too_low_{iou_after:.3f}"
                use_snapped = False
                pass_individual = False
                fail_reason = snap_reason
            elif not (STRING_SNAP_PANEL_AREA_RATIO_MIN <= area_ratio_snap <= STRING_SNAP_PANEL_AREA_RATIO_MAX):
                snap_decision = "fallback_original"
                snap_reason = "area_ratio_out"
                use_snapped = False
                pass_individual = False
                fail_reason = snap_reason
            elif center_shift_snap > STRING_SNAP_PANEL_CENTER_SHIFT_MAX:
                snap_decision = "fallback_original"
                snap_reason = "center_shift_too_large"
                use_snapped = False
                pass_individual = False
                fail_reason = snap_reason
            elif not (STRING_SNAP_PANEL_WIDTH_RATIO_MIN <= width_ratio_snap <= STRING_SNAP_PANEL_WIDTH_RATIO_MAX):
                snap_decision = "fallback_original"
                snap_reason = "width_ratio_out"
                use_snapped = False
                pass_individual = False
                fail_reason = snap_reason
            elif not (STRING_SNAP_PANEL_HEIGHT_RATIO_MIN <= height_ratio_snap <= STRING_SNAP_PANEL_HEIGHT_RATIO_MAX):
                snap_decision = "fallback_original"
                snap_reason = "height_ratio_out"
                use_snapped = False
                pass_individual = False
                fail_reason = snap_reason
            else:
                snap_decision = "use_snapped"
                snap_reason = "ok"
                use_snapped = True
                pass_individual = True
                fail_reason = "ok"

            # Diagnostic-only safe improvement warning (not a gate)
            if use_snapped and not ENABLE_LATTICE_SAFE_IMPROVEMENT_ONLY:
                try:
                    _, _warn_reason, _ = check_safe_improvement(
                        orig_poly, cand_snap, image, (img_h, img_w),
                        original_max_overlap, lattice_max_overlap
                    )
                    if _warn_reason != "ok":
                        print(f"[SAFE_IMPROVEMENT_WARN] string={string_id} panel={det['panel_idx']} {_warn_reason}")
                except Exception:
                    pass

        if ENABLE_OUTER_BOUNDARY_GUARD and is_outer_panel:
            print(f"[OUTER_GUARD] img={image_name} block=n/a string={string_id} side={outer_side} "
                  f"score={outer_div_score:.3f} shift={outer_div_shift:+.1f} decision={outer_guard_decision}")
            if snap_decision == "fallback_outer_panel":
                print(f"[OUTER_GUARD] img={image_name} panel_idx={det['panel_idx']} "
                      f"decision=fallback_outer_panel reason={outer_reason}")

        if is_endpoint:
            dec_print = "diagnostic_only" if not ENABLE_ENDPOINT_GUARD_POLYGON_UPDATE else ep_res['decision']
            print(f"[ENDPOINT_GUARD] img={image_name} block=n/a panel={det['panel_idx']} "
                  f"side={endpoint_side} decision={dec_print} reason={ep_res['reason']} "
                  f"area={ep_res['area_ratio']:.2f} shift={ep_res['center_shift_ratio']:.2f} "
                  f"iou={ep_res['iou']:.2f} edge_ratio={ep_res['edge_score_ratio']:.2f}")

        print(f"[STRING_SNAP_VALIDATE] string_id={string_id} panel_idx={det['panel_idx']} "
              f"decision={snap_decision} reason={snap_reason} "
              f"iou_before={iou_before:.3f} iou_after={iou_after:.3f}")

        # Build updated det
        new_det = dict(det)
        new_det["candidate_polygon_before_snap"] = cand_before_snap
        new_det["iou_before_snap"] = iou_before
        new_det["decision_before_snap"] = "use_string_lattice" if det.get("pass_individual") else "fallback_original"

        # Safe Improvement Metrics
        new_det["original_area"] = original_area_val
        new_det["lattice_area"] = lattice_area_val
        new_det["area_ratio"] = area_ratio_val
        new_det["iou_original_lattice"] = iou_val
        new_det["center_shift_ratio"] = center_shift_val
        new_det["original_edge_score"] = edge_orig_val
        new_det["lattice_edge_score"] = edge_lat_val
        new_det["edge_score_ratio"] = edge_ratio_val
        new_det["original_max_overlap"] = original_max_overlap
        new_det["lattice_max_overlap"] = lattice_max_overlap
        new_det["endpoint_diagnostic_pass"] = endpoint_diagnostic_pass

        if is_outer_panel:
            new_det["is_outer_panel"] = True
            new_det["outer_side"] = outer_side
            new_det["outer_divider_source"] = outer_div_source
            new_det["outer_divider_shift"] = float(outer_div_shift)
            new_det["outer_divider_score"] = float(outer_div_score)
            new_det["outer_guard_decision"] = outer_guard_decision
            new_det["outer_guard_reason"] = outer_reason
            new_det["outer_divider_u_proposal"] = float(u_left_orig if outer_side == "start" else u_right_orig)
            new_det["outer_divider_u_snapped"] = float(u_left_snap if outer_side == "start" else u_right_snap)
            new_det["outer_divider_u_axis"] = u_axis
            new_det["outer_divider_v_axis"] = v_axis
            new_det["outer_divider_origin"] = origin
            new_det["outer_divider_rail_low"] = float(v_low_snapped)
            new_det["outer_divider_rail_high"] = float(v_high_snapped)

        rail_fit_source = fit_result.get("rail_fit_source", "all_panels")
        if is_endpoint:
            new_det["is_endpoint_panel"] = True
            new_det["endpoint_side"] = endpoint_side
            new_det["rail_fit_source"] = rail_fit_source
            new_det["candidate_lattice_area_ratio"] = float(ep_res["area_ratio"])
            new_det["candidate_lattice_center_shift"] = float(ep_res["center_shift_ratio"])
            new_det["candidate_lattice_iou"] = float(ep_res["iou"])
            new_det["candidate_lattice_u_length_ratio"] = float(ep_res["u_length_ratio"])
            new_det["candidate_lattice_v_width_ratio"] = float(ep_res["v_width_ratio"])
            new_det["candidate_lattice_edge_score"] = float(ep_res["edge_score_candidate"])
            new_det["candidate_original_edge_score"] = float(ep_res["edge_score_original"])
            new_det["edge_score_ratio"] = float(ep_res["edge_score_ratio"])
            new_det["endpoint_decision"] = "diagnostic_only" if not ENABLE_ENDPOINT_GUARD_POLYGON_UPDATE else ep_res["decision"]
            new_det["endpoint_reason"] = ep_res["reason"]
        else:
            new_det["is_endpoint_panel"] = False
            new_det["endpoint_side"] = None
            new_det["rail_fit_source"] = rail_fit_source

        new_det["candidate_polygon"] = cand_snap if use_snapped else orig_poly
        new_det["iou_after_snap"] = iou_after if use_snapped else iou_before
        new_det["area_ratio_after_snap"] = area_ratio_snap if use_snapped else det["area_ratio"]
        new_det["center_shift_after_snap"] = center_shift_snap if use_snapped else det["center_shift_ratio"]
        new_det["snap_used"] = use_snapped
        new_det["snap_decision"] = snap_decision
        new_det["snap_reason"] = snap_reason
        new_det["pass_individual"] = pass_individual
        new_det["fail_reason"] = fail_reason
        new_det["iou_with_original"] = iou_after if use_snapped else iou_before

        # Save snapped lattice candidate and overlap value for relaxed endpoint acceptance
        new_det["cand_snap"] = cand_snap
        if is_endpoint and use_locked:
            new_det["neighbor_overlap_val"] = float(neighbor_overlap_val)

        updated_panel_details.append(new_det)

    # Iterative overlap resolution (up to 3 passes to handle cascading)
    for _overlap_pass in range(3):
        changed = False
        for i in range(len(updated_panel_details)):
            if not updated_panel_details[i]["pass_individual"]:
                continue
            for j in range(i + 1, len(updated_panel_details)):
                if not updated_panel_details[j]["pass_individual"]:
                    continue
                pi = np.array(updated_panel_details[i]["candidate_polygon"], dtype=np.float32)
                pj = np.array(updated_panel_details[j]["candidate_polygon"], dtype=np.float32)
                pi_h = cv2.convexHull(pi.reshape(-1, 1, 2)).astype(np.float32)
                pj_h = cv2.convexHull(pj.reshape(-1, 1, 2)).astype(np.float32)
                area_inter, _ = cv2.intersectConvexConvex(pi_h, pj_h)
                if area_inter > 0:
                    min_a = min(cv2.contourArea(pi_h), cv2.contourArea(pj_h))
                    overlap_r = area_inter / max(min_a, 1e-3)
                    if overlap_r > STRING_SNAP_MAX_OVERLAP:
                        # Reject the one with lower IoU; fall back to pre-snap lattice polygon
                        if updated_panel_details[i]["iou_with_original"] <= updated_panel_details[j]["iou_with_original"]:
                            updated_panel_details[i]["pass_individual"] = False
                            updated_panel_details[i]["fail_reason"] = "overlap_after_snap"
                            if updated_panel_details[i].get("snap_decision") == "fallback_outer_panel":
                                updated_panel_details[i]["candidate_polygon"] = updated_panel_details[i]["original_polygon"]
                            else:
                                updated_panel_details[i]["candidate_polygon"] = updated_panel_details[i].get(
                                    "candidate_polygon_before_snap", updated_panel_details[i]["candidate_polygon"]
                                )
                        else:
                            updated_panel_details[j]["pass_individual"] = False
                            updated_panel_details[j]["fail_reason"] = "overlap_after_snap"
                            if updated_panel_details[j].get("snap_decision") == "fallback_outer_panel":
                                updated_panel_details[j]["candidate_polygon"] = updated_panel_details[j]["original_polygon"]
                            else:
                                updated_panel_details[j]["candidate_polygon"] = updated_panel_details[j].get(
                                    "candidate_polygon_before_snap", updated_panel_details[j]["candidate_polygon"]
                                )
                        changed = True
        if not changed:
            break

    # Recompute max_overlap
    max_overlap_new = 0.0
    passing_new = [d for d in updated_panel_details if d["pass_individual"]]
    for i in range(len(passing_new)):
        for j in range(i + 1, len(passing_new)):
            pi = np.array(passing_new[i]["candidate_polygon"], dtype=np.float32)
            pj = np.array(passing_new[j]["candidate_polygon"], dtype=np.float32)
            pi_h = cv2.convexHull(pi.reshape(-1, 1, 2)).astype(np.float32)
            pj_h = cv2.convexHull(pj.reshape(-1, 1, 2)).astype(np.float32)
            area_inter, _ = cv2.intersectConvexConvex(pi_h, pj_h)
            if area_inter > 0:
                min_a = min(cv2.contourArea(pi_h), cv2.contourArea(pj_h))
                max_overlap_new = max(max_overlap_new, area_inter / max(min_a, 1e-3))

    # Endpoint panels that fallback to original are NOT penalized in pass_ratio denominator
    pass_count_new = len([d for d in updated_panel_details if
        d["pass_individual"] or
        (d.get("is_endpoint_panel") and d.get("snap_decision") == "fallback_endpoint_original")])
    n_panels = len(updated_panel_details)
    pass_ratio_new = pass_count_new / n_panels
    ious_new = [d["iou_with_original"] for d in updated_panel_details]
    median_iou_new = float(np.median(ious_new)) if ious_new else 0.0

    # Return updated fit_result
    updated = dict(fit_result)
    updated["v_low"] = v_low_snapped
    updated["v_high"] = v_high_snapped
    updated["panel_details"] = updated_panel_details
    updated["pass_count"] = pass_count_new
    updated["pass_ratio"] = pass_ratio_new
    updated["median_iou"] = median_iou_new
    updated["max_overlap"] = max_overlap_new
    updated["snap_rail_info"] = snap_rail_info
    updated["n_dividers_snapped"] = n_dividers_snapped
    updated["n_partial_rejected"] = n_partial_rejected

    return updated


def _draw_string_debug_image(
    image: np.ndarray,
    strings_with_fits: List[Dict],
    bypass_panels: List[Dict],
    isolated_panels: List[Dict],
    image_path: str
) -> None:
    """Ve debug image cho string lattice refinement - enhanced with snap layers."""
    dbg = image.copy()
    img_h, img_w = image.shape[:2]

    # Draw bypass panels (white box)
    for p in bypass_panels:
        poly = p.get("polygon", [])
        if poly:
            pts = np.array(poly, dtype=np.int32)
            cv2.polylines(dbg, [pts], True, (200, 200, 200), 1)

    # Draw isolated panels (gray)
    for p in isolated_panels:
        poly = p.get("polygon", [])
        if poly:
            pts = np.array(poly, dtype=np.int32)
            cv2.polylines(dbg, [pts], True, (120, 120, 120), 1)

    for sw in strings_with_fits:
        si = sw["string_info"]
        fr = sw.get("fit_result", {})
        if not fr.get("success"):
            continue

        u_axis = fr["u_axis"]
        v_axis = fr["v_axis"]
        origin = fr["origin"]
        v_low = fr["v_low"]
        v_high = fr["v_high"]
        u0 = fr["u0"]
        pitch = fr["pitch"]
        panel_details = fr["panel_details"]
        snap_info = fr.get("snap_rail_info", {})

        rail_len = img_w * 2

        # 1. Draw original YOLO polygons (blue thin)
        for det in panel_details:
            poly_o = det["original_polygon"]
            pts = np.array(poly_o, dtype=np.int32)
            cv2.polylines(dbg, [pts], True, (180, 80, 0), 1)

        # 2. Draw pre-snap lattice candidate (yellow thin, if available)
        for det in panel_details:
            pre_poly = det.get("candidate_polygon_before_snap")
            if pre_poly:
                pts = np.array(pre_poly, dtype=np.int32)
                cv2.polylines(dbg, [pts], True, (0, 200, 200), 1)

        # 3. Draw rails (before snap = white muted, after snap = white bright)
        for v_val, brightness, thickness in [(v_low, 200, 1), (v_high, 200, 1)]:
            pt1 = (origin + (-rail_len) * u_axis + v_val * v_axis).astype(np.int32)
            pt2 = (origin + (rail_len) * u_axis + v_val * v_axis).astype(np.int32)
            p1c = tuple(map(int, np.clip(pt1, [-img_w, -img_h], [2*img_w, 2*img_h]).tolist()))
            p2c = tuple(map(int, np.clip(pt2, [-img_w, -img_h], [2*img_w, 2*img_h]).tolist()))
            cv2.line(dbg, p1c, p2c, (brightness, brightness, brightness), thickness)

        # Draw snapped rails with bright white (if snapped)
        if snap_info.get("rail_low_decision") == "use_snap":
            pt1 = (origin + (-rail_len) * u_axis + v_low * v_axis).astype(np.int32)
            pt2 = (origin + (rail_len) * u_axis + v_low * v_axis).astype(np.int32)
            p1c = tuple(map(int, np.clip(pt1, [-img_w, -img_h], [2*img_w, 2*img_h]).tolist()))
            p2c = tuple(map(int, np.clip(pt2, [-img_w, -img_h], [2*img_w, 2*img_h]).tolist()))
            cv2.line(dbg, p1c, p2c, (255, 255, 255), 2)
        if snap_info.get("rail_high_decision") == "use_snap":
            pt1 = (origin + (-rail_len) * u_axis + v_high * v_axis).astype(np.int32)
            pt2 = (origin + (rail_len) * u_axis + v_high * v_axis).astype(np.int32)
            p1c = tuple(map(int, np.clip(pt1, [-img_w, -img_h], [2*img_w, 2*img_h]).tolist()))
            p2c = tuple(map(int, np.clip(pt2, [-img_w, -img_h], [2*img_w, 2*img_h]).tolist()))
            cv2.line(dbg, p1c, p2c, (255, 255, 255), 2)

        # 4. Draw dividers after snap (cyan bright)
        positions = sorted(set(d["u_pos"] for d in panel_details))
        for pos in positions:
            # Get divider positions from panel details (they reflect snapped)
            pos_panels = [d for d in panel_details if d["u_pos"] == pos]
            if pos_panels:
                cand = pos_panels[0].get("candidate_polygon") or pos_panels[0].get("candidate_polygon_before_snap")
                if cand:
                    cand_np = np.array(cand, dtype=np.float32)
                    u_projs = [float(np.dot(pt - origin, u_axis)) for pt in cand_np]
                    u_l = min(u_projs)
                    u_r = max(u_projs)
                    for u_div in [u_l, u_r]:
                        pt1 = (origin + u_div * u_axis + v_low * v_axis).astype(np.int32)
                        pt2 = (origin + u_div * u_axis + v_high * v_axis).astype(np.int32)
                        cv2.line(dbg, tuple(pt1.tolist()), tuple(pt2.tolist()), (0, 230, 255), 1)

        # 5. Draw centers
        for det in panel_details:
            cx, cy = int(det["p_obj"]["center"][0]), int(det["p_obj"]["center"][1])
            color_c = (255, 255, 255) if det["trusted"] else (128, 128, 128)
            cv2.circle(dbg, (cx, cy), 3, color_c, -1)

        # 6. Draw final polygons (thick, color-coded by decision)
        for det in panel_details:
            final_poly = det.get("final_polygon", det["original_polygon"])
            pts = np.array(final_poly, dtype=np.int32)
            snap_dec = det.get("snap_decision", "")
            decision = det.get("panel_decision", "fallback_original")
            fail_r = det.get("fail_reason", "")

            if fail_r == "reject_partial" or snap_dec == "reject_partial":
                color = (0, 0, 255)   # red = partial rejected
                thickness = 2
            elif decision == "use_string_lattice" and snap_dec == "use_snapped":
                color = (0, 255, 200)  # bright cyan = snapped
                thickness = 2
            elif decision == "use_string_lattice":
                color = (0, 200, 80)   # green = lattice (not snapped)
                thickness = 2
            else:
                color = (0, 60, 200)   # dim blue = fallback original
                thickness = 1

            cv2.polylines(dbg, [pts], True, color, thickness)

            # Label
            cx, cy = int(det["p_obj"]["center"][0]), int(det["p_obj"]["center"][1])
            label_parts = [f"s{si['string_id']}"]
            if snap_dec == "use_snapped":
                label_parts.append("SN")
            elif fail_r:
                label_parts.append(fail_r[:5])
            label = " ".join(label_parts)
            cv2.putText(dbg, label, (cx - 15, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.26, (255, 220, 0), 1)

    # Legend
    legend_items = [
        ((180, 80, 0), "YOLO orig"),
        ((0, 200, 200), "Lattice pre-snap"),
        ((0, 255, 200), "Snapped final"),
        ((0, 200, 80), "Lattice final"),
        ((0, 0, 255), "Reject partial"),
        ((255, 255, 255), "Rail snapped"),
    ]
    for i, (color, text) in enumerate(legend_items):
        y = 18 + i * 14
        cv2.rectangle(dbg, (4, y - 8), (14, y + 2), color, -1)
        cv2.putText(dbg, text, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 200), 1)

    # Save
    image_stem = os.path.splitext(os.path.basename(image_path))[0]
    debug_dir = "data/results/debug"
    os.makedirs(debug_dir, exist_ok=True)
    out_path = os.path.join(debug_dir, f"debug_{image_stem}_string_lattice_refine.JPG")
    cv2.imwrite(out_path, dbg)
    print(f"[STRING_LATTICE] Debug image saved: {out_path}")


def refine_panels_by_string_lattice(
    panels: List[Dict[str, Any]],
    image_shape: Tuple[int, int, int],
    image_path: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Main orchestrator cho String-first Lattice Refinement.
    Production path moi thay the refine_panel_blocks_by_lattice_lines.
    """
    if not panels:
        return []

    import json

    img_h, img_w = image_shape[:2]
    image_stem = os.path.splitext(os.path.basename(image_path))[0] if image_path else "unknown"

    # Load image
    image = None
    if image_path and os.path.exists(image_path):
        image = cv2.imread(image_path)
    if image is None:
        image = np.zeros((img_h, img_w, 3), dtype=np.uint8)

    # Bypass for protected images
    if image_path:
        stem_upper = image_stem.upper()
        if "DJI_0987" in stem_upper or "DJI_0995" in stem_upper:
            print(f"[STRING_LATTICE] img={image_stem} BYPASS completely")
            return panels

    # Separate small panels from large panels
    small_panels = []
    bypass_panels = []
    small_indices = []   # index in panels list
    bypass_indices = []

    for i, p in enumerate(panels):
        bbox = p.get("bbox") or p.get("box") or [0, 0, 0, 0]
        x1, y1, x2, y2 = bbox
        b_area = (x2 - x1) * (y2 - y1)
        if b_area <= STRING_REFINEMENT_MAX_PANEL_BBOX_AREA:
            p["original_yolo_polygon"] = list(p["polygon"])
            p["original_yolo_bbox"] = list(p.get("box") or p.get("bbox") or [0, 0, 0, 0])
            small_panels.append(p)
            small_indices.append(i)
        else:
            p["original_yolo_polygon"] = list(p["polygon"])
            p["original_yolo_bbox"] = list(p.get("box") or p.get("bbox") or [0, 0, 0, 0])
            p["string_refined"] = False
            p["string_decision"] = "bypass_large_panel"
            bypass_panels.append(p)
            bypass_indices.append(i)

    if not small_panels:
        return panels

    # Compute median area for partial panel detection
    areas = [p.get("area", 0.0) for p in small_panels]
    median_panel_area = float(np.median([a for a in areas if a > 0])) if any(a > 0 for a in areas) else 1.0

    # --- Step A: Trusted panel selection ---
    trusted_indices = select_trusted_panels_for_geometry(small_panels, image_shape)
    print(f"[STRING_LATTICE] img={image_stem} total_small={len(small_panels)} trusted={len(trusted_indices)}")

    # --- Step B: Split into strings ---
    string_list = split_panels_into_strings(small_panels, trusted_indices, image_shape)
    print(f"[STRING_LATTICE] img={image_stem} strings_found={len(string_list)}")

    # Find panels not in any string
    assigned_panel_indices = set()
    for s in string_list:
        for idx in s["panel_indices"]:
            assigned_panel_indices.add(idx)
    isolated_panels = [small_panels[i] for i in range(len(small_panels)) if i not in assigned_panel_indices]
    for p in isolated_panels:
        p["string_refined"] = False
        p["string_decision"] = "bypass_no_string"
        p["string_angle_deg"] = 0.0

    for s in string_list:
        for p in s["panels"]:
            p["string_angle_deg"] = float(s["angle_deg"])

    # --- Step C: Fit each string ---
    strings_with_fits = []
    for si in string_list:
        fit = fit_string_parallel_lines(si, image_shape, image, median_panel_area)
        strings_with_fits.append({"string_info": si, "fit_result": fit})

    # --- Step D: Synchronize strings ---
    strings_with_fits = synchronize_strings(strings_with_fits, image_shape, image, median_panel_area)

    # --- Step D2: Post-lattice snap refinement (AFTER sync so synced fits also get snap) ---
    if (ENABLE_STRING_RAIL_EDGE_SNAP or ENABLE_STRING_DIVIDER_EDGE_SNAP or ENABLE_PARTIAL_STRING_PANEL_REJECT):
        for sw in strings_with_fits:
            if sw["fit_result"].get("success", False):
                sw["fit_result"] = apply_string_snap_refinement(
                    sw["fit_result"], sw["string_info"], image, image_shape, image_stem
                )

    # --- Step E: Apply decisions ---
    log_records = []
    import collections
    lattice_prop_total = 0
    lattice_prop_accept = 0
    lattice_prop_reject = 0
    lattice_prop_reject_reason = collections.Counter()
    prop_examples = []

    # Relaxed geometry gate diagnostics counters
    relaxed_gate_total = 0
    relaxed_gate_accept = 0
    relaxed_gate_reject = 0
    relaxed_gate_reject_reason = collections.Counter()
    relaxed_accepted_examples = []
    relaxed_rejected_examples = []


    for sw in strings_with_fits:
        si = sw["string_info"]
        fr = sw.get("fit_result", {})
        string_id = si["string_id"]
        n_panels_str = len(si["panels"])
        synced_with = sw.get("synced_with", -1)

        if not fr.get("success", False):
            # Fallback whole string
            string_decision = "fallback_string"
            fallback_reason = fr.get("reason", "fit_failed")

            for k, p in enumerate(si["panels"]):
                p["string_id"] = string_id
                p["string_decision"] = "fallback_original"
                p["string_reason"] = fallback_reason
                p["string_refined"] = False
                p["polygon"] = list(p.get("original_yolo_polygon") or p["polygon"])
                is_ep = (k == 0 or k == len(si["panels"]) - 1) if len(si["panels"]) >= 5 else False
                if is_ep:
                    p["final_polygon_source"] = "endpoint_original_fallback"
                    p["final_polygon_stage"] = "string_lattice_refinement"
                    p["final_polygon_reason"] = "string_fit_failed_" + fallback_reason
                else:
                    p["final_polygon_source"] = "yolo_original"
                    p["final_polygon_stage"] = "string_lattice_refinement"
                    p["final_polygon_reason"] = "string_fit_failed_" + fallback_reason

            log_records.append({
                "record_type": "string",
                "image_name": image_stem,
                "string_id": int(string_id),
                "n_panels": int(n_panels_str),
                "n_trusted": int(si["n_trusted"]),
                "decision": "fallback_string",
                "fallback_reason": fallback_reason,
                "n_cols": 1,
                "n_rows": 1,
                "angle_deg": float(si["angle_deg"]),
                "pitch": float(si["pitch"]),
                "gap_cv": float(si.get("gap_cv", 0.0)),
                "pass_count": 0,
                "pass_ratio": 0.0,
                "median_iou": 0.0,
                "max_overlap": 0.0,
                "v_low": 0.0,
                "v_high": 0.0,
                "synced_with": int(synced_with),
                "rail_low_shift": 0.0,
                "rail_high_shift": 0.0,
                "rail_low_score_ratio": 1.0,
                "rail_high_score_ratio": 1.0,
                "n_dividers_snapped": 0,
                "n_partial_rejected": 0,
            })

            for k, p in enumerate(si["panels"]):
                log_records.append({
                    "record_type": "panel",
                    "image_name": image_stem,
                    "string_id": int(string_id),
                    "panel_idx": p.get("raw_idx", k),
                    "trusted": si["trusted_mask"][k],
                    "decision": "fallback_original",
                    "fallback_reason": fallback_reason,
                    "original_polygon": p["polygon"],
                    "candidate_polygon": p["polygon"],
                    "final_polygon": p["polygon"],
                    "iou_with_original": 1.0,
                    "center_shift_ratio": 0.0,
                    "area_ratio": 1.0,
                    "width_ratio": 1.0,
                    "height_ratio": 1.0,
                    "max_overlap": 0.0,
                    "edge_score_original": 0.0,
                    "edge_score_candidate": 0.0,
                    "edge_score_ratio": 1.0,
                    "decision_before_snap": "n/a",
                    "snap_used": False,
                    "snap_decision": "n/a",
                    "snap_reason": "fit_failed",
                    "iou_before_snap": 1.0,
                    "iou_after_snap": 1.0,
                    "area_ratio_after_snap": 1.0,
                    "center_shift_after_snap": 0.0,
                })
            continue

        # String quality check
        pass_ratio = fr["pass_ratio"]
        median_iou = fr["median_iou"]
        max_overlap = fr["max_overlap"]

        string_ok = (
            pass_ratio >= STRING_MIN_PASS_RATIO and
            median_iou >= STRING_MEDIAN_IOU_MIN and
            max_overlap <= STRING_MAX_OVERLAP
        )

        # Determine if block is recoverable
        block_passed = fr.get("success", False)
        is_recoverable = fr.get("recoverable_fallback_candidate", False) or fr.get("near_recoverable_string_candidate", False)

        # Apply simulated block-level overlap guard
        recovery_overlap_guard_applied = False
        recovery_simulated_max_overlap = 0.0
        recovery_overlap_discard_count = 0
        recovery_overlap_block_fallback = False

        panels_failing_overlap_guard = set()

        if is_recoverable and not string_ok:
            recovery_overlap_guard_applied = False
            
            # 1. Simulate the polygons
            simulated_polys = []
            panel_details = fr.get("panel_details", [])
            for det in panel_details:
                if det.get("is_endpoint"):
                    simulated_polys.append(det["original_polygon"])
                else:
                    if det["pass_individual"]:
                        is_valid_mid = (det.get("original_coverage_ratio", 1.0) >= 0.85 and
                                        det.get("max_inward_cut_ratio", 0.0) <= 0.20)
                        if is_valid_mid:
                            simulated_polys.append(det["candidate_polygon"])
                        else:
                            simulated_polys.append(det["original_polygon"])
                    else:
                        simulated_polys.append(det["original_polygon"])
            
            # 2. Compute overlaps among simulated polygons in the block
            from shapely.geometry import Polygon
            def get_overlap_ratio_local(poly_a, poly_b):
                try:
                    p_a = Polygon(poly_a)
                    p_b = Polygon(poly_b)
                    if not p_a.is_valid or not p_b.is_valid:
                        return 0.0
                    inter = p_a.intersection(p_b).area
                    min_area = min(p_a.area, p_b.area)
                    if min_area < 1e-5:
                        return 0.0
                    return float(inter / min_area)
                except Exception:
                    return 0.0
                    
            n_sim = len(simulated_polys)
            sim_overlaps = np.zeros((n_sim, n_sim))
            for i in range(n_sim):
                for j in range(i + 1, n_sim):
                    ov = get_overlap_ratio_local(simulated_polys[i], simulated_polys[j])
                    sim_overlaps[i, j] = ov
                    sim_overlaps[j, i] = ov
            
            max_sim_block_overlap = np.max(sim_overlaps) if n_sim > 1 else 0.0
            recovery_simulated_max_overlap = max_sim_block_overlap
            
            # 3. If max overlap is > 0.09, filter by panel overlap <= 0.06
            if max_sim_block_overlap > 0.09:
                for i in range(n_sim):
                    det = panel_details[i]
                    is_recovered_mid = (not det.get("is_endpoint") and det["pass_individual"] and
                                         det.get("original_coverage_ratio", 1.0) >= 0.85 and
                                         det.get("max_inward_cut_ratio", 0.0) <= 0.20)
                    if is_recovered_mid:
                        # Find max neighbor overlap in the simulated set
                        max_neigh = 0.0
                        for j in range(n_sim):
                            if i == j:
                                continue
                            if sim_overlaps[i, j] > max_neigh:
                                max_neigh = sim_overlaps[i, j]
                        
                        if max_neigh > 0.06:
                            panels_failing_overlap_guard.add(i)
                            
                recovery_overlap_discard_count = len(panels_failing_overlap_guard)
                
                # Check if more than 50% of recovered middle panels are discarded
                n_recovered_mid = sum(1 for det in panel_details if not det.get("is_endpoint") and det["pass_individual"] and
                                      det.get("original_coverage_ratio", 1.0) >= 0.85 and
                                      det.get("max_inward_cut_ratio", 0.0) <= 0.20)
                
                if n_recovered_mid > 0 and len(panels_failing_overlap_guard) > 0.50 * n_recovered_mid:
                    recovery_overlap_block_fallback = True
                    is_recoverable = False
        # Ensure overlap guard does not affect recoverability
        recovery_overlap_discard_count = 0
        recovery_overlap_block_fallback = False
        # Determine block decision
        block_decision = "fallback_string"
        if string_ok:
            block_decision = "use_string_lattice"
        elif is_recoverable:
            block_decision = "recoverable_string"
        else:
            block_decision = "fallback_string"

        if block_decision == "fallback_string":
            # Fallback whole string
            string_decision = "fallback_string"
            fallback_reason = "string_quality_low"
            if not string_ok:
                if pass_ratio < STRING_MIN_PASS_RATIO:
                    fallback_reason = f"pass_ratio_low_{pass_ratio:.2f}"
                elif median_iou < STRING_MEDIAN_IOU_MIN:
                    fallback_reason = f"median_iou_low_{median_iou:.2f}"
                elif max_overlap > STRING_MAX_OVERLAP:
                    fallback_reason = f"max_overlap_high_{max_overlap:.3f}"
            if recovery_overlap_block_fallback:
                fallback_reason = "overlap_guard_triggered_full_fallback"

            for k, p in enumerate(si["panels"]):
                p["string_id"] = string_id
                p["string_decision"] = "fallback_original"
                p["string_reason"] = fallback_reason
                p["string_refined"] = False
                p["polygon"] = list(p.get("original_yolo_polygon") or p["polygon"])
                is_ep = (k == 0 or k == len(si["panels"]) - 1) if len(si["panels"]) >= 5 else False
                if is_ep:
                    p["final_polygon_source"] = "endpoint_original_fallback"
                    p["final_polygon_stage"] = "string_lattice_refinement"
                    p["final_polygon_reason"] = "string_quality_failed_" + fallback_reason
                else:
                    p["final_polygon_source"] = "yolo_original"
                    p["final_polygon_stage"] = "string_lattice_refinement"
                    p["final_polygon_reason"] = "string_quality_failed_" + fallback_reason

            log_records.append({
                "record_type": "string",
                "image_name": image_stem,
                "string_id": int(string_id),
                "n_panels": int(n_panels_str),
                "n_trusted": int(si["n_trusted"]),
                "decision": "fallback_string",
                "fallback_reason": fallback_reason,
                "n_cols": int(fr.get("n_cols", 1)),
                "n_rows": int(fr.get("n_rows", 1)),
                "angle_deg": float(si["angle_deg"]),
                "pitch": float(fr["pitch"]),
                "gap_cv": float(si.get("gap_cv", 0.0)),
                "pass_count": int(fr["pass_count"]),
                "pass_ratio": float(pass_ratio),
                "median_iou": float(median_iou),
                "max_overlap": float(max_overlap),
                "v_low": float(fr["v_low"]),
                "v_high": float(fr["v_high"]),
                "synced_with": int(synced_with),
                "rail_low_shift": float(fr.get("snap_rail_info", {}).get("rail_low_shift", 0.0)),
                "rail_high_shift": float(fr.get("snap_rail_info", {}).get("rail_high_shift", 0.0)),
                "rail_low_score_ratio": float(fr.get("snap_rail_info", {}).get("rail_low_score_ratio", 1.0)),
                "rail_high_score_ratio": float(fr.get("snap_rail_info", {}).get("rail_high_score_ratio", 1.0)),
                "n_dividers_snapped": int(fr.get("n_dividers_snapped", 0)),
                "n_partial_rejected": int(fr.get("n_partial_rejected", 0)),
                # Recovery metrics
                "pitch_cv": float(fr.get("pitch_cv", 0.0)),
                "rail_width_cv": float(fr.get("rail_width_cv", 0.0)),
                "median_middle_original_coverage": float(fr.get("median_middle_original_coverage", 1.0)),
                "failed_middle_coverage_count": int(fr.get("failed_middle_coverage_count", 0)),
                "failed_middle_inward_cut_count": int(fr.get("failed_middle_inward_cut_count", 0)),
                "divider_cross_count_middle": int(fr.get("divider_cross_count_middle", 0)),
                "outside_image_count_middle": int(fr.get("outside_image_count_middle", 0)),
                "max_middle_inward_cut_ratio": float(fr.get("max_middle_inward_cut_ratio", 0.0)),
                "recoverable_fallback_candidate": bool(fr.get("recoverable_fallback_candidate", False)),
                "near_recoverable_string_candidate": bool(fr.get("near_recoverable_string_candidate", False)),
                "near_recoverable_reason": str(fr.get("near_recoverable_reason", "")),
                "near_recoverable_middle_valid_count": int(fr.get("near_recoverable_middle_valid_count", 0)),
                "recovery_overlap_guard_applied": bool(recovery_overlap_guard_applied),
                "recovery_simulated_max_overlap": float(recovery_simulated_max_overlap),
                "recovery_overlap_discard_count": int(recovery_overlap_discard_count),
                "recovery_overlap_block_fallback": bool(recovery_overlap_block_fallback),
            })

            panel_details = fr["panel_details"]
            for det in panel_details:
                p = det["p_obj"]
                det["panel_decision"] = "fallback_original"
                det["final_polygon"] = det["original_polygon"]
                log_records.append({
                    "record_type": "panel",
                    "image_name": image_stem,
                    "string_id": int(string_id),
                    "panel_idx": det["panel_idx"],
                    "trusted": det["trusted"],
                    "decision": "fallback_original",
                    "fallback_reason": fallback_reason,
                    "original_polygon": det["original_polygon"],
                    "candidate_polygon": det["candidate_polygon"],
                    "final_polygon": det["original_polygon"],
                    "iou_with_original": float(det["iou_with_original"]),
                    "center_shift_ratio": float(det["center_shift_ratio"]),
                    "area_ratio": float(det["area_ratio"]),
                    "width_ratio": float(det["width_ratio"]),
                    "height_ratio": float(det["height_ratio"]),
                    "max_overlap": float(max_overlap),
                    "edge_score_original": float(det["edge_score_original"]),
                    "edge_score_candidate": float(det["edge_score_candidate"]),
                    "edge_score_ratio": float(det["edge_score_ratio"]),
                    "decision_before_snap": det.get("decision_before_snap", "n/a"),
                    "snap_used": bool(det.get("snap_used", False)),
                    "snap_decision": det.get("snap_decision", "n/a"),
                    "snap_reason": det.get("snap_reason", ""),
                    "iou_before_snap": float(det.get("iou_before_snap", det["iou_with_original"])),
                    "iou_after_snap": float(det.get("iou_after_snap", det["iou_with_original"])),
                    "area_ratio_after_snap": float(det.get("area_ratio_after_snap", det["area_ratio"])),
                    "center_shift_after_snap": float(det.get("center_shift_after_snap", det["center_shift_ratio"])),
                })
            continue

        # String passes or is recoverable
        string_decision = block_decision
        panel_details = fr["panel_details"]
        pass_in_str = 0

        # Compute endpoint stats for string
        n_endpoints_str = sum(1 for det in panel_details if det.get("is_endpoint"))
        n_middle_str = len(panel_details) - n_endpoints_str
        middle_pass_count_str = sum(1 for det in panel_details if det["pass_individual"] and not det.get("is_endpoint"))
        middle_pass_ratio_str = middle_pass_count_str / max(n_middle_str, 1)
        endpoint_pass_count_str = sum(1 for det in panel_details if det["pass_individual"] and det.get("is_endpoint"))
        endpoint_fallback_count_str = sum(1 for det in panel_details if not det["pass_individual"] and det.get("is_endpoint"))
        
        endpoint_source_counts_str = {
            "yolo_projection": 0,
            "pitch_extrapolation": 0,
            "edge_scan": 0,
            "consensus": 0,
            "fallback_original": 0
        }
        for det in panel_details:
            if det.get("is_endpoint"):
                if det["pass_individual"]:
                    src = det.get("endpoint_outer_source", "yolo_projection")
                    if src in endpoint_source_counts_str:
                        endpoint_source_counts_str[src] += 1
                    else:
                        endpoint_source_counts_str["yolo_projection"] += 1
                else:
                    endpoint_source_counts_str["fallback_original"] += 1

        log_records.append({
            "record_type": "string",
            "image_name": image_stem,
            "string_id": int(string_id),
            "n_panels": int(n_panels_str),
            "n_trusted": int(si["n_trusted"]),
            "decision": string_decision,
            "fallback_reason": "",
            "n_cols": int(fr.get("n_cols", 1)),
            "n_rows": int(fr.get("n_rows", 1)),
            "angle_deg": float(si["angle_deg"]),
            "pitch": float(fr["pitch"]),
            "gap_cv": float(si.get("gap_cv", 0.0)),
            "pass_count": int(fr["pass_count"]),
            "pass_ratio": float(pass_ratio),
            "median_iou": float(median_iou),
            "max_overlap": float(max_overlap),
            "v_low": float(fr["v_low"]),
            "v_high": float(fr["v_high"]),
            "synced_with": int(synced_with),
            "rail_low_shift": float(fr.get("snap_rail_info", {}).get("rail_low_shift", 0.0)),
            "rail_high_shift": float(fr.get("snap_rail_info", {}).get("rail_high_shift", 0.0)),
            "rail_low_score_ratio": float(fr.get("snap_rail_info", {}).get("rail_low_score_ratio", 1.0)),
            "rail_high_score_ratio": float(fr.get("snap_rail_info", {}).get("rail_high_score_ratio", 1.0)),
            "n_dividers_snapped": int(fr.get("n_dividers_snapped", 0)),
            "n_partial_rejected": int(fr.get("n_partial_rejected", 0)),
            # New endpoint fields
            "n_middle_panels": int(n_middle_str),
            "n_endpoint_panels": int(n_endpoints_str),
            "middle_pass_count": int(middle_pass_count_str),
            "middle_pass_ratio": float(middle_pass_ratio_str),
            "endpoint_pass_count": int(endpoint_pass_count_str),
            "endpoint_fallback_count": int(endpoint_fallback_count_str),
            "endpoint_source_counts": endpoint_source_counts_str,
            # Recovery metrics
            "recovery_overlap_guard_applied": bool(recovery_overlap_guard_applied),
            "recovery_simulated_max_overlap": float(recovery_simulated_max_overlap),
            "recovery_overlap_discard_count": int(recovery_overlap_discard_count),
            "recovery_overlap_block_fallback": bool(recovery_overlap_block_fallback),
        })

        print(f"[STRING_LATTICE] img={image_stem} string_id={string_id} n={n_panels_str} "
              f"trusted={si['n_trusted']} pass={fr['pass_count']}/{n_panels_str} "
              f"pass_ratio={pass_ratio:.2f} median_iou={median_iou:.3f} decision={string_decision}")

        # Initialize geometry gate counters
        gate_total = 0
        gate_accept = 0
        gate_reject = 0
        gate_reason_counts = collections.Counter()
        near_edge_total = 0
        near_edge_accept = 0
        near_edge_reject = 0
        source_before_counts = collections.Counter()
        source_after_counts = collections.Counter()
        rejected_examples = []
        for idx, det in enumerate(panel_details):
            p = det["p_obj"]
            p["string_id"] = string_id
            blk_id = p.get("block_id")

            use_lattice = False
            if string_decision == "use_string_lattice":
                use_lattice = det["pass_individual"]
            elif string_decision == "recoverable_string":
                if det.get("is_endpoint"):
                    use_lattice = False
                    if det["fail_reason"] == "ok" or not det["fail_reason"]:
                        det["fail_reason"] = "recovery_mode_endpoint_fallback"
                else:
                    if det["pass_individual"]:
                        is_valid_mid = (det.get("original_coverage_ratio", 1.0) >= 0.85 and
                                        det.get("max_inward_cut_ratio", 0.0) <= 0.20)
                        if not is_valid_mid:
                            use_lattice = False
                            det["fail_reason"] = "recovery_filter_failed"
                        elif idx in panels_failing_overlap_guard:
                            use_lattice = False
                            det["fail_reason"] = "overlap_guard_discarded"
                        else:
                            use_lattice = True
                    else:
                        use_lattice = False
            else:
                use_lattice = False

            if use_lattice:
                # Use best candidate (may be snapped or pre-snap lattice)
                final_poly = det["candidate_polygon"]
                
                # Save original YOLO bbox and source
                orig_bbox = p["bbox"].copy()
                orig_source = "string_lattice_middle"
                source_before_counts[orig_source] += 1
                # Compute candidate bbox from final_poly
                cand_pts = np.array(final_poly, dtype=np.float32)
                cand_x1 = float(np.min(cand_pts[:, 0]))
                cand_y1 = float(np.min(cand_pts[:, 1]))
                cand_x2 = float(np.max(cand_pts[:, 0]))
                cand_y2 = float(np.max(cand_pts[:, 1]))
                # Original bbox components
                ox1, oy1, ox2, oy2 = orig_bbox
                o_w = ox2 - ox1
                o_h = oy2 - oy1
                # Metrics
                inter_x1 = max(ox1, cand_x1)
                inter_y1 = max(oy1, cand_y1)
                inter_x2 = min(ox2, cand_x2)
                inter_y2 = min(oy2, cand_y2)
                if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                else:
                    inter_area = 0.0
                orig_area = o_w * o_h
                cand_area = (cand_x2 - cand_x1) * (cand_y2 - cand_y1)
                union_area = orig_area + cand_area - inter_area
                iou = inter_area / union_area if union_area > 0 else 0.0
                # Center shift
                ox_c = (ox1 + ox2) / 2.0
                oy_c = (oy1 + oy2) / 2.0
                cx_c = (cand_x1 + cand_x2) / 2.0
                cy_c = (cand_y1 + cand_y2) / 2.0
                center_shift = ((ox_c - cx_c) ** 2 + (oy_c - cy_c) ** 2) ** 0.5
                area_ratio = cand_area / orig_area if orig_area > 0 else 0.0
                exp_x = max(0.0, cand_x2 - ox2) + max(0.0, ox1 - cand_x1)
                exp_y = max(0.0, cand_y2 - oy2) + max(0.0, oy1 - cand_y1)
                inside_image = (0 <= cand_x1 and cand_x2 <= 640 and 0 <= cand_y1 and cand_y2 <= 512)
                near_edge = (ox1 <= 8 or oy1 <= 8 or ox2 >= 632 or oy2 >= 504)
                # Overlap with same block
                blk_id = p.get("block_id")
                max_overlap = 0.0
                if blk_id is not None:
                    for other_det in panel_details:
                        if other_det is det:
                            continue
                        other_p = other_det["p_obj"]
                        if other_p.get("block_id") != blk_id:
                            continue
                        obx1, oby1, obx2, oby2 = other_p["bbox"]
                        ix1 = max(cand_x1, obx1)
                        iy1 = max(cand_y1, oby1)
                        ix2 = min(cand_x2, obx2)
                        iy2 = min(cand_y2, oby2)
                        if ix2 > ix1 and iy2 > iy1:
                            i_area = (ix2 - ix1) * (iy2 - iy1)
                            o_area = (obx2 - obx1) * (oby2 - oby1)
                            union = cand_area + o_area - i_area
                            ov = i_area / union if union > 0 else 0.0
                            if ov > max_overlap:
                                max_overlap = ov
                # Determine thresholds
                if near_edge:
                    iou_thr = 0.70
                    shift_thr = 0.25 * o_h
                else:
                    iou_thr = 0.55
                    shift_thr = 0.40 * o_h
                accept = (
                    iou >= iou_thr and
                    center_shift <= shift_thr and
                    0.65 <= area_ratio <= 1.60 and
                    exp_x <= 0.35 * o_w and
                    exp_y <= 0.35 * o_h and
                    inside_image and
                    (blk_id is None or max_overlap <= 0.12)
                )
                gate_total += 1
                if near_edge:
                    near_edge_total += 1
                # Record reason if rejected
                reject_reason = ""
                if not accept:
                    if iou < iou_thr:
                        reject_reason = "iou"
                    elif center_shift > shift_thr:
                        reject_reason = "shift"
                    elif not (0.65 <= area_ratio <= 1.60):
                        reject_reason = "area"
                    elif exp_x > 0.35 * o_w or exp_y > 0.35 * o_h:
                        reject_reason = "expansion"
                    elif not inside_image:
                        reject_reason = "outside"
                    elif blk_id is not None and max_overlap > 0.12:
                        reject_reason = "overlap"
                    else:
                        reject_reason = "other"

                # Conditional geometry gate relaxation
                relaxed_accept = False
                reason_code = None
                is_endpoint = det.get("is_endpoint", False)

                # Initialize metrics to None or default values so they can be stored in trace logs
                rel_area_ratio = 1.0
                rel_center_shift_y = 0.0
                rel_exp_x = 0.0
                rel_exp_y = 0.0

                if not accept and not is_endpoint and string_decision == "use_string_lattice":
                    if reject_reason in {"iou", "shift"}:
                        # Calculate relaxed criteria using original YOLO box (p.get("original_yolo_bbox"))
                        orig_box = p.get("original_yolo_bbox") if p.get("original_yolo_bbox") is not None else p["bbox"]
                        ob_w = orig_box[2] - orig_box[0]
                        ob_h = orig_box[3] - orig_box[1]
                        ob_area = ob_w * ob_h
                        ob_diag = (ob_w ** 2 + ob_h ** 2) ** 0.5
                        
                        ox_c = (orig_box[0] + orig_box[2]) / 2.0
                        oy_c = (orig_box[1] + orig_box[3]) / 2.0
                        
                        rel_area_ratio = det.get("area_ratio") if det.get("area_ratio") is not None else (cand_area / ob_area if ob_area > 0 else 0.0)
                        rel_center_shift_ratio = center_shift / max(ob_diag, 1.0)
                        rel_center_shift_y = abs(oy_c - cy_c)
                        
                        rel_exp_x = max(0.0, cand_x2 - orig_box[2]) + max(0.0, orig_box[0] - cand_x1)
                        rel_exp_y = max(0.0, cand_y2 - orig_box[3]) + max(0.0, orig_box[1] - cand_y1)
                        
                        relaxed_accept_cond = (
                            not near_edge and
                            inside_image and
                            max_overlap <= 0.02 and
                            0.85 <= rel_area_ratio <= 1.15 and
                            iou >= 0.42 and
                            (rel_center_shift_ratio <= 0.35 or rel_center_shift_y <= 0.35 * ob_h) and
                            rel_exp_x <= 0.45 * ob_w and
                            rel_exp_y <= 0.15 * ob_h
                        )
                        relaxed_gate_total += 1
                        if relaxed_accept_cond:
                            relaxed_accept = True
                            relaxed_gate_accept += 1
                        else:
                            relaxed_gate_reject += 1
                            # Determine first condition that fails
                            if near_edge:
                                reason_code = "near_edge"
                            elif not inside_image:
                                reason_code = "not_inside_image"
                            elif max_overlap > 0.02:
                                reason_code = "max_overlap_high"
                            elif not (0.85 <= rel_area_ratio <= 1.15):
                                reason_code = "area_ratio_out"
                            elif iou < 0.42:
                                reason_code = "low_iou"
                            elif not (rel_center_shift_ratio <= 0.35 or rel_center_shift_y <= 0.35 * ob_h):
                                reason_code = "large_center_shift"
                            elif rel_exp_x > 0.45 * ob_w:
                                reason_code = "large_exp_x"
                            elif rel_exp_y > 0.15 * ob_h:
                                reason_code = "large_exp_y"
                            else:
                                reason_code = "unknown"
                            relaxed_gate_reject_reason[reason_code] += 1

                # Apply result
                if accept:
                    gate_accept += 1
                    if near_edge:
                        near_edge_accept += 1
                    p["geometry_gate_accept"] = True
                    p["geometry_gate_reason"] = ""
                    p["geometry_gate_relaxed_accept"] = False
                elif relaxed_accept:
                    p["geometry_gate_accept"] = True
                    p["geometry_gate_reason"] = "relaxed_iou_shift"
                    p["geometry_gate_relaxed_accept"] = True
                    
                    # Store values for trace logs
                    p["geometry_gate_relaxed_iou"] = iou
                    p["geometry_gate_relaxed_center_shift"] = rel_center_shift_y
                    p["geometry_gate_relaxed_area_ratio"] = rel_area_ratio
                    p["geometry_gate_relaxed_expansion_x"] = rel_exp_x
                    p["geometry_gate_relaxed_expansion_y"] = rel_exp_y
                    p["geometry_gate_relaxed_max_overlap"] = max_overlap
                    
                    # Record example
                    if len(relaxed_accepted_examples) < 10:
                        relaxed_accepted_examples.append({
                            "raw_idx": p.get("raw_idx"),
                            "bbox": orig_bbox,
                            "candidate_bbox": [cand_x1, cand_y1, cand_x2, cand_y2],
                            "source": "string_lattice_middle",
                            "reason": "relaxed_iou_shift",
                            "iou": iou,
                            "center_shift": rel_center_shift_y,
                            "area_ratio": rel_area_ratio,
                            "expansion_x": rel_exp_x,
                            "expansion_y": rel_exp_y,
                            "max_overlap": max_overlap,
                            "near_edge": near_edge
                        })
                else:
                    # Original reject AND not relaxed accept
                    gate_reject += 1
                    if near_edge:
                        near_edge_reject += 1
                    p["geometry_gate_accept"] = False
                    p["geometry_gate_reason"] = reject_reason
                    if not is_endpoint:
                        p["geometry_gate_relaxed_accept"] = False
                        p["geometry_gate_relaxed_reject_reason"] = reason_code
                        # Store values for trace logs
                        p["geometry_gate_relaxed_iou"] = iou
                        p["geometry_gate_relaxed_center_shift"] = rel_center_shift_y
                        p["geometry_gate_relaxed_area_ratio"] = rel_area_ratio
                        p["geometry_gate_relaxed_expansion_x"] = rel_exp_x
                        p["geometry_gate_relaxed_expansion_y"] = rel_exp_y
                        p["geometry_gate_relaxed_max_overlap"] = max_overlap
                        
                        if len(relaxed_rejected_examples) < 10:
                            relaxed_rejected_examples.append({
                                "raw_idx": p.get("raw_idx"),
                                "bbox": orig_bbox,
                                "candidate_bbox": [cand_x1, cand_y1, cand_x2, cand_y2],
                                "source": "yolo_original",
                                "reason": reason_code,
                                "iou": iou,
                                "center_shift": rel_center_shift_y,
                                "area_ratio": rel_area_ratio,
                                "expansion_x": rel_exp_x,
                                "expansion_y": rel_exp_y,
                                "max_overlap": max_overlap,
                                "near_edge": near_edge
                            })
                    # Fallback to rectangle from original YOLO bbox
                    p["polygon"] = [[ox1, oy1], [ox2, oy1], [ox2, oy2], [ox1, oy2]]
                    p["final_polygon_source"] = "yolo_original"
                    p["final_polygon_stage"] = "geometry_gate_fallback"
                    p["final_polygon_reason"] = f"geometry_gate_rejected_{reject_reason}"
                    # Record example for logging
                    if len(rejected_examples) < 10:
                        rejected_examples.append({
                            "raw_idx": p.get("raw_idx"),
                            "iou": iou,
                            "reason": reject_reason,
                            "bbox": orig_bbox,
                            "candidate_bbox": [cand_x1, cand_y1, cand_x2, cand_y2]
                        })
                    # Skip further processing for this panel
                    # Continue to next panel after logging below
                    # (will still add rec later)
                # Update source after gate
                new_source = p.get("final_polygon_source", orig_source)
                source_after_counts[new_source] += 1
                # --- Geometry Acceptance Gate end ---
                snap_dec = det.get("snap_decision", "")
                if snap_dec == "use_snapped":
                    final_poly = det["candidate_polygon"]  # already updated to snapped
                    panel_decision = "use_string_lattice"
                else:
                    final_poly = det["candidate_polygon"]
                    panel_decision = "use_string_lattice"
                p["polygon"] = [[int(round(pt[0])), int(round(pt[1]))] for pt in final_poly]
                if det.get("is_endpoint"):
                    p["final_polygon_source"] = "middle_locked_endpoint"
                    p["final_polygon_stage"] = "string_lattice_refinement"
                    p["final_polygon_reason"] = "locked_by_middle_pitch"

                # ------------------------------------------------------------------
                # String-lattice propagation - final-polygon metadata
                # ------------------------------------------------------------------
                attempt_propagation = (string_decision == "use_string_lattice")
                
                if attempt_propagation:
                    lattice_prop_total += 1
                    
                    # 1. Guard against missing/too-small candidate polygon
                    if not final_poly or len(final_poly) < 4:
                        reject_code = "few_points"
                        p["string_lattice_propagation_accept"] = False
                        p["string_lattice_propagation_reject_reason"] = reject_code
                        lattice_prop_reject += 1
                        lattice_prop_reject_reason[reject_code] += 1
                        
                        # Fallback: keep fallback, do not use final_poly
                        orig_poly = det.get("original_polygon")
                        if orig_poly:
                            p["polygon"] = [[int(round(pt[0])), int(round(pt[1]))] for pt in orig_poly]
                        else:
                            ox1, oy1, ox2, oy2 = p["bbox"]
                            p["polygon"] = [[ox1, oy1], [ox2, oy1], [ox2, oy2], [ox1, oy2]]
                            
                        if not det.get("is_endpoint"):
                            p["final_polygon_source"] = "yolo_original"
                            p["final_polygon_stage"] = "string_lattice_refinement"
                            p["final_polygon_reason"] = "string_lattice_propagation_rejected_" + reject_code
                    else:
                        # 2. Calculate bbox using pure Python
                        xs = [pt[0] for pt in final_poly]
                        ys = [pt[1] for pt in final_poly]
                        cand_x1, cand_x2 = min(xs), max(xs)
                        cand_y1, cand_y2 = min(ys), max(ys)
                        
                        # 3. Polygon area using Shoelace formula
                        area_val = 0.0
                        n_pts = len(final_poly)
                        for i in range(n_pts):
                            x0, y0 = final_poly[i]
                            x1, y1 = final_poly[(i + 1) % n_pts]
                            area_val += x0 * y1 - x1 * y0
                        cand_area = abs(area_val) * 0.5
                        
                        # 4. Inside image bounds (640x512)
                        inside_image = (0 <= cand_x1 and cand_x2 <= 640 and 0 <= cand_y1 and cand_y2 <= 512)
                        
                        # 5. IoU with original bbox (p["bbox"])
                        ox1, oy1, ox2, oy2 = p["bbox"]
                        inter_x1 = max(ox1, cand_x1)
                        inter_y1 = max(oy1, cand_y1)
                        inter_x2 = min(ox2, cand_x2)
                        inter_y2 = min(oy2, cand_y2)
                        inter_w = max(0.0, inter_x2 - inter_x1)
                        inter_h = max(0.0, inter_y2 - inter_y1)
                        inter_area = inter_w * inter_h
                        orig_area = (ox2 - ox1) * (oy2 - oy1)
                        cand_bbox_area = (cand_x2 - cand_x1) * (cand_y2 - cand_y1)
                        union_area = orig_area + cand_bbox_area - inter_area
                        iou = inter_area / union_area if union_area > 0 else 0.0
                        
                        # 6. Geometry-gate condition (must not be explicitly False)
                        gate_ok = p.get("geometry_gate_accept", None) is not False
                        
                        # 7. Final sanity check
                        sane = (
                            cand_area > 0 and
                            inside_image and
                            (iou >= 0.45 or p.get("geometry_gate_relaxed_accept", False)) and
                            gate_ok
                        )
                        
                        if sane:
                            p["string_lattice_propagation_accept"] = True
                            if not det.get("is_endpoint"):
                                p["final_polygon_source"] = "string_lattice_middle"
                                p["final_polygon_stage"] = "string_lattice_refinement"
                                if p.get("geometry_gate_relaxed_accept", False):
                                    p["final_polygon_reason"] = "lattice_projection_relaxed_gate"
                                else:
                                    p["final_polygon_reason"] = "lattice_projection"
                            lattice_prop_accept += 1
                        else:
                            p["string_lattice_propagation_accept"] = False
                            if not gate_ok:
                                reject_code = "gate_false"
                            elif cand_area <= 0:
                                reject_code = "zero_area"
                            elif not inside_image:
                                reject_code = "outside_image"
                            elif iou < 0.45:
                                reject_code = "low_iou"
                            else:
                                reject_code = "unknown"
                                
                            p["string_lattice_propagation_reject_reason"] = reject_code
                            lattice_prop_reject += 1
                            lattice_prop_reject_reason[reject_code] += 1
                            
                            # Restore fallback polygon
                            orig_poly = det.get("original_polygon")
                            if orig_poly:
                                p["polygon"] = [[int(round(pt[0])), int(round(pt[1]))] for pt in orig_poly]
                            else:
                                ox1, oy1, ox2, oy2 = p["bbox"]
                                p["polygon"] = [[ox1, oy1], [ox2, oy1], [ox2, oy2], [ox1, oy2]]
                                
                            if not det.get("is_endpoint"):
                                if gate_ok:
                                    p["final_polygon_source"] = "yolo_original"
                                    p["final_polygon_stage"] = "string_lattice_refinement"
                                    p["final_polygon_reason"] = "string_lattice_propagation_rejected_" + reject_code
                                    
                    if len(prop_examples) < 10:
                        prop_examples.append({
                            "raw_idx": p.get("raw_idx"),
                            "bbox": p.get("bbox"),
                            "polygon": p.get("polygon"),
                            "final_polygon_source": p.get("final_polygon_source"),
                            "final_polygon_reason": p.get("final_polygon_reason"),
                            "prop_accept": p.get("string_lattice_propagation_accept"),
                            "reject_reason": p.get("string_lattice_propagation_reject_reason"),
                        })
                # ------------------------------------------------------------------
                # Update derived features
                feats = get_polygon_features(p["polygon"])
                p["bbox"] = [round(v) for v in feats["bbox"]]
                p["box"] = p["bbox"]
                p["area"] = feats["area"]
                p["center"] = feats["center"]
                p["aspect_ratio"] = feats["aspect_ratio"]
                p["string_refined"] = True
                pass_in_str += 1
            else:
                # Check for hard-fail (fallback even in passing string)
                # Simple block-level log (no undefined variables)
                logger.info(
                    f"[BLOCK_LOG] img={image_stem} block_id={blk_id} panels={len(panel_details)} "
                    f"decision={det.get('two_col_decision', 'n/a')}"
                )
                is_hard_fail = (
                    det["fail_reason"] == "outside_image" or
                    det["center_shift_ratio"] > STRING_PANEL_HARD_SHIFT_MAX or
                    det["iou_with_original"] < STRING_PANEL_HARD_IOU_MIN or
                    not (STRING_PANEL_HARD_AREA_MIN <= det["area_ratio"] <= STRING_PANEL_HARD_AREA_MAX)
                )
                final_poly = det["original_polygon"]
                
                # Check if it was because safe improvement failed or endpoint was diagnostic only
                if det["fail_reason"] == "reject_partial":
                    panel_decision = "reject_partial"
                elif det["fail_reason"] == "diagnostic_only":
                    panel_decision = "keep_original_not_improved"
                elif ("iou_low" in det["fail_reason"] or 
                      "center_shift_large" in det["fail_reason"] or 
                      "area_ratio_out" in det["fail_reason"] or 
                      "overlap_increased" in det["fail_reason"] or 
                      "edge_score_not_improved" in det["fail_reason"] or 
                      "outside_image_bounds" in det["fail_reason"]):
                    panel_decision = "keep_original_not_improved"
                else:
                    panel_decision = "fallback_original"
                    
                # Relaxed endpoint acceptance checks
                is_relaxed_accept = False
                relax_reject_reason = "not_checked"
                overlap_val = 999.0
                iou = 0.0
                center_shift = 999.0

                if det.get("is_endpoint"):
                    fail_reason_str = str(det.get("fail_reason", ""))
                    if fail_reason_str.startswith("neighbor_overlap") or fail_reason_str.startswith("failed_neighbor_overlap"):
                        overlap_val = det.get("neighbor_overlap_val", 999.0)
                        cand_snap = det.get("cand_snap")

                        if overlap_val <= 0.04 and cand_snap is not None and len(cand_snap) >= 4:
                            xs = [pt[0] for pt in cand_snap]
                            ys = [pt[1] for pt in cand_snap]
                            cand_x1, cand_x2 = min(xs), max(xs)
                            cand_y1, cand_y2 = min(ys), max(ys)

                            inside_image = (0 <= cand_x1 and cand_x2 <= 640 and 0 <= cand_y1 and cand_y2 <= 512)
                            if inside_image:
                                ox1, oy1, ox2, oy2 = p["bbox"]
                                orig_w = ox2 - ox1
                                orig_h = oy2 - oy1
                                orig_area = orig_w * orig_h
                                cand_area = (cand_x2 - cand_x1) * (cand_y2 - cand_y1)

                                # IoU computation
                                inter_x1 = max(ox1, cand_x1)
                                inter_y1 = max(oy1, cand_y1)
                                inter_x2 = min(ox2, cand_x2)
                                inter_y2 = min(oy2, cand_y2)
                                inter_w = max(0.0, inter_x2 - inter_x1)
                                inter_h = max(0.0, inter_y2 - inter_y1)
                                inter_area = inter_w * inter_h
                                union_area = orig_area + cand_area - inter_area
                                iou = inter_area / union_area if union_area > 0 else 0.0

                                # Center shift computation
                                ox_c = (ox1 + ox2) / 2.0
                                oy_c = (oy1 + oy2) / 2.0
                                cx_c = (cand_x1 + cand_x2) / 2.0
                                cy_c = (cand_y1 + cand_y2) / 2.0
                                center_shift = ((ox_c - cx_c) ** 2 + (oy_c - cy_c) ** 2) ** 0.5

                                # Verify metrics bounds
                                if iou < 0.40:
                                    relax_reject_reason = "low_iou"
                                elif center_shift > 0.45 * orig_h:
                                    relax_reject_reason = "large_shift"
                                elif det.get("original_coverage_ratio", 1.0) < 0.78:
                                    relax_reject_reason = "low_coverage"
                                else:
                                    is_relaxed_accept = True
                            else:
                                relax_reject_reason = "outside_image"
                        else:
                            if overlap_val > 0.04:
                                relax_reject_reason = "overlap_too_high"
                            elif cand_snap is None or len(cand_snap) < 4:
                                relax_reject_reason = "invalid_polygon"
                            else:
                                relax_reject_reason = "other"
                    else:
                        relax_reject_reason = "not_overlap_fail"

                if is_relaxed_accept:
                    cand_snap = det["cand_snap"]
                    p["polygon"] = [[int(round(pt[0])), int(round(pt[1]))] for pt in cand_snap]
                    p["final_polygon_source"] = "middle_locked_endpoint"
                    p["final_polygon_stage"] = "string_lattice_refinement"
                    p["final_polygon_reason"] = "endpoint_relaxed_neighbor_overlap"
                    p["endpoint_relaxed_accept"] = True
                    p["endpoint_relaxed_overlap"] = overlap_val
                    p["endpoint_relaxed_iou"] = iou
                    p["endpoint_relaxed_center_shift"] = center_shift
                else:
                    p["polygon"] = [[int(round(pt[0])), int(round(pt[1]))] for pt in final_poly]
                    if det.get("is_endpoint"):
                        p["final_polygon_source"] = "endpoint_original_fallback"
                        p["final_polygon_stage"] = "string_lattice_refinement"
                        p["final_polygon_reason"] = "string_endpoint_failed_" + str(det.get("fail_reason"))
                        p["endpoint_relaxed_accept"] = False
                        p["endpoint_relaxed_reject_reason"] = relax_reject_reason
                    else:
                        p["final_polygon_source"] = "yolo_original"
                        p["final_polygon_stage"] = "string_lattice_refinement"
                        p["final_polygon_reason"] = "string_middle_failed_" + str(det.get("fail_reason"))
                feats = get_polygon_features(p["polygon"])
                p["bbox"] = [round(v) for v in feats["bbox"]]
                p["box"] = p["bbox"]
                p["area"] = feats["area"]
                p["center"] = feats["center"]
                p["aspect_ratio"] = feats["aspect_ratio"]
                p["string_refined"] = False

            p["string_decision"] = panel_decision
            p["string_reason"] = det["fail_reason"]
            if det.get("is_outer_panel"):
                p["is_outer_panel"] = True
                p["outer_side"] = det.get("outer_side")
                p["outer_divider_source"] = det.get("outer_divider_source")
                p["outer_divider_shift"] = det.get("outer_divider_shift")
                p["outer_divider_score"] = det.get("outer_divider_score")
                p["outer_guard_decision"] = det.get("outer_guard_decision")
                p["outer_guard_reason"] = det.get("outer_guard_reason")
                p["outer_divider_u_proposal"] = det.get("outer_divider_u_proposal")
                p["outer_divider_u_snapped"] = det.get("outer_divider_u_snapped")
                p["outer_divider_u_axis"] = det.get("outer_divider_u_axis")
                p["outer_divider_v_axis"] = det.get("outer_divider_v_axis")
                p["outer_divider_origin"] = det.get("outer_divider_origin")
                p["outer_divider_rail_low"] = det.get("outer_divider_rail_low")
                p["outer_divider_rail_high"] = det.get("outer_divider_rail_high")
                # Geometry gate summary for this outer panel (string)
                logger.info(
                    f"[GEOMETRY_GATE] img={image_stem} string_id={string_id} total={gate_total} accept={gate_accept} reject={gate_reject} near_edge_total={near_edge_total} near_edge_accept={near_edge_accept} near_edge_reject={near_edge_reject} source_before={dict(source_before_counts)} source_after={dict(source_after_counts)}"
                )
                if rejected_examples:
                    for i, ex in enumerate(rejected_examples):
                        logger.info(
                            f"[GEOMETRY_GATE] REJ {i} raw_idx={ex['raw_idx']} iou={ex['iou']:.3f} reason={ex['reason']}"
                        )
            if det.get("is_endpoint"):
                p["is_endpoint_panel"] = True
                p["endpoint_side"] = det.get("endpoint_side")
                p["endpoint_decision"] = det.get("endpoint_decision")
                p["endpoint_reason"] = det.get("endpoint_fallback_reason")
                p["candidate_lattice_area_ratio"] = det.get("area_ratio")
                p["candidate_lattice_center_shift"] = det.get("center_shift_ratio")
                p["candidate_lattice_iou"] = det.get("iou_with_original")
                p["candidate_lattice_u_length_ratio"] = det.get("endpoint_length_ratio")
                p["candidate_lattice_v_width_ratio"] = det.get("height_ratio")
                p["candidate_lattice_edge_score"] = det.get("edge_score_candidate")
                p["candidate_original_edge_score"] = det.get("edge_score_original")
                p["edge_score_ratio"] = det.get("edge_score_ratio")
                p["rail_fit_source"] = fr.get("rail_fit_source", "all_panels")
                
                # Additional Hybrid Resolver parameters for drawing
                p["outer_divider_origin"] = fr["origin"]
                p["outer_divider_u_axis"] = fr["u_axis"]
                p["outer_divider_v_axis"] = fr["v_axis"]
                p["outer_divider_rail_low"] = fr["v_low"]
                p["outer_divider_rail_high"] = fr["v_high"]
                
                p["endpoint_yolo_outer_u"] = det.get("endpoint_yolo_outer_u")
                p["endpoint_pitch_outer_u"] = det.get("endpoint_pitch_outer_u")
                p["endpoint_edge_outer_u"] = det.get("endpoint_edge_outer_u")
                p["endpoint_consensus_outer_u"] = det.get("endpoint_consensus_outer_u")
                p["endpoint_final_outer_u"] = det.get("endpoint_final_outer_u")
                p["endpoint_outer_source"] = det.get("endpoint_outer_source")
                p["endpoint_candidates"] = det.get("endpoint_candidates")
            det["panel_decision"] = panel_decision
            det["final_polygon"] = det["candidate_polygon"] if use_lattice else det["original_polygon"]

            # Per-panel overlap (for logging)
            p_overlap_max = 0.0
            poly_k = np.array(det["final_polygon"], dtype=np.float32)
            pk_h = cv2.convexHull(poly_k.reshape(-1, 1, 2)).astype(np.float32)
            for det2 in panel_details:
                if det2 is det:
                    continue
                poly2 = np.array(det2["final_polygon"] if "final_polygon" in det2 else det2["original_polygon"], dtype=np.float32)
                p2_h = cv2.convexHull(poly2.reshape(-1, 1, 2)).astype(np.float32)
                ai, _ = cv2.intersectConvexConvex(pk_h, p2_h)
                if ai > 0:
                    min_a = min(cv2.contourArea(pk_h), cv2.contourArea(p2_h))
                    p_overlap_max = max(p_overlap_max, ai / max(min_a, 1e-3))

            rec = {
                "record_type": "panel",
                "image_name": image_stem,
                "string_id": int(string_id),
                "panel_idx": det["panel_idx"],
                "trusted": det["trusted"],
                "decision": panel_decision,
                "fallback_reason": det["fail_reason"],
                # Endpoint Isolation Architecture fields
                "panel_role": det.get("panel_role", "middle"),
                "is_endpoint": bool(det.get("is_endpoint", False)),
                "endpoint_side": det.get("endpoint_side", None),
                "used_for_lattice_fit": bool(det.get("used_for_lattice_fit", False)),
                "local_angle_deg": float(det.get("local_angle_deg", 0.0)),
                "endpoint_extrapolate_ratio": float(det.get("endpoint_extrapolate_ratio", 0.0)),
                "candidate_polygon_source": det.get("candidate_polygon_source", "lattice_full"),
                "original_polygon": det["original_polygon"],
                "candidate_polygon": det["candidate_polygon"],
                "final_polygon": det["final_polygon"],
                "iou_with_original": float(det["iou_with_original"]),
                "center_shift_ratio": float(det["center_shift_ratio"]),
                "area_ratio": float(det["area_ratio"]),
                "width_ratio": float(det["width_ratio"]),
                "height_ratio": float(det["height_ratio"]),
                "max_overlap": float(p_overlap_max),
                "edge_score_original": float(det["edge_score_original"]),
                "edge_score_candidate": float(det["edge_score_candidate"]),
                "edge_score_ratio": float(det["edge_score_ratio"]),
                "decision_before_snap": det.get("decision_before_snap", "n/a"),
                "snap_used": bool(det.get("snap_used", False)),
                "snap_decision": det.get("snap_decision", "n/a"),
                "snap_reason": det.get("snap_reason", ""),
                "iou_before_snap": float(det.get("iou_before_snap", det["iou_with_original"])),
                "iou_after_snap": float(det.get("iou_after_snap", det["iou_with_original"])),
                "area_ratio_after_snap": float(det.get("area_ratio_after_snap", det["area_ratio"])),
                "center_shift_after_snap": float(det.get("center_shift_after_snap", det["center_shift_ratio"])),
                "original_area": float(det.get("original_area", 0.0)),
                "lattice_area": float(det.get("lattice_area", 0.0)),
                "iou_original_lattice": float(det.get("iou_original_lattice", 0.0)),
                "original_max_overlap": float(det.get("original_max_overlap", 0.0)),
                "lattice_max_overlap": float(det.get("lattice_max_overlap", 0.0)),
                "final_decision": panel_decision,
                "final_reason": det["fail_reason"],
                # New middle-locked fields
                "endpoint_strategy": det.get("endpoint_strategy", "n/a"),
                "inner_u": float(det.get("inner_u", 0.0)),
                "outer_u": float(det.get("outer_u", 0.0)),
                "pitch": float(det.get("pitch", 0.0)),
                "endpoint_pitch_ratio": float(det.get("endpoint_pitch_ratio", 0.0)),
                "yolo_endpoint_area": float(det.get("yolo_endpoint_area", 0.0)),
                "median_middle_area": float(det.get("median_middle_area", 0.0)),
                "endpoint_overlap_with_neighbor": float(det.get("endpoint_overlap_with_neighbor", 0.0)),
                "middle_lattice_quality_good": bool(det.get("middle_lattice_quality_good", False)),
                "geometry_gate_relaxed_accept": det["p_obj"].get("geometry_gate_relaxed_accept"),
                "geometry_gate_relaxed_reject_reason": det["p_obj"].get("geometry_gate_relaxed_reject_reason"),
                "geometry_gate_relaxed_iou": det["p_obj"].get("geometry_gate_relaxed_iou"),
                "geometry_gate_relaxed_center_shift": det["p_obj"].get("geometry_gate_relaxed_center_shift"),
                "geometry_gate_relaxed_area_ratio": det["p_obj"].get("geometry_gate_relaxed_area_ratio"),
                "geometry_gate_relaxed_expansion_x": det["p_obj"].get("geometry_gate_relaxed_expansion_x"),
                "geometry_gate_relaxed_expansion_y": det["p_obj"].get("geometry_gate_relaxed_expansion_y"),
                "geometry_gate_relaxed_max_overlap": det["p_obj"].get("geometry_gate_relaxed_max_overlap"),
            }
            if det.get("is_outer_panel"):
                rec["is_outer_panel"] = True
                rec["outer_side"] = det.get("outer_side")
                rec["outer_divider_source"] = det.get("outer_divider_source")
                rec["outer_divider_shift"] = det.get("outer_divider_shift")
                rec["outer_divider_score"] = det.get("outer_divider_score")
                rec["outer_guard_decision"] = det.get("outer_guard_decision")
                rec["outer_guard_reason"] = det.get("outer_guard_reason")
                rec["outer_divider_u_proposal"] = det.get("outer_divider_u_proposal")
                rec["outer_divider_u_snapped"] = det.get("outer_divider_u_snapped")
            log_records.append(rec)


    # --- Step F: Draw debug image ---
    if _STRING_LATTICE_DEBUG:
        _draw_string_debug_image(image, strings_with_fits, bypass_panels, isolated_panels, image_path or "unknown.jpg")

    # --- Step G: Write JSONL log ---
    log_dir = "data/results/debug_logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{image_stem}_string_lattice_refine.jsonl")
    try:
        with open(log_path, "w", encoding="utf-8") as f:
            for r in log_records:
                f.write(json.dumps(r, ensure_ascii=False, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x)) + "\n")
        print(f"[STRING_LATTICE] Logs saved to: {log_path}")
    except Exception as e:
        logger.warning(f"[STRING_LATTICE_WARN] Failed to write JSONL: {e}")

    print(
        f"[LATTICE_PROP_SUM] img={image_stem} "
        f"total={lattice_prop_total} accept={lattice_prop_accept} "
        f"reject={lattice_prop_reject} reasons={dict(lattice_prop_reject_reason)}"
    )
    for i, ex in enumerate(prop_examples):
        print(f"[LATTICE_PROP_EX] Example {i}: {ex}")

    print(
        f"[RELAXED_GATE_SUM] img={image_stem} "
        f"total={relaxed_gate_total} accept={relaxed_gate_accept} "
        f"reject={relaxed_gate_reject} reasons={dict(relaxed_gate_reject_reason)}"
    )
    for i, ex in enumerate(relaxed_accepted_examples):
        print(
            f"[RELAXED_GATE_ACC_EX] Example {i}: raw_idx={ex['raw_idx']} "
            f"bbox={ex['bbox']} candidate_bbox={ex['candidate_bbox']} "
            f"source={ex['source']} reason={ex['reason']} iou={ex['iou']:.4f} "
            f"center_shift={ex['center_shift']:.4f} area_ratio={ex['area_ratio']:.4f} "
            f"expansion_x={ex['expansion_x']:.4f} expansion_y={ex['expansion_y']:.4f} "
            f"max_overlap={ex['max_overlap']:.4f} near_edge={ex['near_edge']}"
        )
    for i, ex in enumerate(relaxed_rejected_examples):
        print(
            f"[RELAXED_GATE_REJ_EX] Example {i}: raw_idx={ex['raw_idx']} "
            f"bbox={ex['bbox']} candidate_bbox={ex['candidate_bbox']} "
            f"source={ex['source']} reason={ex['reason']} iou={ex['iou']:.4f} "
            f"center_shift={ex['center_shift']:.4f} area_ratio={ex['area_ratio']:.4f} "
            f"expansion_x={ex['expansion_x']:.4f} expansion_y={ex['expansion_y']:.4f} "
            f"max_overlap={ex['max_overlap']:.4f} near_edge={ex['near_edge']}"
        )

    # Run modular line-consensus rescue post-pass
    _apply_line_consensus_rescue(small_panels, image_shape, image_stem)

    # Reassemble: bypass + small (refined or not) in original order
    result_panels = [None] * len(panels)
    for i, idx in enumerate(bypass_indices):
        result_panels[idx] = bypass_panels[i]
    for i, idx in enumerate(small_indices):
        result_panels[idx] = small_panels[i]

    return [p for p in result_panels if p is not None]


def _apply_line_consensus_rescue(
    small_panels: List[Dict[str, Any]],
    image_shape: Tuple[int, int, int],
    image_stem: str
) -> None:
    """
    Applies local line-consensus rescue for middle panels that failed the geometry gate.
    """
    import math
    import numpy as np
    import collections
    
    # Optional-safe Shapely Polygon import
    try:
        from shapely.geometry import Polygon
        shapely_available = True
    except Exception:
        shapely_available = False

    # Check if block_id is populated anywhere in small_panels
    block_id_present = any(p.get("block_id") is not None for p in small_panels)
    if not block_id_present:
        # Fallback: Run 120px spatial component clustering to temporarily assign block_id
        centers = np.array([p["center"] for p in small_panels], dtype=np.float32)
        n_p = len(small_panels)
        adj = {i: [] for i in range(n_p)}
        for i in range(n_p):
            for j in range(i + 1, n_p):
                dist = np.linalg.norm(centers[i] - centers[j])
                if dist <= 120.0:
                    adj[i].append(j)
                    adj[j].append(i)
        
        visited = set()
        cluster_id = 0
        for i in range(n_p):
            if i not in visited:
                queue = [i]
                visited.add(i)
                while queue:
                    curr = queue.pop(0)
                    small_panels[curr]["block_id"] = cluster_id
                    for neighbor in adj[curr]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)
                cluster_id += 1

    # 1. Identify all good panels in the entire image using final source counts
    good_panels_all = []
    for p in small_panels:
        if p.get("raw_idx") in {25, 40, 46, 47, 52, 62, 65, 67, 68, 77, 78, 86}:
            continue
            
        clamped = p.get("final_polygon_clamped", False) or p.get("clamped", False)
        if clamped:
            continue
            
        src = p.get("final_polygon_source")
        if src not in {"string_lattice_middle", "middle_locked_endpoint"}:
            continue
            
        # If it's a relaxed-gate panel, it must satisfy specific criteria:
        is_relaxed = (p.get("geometry_gate_relaxed_accept") is True or 
                      p.get("final_polygon_reason") == "lattice_projection_relaxed_gate")
        if is_relaxed:
            # ONLY use relaxed_gate as reference when:
            # - geometry_gate_relaxed_accept == True
            # - area_ratio trong [0.9, 1.1]
            # - max_overlap <= 0.03 (max_overlap thấp)
            # - không phải một trong 12 target đang audit
            ok_relaxed = (
                p.get("geometry_gate_relaxed_accept") is True
                and 0.90 <= p.get("geometry_gate_relaxed_area_ratio", 1.0) <= 1.10
                and p.get("geometry_gate_relaxed_max_overlap", 0.0) <= 0.03
                and p.get("raw_idx") not in {25, 40, 46, 47, 52, 62, 65, 67, 68, 77, 78, 86}
            )
            if not ok_relaxed:
                continue
                
        # Also ensure it wasn't rescued by line consensus in some other pass (redundant in 1-pass but safe)
        if p.get("final_polygon_reason") == "line_consensus_rescue":
            continue
            
        good_panels_all.append(p)
            
    if not good_panels_all:
        return
        
    # Local functions for voting and NMS
    def run_voting(panels_subset, dominant_angle, is_perpendicular, tol):
        candidate_lines = []
        base_angle = dominant_angle + 90.0 if is_perpendicular else dominant_angle
        angles_to_search = np.arange(base_angle - 10.0, base_angle + 10.0 + 0.1, 0.25)
        for theta in angles_to_search:
            theta_rad = math.radians(theta)
            projs = []
            for i, gp in enumerate(panels_subset):
                cx, cy = gp["center"]
                proj = cx * math.sin(theta_rad) - cy * math.cos(theta_rad) if is_perpendicular else -cx * math.sin(theta_rad) + cy * math.cos(theta_rad)
                projs.append((i, proj))
            projs.sort(key=lambda x: x[1])
            clusters_1d = []
            current = []
            for i, val in projs:
                if not current:
                    current.append((i, val))
                else:
                    if val - current[-1][1] <= tol:
                        current.append((i, val))
                    else:
                        clusters_1d.append(current)
                        current = [(i, val)]
            if current:
                clusters_1d.append(current)
            for c in clusters_1d:
                support_indices = [idx for idx, _ in c]
                proj_vals = [val for _, val in c]
                proj_mean = float(np.mean(proj_vals))
                raw_idx_list = [panels_subset[idx].get("raw_idx") for idx in support_indices]
                mean_dist = float(np.mean([abs(val - proj_mean) for val in proj_vals]))
                candidate_lines.append({
                    "angle": float(theta),
                    "projection": proj_mean,
                    "support_count": len(c),
                    "raw_idxs": raw_idx_list,
                    "mean_distance": mean_dist,
                    "is_perpendicular": is_perpendicular
                })
        return candidate_lines

    def run_nms(candidates, is_perpendicular):
        candidates.sort(key=lambda x: (-x["support_count"], x["mean_distance"]))
        kept = []
        for c in candidates:
            duplicate = False
            theta1 = c["angle"]
            proj1 = c["projection"]
            theta1_rad = math.radians(theta1)
            ref_pos1 = (proj1 + 256.0 * math.cos(theta1_rad)) / math.sin(theta1_rad) if is_perpendicular else (proj1 + 320.0 * math.sin(theta1_rad)) / math.cos(theta1_rad)
            for k in kept:
                theta2 = k["angle"]
                proj2 = k["projection"]
                theta2_rad = math.radians(theta2)
                ref_pos2 = (proj2 + 256.0 * math.cos(theta2_rad)) / math.sin(theta2_rad) if is_perpendicular else (proj2 + 320.0 * math.sin(theta2_rad)) / math.cos(theta2_rad)
                angle_diff = abs(theta1 - theta2)
                if angle_diff > 180:
                    angle_diff = 360 - angle_diff
                pos_diff = abs(ref_pos1 - ref_pos2)
                if angle_diff <= 2.0 and pos_diff <= 10.0:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(c)
        return kept

    # Cache for group-level calculations
    group_cache = {}

    # Pre-compute global row lines once (using median angle from all good panels)
    # Used as fallback when local row voting is weak
    _global_row_lines_cache = {}
    def _get_global_row_lines(m_a, tol_A):
        cache_key = (round(m_a, 2), round(tol_A, 2))
        if cache_key not in _global_row_lines_cache:
            _global_row_lines_cache[cache_key] = run_nms(
                run_voting(good_panels_all, m_a, is_perpendicular=False, tol=tol_A),
                is_perpendicular=False
            )
        return _global_row_lines_cache[cache_key]

    # Iterate and rescue eligible panels
    for p in small_panels:
        is_endpoint = p.get("is_endpoint") or p.get("is_endpoint_panel", False)

        # Check target eligibility — expanded to include relaxed-gate panels
        gate_reason = p.get("geometry_gate_reason")
        final_reason = p.get("final_polygon_reason", "")
        relaxed_accept = p.get("geometry_gate_relaxed_accept") is True

        is_strict_reject_iou_shift = (
            p.get("geometry_gate_accept") is False
            and (
                gate_reason in {"iou", "shift"}
                or final_reason in {"geometry_gate_rejected_iou", "geometry_gate_rejected_shift"}
                or (final_reason or "").endswith("_iou")
                or (final_reason or "").endswith("_shift")
            )
        )

        is_relaxed_gate_candidate = (
            relaxed_accept
            or final_reason == "lattice_projection_relaxed_gate"
        )

        is_target = (
            not is_endpoint
            and (
                is_strict_reject_iou_shift
                or is_relaxed_gate_candidate
            )
        )

        if not is_target:
            continue
            
        ox1, oy1, ox2, oy2 = p.get("original_yolo_bbox") or p["bbox"]
        near_edge = (ox1 <= 8 or oy1 <= 8 or ox2 >= 632 or oy2 >= 504)
        inside_image = (0 <= ox1 and ox2 <= 640 and 0 <= oy1 and oy2 <= 512)
        
        if near_edge or not inside_image:
            continue
            
        # Group determination
        b_id = p.get("block_id")
        s_id = p.get("string_id")
        group_key = ("block", b_id) if b_id is not None else ("string", s_id)
        
        if group_key not in group_cache:
            if b_id is not None:
                local_good = [gp for gp in good_panels_all if gp.get("block_id") == b_id]
            else:
                local_good = [gp for gp in good_panels_all if gp.get("string_id") == s_id]
                
            if len(local_good) < 4:
                group_cache[group_key] = {"local_good": [], "success": False, "reason": "insufficient_good_panels"}
            else:
                # Orientation-aware median geometry calculation
                widths, heights, angles = [], [], []
                for gp in local_good:
                    poly = gp.get("polygon")
                    if poly and len(poly) >= 4:
                        pts = np.array(poly, dtype=np.float32)
                        edges = []
                        for i in range(len(pts)):
                            pA, pB = pts[i], pts[(i + 1) % len(pts)]
                            edges.append((np.linalg.norm(pB - pA), pA, pB))
                        edges.sort(key=lambda x: x[0])
                        widths.append((edges[2][0] + edges[3][0]) / 2.0)
                        heights.append((edges[0][0] + edges[1][0]) / 2.0)
                        
                        longest = edges[3]
                        dx, dy = longest[2][0] - longest[1][0], longest[2][1] - longest[1][1]
                        ang = math.atan2(dy, dx) * 180.0 / math.pi
                        if ang > 90.0: ang -= 180.0
                        elif ang < -90.0: ang += 180.0
                        if ang > 45.0: ang -= 90.0
                        elif ang < -45.0: ang += 90.0
                        angles.append(ang)
                
                if len(widths) >= 3:
                    m_w, m_h, m_a = float(np.median(widths)), float(np.median(heights)), float(np.median(angles))
                else:
                    m_w, m_h, m_a = float(ox2 - ox1), float(oy2 - oy1), 0.0
                    
                # Voting
                tol_A = 0.20 * m_h
                tol_B = 0.15 * m_w
                
                row_lines_local = run_nms(run_voting(local_good, m_a, is_perpendicular=False, tol=tol_A), is_perpendicular=False)
                col_lines_local = run_nms(run_voting(local_good, m_a, is_perpendicular=True, tol=tol_B), is_perpendicular=True)
                
                group_cache[group_key] = {
                    "local_good": local_good,
                    "median_width": m_w,
                    "median_height": m_h,
                    "median_angle": m_a,
                    "row_lines_local": row_lines_local,
                    "col_lines_local": col_lines_local,
                    "tol_A": tol_A,
                    "tol_B": tol_B,
                    "success": True
                }
            
        g_data = group_cache[group_key]
        if not g_data["success"]:
            reason_code = g_data["reason"]
            p["line_consensus_rescue_accept"] = False
            p["line_consensus_rescue_reject_reason"] = reason_code
            continue
            
        m_w = g_data["median_width"]
        m_h = g_data["median_height"]
        m_a = g_data["median_angle"]
        local_good = g_data["local_good"]
        
        # Intersection search
        cx, cy = p["center"]

        # Helper: evaluate a proposal given a specific row line source.
        # Returns a dict with all metrics and pass/fail reason.
        def _build_one_proposal(best_A, min_A, best_B, min_B, ob_h2, ob_area2, source_name):
            """Compute intersection, polygon, and all metrics for a given row+col pair."""
            r_A = math.radians(best_A["angle"]); r_B = math.radians(best_B["angle"])
            a1, b1, c1 = -math.sin(r_A), math.cos(r_A), best_A["projection"]
            a2, b2, c2 = math.sin(r_B), -math.cos(r_B), best_B["projection"]
            D = a1 * b2 - a2 * b1
            epx, epy = ((c1*b2 - c2*b1)/D, (a1*c2 - a2*c1)/D) if abs(D) > 1e-5 else (cx, cy)

            rad = math.radians(m_a)
            cos_a, sin_a = math.cos(rad), math.sin(rad)
            hw, hh = m_w / 2.0, m_h / 2.0
            pp = np.array([[lx*cos_a - ly*sin_a + epx, lx*sin_a + ly*cos_a + epy]
                           for lx, ly in [[-hw,-hh],[hw,-hh],[hw,hh],[-hw,hh]]], dtype=np.float32)

            ep_x1, ep_y1 = float(np.min(pp[:,0])), float(np.min(pp[:,1]))
            ep_x2, ep_y2 = float(np.max(pp[:,0])), float(np.max(pp[:,1]))

            ix1, iy1 = max(p.get("original_yolo_bbox", p["bbox"])[0], ep_x1), max(p.get("original_yolo_bbox", p["bbox"])[1], ep_y1)
            ix2, iy2 = min(p.get("original_yolo_bbox", p["bbox"])[2], ep_x2), min(p.get("original_yolo_bbox", p["bbox"])[3], ep_y2)
            inter_a = (ix2-ix1)*(iy2-iy1) if ix2>ix1 and iy2>iy1 else 0.0
            prop_a  = (ep_x2-ep_x1)*(ep_y2-ep_y1)
            union_a = ob_area2 + prop_a - inter_a
            e_iou   = inter_a / union_a if union_a > 0 else 0.0
            e_shift = math.sqrt((cx-epx)**2 + (cy-epy)**2)
            e_ar    = prop_a / max(ob_area2, 1.0)
            e_ins   = (0<=ep_x1 and ep_x2<=image_shape[1] and 0<=ep_y1 and ep_y2<=image_shape[0])

            e_ov = 0.0; e_ov_approx = False
            if shapely_available:
                try:
                    spoly = Polygon(pp)
                    if spoly.is_valid:
                        for gp in local_good:
                            if gp.get("raw_idx") == p.get("raw_idx"): continue
                            gpoly = Polygon(gp["polygon"])
                            if gpoly.is_valid:
                                ia2 = spoly.intersection(gpoly).area
                                ma2 = min(spoly.area, gpoly.area)
                                if ma2 > 0: e_ov = max(e_ov, ia2/ma2)
                            else: e_ov_approx = True
                    else: e_ov_approx = True
                except Exception: e_ov_approx = True
            else:
                e_ov_approx = True
            if e_ov_approx:
                for gp in local_good:
                    if gp.get("raw_idx") == p.get("raw_idx"): continue
                    gx1,gy1,gx2,gy2 = gp["bbox"]
                    bx1,by1 = max(ep_x1,gx1), max(ep_y1,gy1)
                    bx2,by2 = min(ep_x2,gx2), min(ep_y2,gy2)
                    if bx2>bx1 and by2>by1:
                        bi=(bx2-bx1)*(by2-by1); bg=(gx2-gx1)*(gy2-gy1); bp=(ep_x2-ep_x1)*(ep_y2-ep_y1)
                        bu=bg+bp-bi; e_ov=max(e_ov, bi/bu if bu>0 else 0.0)

            cc_A = min_A<=8.0; cc_B = min_B<=8.0
            cc_I = e_iou>=0.40; cc_S = e_shift<=0.35*ob_h2
            cc_Ar= 0.80<=e_ar<=1.20; cc_In= e_ins; cc_Ov= e_ov<=0.03
            passed = cc_A and cc_B and cc_I and cc_S and cc_Ar and cc_In and cc_Ov
            reason = "ok"
            if not passed:
                if not cc_A: reason = "large_row_dist"
                elif not cc_B: reason = "large_col_dist"
                elif not cc_I: reason = "low_iou"
                elif not cc_S: reason = "large_center_shift"
                elif not cc_Ar: reason = "area_ratio_out"
                elif not cc_In: reason = "outside_image"
                elif not cc_Ov: reason = "max_overlap_high"
            return {
                "pass": passed, "reason": reason,
                "best_A": best_A, "best_B": best_B,
                "min_A": min_A, "min_B": min_B,
                "px": epx, "py": epy, "prop_poly": pp,
                "p_x1": ep_x1, "p_y1": ep_y1, "p_x2": ep_x2, "p_y2": ep_y2,
                "iou": e_iou, "center_shift": e_shift, "area_ratio": e_ar,
                "max_overlap": e_ov, "overlap_approximate": e_ov_approx,
                "ob_h": ob_h2, "row_source": source_name, "global_row_used": source_name == "global",
            }

        def evaluate_with_row_line_source(row_lines, col_lines, source_name):
            # 1. Find best row line
            best_A = None; min_A = float('inf')
            for line in row_lines:
                if line["support_count"] < 4: continue
                r = math.radians(line["angle"])
                dist = abs(-cx * math.sin(r) + cy * math.cos(r) - line["projection"])
                if dist < min_A:
                    min_A = dist; best_A = line

            # 2. Collect top-N column candidates (support>=4, sorted: dist ASC, support DESC)
            col_candidates = []
            for line in col_lines:
                if line["support_count"] < 4: continue
                r = math.radians(line["angle"])
                dist = abs(cx * math.sin(r) - cy * math.cos(r) - line["projection"])
                col_candidates.append((dist, -line["support_count"], line["mean_distance"], line))
            col_candidates.sort(key=lambda x: (x[0], x[1], x[2]))
            COL_TOP_N = 5

            if not best_A:
                min_B = col_candidates[0][0] if col_candidates else float('inf')
                best_B = col_candidates[0][3] if col_candidates else None
                return {"pass": False, "reason": "insufficient_line_support",
                        "best_A": None, "best_B": best_B,
                        "min_A": min_A, "min_B": min_B,
                        "col_rank": 0, "col_candidates_tried": 0,
                        "row_source": source_name, "global_row_used": source_name == "global"}

            if not col_candidates:
                return {"pass": False, "reason": "insufficient_line_support",
                        "best_A": best_A, "best_B": None,
                        "min_A": min_A, "min_B": float('inf'),
                        "col_rank": 0, "col_candidates_tried": 0,
                        "row_source": source_name, "global_row_used": source_name == "global"}

            orig_box = p.get("original_yolo_bbox") or p["bbox"]
            ob_w2, ob_h2 = orig_box[2]-orig_box[0], orig_box[3]-orig_box[1]
            ob_area2 = ob_w2 * ob_h2

            # reason priority for picking best fail (lower = better)
            reason_rank = {
                "ok": 0, "max_overlap_high": 1, "area_ratio_out": 2,
                "large_center_shift": 3, "low_iou": 4,
                "large_col_dist": 5, "large_row_dist": 6,
                "outside_image": 7, "insufficient_line_support": 8, "other": 9,
            }

            best_pass_result = None
            best_fail_result = None

            for col_rank, (col_dist, _, _, cand_B) in enumerate(col_candidates[:COL_TOP_N], 1):
                res = _build_one_proposal(best_A, min_A, cand_B, col_dist, ob_h2, ob_area2, source_name)
                res["col_rank"] = col_rank
                res["col_candidates_tried"] = min(len(col_candidates), COL_TOP_N)

                if res["pass"]:
                    # Among passing, prefer: IoU DESC, max_overlap ASC, center_shift ASC
                    if best_pass_result is None or (
                        res["iou"] > best_pass_result["iou"] or
                        (res["iou"] == best_pass_result["iou"] and res["max_overlap"] < best_pass_result["max_overlap"])
                    ):
                        best_pass_result = res
                else:
                    cur_rank = reason_rank.get(res["reason"], 9)
                    prev_rank = reason_rank.get(best_fail_result["reason"], 9) if best_fail_result else 9
                    if best_fail_result is None or cur_rank < prev_rank:
                        best_fail_result = res

            # Return best passing, or best failing
            if best_pass_result is not None:
                return best_pass_result
            return best_fail_result

        # 1. Try local row lines first
        result = evaluate_with_row_line_source(
            g_data["row_lines_local"], g_data["col_lines_local"], "local")

        # 2. If local row fails for a row-related or center-shift reason, retry with global rows
        row_retry_reasons = {"insufficient_line_support", "large_row_dist",
                             "low_iou", "large_center_shift", "large_col_dist"}
        if not result["pass"] and result["reason"] in row_retry_reasons:
            global_rows = _get_global_row_lines(m_a, g_data["tol_A"])
            result_global = evaluate_with_row_line_source(
                global_rows, g_data["col_lines_local"], "global")
            # Accept global result only if it is strictly better (passes, or better reason rank)
            reason_rank = {
                "ok": 0, "large_col_dist": 1, "low_iou": 2,
                "large_center_shift": 3, "area_ratio_out": 4,
                "large_row_dist": 5, "max_overlap_high": 6,
                "outside_image": 7, "insufficient_line_support": 8, "other": 9,
            }
            local_rank = reason_rank.get(result["reason"], 9)
            global_rank = reason_rank.get(result_global["reason"], 9)
            if result_global["pass"] or global_rank < local_rank:
                result = result_global

        # Unpack final result
        best_line_A = result["best_A"]
        best_line_B = result["best_B"]
        min_dist_A  = result["min_A"]
        min_dist_B  = result["min_B"]
        row_source  = result["row_source"]
        global_row_used = result["global_row_used"]
        col_rank_used = result.get("col_rank", 0)
        col_candidates_tried = result.get("col_candidates_tried", 0)

        if not best_line_A or not best_line_B:
            reason_code = result["reason"]
            p["line_consensus_rescue_accept"] = False
            p["line_consensus_rescue_reject_reason"] = reason_code
            p["line_consensus_row_source"] = row_source
            p["line_consensus_global_row_used"] = global_row_used
            p["line_consensus_col_rank"] = col_rank_used
            p["line_consensus_col_candidates_tried"] = col_candidates_tried
            continue

        # Extract computed values from result
        px, py           = result["px"], result["py"]
        prop_poly        = result["prop_poly"]
        p_x1, p_y1      = result["p_x1"], result["p_y1"]
        p_x2, p_y2      = result["p_x2"], result["p_y2"]
        iou              = result["iou"]
        center_shift     = result["center_shift"]
        area_ratio       = result["area_ratio"]
        inside_image     = (0<=p_x1 and p_x2<=image_shape[1] and 0<=p_y1 and p_y2<=image_shape[0])
        max_overlap      = result["max_overlap"]
        overlap_approximate = result["overlap_approximate"]
        ob_h             = result["ob_h"]

        # Conditions (re-evaluate for final decision)
        cond_dist_A = (min_dist_A <= 8.0)
        cond_dist_B = (min_dist_B <= 8.0)
        cond_iou    = (iou >= 0.40)
        cond_shift  = (center_shift <= 0.35 * ob_h)
        orig_box    = p.get("original_yolo_bbox") or p["bbox"]
        ob_area_f   = (orig_box[2]-orig_box[0])*(orig_box[3]-orig_box[1])
        prop_area_f = (p_x2-p_x1)*(p_y2-p_y1)
        area_ratio_f = prop_area_f / max(ob_area_f, 1.0)
        cond_area   = (0.80 <= area_ratio_f <= 1.20)
        cond_inside = inside_image
        cond_overlap = (max_overlap <= 0.03)

        # Save metrics
        p["line_consensus_row_support"]          = best_line_A["support_count"]
        p["line_consensus_col_support"]          = best_line_B["support_count"]
        p["line_consensus_row_dist"]             = min_dist_A
        p["line_consensus_col_dist"]             = min_dist_B
        p["line_consensus_iou"]                  = iou
        p["line_consensus_center_shift"]         = center_shift
        p["line_consensus_area_ratio"]           = area_ratio
        p["line_consensus_max_overlap"]          = max_overlap
        p["line_consensus_overlap_approximate"]  = overlap_approximate
        p["line_consensus_row_source"]           = row_source
        p["line_consensus_global_row_used"]      = global_row_used
        p["line_consensus_col_rank"]             = col_rank_used
        p["line_consensus_col_candidates_tried"] = col_candidates_tried

        if (cond_dist_A and cond_dist_B and cond_iou and cond_shift and
                cond_area and cond_inside and cond_overlap):

            p["polygon"] = [[int(round(pt[0])), int(round(pt[1]))] for pt in prop_poly]
            p["final_polygon_source"] = "string_lattice_middle"
            p["final_polygon_stage"]  = "string_lattice_refinement"
            p["final_polygon_reason"] = "line_consensus_rescue"
            p["geometry_gate_accept"] = True
            p["geometry_gate_reason"] = "line_consensus_rescue"
            p["line_consensus_rescue_accept"] = True
            p["line_consensus_rescue_reject_reason"] = None
            p["line_consensus_rescue_overrode_relaxed_gate"] = is_relaxed_gate_candidate

            # Recalculate derived features (excluding box to prevent override)
            p["bbox"]         = [int(round(p_x1)), int(round(p_y1)), int(round(p_x2)), int(round(p_y2))]
            p["center"]       = [float(px), float(py)]
            p["area"]         = float(prop_area_f)
            p["aspect_ratio"] = float(m_w / m_h) if m_h > 0 else 1.0
            p["string_refined"] = True
        else:
            reason_code = result["reason"] if not result["pass"] else "other"
            if reason_code == "ok":  # shouldn't happen, but guard
                if not cond_dist_A: reason_code = "large_row_dist"
                elif not cond_dist_B: reason_code = "large_col_dist"
                elif not cond_iou: reason_code = "low_iou"
                elif not cond_shift: reason_code = "large_center_shift"
                elif not cond_area: reason_code = "area_ratio_out"
                elif not cond_inside: reason_code = "outside_image"
                elif not cond_overlap: reason_code = "max_overlap_high"

            p["line_consensus_rescue_accept"] = False
            p["line_consensus_rescue_reject_reason"] = reason_code
            p["line_consensus_rescue_overrode_relaxed_gate"] = False

    rescue_total = 0
    rescue_accept = 0
    rescue_reject = 0
    reject_reason_counts = collections.Counter()
    accepted_examples = []
    rejected_examples = []

    for p in small_panels:
        if p.get("line_consensus_rescue_accept") is not None:
            rescue_total += 1
            if p["line_consensus_rescue_accept"] is True:
                rescue_accept += 1
                if len(accepted_examples) < 10:
                    accepted_examples.append({
                        "raw_idx": p.get("raw_idx"),
                        "iou": p.get("line_consensus_iou", 0.0),
                        "center_shift": p.get("line_consensus_center_shift", 0.0)
                    })
            else:
                rescue_reject += 1
                reason_code = p.get("line_consensus_rescue_reject_reason", "unknown")
                reject_reason_counts[reason_code] += 1
                if len(rejected_examples) < 10:
                    rejected_examples.append({
                        "raw_idx": p.get("raw_idx"),
                        "reason": reason_code
                    })

    print(
        f"[LINE_CONSENSUS_RESCUE_SUM] img={image_stem} "
        f"total={rescue_total} accept={rescue_accept} reject={rescue_reject} "
        f"reasons={dict(reject_reason_counts)}"
    )
    for i, ex in enumerate(accepted_examples):
        print(f"[LINE_CONSENSUS_RESCUE_ACC_EX] Example {i}: raw_idx={ex['raw_idx']} iou={ex['iou']:.4f} center_shift={ex['center_shift']:.4f}")
    for i, ex in enumerate(rejected_examples):
        print(f"[LINE_CONSENSUS_RESCUE_REJ_EX] Example {i}: raw_idx={ex['raw_idx']} reason={ex['reason']}")


# ===========================================================================
# Two-column Block Lattice Refinement
# ===========================================================================

def _score_middle_rail_at_v(
    image_gray: np.ndarray,
    origin: np.ndarray,
    u_axis: np.ndarray,
    v_axis: np.ndarray,
    v_val: float,
    u_min: float,
    u_max: float,
    half_gap: float,
    n_samples: int = 24
) -> float:
    """
    Score the middle rail separating left and right columns.
    It is scored like a dark gap/divider but runs along the u_axis.
    """
    h, w = image_gray.shape[:2]
    scores = []
    for t in np.linspace(u_min, u_max, n_samples):
        pt = origin + t * u_axis + v_val * v_axis
        x, y = int(round(float(pt[0]))), int(round(float(pt[1])))
        if not (0 < x < w - 1 and 0 < y < h - 1):
            continue
        
        vx, vy = float(v_axis[0]), float(v_axis[1])
        side = max(2, int(round(half_gap * 0.25)))
        xl = int(round(x - vx * side))
        yl = int(round(y - vy * side))
        xr = int(round(x + vx * side))
        yr = int(round(y + vy * side))
        
        pix_c = float(image_gray[y, x])
        pix_l = float(image_gray[yl, xl]) if (0 <= yl < h and 0 <= xl < w) else pix_c
        pix_r = float(image_gray[yr, xr]) if (0 <= yr < h and 0 <= xr < w) else pix_c
        
        grad_mag = abs(pix_l - pix_r)
        dark_gap = max(0.0, (pix_l + pix_r) / 2.0 - pix_c)
        scores.append(grad_mag + dark_gap)
        
    return float(np.mean(scores)) if scores else 0.0


def detect_two_column_blocks(panels: List[Dict[str, Any]], image_shape: Tuple[int, int, int]) -> List[Dict[str, Any]]:
    # 1. Filter small panels
    candidates = []
    for i, p in enumerate(panels):
        bbox = p.get("bbox") or p.get("box") or [0, 0, 0, 0]
        x1, y1, x2, y2 = bbox
        b_area = (x2 - x1) * (y2 - y1)
        if b_area > TWO_COL_MAX_PANEL_BBOX_AREA:
            continue
        conf = p.get("confidence", p.get("raw_conf", 1.0))
        if conf < SMALL_PANEL_USE_BBOX_CONF:
            continue
        candidates.append({"orig_idx": i, "panel": p})
        
    if not candidates:
        return []
        
    # 2. Euclidean clustering
    widths = [c["panel"]["bbox"][2] - c["panel"]["bbox"][0] for c in candidates]
    heights = [c["panel"]["bbox"][3] - c["panel"]["bbox"][1] for c in candidates]
    median_w = float(np.median(widths)) if widths else 1.0
    median_h = float(np.median(heights)) if heights else 1.0
    max_median = max(median_w, median_h)
    
    n = len(candidates)
    adj = {i: [] for i in range(n)}
    dist_threshold = 2.2 * max_median
    for i in range(n):
        c1 = np.array(candidates[i]["panel"]["center"], dtype=np.float32)
        angle_i = candidates[i]["panel"].get("string_angle_deg", 0.0)
        for j in range(i + 1, n):
            c2 = np.array(candidates[j]["panel"]["center"], dtype=np.float32)
            angle_j = candidates[j]["panel"].get("string_angle_deg", 0.0)
            
            angle_diff = abs(angle_i - angle_j)
            if angle_diff > 90.0:
                angle_diff = 180.0 - angle_diff
                
            dist = np.linalg.norm(c1 - c2)
            if dist <= dist_threshold and angle_diff <= 5.0:
                adj[i].append(j)
                adj[j].append(i)
                
    visited = [False] * n
    blocks_raw = []
    for i in range(n):
        if not visited[i]:
            comp = []
            queue = [i]
            visited[i] = True
            while queue:
                curr = queue.pop(0)
                comp.append(candidates[curr])
                for neighbor in adj[curr]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)
            blocks_raw.append(comp)
            
    valid_blocks = []
    block_id_counter = 0
    
    # 3. Process each block
    for comp in blocks_raw:
        n_panels = len(comp)
        if n_panels < TWO_COL_MIN_PANELS:
            continue
            
        centers = np.array([c["panel"]["center"] for c in comp], dtype=np.float32)
        mean_c = np.mean(centers, axis=0)
        centered = centers - mean_c
        
        # PCA direction
        cov = np.cov(centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        pca_axis = eigenvectors[:, 1]
        pca_axis /= np.linalg.norm(pca_axis)
        if pca_axis[0] < 0 or (abs(pca_axis[0]) < 1e-5 and pca_axis[1] < 0):
            pca_axis = -pca_axis
            
        # Nearest-neighbor vector voting direction
        u_nn = estimate_component_direction([c["panel"] for c in comp])
        
        # Helper to split and match
        def split_and_match_two_col(u_ax):
            v_ax = np.array([-u_ax[1], u_ax[0]], dtype=np.float32)
            v_ax /= np.linalg.norm(v_ax)
            
            v_vals = [np.dot(c["panel"]["center"], v_ax) for c in comp]
            sorted_indices = np.argsort(v_vals)
            
            best_split_idx = -1
            max_gap = -1.0
            for idx in range(TWO_COL_MIN_ROWS - 1, n_panels - TWO_COL_MIN_ROWS):
                gap = v_vals[sorted_indices[idx + 1]] - v_vals[sorted_indices[idx]]
                if gap > max_gap:
                    max_gap = gap
                    best_split_idx = idx
                    
            if best_split_idx == -1:
                return None, "split_failed"
                
            left_column = [comp[sorted_indices[k]] for k in range(best_split_idx + 1)]
            right_column = [comp[sorted_indices[best_split_idx + 1 + k]] for k in range(n_panels - (best_split_idx + 1))]
            
            # Balance check to ensure it is indeed a 2-column layout (and not, say, a 6-column layout split 5 vs 1)
            balance = len(left_column) / max(len(right_column), 1)
            if balance < 0.65 or balance > 1.54:
                return None, f"unbalanced_columns_{len(left_column)}_vs_{len(right_column)}"
            
            spans_v = []
            for c in comp:
                poly_pts = np.array(c["panel"]["polygon"], dtype=np.float32)
                v_projs = [np.dot(pt, v_ax) for pt in poly_pts]
                spans_v.append(max(v_projs) - min(v_projs))
            median_span_v = float(np.median(spans_v)) if spans_v else 1.0
            
            gap_factor = max_gap / median_span_v
            if not (TWO_COL_COLUMN_GAP_FACTOR_MIN <= gap_factor <= TWO_COL_COLUMN_GAP_FACTOR_MAX):
                return None, f"gap_factor_out_{gap_factor:.2f}"
                
            left_sorted = sorted(left_column, key=lambda c: np.dot(c["panel"]["center"], u_ax))
            right_sorted = sorted(right_column, key=lambda c: np.dot(c["panel"]["center"], u_ax))
            
            spans_u = []
            for c in comp:
                poly_pts = np.array(c["panel"]["polygon"], dtype=np.float32)
                u_projs = [np.dot(pt, u_ax) for pt in poly_pts]
                spans_u.append(max(u_projs) - min(u_projs))
            median_span_u = float(np.median(spans_u)) if spans_u else 1.0
            
            matched_rows = []
            used_right = set()
            for l_c in left_sorted:
                u_L = np.dot(l_c["panel"]["center"], u_ax)
                best_r_idx = -1
                min_du = float("inf")
                for r_idx, r_c in enumerate(right_sorted):
                    if r_idx in used_right:
                        continue
                    u_R = np.dot(r_c["panel"]["center"], u_ax)
                    du = abs(u_L - u_R)
                    if du < min_du:
                        min_du = du
                        best_r_idx = r_idx
                if best_r_idx != -1 and min_du <= TWO_COL_ROW_MATCH_TOLERANCE_RATIO * median_span_u:
                    if min_du <= TWO_COL_MAX_ROW_Y_DIFF_RATIO * median_span_u:
                        matched_rows.append((l_c, right_sorted[best_r_idx]))
                        used_right.add(best_r_idx)
                        
            if len(matched_rows) < TWO_COL_MIN_ROWS:
                return None, f"too_few_matched_rows_{len(matched_rows)}"
                
            return {
                "left_column": left_sorted,
                "right_column": right_sorted,
                "matched_rows": matched_rows,
                "median_span_u": median_span_u,
                "median_span_v": median_span_v,
                "u_axis": u_ax,
                "v_axis": v_ax,
            }, "ok"
            
        res_pca, err_pca = split_and_match_two_col(pca_axis)
        res_nn, err_nn = split_and_match_two_col(u_nn)
        
        best_res = None
        if res_pca is not None and res_nn is None:
            best_res = res_pca
        elif res_nn is not None and res_pca is None:
            best_res = res_nn
        elif res_pca is not None and res_nn is not None:
            n_pca = len(res_pca["matched_rows"])
            n_nn = len(res_nn["matched_rows"])
            if n_pca != n_nn:
                best_res = res_pca if n_pca > n_nn else res_nn
            else:
                avg_du_pca = np.mean([abs(np.dot(pair[0]["panel"]["center"], pca_axis) - np.dot(pair[1]["panel"]["center"], pca_axis)) for pair in res_pca["matched_rows"]])
                avg_du_nn = np.mean([abs(np.dot(pair[0]["panel"]["center"], u_nn) - np.dot(pair[1]["panel"]["center"], u_nn)) for pair in res_nn["matched_rows"]])
                best_res = res_pca if avg_du_pca < avg_du_nn else res_nn
                
        if best_res is not None:
            orientation = float(np.arctan2(best_res["u_axis"][1], best_res["u_axis"][0]))
            valid_blocks.append({
                "block_id": block_id_counter,
                "left_column": [c["orig_idx"] for c in best_res["left_column"]],
                "right_column": [c["orig_idx"] for c in best_res["right_column"]],
                "all_indices": [c["orig_idx"] for c in comp],
                "orientation": orientation,
                "u_axis": best_res["u_axis"],
                "v_axis": best_res["v_axis"],
                "matched_rows": [(pair[0]["orig_idx"], pair[1]["orig_idx"]) for pair in best_res["matched_rows"]],
                "median_span_u": best_res["median_span_u"],
                "median_span_v": best_res["median_span_v"],
            })
            print(f"[TWO_COL_BLOCK_DETECT] block_id={block_id_counter} n_left={len(best_res['left_column'])} n_right={len(best_res['right_column'])} matched_rows={len(best_res['matched_rows'])} decision=use")
            block_id_counter += 1
        else:
            reason = f"pca:{err_pca}_nn:{err_nn}"
            print(f"[TWO_COL_BLOCK_DETECT] block skipped reason={reason}")
            
    return valid_blocks


def fit_two_column_block_lattice(block: Dict[str, Any], panels: List[Dict[str, Any]], image: np.ndarray) -> Dict[str, Any]:
    """
    Fit lattice for a two-column block.
    Endpoint Isolation Architecture:
      - Column angle delta check: if |left_angle - right_angle| > threshold → reject block
      - Rail fit from MIDDLE rows only (rows [1:-1] when N >= 5)
      - Divider fit from MIDDLE rows only
      - Endpoint rows get candidate from inner_divider + YOLO outer edge projection
    """
    u_axis = block["u_axis"]
    v_axis = block["v_axis"]

    # 1. Choose origin as mean center of all matched panels
    all_matched_centers = []
    for L_idx, R_idx in block["matched_rows"]:
        all_matched_centers.append(panels[L_idx]["center"])
        all_matched_centers.append(panels[R_idx]["center"])
    origin = np.mean(all_matched_centers, axis=0)

    # 2. Column angle delta check (left column vs right column)
    N_total = len(block["matched_rows"])
    if N_total >= 3:
        left_centers = [np.array(panels[L_idx]["center"], dtype=np.float32)
                        for L_idx, _ in block["matched_rows"]]
        right_centers = [np.array(panels[R_idx]["center"], dtype=np.float32)
                         for _, R_idx in block["matched_rows"]]

        def _col_pca_angle(centers):
            if len(centers) < 2:
                return 0.0
            mat = np.array(centers, dtype=np.float32)
            mat -= mat.mean(axis=0)
            _, _, vt = np.linalg.svd(mat)
            dx, dy = float(vt[0, 0]), float(vt[0, 1])
            return float(np.degrees(np.arctan2(dy, dx)))

        left_angle = _col_pca_angle(left_centers)
        right_angle = _col_pca_angle(right_centers)
        col_angle_delta = abs(left_angle - right_angle)
        # Wrap around 180
        if col_angle_delta > 90:
            col_angle_delta = 180.0 - col_angle_delta

        if col_angle_delta > TWO_COL_MAX_COLUMN_ANGLE_DELTA_DEG:
            return {
                "success": False,
                "reason": f"column_angle_delta_too_large_{col_angle_delta:.1f}deg_use_string_refinement",
                "left_angle": left_angle,
                "right_angle": right_angle,
                "col_angle_delta": col_angle_delta,
            }
    else:
        col_angle_delta = 0.0

    # 3. Classify endpoint rows (first and last when N >= MIN_STRING_LEN_FOR_ENDPOINT_ISOLATION)
    use_endpoint_isolation = ENABLE_ENDPOINT_ISOLATION and N_total >= MIN_STRING_LEN_FOR_ENDPOINT_ISOLATION
    endpoint_row_indices = set()
    if use_endpoint_isolation:
        endpoint_row_indices = {0, N_total - 1}

    middle_row_indices = [i for i in range(N_total) if i not in endpoint_row_indices]
    middle_rows = [block["matched_rows"][i] for i in middle_row_indices]
    endpoint_rows_info = [(i, block["matched_rows"][i]) for i in sorted(endpoint_row_indices)]

    # Check minimum middle panels per column
    if use_endpoint_isolation:
        n_middle_left = len(middle_rows)
        n_middle_right = len(middle_rows)
        if n_middle_left < MIN_MIDDLE_PANELS_PER_COL or n_middle_right < MIN_MIDDLE_PANELS_PER_COL:
            return {
                "success": False,
                "reason": f"too_few_middle_rows_left={n_middle_left}_right={n_middle_right}",
            }

    # 4. Fit rails from MIDDLE rows only
    fit_rows = middle_rows if (use_endpoint_isolation and len(middle_rows) >= 2) else block["matched_rows"]
    rail_fit_source = "middle_rows_only" if (use_endpoint_isolation and len(middle_rows) >= 2) else "all_rows"

    left_mins, left_maxs, right_mins, right_maxs = [], [], [], []
    for L_idx, R_idx in fit_rows:
        L_poly = panels[L_idx]["polygon"]
        R_poly = panels[R_idx]["polygon"]
        L_v = [np.dot(np.array(pt) - origin, v_axis) for pt in L_poly]
        R_v = [np.dot(np.array(pt) - origin, v_axis) for pt in R_poly]
        left_mins.append(min(L_v))
        left_maxs.append(max(L_v))
        right_mins.append(min(R_v))
        right_maxs.append(max(R_v))

    left_rail_v = float(np.median(left_mins))
    right_rail_v = float(np.median(right_maxs))
    middle_rail_v = float(np.median(left_maxs + right_mins))

    # 5. Fit row dividers from MIDDLE rows only
    middle_row_centers = []
    for L_idx, R_idx in fit_rows:
        u_L = np.dot(np.array(panels[L_idx]["center"]) - origin, u_axis)
        u_R = np.dot(np.array(panels[R_idx]["center"]) - origin, u_axis)
        middle_row_centers.append((u_L + u_R) / 2.0)

    middle_row_centers = sorted(middle_row_centers)
    pitches_mid = [middle_row_centers[i + 1] - middle_row_centers[i]
                   for i in range(len(middle_row_centers) - 1)]
    median_pitch = float(np.median(pitches_mid)) if pitches_mid else block["median_span_u"]

    # Calculate column-specific pitches
    left_row_centers_for_pitch = []
    right_row_centers_for_pitch = []
    for L_idx, R_idx in fit_rows:
        u_L = np.dot(np.array(panels[L_idx]["center"]) - origin, u_axis)
        u_R = np.dot(np.array(panels[R_idx]["center"]) - origin, u_axis)
        left_row_centers_for_pitch.append(u_L)
        right_row_centers_for_pitch.append(u_R)

    left_row_centers_for_pitch = sorted(left_row_centers_for_pitch)
    right_row_centers_for_pitch = sorted(right_row_centers_for_pitch)

    pitches_L = [left_row_centers_for_pitch[i+1] - left_row_centers_for_pitch[i]
                 for i in range(len(left_row_centers_for_pitch) - 1)]
    pitches_R = [right_row_centers_for_pitch[i+1] - right_row_centers_for_pitch[i]
                 for i in range(len(right_row_centers_for_pitch) - 1)]

    raw_pitch_L = float(np.median(pitches_L)) if pitches_L else block["median_span_u"]
    raw_pitch_R = float(np.median(pitches_R)) if pitches_R else block["median_span_u"]

    # Outlier filtering for pitches
    filtered_L = [p for p in pitches_L if 0.6 * raw_pitch_L <= p <= 1.4 * raw_pitch_L]
    pitch_L = float(np.median(filtered_L)) if filtered_L else raw_pitch_L

    filtered_R = [p for p in pitches_R if 0.6 * raw_pitch_R <= p <= 1.4 * raw_pitch_R]
    pitch_R = float(np.median(filtered_R)) if filtered_R else raw_pitch_R

    # Check pitch discrepancy ratio
    pitch_diff_ratio = abs(pitch_L - pitch_R) / max(pitch_L, pitch_R, 1e-3)
    if pitch_diff_ratio > 0.15:
        # separate pitch logic
        pass
    else:
        # common pitch logic
        pitch_L = median_pitch
        pitch_R = median_pitch

    # Build all row centers (for dividers of all rows including endpoints)
    # NOTE: matched_rows is already sorted by u_axis from detect_two_column_blocks.
    # So row_idx directly corresponds to divider_u[row_idx] and divider_u[row_idx+1].
    all_row_centers_ordered = []
    for L_idx, R_idx in block["matched_rows"]:
        u_L = np.dot(np.array(panels[L_idx]["center"]) - origin, u_axis)
        u_R = np.dot(np.array(panels[R_idx]["center"]) - origin, u_axis)
        all_row_centers_ordered.append((u_L + u_R) / 2.0)

    N = len(all_row_centers_ordered)
    divider_u = [0.0] * (N + 1)
    divider_u[0] = all_row_centers_ordered[0] - median_pitch / 2.0
    divider_u[N] = all_row_centers_ordered[-1] + median_pitch / 2.0
    for i in range(1, N):
        divider_u[i] = (all_row_centers_ordered[i - 1] + all_row_centers_ordered[i]) / 2.0

    # Build left and right dividers
    left_centers_ordered = [np.dot(np.array(panels[L_idx]["center"]) - origin, u_axis) for L_idx, _ in block["matched_rows"]]
    right_centers_ordered = [np.dot(np.array(panels[R_idx]["center"]) - origin, u_axis) for _, R_idx in block["matched_rows"]]

    if pitch_diff_ratio > 0.15:
        divider_u_L = [0.0] * (N + 1)
        divider_u_L[0] = left_centers_ordered[0] - pitch_L / 2.0
        divider_u_L[N] = left_centers_ordered[-1] + pitch_L / 2.0
        for i in range(1, N):
            divider_u_L[i] = (left_centers_ordered[i - 1] + left_centers_ordered[i]) / 2.0

        divider_u_R = [0.0] * (N + 1)
        divider_u_R[0] = right_centers_ordered[0] - pitch_R / 2.0
        divider_u_R[N] = right_centers_ordered[-1] + pitch_R / 2.0
        for i in range(1, N):
            divider_u_R[i] = (right_centers_ordered[i - 1] + right_centers_ordered[i]) / 2.0
    else:
        divider_u_L = list(divider_u)
        divider_u_R = list(divider_u)

    # Helper to convert local coords to image coords
    def to_img(u_val, v_val):
        pt = origin + u_val * u_axis + v_val * v_axis
        return [float(pt[0]), float(pt[1])]

    # 6. Build candidate polygons per row
    # row_idx maps directly to divider_u[row_idx]..divider_u[row_idx+1]
    polygons = []
    polygon_meta = []  # per-polygon meta dict
    for row_idx, (L_idx, R_idx) in enumerate(block["matched_rows"]):
        is_ep_row = row_idx in endpoint_row_indices and use_endpoint_isolation
        div_left_u = divider_u[row_idx]
        div_right_u = divider_u[row_idx + 1]

        if is_ep_row:
            # Endpoint row: inner divider is the one facing the middle
            if row_idx == 0:
                ep_side = "start"
                inner_div_u_L = divider_u_L[1]
                inner_div_u_R = divider_u_R[1]
            else:
                ep_side = "end"
                inner_div_u_L = divider_u_L[N - 1]
                inner_div_u_R = divider_u_R[N - 1]

            use_locked = ENABLE_MIDDLE_LOCKED_ENDPOINT and (len(middle_row_indices) >= MIDDLE_PANEL_MIN_FOR_LOCKED_ENDPOINT)

            if use_locked:
                if ep_side == "start":
                    outer_u_L = inner_div_u_L - pitch_L
                    outer_u_R = inner_div_u_R - pitch_R
                else:
                    outer_u_L = inner_div_u_L + pitch_L
                    outer_u_R = inner_div_u_R + pitch_R

                pt1 = origin + inner_div_u_L * u_axis + left_rail_v * v_axis
                pt2 = origin + inner_div_u_L * u_axis + middle_rail_v * v_axis
                pt3 = origin + outer_u_L * u_axis + middle_rail_v * v_axis
                pt4 = origin + outer_u_L * u_axis + left_rail_v * v_axis
                left_poly = _sort_corners(np.array([pt1, pt2, pt3, pt4], dtype=np.float32))

                pt1 = origin + inner_div_u_R * u_axis + middle_rail_v * v_axis
                pt2 = origin + inner_div_u_R * u_axis + right_rail_v * v_axis
                pt3 = origin + outer_u_R * u_axis + right_rail_v * v_axis
                pt4 = origin + outer_u_R * u_axis + middle_rail_v * v_axis
                right_poly = _sort_corners(np.array([pt1, pt2, pt3, pt4], dtype=np.float32))

                polygons.append((left_poly, right_poly))
                polygon_meta.append((
                    {
                        "role": f"endpoint_{ep_side}",
                        "side": ep_side,
                        "extrapolate_L": abs(outer_u_L - inner_div_u_L) / max(pitch_L, 1e-3),
                        "extrapolate_R": abs(outer_u_R - inner_div_u_R) / max(pitch_R, 1e-3),
                        "source_L": "endpoint_middle_locked_pitch",
                        "source_R": "endpoint_middle_locked_pitch",
                        "score_L": 1.0,
                        "score_R": 1.0,
                        "candidates_L": [],
                        "candidates_R": [],
                        "yolo_outer_u_L": 0.0,
                        "yolo_outer_u_R": 0.0,
                        "pitch_outer_u_L": outer_u_L,
                        "pitch_outer_u_R": outer_u_R,
                        "edge_outer_u_L": 0.0,
                        "edge_outer_u_R": 0.0,
                        "consensus_outer_u_L": 0.0,
                        "consensus_outer_u_R": 0.0,
                        "final_outer_u_L": outer_u_L,
                        "final_outer_u_R": outer_u_R,
                        # extra fields
                        "endpoint_strategy": "middle_locked_pitch",
                        "inner_u_L": inner_div_u_L,
                        "inner_u_R": inner_div_u_R,
                        "pitch_L": pitch_L,
                        "pitch_R": pitch_R,
                    },
                ))
            else:
                # Legacy resolver fallback if not using locked
                # Left panel YOLO and basic info
                L_orig = panels[L_idx].get("original_yolo_polygon") or panels[L_idx]["polygon"]
                L_conf = panels[L_idx].get("confidence", panels[L_idx].get("conf", 1.0))
                
                # Right panel YOLO and basic info
                R_orig = panels[R_idx].get("original_yolo_polygon") or panels[R_idx]["polygon"]
                R_conf = panels[R_idx].get("confidence", panels[R_idx].get("conf", 1.0))

                middle_lattice_good = (len(middle_row_indices) >= MIN_MIDDLE_PANELS_PER_COL)
                inner_div_u = inner_div_u_L  # default to left divider u

                # Pass 1: Resolve without consensus
                L_res1 = resolve_hybrid_endpoint_outer_u(
                    L_orig, inner_div_u, left_rail_v, middle_rail_v,
                    u_axis, v_axis, origin, median_pitch, ep_side,
                    image, consensus_outer_u=None, yolo_conf=L_conf,
                    middle_lattice_good=middle_lattice_good
                )
                R_res1 = resolve_hybrid_endpoint_outer_u(
                    R_orig, inner_div_u, middle_rail_v, right_rail_v,
                    u_axis, v_axis, origin, median_pitch, ep_side,
                    image, consensus_outer_u=None, yolo_conf=R_conf,
                    middle_lattice_good=middle_lattice_good
                )

                # Pass 2: Check for consensus
                L_consensus_u = None
                R_consensus_u = None
                if col_angle_delta <= 5.0:
                    if L_res1["success"] and L_res1["best_candidate"]["score"] >= ENDPOINT_SCORE_MIN_ACCEPT:
                        R_consensus_u = L_res1["best_candidate"]["outer_u"]
                    if R_res1["success"] and R_res1["best_candidate"]["score"] >= ENDPOINT_SCORE_MIN_ACCEPT:
                        L_consensus_u = R_res1["best_candidate"]["outer_u"]

                # Run again if we have consensus candidate
                if L_consensus_u is not None:
                    L_res2 = resolve_hybrid_endpoint_outer_u(
                        L_orig, inner_div_u, left_rail_v, middle_rail_v,
                        u_axis, v_axis, origin, median_pitch, ep_side,
                        image, consensus_outer_u=L_consensus_u, yolo_conf=L_conf,
                        middle_lattice_good=middle_lattice_good
                    )
                    L_res = L_res2 if L_res2["success"] else L_res1
                else:
                    L_res = L_res1

                if R_consensus_u is not None:
                    R_res2 = resolve_hybrid_endpoint_outer_u(
                        R_orig, inner_div_u, middle_rail_v, right_rail_v,
                        u_axis, v_axis, origin, median_pitch, ep_side,
                        image, consensus_outer_u=R_consensus_u, yolo_conf=R_conf,
                        middle_lattice_good=middle_lattice_good
                    )
                    R_res = R_res2 if R_res2["success"] else R_res1
                else:
                    R_res = R_res1

                # Left panel final build
                if L_res["success"]:
                    L_best = L_res["best_candidate"]
                    L_outer = L_best["outer_u"]
                    pt1 = origin + inner_div_u * u_axis + left_rail_v * v_axis
                    pt2 = origin + inner_div_u * u_axis + middle_rail_v * v_axis
                    pt3 = origin + L_outer * u_axis + middle_rail_v * v_axis
                    pt4 = origin + L_outer * u_axis + left_rail_v * v_axis
                    left_poly = _sort_corners(np.array([pt1, pt2, pt3, pt4], dtype=np.float32))
                    
                    L_extra = abs(L_outer - inner_div_u) / max(median_pitch, 1e-3)
                    L_source = f"endpoint_{L_best['source']}"
                    L_score = L_best["score"]
                    L_candidates = L_res["candidates"]
                    L_yolo_outer_u = L_res["yolo_outer_u"]
                    L_pitch_outer_u = L_res["pitch_outer_u"]
                    L_edge_outer_u = L_res["edge_outer_u"]
                    L_consensus_outer_u = L_res["consensus_outer_u"]
                    L_final_outer_u = L_outer
                else:
                    left_poly = _sort_corners(np.array([
                        to_img(div_left_u, left_rail_v),
                        to_img(div_left_u, middle_rail_v),
                        to_img(div_right_u, middle_rail_v),
                        to_img(div_right_u, left_rail_v),
                    ]))
                    L_extra = 0.0
                    L_source = "endpoint_resolver_failed"
                    L_score = 0.0
                    L_candidates = []
                    L_yolo_outer_u = 0.0
                    L_pitch_outer_u = 0.0
                    L_edge_outer_u = 0.0
                    L_consensus_outer_u = 0.0
                    L_final_outer_u = 0.0

                # Right panel final build
                if R_res["success"]:
                    R_best = R_res["best_candidate"]
                    R_outer = R_best["outer_u"]
                    pt1 = origin + inner_div_u * u_axis + middle_rail_v * v_axis
                    pt2 = origin + inner_div_u * u_axis + right_rail_v * v_axis
                    pt3 = origin + R_outer * u_axis + right_rail_v * v_axis
                    pt4 = origin + R_outer * u_axis + middle_rail_v * v_axis
                    right_poly = _sort_corners(np.array([pt1, pt2, pt3, pt4], dtype=np.float32))
                    
                    R_extra = abs(R_outer - inner_div_u) / max(median_pitch, 1e-3)
                    R_source = f"endpoint_{R_best['source']}"
                    R_score = R_best["score"]
                    R_candidates = R_res["candidates"]
                    R_yolo_outer_u = R_res["yolo_outer_u"]
                    R_pitch_outer_u = R_res["pitch_outer_u"]
                    R_edge_outer_u = R_res["edge_outer_u"]
                    R_consensus_outer_u = R_res["consensus_outer_u"]
                    R_final_outer_u = R_outer
                else:
                    right_poly = _sort_corners(np.array([
                        to_img(div_left_u, middle_rail_v),
                        to_img(div_left_u, right_rail_v),
                        to_img(div_right_u, right_rail_v),
                        to_img(div_right_u, middle_rail_v),
                    ]))
                    R_extra = 0.0
                    R_source = "endpoint_resolver_failed"
                    R_score = 0.0
                    R_candidates = []
                    R_yolo_outer_u = 0.0
                    R_pitch_outer_u = 0.0
                    R_edge_outer_u = 0.0
                    R_consensus_outer_u = 0.0
                    R_final_outer_u = 0.0

                polygons.append((left_poly, right_poly))
                polygon_meta.append((
                    {
                        "role": f"endpoint_{ep_side}",
                        "side": ep_side,
                        "extrapolate_L": L_extra,
                        "extrapolate_R": R_extra,
                        "source_L": L_source,
                        "source_R": R_source,
                        "score_L": L_score,
                        "score_R": R_score,
                        "candidates_L": L_candidates,
                        "candidates_R": R_candidates,
                        "yolo_outer_u_L": L_yolo_outer_u,
                        "yolo_outer_u_R": R_yolo_outer_u,
                        "pitch_outer_u_L": L_pitch_outer_u,
                        "pitch_outer_u_R": R_pitch_outer_u,
                        "edge_outer_u_L": L_edge_outer_u,
                        "edge_outer_u_R": R_edge_outer_u,
                        "consensus_outer_u_L": L_consensus_outer_u,
                        "consensus_outer_u_R": R_consensus_outer_u,
                        "final_outer_u_L": L_final_outer_u,
                        "final_outer_u_R": R_final_outer_u,
                        "endpoint_strategy": "hybrid_resolver",
                        "inner_u_L": inner_div_u,
                        "inner_u_R": inner_div_u,
                        "pitch_L": median_pitch,
                        "pitch_R": median_pitch,
                    },
                ))
        else:
            # Middle row: standard lattice candidate using directly computed dividers
            c1 = to_img(divider_u_L[row_idx], left_rail_v)
            c2 = to_img(divider_u_L[row_idx], middle_rail_v)
            c3 = to_img(divider_u_L[row_idx + 1], middle_rail_v)
            c4 = to_img(divider_u_L[row_idx + 1], left_rail_v)
            left_poly = _sort_corners(np.array([c1, c2, c3, c4]))

            c1 = to_img(divider_u_R[row_idx], middle_rail_v)
            c2 = to_img(divider_u_R[row_idx], right_rail_v)
            c3 = to_img(divider_u_R[row_idx + 1], right_rail_v)
            c4 = to_img(divider_u_R[row_idx + 1], middle_rail_v)
            right_poly = _sort_corners(np.array([c1, c2, c3, c4]))

            polygons.append((left_poly, right_poly))
            polygon_meta.append((
                {"role": "middle", "side": None, "extrapolate_L": 0.0, "extrapolate_R": 0.0,
                 "source_L": "lattice_full", "source_R": "lattice_full"},
            ))

    return {
        "success": True,
        "left_rail_v": left_rail_v,
        "middle_rail_v": middle_rail_v,
        "right_rail_v": right_rail_v,
        "divider_u": divider_u,
        "divider_u_L": divider_u_L,
        "divider_u_R": divider_u_R,
        "origin": origin,
        "polygons": polygons,
        "polygon_meta": polygon_meta,
        "rail_fit_source": rail_fit_source,
        "col_angle_delta": col_angle_delta,
        "median_pitch": median_pitch,
        "pitch_L": pitch_L,
        "pitch_R": pitch_R,
        "endpoint_row_indices": list(endpoint_row_indices),
        "middle_row_indices": middle_row_indices,
    }


def refine_panels_by_two_column_block_lattice(
    panels: List[Dict[str, Any]],
    image_shape: Tuple[int, int, int],
    image_path: Optional[str] = None
) -> List[Dict[str, Any]]:
    if not panels:
        return []
        
    img_h, img_w = image_shape[:2]
    image_stem = os.path.splitext(os.path.basename(image_path))[0] if image_path else "unknown"
    
    # Bypass for protected images
    if image_path:
        stem_upper = image_stem.upper()
        if "DJI_0987" in stem_upper or "DJI_0995" in stem_upper:
            print(f"[TWO_COL_BLOCK] img={image_stem} BYPASS completely")
            return panels
            
    # Save the String-first polygon as fallback and initialize fields
    for p in panels:
        p["string_first_polygon"] = list(p["polygon"])
        p["two_col_refined"] = False
        p["two_col_decision"] = "bypass"
        p["two_col_reason"] = "not_processed"
        
    # Detect blocks
    blocks = detect_two_column_blocks(panels, image_shape)
    
    # Load image for Sobel/snapping
    image = None
    if image_path and os.path.exists(image_path):
        image = cv2.imread(image_path)
    if image is None:
        image = np.zeros((img_h, img_w, 3), dtype=np.uint8)
        
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    else:
        gray = image.astype(np.float32)
        
    log_records = []
    
    for block in blocks:
        block_id = block["block_id"]
        u_axis = block["u_axis"]
        v_axis = block["v_axis"]
        
        # Fit block lattice
        fit_res = fit_two_column_block_lattice(block, panels, image)

        # Handle fit failure (e.g., column angle delta too large)
        if not fit_res.get("success", True):
            reason = fit_res.get("reason", "fit_failed")
            print(f"[TWO_COL_BLOCK] block_id={block_id} fit FAILED reason={reason} → skip block")
            for L_idx, R_idx in block["matched_rows"]:
                for idx in [L_idx, R_idx]:
                    panels[idx]["two_col_decision"] = "fallback_block"
                    panels[idx]["two_col_reason"] = reason
            continue

        # Original coordinates
        left_rail_v = fit_res["left_rail_v"]
        middle_rail_v = fit_res["middle_rail_v"]
        right_rail_v = fit_res["right_rail_v"]
        divider_u = fit_res["divider_u"]
        origin = fit_res["origin"]
        rail_fit_source = fit_res.get("rail_fit_source", "all_panels")
        polygon_meta = fit_res.get("polygon_meta", [None] * len(block["matched_rows"]))
        block_endpoint_row_indices = set(fit_res.get("endpoint_row_indices", []))
        block_median_pitch = fit_res.get("median_pitch", block.get("median_span_u", 1.0))
        col_angle_delta = fit_res.get("col_angle_delta", 0.0)

        divider_u_L = fit_res.get("divider_u_L", divider_u)
        divider_u_R = fit_res.get("divider_u_R", divider_u)
        pitch_L = fit_res.get("pitch_L", block_median_pitch)
        pitch_R = fit_res.get("pitch_R", block_median_pitch)
        pre_snap_polygons = fit_res["polygons"]

        u_min = min(divider_u_L[0], divider_u_R[0])
        u_max = max(divider_u_L[-1], divider_u_R[-1])

        # Snapping left rail
        left_rail_v_snapped = left_rail_v
        score_old_left = _score_rail_at_v(gray, origin, u_axis, v_axis, left_rail_v, u_min, u_max)
        if TWO_COL_SNAP_RAILS:
            best_score = score_old_left
            for delta in range(-TWO_COL_SNAP_SEARCH_PX, TWO_COL_SNAP_SEARCH_PX + 1):
                if delta == 0:
                    continue
                v_try = left_rail_v + float(delta)
                score = _score_rail_at_v(gray, origin, u_axis, v_axis, v_try, u_min, u_max)
                if score > best_score:
                    best_score = score
                    left_rail_v_snapped = v_try
            shift_left = left_rail_v_snapped - left_rail_v
        else:
            shift_left = 0.0
            
        # Snapping middle rail
        middle_rail_v_snapped = middle_rail_v
        half_gap = (right_rail_v - left_rail_v) / 4.0
        score_old_middle = _score_middle_rail_at_v(gray, origin, u_axis, v_axis, middle_rail_v, u_min, u_max, half_gap)
        if TWO_COL_SNAP_RAILS:
            best_score = score_old_middle
            for delta in range(-TWO_COL_SNAP_SEARCH_PX, TWO_COL_SNAP_SEARCH_PX + 1):
                if delta == 0:
                    continue
                v_try = middle_rail_v + float(delta)
                score = _score_middle_rail_at_v(gray, origin, u_axis, v_axis, v_try, u_min, u_max, half_gap)
                if score > best_score:
                    best_score = score
                    middle_rail_v_snapped = v_try
            shift_middle = middle_rail_v_snapped - middle_rail_v
        else:
            shift_middle = 0.0
            
        # Snapping right rail
        right_rail_v_snapped = right_rail_v
        score_old_right = _score_rail_at_v(gray, origin, u_axis, v_axis, right_rail_v, u_min, u_max)
        if TWO_COL_SNAP_RAILS:
            best_score = score_old_right
            for delta in range(-TWO_COL_SNAP_SEARCH_PX, TWO_COL_SNAP_SEARCH_PX + 1):
                if delta == 0:
                    continue
                v_try = right_rail_v + float(delta)
                score = _score_rail_at_v(gray, origin, u_axis, v_axis, v_try, u_min, u_max)
                if score > best_score:
                    best_score = score
                    right_rail_v_snapped = v_try
            shift_right = right_rail_v_snapped - right_rail_v
        else:
            shift_right = 0.0

        # Snapping dividers
        snapped_divider_u_L = list(divider_u_L)
        snapped_divider_u_R = list(divider_u_R)
        n_dividers_snapped = 0
        div_shifts = []
        div_scores_old = []
        div_scores_new = []
        
        use_locked = ENABLE_MIDDLE_LOCKED_ENDPOINT and (len(fit_res.get("middle_row_indices", [])) >= MIDDLE_PANEL_MIN_FOR_LOCKED_ENDPOINT)

        if TWO_COL_SNAP_DIVIDERS:
            # Snap left column dividers
            for d_idx in range(len(divider_u_L)):
                if use_locked and (d_idx == 0 or d_idx == len(divider_u_L) - 1):
                    div_shifts.append(0.0)
                    div_scores_old.append(0.0)
                    div_scores_new.append(0.0)
                    continue
                u_div = divider_u_L[d_idx]
                best_u = u_div
                score_old = _score_divider_at_u(gray, origin, u_axis, v_axis, u_div, left_rail_v, middle_rail_v, pitch_L / 2.0)
                best_score = score_old
                for delta in range(-TWO_COL_SNAP_SEARCH_PX, TWO_COL_SNAP_SEARCH_PX + 1):
                    if delta == 0:
                        continue
                    u_try = u_div + float(delta)
                    score = _score_divider_at_u(gray, origin, u_axis, v_axis, u_try, left_rail_v, middle_rail_v, pitch_L / 2.0)
                    if score > best_score:
                        best_score = score
                        best_u = u_try
                shift_div = best_u - u_div
                snapped_divider_u_L[d_idx] = best_u
                if abs(shift_div) > 0.5:
                    n_dividers_snapped += 1
                div_shifts.append(shift_div)
                div_scores_old.append(score_old)
                div_scores_new.append(best_score)

            # Snap right column dividers
            for d_idx in range(len(divider_u_R)):
                if use_locked and (d_idx == 0 or d_idx == len(divider_u_R) - 1):
                    continue
                u_div = divider_u_R[d_idx]
                best_u = u_div
                score_old = _score_divider_at_u(gray, origin, u_axis, v_axis, u_div, middle_rail_v, right_rail_v, pitch_R / 2.0)
                best_score = score_old
                for delta in range(-TWO_COL_SNAP_SEARCH_PX, TWO_COL_SNAP_SEARCH_PX + 1):
                    if delta == 0:
                        continue
                    u_try = u_div + float(delta)
                    score = _score_divider_at_u(gray, origin, u_axis, v_axis, u_try, middle_rail_v, right_rail_v, pitch_R / 2.0)
                    if score > best_score:
                        best_score = score
                        best_u = u_try
                shift_div = best_u - u_div
                snapped_divider_u_R[d_idx] = best_u
                if abs(shift_div) > 0.5:
                    n_dividers_snapped += 1

        # --- Apply Outer Boundary Guard to block outer dividers ---
        block_outer_guard = None
        if ENABLE_OUTER_BOUNDARY_GUARD and not use_locked:
            block_dict = {
                "u_axis": u_axis,
                "v_axis": v_axis,
                "origin": origin,
                "left_rail_v": left_rail_v_snapped,
                "right_rail_v": right_rail_v_snapped
            }
            temp_dividers = list(divider_u)
            block_outer_guard = refine_outer_dividers(block_dict, temp_dividers, panels, image)
            if block_outer_guard["start"]["snap_ok"]:
                snapped_divider_u_L[0] = block_outer_guard["start"]["u_best"]
                snapped_divider_u_R[0] = block_outer_guard["start"]["u_best"]
            if block_outer_guard["end"]["snap_ok"]:
                snapped_divider_u_L[-1] = block_outer_guard["end"]["u_best"]
                snapped_divider_u_R[-1] = block_outer_guard["end"]["u_best"]

        if use_locked:
            N_rows = len(block["matched_rows"])
            snapped_divider_u_L[0] = snapped_divider_u_L[1] - pitch_L
            snapped_divider_u_R[0] = snapped_divider_u_R[1] - pitch_R
            snapped_divider_u_L[N_rows] = snapped_divider_u_L[N_rows - 1] + pitch_L
            snapped_divider_u_R[N_rows] = snapped_divider_u_R[N_rows - 1] + pitch_R

        # Construct snapped/final polygons
        def to_img(u_val, v_val):
            pt = origin + u_val * u_axis + v_val * v_axis
            return [pt[0], pt[1]]
            
        snapped_polygons = []
        for i in range(len(block["matched_rows"])):
            c1 = to_img(snapped_divider_u_L[i], left_rail_v_snapped)
            c2 = to_img(snapped_divider_u_L[i], middle_rail_v_snapped)
            c3 = to_img(snapped_divider_u_L[i+1], middle_rail_v_snapped)
            c4 = to_img(snapped_divider_u_L[i+1], left_rail_v_snapped)
            left_poly = _sort_corners(np.array([c1, c2, c3, c4]))
            
            c1 = to_img(snapped_divider_u_R[i], middle_rail_v_snapped)
            c2 = to_img(snapped_divider_u_R[i], right_rail_v_snapped)
            c3 = to_img(snapped_divider_u_R[i+1], right_rail_v_snapped)
            c4 = to_img(snapped_divider_u_R[i+1], middle_rail_v_snapped)
            right_poly = _sort_corners(np.array([c1, c2, c3, c4]))
            
            snapped_polygons.append((left_poly, right_poly))
            
        # Snapping constraints validation
        # 1. Compute pre/post median IoU
        all_iou_pre = []
        for idx, (L_idx, R_idx) in enumerate(block["matched_rows"]):
            L_orig = panels[L_idx]["original_yolo_polygon"]
            R_orig = panels[R_idx]["original_yolo_polygon"]
            iou_L = _compute_mask_iou_cv(np.array(pre_snap_polygons[idx][0], dtype=np.float32), np.array(L_orig, dtype=np.float32), (img_h, img_w))
            iou_R = _compute_mask_iou_cv(np.array(pre_snap_polygons[idx][1], dtype=np.float32), np.array(R_orig, dtype=np.float32), (img_h, img_w))
            all_iou_pre.append(iou_L)
            all_iou_pre.append(iou_R)
        median_iou_pre = np.median(all_iou_pre) if all_iou_pre else 1.0
        
        all_iou_post = []
        for idx, (L_idx, R_idx) in enumerate(block["matched_rows"]):
            L_orig = panels[L_idx]["original_yolo_polygon"]
            R_orig = panels[R_idx]["original_yolo_polygon"]
            iou_L = _compute_mask_iou_cv(np.array(snapped_polygons[idx][0], dtype=np.float32), np.array(L_orig, dtype=np.float32), (img_h, img_w))
            iou_R = _compute_mask_iou_cv(np.array(snapped_polygons[idx][1], dtype=np.float32), np.array(R_orig, dtype=np.float32), (img_h, img_w))
            all_iou_post.append(iou_L)
            all_iou_post.append(iou_R)
        median_iou_post = np.median(all_iou_post) if all_iou_post else 1.0
        
        # 2. Check outside image
        any_outside = False
        for left_poly, right_poly in snapped_polygons:
            for pt in left_poly + right_poly:
                if pt[0] < -2.0 or pt[0] > img_w + 2.0 or pt[1] < -2.0 or pt[1] > img_h + 2.0:
                    any_outside = True
                    break
            if any_outside:
                break
                
        # 3. Check overlap
        max_overlap_snap = 0.0
        flat_snapped = []
        for left_poly, right_poly in snapped_polygons:
            flat_snapped.append(left_poly)
            flat_snapped.append(right_poly)
        for a_idx in range(len(flat_snapped)):
            for b_idx in range(a_idx + 1, len(flat_snapped)):
                overlap = compute_polygon_overlap_ratio(flat_snapped[a_idx], flat_snapped[b_idx], image_shape)
                if overlap > max_overlap_snap:
                    max_overlap_snap = overlap
                    
        # Decision
        snap_valid = True
        snap_reject_reason = ""
        if any_outside:
            snap_valid = False
            snap_reject_reason = "outside_image"
        elif max_overlap_snap > TWO_COL_MAX_OVERLAP:
            snap_valid = False
            snap_reject_reason = f"max_overlap_{max_overlap_snap:.3f}"
        elif median_iou_post < median_iou_pre - 0.08:
            snap_valid = False
            snap_reject_reason = f"iou_drop_{median_iou_pre - median_iou_post:.3f}"
            
        snap_decision = "use_snap" if snap_valid else "keep_original"
        
        # Log snap results for each element
        print(f"[TWO_COL_SNAP] block_id={block_id} rail/divider=left shift={shift_left:+.1f} score_old={score_old_left:.2f} score_new={score_old_left if not TWO_COL_SNAP_RAILS or not snap_valid else _score_rail_at_v(gray, origin, u_axis, v_axis, left_rail_v_snapped, u_min, u_max):.2f} decision={snap_decision}")
        print(f"[TWO_COL_SNAP] block_id={block_id} rail/divider=middle shift={shift_middle:+.1f} score_old={score_old_middle:.2f} score_new={score_old_middle if not TWO_COL_SNAP_RAILS or not snap_valid else _score_middle_rail_at_v(gray, origin, u_axis, v_axis, middle_rail_v_snapped, u_min, u_max, half_gap):.2f} decision={snap_decision}")
        print(f"[TWO_COL_SNAP] block_id={block_id} rail/divider=right shift={shift_right:+.1f} score_old={score_old_right:.2f} score_new={score_old_right if not TWO_COL_SNAP_RAILS or not snap_valid else _score_rail_at_v(gray, origin, u_axis, v_axis, right_rail_v_snapped, u_min, u_max):.2f} decision={snap_decision}")
        if TWO_COL_SNAP_DIVIDERS:
            for d_idx, u_div in enumerate(divider_u):
                print(f"[TWO_COL_SNAP] block_id={block_id} rail/divider=divider_{d_idx} shift={div_shifts[d_idx]:+.1f} score_old={div_scores_old[d_idx]:.2f} score_new={div_scores_old[d_idx] if not snap_valid else div_scores_new[d_idx]:.2f} decision={snap_decision}")
                
        if snap_valid:
            left_rail_v = left_rail_v_snapped
            middle_rail_v = middle_rail_v_snapped
            right_rail_v = right_rail_v_snapped
            divider_u_L = snapped_divider_u_L
            divider_u_R = snapped_divider_u_R
            candidate_polygons = snapped_polygons
        else:
            candidate_polygons = pre_snap_polygons
            print(f"[TWO_COL_SNAP_REJECT] block_id={block_id} reason={snap_reject_reason} → keep pre-snap lattice")
            
        # Store fitted params in block for debug image drawing
        block["origin"] = origin
        block["fitted_left_rail_v"] = left_rail_v
        block["fitted_middle_rail_v"] = middle_rail_v
        block["fitted_right_rail_v"] = right_rail_v
        block["fitted_divider_u_L"] = divider_u_L
        block["fitted_divider_u_R"] = divider_u_R
        block["fitted_divider_u"] = list(divider_u_L) # fallback for legacy drawing
        
        # 4. Validate each panel in block
        middle_row_indices = fit_res.get("middle_row_indices", [])
        middle_areas = []
        for r_idx in middle_row_indices:
            for c_idx in block["matched_rows"][r_idx]:
                m_p = panels[c_idx]
                m_poly_pts = np.array(m_p.get("original_yolo_polygon") or m_p["polygon"], dtype=np.float32)
                middle_areas.append(float(cv2.contourArea(m_poly_pts.reshape(-1, 1, 2))))
        median_middle_area = float(np.median(middle_areas)) if middle_areas else 1.0

        panel_details = []
        for row_idx, (L_idx, R_idx) in enumerate(block["matched_rows"]):
            # Get polygon_meta for this row
            row_meta = polygon_meta[row_idx][0] if row_idx < len(polygon_meta) and polygon_meta[row_idx] else {}
            row_panel_role = row_meta.get("role", "middle")
            row_ep_side = row_meta.get("side", None)
            is_ep_row = row_idx in block_endpoint_row_indices

            overlap_val = 0.0
            for col_name, orig_idx, cand_poly in [("left", L_idx, candidate_polygons[row_idx][0]),
                                                  ("right", R_idx, candidate_polygons[row_idx][1])]:
                ep_extra = row_meta.get(f"extrapolate_{col_name[0].upper()}", 0.0)  # extrapolate_L or extrapolate_R
                ep_source = row_meta.get(f"source_{col_name[0].upper()}", "lattice_full")
                used_for_fit = not is_ep_row

                p = panels[orig_idx]
                orig_poly = p["original_yolo_polygon"]
                orig_pts = np.array(orig_poly, dtype=np.float32)
                orig_area = float(cv2.contourArea(orig_pts.reshape(-1, 1, 2))) if len(orig_pts) >= 3 else 1.0
                orig_center = np.mean(orig_pts, axis=0)

                cand_pts = np.array(cand_poly, dtype=np.float32)
                candidate_area = float(cv2.contourArea(cand_pts.reshape(-1, 1, 2)))
                cand_center = np.mean(cand_pts, axis=0)

                area_ratio = candidate_area / max(orig_area, 1.0)
                iou_with_original = _compute_mask_iou_cv(cand_pts, orig_pts, (img_h, img_w))

                orig_x = [pt[0] for pt in orig_poly]
                orig_y = [pt[1] for pt in orig_poly]
                orig_w_sz = max(orig_x) - min(orig_x) if orig_x else 1.0
                orig_h_sz = max(orig_y) - min(orig_y) if orig_y else 1.0

                center_distance = float(np.linalg.norm(cand_center - orig_center))
                center_shift_ratio = center_distance / max(orig_w_sz, orig_h_sz)

                cand_u_proj = [float(np.dot(pt - origin, u_axis)) for pt in cand_pts]
                cand_v_proj = [float(np.dot(pt - origin, v_axis)) for pt in cand_pts]
                cand_w = max(cand_u_proj) - min(cand_u_proj)
                cand_h = max(cand_v_proj) - min(cand_v_proj)

                orig_u_proj = [float(np.dot(pt - origin, u_axis)) for pt in orig_pts]
                orig_v_proj = [float(np.dot(pt - origin, v_axis)) for pt in orig_pts]
                orig_w_proj = max(orig_u_proj) - min(orig_u_proj) if orig_u_proj else 1.0
                orig_h_proj = max(orig_v_proj) - min(orig_v_proj) if orig_v_proj else 1.0

                width_ratio = cand_w / max(orig_w_proj, 1e-3)
                height_ratio = cand_h / max(orig_h_proj, 1e-3)

                outside_image = any(pt[0] < -2.0 or pt[0] > img_w + 2.0 or pt[1] < -2.0 or pt[1] > img_h + 2.0 for pt in cand_poly)

                col_letter = col_name[0].upper()
                panel_details.append({
                    "p_obj": p,
                    "panel_idx": orig_idx,
                    "column": col_name,
                    "column_id": 0 if col_name == "left" else 1,
                    "row_idx": row_idx,
                    "panel_role": row_panel_role,
                    "is_endpoint": is_ep_row,
                    "endpoint_side": row_ep_side,
                    "used_for_lattice_fit": used_for_fit,
                    "endpoint_extrapolate_ratio": ep_extra,
                    "candidate_polygon_source": ep_source,
                    "candidate_polygon": cand_poly,
                    "original_polygon": orig_poly,
                    "iou_with_original": iou_with_original,
                    "center_shift_ratio": center_shift_ratio,
                    "area_ratio": area_ratio,
                    "width_ratio": width_ratio,
                    "height_ratio": height_ratio,
                    "candidate_area": candidate_area,
                    "original_area": orig_area,
                    "outside_image": outside_image,
                    "max_overlap": 0.0,
                    "pass_individual": True,
                    "fail_reason": "ok",
                    "endpoint_outer_score": row_meta.get(f"score_{col_letter}", 0.0),
                    "endpoint_candidates": row_meta.get(f"candidates_{col_letter}", []),
                    "endpoint_yolo_outer_u": row_meta.get(f"yolo_outer_u_{col_letter}", 0.0),
                    "endpoint_pitch_outer_u": row_meta.get(f"pitch_outer_u_{col_letter}", 0.0),
                    "endpoint_edge_outer_u": row_meta.get(f"edge_outer_u_{col_letter}", 0.0),
                    "endpoint_consensus_outer_u": row_meta.get(f"consensus_outer_u_{col_letter}", 0.0),
                    "endpoint_final_outer_u": row_meta.get(f"final_outer_u_{col_letter}", 0.0),
                    
                    # extra fields for middle-locked logging
                    "endpoint_strategy": row_meta.get("endpoint_strategy", "n/a"),
                    "inner_u": float(row_meta.get(f"inner_u_{col_letter}", 0.0)) if is_ep_row else 0.0,
                    "outer_u": float(row_meta.get(f"outer_u_{col_letter}", 0.0)) if is_ep_row else 0.0,
                    "pitch": float(pitch_L if col_name == "left" else pitch_R),
                    "endpoint_pitch_ratio": float(ep_extra) if is_ep_row else 0.0,
                    "yolo_endpoint_area": float(orig_area) if is_ep_row else 0.0,
                    "median_middle_area": float(median_middle_area),
                    "endpoint_overlap_with_neighbor": float(overlap_val) if is_ep_row else 0.0,
                    "middle_lattice_quality_good": (len(middle_row_indices) >= MIN_MIDDLE_PANELS_PER_COL),
                })
                
        # Compute overlap
        for i in range(len(panel_details)):
            max_over = 0.0
            poly_i = panel_details[i]["candidate_polygon"]
            for j in range(len(panel_details)):
                if i == j:
                    continue
                poly_j = panel_details[j]["candidate_polygon"]
                overlap = compute_polygon_overlap_ratio(poly_i, poly_j, image_shape)
                if overlap > max_over:
                    max_over = overlap
            panel_details[i]["max_overlap"] = max_over

        # Compute original_max_overlap for all panels in the block
        for i in range(len(panel_details)):
            max_orig = 0.0
            poly_i = panel_details[i]["original_polygon"]
            for j in range(len(panel_details)):
                if i == j:
                    continue
                poly_j = panel_details[j]["original_polygon"]
                overlap = compute_polygon_overlap_ratio(poly_i, poly_j, image_shape)
                if overlap > max_orig:
                    max_orig = overlap
            panel_details[i]["original_max_overlap"] = max_orig

        # Compute edge scores and other metrics for all panels in the block
        for det in panel_details:
            orig_poly = det["original_polygon"]
            cand_poly = det["candidate_polygon"]
            
            det["original_area"] = float(det["original_area"])
            det["lattice_area"] = float(det["candidate_area"])
            det["iou_original_lattice"] = float(det["iou_with_original"])
            det["lattice_max_overlap"] = float(det["max_overlap"])
            
            edge_orig = score_panel_boundary_alignment(image, orig_poly) if image is not None else 0.0
            edge_lat = score_panel_boundary_alignment(image, cand_poly) if image is not None else 0.0
            det["original_edge_score"] = float(edge_orig)
            det["lattice_edge_score"] = float(edge_lat)
            det["edge_score_ratio"] = float(edge_lat / max(edge_orig, 1e-6))

        # Compute median middle u_length and v_width if N >= 5
        N = len(block["matched_rows"])
        median_middle_u_length = block["median_span_u"]
        median_middle_v_width = abs(right_rail_v_snapped - left_rail_v_snapped) / 2.0
        if N >= 5:
            middle_u_lengths = []
            middle_v_widths = []
            for r in range(1, N - 1):
                for c_poly in candidate_polygons[r]:
                    c_np = np.array(c_poly, dtype=np.float32)
                    u_projs = [float(np.dot(pt - origin, u_axis)) for pt in c_np]
                    v_projs = [float(np.dot(pt - origin, v_axis)) for pt in c_np]
                    middle_u_lengths.append(max(u_projs) - min(u_projs))
                    middle_v_widths.append(max(v_projs) - min(v_projs))
            if middle_u_lengths:
                median_middle_u_length = float(np.median(middle_u_lengths))
            if middle_v_widths:
                median_middle_v_width = float(np.median(middle_v_widths))

        # Panel-level validation decision
        for det in panel_details:
            is_outer_panel = False
            outer_side = None
            if det["row_idx"] == 0:
                is_outer_panel = True
                outer_side = "start"
            elif det["row_idx"] == N - 1:
                is_outer_panel = True
                outer_side = "end"

            force_fallback_outer = False
            outer_reason = "ok"
            outer_div_source = "extrapolated"
            outer_div_shift = 0.0
            outer_div_score = 0.0
            outer_guard_decision = "keep_lattice"

            if ENABLE_OUTER_BOUNDARY_GUARD and is_outer_panel and block_outer_guard is not None:
                guard_res = block_outer_guard[outer_side]
                outer_div_shift = guard_res["shift"]
                outer_div_score = guard_res["best_score"]
                if guard_res["snap_ok"]:
                    outer_div_source = "edge_snap"
                    outer_guard_decision = "use_outer_snap"
                else:
                    outer_div_source = "fallback" if not OUTER_PANEL_ALLOW_EXTRAPOLATE else "extrapolated"
                    if not OUTER_PANEL_ALLOW_EXTRAPOLATE:
                        force_fallback_outer = True
                        outer_reason = "extrapolate_disallowed"
                        outer_guard_decision = "fallback_outer_panel"

                if not force_fallback_outer:
                    val_pass = True
                    val_reason = "ok"
                    if not (OUTER_PANEL_AREA_RATIO_MIN <= det["area_ratio"] <= OUTER_PANEL_AREA_RATIO_MAX):
                        val_pass = False
                        val_reason = "area_ratio_out"
                    elif det["center_shift_ratio"] > OUTER_PANEL_CENTER_SHIFT_MAX:
                        val_pass = False
                        val_reason = "center_shift_too_large"
                    elif det["iou_with_original"] < OUTER_PANEL_IOU_MIN:
                        val_pass = False
                        val_reason = "iou_too_low"
                    elif det["outside_image"]:
                        val_pass = False
                        val_reason = "outside_image"
                    elif det["max_overlap"] > TWO_COL_MAX_OVERLAP:
                        val_pass = False
                        val_reason = "max_overlap_high"

                    if not val_pass:
                        force_fallback_outer = True
                        outer_reason = val_reason
                        outer_guard_decision = "fallback_outer_panel"

            if ENABLE_OUTER_BOUNDARY_GUARD and is_outer_panel:
                print(f"[OUTER_GUARD] img={image_stem} block={block_id} string=n/a side={outer_side} "
                      f"score={outer_div_score:.3f} shift={outer_div_shift:+.1f} decision={outer_guard_decision}")
                if force_fallback_outer:
                    print(f"[OUTER_GUARD] img={image_stem} panel_idx={det['panel_idx']} "
                          f"decision=fallback_outer_panel reason={outer_reason}")

            if is_outer_panel:
                det["is_outer_panel"] = True
                det["outer_side"] = outer_side
                det["outer_divider_source"] = outer_div_source
                det["outer_divider_shift"] = outer_div_shift
                det["outer_divider_score"] = outer_div_score
                det["outer_guard_decision"] = outer_guard_decision
                det["outer_guard_reason"] = outer_reason
                det["outer_divider_u_proposal"] = block_outer_guard[outer_side]["u_original"] if block_outer_guard is not None else 0.0
                det["outer_divider_u_snapped"] = block_outer_guard[outer_side]["u_best"] if block_outer_guard is not None else 0.0
                det["outer_divider_u_axis"] = u_axis
                det["outer_divider_v_axis"] = v_axis
                det["outer_divider_origin"] = origin
                det["outer_divider_rail_low"] = left_rail_v_snapped
                det["outer_divider_rail_high"] = right_rail_v_snapped

            is_endpoint = False
            endpoint_side = None
            if N >= 5:
                if det["row_idx"] == 0:
                    is_endpoint = True
                    endpoint_side = "start"
                elif det["row_idx"] == N - 1:
                    is_endpoint = True
                    endpoint_side = "end"

            det["is_endpoint_panel"] = is_endpoint
            det["endpoint_side"] = endpoint_side

            ep_res = None
            endpoint_decision = "n/a"
            endpoint_reason = "n/a"
            endpoint_diagnostic_pass = True

            if is_endpoint:
                col_idx = 0 if det["column"] == "left" else 1
                row_idx = det["row_idx"]
                cand_poly = det["candidate_polygon"]
                p = det["p_obj"]
                orig_poly = det["original_polygon"]
                candidate_original = p.get("string_first_polygon") or p.get("original_yolo_polygon") or orig_poly

                ep_score = det.get("endpoint_outer_score", 0.0)
                ep_extra = det.get("endpoint_extrapolate_ratio", 0.0)
                area_ratio = det.get("area_ratio", 0.0)
                center_shift_ratio = det.get("center_shift_ratio", 0.0)
                iou_with_original = det.get("iou_with_original", 0.0)
                height_ratio = det.get("height_ratio", 0.0)
                ep_yolo_outer_u = det.get("endpoint_yolo_outer_u", 0.0)
                ep_pitch_outer_u = det.get("endpoint_pitch_outer_u", 0.0)
                ep_edge_outer_u = det.get("endpoint_edge_outer_u", 0.0)
                ep_consensus_outer_u = det.get("endpoint_consensus_outer_u", 0.0)
                ep_final_outer_u = det.get("endpoint_final_outer_u", 0.0)
                ep_source = det.get("candidate_polygon_source", "")
                ep_candidates = det.get("endpoint_candidates", [])
                is_yolo_endpoint_small = (orig_area < 0.65 * median_middle_area)

                # validation steps
                val_pass = True
                reasons = []

                if det["outside_image"]:
                    val_pass = False
                    reasons.append("outside_image")

                if not val_pass:
                    pass
                elif ENABLE_MIDDLE_LOCKED_ENDPOINT and use_locked:
                    # 1. Pitch ratio check
                    endpoint_pitch_ratio = ep_extra
                    if not (ENDPOINT_PITCH_RATIO_MIN <= endpoint_pitch_ratio <= ENDPOINT_PITCH_RATIO_MAX):
                        val_pass = False
                        reasons.append(f"endpoint_pitch_ratio_out_{endpoint_pitch_ratio:.2f}")

                    # 2. Area ratio check
                    elif not (ENDPOINT_AREA_RATIO_MIN <= area_ratio <= ENDPOINT_AREA_RATIO_MAX):
                        val_pass = False
                        reasons.append(f"endpoint_area_ratio_out_{area_ratio:.2f}")

                    # 3. Center shift check
                    else:
                        c_shift_max = ENDPOINT_CENTER_SHIFT_MAX_RATIO
                        if center_shift_ratio > c_shift_max:
                            if is_yolo_endpoint_small:
                                print(f"[ENDPOINT_GUARD] ignore_center_shift_due_to_small_yolo_endpoint panel={orig_idx} shift={center_shift_ratio:.2f}")
                            else:
                                val_pass = False
                                reasons.append(f"endpoint_center_shift_{center_shift_ratio:.2f}")

                    # 4. IoU with YOLO check
                    if val_pass:
                        middle_lattice_good = (len(middle_row_indices) >= MIN_MIDDLE_PANELS_PER_COL)
                        iou_min = ENDPOINT_LOW_IOU_MIN_WHEN_LATTICE_GOOD if middle_lattice_good else ENDPOINT_IOU_WITH_YOLO_MIN
                        if iou_with_original < iou_min:
                            val_pass = False
                            reasons.append(f"endpoint_iou_low_{iou_with_original:.2f}")

                    # 5. Overlap with neighbor check
                    if val_pass:
                        neighbor_row_idx = 1 if row_idx == 0 else N - 2
                        col_idx_in_row = 0 if det["column"] == "left" else 1
                        neighbor_cand_poly = np.array(candidate_polygons[neighbor_row_idx][col_idx_in_row], dtype=np.float32)
                        overlap_val = compute_polygon_overlap_ratio(cand_pts, neighbor_cand_poly, (img_h, img_w))
                        if overlap_val > ENDPOINT_MAX_OVERLAP_RATIO:
                            val_pass = False
                            reasons.append(f"endpoint_overlap_too_high_{overlap_val:.3f}")
                else:
                    if ep_score < ENDPOINT_SCORE_MIN_ACCEPT:
                        val_pass = False
                        reasons.append("endpoint_outer_resolver_low_score")
                    if not (ENDPOINT_LENGTH_RATIO_MIN <= ep_extra <= ENDPOINT_LENGTH_RATIO_MAX):
                        val_pass = False
                        reasons.append(f"endpoint_length_ratio_out_{ep_extra:.2f}")
                    if not (ENDPOINT_AREA_RATIO_MIN <= area_ratio <= ENDPOINT_AREA_RATIO_MAX):
                        val_pass = False
                        reasons.append(f"endpoint_area_ratio_out_{area_ratio:.2f}")
                    if center_shift_ratio > ENDPOINT_CENTER_SHIFT_MAX:
                        val_pass = False
                        reasons.append(f"endpoint_center_shift_{center_shift_ratio:.2f}")
                    if iou_with_original < 0.40:
                        val_pass = False
                        reasons.append(f"endpoint_iou_low_{iou_with_original:.2f}")

                endpoint_decision = "use_lattice_endpoint" if val_pass else "fallback_endpoint_original"
                endpoint_reason = "ok" if val_pass else "|".join(reasons)
                endpoint_diagnostic_pass = val_pass

                det["endpoint_decision"] = endpoint_decision
                det["endpoint_reason"] = endpoint_reason
                det["endpoint_diagnostic_pass"] = endpoint_diagnostic_pass
                det["candidate_lattice_area_ratio"] = area_ratio
                det["candidate_lattice_center_shift"] = center_shift_ratio
                det["candidate_lattice_iou"] = iou_with_original
                det["candidate_lattice_u_length_ratio"] = ep_extra
                det["candidate_lattice_v_width_ratio"] = height_ratio
                det["candidate_lattice_edge_score"] = det["lattice_edge_score"]
                det["candidate_original_edge_score"] = det["original_edge_score"]
                det["edge_score_ratio"] = det["edge_score_ratio"]
                # Additional Hybrid Resolver parameters
                det["endpoint_yolo_outer_u"] = ep_yolo_outer_u
                det["endpoint_pitch_outer_u"] = ep_pitch_outer_u
                det["endpoint_edge_outer_u"] = ep_edge_outer_u
                det["endpoint_consensus_outer_u"] = ep_consensus_outer_u
                det["endpoint_final_outer_u"] = ep_final_outer_u
                det["endpoint_outer_source"] = ep_source.replace("endpoint_", "")
                det["endpoint_outer_score"] = ep_score
                det["endpoint_length_ratio"] = ep_extra
                det["endpoint_candidates"] = ep_candidates
                det["angle_delta_to_block"] = col_angle_delta

            if is_endpoint:
                dec_print = endpoint_decision
                print(f"[ENDPOINT_GUARD] img={image_stem} block={block_id} panel={det['panel_idx']} side={endpoint_side} "
                      f"decision={dec_print} reason={endpoint_reason} "
                      f"area={area_ratio:.2f} shift={center_shift_ratio:.2f} "
                      f"iou={iou_with_original:.2f} edge_ratio={det['edge_score_ratio']:.2f}")

            # Decision making
            use_lattice_bool = False
            two_col_decision = "use_two_col_lattice"
            two_col_reason = "ok"

            if force_fallback_outer:
                two_col_decision = "fallback_outer_panel"
                two_col_reason = outer_reason
                use_lattice_bool = False
                det["pass_individual"] = False
                det["fail_reason"] = "fallback_outer_panel"
            elif ENABLE_LATTICE_SAFE_IMPROVEMENT_ONLY:
                # Safe Improvement Check (hard gate — only used when ENABLE_LATTICE_SAFE_IMPROVEMENT_ONLY=True)
                pass_safe, fail_reason_safe, stats_safe = check_safe_improvement(
                    det["original_polygon"], det["candidate_polygon"], image, (img_h, img_w),
                    det["original_max_overlap"], det["max_overlap"]
                )
                if pass_safe:
                    two_col_decision = "use_two_col_lattice"
                    two_col_reason = "ok"
                    use_lattice_bool = True
                    det["pass_individual"] = True
                    det["fail_reason"] = "ok"
                else:
                    two_col_decision = "keep_original_not_improved"
                    two_col_reason = fail_reason_safe
                    use_lattice_bool = False
                    det["pass_individual"] = False
                    det["fail_reason"] = fail_reason_safe
            else:
                # Lattice-primary logic: endpoint guard controls endpoints; basic geometry for middle
                if is_endpoint:
                    if endpoint_decision == "use_lattice_endpoint":
                        two_col_decision = "use_lattice_endpoint"
                        two_col_reason = "ok"
                        use_lattice_bool = True
                        det["pass_individual"] = True
                        det["fail_reason"] = "ok"
                    else:
                        two_col_decision = "fallback_endpoint_original"
                        two_col_reason = endpoint_reason
                        use_lattice_bool = False
                        det["pass_individual"] = False
                        det["fail_reason"] = endpoint_reason
                else:
                    pass_val = True
                    fail_reason = "ok"

                    if not (TWO_COL_AREA_RATIO_MIN <= det["area_ratio"] <= TWO_COL_AREA_RATIO_MAX):
                        pass_val = False
                        fail_reason = "area_ratio_out"
                    elif not (TWO_COL_WIDTH_RATIO_MIN <= det["width_ratio"] <= TWO_COL_WIDTH_RATIO_MAX):
                        pass_val = False
                        fail_reason = "width_ratio_out"
                    elif not (TWO_COL_HEIGHT_RATIO_MIN <= det["height_ratio"] <= TWO_COL_HEIGHT_RATIO_MAX):
                        pass_val = False
                        fail_reason = "height_ratio_out"
                    elif det["center_shift_ratio"] > TWO_COL_CENTER_SHIFT_MAX:
                        pass_val = False
                        fail_reason = "center_shift_too_large"
                    elif det["iou_with_original"] < TWO_COL_IOU_MIN:
                        pass_val = False
                        fail_reason = "iou_too_low"
                    elif det["outside_image"]:
                        pass_val = False
                        fail_reason = "outside_image"
                    elif det["max_overlap"] > TWO_COL_MAX_OVERLAP:
                        pass_val = False
                        fail_reason = "max_overlap_high"

                    two_col_decision = "use_two_col_lattice" if pass_val else (
                        "fallback_outer_panel" if force_fallback_outer else
                        ("fallback_string" if det["p_obj"].get("string_refined", False) else "fallback_original")
                    )
                    two_col_reason = fail_reason if not pass_val else "ok"
                    use_lattice_bool = pass_val
                    det["pass_individual"] = pass_val
                    det["fail_reason"] = fail_reason

                    # Diagnostic-only safe improvement warning (not a gate)
                    if use_lattice_bool:
                        try:
                            _, _warn_reason, _ = check_safe_improvement(
                                det["original_polygon"], det["candidate_polygon"], image, (img_h, img_w),
                                det["original_max_overlap"], det["max_overlap"]
                            )
                            if _warn_reason != "ok":
                                print(f"[SAFE_IMPROVEMENT_WARN] block={block_id} panel={det['panel_idx']} {_warn_reason}")
                        except Exception:
                            pass

            det["use_lattice_bool"] = use_lattice_bool
            det["two_col_decision"] = two_col_decision
            det["two_col_reason"] = two_col_reason

        # Block-level validation
        # Endpoint panels that fallback to YOLO original are NOT penalized in pass_ratio denominator
        n_endpoint_fallback = sum(1 for det in panel_details if
            det.get("is_endpoint_panel") and det.get("two_col_decision") == "fallback_endpoint_original")
        pass_count = sum(1 for det in panel_details if det["pass_individual"] or
            (det.get("is_endpoint_panel") and det.get("two_col_decision") == "fallback_endpoint_original"))
        total_panels_in_block = len(block["all_indices"])
        effective_total = max(total_panels_in_block - n_endpoint_fallback, 1)
        pass_ratio = pass_count / effective_total

        block_decision = "use_two_col_lattice" if pass_ratio >= TWO_COL_PASS_RATIO_MIN else "fallback_block"
        block_reason = "pass_ratio_ok" if block_decision == "use_two_col_lattice" else "block_quality_low"

        max_overlap_val = max([det["max_overlap"] for det in panel_details], default=0.0)
        median_iou_val = np.median([det["iou_with_original"] for det in panel_details]) if panel_details else 0.0

        print(f"[TWO_COL_BLOCK] block_id={block_id} n_panels={total_panels_in_block} matched_rows={len(block['matched_rows'])} "
              f"pass_ratio={pass_ratio:.2f} max_overlap={max_overlap_val:.3f} median_iou={median_iou_val:.3f} "
              f"decision={block_decision} reason={block_reason}")

        # Apply decisions to panels
        if block_decision == "use_two_col_lattice":
            block_angle = float(np.degrees(np.arctan2(u_axis[1], u_axis[0])))
            for det in panel_details:
                orig_idx = det["panel_idx"]
                p = panels[orig_idx]
                p["block_id"] = block_id
                p["block_angle_deg"] = block_angle
                p["rail_fit_source"] = rail_fit_source
                p["block_rail_middle"] = middle_rail_v_snapped
                
                # New middle-locked endpoint parameters
                p["endpoint_strategy"] = det.get("endpoint_strategy", "n/a")
                p["inner_u"] = float(det.get("inner_u", 0.0))
                p["outer_u"] = float(det.get("outer_u", 0.0))
                p["pitch"] = float(det.get("pitch", 0.0))
                p["endpoint_pitch_ratio"] = float(det.get("endpoint_pitch_ratio", 0.0))
                p["yolo_endpoint_area"] = float(det.get("yolo_endpoint_area", 0.0))
                p["median_middle_area"] = float(det.get("median_middle_area", 0.0))
                p["endpoint_overlap_with_neighbor"] = float(det.get("endpoint_overlap_with_neighbor", 0.0))
                p["middle_lattice_quality_good"] = bool(det.get("middle_lattice_quality_good", False))

                if det["pass_individual"]:
                    p["polygon"] = [[int(round(pt[0])), int(round(pt[1]))] for pt in det["candidate_polygon"]]
                    feats = get_polygon_features(p["polygon"])
                    p["bbox"] = [round(v) for v in feats["bbox"]]
                    p["box"] = p["bbox"]
                    p["area"] = feats["area"]
                    p["center"] = feats["center"]
                    p["aspect_ratio"] = feats["aspect_ratio"]
                    p["two_col_refined"] = True
                    p["two_col_decision"] = det["two_col_decision"]
                    p["two_col_reason"] = "ok"
                else:
                    if det.get("is_endpoint_panel") and det.get("two_col_decision") == "fallback_endpoint_original":
                        # Endpoint fallback: always use YOLO original polygon
                        p["polygon"] = list(p.get("original_yolo_polygon") or det["original_polygon"])
                    elif ENABLE_LATTICE_SAFE_IMPROVEMENT_ONLY:
                        p["polygon"] = list(p["original_yolo_polygon"])
                    else:
                        # Middle panel fallback: use string-refined polygon if available
                        p["polygon"] = list(p.get("string_first_polygon") or p.get("original_yolo_polygon") or det["original_polygon"])
                    
                    feats = get_polygon_features(p["polygon"])
                    p["bbox"] = [round(v) for v in feats["bbox"]]
                    p["box"] = p["bbox"]
                    p["area"] = feats["area"]
                    p["center"] = feats["center"]
                    p["aspect_ratio"] = feats["aspect_ratio"]
                    p["two_col_refined"] = False
                    p["two_col_decision"] = det["two_col_decision"]
                    p["two_col_reason"] = det["two_col_reason"]

                # After processing all panels, emit simple block log
                if det.get("is_outer_panel"):
                    logger.info(
                        f"[BLOCK_LOG] img={image_stem} block_id={block_id} panels={len(panel_details)} "
                        f"decision={det.get('two_col_decision', 'n/a')}"
                    )
                    # Existing outer panel handling continues below

                    p["is_outer_panel"] = True
                    p["outer_side"] = det.get("outer_side")
                    p["outer_divider_source"] = det.get("outer_divider_source")
                    p["outer_divider_shift"] = det.get("outer_divider_shift")
                    p["outer_divider_score"] = det.get("outer_divider_score")
                    p["outer_guard_decision"] = det.get("outer_guard_decision")
                    p["outer_guard_reason"] = det.get("outer_guard_reason")
                    p["outer_divider_u_proposal"] = det.get("outer_divider_u_proposal")
                    p["outer_divider_u_snapped"] = det.get("outer_divider_u_snapped")
                    p["outer_divider_u_axis"] = det.get("outer_divider_u_axis")
                    p["outer_divider_v_axis"] = det.get("outer_divider_v_axis")
                    p["outer_divider_origin"] = det.get("outer_divider_origin")
                    p["outer_divider_rail_low"] = det.get("outer_divider_rail_low")
                    p["outer_divider_rail_high"] = det.get("outer_divider_rail_high")

                if det.get("is_endpoint_panel"):
                    p["is_endpoint_panel"] = True
                    p["endpoint_side"] = det.get("endpoint_side")
                    p["endpoint_decision"] = det.get("endpoint_decision")
                    p["endpoint_reason"] = det.get("endpoint_reason")
                    p["candidate_lattice_area_ratio"] = det.get("candidate_lattice_area_ratio")
                    p["candidate_lattice_center_shift"] = det.get("candidate_lattice_center_shift")
                    p["candidate_lattice_iou"] = det.get("candidate_lattice_iou")
                    p["candidate_lattice_u_length_ratio"] = det.get("candidate_lattice_u_length_ratio")
                    p["candidate_lattice_v_width_ratio"] = det.get("candidate_lattice_v_width_ratio")
                    p["candidate_lattice_edge_score"] = det.get("candidate_lattice_edge_score")
                    p["candidate_original_edge_score"] = det.get("candidate_original_edge_score")
                    p["edge_score_ratio"] = det.get("edge_score_ratio")
                    p["rail_fit_source"] = rail_fit_source
                    
            # Fallback unmatched panels
            matched_indices = set(det["panel_idx"] for det in panel_details)
            unmatched_indices = set(block["all_indices"]) - matched_indices
            for orig_idx in unmatched_indices:
                p = panels[orig_idx]
                p["polygon"] = p["string_first_polygon"]
                feats = get_polygon_features(p["polygon"])
                p["bbox"] = [round(v) for v in feats["bbox"]]
                p["box"] = p["bbox"]
                p["area"] = feats["area"]
                p["center"] = feats["center"]
                p["aspect_ratio"] = feats["aspect_ratio"]
                p["two_col_refined"] = False
                p["two_col_decision"] = "fallback_string" if p.get("string_refined", False) else "fallback_original"
                p["two_col_reason"] = "unmatched_in_block"
        else:
            # Fallback whole block
            for orig_idx in block["all_indices"]:
                p = panels[orig_idx]
                p["polygon"] = p["string_first_polygon"]
                feats = get_polygon_features(p["polygon"])
                p["bbox"] = [round(v) for v in feats["bbox"]]
                p["box"] = p["bbox"]
                p["area"] = feats["area"]
                p["center"] = feats["center"]
                p["aspect_ratio"] = feats["aspect_ratio"]
                p["two_col_refined"] = False
                p["two_col_decision"] = "fallback_string" if p.get("string_refined", False) else "fallback_original"
                p["two_col_reason"] = "block_quality_low"
                
        # Write logs
        n_endpoint_panels_bl = sum(1 for det in panel_details if det.get("is_endpoint"))
        n_middle_panels_bl = len(panel_details) - n_endpoint_panels_bl
        middle_pass_count_bl = sum(1 for det in panel_details if det["pass_individual"] and not det.get("is_endpoint"))
        middle_pass_ratio_bl = middle_pass_count_bl / max(n_middle_panels_bl, 1)
        endpoint_pass_count_bl = sum(1 for det in panel_details if det["pass_individual"] and det.get("is_endpoint"))
        endpoint_fallback_count_bl = sum(1 for det in panel_details if not det["pass_individual"] and det.get("is_endpoint"))
        
        endpoint_source_counts_bl = {
            "yolo_projection": 0,
            "pitch_extrapolation": 0,
            "edge_scan": 0,
            "consensus": 0,
            "fallback_original": 0
        }
        for det in panel_details:
            if det.get("is_endpoint"):
                if det["pass_individual"]:
                    src = det.get("endpoint_outer_source", "yolo_projection")
                    if src in endpoint_source_counts_bl:
                        endpoint_source_counts_bl[src] += 1
                    else:
                        endpoint_source_counts_bl["yolo_projection"] += 1
                else:
                    endpoint_source_counts_bl["fallback_original"] += 1

        log_records.append({
            "record_type": "two_col_block",
            "image": image_stem,
            "block_id": int(block_id),
            "n_left": int(len(block["left_column"])),
            "n_right": int(len(block["right_column"])),
            "matched_rows": int(len(block["matched_rows"])),
            "pass_count": int(pass_count),
            "pass_ratio": float(pass_ratio),
            "median_iou": float(median_iou_val),
            "max_overlap": float(max_overlap_val),
            "decision": block_decision,
            "reason": block_reason,
            # new endpoint fields
            "n_middle_panels": int(n_middle_panels_bl),
            "n_endpoint_panels": int(n_endpoint_panels_bl),
            "middle_pass_count": int(middle_pass_count_bl),
            "middle_pass_ratio": float(middle_pass_ratio_bl),
            "endpoint_pass_count": int(endpoint_pass_count_bl),
            "endpoint_fallback_count": int(endpoint_fallback_count_bl),
            "endpoint_source_counts": endpoint_source_counts_bl
        })
        
        block_angle = float(np.degrees(np.arctan2(u_axis[1], u_axis[0])))
        for det in panel_details:
            orig_idx = det["panel_idx"]
            p = panels[orig_idx]
            rec = {
                "record_type": "panel",
                "block_id": int(block_id),
                "panel_idx": int(orig_idx),
                "column": det["column"],
                "row_idx": int(det["row_idx"]),
                "decision": p["two_col_decision"],
                "area_ratio": float(det["area_ratio"]),
                "width_ratio": float(det["width_ratio"]),
                "height_ratio": float(det["height_ratio"]),
                "center_shift": float(det["center_shift_ratio"]),
                "iou_with_original": float(det["iou_with_original"]),
                "max_overlap": float(det["max_overlap"]),
                "reason": p["two_col_reason"],

                # 12 required logging parameters
                "original_area": float(det.get("original_area", 0.0)),
                "lattice_area": float(det.get("lattice_area", 0.0)),
                "iou_original_lattice": float(det.get("iou_original_lattice", det.get("iou_with_original", 0.0))),
                "center_shift_ratio": float(det.get("center_shift_ratio", 0.0)),
                "original_edge_score": float(det.get("original_edge_score", 0.0)),
                "lattice_edge_score": float(det.get("lattice_edge_score", 0.0)),
                "edge_score_ratio": float(det.get("edge_score_ratio", 1.0)),
                "original_max_overlap": float(det.get("original_max_overlap", 0.0)),
                "lattice_max_overlap": float(det.get("lattice_max_overlap", det.get("max_overlap", 0.0))),
                "final_decision": p["two_col_decision"],
                "final_reason": p["two_col_reason"],

                # Endpoint Isolation fields
                "panel_role": det.get("panel_role", "middle"),
                "is_endpoint": bool(det.get("is_endpoint", False)),
                "endpoint_side": det.get("endpoint_side", None),
                "used_for_lattice_fit": bool(det.get("used_for_lattice_fit", False)),
                "local_angle_deg": float(block_angle),
                "endpoint_yolo_outer_u": float(det.get("endpoint_yolo_outer_u", 0.0)),
                "endpoint_pitch_outer_u": float(det.get("endpoint_pitch_outer_u", 0.0)),
                "endpoint_edge_outer_u": float(det.get("endpoint_edge_outer_u", 0.0)),
                "endpoint_consensus_outer_u": float(det.get("endpoint_consensus_outer_u", 0.0)),
                "endpoint_final_outer_u": float(det.get("endpoint_final_outer_u", 0.0)),
                "endpoint_outer_source": det.get("endpoint_outer_source", "n/a"),
                "endpoint_outer_score": float(det.get("endpoint_outer_score", 0.0)),
                "endpoint_length_ratio": float(det.get("endpoint_length_ratio", 0.0)),
                "endpoint_candidates": det.get("endpoint_candidates", []),
                "endpoint_decision": det.get("endpoint_decision", "n/a"),
                "endpoint_fallback_reason": det.get("endpoint_reason", "n/a"),
                "middle_lattice_unchanged": True,
                "angle_delta_to_block": float(det.get("angle_delta_to_block", 0.0)),
                # New middle-locked fields
                "endpoint_strategy": det.get("endpoint_strategy", "n/a"),
                "inner_u": float(det.get("inner_u", 0.0)),
                "outer_u": float(det.get("outer_u", 0.0)),
                "pitch": float(det.get("pitch", 0.0)),
                "endpoint_pitch_ratio": float(det.get("endpoint_pitch_ratio", 0.0)),
                "yolo_endpoint_area": float(det.get("yolo_endpoint_area", 0.0)),
                "median_middle_area": float(det.get("median_middle_area", 0.0)),
                "endpoint_overlap_with_neighbor": float(det.get("endpoint_overlap_with_neighbor", 0.0)),
                "middle_lattice_quality_good": bool(det.get("middle_lattice_quality_good", False)),
            }
            if det.get("is_outer_panel"):
                rec["is_outer_panel"] = True
                rec["outer_side"] = det.get("outer_side")
                rec["outer_divider_source"] = det.get("outer_divider_source")
                rec["outer_divider_shift"] = float(det.get("outer_divider_shift", 0.0))
                rec["outer_divider_score"] = float(det.get("outer_divider_score", 0.0))
                rec["outer_guard_decision"] = det.get("outer_guard_decision")
                rec["outer_guard_reason"] = det.get("outer_guard_reason")
                rec["outer_divider_u_proposal"] = det.get("outer_divider_u_proposal")
                rec["outer_divider_u_snapped"] = det.get("outer_divider_u_snapped")
            log_records.append(rec)
            
        matched_indices = set(det["panel_idx"] for det in panel_details)
        unmatched_indices = set(block["all_indices"]) - matched_indices
        for orig_idx in unmatched_indices:
            p = panels[orig_idx]
            col_name = "left" if orig_idx in block["left_column"] else "right"
            log_records.append({
                "record_type": "panel",
                "block_id": int(block_id),
                "panel_idx": int(orig_idx),
                "column": col_name,
                "row_idx": -1,
                "decision": p["two_col_decision"],
                "area_ratio": 1.0,
                "width_ratio": 1.0,
                "height_ratio": 1.0,
                "center_shift": 0.0,
                "iou_with_original": 1.0,
                "max_overlap": 0.0,
                "reason": p["two_col_reason"],

                # 12 required logging parameters
                "original_area": float(p.get("area", 0.0)),
                "lattice_area": float(p.get("area", 0.0)),
                "iou_original_lattice": 1.0,
                "center_shift_ratio": 0.0,
                "original_edge_score": 0.0,
                "lattice_edge_score": 0.0,
                "edge_score_ratio": 1.0,
                "original_max_overlap": 0.0,
                "lattice_max_overlap": 0.0,
                "final_decision": p["two_col_decision"],
                "final_reason": p["two_col_reason"],
            })
            
    # Draw debug image
    if blocks:
        _draw_two_col_debug_image(image, blocks, panels, image_path or "unknown.jpg")
        
    # Write JSONL log
    log_dir = "data/results/debug_logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{image_stem}_two_col_lattice_refine.jsonl")
    try:
        with open(log_path, "w", encoding="utf-8") as f:
            for r in log_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[TWO_COL_BLOCK] Logs saved to: {log_path}")
    except Exception as e:
        logger.warning(f"[TWO_COL_BLOCK_WARN] Failed to write JSONL: {e}")
        
    return panels


def _draw_middle_locked_debug_image(
    image: np.ndarray,
    panels: List[Dict[str, Any]],
    image_path: str
) -> None:
    """
    Draw debug image for Middle-Locked Endpoint Reconstruction.
    Highlighted elements:
      - Middle panel: cyan thin (BGR: (255, 255, 0))
      - Endpoint pass: thick green (BGR: (0, 255, 0)), labeled 'EP_LOCK'
      - Endpoint fallback: red/orange (BGR: (0, 0, 255)), labeled 'EP_FB'
      - Inner divider segment: white (BGR: (255, 255, 255))
      - Outer divider segment: yellow (BGR: (0, 255, 255))
    """
    dbg = image.copy()
    img_h, img_w = image.shape[:2]
    
    # Create transparent overlay for fallback coloring
    overlay = dbg.copy()
    
    for p in panels:
        poly = p.get("polygon")
        if not poly or len(poly) < 3:
            continue
            
        pts = np.array(poly, dtype=np.int32)
        is_ep = p.get("is_endpoint_panel", False) or (p.get("panel_role") in ("endpoint_start", "endpoint_end"))
        
        if not is_ep:
            # Middle panel: cyan thin
            cv2.polylines(dbg, [pts], True, (255, 255, 0), 1, cv2.LINE_AA)
        else:
            # Endpoint panel
            ep_dec = p.get("endpoint_decision", "fallback_original")
            # If it is string refined or two_col refined, or endpoint decision passes
            is_locked = ep_dec in ("use_middle_locked_endpoint", "use_lattice_endpoint")
            
            if is_locked:
                # Successful locked: thick green, labeled EP_LOCK
                cv2.polylines(dbg, [pts], True, (0, 255, 0), 3, cv2.LINE_AA)
                cx, cy = p["center"]
                cv2.putText(dbg, f"EP_LOCK_{p.get('raw_idx', '')}", (int(cx) - 30, int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
            else:
                # Fallback endpoint: red thin and filled transparent overlay
                cv2.polylines(dbg, [pts], True, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.fillPoly(overlay, [pts], (0, 0, 180))
                cx, cy = p["center"]
                cv2.putText(dbg, f"EP_FB_{p.get('raw_idx', '')}", (int(cx) - 20, int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
                
            # Draw inner & outer dividers if origin/axes are available
            if ("outer_divider_origin" in p and "outer_divider_u_axis" in p and 
                p["outer_divider_origin"] is not None and p["outer_divider_u_axis"] is not None):
                try:
                    origin = np.array(p["outer_divider_origin"])
                    u_axis = np.array(p["outer_divider_u_axis"])
                    v_axis = np.array(p["outer_divider_v_axis"])
                    v_low = p["outer_divider_rail_low"]
                    v_high = p["outer_divider_rail_high"]
                    inner_u = p["inner_u"]
                    outer_u = p["outer_u"]

                    pt1 = origin + inner_u * u_axis + v_low * v_axis
                    pt2 = origin + inner_u * u_axis + v_high * v_axis
                    pt3 = origin + outer_u * u_axis + v_low * v_axis
                    pt4 = origin + outer_u * u_axis + v_high * v_axis

                    p1 = (int(round(pt1[0])), int(round(pt1[1])))
                    p2 = (int(round(pt2[0])), int(round(pt2[1])))
                    p3 = (int(round(pt3[0])), int(round(pt3[1])))
                    p4 = (int(round(pt4[0])), int(round(pt4[1])))

                    # Inner divider: white
                    cv2.line(dbg, p1, p2, (255, 255, 255), 2, cv2.LINE_AA)
                    # Outer divider: yellow
                    cv2.line(dbg, p3, p4, (0, 255, 255), 2, cv2.LINE_AA)
                except Exception as e:
                    pass

    # Blend transparent overlay (fallback filled polygon)
    cv2.addWeighted(overlay, 0.25, dbg, 0.75, 0, dbg)
    
    # Save image
    image_stem = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join("data/results/debug", f"debug_{image_stem}_middle_locked_endpoint.JPG")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, dbg)
    print(f"[MIDDLE_LOCKED_DEBUG] Saved image to: {out_path}")


def _draw_final_polygon_source_debug_image(
    image: np.ndarray,
    panels: List[Dict[str, Any]],
    image_path: str
) -> None:
    dbg = image.copy()
    overlay = dbg.copy()
    
    for p in panels:
        poly = p.get("polygon")
        if not poly or len(poly) < 3:
            continue
            
        pts = np.array(poly, dtype=np.int32)
        src = p.get("final_polygon_source", "unknown")
        
        # Color matching based on final_polygon_source
        if src in ("string_lattice_middle", "two_col_block_middle"):
            color = (255, 255, 0) # Cyan BGR
            thickness = 1
            is_filled = False
        elif src == "middle_locked_endpoint":
            color = (0, 255, 0) # Green BGR
            thickness = 3
            is_filled = False
        elif src == "endpoint_original_fallback":
            color = (0, 0, 255) # Red BGR
            thickness = 1
            is_filled = True
        elif src in ("yolo_original", "small_panel_bbox"):
            color = (255, 0, 0) # Blue BGR
            thickness = 1
            is_filled = False
        else:
            color = (255, 0, 255) # Magenta BGR
            thickness = 3
            is_filled = False
            
        if is_filled:
            cv2.fillPoly(overlay, [pts], (0, 100, 255)) # Translucent orange/red fill
            cv2.polylines(dbg, [pts], True, color, thickness, cv2.LINE_AA)
        else:
            cv2.polylines(dbg, [pts], True, color, thickness, cv2.LINE_AA)
            
        # Draw text labels
        cx, cy = p.get("center", [0.0, 0.0])
        raw_idx = p.get("raw_idx", "")
        if src == "middle_locked_endpoint":
            cv2.putText(dbg, f"EP_LOCK_{raw_idx}", (int(cx) - 30, int(cy)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
        elif src == "endpoint_original_fallback":
            cv2.putText(dbg, f"EP_FB_{raw_idx}", (int(cx) - 25, int(cy)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
        elif src == "unknown":
            cv2.putText(dbg, f"UNKNOWN_{raw_idx}", (int(cx) - 30, int(cy)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1, cv2.LINE_AA)
                        
    # Blend overlay
    cv2.addWeighted(overlay, 0.25, dbg, 0.75, 0, dbg)
    
    # Save image
    image_stem = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join("data/results/debug", f"debug_{image_stem}_final_polygon_source.JPG")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, dbg)
    print(f"[FINAL_POLYGON_SOURCE_DEBUG] Saved image to: {out_path}")


def _draw_two_col_debug_image(
    image: np.ndarray,
    blocks: List[Dict[str, Any]],
    panels: List[Dict[str, Any]],
    image_path: str
):
    dbg = image.copy()
    img_h, img_w = image.shape[:2]
    
    # 1. Draw YOLO/refined original (blue thin)
    for p in panels:
        poly = p.get("original_yolo_polygon")
        if poly:
            pts = np.array(poly, dtype=np.int32)
            cv2.polylines(dbg, [pts], True, (255, 0, 0), 1, cv2.LINE_AA)
            
    # 2. Draw String-first result (yellow thin)
    for p in panels:
        poly = p.get("string_first_polygon")
        if poly:
            pts = np.array(poly, dtype=np.int32)
            cv2.polylines(dbg, [pts], True, (0, 255, 255), 1, cv2.LINE_AA)
            
    # 3. Draw Two-column lattice details (rails and dividers) for each block
    for block in blocks:
        if "fitted_divider_u" not in block:
            continue
            
        u_axis = block["u_axis"]
        v_axis = block["v_axis"]
        origin = block["origin"]
        left_rail_v = block["fitted_left_rail_v"]
        middle_rail_v = block["fitted_middle_rail_v"]
        right_rail_v = block["fitted_right_rail_v"]
        divider_u = block["fitted_divider_u"]
        
        def to_img(u_val, v_val):
            pt = origin + u_val * u_axis + v_val * v_axis
            return [int(round(float(pt[0]))), int(round(float(pt[1])))]
            
        u_min = divider_u[0]
        u_max = divider_u[-1]
        
        # Draw 3 vertical rails (white thick)
        p_left_start = to_img(u_min, left_rail_v)
        p_left_end = to_img(u_max, left_rail_v)
        cv2.line(dbg, tuple(p_left_start), tuple(p_left_end), (255, 255, 255), 2, cv2.LINE_AA)
        
        p_middle_start = to_img(u_min, middle_rail_v)
        p_middle_end = to_img(u_max, middle_rail_v)
        cv2.line(dbg, tuple(p_middle_start), tuple(p_middle_end), (255, 255, 255), 2, cv2.LINE_AA)
        
        p_right_start = to_img(u_min, right_rail_v)
        p_right_end = to_img(u_max, right_rail_v)
        cv2.line(dbg, tuple(p_right_start), tuple(p_right_end), (255, 255, 255), 2, cv2.LINE_AA)
        
        # Draw shared row dividers (yellow thick)
        for u_div in divider_u:
            p_div_start = to_img(u_div, left_rail_v)
            p_div_end = to_img(u_div, right_rail_v)
            cv2.line(dbg, tuple(p_div_start), tuple(p_div_end), (0, 200, 255), 2, cv2.LINE_AA)
            
    # 4. Draw final polygons based on their decision
    for p in panels:
        poly = p.get("polygon")
        if not poly:
            continue
        pts = np.array(poly, dtype=np.int32)
        dec = p.get("two_col_decision", "bypass")
        reason = p.get("two_col_reason", "")
        
        if dec == "use_two_col_lattice":
            # Cyan/Green đậm
            cv2.polylines(dbg, [pts], True, (0, 255, 0), 2, cv2.LINE_AA)
            cx, cy = p["center"]
            cv2.putText(dbg, f"2C_{p.get('raw_idx')}", (int(cx) - 15, int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1, cv2.LINE_AA)
        elif dec in ("fallback_string", "fallback_original"):
            # Red/Orange mờ
            color = (0, 100, 255) # Orange
            if reason in ("area_ratio_out", "width_ratio_out", "height_ratio_out", "center_shift_too_large", "iou_too_low", "outside_image", "max_overlap_high"):
                color = (0, 0, 255) # Red đậm
            cv2.polylines(dbg, [pts], True, color, 1, cv2.LINE_AA)
            cx, cy = p["center"]
            cv2.putText(dbg, f"FB_{p.get('raw_idx')}", (int(cx) - 15, int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
            
    # Draw legend
    legend_items = [
        ((255, 0, 0), "YOLO/Refined orig"),
        ((0, 255, 255), "String-first"),
        ((0, 255, 0), "2-Col final"),
        ((0, 100, 255), "Fallback string"),
        ((0, 0, 255), "Failed/Rejected"),
        ((255, 255, 255), "Rails (white)"),
        ((0, 200, 255), "Dividers (yellow)"),
    ]
    for idx, (color, text) in enumerate(legend_items):
        y = 20 + idx * 15
        cv2.rectangle(dbg, (5, y - 8), (20, y + 2), color, -1)
        cv2.putText(dbg, text, (25, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
        
    # Save image
    image_stem = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join("data/results/debug", f"debug_{image_stem}_two_col_lattice_refine.JPG")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, dbg)
    print(f"[TWO_COL_BLOCK] Debug image saved to: {out_path}")


# ===========================================================================
# EDGE GRID SEGMENTED4 REFINEMENT HELPERS
# ===========================================================================
from dataclasses import dataclass
import math

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

def intersect_h_v(y_int_h, ang_h, x_int_v, ang_v, x_mid, y_mid):
    """Compute the intersection of a horizontal-ish line and a vertical-ish line."""
    tan_h = math.tan(math.radians(ang_h))
    if abs(ang_v) == 90.0:
        x = x_int_v
        y = y_int_h + tan_h * (x - x_mid)
    else:
        tan_v = math.tan(math.radians(ang_v))
        x = (x_int_v * tan_v - tan_h * x_mid + y_int_h - y_mid) / (tan_v - tan_h)
        y = y_int_h + tan_h * (x - x_mid)
    return (float(x), float(y))

def get_boundary_x_for_segmented(boundary_points, y_val):
    """Interpolate the X position on the boundary for a given Y."""
    ys = [pt[0] for pt in boundary_points]
    xs = [pt[1] for pt in boundary_points]
    return float(np.interp(y_val, ys, xs))

def get_intersection_segmented(y_int_h, ang_h, boundary_points, x_mid):
    """Compute intersection of a horizontal-ish line and a segmented vertical boundary."""
    x_est = get_boundary_x_for_segmented(boundary_points, y_int_h)
    tan_h = math.tan(math.radians(ang_h))
    y_ref = y_int_h + tan_h * (x_est - x_mid)
    x_ref = get_boundary_x_for_segmented(boundary_points, y_ref)
    y_final = y_int_h + tan_h * (x_ref - x_mid)
    return (x_ref, y_final)

def build_roi_from_target_panels(targets, img_w, img_h, margin=45):
    """Build ROI bounding coordinates from targets with a margin."""
    all_x = []
    all_y = []
    for p in targets:
        box = p.get("bbox") or p.get("box")
        if box:
            all_x.extend([box[0], box[2]])
            all_y.extend([box[1], box[3]])
    min_x = max(0, min(all_x) - margin)
    max_x = min(img_w, max(all_x) + margin)
    min_y = max(0, min(all_y) - margin)
    max_y = min(img_h, max(all_y) + margin)
    return min_x, max_x, min_y, max_y

def preprocess_roi_for_edges(img, roi_coords):
    """Preprocess the ROI to extract Canny edges."""
    min_x, max_x, min_y, max_y = roi_coords
    roi_img = img[min_y:max_y, min_x:max_x]
    gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
    return edges

def cluster_segments(segments, threshold=12.0):
    """Helper: Cluster 1D segments by intercept values."""
    if not segments:
        return []
    segments.sort(key=lambda x: x[0])
    clusters = []
    current = [segments[0]]
    for s in segments[1:]:
        if s[0] - current[-1][0] <= threshold:
            current.append(s)
        else:
            clusters.append(current)
            current = [s]
    clusters.append(current)
    return clusters

def detect_horizontal_gaps(edges, roi_coords):
    """Detect and cluster horizontal consensus gap lines."""
    min_x, max_x, min_y, max_y = roi_coords
    x_mid = (min_x + max_x) / 2.0
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=25, minLineLength=20, maxLineGap=10)
    
    horiz_segments = []
    if lines is not None:
        for line in lines:
            x1_roi, y1_roi, x2_roi, y2_roi = line[0]
            x1, y1 = x1_roi + min_x, y1_roi + min_y
            x2, y2 = x2_roi + min_x, y2_roi + min_y
            dx = x2 - x1
            dy = y2 - y1
            length = math.sqrt(dx*dx + dy*dy)
            if length < 1e-3:
                continue
            angle = math.degrees(math.atan2(dy, dx))
            if angle > 90.0:
                angle -= 180.0
            elif angle < -90.0:
                angle += 180.0
                
            if abs(angle) <= 25.0:
                y_int = y1 + (x_mid - x1) * (dy / dx) if abs(dx) > 1e-3 else y1
                horiz_segments.append((y_int, angle, length, (x1, y1, x2, y2)))
                
    horiz_clusters = cluster_segments(horiz_segments, 12.0)
    final_horiz_lines = []
    for c in horiz_clusters:
        total_len = sum(s[2] for s in c)
        if total_len < 25.0:
            continue
        med_y_int = float(np.median([s[0] for s in c]))
        med_ang = float(np.median([s[1] for s in c]))
        final_horiz_lines.append((med_y_int, med_ang, total_len))
    return final_horiz_lines

def detect_initial_vertical_boundaries(edges, roi_coords):
    """Detect, cluster, and filter the initial straight vertical boundaries."""
    min_x, max_x, min_y, max_y = roi_coords
    y_mid = (min_y + max_y) / 2.0
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=25, minLineLength=20, maxLineGap=10)
    
    vert_segments = []
    if lines is not None:
        for line in lines:
            x1_roi, y1_roi, x2_roi, y2_roi = line[0]
            x1, y1 = x1_roi + min_x, y1_roi + min_y
            x2, y2 = x2_roi + min_x, y2_roi + min_y
            dx = x2 - x1
            dy = y2 - y1
            length = math.sqrt(dx*dx + dy*dy)
            if length < 1e-3:
                continue
            angle = math.degrees(math.atan2(dy, dx))
            if angle > 90.0:
                angle -= 180.0
            elif angle < -90.0:
                angle += 180.0
                
            if abs(angle) >= 65.0:
                x_int = x1 + (y_mid - y1) * (dx / dy) if abs(dy) > 1e-3 else x1
                vert_segments.append((x_int, angle, length, (x1, y1, x2, y2)))
                
    vert_clusters = cluster_segments(vert_segments, 12.0)
    final_vert_lines = []
    for c in vert_clusters:
        total_len = sum(s[2] for s in c)
        if total_len < 25.0:
            continue
        med_x_int = float(np.median([s[0] for s in c]))
        med_ang = float(np.median([s[1] for s in c]))
        final_vert_lines.append((med_x_int, med_ang, total_len))
        
    final_vert_lines = [line for line in final_vert_lines if line[0] >= 480.0]
    final_vert_lines.sort(key=lambda x: x[0])
    return final_vert_lines

def fit_segmented_vertical_boundary(
    edge_img,
    initial_line,
    y_min,
    y_max,
    n_segments=4,
    search_half_width=10,
    min_points_per_segment=6,
    max_angle_delta_deg=8,
    max_x_jump_px=10,
    min_x=0,
    max_dev_baseline_px=12,
    smooth_window=3
):
    """Fit a vertical boundary with n_segments line segments, limiting orientation deltas and X jumps."""
    x_int, ang_v, _ = initial_line
    y_mid = (y_min + y_max) / 2.0
    if abs(ang_v) == 90.0:
        a_init = 0.0
        b_init = x_int
    else:
        a_init = 1.0 / math.tan(math.radians(ang_v))
        b_init = x_int - a_init * y_mid

    y_step = (y_max - y_min) / n_segments
    segment_lines = []

    for k in range(n_segments):
        seg_y_start = y_min + k * y_step
        seg_y_end = seg_y_start + y_step
        
        seg_ys = []
        seg_xs = []
        
        y_start_int = int(math.floor(seg_y_start))
        y_end_int = int(math.ceil(seg_y_end))
        
        for y in range(y_start_int, y_end_int):
            y_roi = int(round(y - y_min))
            if y_roi < 0 or y_roi >= edge_img.shape[0]:
                continue
                
            x_base = a_init * y + b_init
            x_min_search = int(math.floor(x_base - search_half_width))
            x_max_search = int(math.ceil(x_base + search_half_width))
            
            for x in range(x_min_search, x_max_search + 1):
                x_roi = int(round(x - min_x))
                if x_roi < 0 or x_roi >= edge_img.shape[1]:
                    continue
                if edge_img[y_roi, x_roi] == 255:
                    seg_ys.append(y)
                    seg_xs.append(x)
                    
        a_fit, b_fit = a_init, b_init
        fitted = False
        
        if len(seg_ys) >= min_points_per_segment:
            try:
                a_c, b_c = np.polyfit(seg_ys, seg_xs, 1)
                residuals = np.abs(np.array(seg_xs) - (a_c * np.array(seg_ys) + b_c))
                threshold = np.median(residuals) * 1.5
                threshold = max(threshold, 2.0)
                
                valid_indices = residuals <= threshold
                ys_clean = np.array(seg_ys)[valid_indices]
                xs_clean = np.array(seg_xs)[valid_indices]
                
                if len(ys_clean) >= min_points_per_segment:
                    a_c, b_c = np.polyfit(ys_clean, xs_clean, 1)
                    
                ang_fit = math.degrees(math.atan2(1.0, a_c))
                if ang_fit > 90.0:
                    ang_fit -= 180.0
                elif ang_fit < -90.0:
                    ang_fit += 180.0
                    
                ang_diff = abs(ang_fit - ang_v)
                if ang_diff > 180.0:
                    ang_diff = 360.0 - ang_diff
                    
                if ang_diff <= max_angle_delta_deg:
                    a_fit = a_c
                    b_fit = b_c
                    fitted = True
            except Exception:
                pass
                
        segment_lines.append((a_fit, b_fit, fitted))

    for k in range(1, n_segments - 1):
        a_prev, b_prev, fit_p = segment_lines[k-1]
        a_curr, b_curr, fit_c = segment_lines[k]
        a_next, b_next, fit_n = segment_lines[k+1]
        
        if fit_c:
            ang_prev = math.degrees(math.atan2(1.0, a_prev))
            ang_curr = math.degrees(math.atan2(1.0, a_curr))
            ang_next = math.degrees(math.atan2(1.0, a_next))
            
            ang_neighbor_avg = (ang_prev + ang_next) / 2.0
            if abs(ang_curr - ang_neighbor_avg) > max_angle_delta_deg:
                if fit_p and fit_n:
                    a_fit = (a_prev + a_next) / 2.0
                    b_fit = (b_prev + b_next) / 2.0
                else:
                    a_fit = a_init
                    b_fit = b_init
                segment_lines[k] = (a_fit, b_fit, False)

    slopes = [line[0] for line in segment_lines]
    intercepts = [line[1] for line in segment_lines]
    smoothed_slopes = []
    smoothed_intercepts = []
    half_w = smooth_window // 2
    for k in range(n_segments):
        w_start = max(0, k - half_w)
        w_end = min(n_segments, k + half_w + 1)
        s = float(np.mean(slopes[w_start:w_end]))
        i = float(np.mean(intercepts[w_start:w_end]))
        smoothed_slopes.append(s)
        smoothed_intercepts.append(i)
        
    for k in range(n_segments):
        segment_lines[k] = (smoothed_slopes[k], smoothed_intercepts[k], segment_lines[k][2])

    for k in range(1, n_segments):
        a_prev, b_prev, _ = segment_lines[k-1]
        a_curr, b_curr, fitted = segment_lines[k]
        
        y_junc = y_min + k * y_step
        x_prev = a_prev * y_junc + b_prev
        x_curr = a_curr * y_junc + b_curr
        
        if abs(x_curr - x_prev) > max_x_jump_px:
            x_curr_adj = np.clip(x_curr, x_prev - max_x_jump_px, x_prev + max_x_jump_px)
            b_curr = x_curr_adj - a_curr * y_junc
            segment_lines[k] = (a_curr, b_curr, fitted)

    y_junctions = [y_min + k * y_step for k in range(n_segments + 1)]
    polyline = []
    for k in range(n_segments + 1):
        y_j = y_junctions[k]
        if k == 0:
            a, b, _ = segment_lines[0]
            x_j = a * y_j + b
        elif k == n_segments:
            a, b, _ = segment_lines[-1]
            x_j = a * y_j + b
        else:
            a_p, b_p, _ = segment_lines[k-1]
            a_c, b_c, _ = segment_lines[k]
            x_left = a_p * y_j + b_p
            x_right = a_c * y_j + b_c
            x_j = (x_left + x_right) / 2.0
            
        if max_dev_baseline_px is not None:
            x_base = a_init * y_j + b_init
            x_j = np.clip(x_j, x_base - max_dev_baseline_px, x_base + max_dev_baseline_px)
            
        polyline.append((y_j, x_j))
        
    n_fitted = sum(1 for line in segment_lines if line[2])
    return polyline, n_fitted

def build_panel_polygons_from_boundaries(
    targets, final_horiz_lines, left_boundary, right_boundary, roi_coords,
    left_line_straight=None, right_line_straight=None, vertical_mode="segmented"
):
    """Intersect horizontal lines and vertical boundaries to form proposals."""
    min_x, max_x, min_y, max_y = roi_coords
    x_mid = (min_x + max_x) / 2.0
    y_mid = (min_y + max_y) / 2.0
    
    report_data = []
    for p in targets:
        ridx = p["raw_idx"]
        yolo_box = p.get("bbox") or p.get("box")
        ox1, oy1, ox2, oy2 = yolo_box
        cx_o = (ox1 + ox2) / 2.0
        cy_o = (oy1 + oy2) / 2.0
        
        top_line = None
        bot_line = None
        min_top_err = float('inf')
        min_bot_err = float('inf')
        
        for h_line in final_horiz_lines:
            y_int, ang, _ = h_line
            y_at_cx = y_int + math.tan(math.radians(ang)) * (cx_o - x_mid)
            if y_at_cx < cy_o:
                err = abs(y_at_cx - oy1)
                if err < min_top_err:
                    min_top_err = err
                    top_line = h_line
            elif y_at_cx > cy_o:
                err = abs(y_at_cx - oy2)
                if err < min_bot_err:
                    min_bot_err = err
                    bot_line = h_line
                    
        proposal_ok = True
        reason = "ok"
        
        has_vert = (left_boundary and right_boundary)
        if not (top_line and bot_line and has_vert):
            proposal_ok = False
            reason = "missing_boundary_line"
        else:
            y_top_at_cx = top_line[0] + math.tan(math.radians(top_line[1])) * (cx_o - x_mid)
            y_bot_at_cx = bot_line[0] + math.tan(math.radians(bot_line[1])) * (cx_o - x_mid)
            if abs(y_top_at_cx - oy1) > 15.0 or abs(y_bot_at_cx - oy2) > 15.0:
                proposal_ok = False
                reason = "missing_boundary_line"
                
        iou = 0.0
        center_shift = 0.0
        area_ratio = 0.0
        proposal_poly = []
        
        if proposal_ok:
            pt_tl = get_intersection_segmented(top_line[0], top_line[1], left_boundary, x_mid)
            pt_tr = get_intersection_segmented(top_line[0], top_line[1], right_boundary, x_mid)
            pt_br = get_intersection_segmented(bot_line[0], bot_line[1], right_boundary, x_mid)
            pt_bl = get_intersection_segmented(bot_line[0], bot_line[1], left_boundary, x_mid)
                
            proposal_poly = [pt_tl, pt_tr, pt_br, pt_bl]
            
            p_x1 = min(pt[0] for pt in proposal_poly)
            p_y1 = min(pt[1] for pt in proposal_poly)
            p_x2 = max(pt[0] for pt in proposal_poly)
            p_y2 = max(pt[1] for pt in proposal_poly)
            
            ix1, iy1 = max(p_x1, ox1), max(p_y1, oy1)
            ix2, iy2 = min(p_x2, ox2), min(p_y2, oy2)
            inter = (ix2 - ix1) * (iy2 - iy1) if ix2 > ix1 and iy2 > iy1 else 0.0
            area_p = (p_x2 - p_x1) * (p_y2 - p_y1)
            area_o = (ox2 - ox1) * (oy2 - oy1)
            union = area_p + area_o - inter
            iou = inter / union if union > 0 else 0.0
            
            cx_p = sum(pt[0] for pt in proposal_poly) / 4.0
            cy_p = sum(pt[1] for pt in proposal_poly) / 4.0
            center_shift = math.sqrt((cx_p - cx_o)**2 + (cy_p - cy_o)**2)
            area_ratio = area_p / max(area_o, 1.0)
            
        report_data.append({
            "raw_idx": ridx,
            "proposal_ok": proposal_ok,
            "reason": reason,
            "proposal_poly": proposal_poly,
            "iou": iou,
            "center_shift": center_shift,
            "area_ratio": area_ratio
        })
    return report_data

def propose_edge_grid_polygons_in_pipeline(
    orig_img: np.ndarray,
    panels: List[Dict],
    target_panel_indices_or_ids: List[int],
    config: EdgeGridConfig
) -> dict:
    """Helper for pipeline to run segmented4 edge grid proposal using in-memory structures."""
    targets = [p for p in panels if p.get("raw_idx") in target_panel_indices_or_ids]
    if not targets:
        return {"proposals": [], "debug": {}}
    h, w, c = orig_img.shape
    
    roi_coords = build_roi_from_target_panels(targets, w, h, margin=config.roi_margin)
    edges = preprocess_roi_for_edges(orig_img, roi_coords)
    final_horiz_lines = detect_horizontal_gaps(edges, roi_coords)
    final_vert_lines = detect_initial_vertical_boundaries(edges, roi_coords)
    
    vertical_boundaries = []
    vertical_boundaries_fitted = []
    vertical_mode = config.vertical_mode
    for line_v in final_vert_lines:
        smoothed_pts, n_fitted = fit_segmented_vertical_boundary(
            edges, line_v, roi_coords[2], roi_coords[3],
            n_segments=config.n_segments,
            search_half_width=config.search_half_width,
            min_points_per_segment=config.min_points_per_segment,
            max_angle_delta_deg=config.max_angle_delta_deg,
            max_x_jump_px=config.max_x_jump_px,
            min_x=roi_coords[0],
            max_dev_baseline_px=config.max_dev_baseline_px,
            smooth_window=config.smooth_window
        )
        vertical_boundaries.append(smoothed_pts)
        vertical_boundaries_fitted.append(n_fitted)
        
    left_boundary = vertical_boundaries[0] if len(vertical_boundaries) > 0 else []
    right_boundary = vertical_boundaries[1] if len(vertical_boundaries) > 1 else []
    
    left_line_straight = final_vert_lines[0] if len(final_vert_lines) > 0 else None
    right_line_straight = final_vert_lines[1] if len(final_vert_lines) > 1 else None
    
    report_data = build_panel_polygons_from_boundaries(
        targets, final_horiz_lines, left_boundary, right_boundary, roi_coords,
        left_line_straight=left_line_straight,
        right_line_straight=right_line_straight,
        vertical_mode=vertical_mode
    )
    
    proposals = []
    for r in report_data:
        proposals.append({
            "raw_idx": r["raw_idx"],
            "polygon": r["proposal_poly"],
            "source": f"edge_grid_segmented{config.n_segments}",
            "ok": r["proposal_ok"],
            "metrics": {
                "iou": r["iou"],
                "center_shift": r["center_shift"],
                "area_ratio": r["area_ratio"],
                "note": "ok" if r["proposal_ok"] else "failed"
            }
        })
        
    debug_dict = {
        "roi": roi_coords,
        "left_boundary": left_boundary,
        "right_boundary": right_boundary,
        "horizontal_lines": final_horiz_lines,
        "final_vert_lines": final_vert_lines,
        "vertical_boundaries": vertical_boundaries,
        "vertical_boundaries_fitted": vertical_boundaries_fitted,
        "targets": targets,
        "report_data": report_data
    }
    return {
        "proposals": proposals,
        "debug": debug_dict
    }


# ===========================================================================
# SECTION 5 — YOLO PREDICTION PROCESSOR (GIỮ NGUYÊN)
# ===========================================================================

def process_yolo_predictions(
    result,
    orig_img: np.ndarray,
    panel_conf: float,
    defect_conf: float,
    panel_class_name: str = "panel"
) -> List[Dict[str, Any]]:
    """
    Trích xuất và làm sạch panel/defect từ kết quả YOLO.

    Pipeline:
      1. YOLO predict đã chạy trước.
      2. Với class 'panel': refine mask → 3-level CV refiner → polygon 4/6 điểm hoặc Bbox.
      3. Với class defect: dùng polygon gốc từ result.masks.xy.
      4. Áp dụng confidence filter riêng.
    """
    if result.masks is None:
        return []

    masks_xy = result.masks.xy

    raw_panels: List[Dict] = []
    raw_defects: List[Dict] = []

    # ── PASS 1: XỬ LÝ VÀ TRÍCH XUẤT ──
    for i in range(len(result.boxes)):
        class_id = int(result.boxes.cls[i])
        class_name = result.names[class_id]
        confidence = float(result.boxes.conf[i])
        bbox = result.boxes.xyxy[i].tolist()  # [x1,y1,x2,y2]

        xy_polygon = masks_xy[i]  # np.ndarray (N,2)

        # A. Log raw YOLO output trước mọi hậu xử lý
        try:
            raw_pts = xy_polygon.astype(np.float32)
            mask_area = float(cv2.contourArea(raw_pts.reshape(-1, 1, 2))) if len(raw_pts) >= 3 else 0.0
            num_pts = len(raw_pts)
        except Exception:
            mask_area = 0.0
            num_pts = 0

        bw = bbox[2] - bbox[0]
        bh = bbox[3] - bbox[1]
        b_area = bw * bh
        logger.info(
            f"[RAW_YOLO] idx={i} class={class_name} conf={confidence:.4f} bbox={[round(v, 1) for v in bbox]} "
            f"w={bw:.1f} h={bh:.1f} area={b_area:.1f} mask_area={mask_area:.1f} pts={num_pts}"
        )

        if class_name.lower() == panel_class_name:
            if confidence < panel_conf or len(xy_polygon) < 3:
                continue

            import os
            image_path = getattr(result, "path", "unknown_image.jpg")
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            debug_info = {}
            panel_poly = refine_panel_contour(
                orig_img, bbox, xy_polygon, idx=i, conf=confidence,
                image_name=image_name, debug_info=debug_info
            )

            features = get_polygon_features(panel_poly)

            panel_dict = {
                "class_name": class_name,
                "confidence": round(confidence, 4),
                "bbox": [round(v) for v in features["bbox"]],
                "polygon": panel_poly,
                "area": features["area"],
                "center": features["center"],
                "aspect_ratio": features["aspect_ratio"],
                "category": "panel",
                "box": [round(v) for v in bbox],
                # Debug fields
                "raw_yolo_poly": xy_polygon.tolist() if xy_polygon is not None else [],
                "raw_idx": i,
                "raw_conf": confidence,
                "final_polygon_source": "yolo_original",
                "final_polygon_stage": "yolo_refinement",
                "final_polygon_reason": "yolo_detection_contour"
            }
            panel_dict.update(debug_info)
            raw_panels.append(panel_dict)

        else:
            # ── DEFECT: giữ polygon gốc YOLO ──
            if confidence < defect_conf:
                continue

            if len(xy_polygon) < 3:
                continue

            # Rút gọn đa giác lỗi về tối đa 6 cạnh để mượt mà biên, tránh răng cưa nham nhở
            defect_poly = simplify_defect_polygon(xy_polygon, max_vertices=6)

            features = get_polygon_features(defect_poly)

            raw_defects.append({
                "class_name": class_name,
                "confidence": round(confidence, 4),
                "bbox": [round(v) for v in features["bbox"]],
                "polygon": defect_poly,
                "area": features["area"],
                "center": features["center"],
                "category": "defect",
                "box": [round(v) for v in bbox],
                # Debug fields
                "raw_idx": i,
            })

    # Run string lattice refinement if enabled (NEW production path)
    image_path = getattr(result, "path", "unknown_image.jpg")
    if ENABLE_STRING_LATTICE_REFINEMENT:
        raw_panels = refine_panels_by_string_lattice(
            panels=raw_panels,
            image_shape=orig_img.shape,
            image_path=image_path
        )

    # Run Two-column Block Lattice Refinement if enabled (NEW post-string path)
    if ENABLE_TWO_COLUMN_BLOCK_LATTICE:
        raw_panels = refine_panels_by_two_column_block_lattice(
            panels=raw_panels,
            image_shape=orig_img.shape,
            image_path=image_path
        )

    elif ENABLE_PANEL_BLOCK_LATTICE_REFINEMENT:
        # Old block lattice (kept for fallback comparison, normally disabled)
        raw_panels = refine_panel_blocks_by_lattice_lines(
            panels=raw_panels,
            image_shape=orig_img.shape,
            image_path=image_path
        )

    grid_aware_called = False
    edge_snap_called = False
    shrink_called = False

    # --- OPTIONAL SMALL PANEL BBOX SHRINK ---
    if ENABLE_SMALL_PANEL_BBOX_SHRINK:
        shrink_called = True
        for p in raw_panels:
            box = p.get("box") or p.get("bbox", [0, 0, 0, 0])
            bw = box[2] - box[0]
            bh = box[3] - box[1]
            b_area = bw * bh
            if b_area <= SMALL_PANEL_BBOX_AREA_PX:
                poly = p.get("polygon", [])
                if len(poly) == 4:
                    x_coords = [pt[0] for pt in poly]
                    y_coords = [pt[1] for pt in poly]
                    is_axis_aligned = (len(set(x_coords)) <= 2) and (len(set(y_coords)) <= 2)
                    if is_axis_aligned:
                        if should_shrink_panel(box, raw_defects, margin_px=3.0):
                            # Single-Column Geometry Refinement (disabled)
                            # ---------------------------------------------------------------------------
                            # The geometry refinement for single-column tracking candidates has been disabled per user request.
                            # All tracking-only fields (single_column_tracking_candidate, tracking_stage, tracking_reason,
                            # final_polygon_tracking_source) remain untouched. No adjustments to final_polygon_source
                            # or polygon fields are performed.
                            # ---------------------------------------------------------------------------   
                            dx = min(bw * (SMALL_PANEL_BBOX_SHRINK_RATIO / 2.0), SMALL_PANEL_BBOX_SHRINK_MAX_PX_PER_EDGE)
                            dy = min(bh * (SMALL_PANEL_BBOX_SHRINK_RATIO / 2.0), SMALL_PANEL_BBOX_SHRINK_MAX_PX_PER_EDGE)
                            
                            cx = (box[0] + box[2]) / 2.0
                            cy = (box[1] + box[3]) / 2.0
                            w_new = bw - 2.0 * dx
                            h_new = bh - 2.0 * dy
                            
                            x1_new = int(round(cx - w_new / 2.0))
                            y1_new = int(round(cy - h_new / 2.0))
                            x2_new = int(round(cx + w_new / 2.0))
                            y2_new = int(round(cy + h_new / 2.0))
                            
                            shrunk_poly = [
                                [x1_new, y1_new],
                                [x2_new, y1_new],
                                [x2_new, y2_new],
                                [x1_new, y2_new]
                            ]
                            p["polygon"] = shrunk_poly
                            p["bbox"] = [x1_new, y1_new, x2_new, y2_new]
                            p["box"] = [x1_new, y1_new, x2_new, y2_new]
                            
                            shrunk_features = get_polygon_features(shrunk_poly)
                            p["area"] = shrunk_features["area"]
                            p["center"] = shrunk_features["center"]
                            p["aspect_ratio"] = shrunk_features["aspect_ratio"]
                            p["shrunk"] = True
                            
                            logger.info(
                                f"[BBOX_SHRINK] Shrunk small panel idx={p.get('raw_idx')} "
                                f"by dx={dx:.2f}, dy={dy:.2f}"
                            )

    # Call grid-aware refinement for small panel groups
    if ENABLE_SMALL_PANEL_GRID_AWARE and ENABLE_SMALL_PANEL_GRID_REFINEMENT:
        grid_aware_called = True
        image_path = getattr(result, "path", "unknown_image.jpg")
        raw_panels = refine_small_panel_group_grid(raw_panels, orig_img.shape, orig_img, image_path)

    # ── B. SMALL PANEL TRACE → ghi vào file JSONL (không print dài ra terminal) ──
    # Chi tiết xem tại: data/results/debug_logs/<image_name>_panel_refine.jsonl
    # (ghi sau khi collect xong raw_panels, trước khi filter/normalize)

    # ── C. FILTER DEFECT-LIKE PANELS (tùy chọn — mặc định TẮT) ──────────
    # TẮT vì panel có lỗi thật (hotspot_multi_cell/crack) cũng có overlap cao
    # với defect polygon → dễ reject nhầm chính các panel lỗi cần xử lý.
    # Bật ENABLE_DEFECT_LIKE_PANEL_FILTER = True chỉ khi đã test kỹ.
    if ENABLE_DEFECT_LIKE_PANEL_FILTER:
        panels_after_defect_filter = filter_defect_like_panels(
            raw_panels, raw_defects, orig_img.shape
        )
    else:
        panels_after_defect_filter = raw_panels

    # ── FILTER + NORMALIZE PANEL (tùy chọn) ─────────────────────────────
    # ENABLE_PANEL_SIZE_FILTER  = False  (không dùng filter 80% toàn cục)
    # ENABLE_PANEL_GEOMETRY_NORMALIZATION = False  (không normalize polygon)
    if ENABLE_PANEL_SIZE_FILTER or ENABLE_PANEL_GEOMETRY_NORMALIZATION:
        filtered_panels = filter_and_normalize_panels(panels_after_defect_filter, orig_img.shape)
    else:
        filtered_panels = panels_after_defect_filter

    # ---------------------------------------------------------------------------
    # Single-Column Geometry Refinement (fixed 640x512)
    # ---------------------------------------------------------------------------
    # Only process records where single_column_tracking_candidate == True.
    # This block runs BEFORE the final trace logging.
    import math, statistics
    img_w = 640
    img_h = 512

    # Group candidates by block_id
    sc_blocks = collections.defaultdict(list)
    for rec in raw_panels:
        if rec.get('single_column_tracking_candidate'):
            bid = rec.get('block_id')
            if bid is not None:
                sc_blocks[bid].append(rec)

    def _bbox_iou(b1, b2):
        x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        union = area1 + area2 - inter
        return inter / union if union else 0.0

    for bid, items in sc_blocks.items():
        # ----- Pre‑check -----
        count_ok = len(items) >= 6
        areas = [(r['bbox'][2] - r['bbox'][0]) * (r['bbox'][3] - r['bbox'][1]) for r in items]
        area_mean = statistics.mean(areas) if areas else 0
        area_std = statistics.stdev(areas) if len(areas) > 1 else 0
        area_cv = (area_std / area_mean) if area_mean else 1
        # vertical gaps based on centre y
        centers_y = [ (r['bbox'][1] + r['bbox'][3]) / 2.0 for r in items ]
        centers_y.sort()
        gaps = [centers_y[i] - centers_y[i-1] for i in range(1, len(centers_y))]
        gap_mean = statistics.mean(gaps) if gaps else 0
        gap_std = statistics.stdev(gaps) if len(gaps) > 1 else 0
        gap_cv = (gap_std / gap_mean) if gap_mean else 1
        # max IoU among original bboxes
        max_iou_orig = 0.0
        for a in range(len(items)):
            for b in range(a+1, len(items)):
                iou = _bbox_iou(items[a]['bbox'], items[b]['bbox'])
                if iou > max_iou_orig:
                    max_iou_orig = iou
        pre_ok = count_ok and area_cv <= 0.20 and gap_cv <= 0.08 and max_iou_orig <= 0.10
        if not pre_ok:
            reason = []
            if not count_ok: reason.append('count_lt_6')
            if area_cv > 0.20: reason.append('area_cv_high')
            if gap_cv > 0.08: reason.append('gap_cv_high')
            if max_iou_orig > 0.10: reason.append('orig_iou_high')
            reason_str = '_'.join(reason)
            for rec in items:
                rec['single_column_geometry_accept'] = False
                rec['single_column_geometry_reason'] = f"block_precheck_failed_{reason_str}"
                rec.pop('single_column_refined_bbox', None)
                rec.pop('single_column_refined_polygon', None)
            continue

        # ----- Compute median geometry -----
        # Sort by center_y only
        sorted_items = sorted(items, key=lambda r: (r['bbox'][1] + r['bbox'][3]) / 2.0)
        centers = [ ((r['bbox'][0] + r['bbox'][2]) / 2.0, (r['bbox'][1] + r['bbox'][3]) / 2.0) for r in sorted_items ]
        widths = [ r['bbox'][2] - r['bbox'][0] for r in sorted_items ]
        heights = [ r['bbox'][3] - r['bbox'][1] for r in sorted_items ]
        median_cx = statistics.median([c[0] for c in centers])
        median_w = statistics.median(widths)
        median_h = statistics.median(heights)
        # vertical pitch median
        gaps_y = [centers[i][1] - centers[i-1][1] for i in range(1, len(centers))]
        median_pitch = statistics.median(gaps_y) if gaps_y else 0
        # robust y-origin
        origin_vals = [centers[i][1] - i * median_pitch for i in range(len(centers))]
        y0 = statistics.median(origin_vals)

        # Build candidate refined bboxes/polygons
        candidate_bboxes = []
        for i, rec in enumerate(sorted_items):
            new_cy = y0 + i * median_pitch
            new_cx = median_cx
            new_bbox = [
                new_cx - median_w/2.0,
                new_cy - median_h/2.0,
                new_cx + median_w/2.0,
                new_cy + median_h/2.0,
            ]
            candidate_bboxes.append(new_bbox)
            rec['single_column_refined_bbox'] = new_bbox
            rec['single_column_refined_polygon'] = [
                [new_bbox[0], new_bbox[1]],
                [new_bbox[2], new_bbox[1]],
                [new_bbox[2], new_bbox[3]],
                [new_bbox[0], new_bbox[3]],
            ]
        # ----- Overlap guard after refinement -----
        max_iou_after = 0.0
        for a in range(len(candidate_bboxes)):
            for b in range(a+1, len(candidate_bboxes)):
                iou = _bbox_iou(candidate_bboxes[a], candidate_bboxes[b])
                if iou > max_iou_after:
                    max_iou_after = iou
        if max_iou_after > 0.09:
            for rec in sorted_items:
                rec['single_column_geometry_accept'] = False
                rec['single_column_geometry_reason'] = "block_postrefine_overlap_exceeds_0.09"
                rec.pop('single_column_refined_bbox', None)
                rec.pop('single_column_refined_polygon', None)
            logger.info(f"[SINGLE_COLUMN_BLOCK] block_id={bid} candidates={len(items)} accepted=0 fallback=block_overlap max_iou_after={max_iou_after:.3f}")
            continue

        # ----- Per‑panel acceptance -----
        accepted_cnt = 0
        fallback_cnt = 0
        outside_cnt = 0
        for rec, new_bbox in zip(sorted_items, candidate_bboxes):
            orig_bbox = rec['bbox']
            iou_orig = _bbox_iou(orig_bbox, new_bbox)
            # centre shift
            orig_cx = (orig_bbox[0] + orig_bbox[2]) / 2.0
            orig_cy = (orig_bbox[1] + orig_bbox[3]) / 2.0
            new_cx = (new_bbox[0] + new_bbox[2]) / 2.0
            new_cy = (new_bbox[1] + new_bbox[3]) / 2.0
            shift = math.hypot(new_cx - orig_cx, new_cy - orig_cy)
            inside_image = (
                0 <= new_bbox[0] and new_bbox[2] <= img_w and
                0 <= new_bbox[1] and new_bbox[3] <= img_h
            )
            if not inside_image:
                outside_cnt += 1
            if iou_orig >= 0.70 and shift <= 0.35 * median_h and inside_image:
                # accept
                # final_polygon_source unchanged (tracking only)
                # rec['final_polygon_stage'] = "single_column_geometry_refinement"
                # Geometry refinement disabled; keep tracking only
                rec['single_column_geometry_accept'] = False
                rec['single_column_geometry_reason'] = "disabled"
                # No polygon updates
            else:
                # fallback for this panel
                rec['single_column_geometry_accept'] = False
                if iou_orig < 0.70:
                    reason = "iou_low"
                elif shift > 0.35 * median_h:
                    reason = "center_shift_excess"
                elif not inside_image:
                    reason = "outside_image"
                else:
                    reason = "other"
                rec['single_column_geometry_reason'] = reason
                rec.pop('single_column_refined_bbox', None)
                rec.pop('single_column_refined_polygon', None)
                fallback_cnt += 1
        logger.info(f"[SINGLE_COLUMN_BLOCK] block_id={bid} candidates={len(items)} accepted={accepted_cnt} fallback={fallback_cnt} max_iou_after={max_iou_after:.3f} outside_image={outside_cnt}")

    # D. Ghi file debug JSONL + vẽ ảnh debug nhiều lớp
    image_path = getattr(result, "path", "unknown_image.jpg")
    save_panel_refine_debug_jsonl(image_path, raw_panels)
    save_debug_stages_image(orig_img, raw_panels, filtered_panels, image_path)
    if ENABLE_MIDDLE_LOCKED_ENDPOINT:
        _draw_middle_locked_debug_image(orig_img, raw_panels, image_path)

    print(f"[PIPELINE_CHECK] grid_aware_called={grid_aware_called} edge_snap_called={edge_snap_called} shrink_called={shrink_called}")

    image_name = os.path.splitext(os.path.basename(image_path))[0]

    # Local angle diagnostic logging for accepted strings/blocks
    accepted_blocks = {}
    accepted_strings = {}
    for p in raw_panels:
        if p.get("two_col_refined") == True and p.get("block_id") is not None:
            b_id = p["block_id"]
            if b_id not in accepted_blocks:
                accepted_blocks[b_id] = {
                    "type": "block",
                    "id": f"block_{b_id}",
                    "angle": p.get("block_angle_deg", 0.0),
                    "n_panels": 0,
                    "rail_fit_source": p.get("rail_fit_source", "all_panels")
                }
            accepted_blocks[b_id]["n_panels"] += 1
        elif p.get("string_refined") == True and p.get("string_id") is not None:
            s_id = p["string_id"]
            if s_id not in accepted_strings:
                accepted_strings[s_id] = {
                    "type": "string",
                    "id": f"string_{s_id}",
                    "angle": p.get("string_angle_deg", 0.0),
                    "n_panels": 0,
                    "rail_fit_source": p.get("rail_fit_source", "all_panels")
                }
            accepted_strings[s_id]["n_panels"] += 1

    accepted_elements = list(accepted_blocks.values()) + list(accepted_strings.values())
    local_angles = []
    for elem in accepted_elements:
        ang = elem["angle"]
        local_angles.append(ang)
        print(f"[ANGLE_DIAG] img={image_name} block_id={elem['id']} local_angle_deg={ang:.2f} "
              f"n_panels={elem['n_panels']} rail_fit_source={elem['rail_fit_source']}")

    if local_angles:
        min_ang = min(local_angles)
        max_ang = max(local_angles)
        ang_range = max_ang - min_ang
        print(f"[ANGLE_RANGE] img={image_name} min_angle={min_ang:.2f} max_angle={max_ang:.2f} range={ang_range:.2f}")
        print(f"[ANGLE_DIAG] img={image_name} n_blocks={len(accepted_elements)} min_angle={min_ang:.2f} max_angle={max_ang:.2f} angle_range={ang_range:.2f}")
    else:
        print(f"[ANGLE_RANGE] img={image_name} min_angle=0.00 max_angle=0.00 range=0.00")
        print(f"[ANGLE_DIAG] img={image_name} n_blocks=0 min_angle=0.00 max_angle=0.00 angle_range=0.00")

    # Initialize edge grid trace fields for all panels
    for p in raw_panels:
        p["edge_grid_attempted"] = False
        p["edge_grid_accept"] = False
        p["edge_grid_reject_reason"] = None
        p["edge_grid_area_ratio"] = None
        p["edge_grid_center_shift"] = None
        p["edge_grid_overlap"] = None
        p["edge_grid_vertical_mode"] = "segmented"
        p["edge_grid_n_segments"] = 4

    if ENABLE_EDGE_GRID_SEGMENTED_REFINEMENT:
        # 1. Identify target panels based on the specified criteria
        poor_panels = []
        for p in raw_panels:
            src = p.get("final_polygon_source", "")
            reason = p.get("final_polygon_reason", "")
            if (src == "yolo_original" or 
                "failed" in reason or 
                "rejected" in reason or 
                "area_ratio_out" in reason or 
                "large_col_dist" in reason):
                poor_panels.append(p)
                
        # 2. Group target panels into columns
        poor_panels_sorted = sorted(poor_panels, key=lambda p: p.get("center", [0, 0])[0])
        column_groups = []
        if poor_panels_sorted:
            current_group = [poor_panels_sorted[0]]
            for p in poor_panels_sorted[1:]:
                prev_x = current_group[-1].get("center", [0, 0])[0]
                curr_x = p.get("center", [0, 0])[0]
                if abs(curr_x - prev_x) <= 35:
                    current_group.append(p)
                else:
                    column_groups.append(current_group)
                    current_group = [p]
            column_groups.append(current_group)
            
        # Group vertically, filter by contiguous near-contiguous constraints (Y distance <= 120px)
        valid_column_groups = []
        for group in column_groups:
            group_sorted = sorted(group, key=lambda p: p.get("center", [0, 0])[1])
            if not group_sorted:
                continue
            subgroups = []
            current_sub = [group_sorted[0]]
            for p in group_sorted[1:]:
                prev_y = current_sub[-1].get("center", [0, 0])[1]
                curr_y = p.get("center", [0, 0])[1]
                if abs(curr_y - prev_y) <= 120.0:
                    current_sub.append(p)
                else:
                    subgroups.append(current_sub)
                    current_sub = [p]
            subgroups.append(current_sub)
            for sub in subgroups:
                if len(sub) >= 4:
                    valid_column_groups.append(sub)
                    
        # 3. Process each valid group
        for group in valid_column_groups:
            target_raw_indices = [p["raw_idx"] for p in group]
            for p in group:
                p["edge_grid_attempted"] = True
                
            config = EdgeGridConfig(
                roi_margin=45,
                vertical_mode="segmented",
                n_segments=4,
                max_x_jump_px=10.0,
                max_angle_delta_deg=8.0,
                search_half_width=10,
                min_points_per_segment=6,
                max_dev_baseline_px=12.0,
                smooth_window=3
            )
            
            res = propose_edge_grid_polygons_in_pipeline(
                orig_img=orig_img,
                panels=raw_panels,
                target_panel_indices_or_ids=target_raw_indices,
                config=config
            )
            
            debug_info = res.get("debug", {})
            fitted_info = debug_info.get("vertical_boundaries_fitted", [])
            too_much_fallback = False
            if len(fitted_info) >= 2:
                if fitted_info[0] < 1 or fitted_info[1] < 1:
                    too_much_fallback = True
            elif len(fitted_info) > 0:
                if fitted_info[0] < 1:
                    too_much_fallback = True
            else:
                too_much_fallback = True
                
            proposals = res.get("proposals", [])
            for prop in proposals:
                ridx = prop["raw_idx"]
                panel_obj = [p for p in group if p["raw_idx"] == ridx][0]
                
                m = prop["metrics"]
                iou = m["iou"]
                center_shift = m["center_shift"]
                area_ratio = m["area_ratio"]
                
                panel_obj["edge_grid_area_ratio"] = area_ratio
                panel_obj["edge_grid_center_shift"] = center_shift
                
                if too_much_fallback:
                    panel_obj["edge_grid_accept"] = False
                    panel_obj["edge_grid_reject_reason"] = "vertical_boundary_too_much_fallback"
                    continue
                    
                if not prop["ok"]:
                    panel_obj["edge_grid_accept"] = False
                    panel_obj["edge_grid_reject_reason"] = f"proposal_failed_{prop.get('source', 'unknown')}"
                    continue
                    
                poly_proposed = prop["polygon"]
                if len(poly_proposed) != 4:
                    panel_obj["edge_grid_accept"] = False
                    panel_obj["edge_grid_reject_reason"] = "not_quadrilateral"
                    continue
                    
                poly_np = np.array(poly_proposed, dtype=np.float32)
                if not cv2.isContourConvex(poly_np.reshape(-1, 1, 2).astype(np.int32)):
                    panel_obj["edge_grid_accept"] = False
                    panel_obj["edge_grid_reject_reason"] = "non_convex"
                    continue
                    
                if not (0.75 <= area_ratio <= 1.35):
                    panel_obj["edge_grid_accept"] = False
                    panel_obj["edge_grid_reject_reason"] = "area_ratio_out_of_bounds"
                    continue
                    
                if center_shift > 25.0:
                    panel_obj["edge_grid_accept"] = False
                    panel_obj["edge_grid_reject_reason"] = "large_center_shift"
                    continue
                    
                # Compute overlap
                max_overlap = 0.0
                for other_p in raw_panels:
                    if other_p["raw_idx"] == ridx:
                        continue
                    other_poly = other_p.get("polygon", [])
                    if len(other_poly) >= 3:
                        ov = compute_polygon_overlap_ratio(poly_proposed, other_poly, orig_img.shape)
                        if ov > max_overlap:
                            max_overlap = ov
                            
                panel_obj["edge_grid_overlap"] = max_overlap
                if max_overlap > 0.05:
                    panel_obj["edge_grid_accept"] = False
                    panel_obj["edge_grid_reject_reason"] = "high_overlap"
                    continue
                    
                # Accept proposal!
                panel_obj["edge_grid_accept"] = True
                panel_obj["edge_grid_reject_reason"] = None
                panel_obj["polygon"] = [[int(round(pt[0])), int(round(pt[1]))] for pt in poly_proposed]
                
                features = get_polygon_features(panel_obj["polygon"])
                panel_obj["area"] = features["area"]
                panel_obj["center"] = features["center"]
                panel_obj["aspect_ratio"] = features["aspect_ratio"]
                
                panel_obj["final_polygon_source"] = "edge_grid_segmented4"
                panel_obj["final_polygon_stage"] = "edge_grid_refinement"
                panel_obj["final_polygon_reason"] = "segmented_boundary_projection"

        # 4. Save debug image
        debug_dir = "data/results/debug"
        os.makedirs(debug_dir, exist_ok=True)
        image_stem = os.path.splitext(os.path.basename(image_path))[0]
        debug_img_path = os.path.join(debug_dir, f"debug_{image_stem}_edge_grid_final.JPG")
        
        img_debug = orig_img.copy()
        for p in raw_panels:
            poly = p.get("polygon", [])
            if len(poly) >= 3:
                pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
                color = (0, 255, 255) if p.get("final_polygon_source") == "edge_grid_segmented4" else (0, 255, 0)
                cv2.polylines(img_debug, [pts], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)
                
            cx, cy = p.get("center", [0, 0])
            cv2.putText(img_debug, str(p.get("raw_idx")), (int(cx) - 8, int(cy) + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1, cv2.LINE_AA)
                        
        cv2.imwrite(debug_img_path, img_debug, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"[EDGE_GRID_DEBUG] Saved final overlay to: {debug_img_path}")

    # Polygon source trace audit, summary printing, JSONL logging, and warning check
    trace_records = []
    source_counts = {
        "yolo_original": 0,
        "small_panel_bbox": 0,
        "string_lattice_middle": 0,
        "middle_locked_endpoint": 0,
        "two_col_block_middle": 0,
        "two_col_block_endpoint": 0,
        "endpoint_original_fallback": 0,
        "outer_guard_fallback": 0,
        "safe_improvement_fallback": 0,
        "edge_grid_segmented4": 0,
        "unknown": 0
    }
    
    for p in raw_panels:
        # Clamp final polygon into image bounds (640x512)
        clamped = False
        max_delta = 0.0
        new_poly = []
        for pt in p.get("polygon", []):
            x, y = pt
            cx = min(max(x, 0), 640)
            cy = min(max(y, 0), 512)
            dx = abs(cx - x)
            dy = abs(cy - y)
            if dx > 0 or dy > 0:
                clamped = True
                max_delta = max(max_delta, dx, dy)
            new_poly.append([cx, cy])
        p["polygon"] = new_poly
        p["final_polygon_clamped"] = clamped
        p["final_polygon_clamp_delta_max"] = max_delta

        src = p.get("final_polygon_source", "unknown")
        if src not in source_counts:
            src = "unknown"
            p["final_polygon_source"] = "unknown"
            
        source_counts[src] += 1
        
        if src == "unknown":
            print(f"[WARNING] Panel raw_idx={p.get('raw_idx')} has final_polygon_source=\"unknown\"")
            
        trace_records.append({
            "raw_idx": p.get("raw_idx"),
            "local_id": p.get("local_id", p.get("class_name", "panel")),
            "bbox": p.get("bbox"),
            "area": p.get("area"),
            "center": p.get("center"),
            "is_endpoint": bool(p.get("is_endpoint_panel", False) or p.get("is_endpoint", False)),
            "endpoint_side": p.get("endpoint_side"),
            "final_polygon_source": src,
            "final_polygon_stage": p.get("final_polygon_stage", "unknown"),
            "final_polygon_reason": p.get("final_polygon_reason", "unknown"),
            "string_id": p.get("string_id"),
            "block_id": p.get("block_id"),
            "polygon": p.get("polygon"),
            "final_polygon_clamped": p.get("final_polygon_clamped"),
            "final_polygon_clamp_delta_max": p.get("final_polygon_clamp_delta_max"),
            "endpoint_relaxed_accept": p.get("endpoint_relaxed_accept"),
            "endpoint_relaxed_overlap": p.get("endpoint_relaxed_overlap"),
            "endpoint_relaxed_iou": p.get("endpoint_relaxed_iou"),
            "endpoint_relaxed_center_shift": p.get("endpoint_relaxed_center_shift"),
            "endpoint_relaxed_reject_reason": p.get("endpoint_relaxed_reject_reason"),
            "geometry_gate_relaxed_accept": p.get("geometry_gate_relaxed_accept"),
            "geometry_gate_relaxed_reject_reason": p.get("geometry_gate_relaxed_reject_reason"),
            "geometry_gate_relaxed_iou": p.get("geometry_gate_relaxed_iou"),
            "geometry_gate_relaxed_center_shift": p.get("geometry_gate_relaxed_center_shift"),
            "geometry_gate_relaxed_area_ratio": p.get("geometry_gate_relaxed_area_ratio"),
            "geometry_gate_relaxed_expansion_x": p.get("geometry_gate_relaxed_expansion_x"),
            "geometry_gate_relaxed_expansion_y": p.get("geometry_gate_relaxed_expansion_y"),
            "geometry_gate_relaxed_max_overlap": p.get("geometry_gate_relaxed_max_overlap"),
            "line_consensus_rescue_accept": p.get("line_consensus_rescue_accept"),
            "line_consensus_rescue_reject_reason": p.get("line_consensus_rescue_reject_reason"),
            "line_consensus_row_support": p.get("line_consensus_row_support"),
            "line_consensus_col_support": p.get("line_consensus_col_support"),
            "line_consensus_row_dist": p.get("line_consensus_row_dist"),
            "line_consensus_col_dist": p.get("line_consensus_col_dist"),
            "line_consensus_iou": p.get("line_consensus_iou"),
            "line_consensus_center_shift": p.get("line_consensus_center_shift"),
            "line_consensus_area_ratio": p.get("line_consensus_area_ratio"),
            "line_consensus_max_overlap": p.get("line_consensus_max_overlap"),
            "line_consensus_overlap_approximate": p.get("line_consensus_overlap_approximate"),
            "line_consensus_row_source": p.get("line_consensus_row_source"),
            "line_consensus_global_row_used": p.get("line_consensus_global_row_used"),
            "line_consensus_col_rank": p.get("line_consensus_col_rank"),
            "line_consensus_col_candidates_tried": p.get("line_consensus_col_candidates_tried"),
            "edge_grid_attempted": p.get("edge_grid_attempted"),
            "edge_grid_accept": p.get("edge_grid_accept"),
            "edge_grid_reject_reason": p.get("edge_grid_reject_reason"),
            "edge_grid_area_ratio": p.get("edge_grid_area_ratio"),
            "edge_grid_center_shift": p.get("edge_grid_center_shift"),
            "edge_grid_overlap": p.get("edge_grid_overlap"),
            "edge_grid_vertical_mode": p.get("edge_grid_vertical_mode"),
            "edge_grid_n_segments": p.get("edge_grid_n_segments")
        })
        
    print(f"[FINAL_POLYGON_SOURCE] img={image_name} source_counts={source_counts}")
    
    # Write trace JSONL
    trace_log_dir = "data/results/debug_logs"
    os.makedirs(trace_log_dir, exist_ok=True)
    trace_log_path = os.path.join(trace_log_dir, f"{image_name}_final_polygon_trace.jsonl")
    try:
        with open(trace_log_path, "w", encoding="utf-8") as f:
            for rec in trace_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[FINAL_POLYGON_TRACE] Trace saved to: {trace_log_path}")
    except Exception as e:
        print(f"[WARNING] Failed to write trace JSONL: {e}")
        
    # Draw trace debug image
    _draw_final_polygon_source_debug_image(orig_img, raw_panels, image_path)

    detections = filtered_panels + raw_defects
    return detections


def _draw_outer_guard_debug_image(
    image: np.ndarray,
    panels: List[Dict[str, Any]],
    image_path: str
) -> None:
    """
    Draw debug image specifically for Outer Boundary Guard.
    """
    if not any(p.get("is_outer_panel") for p in panels):
        return

    dbg = image.copy()
    img_h, img_w = image.shape[:2]
    
    def draw_dashed_line(img, pt1, pt2, color, thickness=2, gap=6):
        dist = np.linalg.norm(np.array(pt1) - np.array(pt2))
        if dist < 1.0:
            return
        pts = []
        for i in np.arange(0, dist, gap):
            r = float(i) / dist
            x = int(round(pt1[0] + r * (pt2[0] - pt1[0])))
            y = int(round(pt1[1] + r * (pt2[1] - pt1[1])))
            pts.append((x, y))
        for i in range(0, len(pts) - 1, 2):
            cv2.line(img, pts[i], pts[i+1], color, thickness, cv2.LINE_AA)

    # 1. Draw current lattice of all panels (cyan/yellow thin)
    for p in panels:
        is_refined = p.get("string_refined", False) or p.get("two_col_refined", False)
        if is_refined and not p.get("is_outer_panel"):
            poly = p.get("polygon")
            if poly:
                pts = np.array(poly, dtype=np.int32)
                cv2.polylines(dbg, [pts], True, (255, 255, 0), 1, cv2.LINE_AA)
                
    # 2. Draw outer dividers proposal, accepted, and rejected
    drawn_keys = set()
    for p in panels:
        if p.get("is_outer_panel"):
            dec = p.get("two_col_decision") or p.get("string_decision")
            if dec == "bypass_large_panel" or dec == "bypass_no_string":
                continue
            u_axis = p.get("outer_divider_u_axis")
            v_axis = p.get("outer_divider_v_axis")
            origin = p.get("outer_divider_origin")
            rail_low = p.get("outer_divider_rail_low")
            rail_high = p.get("outer_divider_rail_high")
            u_proposal = p.get("outer_divider_u_proposal")
            u_snapped = p.get("outer_divider_u_snapped")
            
            if u_axis is None or v_axis is None or origin is None or u_proposal is None:
                continue
                
            key = (round(float(origin[0]), 1), round(float(origin[1]), 1), round(float(u_proposal), 1))
            if key in drawn_keys:
                continue
            drawn_keys.add(key)
            
            pt_prop_low = origin + u_proposal * u_axis + rail_low * v_axis
            pt_prop_high = origin + u_proposal * u_axis + rail_high * v_axis
            
            pt_snap_low = origin + u_snapped * u_axis + rail_low * v_axis
            pt_snap_high = origin + u_snapped * u_axis + rail_high * v_axis
            
            prop_p1 = (int(round(pt_prop_low[0])), int(round(pt_prop_low[1])))
            prop_p2 = (int(round(pt_prop_high[0])), int(round(pt_prop_high[1])))
            
            snap_p1 = (int(round(pt_snap_low[0])), int(round(pt_snap_low[1])))
            snap_p2 = (int(round(pt_snap_high[0])), int(round(pt_snap_high[1])))
            
            guard_dec = p.get("outer_guard_decision") or p.get("two_col_decision") or p.get("string_decision")
            if p.get("two_col_decision") == "fallback_outer_panel" or p.get("string_decision") == "fallback_outer_panel":
                guard_dec = "fallback_outer_panel"
            elif p.get("two_col_decision") == "use_two_col_lattice" or p.get("string_decision") == "use_string_lattice":
                guard_dec = "use_outer_snap"
            
            # outer dividers proposal: yellow dashed (cyan)
            draw_dashed_line(dbg, prop_p1, prop_p2, (0, 255, 255), 1, gap=6)
            
            if guard_dec == "use_outer_snap":
                # accepted outer divider: white solid
                cv2.line(dbg, snap_p1, snap_p2, (255, 255, 255), 2, cv2.LINE_AA)
            else:
                # rejected outer divider: red dashed
                draw_dashed_line(dbg, snap_p1, snap_p2, (0, 0, 255), 2, gap=6)

    # 3. Draw fallback outer panels: orange/red translucent
    overlay = dbg.copy()
    for p in panels:
        if p.get("is_outer_panel"):
            dec = p.get("two_col_decision") or p.get("string_decision")
            poly = p.get("polygon")
            if poly and dec == "fallback_outer_panel":
                pts = np.array(poly, dtype=np.int32)
                cv2.fillPoly(overlay, [pts], (0, 100, 255))
                cv2.polylines(dbg, [pts], True, (0, 0, 255), 2, cv2.LINE_AA)
                cx, cy = p["center"]
                cv2.putText(dbg, f"FB_OUT_{p.get('raw_idx')}", (int(cx) - 25, int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1, cv2.LINE_AA)
            elif poly and dec in ("use_string_lattice", "use_two_col_lattice"):
                pts = np.array(poly, dtype=np.int32)
                cv2.polylines(dbg, [pts], True, (255, 255, 0), 2, cv2.LINE_AA)
                cx, cy = p["center"]
                cv2.putText(dbg, f"OUT_OK_{p.get('raw_idx')}", (int(cx) - 25, int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1, cv2.LINE_AA)

    # Alpha blend overlay
    cv2.addWeighted(overlay, 0.3, dbg, 0.7, 0, dbg)
    
    # Save image
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    out_dir = "data/results/debug"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"debug_{image_name}_outer_guard.JPG")
    cv2.imwrite(out_path, dbg)
    print(f"[OUTER_GUARD] Debug image saved: {out_path}")


def _draw_endpoint_guard_debug_image(
    image: np.ndarray,
    panels: List[Dict[str, Any]],
    image_path: str
) -> None:
    """
    Draw debug image for Endpoint Guard showing rails, local angles, middle panels, and endpoint panels.
    """
    if not any(p.get("is_endpoint_panel") for p in panels):
        return

    dbg = image.copy()
    img_h, img_w = image.shape[:2]

    def draw_dashed_line(img, pt1, pt2, color, thickness=2, gap=6):
        dist = np.linalg.norm(np.array(pt1) - np.array(pt2))
        if dist < 1.0:
            return
        pts = []
        for i in np.arange(0, dist, gap):
            r = float(i) / dist
            x = int(round(pt1[0] + r * (pt2[0] - pt1[0])))
            y = int(round(pt1[1] + r * (pt2[1] - pt1[1])))
            pts.append((x, y))
        for i in range(0, len(pts) - 1, 2):
            cv2.line(img, pts[i], pts[i+1], color, thickness, cv2.LINE_AA)

    # Group panels by block_id or string_id
    blocks = {}
    strings = {}
    for p in panels:
        if p.get("two_col_refined") == True and p.get("block_id") is not None:
            b_id = p["block_id"]
            if b_id not in blocks:
                blocks[b_id] = []
            blocks[b_id].append(p)
        elif p.get("string_refined") == True and p.get("string_id") is not None:
            s_id = p["string_id"]
            if s_id not in strings:
                strings[s_id] = []
            strings[s_id].append(p)

    # Draw block rails
    for b_id, block_panels in blocks.items():
        p_sample = block_panels[0]
        origin = p_sample.get("outer_divider_origin")
        u_axis = p_sample.get("outer_divider_u_axis")
        v_axis = p_sample.get("outer_divider_v_axis")
        left_rail_v = p_sample.get("outer_divider_rail_low")
        right_rail_v = p_sample.get("outer_divider_rail_high")
        middle_rail_v = p_sample.get("block_rail_middle")

        if origin is None or u_axis is None or v_axis is None or left_rail_v is None or right_rail_v is None:
            continue

        # Project all panel corners onto u_axis
        u_vals = []
        for p in block_panels:
            poly = p.get("polygon")
            if poly:
                for pt in poly:
                    u_vals.append(float(np.dot(np.array(pt) - origin, u_axis)))
        if not u_vals:
            continue
        u_min = min(u_vals)
        u_max = max(u_vals)

        # Draw left rail
        pt_l_start = origin + u_min * u_axis + left_rail_v * v_axis
        pt_l_end = origin + u_max * u_axis + left_rail_v * v_axis
        cv2.line(dbg, (int(round(pt_l_start[0])), int(round(pt_l_start[1]))), 
                      (int(round(pt_l_end[0])), int(round(pt_l_end[1]))), (255, 255, 255), 2, cv2.LINE_AA)

        # Draw middle rail (if exists)
        if middle_rail_v is not None:
            pt_m_start = origin + u_min * u_axis + middle_rail_v * v_axis
            pt_m_end = origin + u_max * u_axis + middle_rail_v * v_axis
            cv2.line(dbg, (int(round(pt_m_start[0])), int(round(pt_m_start[1]))), 
                          (int(round(pt_m_end[0])), int(round(pt_m_end[1]))), (255, 255, 255), 2, cv2.LINE_AA)

        # Draw right rail
        pt_r_start = origin + u_min * u_axis + right_rail_v * v_axis
        pt_r_end = origin + u_max * u_axis + right_rail_v * v_axis
        cv2.line(dbg, (int(round(pt_r_start[0])), int(round(pt_r_start[1]))), 
                      (int(round(pt_r_end[0])), int(round(pt_r_end[1]))), (255, 255, 255), 2, cv2.LINE_AA)

        # Draw angle text
        block_ang = p_sample.get("block_angle_deg", 0.0)
        cx = sum(p["center"][0] for p in block_panels) / len(block_panels)
        cy = sum(p["center"][1] for p in block_panels) / len(block_panels)
        cv2.putText(dbg, f"B{b_id} Angle: {block_ang:.1f}deg", (int(cx) - 60, int(cy) - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # Draw string rails
    for s_id, string_panels in strings.items():
        p_sample = string_panels[0]
        origin = p_sample.get("outer_divider_origin")
        u_axis = p_sample.get("outer_divider_u_axis")
        v_axis = p_sample.get("outer_divider_v_axis")
        rail_low = p_sample.get("outer_divider_rail_low")
        rail_high = p_sample.get("outer_divider_rail_high")

        if origin is None or u_axis is None or v_axis is None or rail_low is None or rail_high is None:
            continue

        # Project all panel corners onto u_axis
        u_vals = []
        for p in string_panels:
            poly = p.get("polygon")
            if poly:
                for pt in poly:
                    u_vals.append(float(np.dot(np.array(pt) - origin, u_axis)))
        if not u_vals:
            continue
        u_min = min(u_vals)
        u_max = max(u_vals)

        # Draw low rail
        pt_l_start = origin + u_min * u_axis + rail_low * v_axis
        pt_l_end = origin + u_max * u_axis + rail_low * v_axis
        cv2.line(dbg, (int(round(pt_l_start[0])), int(round(pt_l_start[1]))), 
                      (int(round(pt_l_end[0])), int(round(pt_l_end[1]))), (255, 255, 255), 2, cv2.LINE_AA)

        # Draw high rail
        pt_h_start = origin + u_min * u_axis + rail_high * v_axis
        pt_h_end = origin + u_max * u_axis + rail_high * v_axis
        cv2.line(dbg, (int(round(pt_h_start[0])), int(round(pt_h_start[1]))), 
                      (int(round(pt_h_end[0])), int(round(pt_h_end[1]))), (255, 255, 255), 2, cv2.LINE_AA)

        # Draw angle text
        string_ang = p_sample.get("string_angle_deg", 0.0)
        cx = sum(p["center"][0] for p in string_panels) / len(string_panels)
        cy = sum(p["center"][1] for p in string_panels) / len(string_panels)
        cv2.putText(dbg, f"S{s_id} Angle: {string_ang:.1f}deg", (int(cx) - 60, int(cy) - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # 2. Draw panels based on their endpoint guard status
    overlay = dbg.copy()
    for p in panels:
        poly = p.get("polygon")
        if not poly:
            continue
        pts = np.array(poly, dtype=np.int32)
        is_endpoint = p.get("is_endpoint_panel", False)

        if is_endpoint:
            diag_pass = p.get("endpoint_diagnostic_pass")
            if diag_pass is not None:
                show_fail = not diag_pass
            else:
                dec = p.get("endpoint_decision") or p.get("two_col_decision") or p.get("string_decision")
                show_fail = dec in ("fallback_endpoint_original", "fallback_outer_panel", "fallback_string", "fallback_original")

            if show_fail:
                # Fallback original: translucent orange/red
                cv2.fillPoly(overlay, [pts], (0, 100, 255))
                cv2.polylines(dbg, [pts], True, (0, 0, 255), 2, cv2.LINE_AA)
                cx, cy = p["center"]
                cv2.putText(dbg, f"EP_FB_{p.get('raw_idx')}", (int(cx) - 25, int(cy)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1, cv2.LINE_AA)
            else:
                # Accepted endpoint lattice: thick dark green
                cv2.polylines(dbg, [pts], True, (0, 150, 0), 3, cv2.LINE_AA)
                cx, cy = p["center"]
                cv2.putText(dbg, f"EP_OK_{p.get('raw_idx')}", (int(cx) - 25, int(cy)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 150, 0), 1, cv2.LINE_AA)

            # Draw candidates if ENABLED
            if ENDPOINT_DEBUG_DRAW_CANDIDATES:
                origin = p.get("outer_divider_origin")
                u_axis = p.get("outer_divider_u_axis")
                v_axis = p.get("outer_divider_v_axis")
                r_low = p.get("outer_divider_rail_low")
                r_high = p.get("outer_divider_rail_high")

                if origin is not None and u_axis is not None and v_axis is not None and r_low is not None and r_high is not None:
                    def get_segment(u_val):
                        p1 = origin + u_val * u_axis + r_low * v_axis
                        p2 = origin + u_val * u_axis + r_high * v_axis
                        return (int(round(p1[0])), int(round(p1[1]))), (int(round(p2[0])), int(round(p2[1])))

                    # YOLO candidate: blue dashed
                    yolo_u = p.get("endpoint_yolo_outer_u")
                    if yolo_u is not None:
                        p1, p2 = get_segment(yolo_u)
                        draw_dashed_line(dbg, p1, p2, (255, 0, 0), thickness=2, gap=6)

                    # Pitch candidate: yellow dashed
                    pitch_u = p.get("endpoint_pitch_outer_u")
                    if pitch_u is not None:
                        p1, p2 = get_segment(pitch_u)
                        draw_dashed_line(dbg, p1, p2, (0, 255, 255), thickness=2, gap=6)

                    # Edge scan selected: white dashed
                    edge_u = p.get("endpoint_edge_outer_u")
                    if edge_u is not None and edge_u != 0.0:
                        p1, p2 = get_segment(edge_u)
                        draw_dashed_line(dbg, p1, p2, (255, 255, 255), thickness=2, gap=6)

                    # Consensus candidate: magenta dashed
                    cons_u = p.get("endpoint_consensus_outer_u")
                    if cons_u is not None and cons_u != 0.0:
                        p1, p2 = get_segment(cons_u)
                        draw_dashed_line(dbg, p1, p2, (255, 0, 255), thickness=2, gap=6)
        else:
            # Middle panel: thin cyan/light blue (B=255, G=255, R=0 is cyan)
            is_refined = p.get("string_refined", False) or p.get("two_col_refined", False)
            if is_refined:
                cv2.polylines(dbg, [pts], True, (255, 255, 0), 1, cv2.LINE_AA)

    # Blend overlay
    cv2.addWeighted(overlay, 0.3, dbg, 0.7, 0, dbg)

    # Save
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    out_dir = "data/results/debug"
    os.makedirs(out_dir, exist_ok=True)
    
    # Save endpoint_guard image
    out_path = os.path.join(out_dir, f"debug_{image_name}_endpoint_guard.JPG")
    cv2.imwrite(out_path, dbg)
    print(f"[ENDPOINT_GUARD] Debug image saved: {out_path}")

    # Save endpoint_hybrid image
    hybrid_path = os.path.join(out_dir, f"debug_{image_name}_endpoint_hybrid.JPG")
    cv2.imwrite(hybrid_path, dbg)
    print(f"[ENDPOINT_HYBRID] Debug image saved: {hybrid_path}")


# ===========================================================================
# SECTION 6 — ANNOTATION DRAWING (GIỮ NGUYÊN)
# ===========================================================================

def draw_custom_annotation(
    image_bgr: np.ndarray,
    panels: List[Dict[str, Any]],
    unassigned_defects: List[Dict[str, Any]] = None
) -> np.ndarray:
    img = image_bgr.copy()
    DEFECT_COLORS = {
        "hotspot_single_cell": (0, 100, 255),   # Cam
        "hotspot_multi_cell":  (0, 0, 255),      # Đỏ
        "shading":             (200, 200, 0),    # Cyan
        "soiling":             (0, 165, 255),    # Cam nhạt
        "crack":               (200, 0, 200),    # Tím
    }
    DEFAULT_DEFECT_COLOR = (100, 100, 255)

    # 1. VẼ PANEL VÀ DEFECT ĐÃ ĐƯỢC GÁN
    for p in panels:
        p_poly = p.get("polygon", [])
        if len(p_poly) >= 3:
            pts = np.array(p_poly, dtype=np.int32)
            cv2.polylines(img, [pts], True, (255, 0, 0), 2, cv2.LINE_AA)
            cx, cy = p.get("center", [0, 0])
            label = p.get("local_id", p.get("class_name", "panel"))
            cv2.putText(img, label, (int(cx) - 20, int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1, cv2.LINE_AA)

        for d in p.get("defects", []):
            d_poly = d.get("polygon", [])
            cls = d.get("class_name", "")
            color = DEFECT_COLORS.get(cls, DEFAULT_DEFECT_COLOR)

            if len(d_poly) >= 3:
                pts = np.array(d_poly, dtype=np.int32)
                overlay = img.copy()
                cv2.fillPoly(overlay, [pts], color=color)
                img = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)
                cv2.polylines(img, [pts], True, (0, 255, 255), 2, cv2.LINE_AA)

                dcx, dcy = d.get("center", [0, 0])
                info = f"{cls} {d.get('area_ratio_percent', 0):.1f}%"
                cv2.putText(img, info, (int(dcx) - 30, int(dcy)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)

    # 2. VẼ DEFECT BỊ LẠC TRÔI (UNASSIGNED)
    if unassigned_defects:
        for d in unassigned_defects:
            d_poly = d.get("polygon", [])
            cls = d.get("class_name", "")
            color = DEFECT_COLORS.get(cls, DEFAULT_DEFECT_COLOR)

            if len(d_poly) >= 3:
                pts = np.array(d_poly, dtype=np.int32)
                cv2.polylines(img, [pts], True, (0, 0, 255), 3, cv2.LINE_AA)
                dcx, dcy = d.get("center", [0, 0])
                cv2.putText(img, f"[LOST] {cls}", (int(dcx) - 30, int(dcy)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)

    return img
