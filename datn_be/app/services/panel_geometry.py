# app/services/panel_geometry.py
# Chứa các hàm xử lý hình học cho tấm pin (panel):
#   - Validation panel detection
#   - Refine polygon về tứ giác sạch bằng minAreaRect
#   - Trích xuất bbox, area, center, aspect_ratio từ polygon
#   - Gán hàng/cột (R01_C03) theo vị trí tương đối trong ảnh
#   - Định vị lỗi nằm ở đâu trong panel (upper-left, center, lower-right...)
import cv2
import numpy as np
from typing import List, Optional, Dict, Any, Tuple


# ─────────────────────────────────────────
# 1. VALIDATION PANEL
# ─────────────────────────────────────────

def is_valid_panel_detection(det: Dict[str, Any], median_area: Optional[float] = None) -> bool:
    """
    Kiểm tra một detection class 'panel' có hợp lệ không.
    - Polygon phải có ít nhất 3 điểm
    - Area không quá nhỏ (loại nhiễu)
    - Aspect ratio hợp lý (không dài/hẹp bất thường)
    - Nếu có median_area thì loại panel quá lệch so với median
    """
    poly = det.get("polygon", [])
    area = det.get("area", 0)

    # Cần ít nhất 3 điểm để tạo polygon
    if len(poly) < 3:
        return False

    # Loại diện tích quá nhỏ (< 200 px² thường là nhiễu)
    if area < 200:
        return False

    # Kiểm tra aspect ratio: lấy từ bbox
    bbox = det.get("bbox") or det.get("box")
    if bbox and len(bbox) == 4:
        w = max(bbox[2] - bbox[0], 1)
        h = max(bbox[3] - bbox[1], 1)
        aspect = max(w, h) / min(w, h)
        # Panel tấm pin thường có aspect ratio 1:1 đến 3:1, tối đa ~5:1
        if aspect > 6.0:
            return False

    # So sánh với median area nếu có
    if median_area and median_area > 0:
        ratio = area / median_area
        # Loại panel có diện tích nhỏ hơn 40% hoặc lớn hơn 250% so với median
        if ratio < 0.40 or ratio > 2.50:
            return False

    return True


# ─────────────────────────────────────────
# 2. REFINE PANEL POLYGON (minAreaRect → 4 điểm)
# ─────────────────────────────────────────

def refine_panel_polygon_from_mask(mask_data: np.ndarray, img_shape: Tuple[int, int]) -> Optional[List[List[int]]]:
    """
    Từ binary mask raster của panel:
    1. Morphology close/open 5x5 để điền lỗ nhỏ
    2. Lấy contour lớn nhất
    3. Dùng cv2.minAreaRect → 4 điểm tứ giác sạch
    Trả về list 4 điểm [[x,y], ...] hoặc None nếu thất bại.

    Lý do dùng minAreaRect cho panel: panel là vật thể có hình dạng hình chữ nhật
    (tứ giác xoay), minAreaRect cho ra polygon 4 điểm rất sạch và ổn định.
    """
    img_h, img_w = img_shape[:2]

    # Đảm bảo mask là uint8
    if mask_data.dtype != np.uint8:
        mask_u8 = (mask_data > 0).astype(np.uint8) * 255
    else:
        mask_u8 = mask_data.copy()

    # Resize mask về kích thước ảnh gốc nếu cần
    if mask_u8.shape[:2] != (img_h, img_w):
        mask_u8 = cv2.resize(mask_u8, (img_w, img_h), interpolation=cv2.INTER_NEAREST)

    # Morphology close (điền lỗ nhỏ) rồi open (loại nhiễu nhỏ)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

    # Tìm contour
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Lấy contour lớn nhất
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 100:
        return None

    # minAreaRect → 4 điểm tứ giác xoay
    rect = cv2.minAreaRect(largest)
    box_pts = cv2.boxPoints(rect)  # shape (4, 2)
    return np.int32(box_pts).tolist()


def refine_panel_polygon_from_xy(xy_polygon: List, img_shape: Tuple[int, int]) -> Optional[List[List[int]]]:
    """
    Từ polygon xy của YOLO (result.masks.xy):
    1. Vẽ vào binary mask
    2. Gọi refine_panel_polygon_from_mask
    Dùng khi không có mask raster trực tiếp.
    """
    img_h, img_w = img_shape[:2]
    pts = np.array(xy_polygon, dtype=np.int32)
    if len(pts) < 3:
        return None

    binary = np.zeros((img_h, img_w), dtype=np.uint8)
    cv2.fillPoly(binary, [pts], 255)
    return refine_panel_polygon_from_mask(binary, img_shape)


# ─────────────────────────────────────────
# 3. TRÍCH XUẤT FEATURE TỪ POLYGON
# ─────────────────────────────────────────

def get_polygon_features(points: List) -> Dict[str, Any]:
    """
    Tính bbox, area, center, aspect_ratio từ list điểm polygon.
    Trả về dict với các key: bbox, area, center, aspect_ratio.
    """
    pts = np.array(points, dtype=np.float32)
    if len(pts) < 3:
        return {"bbox": [0, 0, 0, 0], "area": 0, "center": [0, 0], "aspect_ratio": 1.0}

    x_coords = pts[:, 0]
    y_coords = pts[:, 1]

    x1, y1 = float(np.min(x_coords)), float(np.min(y_coords))
    x2, y2 = float(np.max(x_coords)), float(np.max(y_coords))

    w = max(x2 - x1, 1)
    h = max(y2 - y1, 1)

    # Diện tích polygon (Shoelace formula)
    area = float(cv2.contourArea(pts.reshape(-1, 1, 2)))

    cx = float(np.mean(x_coords))
    cy = float(np.mean(y_coords))

    aspect_ratio = round(max(w, h) / min(w, h), 3)

    return {
        "bbox": [round(x1), round(y1), round(x2), round(y2)],
        "area": round(area, 2),
        "center": [round(cx, 2), round(cy, 2)],
        "aspect_ratio": aspect_ratio,
    }


# ─────────────────────────────────────────
# 4. GÁN HÀNG/CỘT (ROW/COL GRID)
# ─────────────────────────────────────────

def assign_row_col_ids(panels: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Gán row, col, local_id (dạng R01_C03) cho từng panel dựa trên vị trí.
    Phương pháp:
    1. Sort panel theo center_y
    2. Nhóm thành hàng bằng ngưỡng = 0.5 * median panel height
    3. Trong mỗi hàng, sort theo center_x
    4. Gán row từ trên xuống, col từ trái sang phải

    Lưu ý: Đây là định vị TƯƠNG ĐỐI vì không có GPS/metadata.
    Hoạt động tốt khi drone bay thẳng, camera nhìn thẳng xuống.
    Với góc nghiêng mạnh, kết quả có thể sai nhẹ ở hàng cuối.
    """
    if not panels:
        return panels

    # Lấy center_y và height của từng panel
    for p in panels:
        bbox = p.get("bbox") or p.get("box", [0, 0, 0, 0])
        if "center" in p:
            cy = p["center"][1]
            cx = p["center"][0]
        elif len(bbox) == 4:
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
        else:
            cx, cy = 0, 0
        p["_cx"] = cx
        p["_cy"] = cy

    # Tính ngưỡng nhóm hàng = 50% median chiều cao panel
    heights = []
    for p in panels:
        bbox = p.get("bbox") or p.get("box", [0, 0, 0, 0])
        if len(bbox) == 4:
            h = bbox[3] - bbox[1]
            if h > 0:
                heights.append(h)

    if heights:
        median_h = float(np.median(heights))
        row_threshold = median_h * 0.5
    else:
        row_threshold = 30.0  # fallback px

    # Sort theo y rồi nhóm hàng
    sorted_panels = sorted(panels, key=lambda p: p["_cy"])
    rows: List[List[Dict]] = []
    current_row = [sorted_panels[0]]
    current_y = sorted_panels[0]["_cy"]

    for p in sorted_panels[1:]:
        if abs(p["_cy"] - current_y) <= row_threshold:
            current_row.append(p)
        else:
            rows.append(current_row)
            current_row = [p]
            current_y = p["_cy"]
    rows.append(current_row)

    # Trong mỗi hàng, sort theo x rồi gán col
    for row_idx, row in enumerate(rows):
        row_sorted = sorted(row, key=lambda p: p["_cx"])
        for col_idx, p in enumerate(row_sorted):
            p["row"] = row_idx + 1
            p["col"] = col_idx + 1
            p["local_id"] = f"R{row_idx+1:02d}_C{col_idx+1:02d}"

    # Xóa biến tạm
    for p in panels:
        p.pop("_cx", None)
        p.pop("_cy", None)

    return panels


# ─────────────────────────────────────────
# 5. ĐỊNH VỊ LỖI TRONG PANEL
# ─────────────────────────────────────────

def localize_defect_inside_panel(
    defect_center: List[float],
    panel_bbox: List[float]
) -> Dict[str, Any]:
    """
    Tính vị trí tương đối của lỗi (defect) bên trong tấm pin (panel).

    Args:
        defect_center: [cx, cy] của defect (pixel coords trong ảnh gốc)
        panel_bbox: [x1, y1, x2, y2] của panel (pixel coords trong ảnh gốc)

    Returns:
        dict với:
            u (float): 0.0 = trái, 1.0 = phải
            v (float): 0.0 = trên, 1.0 = dưới
            location_in_panel (str): e.g. "upper-left", "middle-center", "lower-right"
    """
    x1, y1, x2, y2 = panel_bbox
    w = max(x2 - x1, 1)
    h = max(y2 - y1, 1)

    cx, cy = defect_center

    # Tọa độ tương đối, clamp vào [0, 1]
    u = max(0.0, min(1.0, (cx - x1) / w))
    v = max(0.0, min(1.0, (cy - y1) / h))

    # Phân vùng horizontal
    if u < 0.33:
        h_zone = "left"
    elif u < 0.66:
        h_zone = "center"
    else:
        h_zone = "right"

    # Phân vùng vertical
    if v < 0.33:
        v_zone = "upper"
    elif v < 0.66:
        v_zone = "middle"
    else:
        v_zone = "lower"

    return {
        "u": round(u, 3),
        "v": round(v, 3),
        "location_in_panel": f"{v_zone}-{h_zone}",
    }
