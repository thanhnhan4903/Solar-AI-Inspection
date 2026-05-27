import cv2
import numpy as np
from typing import List, Dict, Any, Tuple

from app.services.panel_geometry import get_polygon_features


def filter_panels_by_area(panels: List[Dict]) -> List[Dict]:
    """
    Loại bỏ các panel có diện tích lệch quá nhiều so với median (quá 30%).
    Sử dụng median (trung vị) là phương pháp tốt nhất vì nó không bị nhiễu
    bởi các panel false positive (rất nhỏ) hoặc panel dính chùm (rất to).
    """
    if len(panels) <= 2:
        return panels

    # Tính diện tích thực tế từ polygon (p["area"]) thay vì Bounding Box!
    # Bounding Box của panel nghiêng sẽ to hơn panel nằm ngang, gây filter sai.
    areas = [p.get("area", 0) for p in panels]
    median_area = float(np.median(areas))

    if median_area <= 0:
        return panels

    filtered = []
    for i, p in enumerate(panels):
        ratio = areas[i] / median_area
        # Nới lỏng filter (0.7 - 1.3) để không lỡ tay xóa nhầm các panel bị nghiêng/lệch nhỏ
        if 0.70 <= ratio <= 1.30:
            filtered.append(p)

    # Nếu filter quá mạnh (loại > 50%), trả về tất cả để tránh mất data
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
        
        # Cạnh đối diện không được lệch quá 10% chiều dài (tránh méo góc do dính điểm nóng nhiệt)
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
    # 1. Tính centroid của 4 điểm
    cx, cy = np.mean(box, axis=0)
    
    # 2. Tính góc của mỗi điểm so với centroid
    angles = np.arctan2(box[:, 1] - cy, box[:, 0] - cx)
    
    # 3. Sắp xếp các điểm theo góc tăng dần (từ -pi đến pi)
    # Trong hệ tọa độ màn hình (y hướng xuống), góc tăng dần tương ứng với thứ tự kim đồng hồ:
    # Top-Left (~ -3pi/4), Top-Right (~ -pi/4), Bottom-Right (~ pi/4), Bottom-Left (~ 3pi/4)
    sorted_box = box[np.argsort(angles)]
    
    # 4. Để đảm bảo Top-Left luôn ở vị trí đầu tiên (tránh xoay sai trục):
    # Tìm điểm có khoảng cách nhỏ nhất tới góc trên bên trái của bbox [min_x, min_y]
    min_x, min_y = np.min(box, axis=0)
    distances = np.linalg.norm(sorted_box - [min_x, min_y], axis=1)
    tl_idx = np.argmin(distances)
    
    # Lăn mảng để Top-Left lên đầu
    sorted_box = np.roll(sorted_box, -tl_idx, axis=0)
    
    # Ép kiểu sang int/float chuẩn để tránh lỗi JSON serialization
    return [[int(round(pt[0])), int(round(pt[1]))] for pt in sorted_box]


def refine_panel_contour(orig_img: np.ndarray, bbox: List[float], xy_polygon: np.ndarray) -> List[List[int]]:
    """
    Tinh chỉnh viền tấm pin từ đa giác thô YOLO (xy_polygon) để ôm sát viền nghiêng (perspective)
    mà không bao giờ lem/đè sang tấm pin lân cận (do được cô lập hoàn toàn trong instance mask):
    Ưu tiên 1: Convex Hull + approxPolyDP thích ứng để tìm đúng tứ giác lệch (4 góc) hợp lệ
    Ưu tiên 2: Rotated Min Area Rect (cv2.minAreaRect) cho tứ giác xoay khít chặt chẽ
    Ưu tiên 3: Fallback YOLO Bbox làm tứ giác xoay phẳng
    """
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    bbox_poly = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    
    if xy_polygon is None or len(xy_polygon) < 3:
        return bbox_poly
        
    try:
        pts = xy_polygon.astype(np.float32)
        # Tạo Convex Hull để làm mịn viền răng cưa và cô lập riêng tấm pin đang xét
        hull = cv2.convexHull(pts)
        
        # 1. ƯU TIÊN 1: Thử tìm đúng tứ giác (4 đỉnh) có góc nghiêng chuẩn và chất lượng tốt
        arc_len = cv2.arcLength(hull, True)
        for factor in np.linspace(0.005, 0.15, 120):
            epsilon = factor * arc_len
            approx = cv2.approxPolyDP(hull, epsilon, True)
            approx = approx.reshape(-1, 2)
            if len(approx) == 4:
                if _check_geometry_quality(approx, bbox, orig_img.shape):
                    return _sort_corners(approx)
                    
        # 2. ƯU TIÊN 2: Rotated Min Area Rect xoay khít chặt chẽ
        rect = cv2.minAreaRect(pts)
        box = cv2.boxPoints(rect)
        if len(box) == 4:
            return _sort_corners(box)
                
    except Exception:
        pass
        
    # 3. ƯU TIÊN 3: Fallback về YOLO Bbox làm tứ giác xoay phẳng
    return bbox_poly


def simplify_defect_polygon(xy_polygon: np.ndarray, max_vertices: int = 6) -> List[List[float]]:
    """
    Rút gọn đa giác lỗi (defect polygon) về tối đa max_vertices cạnh (thường là 6),
    loại bỏ răng cưa chi tiết quá mức để đường bao mượt mà và đẹp mắt.
    """
    poly_arr = xy_polygon.astype(np.float32)
    arc_len = cv2.arcLength(poly_arr, True)
    
    # Thử tăng dần epsilon cho đến khi số đỉnh đạt <= max_vertices
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

        if class_name.lower() == panel_class_name:
            if confidence < panel_conf or len(xy_polygon) < 3:
                continue

            # Sử dụng quy trình CV 3 cấp độ ưu tiên để tìm viền tấm pin hoàn hảo
            panel_poly = refine_panel_contour(orig_img, bbox, xy_polygon)

            features = get_polygon_features(panel_poly)

            raw_panels.append({
                "class_name": class_name,
                "confidence": round(confidence, 4),
                "bbox": [round(v) for v in features["bbox"]],
                "polygon": panel_poly,
                "area": features["area"],
                "center": features["center"],
                "aspect_ratio": features["aspect_ratio"],
                "category": "panel",
                "box": [round(v) for v in bbox],
            })

        else:
            # ── DEFECT: giữ polygon gốc YOLO ──
            if confidence < defect_conf:
                continue

            if len(xy_polygon) < 3:
                continue

            # [ĐÃ BỎ THEO YÊU CẦU]: Không làm gọn đa giác nữa, dùng trực tiếp 100% tọa độ mask từ YOLO
            # poly_arr = xy_polygon.astype(np.float32)
            # arc_len = cv2.arcLength(poly_arr, True)
            # epsilon = 0.005 * arc_len  
            # approx = cv2.approxPolyDP(poly_arr, epsilon, True)
            # defect_poly = approx.reshape(-1, 2).tolist()

            # if len(defect_poly) < 3:
            #     defect_poly = xy_polygon.tolist()
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
            })

    # ── FILTER PANEL theo diện tích ──
    # [ĐÃ BỎ THEO YÊU CẦU]: Bỏ lọc panel theo diện tích để trả về 100% kết quả từ YOLO
    # filtered_panels = filter_panels_by_area(raw_panels)
    filtered_panels = raw_panels

    detections = filtered_panels + raw_defects
    return detections


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
            # Vẽ viền panel xanh dương
            cv2.polylines(img, [pts], True, (255, 0, 0), 2, cv2.LINE_AA)
            cx, cy = p.get("center", [0, 0])
            label = p.get("local_id", p.get("class_name", "panel"))
            cv2.putText(img, label, (int(cx) - 20, int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1, cv2.LINE_AA)

        # Vẽ defect trong panel này
        for d in p.get("defects", []):
            d_poly = d.get("polygon", [])
            cls = d.get("class_name", "")
            color = DEFECT_COLORS.get(cls, DEFAULT_DEFECT_COLOR)

            if len(d_poly) >= 3:
                pts = np.array(d_poly, dtype=np.int32)
                overlay = img.copy()
                cv2.fillPoly(overlay, [pts], color=color)
                img = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)
                cv2.polylines(img, [pts], True, (0, 255, 255), 2, cv2.LINE_AA) # Viền vàng

                dcx, dcy = d.get("center", [0, 0])
                info = f"{cls} {d.get('area_ratio_percent', 0):.1f}%"
                cv2.putText(img, info, (int(dcx) - 30, int(dcy)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)

    # 2. VẼ DEFECT BỊ LẠC TRÔI (UNASSIGNED) - ĐỂ CHECK XEM CÓ BỊ NGHẼN Ở LOGIC KHÔNG
    if unassigned_defects:
        for d in unassigned_defects:
            d_poly = d.get("polygon", [])
            cls = d.get("class_name", "")
            color = DEFECT_COLORS.get(cls, DEFAULT_DEFECT_COLOR)
            
            if len(d_poly) >= 3:
                pts = np.array(d_poly, dtype=np.int32)
                # Vẽ viền ĐỎ ĐỨT ĐOẠN hoặc nét dày để cảnh báo lỗi này bị lạc trôi bên ngoài panel
                cv2.polylines(img, [pts], True, (0, 0, 255), 3, cv2.LINE_AA) 
                dcx, dcy = d.get("center", [0, 0])
                cv2.putText(img, f"[LOST] {cls}", (int(dcx) - 30, int(dcy)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)

    return img