# app/services/ai_engine.py
# AI Engine cho hệ thống kiểm tra tấm pin PV bằng YOLOv8-seg.
#
# Nguyên tắc xử lý:
#   - Class 'panel': refine bằng minAreaRect → polygon 4 điểm sạch
#   - Class defect (hotspot/shading/crack/soiling): giữ polygon gốc YOLO
#     (không ép thành hình chữ nhật vì defect có hình dạng bất kỳ)
#   - Confidence threshold riêng: panel dùng cao hơn defect
#   - Input: ảnh nhiệt màu 8-bit (precalib), không cần metadata GPS/temperature
from ultralytics import YOLO
import cv2
import numpy as np
import os
from typing import Optional, List, Dict, Any, Tuple

from app.services.panel_geometry import (
    refine_panel_polygon_from_mask,
    get_polygon_features,
)

# ─────────────────────────────────────────
# CONFIDENCE THRESHOLDS
# ─────────────────────────────────────────
# Panel cần độ chắc chắn cao hơn vì là nền tảng định vị.
# Defect dùng threshold thấp hơn để không bỏ sót lỗi nhỏ.
PANEL_CONF_THRESHOLD = 0.35
DEFECT_CONF_THRESHOLD = 0.20

# Class ID của 'panel' trong dataset
PANEL_CLASS_NAME = "panel"


class AIEngine:
    def __init__(self, model_path: str):
        """
        Khởi tạo và load model YOLOv8-seg vào bộ nhớ.
        model_path: đường dẫn tới file .pt (relative hoặc absolute)
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Không tìm thấy file trọng số tại: {model_path}")
        self.model_path = model_path
        self.model = YOLO(model_path)

    def reload_model(self, new_path: Optional[str] = None):
        """
        Tải lại model. Nếu new_path được cung cấp thì dùng path mới,
        ngược lại reload từ self.model_path hiện tại.
        """
        if new_path:
            if not os.path.exists(new_path):
                raise FileNotFoundError(f"File weights không tồn tại: {new_path}")
            self.model_path = new_path
        self.model = YOLO(self.model_path)

    def detect_and_segment(
        self,
        image_path: str,
        panel_conf: float = PANEL_CONF_THRESHOLD,
        defect_conf: float = DEFECT_CONF_THRESHOLD,
    ) -> Tuple[List[Dict[str, Any]], Any]:
        """
        Chạy YOLOv8-seg trên ảnh nhiệt precalib.

        Pipeline:
          1. YOLO predict với conf = min(panel_conf, defect_conf) để lấy tất cả.
          2. Với class 'panel': refine mask → minAreaRect → polygon 4 điểm.
          3. Với class defect: dùng polygon gốc từ result.masks.xy, không ép rectangle.
          4. Áp dụng confidence filter riêng cho panel và defect.

        Returns:
            (detections, yolo_result)
            - detections: list dict, mỗi dict có đầy đủ:
                class_name, confidence, bbox, polygon, area, center, category
            - yolo_result: Result object gốc của YOLO (dùng cho debug .plot())
        """
        # Chạy với conf thấp nhất để lấy tất cả, filter sau
        global_conf = min(panel_conf, defect_conf)
        results = self.model.predict(source=image_path, conf=global_conf, save=False, verbose=False)

        detections: List[Dict[str, Any]] = []
        result = results[0]

        if result.masks is None:
            # Không có detection nào
            return detections, result

        img_h, img_w = result.orig_shape

        # result.masks.xy: list polygon (np.ndarray) theo pixel coords gốc
        # result.masks.data: tensor mask raster (scaled)
        masks_xy = result.masks.xy         # List[np.ndarray shape (N,2)]
        masks_data = result.masks.data     # Tensor (num_det, mh, mw)

        raw_panels: List[Dict] = []
        raw_defects: List[Dict] = []

        for i in range(len(result.boxes)):
            class_id = int(result.boxes.cls[i])
            class_name = result.names[class_id]
            confidence = float(result.boxes.conf[i])
            bbox = result.boxes.xyxy[i].tolist()  # [x1,y1,x2,y2]

            xy_polygon = masks_xy[i]  # np.ndarray (N,2)

            if class_name.lower() == PANEL_CLASS_NAME:
                # ── PANEL: refine bằng minAreaRect ──
                if confidence < panel_conf:
                    continue

                # Lấy mask raster từ masks.data, resize về ảnh gốc
                mask_tensor = masks_data[i]  # shape (mh, mw), float 0-1
                mask_np = mask_tensor.cpu().numpy()
                mask_u8 = (mask_np > 0.5).astype(np.uint8) * 255

                refined_poly = refine_panel_polygon_from_mask(mask_u8, (img_h, img_w))
                if refined_poly is None or len(refined_poly) < 3:
                    # Fallback: dùng polygon gốc của YOLO
                    refined_poly = xy_polygon.tolist() if len(xy_polygon) >= 3 else []

                if not refined_poly:
                    continue

                features = get_polygon_features(refined_poly)

                raw_panels.append({
                    "class_name": class_name,
                    "confidence": round(confidence, 4),
                    "bbox": [round(v) for v in features["bbox"]],
                    "polygon": refined_poly,
                    "area": features["area"],
                    "center": features["center"],
                    "aspect_ratio": features["aspect_ratio"],
                    "category": "panel",
                    # Backward compat
                    "box": [round(v) for v in bbox],
                })

            else:
                # ── DEFECT: giữ polygon gốc YOLO, không ép rectangle ──
                if confidence < defect_conf:
                    continue

                if len(xy_polygon) < 3:
                    continue

                # Chỉ dùng approxPolyDP nhẹ để giảm điểm dư, kernel nhỏ
                # epsilon nhỏ để giữ sát hình dạng thực
                poly_arr = xy_polygon.astype(np.float32)
                arc_len = cv2.arcLength(poly_arr, True)
                epsilon = 0.005 * arc_len  # 0.5% arc length, rất nhẹ
                approx = cv2.approxPolyDP(poly_arr, epsilon, True)
                defect_poly = approx.reshape(-1, 2).tolist()

                if len(defect_poly) < 3:
                    defect_poly = xy_polygon.tolist()

                features = get_polygon_features(defect_poly)

                raw_defects.append({
                    "class_name": class_name,
                    "confidence": round(confidence, 4),
                    "bbox": [round(v) for v in features["bbox"]],
                    "polygon": defect_poly,
                    "area": features["area"],
                    "center": features["center"],
                    "category": "defect",
                    # Backward compat
                    "box": [round(v) for v in bbox],
                })

        # ── FILTER PANEL theo diện tích (loại panel mẻ góc / nhiễu) ──
        filtered_panels = _filter_panels_by_area(raw_panels)

        detections = filtered_panels + raw_defects
        return detections, result


# ─────────────────────────────────────────
# HELPER: FILTER PANEL THEO MEDIAN AREA
# ─────────────────────────────────────────

def _filter_panels_by_area(panels: List[Dict]) -> List[Dict]:
    """
    Loại bỏ các panel có diện tích lệch quá nhiều so với median.
    Giữ lại panel có area trong khoảng [40%, 250%] của median.
    Nếu chỉ có 1-2 panel thì giữ tất cả (không đủ để tính median tin cậy).
    """
    if len(panels) <= 2:
        return panels

    areas = [p["area"] for p in panels]
    median_area = float(np.median(areas))

    if median_area <= 0:
        return panels

    filtered = []
    for p in panels:
        ratio = p["area"] / median_area
        if 0.40 <= ratio <= 2.50:
            filtered.append(p)

    # Nếu filter quá mạnh (loại > 50%), trả về tất cả để tránh mất data
    if len(filtered) < len(panels) // 2:
        return panels

    return filtered


# ─────────────────────────────────────────
# HELPER: VẼ ANNOTATED IMAGE (DEBUG)
# ─────────────────────────────────────────

def draw_custom_annotation(image_bgr: np.ndarray, panels: List[Dict[str, Any]]) -> np.ndarray:
    """
    Vẽ annotation tùy chỉnh lên ảnh BGR:
    - Panel: polygon refine màu xanh lá, ghi local_id
    - Defect: polygon màu đỏ, ghi class + area_ratio + severity

    Hàm này tạo ảnh annotated chính xác hơn result.plot() vì dùng
    refined polygon và thông tin đã xử lý (assignment, severity).
    """
    img = image_bgr.copy()
    PANEL_COLOR = (0, 200, 80)     # BGR xanh lá
    DEFECT_COLORS = {
        "hotspot_single_cell": (0, 100, 255),   # Cam
        "hotspot_multi_cell":  (0, 0, 255),      # Đỏ
        "shading":             (200, 200, 0),    # Cyan
        "soiling":             (0, 165, 255),    # Cam nhạt
        "crack":               (200, 0, 200),    # Tím
    }
    DEFAULT_DEFECT_COLOR = (100, 100, 255)

    for p in panels:
        p_poly = p.get("polygon", [])
        if len(p_poly) >= 3:
            pts = np.array(p_poly, dtype=np.int32)
            cv2.polylines(img, [pts], True, PANEL_COLOR, 2)

            # Ghi local_id
            cx, cy = p.get("center", [0, 0])
            label = p.get("local_id", p.get("class_name", "panel"))
            cv2.putText(img, label, (int(cx) - 20, int(cy)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, PANEL_COLOR, 1, cv2.LINE_AA)

        for d in p.get("defects", []):
            d_poly = d.get("polygon", [])
            cls = d.get("class_name", "")
            color = DEFECT_COLORS.get(cls, DEFAULT_DEFECT_COLOR)

            if len(d_poly) >= 3:
                pts = np.array(d_poly, dtype=np.int32)
                cv2.fillPoly(img, [pts], (*color[:3], 80))   # không hoạt động trực tiếp nhưng giữ
                cv2.polylines(img, [pts], True, color, 2)

            dcx, dcy = d.get("center", [0, 0])
            info = f"{cls} {d.get('area_ratio_percent', 0):.1f}% {d.get('severity', '')}"
            cv2.putText(img, info, (int(dcx) - 30, int(dcy)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

    return img