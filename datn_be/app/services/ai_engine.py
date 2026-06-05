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
import os
from typing import Optional, List, Dict, Any, Tuple

from app.services.panel_processor import process_yolo_predictions

# ─────────────────────────────────────────
# CONFIDENCE THRESHOLDS
# ─────────────────────────────────────────
# Panel cần độ chắc chắn cao hơn vì là nền tảng định vị.
# Defect dùng threshold thấp hơn để không bỏ sót lỗi nhỏ.
PANEL_CONF_THRESHOLD = 0.85
DEFECT_CONF_THRESHOLD = 0.25

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
          2. Sử dụng helper function `process_yolo_predictions` để trích xuất 
             panel và defect.
        """
        # Chạy với conf thấp nhất để lấy tất cả, filter sau
        global_conf = min(panel_conf, defect_conf)
        results = self.model.predict(source=image_path, conf=global_conf, save=False, verbose=False)

        result = results[0]

        detections = process_yolo_predictions(
            result=result,
            orig_img=result.orig_img,
            panel_conf=panel_conf,
            defect_conf=defect_conf,
            panel_class_name=PANEL_CLASS_NAME
        )
        return detections, result