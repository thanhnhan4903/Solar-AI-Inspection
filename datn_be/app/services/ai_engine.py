# app/services/ai_engine.py
from ultralytics import YOLO
import os

class AIEngine:
    def __init__(self, model_path: str):
        """
        Khởi tạo và load model YOLOv8-seg vào bộ nhớ
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Không tìm thấy file trọng số tại {model_path}")
        self.model_path = model_path
        self.model = YOLO(model_path)

    def reload_model(self):
        """Tải lại model từ file trọng số mới nhất"""
        self.model = YOLO(self.model_path)

    def detect_and_segment(self, image_path: str, conf_threshold=0.25):
        """
        Chạy AI để tìm tấm pin và lỗi.
        Trả về (detections, yolo_result):
          - detections: danh sách dict với polygon, box, class_name, confidence
          - yolo_result: object Result gốc của YOLO (dùng cho .plot())
        """
        results = self.model.predict(source=image_path, conf=conf_threshold, save=False)
        
        detections = []
        result = results[0]  # Lấy kết quả của tấm ảnh đầu tiên

        if result.masks is not None:
            # Duyệt qua từng đối tượng mà AI tìm thấy
            for i, mask in enumerate(result.masks.xy):
                class_id = int(result.boxes.cls[i])
                label = result.names[class_id]
                confidence = float(result.boxes.conf[i])
                
                # Trích xuất danh sách tọa độ đa giác [[x1, y1], [x2, y2], ...]
                polygon_points = mask.tolist()
                
                # Trích xuất bounding box [x1, y1, x2, y2]
                box = result.boxes.xyxy[i].tolist()  # [x1, y1, x2, y2]
                
                detections.append({
                    "class_name": label,
                    "confidence": confidence,
                    "polygon": polygon_points,
                    "box": box
                })
        
        return detections, result