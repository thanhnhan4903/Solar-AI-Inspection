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
            import cv2
            import numpy as np
            
            raw_detections = []
            
            # Duyệt qua từng đối tượng mà AI tìm thấy
            for i, mask in enumerate(result.masks.xy):
                class_id = int(result.boxes.cls[i])
                label = result.names[class_id]
                confidence = float(result.boxes.conf[i])
                
                # --- PHƯƠNG PHÁP HYBRID ---
                # 1. Chuyển list tọa độ của AI thành ma trận ảnh nhị phân (Binary Mask)
                img_h, img_w = result.orig_shape
                binary_mask = np.zeros((img_h, img_w), dtype=np.uint8)
                cv2.fillPoly(binary_mask, [np.array(mask, dtype=np.int32)], 255)

                # 2. Toán tử hình thái học (Morphological Operations)
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
                smoothed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
                smoothed_mask = cv2.morphologyEx(smoothed_mask, cv2.MORPH_OPEN, kernel)

                # 3. Rút trích lại đường viền
                contours, _ = cv2.findContours(smoothed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if len(contours) > 0:
                    largest_contour = max(contours, key=cv2.contourArea)
                    
                    # 4. ÉP BUỘC THÀNH HÌNH TỨ GIÁC (Rotated Rectangle)
                    # Hàm cv2.minAreaRect sẽ vẽ một hình chữ nhật bao quanh khít nhất có thể, bất kể tấm pin bị mẻ góc
                    rect = cv2.minAreaRect(largest_contour)
                    box_points = cv2.boxPoints(rect)
                    polygon_points = np.int32(box_points).tolist()
                    
                    # Tính diện tích thực tế của đa giác bọc ngoài
                    width, height = rect[1]
                    area = width * height
                else:
                    polygon_points = mask.tolist()
                    area = cv2.contourArea(np.array(mask, dtype=np.float32))
                
                # Trích xuất bounding box [x1, y1, x2, y2]
                box = result.boxes.xyxy[i].tolist()
                
                raw_detections.append({
                    "class_name": label,
                    "confidence": confidence,
                    "polygon": polygon_points,
                    "box": box,
                    "area": area
                })
            
            # --- BƯỚC 5: LỌC CÁC TẤM PIN MẺ GÓC / DIỆN TÍCH NHỎ ---
            # Tính toán diện tích trung vị (Median Area) của các tấm pin trong ảnh
            panel_areas = [d["area"] for d in raw_detections if d["class_name"] == "panel"]
            
            if len(panel_areas) > 0:
                median_area = np.median(panel_areas)
                
                for d in raw_detections:
                    if d["class_name"] == "panel":
                        # Thuật toán Logic: Chỉ hiển thị các tấm pin có diện tích dao động không quá lớn so với trung vị (ví dụ 60% đến 150%)
                        # Các tấm pin mẻ góc diện tích nhỏ sẽ bị loại bỏ hoàn toàn
                        if 0.8 * median_area <= d["area"] <= 1.3 * median_area:
                            del d["area"]
                            detections.append(d)
                    else:
                        del d["area"]
                        detections.append(d)
            else:
                # Nếu không có tấm pin nào, trả về toàn bộ
                for d in raw_detections:
                    del d["area"]
                    detections.append(d)
        
        return detections, result