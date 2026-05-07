# app/services/image_processor.py
import cv2
import numpy as np
import os

class ImageProcessor:
    @staticmethod
    def apply_software_nuc(raw_image, flat_frame):
        """
        Khối 2: Hiệu chỉnh độ không đồng nhất (NUC)
        """
        raw_float = raw_image.astype(np.float32)
        flat_float = flat_frame.astype(np.float32)
        
        mean_F = np.mean(flat_float)
        normalized_F = flat_float / mean_F
        # Tránh lỗi chia cho 0
        normalized_F[normalized_F == 0] = 1e-5
        
        corrected_float = raw_float / normalized_F
        return np.clip(corrected_float, 0, 255).astype(np.uint8)

    @staticmethod
    def preprocess_thermal(image_path: str, flat_frame=None, target_shape=(640, 512)):
        """
        Pipeline xử lý ảnh nhiệt theo chuẩn mới nhất của bạn
        """
        # 1. Đọc ảnh (Sử dụng IMREAD_COLOR như code của bạn)
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            return None

        # 2. --- NUC (Nếu có flat-frame) ---
        if flat_frame is not None:
            img = ImageProcessor.apply_software_nuc(img, flat_frame)

        # 3. --- Bilateral Filter (Khử nhiễu bảo toàn biên) ---
        # Thay thế Median Blur cũ để giữ biên đa giác sắc nét hơn
        img_denoised = cv2.bilateralFilter(img, d=5, sigmaColor=15, sigmaSpace=15)

        # 4. --- Letterbox (Giữ tỷ lệ khung hình) ---
        h, w = img_denoised.shape[:2]
        target_w, target_h = target_shape

        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)

        resized = cv2.resize(img_denoised, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Tính toán padding (viền đen)
        top = (target_h - new_h) // 2
        bottom = target_h - new_h - top
        left = (target_w - new_w) // 2
        right = target_w - new_w - left

        # Sử dụng copyMakeBorder cho chuẩn Clean Code của OpenCV
        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )

        return padded