# app/services/image_processor.py
import cv2
import numpy as np
import os

class ImageProcessor:
    @staticmethod
    def get_black_border_mask(img_bgr: np.ndarray, threshold: int = 8) -> np.ndarray:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        mask = gray <= threshold
        return mask.astype(np.uint8) * 255

    @staticmethod
    def restore_black_border(processed_bgr: np.ndarray, original_bgr: np.ndarray, threshold: int = 8) -> np.ndarray:
        mask = ImageProcessor.get_black_border_mask(original_bgr, threshold=threshold)
        out = processed_bgr.copy()
        out[mask > 0] = original_bgr[mask > 0]
        return out

    @staticmethod
    def bad_pixel_replacement_color(img_bgr: np.ndarray, z_thresh: float = 6.0, max_bad_ratio: float = 0.003, inpaint_radius: int = 1):
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l_channel, _, _ = cv2.split(lab)
        l_channel = l_channel.astype(np.float32)

        local_median = cv2.medianBlur(l_channel.astype(np.uint8), 3).astype(np.float32)
        residual = l_channel - local_median

        med = np.median(residual)
        mad = np.median(np.abs(residual - med))

        sigma = 1.4826 * mad
        sigma = max(float(sigma), 1e-6)

        bad_mask = np.abs(residual - med) > z_thresh * sigma
        bad_ratio = float(np.mean(bad_mask))

        if bad_ratio > max_bad_ratio or not np.any(bad_mask):
            return img_bgr.copy()

        mask_u8 = bad_mask.astype(np.uint8) * 255
        repaired = cv2.inpaint(img_bgr, mask_u8, inpaintRadius=inpaint_radius, flags=cv2.INPAINT_TELEA)
        return repaired
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
        Pipeline xử lý ảnh nhiệt chuẩn Roboflow (Color, BPR, Bilateral, Padding Mask)
        """
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            return None

        # 1. NUC (nếu có flat frame)
        if flat_frame is not None:
            img = ImageProcessor.apply_software_nuc(img, flat_frame)

        # 2. Bad Pixel Replacement (BPR)
        img_bpr = ImageProcessor.bad_pixel_replacement_color(img)

        # 3. Bilateral Filter
        img_denoised = cv2.bilateralFilter(img_bpr, d=9, sigmaColor=25, sigmaSpace=25)

        # 4. Khôi phục vùng Black Border ban đầu (Roboflow padding)
        img_restored = ImageProcessor.restore_black_border(img_denoised, img)

        # 5. Letterbox (Resize + Pad) về target shape nếu cần
        h, w = img_restored.shape[:2]
        target_w, target_h = target_shape

        if h == target_h and w == target_w:
            return img_restored

        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)

        resized = cv2.resize(img_restored, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        top = (target_h - new_h) // 2
        bottom = target_h - new_h - top
        left = (target_w - new_w) // 2
        right = target_w - new_w - left

        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )

        return padded