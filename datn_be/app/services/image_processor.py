# app/services/image_processor.py

import cv2
import numpy as np


class ImageProcessor:
    """
    Thermal image preprocessing for 8-bit color thermal images.

    Current practical pipeline:
    - Keep the old pipeline that works well with the current YOLO model.
    - Add non-invasive quality metrics.
    - Do not use CLAHE / Auto-Contrast / palette conversion.
    """

    # ============================================================
    # Black border utilities
    # ============================================================

    @staticmethod
    def get_black_border_mask(img_bgr: np.ndarray, threshold: int = 8) -> np.ndarray:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        mask = gray <= threshold
        return mask.astype(np.uint8) * 255

    @staticmethod
    def restore_black_border(
        processed_bgr: np.ndarray,
        original_bgr: np.ndarray,
        threshold: int = 8
    ) -> np.ndarray:
        mask = ImageProcessor.get_black_border_mask(original_bgr, threshold=threshold)
        out = processed_bgr.copy()
        out[mask > 0] = original_bgr[mask > 0]
        return out

    @staticmethod
    def compute_black_border_ratio(img_bgr: np.ndarray, threshold: int = 8) -> float:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray <= threshold))

    # ============================================================
    # Bad Pixel Replacement
    # ============================================================

    @staticmethod
    def bad_pixel_replacement_color(
        img_bgr: np.ndarray,
        z_thresh: float = 6.0,
        max_bad_ratio: float = 0.003,
        inpaint_radius: int = 1
    ) -> np.ndarray:
        """
        Lightweight bad-pixel replacement on Lab-L channel.

        Note:
        This is not true camera-level BPR because no camera calibration map is available.
        It is a conservative image-processing approximation for isolated abnormal pixels.
        """
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

        # If too many pixels are flagged, it is probably real image content, not bad pixels.
        if bad_ratio > max_bad_ratio or not np.any(bad_mask):
            return img_bgr.copy()

        mask_u8 = bad_mask.astype(np.uint8) * 255
        repaired = cv2.inpaint(
            img_bgr,
            mask_u8,
            inpaintRadius=inpaint_radius,
            flags=cv2.INPAINT_TELEA
        )

        return repaired

    # ============================================================
    # Optional software NUC
    # ============================================================

    @staticmethod
    def apply_software_nuc(raw_image: np.ndarray, flat_frame: np.ndarray) -> np.ndarray:
        """
        Optional software flat-field correction.

        This is only valid if a real flat-frame calibration image is available.
        In the current project, flat_frame is normally None, so this step is not used.
        """
        raw_float = raw_image.astype(np.float32)
        flat_float = flat_frame.astype(np.float32)

        mean_F = np.mean(flat_float)
        normalized_F = flat_float / mean_F
        normalized_F[normalized_F == 0] = 1e-5

        corrected_float = raw_float / normalized_F
        return np.clip(corrected_float, 0, 255).astype(np.uint8)

    # ============================================================
    # Quality metrics
    # ============================================================

    @staticmethod
    def compute_blur_score(img_bgr: np.ndarray) -> float:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    @staticmethod
    def compute_noise_score(img_bgr: np.ndarray) -> float:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        smooth = cv2.GaussianBlur(gray, (3, 3), 0)
        residual = gray.astype(np.float32) - smooth.astype(np.float32)
        return float(np.std(residual))

    @staticmethod
    def compute_exposure_stats(img_bgr: np.ndarray) -> dict:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        p01 = float(np.percentile(gray, 1))
        p99 = float(np.percentile(gray, 99))

        return {
            "brightness_mean": float(np.mean(gray)),
            "brightness_std": float(np.std(gray)),
            "p01": p01,
            "p99": p99,
            "dynamic_range": float(p99 - p01),
            "low_saturation_ratio": float(np.mean(gray <= 5)),
            "high_saturation_ratio": float(np.mean(gray >= 250)),
        }

    # ============================================================
    # Issue labels (Vietnamese)
    # ============================================================

    ISSUE_LABELS = {
        "possible_motion_blur_or_out_of_focus":     "Ảnh bị mờ (rung máy hoặc mất nét)",
        "low_contrast":                             "Độ tương phản thấp",
        "low_dynamic_range":                        "Dải sáng hẹp (thiếu chi tiết nhiệt)",
        "possible_overexposure_or_saturation":      "Ảnh có thể bị cháy sáng",
        "possible_underexposure_or_excessive_dark_area": "Ảnh quá tối hoặc thiếu sáng",
        "excessive_black_border":                   "Viền đen quá nhiều (>50% diện tích)",
        "high_image_noise":                         "Nhiễu ảnh cao",
    }

    @staticmethod
    def assess_quality(img_bgr: np.ndarray) -> dict:
        """
        Image quality assessment with 3-tier classification.

        Tiers:
          ok      — 0 issues detected
          warning — 1-2 issues detected (suggest proceeding with caution)
          poor    — ≥3 issues detected (recommend recapture)

        This function only reports quality metrics. It does not modify the image.
        """
        blur_score = ImageProcessor.compute_blur_score(img_bgr)
        noise_score = ImageProcessor.compute_noise_score(img_bgr)
        exposure = ImageProcessor.compute_exposure_stats(img_bgr)
        black_border_ratio = ImageProcessor.compute_black_border_ratio(img_bgr)

        issues = []

        # Thresholds tuned for 8-bit color thermal images.
        # Revisit with real project data after first production run.
        if blur_score < 60:
            issues.append("possible_motion_blur_or_out_of_focus")

        if exposure["brightness_std"] < 15:
            issues.append("low_contrast")

        if exposure["dynamic_range"] < 40:
            issues.append("low_dynamic_range")

        if exposure["high_saturation_ratio"] > 0.15:
            issues.append("possible_overexposure_or_saturation")

        if exposure["low_saturation_ratio"] > 0.30:
            issues.append("possible_underexposure_or_excessive_dark_area")

        if black_border_ratio > 0.50:
            issues.append("excessive_black_border")

        if noise_score > 18:
            issues.append("high_image_noise")

        # 3-tier classification based on issue count
        n = len(issues)
        if n == 0:
            quality_status = "ok"
        elif n <= 2:
            quality_status = "warning"
        else:
            quality_status = "poor"

        # Human-readable Vietnamese labels
        issues_vi = [
            ImageProcessor.ISSUE_LABELS.get(code, code)
            for code in issues
        ]

        return {
            "quality_status": quality_status,
            "issues": issues,
            "issues_vi": issues_vi,
            "metrics": {
                "blur_score": blur_score,
                "noise_score": noise_score,
                "black_border_ratio": black_border_ratio,
                **exposure
            }
        }

    # ============================================================
    # Resize / Letterbox
    # ============================================================

    @staticmethod
    def letterbox_to_shape(
        img_bgr: np.ndarray,
        target_shape=(640, 512)
    ) -> np.ndarray:
        """
        Letterbox resize to target_shape = (target_w, target_h).

        This matches Roboflow "Resize: Fit black edges in 640x512".
        """
        h, w = img_bgr.shape[:2]
        target_w, target_h = target_shape

        if h == target_h and w == target_w:
            return img_bgr.copy()

        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        resized = cv2.resize(
            img_bgr,
            (new_w, new_h),
            interpolation=cv2.INTER_LINEAR
        )

        top = (target_h - new_h) // 2
        bottom = target_h - new_h - top
        left = (target_w - new_w) // 2
        right = target_w - new_w - left

        padded = cv2.copyMakeBorder(
            resized,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0)
        )

        return padded

    # ============================================================
    # Main preprocessing pipeline
    # ============================================================

    @staticmethod
    def preprocess_thermal(
        image_path: str,
        flat_frame=None,
        target_shape=(640, 512),
        return_quality: bool = False
    ):
        """
        Main pre-YOLO preprocessing pipeline.

        Practical version for current YOLO model:
        1. cv2.imread
        2. Optional software NUC only if flat_frame is provided
        3. Bad Pixel Replacement approximation
        4. Bilateral filtering
        5. Restore black border
        6. Letterbox to 640x512 if needed
        7. Optional quality metrics

        No CLAHE.
        No Auto-Contrast.
        No palette conversion.
        """
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)

        if img is None:
            if return_quality:
                return None, {
                    "quality_status": "error",
                    "issues": ["image_read_failed"],
                    "metrics": {}
                }
            return None

        original = img.copy()

        # Optional NUC only when a real flat-frame exists.
        if flat_frame is not None:
            img = ImageProcessor.apply_software_nuc(img, flat_frame)
            original = img.copy()

        # BPR approximation.
        img_bpr = ImageProcessor.bad_pixel_replacement_color(img)

        # Keep the old bilateral settings because they work better with current model.
        img_denoised = cv2.bilateralFilter(
            img_bpr,
            d=9,
            sigmaColor=25,
            sigmaSpace=25
        )

        # Restore original black padding.
        img_restored = ImageProcessor.restore_black_border(
            img_denoised,
            original
        )

        # Letterbox / resize.
        img_out = ImageProcessor.letterbox_to_shape(
            img_restored,
            target_shape=target_shape
        )

        if return_quality:
            quality = ImageProcessor.assess_quality(img_out)
            return img_out, quality

        return img_out