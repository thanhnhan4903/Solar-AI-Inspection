# app/services/image_processor.py

import os
import json
import cv2
import numpy as np
from pathlib import Path

DEFAULT_THRESHOLD_JSON_PATH = Path(__file__).parent.parent / "core" / "empirical_qc_thresholds.json"

class ImageProcessor:
    """
    Thermal image preprocessing and quality control service.
    Integrated from qualitycontrol.py logic.
    """

    ISSUE_LABELS = {
        "reject_too_blurry": "Ảnh quá mờ (không thể chấp nhận)",
        "warn_blurry": "Ảnh hơi mờ (cảnh báo)",
        "reject_too_noisy": "Ảnh quá nhiễu (không thể chấp nhận)",
        "warn_noisy": "Ảnh hơi nhiễu (cảnh báo)",
        "reject_low_contrast": "Độ tương phản quá thấp (không thể chấp nhận)",
        "warn_low_contrast": "Độ tương phản thấp (cảnh báo)",
        "reject_low_dynamic_range": "Dải động quá hẹp (không thể chấp nhận)",
        "warn_low_dynamic_range": "Dải động hẹp (cảnh báo)",
        "reject_too_dark_or_black": "Ảnh quá tối hoặc đen (không thể chấp nhận)",
        "warn_too_dark_or_black": "Ảnh hơi tối/đen (cảnh báo)",
        "reject_too_saturated_hot": "Ảnh bị cháy sáng/bão hòa nhiệt quá nhiều (không thể chấp nhận)",
        "warn_too_saturated_hot": "Ảnh hơi cháy sáng/bão hòa nhiệt (cảnh báo)",
        "reject_excessive_black_edges": "Viền đen xung quanh quá nhiều (không thể chấp nhận)",
        "warn_black_edges": "Viền đen xung quanh hơi nhiều (cảnh báo)",
        "reject_too_many_quality_warnings": "Ảnh có quá nhiều cảnh báo chất lượng",
        "image_read_failed": "Lỗi đọc file ảnh",
    }

    # ============================================================
    # Threshold Loader
    # ============================================================

    @staticmethod
    def load_thresholds(threshold_json_path=None) -> dict:
        if threshold_json_path is None:
            threshold_json_path = DEFAULT_THRESHOLD_JSON_PATH

        default_thresholds = {
            "blur_warn_below": 121.01289367675781,
            "blur_reject_below": 17.790084838867188,
            "noise_warn_above": 23.92720603942871,
            "noise_reject_above": 29.88382339477539,
            "denoise_apply_above": 19.96678924560547,
            "contrast_warn_below": 42.662113189697266,
            "contrast_reject_below": 22.792781829833984,
            "dynamic_range_warn_below": 169.5500030517578,
            "dynamic_range_reject_below": 77.36499786376953,
            "low_saturation_warn_above": 0.3251809775829315,
            "low_saturation_reject_above": 0.6088218688964844,
            "high_saturation_warn_above": 0.0018350220052525401,
            "high_saturation_reject_above": 0.005204254295676947,
            "edge_black_warn_above": 0.6525292992591858,
            "edge_black_reject_above": 0.805412232875824,
        }

        if not os.path.exists(threshold_json_path):
            return default_thresholds

        try:
            with open(threshold_json_path, "r", encoding="utf-8") as f:
                thresholds = json.load(f)
            # Merge with defaults to ensure completeness
            for key, val in default_thresholds.items():
                if key not in thresholds:
                    thresholds[key] = val
            return thresholds
        except Exception:
            return default_thresholds

    # ============================================================
    # Metric Calculations
    # ============================================================

    @staticmethod
    def to_gray(img_bgr: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def compute_blur_score(img_bgr: np.ndarray) -> float:
        gray = ImageProcessor.to_gray(img_bgr)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    @staticmethod
    def compute_noise_score(img_bgr: np.ndarray) -> float:
        gray = ImageProcessor.to_gray(img_bgr)
        smooth = cv2.GaussianBlur(gray, (3, 3), 0)
        residual = gray.astype(np.float32) - smooth.astype(np.float32)
        return float(np.std(residual))

    @staticmethod
    def compute_exposure_stats(img_bgr: np.ndarray) -> dict:
        gray = ImageProcessor.to_gray(img_bgr)
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

    @staticmethod
    def compute_edge_black_ratio(img_bgr: np.ndarray, threshold: int = 8, edge_percent: float = 0.04) -> float:
        gray = ImageProcessor.to_gray(img_bgr)
        h, w = gray.shape[:2]

        eh = max(1, int(round(h * edge_percent)))
        ew = max(1, int(round(w * edge_percent)))

        top = gray[:eh, :]
        bottom = gray[h - eh:, :]
        left = gray[:, :ew]
        right = gray[:, w - ew:]

        edge_pixels = np.concatenate([
            top.reshape(-1),
            bottom.reshape(-1),
            left.reshape(-1),
            right.reshape(-1),
        ])

        return float(np.mean(edge_pixels <= threshold))

    @staticmethod
    def compute_all_metrics(img_bgr: np.ndarray) -> dict:
        exposure = ImageProcessor.compute_exposure_stats(img_bgr)
        return {
            "blur_score": ImageProcessor.compute_blur_score(img_bgr),
            "noise_score": ImageProcessor.compute_noise_score(img_bgr),
            "edge_black_ratio": ImageProcessor.compute_edge_black_ratio(img_bgr),
            **exposure,
        }

    # ============================================================
    # Black Border & Padding Utilities
    # ============================================================

    @staticmethod
    def get_black_border_mask(img_bgr: np.ndarray, threshold: int = 8) -> np.ndarray:
        gray = ImageProcessor.to_gray(img_bgr)
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
        gray = ImageProcessor.to_gray(img_bgr)
        return float(np.mean(gray <= threshold))

    @staticmethod
    def detect_existing_black_padding(
        img_bgr: np.ndarray,
        black_threshold: int = 8,
        line_black_ratio: float = 0.95,
        min_thickness_px: int = 3,
    ) -> dict:
        gray = ImageProcessor.to_gray(img_bgr)
        black = gray <= black_threshold
        h, w = black.shape[:2]

        def count_from_top():
            count = 0
            for r in range(h):
                if float(np.mean(black[r, :])) >= line_black_ratio:
                    count += 1
                else:
                    break
            return count

        def count_from_bottom():
            count = 0
            for r in range(h - 1, -1, -1):
                if float(np.mean(black[r, :])) >= line_black_ratio:
                    count += 1
                else:
                    break
            return count

        def count_from_left():
            count = 0
            for c in range(w):
                if float(np.mean(black[:, c])) >= line_black_ratio:
                    count += 1
                else:
                    break
            return count

        def count_from_right():
            count = 0
            for c in range(w - 1, -1, -1):
                if float(np.mean(black[:, c])) >= line_black_ratio:
                    count += 1
                else:
                    break
            return count

        top = count_from_top()
        bottom = count_from_bottom()
        left = count_from_left()
        right = count_from_right()

        has_existing_pad = max(top, bottom, left, right) >= min_thickness_px

        return {
            "has_existing_pad": bool(has_existing_pad),
            "detected_pad_top": int(top),
            "detected_pad_bottom": int(bottom),
            "detected_pad_left": int(left),
            "detected_pad_right": int(right),
            "pad_black_threshold": int(black_threshold),
            "pad_line_black_ratio": float(line_black_ratio),
            "pad_min_thickness_px": int(min_thickness_px),
        }

    @staticmethod
    def make_padding_mask(img_bgr: np.ndarray, pad_info: dict) -> np.ndarray:
        h, w = img_bgr.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        top = int(pad_info.get("detected_pad_top", 0))
        bottom = int(pad_info.get("detected_pad_bottom", 0))
        left = int(pad_info.get("detected_pad_left", 0))
        right = int(pad_info.get("detected_pad_right", 0))

        if top > 0:
            mask[:top, :] = 255
        if bottom > 0:
            mask[h - bottom:, :] = 255
        if left > 0:
            mask[:, :left] = 255
        if right > 0:
            mask[:, w - right:] = 255

        return mask

    @staticmethod
    def should_restore_padding(pad_mode: str, pad_info: dict) -> bool:
        if pad_mode == "off":
            return False
        if pad_mode == "on" or pad_mode == "auto":
            return bool(pad_info.get("has_existing_pad", False))
        raise ValueError("PAD_MODE phải là 'off', 'on', hoặc 'auto'")

    # ============================================================
    # Bad Pixel Replacement & Flat Field Correction
    # ============================================================

    @staticmethod
    def bad_pixel_replacement_color(
        img_bgr: np.ndarray,
        z_thresh: float = 6.0,
        max_bad_ratio: float = 0.003,
        inpaint_radius: int = 1
    ) -> np.ndarray:
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l_channel, _, _ = cv2.split(lab)
        l_float = l_channel.astype(np.float32)

        local_median = cv2.medianBlur(l_channel, 3).astype(np.float32)
        residual = l_float - local_median

        med = np.median(residual)
        mad = np.median(np.abs(residual - med))

        sigma = 1.4826 * mad
        sigma = max(float(sigma), 1e-6)

        bad_mask = np.abs(residual - med) > z_thresh * sigma
        bad_ratio = float(np.mean(bad_mask))

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

    @staticmethod
    def apply_software_nuc(raw_image: np.ndarray, flat_frame: np.ndarray) -> np.ndarray:
        raw_float = raw_image.astype(np.float32)
        flat_float = flat_frame.astype(np.float32)

        mean_F = np.mean(flat_float)
        normalized_F = flat_float / mean_F
        normalized_F[normalized_F == 0] = 1e-5

        corrected_float = raw_float / normalized_F
        return np.clip(corrected_float, 0, 255).astype(np.uint8)

    # ============================================================
    # Quality Assessment
    # ============================================================

    @staticmethod
    def assess_quality(img_bgr: np.ndarray, thresholds: dict = None) -> dict:
        if thresholds is None:
            thresholds = ImageProcessor.load_thresholds()

        # Automatically letterbox to (640, 512) for consistent quality metrics
        h, w = img_bgr.shape[:2]
        if (w, h) != (640, 512):
            qc_img = ImageProcessor.letterbox_to_shape(img_bgr, (640, 512))
        else:
            qc_img = img_bgr

        metrics = ImageProcessor.compute_all_metrics(qc_img)

        warnings = []
        critical = []

        # Blur check
        if metrics["blur_score"] < thresholds["blur_reject_below"]:
            critical.append("reject_too_blurry")
        elif metrics["blur_score"] < thresholds["blur_warn_below"]:
            warnings.append("warn_blurry")

        # Noise check
        if metrics["noise_score"] > thresholds["noise_reject_above"]:
            critical.append("reject_too_noisy")
        elif metrics["noise_score"] > thresholds["noise_warn_above"]:
            warnings.append("warn_noisy")

        # Contrast check (brightness_std)
        if metrics["brightness_std"] < thresholds["contrast_reject_below"]:
            critical.append("reject_low_contrast")
        elif metrics["brightness_std"] < thresholds["contrast_warn_below"]:
            warnings.append("warn_low_contrast")

        # Dynamic range check
        if metrics["dynamic_range"] < thresholds["dynamic_range_reject_below"]:
            critical.append("reject_low_dynamic_range")
        elif metrics["dynamic_range"] < thresholds["dynamic_range_warn_below"]:
            warnings.append("warn_low_dynamic_range")

        # Low saturation (underexposure / dark)
        if metrics["low_saturation_ratio"] > thresholds["low_saturation_reject_above"]:
            critical.append("reject_too_dark_or_black")
        elif metrics["low_saturation_ratio"] > thresholds["low_saturation_warn_above"]:
            warnings.append("warn_too_dark_or_black")

        # High saturation (overexposure / hot)
        if metrics["high_saturation_ratio"] > thresholds["high_saturation_reject_above"]:
            critical.append("reject_too_saturated_hot")
        elif metrics["high_saturation_ratio"] > thresholds["high_saturation_warn_above"]:
            warnings.append("warn_too_saturated_hot")

        # Edge black ratio check
        if metrics["edge_black_ratio"] > thresholds["edge_black_reject_above"]:
            critical.append("reject_excessive_black_edges")
        elif metrics["edge_black_ratio"] > thresholds["edge_black_warn_above"]:
            warnings.append("warn_black_edges")

        # Warning threshold logic
        if len(warnings) >= 3:
            critical.append("reject_too_many_quality_warnings")

        if critical:
            status = "poor"  # Map "critical" -> "poor"
        elif warnings:
            status = "warning"
        else:
            status = "ok"

        issues = critical + warnings
        issues_vi = [ImageProcessor.ISSUE_LABELS.get(issue, issue) for issue in issues]

        return {
            "quality_status": status,
            "issues": issues,
            "issues_vi": issues_vi,
            "metrics": metrics,
        }

    # ============================================================
    # Letterbox & Resize
    # ============================================================

    @staticmethod
    def letterbox_to_shape(
        img_bgr: np.ndarray,
        target_shape=(640, 512)
    ) -> np.ndarray:
        h, w = img_bgr.shape[:2]
        target_w, target_h = target_shape

        if w == target_w and h == target_h:
            return img_bgr.copy()

        scale = min(target_w / w, target_h / h)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))

        resized = cv2.resize(
            img_bgr,
            (new_w, new_h),
            interpolation=cv2.INTER_LINEAR,
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
            value=(0, 0, 0),
        )
        return padded

    # ============================================================
    # Main Preprocessing Pipeline
    # ============================================================

    @staticmethod
    def preprocess_thermal(
        image_path: str,
        flat_frame=None,
        target_shape=(640, 512),
        return_quality: bool = False
    ):
        """
        Main preprocessing pipeline matching qualitycontrol.py.
        """
        thresholds = ImageProcessor.load_thresholds()
        
        # Support Unicode paths
        try:
            data = np.fromfile(str(image_path), dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        except Exception:
            img = None

        if img is None:
            if return_quality:
                return None, {
                    "quality_status": "error",
                    "issues": ["image_read_failed"],
                    "issues_vi": ["Lỗi đọc file ảnh"],
                    "metrics": {}
                }
            return None

        original = img.copy()

        # 1. Detect existing black padding (PAD_MODE = 'auto')
        pad_info = ImageProcessor.detect_existing_black_padding(original)
        restore_existing_pad = ImageProcessor.should_restore_padding("auto", pad_info)

        # 2. Assess quality before processing
        # Quality assessment automatically letterboxes raw image internally
        quality = ImageProcessor.assess_quality(original, thresholds)

        # 3. Optional Software NUC
        if flat_frame is not None:
            img = ImageProcessor.apply_software_nuc(img, flat_frame)
            original = img.copy()

        # 4. Bad Pixel Replacement
        img = ImageProcessor.bad_pixel_replacement_color(img)

        # 5. Denoise if noise exceeds threshold
        noise_score = quality["metrics"].get("noise_score", 0.0)
        apply_denoise_threshold = float(thresholds["denoise_apply_above"])
        if noise_score > apply_denoise_threshold:
            img = cv2.bilateralFilter(
                img,
                d=7,
                sigmaColor=20,
                sigmaSpace=20
            )

        # 6. Restore original black border if existed
        if restore_existing_pad:
            mask = ImageProcessor.make_padding_mask(original, pad_info)
            img[mask > 0] = original[mask > 0]

        # 7. Letterbox to target shape
        img_out = ImageProcessor.letterbox_to_shape(img, target_shape)

        # 8. Recalculate metrics on preprocessed image for final report
        final_quality = ImageProcessor.assess_quality(img_out, thresholds)
        
        if return_quality:
            return img_out, final_quality

        return img_out