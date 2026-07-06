import cv2
import csv
import json
import shutil
import numpy as np
from pathlib import Path


# ============================================================
# CẤU HÌNH CHÍNH
# ============================================================

# Thư mục ảnh đầu vào của hệ thống
INPUT_DIR = r"E:\DATN\ImageXau"

# Thư mục lưu ảnh sau tiền xử lý + ảnh bị loại + report
OUTPUT_DIR = r"E:\DATN\ImageXau_preprocessed"

# File threshold JSON bạn đã calibrate từ tập train/reference
THRESHOLD_JSON = r"E:\DATN\code\empirical_qc_thresholds.json"

# Kích thước ảnh train/model: width, height
TARGET_SHAPE = (640, 512)

# Các định dạng ảnh được nhận
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]

# Lưu report
SAVE_REPORT_JSON = True
SAVE_REPORT_CSV = True

# Nếu ảnh có nhiều warning cùng lúc thì loại.
# Đặt None nếu không muốn dùng luật này.
REJECT_IF_WARNING_COUNT_AT_LEAST = 3


# ============================================================
# CẤU HÌNH PAD ĐEN
# ============================================================

# PAD_MODE:
# - "off" : không bao giờ khôi phục pad đen gốc
# - "on"  : luôn khôi phục pad đen gốc nếu phát hiện được vùng đen ở mép
# - "auto": tự phát hiện ảnh đầu vào có pad đen hay không, có thì khôi phục, không thì bỏ qua
PAD_MODE = "auto"

# Cấu hình phát hiện pad đen ở mép ảnh.
# Một hàng/cột được xem là pad nếu phần lớn pixel gần đen.
PAD_BLACK_THRESHOLD = 8
PAD_LINE_BLACK_RATIO = 0.95
PAD_MIN_THICKNESS_PX = 3


# ============================================================
# GHI CHÚ QUAN TRỌNG
# ============================================================
# Code này KHÔNG làm sharpen, KHÔNG tăng tương phản mạnh, KHÔNG histogram equalization.
# Mục tiêu là:
# 1. Loại ảnh quá lệch so với phân bố ảnh train/reference.
# 2. Sửa bad pixel rất nhẹ nếu chỉ là pixel lẻ.
# 3. Khử nhiễu nhẹ nếu noise vượt ngưỡng đã calibrate.
# 4. Nếu ảnh đầu vào đã có pad đen thì giữ lại pad đen gốc.
# 5. Letterbox về 640x512 giống dữ liệu train/model.


# ============================================================
# IO ẢNH, HỖ TRỢ ĐƯỜNG DẪN CÓ KÝ TỰ UNICODE
# ============================================================

def read_image(image_path):
    image_path = Path(image_path)
    try:
        data = np.fromfile(str(image_path), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def write_image(image_path, img_bgr):
    image_path = Path(image_path)
    image_path.parent.mkdir(parents=True, exist_ok=True)

    ext = image_path.suffix.lower()
    if ext == "":
        ext = ".jpg"

    ok, buf = cv2.imencode(ext, img_bgr)
    if not ok:
        return False

    buf.tofile(str(image_path))
    return True


def make_output_name(image_path, input_dir):
    """
    Tránh trùng tên nếu ảnh nằm trong nhiều thư mục con.
    Ví dụ: a/img.jpg và b/img.jpg sẽ thành a__img.jpg, b__img.jpg.
    """
    image_path = Path(image_path)
    input_dir = Path(input_dir)

    try:
        rel = image_path.relative_to(input_dir)
        parts = rel.parts
        return "__".join(parts)
    except Exception:
        return image_path.name


# ============================================================
# LOAD THRESHOLD JSON
# ============================================================

def load_thresholds(threshold_json):
    with open(threshold_json, "r", encoding="utf-8") as f:
        thresholds = json.load(f)

    required_keys = [
        "blur_warn_below",
        "blur_reject_below",
        "noise_warn_above",
        "noise_reject_above",
        "denoise_apply_above",
        "contrast_warn_below",
        "contrast_reject_below",
        "dynamic_range_warn_below",
        "dynamic_range_reject_below",
        "low_saturation_warn_above",
        "low_saturation_reject_above",
        "high_saturation_warn_above",
        "high_saturation_reject_above",
        "edge_black_warn_above",
        "edge_black_reject_above",
    ]

    missing = [k for k in required_keys if k not in thresholds]
    if missing:
        raise ValueError(
            "File threshold JSON thiếu các key sau: " + ", ".join(missing)
        )

    return thresholds


# ============================================================
# TÍNH METRIC CHẤT LƯỢNG ẢNH
# ============================================================

class QualityMetrics:
    @staticmethod
    def to_gray(img_bgr):
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def compute_blur_score(img_bgr):
        """
        Độ nét tương đối bằng variance of Laplacian.
        Càng thấp càng mờ.
        Ngưỡng không phải IEC/OpenCV universal threshold,
        mà lấy từ threshold JSON đã calibrate từ tập train/reference.
        """
        gray = QualityMetrics.to_gray(img_bgr)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    @staticmethod
    def compute_noise_score(img_bgr):
        """
        Ước lượng nhiễu bằng độ lệch chuẩn của phần dư:
        gray - GaussianBlur(gray).
        Càng cao càng nhiễu/chi tiết cao bất thường.
        """
        gray = QualityMetrics.to_gray(img_bgr)
        smooth = cv2.GaussianBlur(gray, (3, 3), 0)
        residual = gray.astype(np.float32) - smooth.astype(np.float32)
        return float(np.std(residual))

    @staticmethod
    def compute_exposure_stats(img_bgr):
        gray = QualityMetrics.to_gray(img_bgr)

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
    def compute_edge_black_ratio(img_bgr, threshold=8, edge_percent=0.04):
        """
        Tỉ lệ pixel gần đen ở mép ảnh, không đo toàn ảnh.
        Metric này dùng để phát hiện ảnh có mép đen bất thường so với tập reference.
        """
        gray = QualityMetrics.to_gray(img_bgr)
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
    def compute_all(img_bgr):
        exposure = QualityMetrics.compute_exposure_stats(img_bgr)

        return {
            "blur_score": QualityMetrics.compute_blur_score(img_bgr),
            "noise_score": QualityMetrics.compute_noise_score(img_bgr),
            "edge_black_ratio": QualityMetrics.compute_edge_black_ratio(img_bgr),
            **exposure,
        }


# ============================================================
# LETTERBOX RESIZE VỀ SIZE TRAIN
# ============================================================

class ResizeUtils:
    @staticmethod
    def letterbox_to_shape(img_bgr, target_shape=(640, 512)):
        """
        Resize kiểu Fit black edges/letterbox.
        target_shape = (width, height).
        Không làm méo ảnh, chỉ thêm pad đen nếu cần.
        """
        h, w = img_bgr.shape[:2]
        target_w, target_h = target_shape

        if w == target_w and h == target_h:
            info = {
                "scale": 1.0,
                "new_width": w,
                "new_height": h,
                "pad_top": 0,
                "pad_bottom": 0,
                "pad_left": 0,
                "pad_right": 0,
            }
            return img_bgr.copy(), info

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

        info = {
            "scale": float(scale),
            "new_width": int(new_w),
            "new_height": int(new_h),
            "pad_top": int(top),
            "pad_bottom": int(bottom),
            "pad_left": int(left),
            "pad_right": int(right),
        }

        return padded, info


# ============================================================
# PHÁT HIỆN PAD ĐEN ĐÃ CÓ Ở ẢNH ĐẦU VÀO
# ============================================================

class PadDetector:
    @staticmethod
    def detect_existing_black_padding(
        img_bgr,
        black_threshold=8,
        line_black_ratio=0.95,
        min_thickness_px=3,
    ):
        """
        Tự phát hiện ảnh đầu vào đã có pad đen hay chưa.

        Cách làm:
        - Tạo mask pixel gần đen.
        - Đếm số hàng đen liên tục từ mép trên/dưới.
        - Đếm số cột đen liên tục từ mép trái/phải.
        - Nếu có một cạnh có pad đủ dày thì xem là ảnh đã có pad đen.

        Hàm này chỉ phục vụ việc khôi phục pad đen sau denoise.
        Nó khác với edge_black_ratio dùng cho QC.
        """
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
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
    def make_padding_mask(img_bgr, pad_info):
        """
        Tạo mask chỉ cho phần pad đen đã phát hiện ở mép ảnh.
        Không mask tất cả pixel đen trong ảnh.
        """
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
    def should_restore_padding(pad_mode, pad_info):
        if pad_mode == "off":
            return False
        if pad_mode == "on":
            return bool(pad_info.get("has_existing_pad", False))
        if pad_mode == "auto":
            return bool(pad_info.get("has_existing_pad", False))
        raise ValueError("PAD_MODE phải là 'off', 'on', hoặc 'auto'")


# ============================================================
# ĐÁNH GIÁ CHẤT LƯỢNG ẢNH DỰA TRÊN THRESHOLD ĐÃ CÓ
# ============================================================

def assess_quality(img_bgr, thresholds):
    metrics = QualityMetrics.compute_all(img_bgr)

    warnings = []
    critical = []

    # Ảnh mờ: blur càng thấp càng xấu
    if metrics["blur_score"] < thresholds["blur_reject_below"]:
        critical.append("reject_too_blurry")
    elif metrics["blur_score"] < thresholds["blur_warn_below"]:
        warnings.append("warn_blurry")

    # Ảnh nhiễu: noise càng cao càng xấu
    if metrics["noise_score"] > thresholds["noise_reject_above"]:
        critical.append("reject_too_noisy")
    elif metrics["noise_score"] > thresholds["noise_warn_above"]:
        warnings.append("warn_noisy")

    # Tương phản thấp
    if metrics["brightness_std"] < thresholds["contrast_reject_below"]:
        critical.append("reject_low_contrast")
    elif metrics["brightness_std"] < thresholds["contrast_warn_below"]:
        warnings.append("warn_low_contrast")

    # Dải sáng thấp
    if metrics["dynamic_range"] < thresholds["dynamic_range_reject_below"]:
        critical.append("reject_low_dynamic_range")
    elif metrics["dynamic_range"] < thresholds["dynamic_range_warn_below"]:
        warnings.append("warn_low_dynamic_range")

    # Quá nhiều pixel tối/gần đen
    if metrics["low_saturation_ratio"] > thresholds["low_saturation_reject_above"]:
        critical.append("reject_too_dark_or_black")
    elif metrics["low_saturation_ratio"] > thresholds["low_saturation_warn_above"]:
        warnings.append("warn_too_dark_or_black")

    # Quá nhiều pixel sáng bão hòa
    if metrics["high_saturation_ratio"] > thresholds["high_saturation_reject_above"]:
        critical.append("reject_too_saturated_hot")
    elif metrics["high_saturation_ratio"] > thresholds["high_saturation_warn_above"]:
        warnings.append("warn_too_saturated_hot")

    # Quá nhiều mép đen so với phân bố reference
    if metrics["edge_black_ratio"] > thresholds["edge_black_reject_above"]:
        critical.append("reject_excessive_black_edges")
    elif metrics["edge_black_ratio"] > thresholds["edge_black_warn_above"]:
        warnings.append("warn_black_edges")

    if (
        REJECT_IF_WARNING_COUNT_AT_LEAST is not None
        and len(warnings) >= REJECT_IF_WARNING_COUNT_AT_LEAST
    ):
        critical.append("reject_too_many_quality_warnings")

    if critical:
        status = "critical"
    elif warnings:
        status = "warning"
    else:
        status = "ok"

    return {
        "quality_status": status,
        "critical": critical,
        "warnings": warnings,
        "issues": critical + warnings,
        "metrics": metrics,
    }


# ============================================================
# TIỀN XỬ LÝ ẢNH NHIỆT
# ============================================================

class ThermalPreprocessor:
    @staticmethod
    def bad_pixel_replacement(img_bgr, z_thresh=6.0, max_bad_ratio=0.003):
        """
        Sửa bad pixel/hot pixel lẻ.
        Không sửa nếu vùng bất thường quá nhiều, vì có thể là hotspot thật.
        """
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

        info = {
            "bad_pixel_ratio": bad_ratio,
            "bad_pixel_repaired": False,
        }

        if bad_ratio > max_bad_ratio or not np.any(bad_mask):
            return img_bgr.copy(), info

        mask_u8 = bad_mask.astype(np.uint8) * 255

        repaired = cv2.inpaint(
            img_bgr,
            mask_u8,
            inpaintRadius=1,
            flags=cv2.INPAINT_TELEA,
        )

        info["bad_pixel_repaired"] = True
        return repaired, info

    @staticmethod
    def denoise_if_needed(img_bgr, noise_score, thresholds):
        """
        Khử nhiễu nhẹ nếu noise_score vượt ngưỡng denoise_apply_above.
        Ngưỡng này lấy từ threshold JSON đã calibrate.
        """
        apply_threshold = float(thresholds["denoise_apply_above"])

        if noise_score <= apply_threshold:
            return img_bgr, False

        denoised = cv2.bilateralFilter(
            img_bgr,
            d=7,
            sigmaColor=20,
            sigmaSpace=20,
        )

        return denoised, True

    @staticmethod
    def restore_existing_padding(processed_bgr, original_bgr, pad_info):
        """
        Chỉ khôi phục đúng các dải pad đen đã phát hiện ở mép ảnh.
        Không khôi phục toàn bộ pixel đen trong ảnh.
        """
        mask = PadDetector.make_padding_mask(original_bgr, pad_info)
        out = processed_bgr.copy()
        out[mask > 0] = original_bgr[mask > 0]
        return out

    @staticmethod
    def preprocess(image_path, thresholds):
        img = read_image(image_path)

        if img is None:
            report = {
                "quality_status": "error",
                "critical": ["image_read_failed"],
                "warnings": [],
                "issues": ["image_read_failed"],
                "metrics": {},
                "metrics_space": "letterboxed_to_target_shape_before_processing",
                "pad_detection": {},
                "processing": {
                    "bpr_applied": False,
                    "bad_pixel_ratio": None,
                    "denoise_applied": False,
                    "restore_existing_pad_applied": False,
                    "letterbox_applied": False,
                },
                "rejected": True,
            }
            return None, report

        original = img.copy()
        original_h, original_w = original.shape[:2]

        # 1. Tự phát hiện ảnh đầu vào có pad đen sẵn hay không.
        pad_info = PadDetector.detect_existing_black_padding(
            original,
            black_threshold=PAD_BLACK_THRESHOLD,
            line_black_ratio=PAD_LINE_BLACK_RATIO,
            min_thickness_px=PAD_MIN_THICKNESS_PX,
        )

        restore_existing_pad = PadDetector.should_restore_padding(PAD_MODE, pad_info)

        # 2. Tạo ảnh chuẩn hoá kích thước chỉ để tính QC metric.
        # Vì threshold được tạo từ tập train/reference, thường đã ở size train,
        # nên metric QC nên đo trên ảnh đã letterbox về TARGET_SHAPE.
        qc_img, qc_letterbox_info = ResizeUtils.letterbox_to_shape(original, TARGET_SHAPE)
        quality_before = assess_quality(qc_img, thresholds)

        if quality_before["quality_status"] == "critical":
            report = {
                "quality_status": "critical",
                "critical": quality_before["critical"],
                "warnings": quality_before["warnings"],
                "issues": quality_before["issues"],
                "metrics": quality_before["metrics"],
                "metrics_space": "letterboxed_to_target_shape_before_processing",
                "original_shape": {
                    "height": int(original_h),
                    "width": int(original_w),
                },
                "pad_detection": pad_info,
                "qc_letterbox_info": qc_letterbox_info,
                "quality_before": quality_before,
                "quality_after": None,
                "processing": {
                    "bpr_applied": False,
                    "bad_pixel_ratio": None,
                    "denoise_applied": False,
                    "restore_existing_pad_applied": False,
                    "letterbox_applied": False,
                },
                "rejected": True,
            }
            return None, report

        # 3. Sửa bad pixel/hot pixel lẻ.
        img, bpr_info = ThermalPreprocessor.bad_pixel_replacement(img)

        # 4. Khử nhiễu nhẹ nếu cần.
        # Dùng noise_score trong không gian QC đã letterbox để nhất quán threshold.
        noise_score = quality_before["metrics"]["noise_score"]
        img, denoise_applied = ThermalPreprocessor.denoise_if_needed(
            img,
            noise_score,
            thresholds,
        )

        # 5. Nếu ảnh đầu vào đã có pad đen thì khôi phục đúng dải pad đen đó.
        # Nếu ảnh đầu vào là ảnh drone gốc không pad thì bước này tự tắt khi PAD_MODE="auto".
        if restore_existing_pad:
            img = ThermalPreprocessor.restore_existing_padding(img, original, pad_info)

        # 6. Letterbox/resize về đúng size train/model.
        img_out, out_letterbox_info = ResizeUtils.letterbox_to_shape(img, TARGET_SHAPE)

        # 7. Đánh giá lại sau xử lý chỉ để report, không dùng để reject.
        quality_after = assess_quality(img_out, thresholds)

        report = {
            "quality_status": quality_after["quality_status"],
            "critical": quality_after["critical"],
            "warnings": quality_after["warnings"],
            "issues": quality_after["issues"],
            "metrics": quality_after["metrics"],
            "metrics_space": "letterboxed_to_target_shape_after_processing",
            "original_shape": {
                "height": int(original_h),
                "width": int(original_w),
            },
            "pad_detection": pad_info,
            "qc_letterbox_info": qc_letterbox_info,
            "quality_before": quality_before,
            "quality_after": quality_after,
            "processing": {
                "bpr_applied": bool(bpr_info["bad_pixel_repaired"]),
                "bad_pixel_ratio": float(bpr_info["bad_pixel_ratio"]),
                "denoise_applied": bool(denoise_applied),
                "restore_existing_pad_applied": bool(restore_existing_pad),
                "pad_mode": PAD_MODE,
                "letterbox_applied": True,
                **out_letterbox_info,
            },
            "rejected": False,
        }

        return img_out, report


# ============================================================
# CHẠY BATCH TRÊN THƯ MỤC
# ============================================================

def run_preprocess():
    input_dir = Path(INPUT_DIR)
    output_dir = Path(OUTPUT_DIR)

    preprocessed_dir = output_dir / "preprocessed"
    rejected_dir = output_dir / "rejected"
    reports_dir = output_dir / "reports"

    preprocessed_dir.mkdir(parents=True, exist_ok=True)
    rejected_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    thresholds = load_thresholds(THRESHOLD_JSON)

    image_paths = [
        p for p in input_dir.rglob("*")
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    print(f"Found {len(image_paths)} images in: {input_dir}")

    if not image_paths:
        print("Không tìm thấy ảnh nào. Kiểm tra lại INPUT_DIR.")
        return

    rows = []

    for idx, image_path in enumerate(image_paths, start=1):
        print(f"[{idx}/{len(image_paths)}] {image_path.name}")

        img_out, report = ThermalPreprocessor.preprocess(image_path, thresholds)

        out_name = make_output_name(image_path, input_dir)

        if img_out is not None:
            save_path = preprocessed_dir / out_name
            write_image(save_path, img_out)
        else:
            # Lưu/copy ảnh bị loại để kiểm tra lại.
            rejected_path = rejected_dir / out_name
            rejected_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(str(image_path), str(rejected_path))
            except Exception:
                img_original = read_image(image_path)
                if img_original is not None:
                    write_image(rejected_path, img_original)

        if SAVE_REPORT_JSON:
            report_name = str(Path(out_name).with_suffix(".json"))
            json_path = reports_dir / report_name
            json_path.parent.mkdir(parents=True, exist_ok=True)

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "image": str(image_path),
                        "threshold_json": str(THRESHOLD_JSON),
                        "target_shape_width_height": list(TARGET_SHAPE),
                        "pad_mode": PAD_MODE,
                        "report": report,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

        row = {
            "filename": image_path.name,
            "output_name": out_name,
            "path": str(image_path),
            "quality_status": report.get("quality_status"),
            "issues": ";".join(report.get("issues", [])),
            "warnings": ";".join(report.get("warnings", [])),
            "critical": ";".join(report.get("critical", [])),
            "rejected": report.get("rejected", False),
            "metrics_space": report.get("metrics_space"),
        }

        for k, v in report.get("metrics", {}).items():
            row[k] = v

        for k, v in report.get("pad_detection", {}).items():
            row[k] = v

        for k, v in report.get("processing", {}).items():
            row[k] = v

        rows.append(row)

    if SAVE_REPORT_CSV and rows:
        csv_path = reports_dir / "preprocess_quality_report.csv"
        fieldnames = sorted(set().union(*(row.keys() for row in rows)))

        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    print("\nDone.")
    print(f"Ảnh đạt nằm ở: {preprocessed_dir}")
    print(f"Ảnh bị loại nằm ở: {rejected_dir}")
    print(f"Report nằm ở: {reports_dir}")


if __name__ == "__main__":
    run_preprocess()
