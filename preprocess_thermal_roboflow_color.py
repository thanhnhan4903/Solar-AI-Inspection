from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml


# ============================================================
# CẤU HÌNH ĐƯỜNG DẪN - ĐÃ THAY SẴN CHO MÁY CỦA BẠN
# ============================================================

INPUT_ROOT = Path(r"E:\DATN\dataset")
OUTPUT_ROOT = Path(r"E:\DATN\dataset_preprocessed_no_clahe")

WIDTH = 640
HEIGHT = 512

OUTPUT_EXT = ".jpg"

# Preprocessing options
ENABLE_BPR = True
ENABLE_BILATERAL = True
RESTORE_PADDING = True

# BPR parameters
BPR_Z_THRESH = 6.0
BPR_MAX_BAD_RATIO = 0.003
INPAINT_RADIUS = 1

# Bilateral parameters theo tài liệu
BILATERAL_D = 9
BILATERAL_SIGMA_COLOR = 75
BILATERAL_SIGMA_SPACE = 75

# Black border threshold cho vùng padding đen của Roboflow Fit
BLACK_THRESHOLD = 8

# Bật True nếu muốn lưu ảnh debug từng bước
SAVE_DEBUG = True

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# ============================================================
# 1. Đọc ảnh
# ============================================================

def read_color_image(path: Path) -> np.ndarray:
    """
    Đọc ảnh thermal pseudo-color từ Roboflow.

    Ảnh của bạn là ảnh màu thermal kiểu tím/vàng/đỏ,
    nên giữ ảnh màu thay vì chuyển grayscale.
    """
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)

    if img is None:
        raise RuntimeError(f"Không đọc được ảnh: {path}")

    return img


# ============================================================
# 2. Xử lý viền đen do Roboflow Resize Fit
# ============================================================

def get_black_border_mask(
    img_bgr: np.ndarray,
    threshold: int = 8,
) -> np.ndarray:
    """
    Tạo mask vùng padding đen.

    Roboflow ghi Resize: Fit (black edges) in 640x512,
    nên vùng viền đen cần được giữ nguyên để tránh tạo artifact.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mask = gray <= threshold
    return mask.astype(np.uint8) * 255


def restore_black_border(
    processed_bgr: np.ndarray,
    original_bgr: np.ndarray,
    threshold: int = 8,
) -> np.ndarray:
    """
    Khôi phục vùng padding đen sau BPR/Bilateral.
    """
    mask = get_black_border_mask(original_bgr, threshold=threshold)

    out = processed_bgr.copy()
    out[mask > 0] = original_bgr[mask > 0]

    return out


# ============================================================
# 3. BPR - Bad Pixel Replacement
# ============================================================

def get_l_channel(img_bgr: np.ndarray) -> np.ndarray:
    """
    Lấy kênh L trong không gian LAB để phân tích độ sáng.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l_channel, _, _ = cv2.split(lab)
    return l_channel


def bad_pixel_replacement_color(
    img_bgr: np.ndarray,
    z_thresh: float = 6.0,
    max_bad_ratio: float = 0.003,
    inpaint_radius: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    BPR cho ảnh thermal pseudo-color.

    Vì dataset không có video/chuỗi frame, không dùng temporal median.
    Thay vào đó dùng local median 3x3 để tìm pixel bất thường đơn lẻ.

    Nếu số pixel bị nghi là lỗi quá nhiều, bỏ qua để tránh sửa nhầm hotspot thật.
    """
    l_channel = get_l_channel(img_bgr).astype(np.float32)

    local_median = cv2.medianBlur(l_channel.astype(np.uint8), 3).astype(np.float32)
    residual = l_channel - local_median

    med = np.median(residual)
    mad = np.median(np.abs(residual - med))

    sigma = 1.4826 * mad
    sigma = max(float(sigma), 1e-6)

    bad_mask = np.abs(residual - med) > z_thresh * sigma
    bad_ratio = float(np.mean(bad_mask))

    if bad_ratio > max_bad_ratio:
        empty_mask = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
        return img_bgr.copy(), empty_mask

    if not np.any(bad_mask):
        empty_mask = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
        return img_bgr.copy(), empty_mask

    mask_u8 = bad_mask.astype(np.uint8) * 255

    repaired = cv2.inpaint(
        img_bgr,
        mask_u8,
        inpaintRadius=inpaint_radius,
        flags=cv2.INPAINT_TELEA,
    )

    return repaired, mask_u8


# ============================================================
# 4. Bilateral filter
# ============================================================

def apply_bilateral_color(
    img_bgr: np.ndarray,
    d: int = 9,
    sigma_color: float = 75,
    sigma_space: float = 75,
) -> np.ndarray:
    """
    Bilateral filter:
    - Giảm nhiễu nhẹ.
    - Giữ biên hotspot, biên cell, biên module.

    Thông số theo tài liệu:
    d=9, sigmaColor=75, sigmaSpace=75.
    """
    return cv2.bilateralFilter(
        img_bgr,
        d=d,
        sigmaColor=sigma_color,
        sigmaSpace=sigma_space,
    )


# ============================================================
# 5. Resize dự phòng
# ============================================================

def resize_if_needed(
    img_bgr: np.ndarray,
    width: int = 640,
    height: int = 512,
) -> np.ndarray:
    """
    Ảnh Roboflow của bạn đã 640x512 rồi.
    Hàm này chỉ resize nếu gặp ảnh khác kích thước.
    """
    h, w = img_bgr.shape[:2]

    if w == width and h == height:
        return img_bgr

    return cv2.resize(
        img_bgr,
        (width, height),
        interpolation=cv2.INTER_AREA,
    )


# ============================================================
# 6. Xử lý một ảnh
# ============================================================

def preprocess_one_image(
    src_path: Path,
    dst_path: Path,
    debug_dir: Path | None = None,
) -> None:
    original_img = read_color_image(src_path)
    img = original_img.copy()

    if SAVE_DEBUG and debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(debug_dir / f"{src_path.stem}_00_original.png"), img)

    # Step 1: BPR
    if ENABLE_BPR:
        img, bad_mask = bad_pixel_replacement_color(
            img,
            z_thresh=BPR_Z_THRESH,
            max_bad_ratio=BPR_MAX_BAD_RATIO,
            inpaint_radius=INPAINT_RADIUS,
        )

        if SAVE_DEBUG and debug_dir is not None:
            cv2.imwrite(str(debug_dir / f"{src_path.stem}_01_bpr.png"), img)
            cv2.imwrite(str(debug_dir / f"{src_path.stem}_01_bad_mask.png"), bad_mask)

    # Step 2: Bilateral filter
    if ENABLE_BILATERAL:
        img = apply_bilateral_color(
            img,
            d=BILATERAL_D,
            sigma_color=BILATERAL_SIGMA_COLOR,
            sigma_space=BILATERAL_SIGMA_SPACE,
        )

        if SAVE_DEBUG and debug_dir is not None:
            cv2.imwrite(str(debug_dir / f"{src_path.stem}_02_bilateral.png"), img)

    # Không dùng CLAHE theo yêu cầu của anh hướng dẫn

    # Step 3: Restore black padding
    if RESTORE_PADDING:
        img = restore_black_border(
            processed_bgr=img,
            original_bgr=original_img,
            threshold=BLACK_THRESHOLD,
        )

        if SAVE_DEBUG and debug_dir is not None:
            cv2.imwrite(str(debug_dir / f"{src_path.stem}_03_restore_black_border.png"), img)

    # Step 4: Resize dự phòng
    img = resize_if_needed(
        img,
        width=WIDTH,
        height=HEIGHT,
    )

    if SAVE_DEBUG and debug_dir is not None:
        cv2.imwrite(str(debug_dir / f"{src_path.stem}_04_final.png"), img)

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst_path), img)


# ============================================================
# 7. Xử lý dataset Roboflow
# ============================================================

def find_images(input_root: Path) -> list[Path]:
    image_paths: list[Path] = []

    for split in ["train", "valid", "test"]:
        image_dir = input_root / split / "images"

        if not image_dir.exists():
            print(f"[WARN] Không thấy thư mục: {image_dir}")
            continue

        for path in image_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                image_paths.append(path)

    return sorted(image_paths)


def copy_label_for_image(
    src_image_path: Path,
    input_root: Path,
    output_root: Path,
) -> None:
    """
    Copy label YOLO tương ứng.

    Vì code không crop, không letterbox lại, không thay đổi tọa độ,
    label YOLO normalized được copy nguyên.
    """
    rel = src_image_path.relative_to(input_root)
    parts = list(rel.parts)

    try:
        idx = parts.index("images")
    except ValueError:
        return

    parts[idx] = "labels"

    src_label = input_root / Path(*parts).with_suffix(".txt")
    dst_label = output_root / Path(*parts).with_suffix(".txt")

    if src_label.exists():
        dst_label.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_label, dst_label)
    else:
        print(f"[WARN] Không thấy label cho ảnh: {src_image_path}")


def fix_data_yaml(
    input_yaml: Path,
    output_yaml: Path,
    output_root: Path,
) -> None:
    """
    Tạo data.yaml mới trỏ tới dataset đã preprocess.
    Giữ nguyên names/nc từ data.yaml gốc.
    """
    if not input_yaml.exists():
        print(f"[WARN] Không tìm thấy data.yaml: {input_yaml}")
        return

    with open(input_yaml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        data = {}

    data["path"] = str(output_root).replace("\\", "/")
    data["train"] = "train/images"
    data["val"] = "valid/images"
    data["test"] = "test/images"

    output_yaml.parent.mkdir(parents=True, exist_ok=True)

    with open(output_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            data,
            f,
            sort_keys=False,
            allow_unicode=True,
        )


def process_dataset() -> None:
    if not INPUT_ROOT.exists():
        raise RuntimeError(f"Không tồn tại thư mục input: {INPUT_ROOT}")

    image_paths = find_images(INPUT_ROOT)

    if not image_paths:
        raise RuntimeError(f"Không tìm thấy ảnh trong dataset: {INPUT_ROOT}")

    print("=" * 80)
    print("THERMAL ROBOFLOW PREPROCESSING - NO CLAHE")
    print("=" * 80)
    print(f"Input dataset : {INPUT_ROOT}")
    print(f"Output dataset: {OUTPUT_ROOT}")
    print(f"Images found  : {len(image_paths)}")
    print(f"Output size   : {WIDTH}x{HEIGHT}")
    print(f"BPR           : {ENABLE_BPR}")
    print(f"Bilateral     : {ENABLE_BILATERAL}")
    print(f"Restore border: {RESTORE_PADDING}")
    print(f"CLAHE         : False")
    print(f"Save debug    : {SAVE_DEBUG}")
    print("=" * 80)

    for idx, src_path in enumerate(image_paths, start=1):
        rel = src_path.relative_to(INPUT_ROOT)
        dst_path = (OUTPUT_ROOT / rel).with_suffix(OUTPUT_EXT)

        debug_dir = None
        if SAVE_DEBUG:
            debug_dir = OUTPUT_ROOT / "debug" / rel.parent / src_path.stem

        print(f"[{idx}/{len(image_paths)}] {src_path.name}")

        preprocess_one_image(
            src_path=src_path,
            dst_path=dst_path,
            debug_dir=debug_dir,
        )

        copy_label_for_image(
            src_image_path=src_path,
            input_root=INPUT_ROOT,
            output_root=OUTPUT_ROOT,
        )

    fix_data_yaml(
        input_yaml=INPUT_ROOT / "data.yaml",
        output_yaml=OUTPUT_ROOT / "data.yaml",
        output_root=OUTPUT_ROOT,
    )

    print("\n[DONE] Preprocessing hoàn tất.")
    print(f"Dataset mới nằm ở: {OUTPUT_ROOT}")
    print(f"File data.yaml mới: {OUTPUT_ROOT / 'data.yaml'}")


if __name__ == "__main__":
    process_dataset()