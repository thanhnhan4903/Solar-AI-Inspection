import cv2
import numpy as np
from pathlib import Path


# =========================
# CONFIG
# =========================

INPUT_DIR = Path(r"D:\Image\archive\dataset_1\images")
OUTPUT_DIR = Path(r"D:\Image\archive\dataset_1\images_color_8bit")

# Chọn colormap:
# cv2.COLORMAP_JET
# cv2.COLORMAP_INFERNO
# cv2.COLORMAP_MAGMA
# cv2.COLORMAP_TURBO
# cv2.COLORMAP_HOT
COLORMAP = cv2.COLORMAP_INFERNO

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def normalize_to_8bit(gray: np.ndarray) -> np.ndarray:
    """
    Đưa ảnh grayscale về uint8 [0, 255].
    Hỗ trợ ảnh uint8, uint16, float.
    """
    if gray is None:
        raise ValueError("Ảnh đầu vào rỗng.")

    if len(gray.shape) == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    if gray.dtype == np.uint8:
        return gray

    gray_float = gray.astype(np.float32)

    min_val = np.min(gray_float)
    max_val = np.max(gray_float)

    if max_val - min_val < 1e-6:
        return np.zeros_like(gray_float, dtype=np.uint8)

    gray_8bit = (gray_float - min_val) / (max_val - min_val) * 255.0
    return np.clip(gray_8bit, 0, 255).astype(np.uint8)


def convert_gray_to_color_8bit(input_path: Path, output_path: Path):
    """
    Đọc ảnh grayscale, chuẩn hóa về 8-bit,
    apply colormap và lưu ảnh màu 8-bit.
    """
    img = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)

    if img is None:
        print(f"[SKIP] Không đọc được ảnh: {input_path}")
        return

    gray_8bit = normalize_to_8bit(img)

    color_img = cv2.applyColorMap(gray_8bit, COLORMAP)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    success = cv2.imwrite(str(output_path), color_img)

    if success:
        print(f"[OK] {input_path.name} -> {output_path}")
    else:
        print(f"[ERROR] Không lưu được: {output_path}")


def main():
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Không tồn tại thư mục: {INPUT_DIR}")

    image_files = [
        p for p in INPUT_DIR.rglob("*")
        if p.suffix.lower() in VALID_EXTENSIONS
    ]

    print(f"Tìm thấy {len(image_files)} ảnh.")

    for img_path in image_files:
        relative_path = img_path.relative_to(INPUT_DIR)

        # Lưu tất cả thành .jpg
        output_path = OUTPUT_DIR / relative_path.with_suffix(".jpg")

        convert_gray_to_color_8bit(img_path, output_path)

    print("Hoàn tất chuyển đổi ảnh xám sang ảnh màu 8-bit.")


if __name__ == "__main__":
    main()