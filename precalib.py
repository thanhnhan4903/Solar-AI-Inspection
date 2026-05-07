import cv2
import numpy as np
import os
import re

# =========================
# NUC
# =========================
def apply_software_nuc(raw_image, flat_frame):
    raw_float = raw_image.astype(np.float32)
    flat_float = flat_frame.astype(np.float32)
    
    mean_F = np.mean(flat_float)
    normalized_F = flat_float / mean_F
    normalized_F[normalized_F == 0] = 1e-5
    
    corrected_float = raw_float / normalized_F
    return np.clip(corrected_float, 0, 255).astype(np.uint8)


# =========================
# Natural sort (quan trọng)
# =========================
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]


# =========================
# Xử lý 1 ảnh
# =========================
def process_universal_thermal_image(image_path, flat_frame, target_shape=(640, 512)):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"[Lỗi] Không thể đọc: {image_path}")
        return None

    # --- NUC ---
    if flat_frame is not None:
        img = apply_software_nuc(img, flat_frame)

    # --- Bilateral ---
    img_denoised = cv2.bilateralFilter(img, d=5, sigmaColor=15, sigmaSpace=15)

    # --- Letterbox ---
    h, w = img_denoised.shape[:2]
    target_w, target_h = target_shape

    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(img_denoised, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    top = (target_h - new_h) // 2
    bottom = target_h - new_h - top
    left = (target_w - new_w) // 2
    right = target_w - new_w - left

    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )

    # --- Normalize ---
    normalized = cv2.normalize(
        padded, None, 0, 255,
        cv2.NORM_MINMAX, dtype=cv2.CV_8UC3
    )

    return normalized


# =========================
# Xử lý cả thư mục
# =========================
def process_directory(input_dir, output_dir, flat_frame_path=None):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    valid_ext = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')

    # Load flat-frame 1 lần
    flat_frame = None
    if flat_frame_path:
        flat_frame = cv2.imread(flat_frame_path, cv2.IMREAD_COLOR)
        if flat_frame is None:
            print("[Cảnh báo] Flat-frame lỗi → bỏ qua NUC")
            flat_frame = None
        else:
            print("✅ Đã load flat-frame")

    files = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_ext)]
    files.sort(key=natural_sort_key)

    for filename in files:
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        print(f"Đang xử lý: {filename}")

        result = process_universal_thermal_image(input_path, flat_frame)
        if result is not None:
            cv2.imwrite(output_path, result)

    print("✅ Hoàn tất toàn bộ pipeline")


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    input_dir = r"C:\Solar_Inspection_Project\input_dir"
    output_dir = r"C:\Solar_Inspection_Project\output_dir"

    # Nếu có flat-frame thì điền path, không thì để None
    flat_frame_path = None
    # ví dụ:
    # flat_frame_path = r"C:\Solar_Inspection_Project\flat\flat.jpg"

    process_directory(input_dir, output_dir, flat_frame_path)