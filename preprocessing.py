import cv2
import numpy as np
import os
from tqdm import tqdm

class SolarThermalFinalPipeline:
    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def process_all(self):
        image_files = [f for f in os.listdir(self.input_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        
        for filename in tqdm(image_files):
            img_path = os.path.join(self.input_dir, filename)
            img = cv2.imread(img_path)
            if img is None: continue

            # --- BƯỚC 1: Chuẩn hóa dải sáng (Global Normalization) ---
            # Thay cho NUC: Đưa ảnh về dải 0-255 đồng nhất để AI dễ học 
            # nhưng không làm thay đổi tỷ lệ sáng/tối giữa các pixel.
            step1 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

            # --- BƯỚC 2: Khử nhiễu Pixel (Bad Pixel Replacement - BPR) ---
            # Thay thế các điểm nhiễu hạt bằng lọc trung vị 3x3.
            # Bảo toàn các đường biên cạnh của tấm pin.
            step2 = cv2.medianBlur(step1, 3)

            # --- BƯỚC 3: Khử tối góc (Flat-Field Correction - FFC) ---
            # Cực kỳ quan trọng: Đảm bảo độ sáng pixel phản ánh đúng nhiệt độ 
            # bất kể vị trí của tấm pin nằm ở giữa hay ở rìa ảnh.
            rows, cols = step2.shape[:2]
            mask = cv2.getGaussianKernel(rows, rows/2) * cv2.getGaussianKernel(cols, cols/2).T
            mask = mask / mask.max()
            step3 = (step2.astype(np.float32))
            for i in range(3): step3[:,:,i] /= mask
            step3 = np.clip(step3, 0, 255).astype(np.uint8)

            # --- BƯỚC 4: Kéo giãn tuyến tính (Linear Stretching) ---
            # Thay cho CLAHE: Tăng độ rõ nét nhưng bảo toàn quan hệ chênh lệch pixel (Delta T).
            # Không làm biến dạng biểu đồ nhiệt như các phương pháp cân bằng sáng cục bộ.
            min_val, max_val, _, _ = cv2.minMaxLoc(cv2.cvtColor(step3, cv2.COLOR_BGR2GRAY))
            alpha = 255.0 / (max_val - min_val) if max_val > min_val else 1
            beta = -min_val * alpha
            step4 = cv2.convertScaleAbs(step3, alpha=alpha, beta=beta)

            # --- BƯỚC 5: Lọc nhiễu mịn (Temporal/Gaussian Filtering) ---
            # Làm mịn nhẹ cuối cùng để mô hình tập trung vào đặc trưng vật thể (Feature extraction).
            final_img = cv2.GaussianBlur(step4, (3, 3), 0)

            cv2.imwrite(os.path.join(self.output_dir, filename), final_img)

# Triển khai cho bộ dữ liệu DATN
dataset_path = r'E:\DATN\Image\dataset_from_rbf'
for split in ['train', 'valid', 'test']:
    input_f = os.path.join(dataset_path, split, 'images')
    output_f = os.path.join(dataset_path, split, 'images_processed')
    if os.path.exists(input_f):
        SolarThermalFinalPipeline(input_f, output_f).process_all()