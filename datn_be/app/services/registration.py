# app/services/registration.py
import os
import re
import cv2

class RegistrationService:
    @staticmethod
    def natural_sort_key(s):
        """Hàm bổ trợ để sắp xếp tên file theo thứ tự số tăng dần"""
        return [int(text) if text.isdigit() else text.lower()
                for text in re.split(r'(\d+)', s)]

    @staticmethod
    def match_thermal_rgb(folder_path: str):
        """
        Khối 3: Ghép cặp dựa trên độ phân giải (Resolution-Based) của ảnh
        - DJI chụp 2 ảnh cùng lúc: sequence liền kề sequence + 1.
        - Ảnh nhiệt (Thermal) luôn có độ phân giải thấp (chiều rộng <= 1280px, vd: 640x512).
        - Ảnh quang học (RGB) luôn có độ phân giải cao (chiều rộng > 1280px, vd: 4000x3000 hoặc 8000x6000).
        """
        if not os.path.exists(folder_path):
            return []
            
        # Lấy danh sách file và sắp xếp tự nhiên
        all_files = [f for f in os.listdir(folder_path) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        all_files.sort(key=RegistrationService.natural_sort_key)
        
        pairs = []
        # Duyệt qua danh sách để tìm các cặp ảnh sequence liền kề nhau
        for i in range(len(all_files) - 1):
            file_1 = all_files[i]
            file_2 = all_files[i+1]
            
            # Trích xuất số hiệu từ tên file (ví dụ: 0953 từ DJI_0953.JPG)
            match_1 = re.search(r'DJI_(\d+)', file_1)
            match_2 = re.search(r'DJI_(\d+)', file_2)
            
            if match_1 and match_2:
                num_1 = int(match_1.group(1))
                num_2 = int(match_2.group(1))
                
                # Kiểm tra nếu 2 ảnh là số hiệu liền kề nhau (khoảng cách là 1)
                if abs(num_1 - num_2) == 1:
                    path_1 = os.path.join(folder_path, file_1)
                    path_2 = os.path.join(folder_path, file_2)
                    
                    img_1 = cv2.imread(path_1)
                    img_2 = cv2.imread(path_2)
                    
                    if img_1 is not None and img_2 is not None:
                        w1 = img_1.shape[1]
                        w2 = img_2.shape[1]
                        
                        is_thermal_1 = w1 <= 1280
                        is_thermal_2 = w2 <= 1280
                        
                        # Cặp hợp lệ: ảnh RGB (file_1, rộng > 1280) đi liền kề trước ảnh Thermal (file_2, rộng <= 1280)
                        # Tức là ảnh RGB có số thứ tự là ảnh Thermal - 1 (ví dụ: DJI_0953.JPG là RGB, DJI_0954.JPG là Thermal)
                        if not is_thermal_1 and is_thermal_2:
                            pairs.append({
                                "thermal": file_2,
                                "rgb": file_1
                            })
        return pairs