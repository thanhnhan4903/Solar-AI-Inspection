# app/services/registration.py
import os
import re

class RegistrationService:
    @staticmethod
    def natural_sort_key(s):
        """Hàm bổ trợ để sắp xếp tên file theo thứ tự số tăng dần"""
        return [int(text) if text.isdigit() else text.lower()
                for text in re.split(r'(\d+)', s)]

    @staticmethod
    def match_thermal_rgb(folder_path: str):
        """
        Khối 3: Ghép cặp theo quy tắc DJI (Lẻ = RGB, Chẵn = Nhiệt)
        """
        # Lấy danh sách file và sắp xếp tự nhiên
        all_files = [f for f in os.listdir(folder_path) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        all_files.sort(key=RegistrationService.natural_sort_key)
        
        pairs = []
        # Duyệt qua danh sách để tìm các cặp (Lẻ, Chẵn) đứng cạnh nhau
        for i in range(len(all_files) - 1):
            file_1 = all_files[i]
            file_2 = all_files[i+1]
            
            # Trích xuất số hiệu từ tên file (ví dụ: 0619 từ DJI_0619.JPG)
            match_1 = re.search(r'DJI_(\d+)', file_1)
            match_2 = re.search(r'DJI_(\d+)', file_2)
            
            if match_1 and match_2:
                num_1 = int(match_1.group(1))
                num_2 = int(match_2.group(1))
                
                # Kiểm tra nếu num_1 lẻ và num_2 = num_1 + 1 (cặp chẵn lẻ liên tiếp)
                if num_1 % 2 != 0 and num_2 == num_1 + 1:
                    pairs.append({
                        "rgb": file_1,      # Số lẻ là RGB
                        "thermal": file_2    # Số chẵn là Nhiệt
                    })
        return pairs