# app/services/file_handler.py
import zipfile
import os
import shutil

class FileService:
    @staticmethod
    def save_and_extract_zip(zip_file, extract_to: str):
        """
        Lưu file ZIP tạm thời và giải nén vào thư mục chỉ định
        """
        # Tạo thư mục nếu chưa có
        os.makedirs(extract_to, exist_ok=True)
        
        zip_path = os.path.join(extract_to, "upload.zip")
        
        # Lưu file ZIP từ Frontend gửi xuống máy chủ
        with open(zip_path, "wb") as buffer:
            shutil.copyfileobj(zip_file.file, buffer)
            
        # Giải nén
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
            
        # Xóa file ZIP sau khi đã giải nén xong cho sạch máy
        os.remove(zip_path)
        
        return os.listdir(extract_to)