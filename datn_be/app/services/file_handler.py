# app/services/file_handler.py
import zipfile
import os
import shutil
import patoolib
from typing import List
from fastapi import UploadFile

class FileService:
    @staticmethod
    def process_uploads(files: List[UploadFile], extract_to: str):
        """
        Xử lý danh sách file upload (có thể là ảnh lẻ, zip, rar, thư mục)
        """
        os.makedirs(extract_to, exist_ok=True)
        
        saved_files_count = 0
        
        for upload_file in files:
            # Bóc tách tên file thực sự (bỏ qua cấu trúc thư mục con do webkitdirectory tạo ra)
            clean_filename = upload_file.filename.replace('\\', '/').split('/')[-1]
            filename = clean_filename.lower()
            temp_path = os.path.join(extract_to, clean_filename)
            
            # Lưu file vào ổ cứng trước
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(upload_file.file, buffer)
            
            if filename.endswith('.zip'):
                # Xử lý giải nén ZIP
                try:
                    with zipfile.ZipFile(temp_path, 'r') as zip_ref:
                        zip_ref.extractall(extract_to)
                except Exception as e:
                    print(f"Lỗi giải nén ZIP {filename}: {e}")
                finally:
                    os.remove(temp_path) # Xóa file gốc sau khi giải nén
                    
            elif filename.endswith('.rar'):
                # Xử lý giải nén RAR bằng patool
                try:
                    patoolib.extract_archive(temp_path, outdir=extract_to)
                except Exception as e:
                    print(f"Lỗi giải nén RAR {filename}: {e}")
                finally:
                    os.remove(temp_path)
            else:
                # File ảnh thường (hoặc file khác), cứ để nguyên trong thư mục
                pass
                
        # Duyệt lại thư mục để đếm file và có thể dọn dẹp thư mục con nếu bị lồng
        # (Ở đây ta tạm thời trả về danh sách các file trong extract_to)
        return os.listdir(extract_to)