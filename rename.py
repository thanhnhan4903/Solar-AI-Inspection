import os
import re

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]

def rename_files_sequential(input_dir, prefix="anh", suffix="_pc", padding=4):
    valid_ext = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')

    files = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_ext)]
    files.sort(key=natural_sort_key)

    # =========================
    # BƯỚC 1: đổi tên tạm
    # =========================
    temp_names = []
    for i, filename in enumerate(files):
        old_path = os.path.join(input_dir, filename)
        ext = os.path.splitext(filename)[1]

        temp_name = f"temp_{i}{ext}"
        temp_path = os.path.join(input_dir, temp_name)

        os.rename(old_path, temp_path)
        temp_names.append(temp_name)

    # =========================
    # BƯỚC 2: đổi tên chính thức
    # =========================
    for i, temp_name in enumerate(temp_names, start=1):
        old_path = os.path.join(input_dir, temp_name)
        ext = os.path.splitext(temp_name)[1]

        # padding chuẩn dataset
        new_name = f"{prefix}{i:0{padding}d}{suffix}{ext}"
        new_path = os.path.join(input_dir, new_name)

        os.rename(old_path, new_path)
        print(f"{temp_name} -> {new_name}")

    print("Đổi tên hoàn tất.")


if __name__ == "__main__":
    input_dir = r"C:\Solar_Inspection_Project\output_dir"
    rename_files_sequential(input_dir)