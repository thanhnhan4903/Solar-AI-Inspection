# 🌞 Solar AI Inspection Platform

Chào mừng bạn đến với **Solar AI Inspection** - Hệ thống phân tích và giám sát lỗi pin năng lượng mặt trời tự động bằng máy bay không người lái (Drone) và Trí tuệ Nhân tạo (YOLOv8).

---

## ✨ Các tính năng nổi bật
1. **Nhận diện lỗi AI (YOLOv8 Segmentation):** Tự động phát hiện, khoanh vùng chính xác (Polygon) các lỗi như Điểm nóng (Hotspot), Nứt vỡ (Crack), Bụi bẩn (Soiling) trên bề mặt tấm pin.
2. **Bản đồ Ảo (Virtual Map):** Tự động "khâu" (stitch) các bức ảnh riêng lẻ thành một không gian bản đồ 2D khổng lồ để người dùng dễ dàng thu phóng, cuộn và theo dõi toàn bộ nông trại điện.
3. **Chế độ xem Side-by-side:** Rê chuột vào một tấm pin bị lỗi trên bản đồ, hệ thống sẽ tự động hiển thị cắt cúp (crop) ảnh Nhiệt (Thermal) và ảnh Quang học (RGB) đặt cạnh nhau để đối chiếu.
4. **Cập nhật AI không gián đoạn (Hot-reload):** Hỗ trợ thay thế/tải lên file trọng số (`.pt`) của AI trực tiếp qua giao diện Web mà không cần phải tắt mở lại server.
5. **Tiền xử lý tiêu chuẩn:** Tự động khử nhiễu cảm biến nhiệt và điều chỉnh tỷ lệ ảnh chuẩn xác trước khi phân tích.

---

## 🛠 Yêu cầu hệ thống (Prerequisites)
Trước khi chạy dự án, máy tính của bạn cần được cài đặt sẵn:
- **Python 3.9+**
- **Node.js 18+**
- **MySQL Server** (XAMPP, MySQL Workbench, v.v.)

---

## 🚀 Hướng dẫn Cài đặt & Khởi chạy

### 1. Cấu hình Cơ sở dữ liệu (Database)
- Mở MySQL của bạn lên và tạo một database trống với tên là `solar_db`.
  ```sql
  CREATE DATABASE solar_db;
  ```
- Mở file `datn_be/app/core/database.py` ở dòng số 7. Sửa lại mật khẩu MySQL cho khớp với máy của bạn.
  *Ví dụ: Nếu dùng XAMPP mặc định không có mật khẩu:*
  `SQLALCHEMY_DATABASE_URL = "mysql+pymysql://root:@localhost:3306/solar_db"`

### 2. Khởi động Backend (FastAPI + AI)
Mở một cửa sổ Terminal (Cmd/Powershell) mới và gõ lần lượt:
```bash
cd datn_be

# Tạo và kích hoạt môi trường ảo (Virtual Environment)
python -m venv venv
venv\Scripts\activate

# Cài đặt các thư viện cần thiết
pip install -r requirements.txt

# Chạy Server
uvicorn app.main:app --reload
```
*Backend sẽ chạy tại địa chỉ: `http://127.0.0.1:8000`*

### 3. Khởi động Frontend (React + Vite)
Mở một cửa sổ Terminal khác và gõ:
```bash
cd datn_fe

# Cài đặt các gói giao diện
npm install

# Chạy giao diện Web
npm run dev
```
*Frontend sẽ chạy tại địa chỉ: `http://localhost:3000` (hoặc cổng mà Terminal báo)*

---

## 📖 Hướng dẫn sử dụng nhanh
1. **Đăng nhập:** Mở link Frontend, dùng tài khoản mặc định `admin123` / `admin123`.
2. **Tải dữ liệu:** Tại màn hình Dashboard, bấm nút **"Tải lên & Phân tích"**. Chọn file `.zip` chứa các cặp ảnh Drone (bao gồm cả ảnh lẻ RGB và ảnh chẵn Thermal của DJI).
3. **Khám phá:** Sau khi AI báo thành công, chuyển sang tab **"Solar Operations"** ở menu bên trái. Lúc này toàn bộ dữ liệu sẽ được hiển thị trên Bản đồ ảo.
4. **Cập nhật AI:** Nếu có file `best.pt` mới được huấn luyện thông minh hơn, bấm nút **"Thay model AI"** ngay tại trang chủ để cập nhật trực tiếp.

---
*Dự án thực tập - Phát triển bởi team Backend / AI*
