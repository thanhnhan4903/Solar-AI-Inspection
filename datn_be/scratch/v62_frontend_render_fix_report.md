# Báo cáo sửa đổi hiển thị và tỉ lệ (scale) Polygon trên Frontend v62

Tài liệu này tổng hợp toàn bộ các thay đổi sửa lỗi tỷ lệ hiển thị của đa giác (polygon) khi hiển thị nhiều ảnh có kích thước khác nhau trên bản đồ, đồng thời cố định độ dày viền vẽ panel luôn đạt đúng 3px trên cả Dashboard và Detail.

---

## 1. Các file đã sửa đổi
- **Backend:**
  - `C:\Solar_Inspection_Project\datn_be\app\main.py`
- **Frontend:**
  - `C:\Solar_Inspection_Project\datn_fe\src\pages\Unified\UnifiedDashboard.jsx`
  - `C:\Solar_Inspection_Project\datn_fe\src\pages\Panel\PanelDetail.jsx`

---

## 2. Điểm sửa đổi nét vẽ (stroke) trên Leaflet Dashboard
Trong `UnifiedDashboard.jsx`, cấu hình `pathOptions` của `<Polygon>` đã được điều chỉnh cố định độ dày viền là `3px`, loại bỏ hoàn toàn tô phủ màu bên trong (`fillOpacity: 0.0`) và thêm định kiểu góc nối:
```jsx
pathOptions={{
    color: color,
    fillColor: color,
    fillOpacity: 0.0,
    weight: 3,
    opacity: 1.0,
    lineCap: "round",
    lineJoin: "round",
}}
```

---

## 3. Điểm sửa đổi nét vẽ (stroke) trên SVG PanelDetail
Trong `PanelDetail.jsx`, tất cả các thẻ `<polygon>` dùng để vẽ panel đã được chuyển đổi cấu hình sang:
- Viền mặc định: `strokeWidth={3}`
- Viền khi hover: `strokeWidth={3.5}`

---

## 4. Bổ sung `vectorEffect="non-scaling-stroke"`
Chúng tôi đã thêm thuộc tính `vectorEffect="non-scaling-stroke"` vào cả 2 thẻ `<polygon>` hiển thị panel trong `PanelDetail.jsx` (Giao diện hiển thị chính và Modal Zoom cận cảnh).
- **Tác dụng:** Giữ nguyên độ dày viền của panel SVG là 3px bất kể mức độ thu phóng (zoom) của container hình ảnh, tránh hiện tượng viền bị vỡ hạt khi phóng to hoặc quá mảnh dẻ khi thu nhỏ.

---

## 5. Logic thứ tự ưu tiên (Priority) hiển thị Panel
Sử dụng helper functions `getPanelDrawPolygon` / `getPanelDrawPoly`:
1. **`outer_polygon`** (Từ Line-snap V61/V62, yêu cầu $\ge 3$ điểm).
2. **`polygon`** (YOLO polygon gốc, yêu cầu $\ge 3$ điểm).
3. **`bbox`** (Bounding box dạng pixel `[x1, y1, x2, y2]`, tự động sinh hình chữ nhật).

---

## 6. Logic scale polygon theo từng ảnh (Multi-layout scaling)
Để khắc phục lỗi lệch tọa độ khi nhiều ảnh có kích thước hiển thị khác nhau, chúng tôi cài đặt thuật toán scale động như sau:
- Khai báo layout của từng ảnh trong lưới map:
  ```js
  const imageLayout = {
      srcWidth: img.image_width || 640,    // Chiều rộng gốc của ảnh từ backend
      srcHeight: img.image_height || 512,  // Chiều cao gốc của ảnh từ backend
      displayWidth: imgW,                 // Chiều rộng hiển thị thực tế trên map
      displayHeight: imgH,                // Chiều cao hiển thị thực tế trên map
      offsetX: xOffset,
      offsetY: yOffset,
  };
  ```
- Thực hiện chuyển đổi từng điểm của polygon qua hàm `transformPointToDisplay`:
  $$displayX = offsetX + x \times \frac{displayWidth}{srcWidth}$$
  $$displayY = offsetY - y \times \frac{displayHeight}{srcHeight}$$
- Trả về tọa độ dạng `[displayY, displayX]` tương thích với hệ tọa độ `[lat, lng]` của Leaflet CRS.Simple.
- Cơ chế này giúp polygon tự co giãn theo tỷ lệ hiển thị bất kể ảnh hiển thị ở kích thước nào.

---

## 7. Xử lý object-fit / contain / letterbox
- Bản đồ sử dụng thành phần `ImageOverlay` của Leaflet, tự động kéo dãn hình ảnh khớp hoàn toàn vào tọa độ `bounds` (hoạt động tương tự `object-fit: fill`).
- Do đó, tỷ lệ co giãn luôn là phẳng và tuyến tính theo trục ngang/dọc, không cần bù trừ letterbox/padding (như trong trường hợp dùng `object-fit: contain`). Thuật toán scale ngang và dọc độc lập đã xử lý triệt để bài toán này.

---

## 8. Backend API `/latest-batch` trả về kích thước ảnh
- Cập nhật `/api/v1/latest-batch` trong `main.py` để tự động đọc kích thước thực tế của tệp ảnh JPEG/PNG trên đĩa bằng cách sử dụng `PIL.Image.open` (chỉ đọc header, cực kỳ nhanh):
  ```python
  precalib_path = os.path.join("data", "precalib", img.filename)
  # Đọc từ precalib hoặc kết quả hoặc fallback 640x512
  ```
- Gán hai trường `"image_width"` và `"image_height"` vào từng phần tử ảnh trong JSON response gửi về frontend.

---

## 9. Các lệnh kiểm tra đã chạy và kết quả
1. **Biên dịch thử Backend:**
   - Lệnh: `.\venv\Scripts\python.exe -m py_compile app/main.py app/services/pv_panel_snapper.py`
   - Kết quả: **Thành công** (Không lỗi cú pháp).
2. **Build Production Frontend:**
   - Lệnh: `npx vite build --minify false`
   - Kết quả: **Thành công** (Built in 3.48s).

---

## 10. Cách kiểm tra và Verify (Verification Plan)
- **DevTools Network `/latest-batch`:**
  Xem response API và xác thực mỗi đối tượng ảnh có trả về:
  - `"image_width"` và `"image_height"` tương ứng với độ phân giải thực của ảnh thermal.
  - Các panel bên trong có `geometry_source: "v61_line_snap"` và `outer_polygon` đầy đủ điểm.
- **Kiểm tra giao diện bản đồ:**
  - Bật / tắt các ảnh thermal trên Dashboard, kiểm tra xem viền xanh/đỏ vẽ panel có khớp khít 100% vào vị trí tấm pin hay không (không bị lệch sang trái/phải/trên/dưới).
  - Kiểm tra xem viền polygon panel trên map có dày đúng 3px rõ nét và trong suốt (`fillOpacity: 0.0`) không bị che khuất ảnh nhiệt bên dưới hay không.
  - Phóng to/thu nhỏ trên Dashboard và Detail, kiểm tra độ dày viền của panel trong SVG Detail luôn giữ nguyên 3px không bị mờ nhòe.

---

## 11. TODO còn lại
- Chạy lệnh Re-analyze để cập nhật các panel cũ sang định dạng v62 chính thức.
