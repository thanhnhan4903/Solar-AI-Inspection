# Báo cáo sửa đổi Y-flip sửa lỗi lật ngược Polygon trên Frontend v62

Báo cáo này ghi nhận việc sửa đổi công thức ánh xạ tọa độ trục Y từ pixel ảnh sang tọa độ Leaflet Simple CRS để sửa lỗi đa giác panel bị lật ngược theo chiều dọc.

---

## 1. Trước đó công thức Y là gì
Công thức Y trong phiên bản trước sử dụng phép cộng trực tiếp:
```javascript
displayY = offY + (y / srcH) * dstH
```
Biểu thức này giả định rằng trục tọa độ pixel Y và trục tọa độ Leaflet Y chạy cùng chiều (tăng dần từ trên xuống), dẫn đến đa giác panel bị vẽ lật ngược chiều dọc (upside down).

---

## 2. Sau khi sửa công thức Y là gì
Chúng tôi đã sửa đổi công thức trong hàm `pixelPolyToLeafletScaled` thành:
```javascript
const mapY = offY + dstH - (py / srcH) * dstH;
```

---

## 3. ImageOverlay bounds hiện tại
Bounds được thiết lập đồng bộ theo layout từng ảnh:
```javascript
const bounds = [
    [layout.y, layout.x],
    [layout.y + layout.h, layout.x + layout.w],
];
```
Với:
- Mép dưới của ảnh trên bản đồ nằm ở `layout.y` (tọa độ Leaflet nhỏ nhất).
- Mép trên của ảnh trên bản đồ nằm ở `layout.y + layout.h` (tọa độ Leaflet lớn nhất).

---

## 4. Vì sao phải dùng `mapY = offY + dstH - py * scaleY` (Y-flip)
Trong hệ tọa độ pixel của tệp ảnh:
- Điểm `y = 0` tương ứng với mép **trên** của ảnh.
- Điểm `y = srcH` tương ứng với mép **dưới** của ảnh.

Trong khi đó, đối với bounds hiển thị `[offY, offY + dstH]` của Leaflet CRS.Simple:
- Điểm `offY` là biên **dưới** của ảnh trên bản đồ.
- Điểm `offY + dstH` là biên **trên** của ảnh trên bản đồ.

Do đó, cần có phép biến đổi lật ngược Y (Y-flip):
- Khi `py = 0` (đỉnh ảnh), `mapY = offY + dstH - 0 = offY + dstH` (lên đỉnh bản đồ) -> **Đúng**.
- Khi `py = srcH` (đáy ảnh), `mapY = offY + dstH - dstH = offY` (xuống đáy bản đồ) -> **Đúng**.

---

## 5. Mức độ ưu tiên vẽ panel (Polygon Priority)
Thứ tự ưu tiên được giữ nguyên:
1. **`outer_polygon`**: v62 line-snap polygon ($\ge 3$ điểm).
2. **`polygon`**: Fallback polygon từ YOLO ($\ge 3$ điểm).
3. **`bbox`**: Fallback cuối cùng nếu không có polygon (tạo hình chữ nhật 4 điểm).

Không có nơi nào sử dụng `inner_polygon` làm viền vẽ panel chính.

---

## 6. Nét vẽ trên Leaflet (Stroke Style)
- **React Leaflet weight**: `3`
- **CSS global selector** (`src/index.css`):
  ```css
  .leaflet-interactive.panel-outer-polygon-line {
      stroke-width: 3px !important;
      stroke-opacity: 1 !important;
      fill-opacity: 0 !important;
      vector-effect: non-scaling-stroke;
  }
  ```
  Nhờ lớp CSS đè này, viền panel luôn dày đúng 3px rõ ràng và không bị biến dạng khi thu phóng.

---

## 7. Kết quả Build / Compile
- **Backend compile check:**
  `.\venv\Scripts\python.exe -m py_compile app/main.py app/services/pv_panel_snapper.py` -> **Thành công**.
- **Frontend Vite build:**
  `npx vite build --minify false` -> **Thành công (Built in 3.68s)**.
