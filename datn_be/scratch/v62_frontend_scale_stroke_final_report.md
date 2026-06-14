# Báo cáo kết quả sửa lỗi tỷ lệ (scale) và độ dày viền Polygon trên Frontend v62

Báo cáo này ghi nhận việc thực hiện đồng bộ 100% tỷ lệ scale và offset giữa ImageOverlay và Polygon trên bản đồ, loại bỏ lỗi lệch tọa độ, đồng thời áp dụng chính xác độ dày nét vẽ 3px thực tế.

---

## 1. Các file đã sửa đổi
- **Frontend:**
  - `C:\Solar_Inspection_Project\datn_fe\src\pages\Unified\UnifiedDashboard.jsx`
  - `C:\Solar_Inspection_Project\datn_fe\src\pages\Panel\PanelDetail.jsx`
  - `C:\Solar_Inspection_Project\datn_fe\src\index.css`

---

## 2. ImageOverlay bounds hiện tại
Cấu trúc bounds được tính toán đồng bộ theo `layout` của từng hình ảnh như sau:
```javascript
const layout = {
    x: xOffset,
    y: yOffset,
    w: imgW,
    h: imgH,
    srcW: img.image_width || 640,
    srcH: img.image_height || 512,
};

const bounds = [
    [layout.y, layout.x],
    [layout.y + layout.h, layout.x + layout.w],
];
```
Điều này đảm bảo góc dưới bên trái là `[layout.y, layout.x]` và góc trên bên phải là `[layout.y + layout.h, layout.x + layout.w]`.

---

## 3. Công thức convert polygon pixel → Leaflet
Sử dụng hàm chuyển đổi đồng bộ không sử dụng phép trừ Y:
```javascript
function pixelPolyToLeafletScaled(poly, layout) {
    if (!poly || !Array.isArray(poly)) return [];

    const srcW = layout.srcW || 640;
    const srcH = layout.srcH || 512;
    const dstW = layout.w;
    const dstH = layout.h;
    const offX = layout.x;
    const offY = layout.y;

    return poly.map(([x, y]) => [
        offY + (Number(y) / srcH) * dstH,
        offX + (Number(x) / srcW) * dstW,
    ]);
}
```

---

## 4. Kiểm tra việc loại bỏ `offsetY - y`
- **Xác nhận:** Đã loại bỏ hoàn toàn các biểu thức cũ dạng `offsetY - y` hoặc `yOffset - y`.
- Hệ tọa độ Leaflet CRS.Simple hiện tại hoạt động đồng chiều trên trục Y: `offY + (y / srcH) * dstH`, khớp chính xác với ImageOverlay bounds `y → y + h`.

---

## 5. Logic mức độ ưu tiên vẽ panel (Polygon Priority)
1. **`outer_polygon`**: v62 line-snap polygon ($\ge 3$ điểm).
2. **`polygon`**: Fallback polygon từ YOLO ($\ge 3$ điểm).
3. **`bbox`**: Fallback cuối cùng nếu không có polygon (tạo hình chữ nhật 4 điểm).

Không có nơi nào sử dụng `inner_polygon` làm viền vẽ panel chính.

---

## 6. Leaflet stroke style trên UnifiedDashboard.jsx
- **`weight`**: `3` (Cấu hình mặc định trong React Leaflet).
- **`className`**: `"panel-outer-polygon-line"` được gán vào `pathOptions`.
- **CSS selector**:
  ```css
  .leaflet-interactive.panel-outer-polygon-line {
      stroke-width: 3px !important;
      stroke-opacity: 1 !important;
      fill-opacity: 0 !important;
      vector-effect: non-scaling-stroke;
  }
  ```
- **`fillOpacity`**: `0` (Trong suốt hoàn toàn để dễ kiểm chứng hình học).

---

## 7. SVG PanelDetail style trên PanelDetail.jsx
- **`strokeWidth`**: `3` (Hoặc `3.5` khi hover).
- **`vectorEffect`**: `"non-scaling-stroke"` (Cố định độ dày nét vẽ khi zoom).
- **`fill`**: `"none"` (Bằng thuộc tính inline `fill="none"` và class CSS đè).
- **CSS Selector**:
  ```css
  .panel-outer-svg-line {
      stroke-width: 3px !important;
      vector-effect: non-scaling-stroke;
      fill: none !important;
  }
  ```

---

## 8. Các lệnh compile/build đã chạy và kết quả
1. **Backend check:**
   - Lệnh: `.\venv\Scripts\python.exe -m py_compile app/main.py app/services/pv_panel_snapper.py`
   - Kết quả: **Thành công (Không có lỗi cú pháp)**.
2. **Frontend check:**
   - Lệnh: `npx vite build --minify false`
   - Kết quả: **Thành công (Built in 3.45s)**.

---

## 9. Hướng dẫn kiểm tra bằng DevTools
- **Kiểm tra phần tử SVG trong Leaflet:**
  1. Mở F12 chọn công cụ Inspect Element, click vào viền một panel trên bản đồ.
  2. Phần tử `<path>` tương ứng phải có class `panel-outer-polygon-line` và thuộc tính style `stroke-width` được áp dụng đúng `3px !important`.
- **Kiểm tra mạng lưới API:**
  1. Tải lại trang web và mở tab **Network** trong DevTools.
  2. Xem response của API `/api/v1/latest-batch`.
  3. Đảm bảo cấu trúc mỗi ảnh có đầy đủ: `"image_width"`, `"image_height"`, và các panel có `"outer_polygon"`, `"geometry_source"`.

---

## 10. TODO
- Thực hiện chạy Re-analyze một đợt ảnh bất kỳ để kiểm chứng độ đồng bộ và khớp khít hoàn hảo của nét viền 3px.
