# Báo cáo tích hợp đường dữ liệu v62 outer_polygon (backend → API → frontend)

Tài liệu này ghi nhận các thay đổi đã thực hiện để đưa kết quả thuật toán v62 line-snap (outer_polygon) hiển thị chuẩn xác lên Web UI frontend, tắt bỏ các fallback cũ không cần thiết và tối ưu giao diện hiển thị viền tấm pin.

---

## 1. Các file đã sửa
Chúng tôi đã sửa đổi và kiểm tra thành công 4 files:
- **Backend:**
  - `C:\Solar_Inspection_Project\datn_be\app\services\pv_panel_snapper.py`
  - `C:\Solar_Inspection_Project\datn_be\app\main.py`
- **Frontend:**
  - `C:\Solar_Inspection_Project\datn_fe\src\pages\Unified\UnifiedDashboard.jsx`
  - `C:\Solar_Inspection_Project\datn_fe\src\pages\Panel\PanelDetail.jsx`

---

## 2. Chi tiết thay đổi trong `pv_panel_snapper.py`
- Bỏ các panel bị reject (`valid=False`) từ v62 engine. Thêm filter `if p.get("valid") is False: continue` trong vòng lặp convert panel.
- Đồng bộ hóa log thông tin để biến `n_returned` khớp đúng với số panel hợp lệ (`n_valid`).
- Thêm đoạn kiểm tra phòng vệ và cảnh báo cảnh báo lệch số lượng:
  ```python
  try:
      n_valid = int(summary.get("n_valid", -1))
      if n_valid >= 0 and len(snapped) != n_valid:
          logger.warning(f"[PV_SNAPPER] returned panels mismatch: n_valid={n_valid}, n_returned={len(snapped)}")
  except Exception:
      pass
  ```

---

## 3. Chi tiết thay đổi trong `main.py` - Endpoint `/api/v1/latest-batch`
- Thêm chuẩn hóa dữ liệu panel trước khi append:
  ```python
  raw_polygon = panel_detail.get("polygon", [])
  outer_polygon = panel_detail.get("outer_polygon") or raw_polygon
  inner_polygon = panel_detail.get("inner_polygon", [])
  calc_polygon = panel_detail.get("calc_polygon") or inner_polygon

  if not outer_polygon or len(outer_polygon) < 3:
      outer_polygon = raw_polygon

  geometry_source = panel_detail.get("geometry_source", "unknown")
  snap_source = panel_detail.get("snap_source", "")
  outer_area = panel_detail.get("outer_area", 0.0)
  inner_area = panel_detail.get("inner_area", panel_detail.get("area", 0.0))
  ```
- Trả về đúng các field mới trong response JSON gửi về frontend:
  ```python
  "polygon": outer_polygon, # Đảm bảo frontend cũ đọc p.polygon vẫn vẽ chuẩn line-snap
  "outer_polygon": outer_polygon,
  "inner_polygon": inner_polygon,
  "calc_polygon": calc_polygon,
  "geometry_source": geometry_source,
  "snap_source": snap_source,
  "outer_area": outer_area,
  "inner_area": inner_area,
  ```
- Giữ nguyên toàn bộ các field metadata cũ.

---

## 4. Chi tiết thay đổi trong `main.py` - Hàm `_serialize_panels`
- Cập nhật thống nhất cách thức serialize để đảm bảo API `/api/v1/analyze-all` cũng trả về cấu trúc tương tự:
  ```python
  "polygon":      _to_list(p.get("outer_polygon") or p.get("polygon", [])),
  "outer_polygon":   _to_list(p.get("outer_polygon") or p.get("polygon", [])),
  "inner_polygon":   _to_list(p.get("inner_polygon", [])),
  "calc_polygon":    _to_list(p.get("calc_polygon") or p.get("inner_polygon", [])),
  "geometry_source": p.get("geometry_source", "yolo_fallback"),
  "snap_source":      p.get("snap_source", ""),
  "outer_area":      p.get("outer_area", p.get("area", 0.0)),
  "inner_area":      p.get("inner_area", p.get("area", 0.0)),
  ```

---

## 5. Chi tiết thay đổi trong `UnifiedDashboard.jsx` (Frontend)
- Định nghĩa helper functions sạch sẽ:
  - `getPanelDrawPolygon`: Ưu tiên `outer_polygon` -> `polygon` -> `null`.
  - `polygonFromBBox`: Chuyển đổi bounding box `[x1, y1, x2, y2]` thành mảng tọa độ 4 góc pixel.
- Sử dụng hai helper trên để tính toán tọa độ vẽ panel, cam kết thứ tự ưu tiên:
  `outer_polygon` → `polygon` → `bbox` (bbox chỉ là fallback cuối cùng).
- Không vẽ `inner_polygon` làm viền panel chính trên map.
- Cập nhật Tooltip hiển thị chính xác chuỗi debug `geometry_source`: hiển thị `✓ Line-snap V61` nếu là `v61_line_snap`, ngược lại cảnh báo nguồn gốc thực tế (ví dụ: `⚠ yolo_fallback`).

---

## 6. Chi tiết thay đổi trong `PanelDetail.jsx` (Frontend)
- Khai báo helper `getPanelDrawPoly(panel)` kế thừa từ `getPanelOuterPoly(panel)` để fallback về polygon 4 điểm vẽ từ `bbox` khi không có polygon.
- Thay thế toàn bộ các nơi vẽ SVG panel chính (`<polygon>` & `<rect>` fallback cũ) bằng một thẻ `<polygon>` duy nhất sử dụng `getPanelDrawPoly(p)`.
- Điều này áp dụng đồng bộ cho cả giao diện vẽ ảnh chính và giao diện vẽ Zoom cận cảnh (Modal Zoom).
- Cập nhật hiển thị chuỗi debug `geometry_source` tương tự trên Dashboard chính.

---

## 7. Logic thứ tự ưu tiên (Priority) của Polygon hiện tại
Thứ tự bắt buộc và duy nhất trên cả Dashboard và Detail:
1. **`outer_polygon`** (Nếu có và $\ge 3$ điểm)
2. **`polygon`** (Fallback nếu có và $\ge 3$ điểm)
3. **`bbox`** (Fallback cuối cùng, vẽ thành hình chữ nhật từ tọa độ hộp)

---

## 8. Style của Polygon hiện tại trên Web
Để người dùng nhìn rõ viền của panel sau khi line-snap:
- **`fillOpacity`**: 
  - Tấm pin lành lặn (healthy): `0.04` (gần như trong suốt)
  - Tấm pin có lỗi (faulty): `0.10` (độ phủ nhẹ để vẫn nhận biết được vùng có lỗi)
- **`weight`** (Độ dày viền): `3` pixel cố định.
- **`opacity`** (Độ mờ viền): `1.0`.

---

## 9. Hướng dẫn kiểm tra bằng DevTools Network
1. Mở trình duyệt Web, truy cập Dashboard (`localhost:5173`).
2. Nhấn `F12` mở DevTools, chuyển sang tab **Network**.
3. Thực hiện phân tích hoặc tải lại trang để API `/api/v1/latest-batch` được gọi.
4. Xem JSON response tại: `data[0].panels[0]`.
5. Đảm bảo:
   - Cột `geometry_source` chứa `"v61_line_snap"`.
   - `outer_polygon` là một mảng tọa độ có $\ge 4$ điểm.
   - `polygon` có giá trị trùng khớp hoàn toàn với `outer_polygon`.

---

## 10. Các lệnh compile/build đã chạy và kết quả
- **Backend (Python Compile Test):**
  Lệnh chạy: `.\venv\Scripts\python.exe -m py_compile app/main.py app/services/pv_panel_snapper.py`
  Kết quả: **Thành công (Không báo lỗi)**.
- **Frontend (Vite Production Build):**
  Lệnh chạy: `npx vite build --minify false`
  Kết quả: **Thành công (Built in 4.08s)**, sinh thư mục `dist` chứa bundle sạch sẽ.

---

## 11. Các ảnh fallback YOLO
Vì dữ liệu cũ trong DB vẫn lưu cấu trúc và tọa độ cũ, bạn cần kích hoạt:
1. Gọi API `POST /api/v1/reanalyze` để dọn dẹp phân tích cũ.
2. Gọi API `POST /api/v1/analyze-all` để hệ thống chạy lại toàn bộ các ảnh bằng thuật toán v62 mới.
Sau khi phân tích lại, hãy theo dõi log console của uvicorn để biết có ảnh nào phải fallback YOLO hay không.

---

## 12. TODO / Khuyến nghị tiếp theo
- Thực hiện Hard Refresh trình duyệt (`Ctrl + F5`) để xóa cache React/Leaflet cũ.
- Thực hiện chạy Re-analyze một lô ảnh mẫu đầy đủ để kiểm chứng độ mượt của các đường viền 3px.
