# Báo cáo Sửa lỗi V62 Backend Result Collector (Route Local/Closeup V42)

## 1. Nguyên nhân lỗi
Trong kiến trúc động của V62 engine (`pv_fullsnap_universal_v62_backend_ready.py`), có hai route xử lý chính:
- **Core Route:** Chạy hàm `run_current_image()` của V34/V42 và đã được chèn hook lưu trữ dữ liệu vào global collector `_BACKEND_RESULT_PANELS`.
- **Local/Closeup Route:** Chạy hàm `_v42_build_closeup_polygon_prior_grid_outputs()` thay thế cho `run_current_image()`. Tuy nhiên, trong hàm này hoàn toàn thiếu đi phần hook lưu trữ panel vào `_BACKEND_RESULT_PANELS`. Do đó, mặc dù ảnh debug vẫn được vẽ thành công 8 panel (`drawn=8`), nhưng mảng kết quả trả về backend rỗng (`n_total=0`), làm backend tự động kích hoạt chế độ dự phòng YOLO (`yolo_fallback`, trả về 12 panel thô không tối ưu).

---

## 2. File đã sửa đổi
- [pv_fullsnap_universal_v62_backend_ready.py](file:///C:/Solar_Inspection_Project/datn_be/scratch/pv_fullsnap_universal_v62_backend_ready.py)

---

## 3. Hook Collector đã được đặt ở đâu
1. **Hàm helper toàn cục `_backend_append_panel`:**
   Đã được định nghĩa ở mức module để chuẩn hóa đầu vào đa giác, tính toán diện tích bằng công thức Shoelace qua OpenCV, thực hiện inset quad margin 3px (nếu chưa được inset), và đưa kết quả vào `_BACKEND_RESULT_PANELS`.
2. **Hook trong Route Local (`_v42_build_closeup_polygon_prior_grid_outputs`):**
   Được chèn vào trong vòng lặp vẽ đa giác kết quả (`Pass 3: Draw remaining panels`):
   ```python
   # Hook panel vào backend result collector
   _backend_append_panel(
       outer_polygon=poly,
       inner_polygon=inner if valid else poly,
       source="v42_local_snap",
       engine="large_local_v42_full",
       valid=True,
       filter_reason="",
   )
   ```
3. **Refactor hook trong `run_current_image` (V34 & V42):**
   Tất cả các đoạn code hook inline gom panel trước đây trong V34 và V42 đã được cấu trúc lại để gọi chung hàm helper `_backend_append_panel()`.

---

## 4. Cách thức reset/collect của `process_image_for_backend()`
- Trước khi chạy bất cứ engine nào (V34 hay V42), `_BACKEND_RESULT_PANELS` được xóa sạch:
  ```python
  _BACKEND_RESULT_PANELS = []
  ```
- Sau khi engine chạy xong, mảng kết quả được sao chép và đóng gói trả về API:
  ```python
  panels = list(_BACKEND_RESULT_PANELS)
  ```

---

## 5. Kết quả Log trước và sau cho ảnh DJI_0959

### Trước khi sửa:
```text
[V62_ROUTE] stem=DJI_0959 route=local rep_area_frac=0.06759
[V42_ROUTE] stem=DJI_0959 mode=local_polygon_prior_conservative_grid
[DONE] stem=DJI_0959 drawn=8 outputs=debug_DJI_0959_line_snap.JPG,debug_DJI_0959_calc_inner_polygon.JPG
[V62_DONE] stem=DJI_0959 engine=large_local_v42_full rc=0 n_total=0 n_valid=0 n_rejected=0
[PV_SNAPPER] v62 returned 0 panels for stem=DJI_0959.
→ V62 snap returned 0 panels, falling back to 12 YOLO panels.
```

### Sau khi sửa:
```text
INFO:solar_ai:[V62_ROUTE] stem=DJI_0959 route=local rep_area_frac=0.06759
INFO:solar_ai:[V62_DONE] stem=DJI_0959 engine=large_local_v42_full rc=0 n_total=8 n_valid=8 n_rejected=0
INFO:solar_ai:[PV_SNAPPER] stem=DJI_0959 route=local engine=large_local_v42_full n_total=8 n_valid=8 n_returned=8
```

---

## 6. Các lệnh đã chạy để kiểm tra
1. Kiểm tra biên dịch cú pháp Python:
   ```powershell
   python -m py_compile scratch\pv_fullsnap_universal_v62_backend_ready.py
   python -m py_compile app\services\pv_panel_snapper.py
   python -m py_compile app\main.py
   ```
2. Chạy thử nghiệm trực tiếp trên ảnh DJI_0959:
   ```powershell
   $env:PYTHONPATH="."; python scratch/test_dji_0959.py
   $env:PYTHONPATH="."; python scratch/test_snapper.py
   ```

---

## 7. Cách kiểm tra trên Web/DevTools
1. Gửi request phân tích lại toàn bộ: `POST /api/v1/reanalyze` để xóa dữ liệu cũ.
2. Gửi request phân tích: `POST /api/v1/analyze-all`.
3. Kiểm tra API lấy dữ liệu mới nhất: `GET /api/v1/latest-batch`.
4. Trong response JSON, tìm ảnh `DJI_0959`:
   - `geometry_source` phải là `"v61_line_snap"`.
   - `panels` phải có độ dài đúng bằng **8**.
   - `inner_polygon` và `outer_polygon` của mỗi panel phải chứa danh sách 4 điểm tọa độ thực tế từ line-snap thay vì mảng rỗng hay bbox thô.

---

## 8. TODO (Nếu còn)
- Hiện tại tất cả các tiêu chí hoàn thành đã được đáp ứng 100%. Không có vấn đề tồn đọng.
