# Báo cáo Tính năng Duyệt lỗi thủ công (Manual Defect Review)

Tài liệu này ghi nhận chi tiết thiết kế kỹ thuật, danh sách các file đã sửa đổi, các API mới, quy tắc tính toán báo cáo, hướng dẫn xác thực thủ công và các bước kiểm tra cho tính năng **Duyệt lỗi phát hiện** (Manual Defect Review).

---

## 1. Danh sách các file đã sửa đổi

### Backend (`datn_be`)
* **[app/main.py](file:///C:/Solar_Inspection_Project/datn_be/app/main.py)**:
  * Cập nhật endpoint `/api/v1/latest-batch` để parse các thông tin review từ trường JSON `defect_type` trong DB và trả về cho frontend. Nếu trạng thái review là `false_positive` (Không phải lỗi), trạng thái hiển thị của panel sẽ tự động chuyển thành `"healthy"`.
  * Thêm endpoint `GET /api/v1/review/items` lấy danh sách tấm pin lỗi cần duyệt (được sắp xếp theo thứ tự hình ảnh tăng dần, sau đó theo `local_id` tăng dần).
  * Thêm endpoint `POST /api/v1/review/items/{review_item_id}` để lưu trạng thái duyệt và ghi chú của người dùng vào trường `defect_type` JSON của dòng `AiResult` tương ứng, đồng thời kích hoạt tiến trình tạo lại tệp báo cáo PDF.
  * Thêm endpoint `GET /api/v1/review/summary` trả về thống kê số lượng panel theo 4 trạng thái duyệt.
  * Import `HTTPException` từ `fastapi` để tránh lỗi biên dịch/runtime khi xử lý lỗi trong các API review mới.
* **[app/services/report_generator.py](file:///C:/Solar_Inspection_Project/datn_be/app/services/report_generator.py)**:
  * Cập nhật logic trong `generate_inspection_report` để bỏ qua toàn bộ panel lỗi có `include_in_report = False` (tức là trạng thái duyệt `false_positive`) khi tính toán tổng quan lỗi, công suất thất thoát hao hụt kW, phân phối loại lỗi và in danh sách chi tiết lỗi chính.
  * Hiển thị trạng thái review bằng nhãn tiếng Việt tương ứng (`Đã xác nhận`, `Cần xem xét`, `AI phát hiện - chưa duyệt`) trong mục chi tiết của từng panel lỗi được in trong PDF.

### Frontend (`datn_fe`)
* **[src/components/DefectReviewModal.jsx](file:///C:/Solar_Inspection_Project/datn_fe/src/components/DefectReviewModal.jsx)**:
  * Component modal duyệt lỗi hoàn chỉnh. Hỗ trợ hiển thị ảnh nhiệt crop gán lỗi, thông tin độ tin cậy AI, công suất hao hụt, form chọn 3 trạng thái kèm text ghi chú, các nút điều hướng chuyển tiếp/quay lại và lưu kết quả lên backend.
* **[src/pages/Unified/UnifiedDashboard.jsx](file:///C:/Solar_Inspection_Project/datn_fe/src/pages/Unified/UnifiedDashboard.jsx)**:
  * Tích hợp nút **"Duyệt lỗi phát hiện"** ở góc trên bên phải khi batchId hoạt động để mở modal duyệt lỗi.
  * Hiển thị trạng thái review (`review_label`) trực tiếp trên Leaflet tooltips của từng panel khi di chuột qua.
  * Liên kết `onRefresh` để tự động cập nhật bản đồ và các thống kê sau khi lưu đánh giá lỗi.
* **[src/pages/Panel/PanelDetail.jsx](file:///C:/Solar_Inspection_Project/datn_fe/src/pages/Panel/PanelDetail.jsx)**:
  * Hiển thị nhãn trạng thái review cùng màu sắc tương ứng (`confirmed_defect` -> Đỏ nhạt/Đỏ, `needs_review` -> Vàng nhạt/Cam, `false_positive` -> Xám nhạt/Xám) và ghi chú đi kèm trực tiếp bên dưới toạ độ hàng/cột của tấm pin trong thanh thông tin chi tiết.
* **[App.jsx](file:///C:/Solar_Inspection_Project/datn_fe/App.jsx)**:
  * Tạo hàm `refreshData` để tải lại dữ liệu batch mới nhất từ API `/api/v1/latest-batch`.
  * Truyền `batchId={currentBatchId}` và `onRefresh={refreshData}` vào component `<UnifiedDashboard ... />` để đồng bộ dữ liệu sau khi đóng modal duyệt lỗi.

---

## 2. API mới đã thêm

### 2.1. Lấy danh sách item cần review
* **Endpoint:** `GET /api/v1/review/items`
* **Query Params:** `batch_id` (tùy chọn, mặc định lấy batch mới nhất).
* **Mô tả:** Trả về danh sách các panel có lỗi phát hiện bởi AI, sắp xếp theo thứ tự hình ảnh và toạ độ panel.

### 2.2. Cập nhật review của một panel lỗi
* **Endpoint:** `POST /api/v1/review/items/{review_item_id}`
* **Path Params:** `review_item_id` (định dạng `filename::local_id`).
* **Request Body:**
  ```json
  {
    "review_status": "confirmed_defect",
    "review_note": "Ghi chú nếu có"
  }
  ```
* **Mô tả:** Lưu trạng thái duyệt mới và ghi chú của người dùng. Kích hoạt chạy tiến trình sinh lại PDF báo cáo dưới nền.

### 2.3. Lấy tóm tắt thống kê review
* **Endpoint:** `GET /api/v1/review/summary`
* **Query Params:** `batch_id` (tùy chọn, mặc định lấy batch mới nhất).
* **Mô tả:** Trả về số lượng panel theo 4 trạng thái duyệt: `unreviewed` (chưa duyệt), `confirmed_defect` (đúng có lỗi), `needs_review` (xem xét), `false_positive` (không phải lỗi).

---

## 3. Cấu trúc review_status kỹ thuật & nhãn hiển thị UI

| Kỹ thuật (Enum/String) | Nhãn hiển thị UI | Quy tắc báo cáo (`include_in_report`) | Màu sắc hiển thị |
| :--- | :--- | :--- | :--- |
| `unreviewed` | Chưa duyệt / AI phát hiện - chưa duyệt | `True` (Đưa vào báo cáo chính) | Xám Slate |
| `confirmed_defect` | Đúng có lỗi / Đã xác nhận | `True` (Đưa vào báo cáo chính) | Đỏ / Đỏ nhạt |
| `needs_review` | Xem xét / Cần xem xét | `True` (Đưa vào báo cáo chính) | Vàng / Cam |
| `false_positive` | Không phải lỗi | `False` (Loại bỏ khỏi báo cáo chính) | Xám nhạt / Ẩn lỗi |

---

## 4. Thuật toán sắp xếp thứ tự review

Thứ tự review được sắp xếp ổn định cả ở backend bằng phương pháp sort:
1. Sắp xếp tăng dần theo **Chỉ mục hình ảnh** (`image_index`) và tên file ảnh.
2. Sắp xếp tăng dần theo toạ độ panel **`local_id`** (parse theo thứ tự: chỉ mục Block `B`, chỉ mục Row `R`, chỉ mục Col `C`).

Hàm helper được định nghĩa ở Backend để đảm bảo thứ tự duyệt thống nhất:
```python
def parse_local_id(local_id):
    if not local_id:
        return (0, 0, 0)
    b = re.search(r"B(\d+)", local_id)
    r = re.search(r"R(\d+)", local_id)
    c = re.search(r"C(\d+)", local_id)
    block = int(b.group(1)) if b else 0
    row = int(r.group(1)) if r else 0
    col = int(c.group(1)) if c else 0
    return (block, row, col)
```

---

## 5. Quy tắc báo cáo PDF sau review

Khi sinh báo cáo PDF:
* Các thống kê tổng số lượng tấm pin lỗi, tỷ lệ lỗi, tổng hao hụt công suất kW chỉ đếm các panel bị lỗi phát hiện bởi AI có `include_in_report` không phải là `False`.
* Danh sách chi tiết dị thường (Anomaly Log) hoàn toàn loại trừ các panel lỗi bị đánh dấu `false_positive`.
* Nhãn trạng thái review tương ứng (`Đã xác nhận`, `Can xem xet`, `AI phat hien - chua duyet`) được in rõ ràng trong mục chi tiết của từng tấm pin còn lại.

---

## 6. Hướng dẫn xác thực thủ công (Manual Verification)

1. Mở giao diện web và thực hiện chạy phân tích dữ liệu ảnh bằng nút **"Phân tích hệ thống"** (hoặc chọn đợt phân tích cũ nhất có sẵn).
2. Vào trang **"Bản đồ toàn cảnh"** (Unified Dashboard).
3. Bấm vào nút **"Duyệt lỗi phát hiện"** ở góc phải phía trên màn hình.
4. Một modal duyệt lỗi sẽ mở ra:
   * Kiểm tra xem hình ảnh thermal panel lỗi và các trường thông tin tổn thất, độ tin cậy hiển thị chính xác không.
   * Sử dụng nút **"Tiếp theo"** và **"Trước"** để điều hướng lần lượt. Kiểm tra thứ tự duyệt có đúng theo chiều tăng dần của ảnh và local_id không.
5. Chọn một panel lỗi bất kỳ và cập nhật trạng thái là **"Không phải lỗi" (False Positive)**, sau đó bấm **"Hoàn tất review"**:
   * Kiểm tra trên bản đồ, tấm pin vừa chọn phải đổi từ viền đỏ (lỗi) sang viền xanh lá (hoạt động bình thường).
   * Kiểm tra bộ đếm "Phát hiện lỗi" ở cột bên trái phải giảm đi 1.
6. Chọn một panel lỗi khác và đặt trạng thái là **"Xem xét" (Needs Review)**, bấm lưu và đóng modal:
   * Tấm pin trên bản đồ vẫn giữ viền đỏ/cam và khi rê chuột lên, tooltip sẽ hiện thêm nhãn `Duyệt: Xem xét`.
7. Đi tới trang **"Báo cáo"** hoặc bấm nút tải báo cáo PDF:
   * Báo cáo PDF sẽ được tạo lại dưới nền tự động. Tải về và kiểm tra xem tấm pin bị đánh dấu `Không phải lỗi` có hoàn toàn biến mất khỏi danh sách chi tiết lỗi và khỏi biểu đồ/thống kê tóm tắt đầu trang không.
   * Các tấm pin còn lại phải hiển thị đúng dòng `Review Status: Da xac nhan` hoặc `Can xem xet`.

---

## 7. Các lệnh đã chạy và xác thực build

* **Biên dịch Backend:**
  ```powershell
  cd C:\Solar_Inspection_Project\datn_be
  python -m py_compile app\main.py app\services\report_generator.py
  ```
  => *Kết quả: Biên dịch thành công, không phát hiện lỗi cú pháp.*

* **Build Frontend:**
  ```powershell
  cd C:\Solar_Inspection_Project\datn_fe
  npx vite build --minify false
  ```
  => *Kết quả: Biên dịch & build thành công không lỗi.*
