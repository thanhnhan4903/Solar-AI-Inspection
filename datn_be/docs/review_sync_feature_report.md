# Review Sync Feature — Implementation Report

**Date:** 2026-06-11  
**Author:** Antigravity AI (Claude Sonnet 4.6)

---

## 1. File backend đã sửa

| File | Thay đổi |
|------|----------|
| `app/main.py` | Thêm `POST /api/v1/review/sync` endpoint (dòng ~1354) |
| `app/services/report_generator.py` | Fix filter `false_positive` trong `generate_inspection_report` |

---

## 2. File frontend đã sửa

| File | Thay đổi |
|------|----------|
| `src/components/DefectReviewModal.jsx` | Đổi nút "Hoàn tất review" → "Hoàn tất review và đồng bộ", thêm `handleFinishReviewAndSync`, loading state, error state |
| `App.jsx` | Thêm `useEffect` lắng nghe event `review-sync-completed` → gọi `refreshData()` |
| `src/pages/Dashboard/Home.jsx` | Filter `false_positive` và `include_in_report=false` ra khỏi `faultyPanels` |
| `src/pages/Report/ReportPage.jsx` | Filter `false_positive`, thêm `REVIEW_STATUS_LABEL` map, hiển thị badge trạng thái review |

---

## 3. Endpoint `/api/v1/review/sync` hoạt động thế nào

**Method:** `POST`  
**Body:** `{ "batch_id": 325 }` (optional, nếu không truyền dùng latest batch)

**Logic:**
1. Load tất cả `AiResult` thuộc `batch_id`.
2. Với mỗi AiResult có defects:
   - Đọc `review_status` từ JSON field `defect_type`.
   - Chuẩn hóa `include_in_report`:
     - `false_positive` → `include_in_report = false`
     - `confirmed_defect / needs_review / unreviewed` → `include_in_report = true`
   - Ghi lại JSON vào DB.
3. Tính summary (total_ai_detected, confirmed_defect, needs_review, false_positive, unreviewed, included, excluded).
4. `db.commit()`.
5. Trả về `{ ok, batch_id, summary, message }`.

---

## 4. Rule include_in_report

```python
false_positive       → include_in_report = False  (không tính lỗi, không vào report)
confirmed_defect     → include_in_report = True   (tính lỗi, vào report)
needs_review         → include_in_report = True   (tính lỗi, vào report, nhãn "Xem xét")
unreviewed           → include_in_report = True   (tính lỗi, vào report, nhãn "Chưa duyệt")
```

---

## 5. Dashboard được đồng bộ thế nào

- `Home.jsx` tính `faultyPanels` bằng cách filter:
  ```js
  allPanels.filter(p =>
    (p.total_panel_loss > 0 || p.status === "faulty") &&
    p.review_status !== "false_positive" &&
    p.include_in_report !== false
  )
  ```
- Sau khi sync, `App.jsx` lắng nghe `review-sync-completed` → gọi `fetchLatestBatch()` → set `aiResults` mới → Dashboard render lại với data mới.

---

## 6. Report page được đồng bộ thế nào

- `ReportPage.jsx` tính `faultyPanels` cùng rule với Dashboard.
- Thêm badge `review_status` bên cạnh badge severity trong header mỗi tấm pin:
  - `confirmed_defect` → "Đúng có lỗi" (đỏ)
  - `needs_review` → "Xem xét" (vàng)
  - `unreviewed` → "Chưa duyệt" (xám)
- `false_positive` không xuất hiện trong danh sách lỗi chính.

---

## 7. Report download được đồng bộ thế nào

- Endpoint `GET /api/v1/download-report/{batch_id}` gọi `ReportGenerator.generate_inspection_report(...)`.
- Trong `report_generator.py`, điều kiện `is_faulty` được sửa:
  ```python
  review_status = panel_detail.get("review_status", "unreviewed")
  include_in_report = panel_detail.get("include_in_report", True)
  if review_status == "false_positive" or include_in_report is False:
      is_faulty = False
  elif panel_detail.get("status") == "faulty":
      is_faulty = True
  ```
- Sau khi sync, `include_in_report` đã được chuẩn hóa trong DB → report tải về không chứa `false_positive`.

---

## 8. Nút "Hoàn tất review và đồng bộ" làm những bước nào

```
1. setSyncing(true) — hiển thị "Đang đồng bộ..."
2. handleSave(silent=true) — lưu item hiện tại nếu có thay đổi
3. api.post('/review/sync', { batch_id }) — gọi backend sync
4. window.dispatchEvent(new Event('review-sync-completed')) — notify toàn app
5. onRefresh() — trigger immediate refresh Bản đồ phân tích
6. onClose() — đóng modal
Nếu lỗi: setSyncError('Đồng bộ thất bại. Vui lòng thử lại.')
```

---

## 9. Commands đã chạy

```powershell
# Backend compile
cd C:\Solar_Inspection_Project\datn_be
python -m py_compile app\main.py
python -m py_compile app\services\report_generator.py
# → COMPILE OK

# Frontend build
cd C:\Solar_Inspection_Project\datn_fe
npx vite build --minify false
# → ✓ built in 4.24s
```

---

## 10. Cách test thủ công

```
1. Chạy analyze-all để tạo batch mới.
2. Mở Bản đồ phân tích → click "Duyệt lỗi phát hiện".
3. Trong modal review:
   - Panel A → chọn "Không phải lỗi (False Positive)"
   - Panel B → chọn "Xem xét"
   - Panel C → chọn "Đúng có lỗi"
4. Bấm "Hoàn tất review và đồng bộ".
5. Kiểm tra:
   - Bản đồ: Panel A chuyển sang màu healthy (nếu có xử lý)
   - Dashboard: số lỗi giảm (Panel A không còn tính)
   - Báo cáo: Panel A không xuất hiện, Panel B có nhãn "Xem xét", Panel C có nhãn "Đúng có lỗi"
6. Tải report PDF: Panel A không có trong danh sách lỗi.
7. Refresh browser: review status vẫn còn.
```

---

## 11. TODO còn lại

- **Bản đồ phân tích:** Hiện tại `status` của panel `false_positive` đã là `"healthy"` (được set trong `latest-batch` endpoint dòng 203-206). Nhưng nếu muốn thể hiện rõ hơn màu xám cho `false_positive`, có thể thêm logic màu riêng trong `UnifiedDashboard.jsx`.
- **UnifiedDashboard sidebar stats:** `stats.Issues` đang đếm tất cả panel có `status !== "healthy"` - sau sync thì `false_positive` đã được đặt `status = "healthy"` trong response, nên tự động đúng.
- **Review note audit:** DB đã lưu `review_note`, `reviewed_at`, `reviewed_by` nếu được set.
