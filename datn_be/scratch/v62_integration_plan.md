# v62 Integration Plan — Tích hợp v61 line-snap engine vào Backend

## Tóm tắt mục tiêu

Thay thế panel polygon cũ (YOLO polygon / bbox / minAreaRect) bằng polygon từ thuật toán line-snap v61,
giữ nguyên toàn bộ luồng YOLO detect, defect assignment, và DB schema.

---

## Phân tích backend hiện tại

### Luồng pipeline cũ (trước tích hợp)

```
POST /api/v1/analyze-all
  → main.py (line ~495)
    → ai_engine.detect_and_segment(img_path)
        → AIEngine.detect_and_segment()          [ai_engine.py L50]
          → YOLO model.predict()
          → process_yolo_predictions()            [panel_processor.py L888]
            → refine_panel_contour()             ← TẠO POLYGON PANEL CŨ (YOLO mask → bbox/minAreaRect)
            → simplify_defect_polygon()
    → panels_raw = [d ... if d["category"] == "panel"]
    → defects_raw = [d ... if d["category"] == "defect"]
    → assign_defects_to_panels(panels_raw, ...)  [defect_logic.py L48]
        → dùng panel["polygon"] để tính overlap
    → assign_row_col_ids(panels_with_defects)    [panel_geometry.py L167]
    → draw_custom_annotation(orig_img, final_panels)
        → dùng panel["polygon"] để vẽ
    → Lưu panel["polygon"] vào DB qua defect_str JSON
```

### Các điểm quan trọng xác định

| Câu hỏi | Trả lời |
|:--------|:--------|
| YOLO detect được gọi ở đâu? | app/main.py L495: ai_engine.detect_and_segment(img_path) |
| Panel polygon cũ được tạo ở đâu? | app/services/panel_processor.py L946: refine_panel_contour() |
| Field nào lưu panel polygon? | panel["polygon"] (L966 panel_processor.py) |
| Field nào dùng để vẽ panel? | p.get("polygon", []) trong draw_custom_annotation() (L1204) |
| Field nào dùng để assign defect? | p.get("polygon", []) trong defect_logic.py L80 |
| Field nào lưu DB? | defect_str = json.dumps({"polygon": ..., "bbox": ..., ...}) trong main.py L583-596 |
| local_id / row / col được gán ở đâu? | assign_row_col_ids() trong panel_geometry.py L167, gọi từ main.py L520 |

### DB Schema (không sửa)

- panels.local_id — string R01_C03
- ai_results.defect_type — TEXT chứa JSON với polygon, defects, bbox...
- Polygon được lưu vào ai_results.defect_type (JSON field)

### Nơi sẽ cắm v61 panel snapper

Sau bước ai_engine.detect_and_segment() và TRƯỚC assign_defects_to_panels(), trong app/main.py khoảng dòng 498-512:

```python
panels_raw = [d for d in raw_detections if d["category"] == "panel"]
defects_raw = [d for d in raw_detections if d["category"] == "defect"]

# [MỚI] Thay panel polygon bằng v62 line-snap
panels_for_backend = snap_panels_with_v61(
    image_path=img_path,
    yolo_panels=panels_raw,
    output_dir=results_dir,
    debug=False,
)
if not panels_for_backend:
    panels_for_backend = panels_raw  # fallback

# [CŨ] Tiếp tục pipeline như cũ
panels_with_defects, unassigned = assign_defects_to_panels(panels_for_backend, ...)
```

---

## Files sẽ tạo mới

1. scratch/pv_fullsnap_universal_v61_full_backup_before_be_integration.py
   - Backup nguyên vẹn của v61

2. scratch/pv_fullsnap_universal_v62_backend_ready.py
   - Copy của v61 + thêm hàm process_image_for_backend()
   - Không thay đổi bất kỳ thuật toán line-snap nào
   - Chỉ thêm global list _BACKEND_RESULT_PANELS và hook vào panel_calc_entries

3. app/services/pv_panel_snapper.py
   - Adapter: nhận image_path + yolo_panels, gọi v62, trả panels_snapped
   - Hàm chính: snap_panels_with_v61()

---

## Files sẽ sửa

4. app/main.py
   - Import pv_panel_snapper.snap_panels_with_v61
   - Thêm call snapper sau detect_and_segment
   - Thêm field outer_polygon, inner_polygon, geometry_source vào serialize

5. app/services/panel_processor.py
   - draw_custom_annotation(): dùng outer_polygon nếu có, fallback polygon
   - Đổi panel thickness từ 2 → 3px

---

## Cơ chế lấy polygon từ v62

Trong v62, khi panel_calc_entries được duyệt (L3704-3730 v34, L11757-11808 v42):
- e["outer_polygon"] — np.array float32 shape (4,2) — outer polygon từ line-snap
- e["inner_polygon"] — np.array float32 shape (4,2) — inner 3px inset
- e["outer_area"] — float
- e["calc_area"] — float = inner_area
- e["valid"] — bool
- e["src_name"] — string (source info)

Chiến lược: thêm global _BACKEND_RESULT_PANELS = [] để collect kết quả.
Hàm process_image_for_backend() đọc list này và trả về dict chuẩn.
