# v62 Integration Report

## Kết quả tích hợp

Tất cả compile PASS. v62 module load OK. Pipeline sẵn sàng chạy.

---

## Tóm tắt những gì đã làm

### Bước 1: Backup v61
- `scratch/pv_fullsnap_universal_v61_full_backup_before_be_integration.py` — bản gốc không sửa

### Bước 2: Tạo v62 backend-ready
- `scratch/pv_fullsnap_universal_v62_backend_ready.py` — copy v61 + thêm:
  - `_BACKEND_RESULT_PANELS = []` — global list collect kết quả
  - Hook ở cuối vòng loop `panel_calc_entries` trong **cả hai engine** (v34 và v42)
    để append mỗi panel `{outer_polygon, inner_polygon, outer_area, inner_area, valid, source}`
  - Hàm `process_image_for_backend(image_path, panel_log_path, output_dir, debug)` ở trên `def main()`

### Bước 3: Tạo adapter service
- `app/services/pv_panel_snapper.py` — hàm `snap_panels_with_v61()`:
  - Load v62 module một lần và cache
  - Ghi YOLO panels ra file JSONL để v62 đọc làm prior
  - Gọi `v62.process_image_for_backend()` trong try/except
  - Convert kết quả sang backend dict format
  - Trả `[]` nếu thất bại (backend tự fallback)

### Bước 4: Sửa main.py
- Import `snap_panels_with_v61`
- Sau YOLO detect, thêm bước 1b: gọi snapper
- Fallback logic: nếu snapped = [] → dùng panels_yolo
- Lưu thêm `outer_polygon`, `inner_polygon`, `geometry_source` vào DB JSON
- `_serialize_panels()` trả thêm 5 fields mới

### Bước 5: Sửa panel_processor.py
- `draw_custom_annotation()`: ưu tiên `outer_polygon` khi vẽ
- Panel border thickness: 2px → 3px

---

## Luồng mới

```
POST /api/v1/analyze-all
  → YOLO detect → panels_yolo, defects_raw
  → snap_panels_with_v61(image_path, panels_yolo)   ← MỚI
      → ghi panel_refine.jsonl (YOLO prior)
      → v62.process_image_for_backend()
          → route metrics → chọn v34 hay v42
          → run engine (line-snap không đổi)
          → hook collect _BACKEND_RESULT_PANELS
          → return {panels: [...], ok: True}
      → convert → panels_snapped
  → nếu panels_snapped empty → fallback → panels_yolo
  → assign_defects_to_panels(panels_raw, ...)       ← như cũ
  → assign_row_col_ids(...)                         ← như cũ
  → draw_custom_annotation(...)  ← dùng outer_polygon
  → lưu DB (có thêm outer_polygon/inner_polygon)    ← như cũ + fields mới
```

---

## Schema panel dict mới

```python
{
    # Backward compat
    "polygon": outer_polygon,       # = outer_polygon (không đổi)
    "bbox": [x1, y1, x2, y2],
    "area": inner_area,             # nội diện sau inset 3px

    # Mới từ v62
    "outer_polygon": [[x,y], ...],  # 4-point từ line-snap
    "inner_polygon": [[x,y], ...],  # 4-point inset 3px
    "calc_polygon": inner_polygon,  # alias
    "outer_area": float,
    "inner_area": float,
    "geometry_source": "v61_line_snap" | "yolo_fallback",
    "snap_source": str,             # nguồn snap cụ thể
    "snap_engine": "small_core_v34_full" | "large_local_v42_full",
}
```

---

## Files đã sửa / tạo

| File | Thao tác |
|:-----|:---------|
| `scratch/pv_fullsnap_universal_v61_full_backup_before_be_integration.py` | TẠO MỚI (backup) |
| `scratch/pv_fullsnap_universal_v62_backend_ready.py` | TẠO MỚI (v61 + BE hooks) |
| `app/services/pv_panel_snapper.py` | TẠO MỚI |
| `app/main.py` | SỬA (import, pipeline, serialize, DB) |
| `app/services/panel_processor.py` | SỬA (draw_custom_annotation) |

---

## Không đổi gì

- Thuật toán line-snap trong v62 (100% giữ nguyên từ v61)
- Schema DB (panels table, ai_results table)
- assign_defects_to_panels logic
- assign_row_col_ids logic
- defect_logic.py
- panel_geometry.py
- Report generation
