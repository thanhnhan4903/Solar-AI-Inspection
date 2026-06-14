# Report: Sửa đường dữ liệu API/Frontend dùng outer_polygon từ v62

## Kết quả
- Vite build: **✓ built in 4.63s** (no errors)
- Backend compile: **PASS**

---

## Files đã sửa

| File | Thay đổi |
|:-----|:---------|
| `app/main.py` | Endpoint `/api/v1/latest-batch`: trả thêm `outer_polygon`, `inner_polygon`, `calc_polygon`, `geometry_source`, `snap_source`, `outer_area`, `inner_area`. Field `polygon` = `outer_polygon` nếu có và >= 3 điểm, else fallback `raw_polygon`. |
| `src/pages/Unified/UnifiedDashboard.jsx` | `mappedPanels`: ưu tiên `outer_polygon` → `polygon` → bbox. Tooltip thêm dòng geometry_source. |
| `src/pages/Panel/PanelDetail.jsx` | Thêm helper `getPanelOuterPoly(p)`. SVG overlay chính và modal zoom dùng `getPanelOuterPoly`. Hover panel info thêm badge geometry_source (green = line-snap, yellow = YOLO). |

---

## Logic ưu tiên polygon (áp dụng ở tất cả nơi vẽ)

```
panel.outer_polygon (nếu có và length >= 3)   ← v62 line-snap
  ↓ fallback
panel.polygon (nếu có và length >= 3)         ← YOLO raw (backward compat)
  ↓ fallback
bbox → tạo 4-point rect                       ← last resort
```

---

## Các nơi vẽ panel và field đang dùng

| Component | Nơi vẽ | Field ưu tiên |
|:----------|:--------|:--------------|
| `UnifiedDashboard.jsx` | Leaflet `<Polygon>` overlay trên bản đồ | `outer_polygon → polygon → bbox` |
| `PanelDetail.jsx` SVG | SVG `<polygon>` trên ảnh thermal | `getPanelOuterPoly()` = outer → polygon |
| `PanelDetail.jsx` Modal | Modal zoom SVG `<polygon>` | `getPanelOuterPoly()` = outer → polygon |

---

## Cách kiểm tra bằng Network `/api/v1/latest-batch`

1. Mở DevTools → Network tab → Gọi `/api/v1/latest-batch`
2. Mở response → tìm `panels[0]`
3. Xác nhận có các field:
   - `outer_polygon`: `[[x,y], [x,y], [x,y], [x,y]]` (4-point từ line-snap)
   - `inner_polygon`: `[[x,y], ...]` (3px inset)
   - `geometry_source`: `"v61_line_snap"` (hoặc `"yolo_fallback"`)
   - `polygon`: phải trùng với `outer_polygon` (không phải bbox)
4. Hover lên panel trên bản đồ → tooltip hiện `✓ Line-snap V61` hoặc `⚠ YOLO fallback`

---

## Inner polygon

- **Không vẽ** inner_polygon mặc định (theo yêu cầu).
- `inner_polygon` vẫn được trả về trong API để frontend có thể dùng sau nếu cần.
- `panel["area"]` = `inner_area` trong pv_panel_snapper.py.

---

## Không thay đổi

- `GISMapping.jsx`: dùng mock data, không kết nối API thực → giữ nguyên.
- `defect_logic.py`, `panel_geometry.py`: không đổi.
- Schema DB, report generation: không đổi.
