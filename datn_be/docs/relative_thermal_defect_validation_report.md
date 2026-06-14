# Relative Thermal Defect Validation — Implementation Report

**Date:** 2026-06-11  
**Phương pháp:** Relative Spatial-Thermal Contrast Analysis  
**Tên tiếng Việt:** Phân tích Tương phản Không gian - Nhiệt độ Tương đối

---

> **Lưu ý học thuật quan trọng:**  
> Hệ thống sử dụng các chỉ báo tương phản tương đối như một proxy cho bất thường nhiệt.  
> Relative normalized intensity is used as a non-radiometric proxy for thermal anomaly severity.  
> **Không có radiometric metadata. Không suy ra nhiệt độ tuyệt đối (°C).**

---

## 1. File backend đã thêm/sửa

| File | Loại | Mô tả |
|------|------|-------|
| `app/services/defect_thermal_validator.py` | **[NEW]** | Module validator hoàn chỉnh |
| `app/main.py` | **[MODIFY]** | Thêm import + BƯỚC 3b + serialize fields |

---

## 2. File frontend đã sửa

| File | Mô tả |
|------|-------|
| `src/components/DefectReviewModal.jsx` | Thêm section "Kiểm chứng nhiệt tương đối" + preselect logic |

---

## 3. Module defect_thermal_validator hoạt động thế nào

### Vị trí trong pipeline

```
YOLO detect
  ↓
v61/v62 snap panels
  ↓
assign_defects_to_panels()      [defect_logic.py]
  ↓
validate_defects_by_relative_thermal_contrast()   ← MỚI (BƯỚC 3b)
  ↓
assign_row_col_ids()
  ↓
save to DB → dashboard / report / map
```

### Hàm chính

```python
def validate_defects_by_relative_thermal_contrast(
    image_path: str,
    panels: list,
    config: dict | None = None,
    debug_dir: str | None = None,
    image_stem: str | None = None,
) -> list
```

### Luồng xử lý

1. **Đọc ảnh BGR** từ `data/precalib/<filename>`
2. **Tạo thermal maps** (hot_score, dark_score, blue_ratio, ...)
3. **Tính panel features** cho từng panel (inner → outer → bbox fallback)
4. **Tính baseline** = median(panel_mean_hot) của toàn bộ panels
5. **Với mỗi defect** → tính defect features → apply class rules → gắn kết quả
6. **Ghi debug JSONL** → `data/results/debug_logs/<stem>_defect_thermal_validation.jsonl`
7. **Return panels** đã được gắn thêm thermal fields

---

## 4. Cách normalize ảnh 0..1

Không dùng min/max tuyệt đối. Dùng **robust percentile normalization**:

```python
def _robust_normalize(arr, p_low=2, p_high=98, mask=None):
    lo = percentile(vals, p_low)   # Loại bỏ outlier lạnh
    hi = percentile(vals, p_high)  # Loại bỏ outlier nóng
    return clip((arr - lo) / (hi - lo), 0, 1)
```

Nếu có mask → chỉ lấy percentile từ vùng mask (normalize theo ROI).

---

## 5. Features được sử dụng

### 5.1. hot_delta
```
hot_delta = defect_mean_hot - background_mean_hot
```
- Dương → defect nóng hơn background trong cùng panel
- Ngưỡng: `HOT_MEAN_DELTA_MIN=0.10`, `HOT_MEAN_DELTA_STRONG=0.20`

### 5.2. dark_delta
```
dark_delta = background_mean_brightness - defect_mean_brightness
```
- Dương → defect tối hơn background (shading/soiling)
- Ngưỡng: `DARK_MEAN_DELTA_MIN=0.08`, `DARK_MEAN_DELTA_STRONG=0.16`

### 5.3. blue_suppression (blue_drop)
```
blue_drop = background_blue_ratio - defect_blue_ratio
```
- Vùng nóng trong ảnh IronBow/Jet có ít blue hơn
- Ngưỡng: `BLUE_SUPPRESSION_MIN=0.06`, `BLUE_SUPPRESSION_STRONG=0.12`

### 5.4. area_ratio
```
area_ratio = defect_pixel_count / panel_pixel_count
```
- Tính theo inner_polygon (fallback outer)
- Dùng để phân biệt single_cell vs multi_cell vs crack

### 5.5. TNI (Thermal Non-uniformity Index)
```
TNI = std(hot_score) trong vùng defect
```
- TNI cao → vùng không đồng nhất → bất thường nhiệt rõ
- Ngưỡng: `TNI_LOW=0.06`, `TNI_MEDIUM=0.12`, `TNI_HIGH=0.20`

### 5.6. hot_score (composite)
```python
hot_score = 0.45 * brightness_norm
          + 0.35 * yellow_hot_norm   # (r+g)/2 - b
          + 0.20 * blue_supp_norm    # 1 - blue_ratio
```

### 5.7. bright_area_ratio (dùng cho crack)
```
bright_area_ratio = count(panel_pixel > p90_hot) / panel_area
```
- Crack gây phần lớn panel sáng lên → bright_area_ratio ≥ 0.65

---

## 6. Rules cho từng class

### 6.1. hotspot_single_cell

| Điều kiện | Weight | Ngưỡng |
|----------|--------|--------|
| area_ratio trong [0.5%, 3%] | 30% | SINGLE_CELL_SOFT_AREA_RATIO |
| hot_delta | 40% | ≥0.10 (min), ≥0.20 (strong) |
| blue_drop | 20% | ≥0.06 (min), ≥0.12 (strong) |
| aspect_ratio bình thường | bonus/penalty | 0.3–3.0 |
| TNI ≥ 0.12 | +5% bonus | TNI_MEDIUM |

**Tổng score → validation_status:**
- ≥ 0.60 → `confirmed_by_relative_thermal`
- 0.38–0.60 → `needs_review`
- 0.22–0.38 → `needs_review`
- < 0.22 → `suspect_false_positive`

### 6.2. hotspot_multi_cell

| Điều kiện | Weight | Ngưỡng |
|----------|--------|--------|
| area_ratio trong [4%, 55%] | 30% | MULTI_CELL_SOFT_AREA_RATIO |
| hot_delta | 40% | ≥0.10 (min), ≥0.20 (strong) |
| d_p90_hot delta | 20% | So với background |
| blue_drop | 10% | ≥0.06 |

### 6.3. crack

| Điều kiện | Weight | Ngưỡng |
|----------|--------|--------|
| bright_area_ratio trong panel | 45% | ≥0.65 (min), ≥0.80 (target) |
| panel_tni cao | 30% | TNI_HIGH=0.20 |
| hot_delta ≥ 0.10 | 15% | HOT_MEAN_DELTA_MIN |

*Theo định nghĩa project: crack gây phần lớn panel sáng lên.*

### 6.4. shading / soiling

| Điều kiện | Weight | Ngưỡng |
|----------|--------|--------|
| dark_delta | 45% | ≥0.08 (min), ≥0.16 (strong) |
| hot_delta cao → penalty | -20% | Nếu hot_delta ≥ 0.20 |
| area_ratio | bonus | full-panel vs partial |
| TNI ≥ 0.12 | +5% bonus | - |

---

## 7. Field mới được thêm vào defect

| Field | Kiểu | Mô tả |
|-------|------|-------|
| `thermal_validation_status` | str | confirmed_by_relative_thermal / needs_review / suspect_false_positive / class_mismatch / insufficient_pixels / not_run |
| `thermal_validation_score` | float | 0.0–1.0 |
| `rule_class` | str | Class theo YOLO (giữ nguyên, log mismatch) |
| `final_class_suggestion` | str | Class đề xuất nếu mismatch |
| `relative_hot_delta` | float | Δ nhiệt nóng |
| `relative_dark_delta` | float | Δ tối |
| `blue_suppression` | float | Blue channel drop |
| `tni` | float | Thermal Non-uniformity Index |
| `area_ratio_inner` | float | Diện tích defect / diện tích inner panel |
| `severity_by_relative_contrast` | str | low / medium / high |
| `suggested_review_status` | str | confirmed_defect / needs_review |
| `thermal_validation` | dict | Full validation detail (cho debug) |

Panel cũng được gắn thêm:
```python
panel["thermal_panel_features"] = {
    "panel_mean_hot", "panel_std_hot", "panel_tni",
    "panel_mean_shift_from_baseline", "panel_p90_hot", "panel_p95_hot",
    "panel_area_px", "poly_used", "baseline_mean_hot"
}
```

---

## 8. Debug JSONL

**Vị trí:** `data/results/debug_logs/<stem>_defect_thermal_validation.jsonl`

**Format mỗi dòng:**
```json
{
  "image": "DJI_0959",
  "local_id": "R02_C03",
  "defect_idx": 0,
  "yolo_class": "hotspot_single_cell",
  "rule_class": "hotspot_single_cell",
  "final_class_suggestion": "hotspot_single_cell",
  "validation_status": "confirmed_by_relative_thermal",
  "score": 0.72,
  "hot_delta": 0.19,
  "dark_delta": -0.01,
  "blue_suppression": 0.11,
  "area_ratio_inner": 0.013,
  "tni": 0.14,
  "severity": "medium",
  "suggested_review": "confirmed_defect",
  "bright_area_ratio": 0.21,
  "pixel_count": 234,
  "mask_source": "polygon",
  "panel_mean_shift": 0.08
}
```

---

## 9. Review modal hiển thị validation thế nào

Trong `DefectReviewModal.jsx`, section "KIỂM CHỨNG NHIỆT TƯƠNG ĐỐI" hiển thị:

| Trường | Hiển thị |
|-------|---------|
| Status | Badge màu (xanh/vàng/cam/đỏ/xám) |
| YOLO class | Tên class gốc |
| Rule class | Màu cam nếu mismatch |
| Validation score | % |
| Tương phản nóng (Δhot) | Đỏ nếu ≥ 0.12 |
| Blue suppression | Cam nếu ≥ 0.06 |
| TNI | Vàng nếu ≥ 0.12 |
| Mức tương phản nhiệt | low/medium/high |
| Đề xuất tự động | Xanh dương |

Khi `suspect_false_positive`:
```
⚠ Thuật toán tương phản nhiệt không đủ bằng chứng xác nhận lỗi này.
Đề xuất: Xem xét thủ công trước khi xác nhận.
```

**Preselect logic (chỉ khi chưa review thủ công):**
- `suggested_review_status = confirmed_defect` → preselect "Đúng có lỗi"
- `suggested_review_status = needs_review` → preselect "Xem xét"
- Đã review thủ công → giữ nguyên, không preselect

---

## 10. Commands đã chạy

```powershell
# Backend compile
cd C:\Solar_Inspection_Project\datn_be
python -m py_compile app\services\defect_thermal_validator.py
python -m py_compile app\main.py
# → COMPILE OK

# Frontend build
cd C:\Solar_Inspection_Project\datn_fe
npx vite build --minify false
# → ✓ built in 3.29s (1695 modules)
```

---

## 11. Hạn chế (Limitations)

1. **Không có radiometric metadata**: Không có nhiệt độ tuyệt đối. Mọi chỉ số chỉ là tương đối.
2. **Không suy ra °C**: Không dùng bất kỳ công thức nào liên quan đến ΔT theo °C.
3. **Phụ thuộc vào colormap**: Nếu ảnh dùng colormap khác IronBow/Jet (e.g., grayscale), hot_score và blue_drop có thể không chính xác. Hiện tại module chỉ dùng tương phản chung (brightness + color shift).
4. **Panel mask accuracy**: Chất lượng inner_polygon từ v62 ảnh hưởng trực tiếp đến chất lượng validation.
5. **Small defect**: Defect < 10px pixel count → `insufficient_pixels`, không validate được.

---

## 12. TODO còn lại (Optional)

- [ ] `ENABLE_THERMAL_CANDIDATE_SCAN = True`: Phát hiện hotspot mà YOLO bỏ sót (hiện tại mặc định False).
- [ ] `SAVE_THERMAL_VALIDATION_DEBUG_IMAGE = True`: Lưu ảnh debug overlay (hiện tại False).
- [ ] Thêm cột "Kiểm chứng nhiệt" vào bảng danh sách lỗi trong `ReportPage.jsx`.
- [ ] Thêm filter "Nghi ngờ nhận nhầm" trong review modal để lọc nhanh.
- [ ] Calibrate ngưỡng với dataset thực tế (DJI_0959, DJI_0987, DJI_0087_R).
