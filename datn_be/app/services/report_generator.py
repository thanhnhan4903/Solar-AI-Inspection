# app/services/report_generator.py
import os
import cv2
import json
import re
import datetime
from fpdf import FPDF
from app.services.registration import RegistrationService

# ─────────────────────────────────────────
# Font paths — Times New Roman trên Windows (luôn có mặt)
# ─────────────────────────────────────────
FONT_DIR = "C:/Windows/Fonts"
TIMES_REGULAR = os.path.join(FONT_DIR, "times.ttf")
TIMES_BOLD    = os.path.join(FONT_DIR, "timesbd.ttf")
TIMES_ITALIC  = os.path.join(FONT_DIR, "timesi.ttf")

_USE_TIMES = os.path.exists(TIMES_REGULAR) and os.path.exists(TIMES_BOLD)


def safe(text):
    """Đảm bảo text là string, không cần strip dấu nữa vì font hỗ trợ Unicode."""
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    return text


class CustomPDF(FPDF):
    def __init__(self):
        super().__init__()
        # Đăng ký font Unicode Times New Roman
        if _USE_TIMES:
            self.add_font("TimesNewRoman", style="",  fname=TIMES_REGULAR)
            self.add_font("TimesNewRoman", style="B", fname=TIMES_BOLD)
            self.add_font("TimesNewRoman", style="I", fname=TIMES_ITALIC)
            self._font_name = "TimesNewRoman"
        else:
            self._font_name = "helvetica"

    def _f(self, style="", size=14):
        """Helper: set font nhanh (mặc định size 14 theo yêu cầu)."""
        self.set_font(self._font_name, style=style, size=size)

    def header(self):
        if self.page_no() > 1:
            logo_path = "data/epc_solar.png"
            if os.path.exists(logo_path):
                # Căn lề phải: logo kết thúc ở x=190
                self.image(logo_path, x=155, y=8, w=35)
            self.set_y(12)
            self._f("I", 9)
            self.set_text_color(148, 163, 184)
            self.cell(0, 5, "SOLAR AI PV INSPECTION REPORT", new_x="LMARGIN", new_y="NEXT", align="L")
            self.set_draw_color(226, 232, 240)
            # Dòng kẻ ngang từ lề trái (35) sang lề phải (190)
            self.line(35, 19, 190, 19)
            self.set_y(25)

    def footer(self):
        if self.page_no() > 1:
            self.set_y(-15)
            self._f("I", 9)
            self.set_text_color(148, 163, 184)
            # Đánh số trang ở chân trang
            self.cell(0, 10, f"Page {self.page_no()}", align="C")


class ReportGenerator:
    @staticmethod
    def generate_inspection_report(batch_id, data_list, output_path):
        """
        Tạo báo cáo kiểm tra PDF tuân thủ quy chuẩn:
        - Khổ giấy A4, Font Times New Roman.
        - Nội dung chính: Size 14, Giãn dòng 1.5 (khoảng 7.5mm).
        - Tiêu đề chương: Size 16, Bold.
        - Căn lề: Trái 3.5cm, Phải 2.0cm, Trên 2.5cm, Dưới 2.5cm.
        """
        pairs = RegistrationService.match_thermal_rgb("data/raw")
        thermal_to_rgb = {p['thermal']: p['rgb'] for p in pairs}

        batch = None
        if data_list:
            try:
                batch = data_list[0].image.batch
            except Exception:
                pass

        project_name = batch.project_name if (batch and batch.project_name) else "Dự án kiểm tra điện mặt trời"
        location     = batch.location    if (batch and batch.location)     else "Chưa xác định"
        scan_time    = batch.scan_time   if (batch and batch.scan_time)    else datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        operator     = batch.operator    if (batch and batch.operator)     else "Đội ngũ kỹ thuật vận hành EPC Solar"
        device       = batch.device      if (batch and batch.device)       else "DJI Matrice 300 RTK + Zenmuse H20T"
        scope        = batch.scope       if (batch and batch.scope)        else "Chưa xác định"
        panel_power  = batch.panel_power if (batch and batch.panel_power)  else 600.0

        pdf = CustomPDF()
        # Áp dụng căn lề: Trái 35mm, Trên 25mm, Phải 20mm, Dưới 25mm
        pdf.set_margins(left=35, top=25, right=20)
        pdf.set_auto_page_break(auto=True, margin=25)

        # ─────────────────────────────────────────
        # TRANG BÌA
        # ─────────────────────────────────────────
        pdf.add_page()

        logo_path = "data/epc_solar.png"
        if os.path.exists(logo_path):
            # Căn giữa logo trên trang bìa (printable width là 155mm, tâm là 112.5)
            pdf.image(logo_path, x=92.5, y=30, w=40)
            pdf.ln(50)
        else:
            pdf.ln(30)

        pdf._f("B", 22)
        pdf.set_text_color(15, 23, 42)
        pdf.cell(0, 12, "BÁO CÁO KIỂM TRA ĐIỆN MẶT TRỜI AI", new_x="LMARGIN", new_y="NEXT", align="C")

        pdf._f("B", 12)
        pdf.set_text_color(14, 165, 233)
        pdf.cell(0, 8, "TỰ ĐỘNG PHÁT HIỆN & CHẨN ĐOÁN LỖI TẤM PIN", new_x="LMARGIN", new_y="NEXT", align="C")
        pdf.ln(15)

        # Hộp thông tin căn giữa theo lề mới
        pdf.set_fill_color(248, 250, 252)
        pdf.set_draw_color(226, 232, 240)
        pdf.rect(35, 110, 155, 95, "DF")

        metadata_fields = [
            ("Tên dự án:",       project_name),
            ("Vị trí:",          location),
            ("Thời gian quét:",  scan_time),
            ("Đơn vị thực hiện:", operator),
            ("Thiết bị bay:",    device),
            ("Phạm vi quét:",    scope),
            ("Công suất tấm pin:", f"{panel_power} W"),
        ]

        y_offset = 115
        for label, val in metadata_fields:
            pdf.set_xy(42, y_offset)
            pdf._f("B", 11)
            pdf.set_text_color(71, 85, 105)
            pdf.cell(45, 7, safe(label))
            pdf._f("", 11)
            pdf.set_text_color(15, 23, 42)
            pdf.cell(100, 7, safe(str(val)), new_x="LMARGIN", new_y="NEXT")
            y_offset += 10

        pdf.set_xy(0, 255)
        pdf._f("I", 10)
        pdf.set_text_color(148, 163, 184)
        pdf.cell(0, 10, "Bảo mật - EPC Solar JSC", align="C", new_x="LMARGIN", new_y="NEXT")

        # ─────────────────────────────────────────
        # TRANG 2: MỤC LỤC (TABLE OF CONTENTS)
        # ─────────────────────────────────────────
        pdf.add_page()
        pdf.set_xy(35, 25)
        pdf._f("B", 16)
        pdf.set_text_color(15, 23, 42)
        pdf.cell(0, 10, "MỤC LỤC", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(5)

        # Đoạn văn giới thiệu mục lục có thụt lề
        pdf.set_x(35 + 12.7)
        pdf._f("", 14)
        pdf.set_text_color(51, 65, 85)
        pdf.write(7.5, "Mục lục dưới đây tóm tắt toàn bộ cấu trúc các chương mục chính của báo cáo kết quả quét nhiệt lỗi tấm pin mặt trời để phục vụ công tác tra cứu nhanh.")
        pdf.ln(15)

        toc_items = [
            ("1. Tóm tắt dự án (Executive Summary)", 3),
            ("2. Phân tích phân bổ các loại lỗi (Defect Distribution)", 3),
            ("3. Nhật ký chi tiết lỗi tấm pin (Detailed Anomalies Log)", 4),
        ]

        for title, page_num in toc_items:
            pdf.set_x(35)
            pdf._f("", 14)
            pdf.set_text_color(51, 65, 85)
            pdf.cell(125, 7.5, title)
            
            pdf._f("", 14)
            pdf.set_text_color(148, 163, 184)
            pdf.cell(20, 7.5, " . . . . . . . . .", align="R")
            
            pdf._f("B", 14)
            pdf.set_text_color(15, 23, 42)
            pdf.cell(10, 7.5, str(page_num), align="R", new_x="LMARGIN", new_y="NEXT")
            pdf.ln(2)

        # ─────────────────────────────────────────
        # TRANG 3: TÓM TẮT
        # ─────────────────────────────────────────
        pdf.add_page()

        pdf.set_xy(35, 25)
        pdf._f("B", 16)
        pdf.set_text_color(15, 23, 42)
        pdf.cell(0, 10, "1. Tóm tắt dự án", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)

        # Đoạn văn giới thiệu có thụt lề đầu dòng 1.27cm và giãn dòng 1.5 (line height 7.5mm)
        pdf.set_x(35 + 12.7)
        pdf._f("", 14)
        pdf.set_text_color(51, 65, 85)
        pdf.write(7.5, "Báo cáo này trình bày kết quả phân tích tự động hệ thống điện mặt trời bằng công nghệ AI. Dưới đây là các chỉ số thống kê tổng hợp về số lượng tấm pin, tỷ lệ lỗi và mức độ suy hao công suất phát hiện trong đợt kiểm tra.")
        pdf.ln(12)

        total_panels = len(data_list)
        faulty_panels = []
        healthy_count = 0
        total_power_loss_w = 0.0

        stats_defects = {
            "hotspot_single_cell": 0,
            "hotspot_multi_cell": 0,
            "shading": 0,
            "soiling": 0,
            "crack": 0
        }

        for p in data_list:
            panel_detail = {}
            is_faulty = False

            if p.defect_type:
                if p.defect_type.startswith("{"):
                    try:
                        panel_detail = json.loads(p.defect_type)
                        if panel_detail.get("status") == "faulty":
                            is_faulty = True
                    except Exception:
                        pass
                elif p.defect_type != "Healthy":
                    is_faulty = True
                    panel_detail = {
                        "status": "faulty",
                        "total_panel_loss": p.loss_pct,
                        "worst_severity": "minor",
                        "recommendation": "Kiểm tra",
                        "bbox": [],
                        "polygon": [],
                        "defects": [{
                            "class_name": "hotspot_single_cell",
                            "confidence": p.confidence,
                            "severity": "minor",
                            "recommendation": "Kiểm tra",
                            "relative_position": {"u": 0.5, "v": 0.5},
                            "location_in_panel": "center"
                        }]
                    }

            if is_faulty:
                faulty_panels.append((p, panel_detail))
                total_power_loss_w += panel_detail.get("total_panel_loss", 0.0)
                for d in panel_detail.get("defects", []):
                    cname = d.get("class_name", "")
                    if cname in stats_defects:
                        stats_defects[cname] += 1
            else:
                healthy_count += 1

        total_faulty = len(faulty_panels)
        health_rate = round((total_panels - total_faulty) / total_panels * 100, 1) if total_panels > 0 else 100.0

        # --- Grid thống kê (Khổ 155mm lề mới) ---
        # Card 1: TOTAL SCANNED PANELS
        pdf.set_fill_color(248, 250, 252)
        pdf.rect(35, 75, 73, 40, "F")
        pdf.set_xy(40, 79)
        pdf._f("B", 10)
        pdf.set_text_color(100, 116, 139)
        pdf.cell(63, 5, "TỔNG SỐ TẤM PIN ĐÃ QUÉT", new_x="LMARGIN", new_y="NEXT")
        pdf._f("B", 20)
        pdf.set_text_color(14, 165, 233)
        pdf.set_x(40)
        pdf.cell(63, 12, str(total_panels), new_x="LMARGIN", new_y="NEXT")
        pdf._f("", 10)
        pdf.set_text_color(71, 85, 105)
        pdf.set_x(40)
        pdf.cell(63, 5, f"Bình thường: {total_panels - total_faulty} ({health_rate}%)", new_x="LMARGIN", new_y="NEXT")

        # Card 2: ANOMALOUS / FAULTY PANELS
        pdf.set_fill_color(248, 250, 252)
        pdf.rect(117, 75, 73, 40, "F")
        pdf.set_xy(122, 79)
        pdf._f("B", 10)
        pdf.set_text_color(100, 116, 139)
        pdf.cell(63, 5, "TẤM PIN BẤT THƯỜNG / LỖI", new_x="LMARGIN", new_y="NEXT")
        pdf._f("B", 20)
        pdf.set_text_color(239, 68, 68)
        pdf.set_x(122)
        pdf.cell(63, 12, str(total_faulty), new_x="LMARGIN", new_y="NEXT")
        pdf._f("", 10)
        pdf.set_text_color(71, 85, 105)
        pdf.set_x(122)
        fault_rate = round(total_faulty / total_panels * 100, 1) if total_panels > 0 else 0.0
        pdf.cell(63, 5, f"Tỷ lệ lỗi: {fault_rate}%", new_x="LMARGIN", new_y="NEXT")

        # Card 3: TOTAL ESTIMATED POWER LOSS
        pdf.set_fill_color(248, 250, 252)
        pdf.rect(35, 122, 73, 40, "F")
        pdf.set_xy(40, 126)
        pdf._f("B", 10)
        pdf.set_text_color(100, 116, 139)
        pdf.cell(63, 5, "TỔNG HAO HỤT CÔNG SUẤT DỰ TÍNH", new_x="LMARGIN", new_y="NEXT")
        pdf._f("B", 20)
        pdf.set_text_color(245, 158, 11)
        pdf.set_x(40)
        pdf.cell(63, 12, f"{round(total_power_loss_w / 1000, 2)} kW", new_x="LMARGIN", new_y="NEXT")
        pdf._f("", 10)
        pdf.set_text_color(71, 85, 105)
        pdf.set_x(40)
        avg_loss = round(total_power_loss_w / total_faulty, 1) if total_faulty > 0 else 0.0
        pdf.cell(63, 5, f"Hao hụt trung bình: {avg_loss} W / lỗi", new_x="LMARGIN", new_y="NEXT")

        # --- Bảng phân loại lỗi (Tiêu đề chương 16 Bold, căn lề 3.5cm) ---
        pdf.set_xy(35, 172)
        pdf._f("B", 16)
        pdf.set_text_color(15, 23, 42)
        pdf.cell(0, 10, "2. Phân tích phân bổ các loại lỗi", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(3)

        pdf.set_fill_color(15, 23, 42)
        pdf.set_text_color(255, 255, 255)
        pdf._f("B", 12)
        pdf.cell(60, 8.5, "Loại lỗi", border=1, align="L", fill=True)
        pdf.cell(45, 8.5, "Thuật ngữ tiếng Anh", border=1, align="C", fill=True)
        pdf.cell(25, 8.5, "Số lượng", border=1, align="C", fill=True)
        pdf.cell(25, 8.5, "Tỷ lệ (%)", border=1, align="C", fill=True, new_x="LMARGIN", new_y="NEXT")

        pdf.set_text_color(15, 23, 42)
        pdf._f("", 12)

        defect_mapping_list = [
            ("hotspot_single_cell", "Hotspot single cell", "Hotspot single cell", stats_defects["hotspot_single_cell"]),
            ("hotspot_multi_cell",  "Hotspot multi cell",  "Hotspot multi cell",  stats_defects["hotspot_multi_cell"]),
            ("shading",             "Shading",             "Shading",             stats_defects["shading"]),
            ("soiling",             "Soiling",             "Soiling",             stats_defects["soiling"]),
            ("crack",               "Crack",               "Crack",               stats_defects["crack"]),
        ]

        sum_defects = sum(stats_defects.values())

        for index, (key, vi_name, en_name, count) in enumerate(defect_mapping_list):
            if index % 2 == 0:
                pdf.set_fill_color(248, 250, 252)
            else:
                pdf.set_fill_color(255, 255, 255)
            ratio = round(count / sum_defects * 100, 1) if sum_defects > 0 else 0.0
            pdf.cell(60, 8.5, f" {vi_name}", border=1, align="L", fill=True)
            pdf.cell(45, 8.5, en_name, border=1, align="C", fill=True)
            pdf.cell(25, 8.5, str(count), border=1, align="C", fill=True)
            pdf.cell(25, 8.5, f"{ratio}%", border=1, align="C", fill=True, new_x="LMARGIN", new_y="NEXT")

        # ─────────────────────────────────────────
        # PHẦN 3: DANH SÁCH CHI TIẾT CÁC LỖI
        # ─────────────────────────────────────────
        pdf.add_page()
        pdf._f("B", 16)
        pdf.set_text_color(15, 23, 42)
        pdf.cell(0, 10, "3. Nhật ký chi tiết lỗi tấm pin", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)

        # Đoạn văn giới thiệu có thụt lề đầu dòng 1.27cm và giãn dòng 1.5 (line height 7.5mm)
        pdf.set_x(35 + 12.7)
        pdf._f("", 14)
        pdf.set_text_color(51, 65, 85)
        pdf.write(7.5, "Danh sách chi tiết dưới đây mô tả vị trí, tọa độ GPS, mức độ hao hụt công suất và hình ảnh nhiệt/RGB tương ứng của từng tấm pin mặt trời được xác định là có bất thường hoặc bị lỗi trong quá trình quét bằng thiết bị bay không người lái.")
        pdf.ln(12)

        temp_dir = "data/temp_crop"
        os.makedirs(temp_dir, exist_ok=True)

        VI_DEFECT_MAP = {
            "hotspot_single_cell": "Hotspot single cell",
            "hotspot_multi_cell":  "Hotspot multi cell",
            "shading":             "Shading",
            "soiling":             "Soiling",
            "crack":               "Crack",
        }

        VI_SEVERITY_MAP = {
            "very_minor": "Rất nhẹ",
            "minor":      "Nhẹ",
            "moderate":   "Cần theo dõi",
            "severe":     "Ưu tiên bảo trì",
            "replace":    "Cần thay thế"
        }

        VI_LOC_MAP = {
            "center": "Trung tâm",
            "top-left": "Trên - Trái",
            "top-center": "Trên - Giữa",
            "top-right": "Trên - Phải",
            "middle-left": "Giữa - Trái",
            "middle-center": "Giữa - Trung tâm",
            "middle-right": "Giữa - Phải",
            "bottom-left": "Dưới - Trái",
            "bottom-center": "Dưới - Giữa",
            "bottom-right": "Dưới - Phải",
        }

        for idx, (p, p_detail) in enumerate(faulty_panels):
            # Mỗi khối cần khoảng 100mm chiều cao, nếu không đủ thì ngắt trang tự động
            if pdf.get_y() > 180:
              pdf.add_page()

            pdf._f("B", 13)
            pdf.set_text_color(15, 23, 42)
            pdf.set_fill_color(241, 245, 249)

            local_id = p.panel.local_id
            row = p_detail.get("row", 0)
            col = p_detail.get("col", 0)

            lat = p_detail.get("gps_lat")
            lng = p_detail.get("gps_lng")
            if lat is None or lng is None:
                lat = round(10.8231 + row * 0.00015, 6)
                lng = round(106.6297 + col * 0.00025, 6)
            else:
                lat = round(lat, 6)
                lng = round(lng, 6)

            pdf.cell(
                0, 7.5,
                f" Anomalous Panel {local_id} (Row {row}, Col {col}) | GPS: {lat}, {lng}",
                border=1, align="L", fill=True, new_x="LMARGIN", new_y="NEXT"
            )

            pdf._f("", 12)
            pdf.set_text_color(51, 65, 85)

            # Cắt ảnh
            thermal_filename = p.image.filename
            rgb_filename = thermal_to_rgb.get(thermal_filename)
            thermal_path = os.path.join("data/results", thermal_filename)
            rgb_path = os.path.join("data/raw", rgb_filename) if rgb_filename else None
            bbox = p_detail.get("bbox", [])

            crop_success = False
            temp_thermal_path = ""
            temp_rgb_path = ""

            if bbox and len(bbox) == 4 and os.path.exists(thermal_path):
                try:
                    t_img = cv2.imread(thermal_path)
                    if t_img is not None:
                        h_t, w_t = t_img.shape[:2]
                        x1, y1, x2, y2 = [int(v) for v in bbox]
                        crop_w = int(w_t * 0.5)
                        crop_h = int(h_t * 0.5)
                        cx = (x1 + x2) // 2
                        cy = (y1 + y2) // 2
                        x1_t = max(0, cx - crop_w // 2)
                        y1_t = max(0, cy - crop_h // 2)
                        x2_t = min(w_t, cx + crop_w // 2)
                        y2_t = min(h_t, cy + crop_h // 2)
                        crop_t = t_img[y1_t:y2_t, x1_t:x2_t]
                        if crop_t is not None and crop_t.size > 0:
                            temp_thermal_path = os.path.join(temp_dir, f"t_{local_id}_{idx}.jpg")
                            cv2.imwrite(temp_thermal_path, crop_t)

                        if rgb_path and os.path.exists(rgb_path):
                            r_img = cv2.imread(rgb_path)
                            if r_img is not None:
                                h_r, w_r = r_img.shape[:2]
                                scale_x = w_r / w_t
                                scale_y = h_r / h_t
                                cx_r = int(cx * scale_x)
                                cy_r = int(cy * scale_y)
                                crop_w_r = int(crop_w * scale_x)
                                crop_h_r = int(crop_h * scale_y)
                                x1_r = max(0, cx_r - crop_w_r // 2)
                                y1_r = max(0, cy_r - crop_h_r // 2)
                                x2_r = min(w_r, cx_r + crop_w_r // 2)
                                y2_r = min(h_r, cy_r + crop_h_r // 2)
                                crop_r = r_img[y1_r:y2_r, x1_r:x2_r]
                                if crop_r is not None and crop_r.size > 0:
                                    r_pad_x = int((x1 - x1_t) * scale_x)
                                    r_pad_y = int((y1 - y1_t) * scale_y)
                                    r_w = int((x2 - x1) * scale_x)
                                    r_h = int((y2 - y1) * scale_y)
                                    cv2.rectangle(crop_r, (r_pad_x, r_pad_y), (r_pad_x + r_w, r_pad_y + r_h), (0, 255, 0), 2)
                                    temp_rgb_path = os.path.join(temp_dir, f"r_{local_id}_{idx}.jpg")
                                    cv2.imwrite(temp_rgb_path, crop_r)

                        crop_success = os.path.exists(temp_thermal_path) and os.path.exists(temp_rgb_path)
                except Exception:
                    crop_success = False

            # In ảnh kẹp song song ở trên trong lề mới (printable width 155mm)
            image_y = pdf.get_y() + 2
            if crop_success:
                IMG_W = 70

                pdf.image(
                    temp_thermal_path,
                    x=35,
                    y=image_y,
                    w=IMG_W,
                    h=30
                )

                pdf.image(
                    temp_rgb_path,
                    x=120,
                    y=image_y,
                    w=IMG_W,
                    h=30
                )
                
                # Bổ sung nhãn ảnh
                pdf.set_xy(35, image_y + 30.5)
                pdf._f("I", 8)
                pdf.set_text_color(100, 116, 139)
                pdf.cell(73, 4, "Ảnh nhiệt (Có nhãn)", align="C")
                
                pdf.set_xy(116, image_y + 30.5)
                pdf.cell(73, 4, "Ảnh quang học (RGB)", align="C", new_x="LMARGIN", new_y="NEXT")
                
                # Dịch chuyển y xuống dưới ảnh + nhãn
                pdf.set_y(image_y + 36)
            else:
                pdf.ln(2)

            # Vùng text chi tiết (Căn chữ Justify, Nội dung Size 14, Giãn dòng 1.5)
            LABEL_W = 40
            TEXT_W = pdf.w - pdf.l_margin - pdf.r_margin
            text_x = pdf.l_margin

            pdf.ln(1)

            defects = p_detail.get("defects", [])
            defect_types      = []
            confidence_values = []
            relative_positions = []

            def draw_table_row(pdf, cols):

                widths = [40, 35, 25, 55]
                line_h = 6

                max_lines = max(
                    len(str(c).split("\n"))
                    for c in cols
                )

                row_h = max_lines * line_h + 2

                if pdf.get_y() + row_h > 260:
                    pdf.add_page()

                x0 = pdf.l_margin
                y0 = pdf.get_y()
                x = x0

                for txt, w in zip(cols, widths):

                    pdf.rect(x, y0, w, row_h)

                    pdf.set_xy(x + 1, y0 + 1)

                    pdf.multi_cell(
                        w - 2,
                        line_h,
                        str(txt),
                        border=0,
                        align="C"
                    )

                    x += w
                    pdf.set_xy(x, y0)

                pdf.set_xy(pdf.l_margin, y0 + row_h)
                        
            pdf.ln(5)

                        # ==================================================
                        # BẢNG CHI TIẾT LỖI GIỐNG HÌNH 2
                        # ==================================================
            loss = p_detail.get("total_panel_loss", 0.0)
            loss_pct_calc = (
                round((loss / panel_power) * 100, 1)
                if panel_power > 0 else 0.0
            )
            rec = p_detail.get("recommendation", "Kiểm tra")

            if rec == "Check":
                rec = "Kiểm tra"
            elif rec == "Replace":
                rec = "Cần thay thế"
            pdf._f("B", 11)

            pdf.set_draw_color(0, 0, 0)
            pdf.set_line_width(0.6)

            COL1 = 40
            COL2 = 35
            COL3 = 25
            COL4 = 55

            pdf.cell(COL1, 10, "Lỗi phát hiện", 1, 0, "C")
            pdf.cell(COL2, 10, "Mức độ & độ tin cậy", 1, 0, "C")
            pdf.cell(COL3, 10, "Hao hụt", 1, 0, "C")
            pdf.cell(COL4, 10, "Vị trí", 1, 1, "C")

            pdf._f("", 10)
            for d in defects:

                defect_name = VI_DEFECT_MAP.get(
                    d.get("class_name", ""),
                    d.get("class_name", "")
                )

                confidence = f"{round(d.get('confidence',0)*100,1)}%"

                severity = VI_SEVERITY_MAP.get(
                    d.get("severity","minor"),
                    "Nhẹ"
                )

                power_loss = f"{loss:.1f} W"

                location = d.get(
                    "location_in_panel",
                    "center"
                )

                u = round(
                    d.get("relative_position", {}).get("u", 0.5),
                    2
                )

                v = round(
                    d.get("relative_position", {}).get("v", 0.5),
                    2
                )

                location_text = (
                    f"{location}\n"
                    f"(u:{u}, v:{v})"
                )

                draw_table_row(
                    pdf,
                    [
                        defect_name,
                        f"{severity}\n{confidence}",
                        power_loss,
                        location_text
                    ]
                )
            # Tính toán hao hụt và sản lượng hao hụt
            loss = p_detail.get("total_panel_loss", 0.0)
            loss_pct_calc = round((loss / panel_power) * 100, 1) if panel_power > 0 else 0.0
            rec  = p_detail.get("recommendation", "Kiểm tra")
            if rec == "Check":
                rec = "Kiểm tra"
            elif rec == "Replace":
                rec = "Cần thay thế"
            #print_row("Hao hụt công suất:",      f"{loss} W ({loss_pct_calc}% công suất tấm pin) | {rec}")
            
            # Giả định 4 giờ nắng đỉnh mỗi ngày và đơn giá điện tự tiêu thụ/phát thải trung bình là 2,000 VND / kWh
            yearly_loss_kwh = (loss * 4.0 * 365) / 1000.0
            yearly_cost_vnd = int(yearly_loss_kwh * 2000)
            #print_row("Hao hụt sản lượng ước tính:", f"{round(yearly_loss_kwh, 1)} kWh/năm (Thiệt hại ước tính: {yearly_cost_vnd:,} VNĐ/năm)")
            
           # print_row("Vị trí tương đối:",   ", ".join(relative_positions) if relative_positions else "N/A")
           #print_row("Ảnh nhiệt:",   thermal_filename)
           #print_row("Ảnh quang học:",       rgb_filename or "Không khớp ảnh")

            pdf.ln(5)

        # Xóa file tạm
        try:
            for f in os.listdir(temp_dir):
                os.remove(os.path.join(temp_dir, f))
            os.rmdir(temp_dir)
        except Exception:
            pass

        pdf.output(output_path)
        return output_path