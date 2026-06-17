# app/services/report_generator.py
import os
import json
import re
import datetime
import tempfile
import shutil
import cv2
import numpy as np
import logging
from fpdf import FPDF
from app.services.registration import RegistrationService

logger = logging.getLogger("solar_ai")


# ─────────────────────────────────────────
# Font paths — Times New Roman trên Windows (luôn có mặt)
# ─────────────────────────────────────────
def find_font(font_name):
    paths = [
        os.path.join("C:/Windows/Fonts", font_name),
        os.path.join("C:/windows/fonts", font_name),
        os.path.join("c:/windows/fonts", font_name.lower()),
        os.path.join("C:/Windows/Fonts", font_name.lower()),
    ]
    for p in paths:
        if os.path.exists(p):
            return p
    return None

TIMES_REGULAR = find_font("times.ttf") or "C:/Windows/Fonts/times.ttf"
TIMES_BOLD    = find_font("timesbd.ttf") or "C:/Windows/Fonts/timesbd.ttf"
TIMES_ITALIC  = find_font("timesi.ttf") or "C:/Windows/Fonts/timesi.ttf"

_USE_TIMES = os.path.exists(TIMES_REGULAR) and os.path.exists(TIMES_BOLD)

# ─────────────────────────────────────────
# Tìm kiếm logo và ảnh bản đồ toàn cảnh
# ─────────────────────────────────────────
def find_asset(filename):
    possible_paths = [
        os.path.join("c:/Solar_Inspection_Project/datn_fe/src/assets", filename),
        os.path.join("../datn_fe/src/assets", filename),
        os.path.join("datn_fe/src/assets", filename),
        os.path.join("app/assets", filename),
        filename
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None

LOGO_PATH = find_asset("epc_solar_logo.png") or "datn_fe/src/assets/epc_solar_logo.png"
AERIAL_PATH = find_asset("solar_farm_aerial.png") or "datn_fe/src/assets/solar_farm_aerial.png"

# ─────────────────────────────────────────
# Các bản dịch tiếng Việt
# ─────────────────────────────────────────
VI_DEFECT_MAP = {
    "hotspot_single_cell": "Điểm nóng đơn bào (hotspot_single_cell)",
    "hotspot_multi_cell":  "Điểm nóng đa bào (hotspot_multi_cell)",
    "shading":             "Bóng che (shading)",
    "soiling":             "Bám bẩn (soiling)",
    "crack":               "Vết nứt (crack)",
}

VI_SEVERITY_MAP = {
    "very_minor": "Rất nhẹ",
    "minor":      "Nhẹ",
    "moderate":   "Cần theo dõi",
    "severe":     "Ưu tiên bảo trì",
    "replace":    "Cần thay thế"
}

VI_LOC_MAP = {
    "top-left": "Góc trên bên trái",
    "top-center": "Phía trên trung tâm",
    "top-middle": "Phía trên trung tâm",
    "top": "Phía trên trung tâm",
    "top-right": "Góc trên bên phải",
    "middle-left": "Giữa bên trái",
    "left": "Giữa bên trái",
    "middle-center": "Chính giữa",
    "center": "Chính giữa",
    "middle": "Chính giữa",
    "middle-right": "Giữa bên phải",
    "right": "Giữa bên phải",
    "bottom-left": "Góc dưới bên trái",
    "bottom-center": "Phía dưới trung tâm",
    "bottom-middle": "Phía dưới trung tâm",
    "bottom": "Phía dưới trung tâm",
    "bottom-right": "Góc dưới bên phải",
}

def translate_location(loc_str):
    if not loc_str:
        return "Chính giữa"
    loc_str = str(loc_str).lower().strip()
    return VI_LOC_MAP.get(loc_str, loc_str.capitalize())

def translate_rec(rec_str):
    if not rec_str:
        return "Kiểm tra"
    rec_str = str(rec_str).lower().strip()
    rec_map = {
        "check": "Kiểm tra",
        "replace": "Cần thay thế",
        "monitor": "Theo dõi",
        "none": "Không cần xử lý",
    }
    return rec_map.get(rec_str, rec_str.capitalize())

def clean_val(val, default="Chưa cập nhật"):
    if val is None:
        return default
    val_str = str(val).strip()
    if not val_str or val_str.lower() in ["none", "null", "undefined", "+không có ghi chú", "không có ghi chú"]:
        return default
    return val_str

def safe(text):
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    return text

def get_wrapped_lines(pdf, text, max_width):
    if not text:
        return []
    paragraphs = text.split("\n")
    all_lines = []
    for para in paragraphs:
        words = para.split(" ")
        current_line = ""
        for word in words:
            if not word:
                continue
            test_line = current_line + " " + word if current_line else word
            if pdf.get_string_width(test_line) <= max_width:
                current_line = test_line
            else:
                if current_line:
                    all_lines.append(current_line)
                current_line = word
        if current_line:
            all_lines.append(current_line)
    return all_lines


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
            self.set_y(15)
            self._f("I", 10)
            self.set_text_color(148, 163, 184)
            self.cell(0, 10, "BÁO CÁO KIỂM TRA ĐIỆN MẶT TRỜI BẰNG AI", align="L")
            
            if LOGO_PATH and os.path.exists(LOGO_PATH):
                self.image(LOGO_PATH, x=160, y=10, w=30)
            
            self.set_y(25)
            self.set_draw_color(226, 232, 240)
            self.line(35, 23, 190, 23)
            self.set_y(30)

    def footer(self):
        if self.page_no() > 1:
            self.set_y(-15)
            self._f("I", 9)
            self.set_text_color(148, 163, 184)
            self.cell(0, 10, f"Trang {self.page_no()}", align="C")


class ReportGenerator:
    @staticmethod
    def _build_pdf_content(pdf, data_list, thermal_to_rgb, metadata_fields, faulty_panels, cropped_images, panel_power, toc_page_numbers=None):
        """
        Dựng layout báo cáo PDF. Hàm này sẽ được chạy 2 lần (Two-pass):
        - Lần 1: Không có toc_page_numbers (để đo vị trí trang thực tế của các phần).
        - Lần 2: Có toc_page_numbers để vẽ trang mục lục với số trang chính xác.
        """
        # Cấu hình lề
        pdf.set_margins(left=35, top=25, right=20)
        pdf.set_auto_page_break(auto=True, margin=25)
        
        # ─────────────────────────────────────────
        # TRANG BÌA (Page 1)
        # ─────────────────────────────────────────
        pdf.add_page()
        if LOGO_PATH and os.path.exists(LOGO_PATH):
            pdf.image(LOGO_PATH, x=77.5, y=30, w=55)
        pdf.set_y(75)

        pdf._f("B", 22)
        pdf.set_text_color(15, 23, 42)
        pdf.cell(0, 12, "BÁO CÁO KIỂM TRA ĐIỆN MẶT TRỜI", new_x="LMARGIN", new_y="NEXT", align="C")

        pdf._f("B", 12)
        pdf.set_text_color(14, 165, 233)
        pdf.cell(0, 8, "PHÂN TÍCH ẢNH NHIỆT UAV BẰNG AI", new_x="LMARGIN", new_y="NEXT", align="C")
        pdf.ln(10)
        
        # Grid thông tin dự án
        pdf.set_xy(35, 100)
        pdf.set_draw_color(226, 232, 240)
        pdf.set_line_width(0.15)
        
        for label, val in metadata_fields:
            cleaned_val = clean_val(val)
            val_lines = get_wrapped_lines(pdf, cleaned_val, 101)
            row_h = max(1, len(val_lines)) * 7.5
            
            y0 = pdf.get_y()
            # Cột nhãn (Label)
            pdf._f("B", 10)
            pdf.set_text_color(71, 85, 105)
            pdf.rect(35, y0, 50, row_h, "D")
            pdf.set_xy(35 + 2, y0 + (row_h - 7.5) / 2)
            pdf.cell(46, 7.5, safe(label), border=0)
            
            # Cột giá trị (Value)
            pdf._f("", 10)
            pdf.set_text_color(15, 23, 42)
            pdf.rect(85, y0, 105, row_h, "D")
            
            for i, line in enumerate(val_lines):
                pdf.set_xy(85 + 2, y0 + i * 7.5)
                pdf.cell(101, 7.5, safe(line), border=0)
                
            pdf.set_xy(35, y0 + row_h)
            
        pdf.set_xy(0, 260)
        pdf._f("I", 10)
        pdf.set_text_color(148, 163, 184)
        pdf.cell(0, 10, "Bảo mật - EPC Solar JSC", align="C", new_x="LMARGIN", new_y="NEXT")

        # ─────────────────────────────────────────
        # TRANG 2: MỤC LỤC
        # ─────────────────────────────────────────
        pdf.add_page()
        pdf.set_xy(35, 25)
        pdf._f("B", 16)
        pdf.set_text_color(15, 23, 42)
        pdf.cell(0, 10, "MỤC LỤC", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(5)
        
        pdf.set_x(35 + 12.7)
        pdf._f("", 14)
        pdf.set_text_color(51, 65, 85)
        pdf.write(7.5, "Mục lục dưới đây tóm tắt cấu trúc chính của báo cáo kết quả kiểm tra nhiệt tấm pin mặt trời để phục vụ công tác tra cứu nhanh.")
        pdf.ln(15)
        
        sec_1_p = toc_page_numbers.get('sec_1', "") if toc_page_numbers else ""
        sec_2_p = toc_page_numbers.get('sec_2', "") if toc_page_numbers else ""
        sec_3_p = toc_page_numbers.get('sec_3', "") if toc_page_numbers else ""
        sec_4_p = toc_page_numbers.get('sec_4', "") if toc_page_numbers else ""
        
        toc_items = [
            ("1. Tóm tắt báo cáo", sec_1_p),
            ("2. Tổng quan kết quả phân tích", sec_2_p),
            ("3. Bản đồ tổng thể và phân bố lỗi", sec_3_p),
            ("4. Danh sách chi tiết các tấm pin lỗi", sec_4_p),
        ]
        
        for title, page_num in toc_items:
            pdf.set_x(35)
            pdf._f("", 14)
            pdf.set_text_color(51, 65, 85)
            pdf.cell(115, 7.5, title)
            
            pdf._f("", 14)
            pdf.set_text_color(148, 163, 184)
            pdf.cell(30, 7.5, " . . . . . . . . . . . . . . . . . . . .", align="R")
            
            pdf._f("B", 14)
            pdf.set_text_color(15, 23, 42)
            page_str = str(page_num) if page_num else ""
            pdf.cell(10, 7.5, page_str, align="R", new_x="LMARGIN", new_y="NEXT")
            pdf.ln(2)

        # ─────────────────────────────────────────
        # TRANG 3: TÓM TẮT & TỔNG QUAN KẾT QUẢ PHÂN TÍCH
        # ─────────────────────────────────────────
        pdf.add_page()
        sec_1_page = pdf.page_no()
        sec_2_page = pdf.page_no()
        
        pdf.set_xy(35, 25)
        pdf._f("B", 16)
        pdf.set_text_color(15, 23, 42)
        pdf.cell(0, 10, "1. Tóm tắt báo cáo", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)
        
        pdf.set_x(35 + 12.7)
        pdf._f("", 14)
        pdf.set_text_color(51, 65, 85)
        pdf.write(7.5, "Báo cáo này trình bày kết quả phân tích tự động hệ thống điện mặt trời bằng công nghệ AI. Dưới đây là các chỉ số thống kê tổng hợp về số lượng tấm pin, tỷ lệ lỗi và mức độ suy hao công suất phát hiện trong đợt kiểm tra.")
        pdf.ln(15)
        
        pdf._f("B", 16)
        pdf.set_text_color(15, 23, 42)
        pdf.cell(0, 10, "2. Tổng quan kết quả phân tích", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(5)
        
        # Tính toán thống kê
        total_panels = len(data_list)
        total_faulty = len(faulty_panels)
        healthy_count = total_panels - total_faulty
        health_rate = round(healthy_count / total_panels * 100, 1) if total_panels > 0 else 100.0
        fault_rate = round(total_faulty / total_panels * 100, 1) if total_panels > 0 else 0.0
        
        total_power_loss_w = 0.0
        stats_defects = {
            "hotspot_single_cell": 0,
            "hotspot_multi_cell": 0,
            "shading": 0,
            "crack": 0
        }
        
        for p, p_detail in faulty_panels:
            total_power_loss_w += p_detail.get("total_panel_loss", 0.0)
            for d in p_detail.get("defects", []):
                cname = d.get("class_name", "")
                cname_lower = cname.lower() if isinstance(cname, str) else str(cname).lower()
                if "hotspot_single" in cname_lower or "single_cell" in cname_lower or "single-cell" in cname_lower:
                    stats_defects["hotspot_single_cell"] += 1
                elif "hotspot_multi" in cname_lower or "multi_cell" in cname_lower or "multicell" in cname_lower or "multi-cell" in cname_lower:
                    stats_defects["hotspot_multi_cell"] += 1
                elif "crack" in cname_lower or "nut" in cname_lower:
                    stats_defects["crack"] += 1
                elif any(k in cname_lower for k in ["shading", "shadow", "shade", "soil", "soiling", "dirt"]):
                    stats_defects["shading"] += 1
                    
        # Vẽ các card thống kê
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
        pdf.cell(63, 5, f"Bình thường: {healthy_count} ({health_rate}%)", new_x="LMARGIN", new_y="NEXT")

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
        pdf.cell(63, 5, f"Tỷ lệ lỗi: {fault_rate}%", new_x="LMARGIN", new_y="NEXT")

        pdf.set_fill_color(248, 250, 252)
        pdf.rect(35, 122, 155, 30, "F")
        pdf.set_xy(40, 126)
        pdf._f("B", 10)
        pdf.set_text_color(100, 116, 139)
        pdf.cell(145, 5, "TỔNG HAO HỤT CÔNG SUẤT DỰ TÍNH", new_x="LMARGIN", new_y="NEXT")
        pdf._f("B", 20)
        pdf.set_text_color(245, 158, 11)
        pdf.set_x(40)
        pdf.cell(145, 12, f"{round(total_power_loss_w / 1000, 2)} kW", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(12)
        
        # Bảng phân phối loại lỗi
        pdf.set_xy(35, 160)
        pdf._f("B", 12)
        pdf.set_text_color(15, 23, 42)
        pdf.cell(0, 10, "Phân loại lỗi phát hiện", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)

        pdf.set_fill_color(15, 23, 42)
        pdf.set_text_color(255, 255, 255)
        pdf._f("B", 10)
        pdf.cell(80, 8.5, " Loại lỗi", border=1, align="L", fill=True)
        pdf.cell(35, 8.5, "Số lượng", border=1, align="C", fill=True)
        pdf.cell(40, 8.5, "Tỷ lệ (%)", border=1, align="C", fill=True, new_x="LMARGIN", new_y="NEXT")

        pdf.set_text_color(15, 23, 42)
        pdf._f("", 10)

        defect_mapping_list = [
            ("hotspot_single_cell", "Điểm nóng đơn bào (hotspot_single_cell)", stats_defects.get("hotspot_single_cell", 0)),
            ("hotspot_multi_cell",  "Điểm nóng đa bào (hotspot_multi_cell)",  stats_defects.get("hotspot_multi_cell", 0)),
            ("crack",               "Vết nứt (crack)",               stats_defects.get("crack", 0)),
            ("shading",             "Bóng che (shading)",             stats_defects.get("shading", 0)),
        ]

        sum_defects = sum(stats_defects.values())

        for idx_row, (key, label_name, count) in enumerate(defect_mapping_list):
            if idx_row % 2 == 0:
                pdf.set_fill_color(248, 250, 252)
            else:
                pdf.set_fill_color(255, 255, 255)
            ratio = round(count / sum_defects * 100, 1) if sum_defects > 0 else 0.0
            pdf.cell(80, 8.5, f" {label_name}", border=1, align="L", fill=True)
            pdf.cell(35, 8.5, str(count), border=1, align="C", fill=True)
            pdf.cell(40, 8.5, f"{ratio}%", border=1, align="C", fill=True, new_x="LMARGIN", new_y="NEXT")

        # ─────────────────────────────────────────
        # TRANG 4: BẢN ĐỒ TỔNG THỂ VÀ PHÂN BỐ LỖI
        # ─────────────────────────────────────────
        pdf.add_page()
        sec_3_page = pdf.page_no()
        
        pdf.set_xy(35, 25)
        pdf._f("B", 16)
        pdf.set_text_color(15, 23, 42)
        pdf.cell(0, 10, "3. Bản đồ tổng thể và phân bố lỗi", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)
        
        pdf.set_x(35 + 12.7)
        pdf._f("", 14)
        pdf.set_text_color(51, 65, 85)
        pdf.write(7.5, "Dưới đây là hình ảnh bản đồ nhiệt toàn cảnh của khu vực quét điện mặt trời. Bản đồ cung cấp cái nhìn tổng quan về vị trí phân bố các tấm pin lỗi phục vụ hiệu quả cho công tác lập kế hoạch O&M.")
        pdf.ln(12)
        
        if AERIAL_PATH and os.path.exists(AERIAL_PATH):
            y_img = pdf.get_y()
            pdf.image(AERIAL_PATH, x=35, y=y_img, w=155)
            
            try:
                temp_img = cv2.imread(AERIAL_PATH)
                if temp_img is not None:
                    h_i, w_i = temp_img.shape[:2]
                    img_aspect = h_i / w_i
                    scaled_h = 155 * img_aspect
                    
                    pdf.set_xy(35, y_img + scaled_h + 3)
                    pdf._f("I", 9)
                    pdf.set_text_color(71, 85, 105)
                    pdf.cell(155, 5, "Hình 3.1: Bản đồ nhiệt toàn cảnh hệ thống pin mặt trời", align="C", new_x="LMARGIN", new_y="NEXT")
                    pdf.ln(5)
            except Exception:
                pdf.ln(90)
        else:
            pdf._f("I", 10)
            pdf.set_text_color(239, 68, 68)
            pdf.cell(155, 10, "[Không tìm thấy ảnh bản đồ toàn cảnh]", align="C", new_x="LMARGIN", new_y="NEXT")
            pdf.ln(5)

        # ─────────────────────────────────────────
        # TRANG 5+: DANH SÁCH CHI TIẾT CÁC TẤM PIN LỖI
        # ─────────────────────────────────────────
        pdf.add_page()
        sec_4_page = pdf.page_no()
        
        pdf.set_xy(35, 25)
        pdf._f("B", 16)
        pdf.set_text_color(15, 23, 42)
        pdf.cell(0, 10, "4. Danh sách chi tiết các tấm pin lỗi", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)
        
        pdf.set_x(35 + 12.7)
        pdf._f("", 14)
        pdf.set_text_color(51, 65, 85)
        pdf.write(7.5, "Phần dưới đây trình bày thông tin chi tiết từng tấm pin bị lỗi, bao gồm các thông số ghi nhận O&M, hình ảnh nhiệt phóng to được định vị làm nổi bật lỗi tương ứng, ảnh quang học bối cảnh và bảng dữ liệu chi tiết của lỗi.")
        pdf.ln(12)

        def draw_defect_row_local(pdf, columns_data):
            widths = [35, 40, 25, 30, 25]
            cell_lines = []
            for txt, w in zip(columns_data, widths):
                cell_lines.append(get_wrapped_lines(pdf, str(txt), w - 4))
                
            max_lines = max(len(lines) for lines in cell_lines)
            line_h = 5
            row_h = max_lines * line_h + 4
            
            if pdf.get_y() + row_h > 265:
                pdf.add_page()
                
            y0 = pdf.get_y()
            x0 = pdf.l_margin
            
            pdf.set_draw_color(226, 232, 240)
            pdf.set_line_width(0.15)
            
            x = x0
            pdf._f("", 9)
            pdf.set_text_color(15, 23, 42)
            for idx, lines in enumerate(cell_lines):
                w = widths[idx]
                pdf.rect(x, y0, w, row_h)
                
                y_text = y0 + 2 + (row_h - 4 - len(lines) * line_h) / 2
                for line_idx, line in enumerate(lines):
                    pdf.set_xy(x + 2, y_text + line_idx * line_h)
                    pdf.cell(w - 4, line_h, safe(line), border=0, align="C")
                x += w
                
            pdf.set_xy(x0, y0 + row_h)

        for idx, (p, p_detail) in enumerate(faulty_panels):
            defects = p_detail.get("defects", [])
            if not defects:
                continue
                
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
            gps_str = f"{lat}, {lng}"
            
            if pdf.get_y() > 210:
                pdf.add_page()
                
            # Thanh tiêu đề block tấm pin
            pdf.set_fill_color(241, 245, 249)
            pdf.set_draw_color(203, 213, 225)
            pdf.set_line_width(0.15)
            pdf._f("B", 12)
            pdf.set_text_color(15, 23, 42)
            pdf.cell(0, 8, f" Tấm pin lỗi: {local_id} (Hàng {row}, Cột {col}) | GPS: {gps_str}", border=1, fill=True, new_x="LMARGIN", new_y="NEXT")
            pdf.ln(2)
            
            # Khối thông tin O&M duyệt lỗi
            status_val = p_detail.get("review_status", "unreviewed")
            status_label = {
                "unreviewed": "Chưa duyệt",
                "confirmed_defect": "Đã xác nhận",
                "needs_review": "Cần kiểm tra lại",
                "false_positive": "Bỏ qua"
            }.get(status_val, "Chưa duyệt")

            priority_val = p_detail.get("maintenance_priority", "medium")
            priority_label = {
                "low": "Thấp",
                "medium": "Trung bình",
                "high": "Cao",
                "urgent": "Khẩn cấp"
            }.get(priority_val, "Trung bình")

            reviewer = clean_val(p_detail.get("reviewer_name"))
            reviewed_at = clean_val(p_detail.get("reviewed_at"))
            notes = clean_val(p_detail.get("review_note"), "Không có ghi chú")
            
            pdf._f("B", 10)
            pdf.set_text_color(71, 85, 105)
            pdf.cell(45, 5, "Trạng thái duyệt:")
            pdf._f("", 10)
            pdf.set_text_color(15, 23, 42)
            pdf.cell(40, 5, status_label)
            
            pdf._f("B", 10)
            pdf.set_text_color(71, 85, 105)
            pdf.cell(30, 5, "Người duyệt:")
            pdf._f("", 10)
            pdf.set_text_color(15, 23, 42)
            pdf.cell(40, 5, reviewer, new_x="LMARGIN", new_y="NEXT")
            
            pdf._f("B", 10)
            pdf.set_text_color(71, 85, 105)
            pdf.cell(45, 5, "Ưu tiên xử lý:")
            pdf._f("", 10)
            pdf.set_text_color(15, 23, 42)
            pdf.cell(40, 5, priority_label)
            
            pdf._f("B", 10)
            pdf.set_text_color(71, 85, 105)
            pdf.cell(30, 5, "Thời gian duyệt:")
            pdf._f("", 10)
            pdf.set_text_color(15, 23, 42)
            pdf.cell(40, 5, reviewed_at, new_x="LMARGIN", new_y="NEXT")
            
            pdf._f("B", 10)
            pdf.set_text_color(71, 85, 105)
            pdf.cell(45, 5, "Ghi chú kỹ sư:")
            pdf._f("", 10)
            pdf.set_text_color(51, 65, 85)
            pdf.multi_cell(110, 5, notes, align="L")
            pdf.ln(3)
            
            # Chi tiết từng lỗi trong tấm pin
            for d_idx, d in enumerate(defects):
                if pdf.get_y() > 210:
                    pdf.add_page()
                    
                if len(defects) > 1:
                    pdf._f("B", 11)
                    pdf.set_text_color(14, 165, 233)
                    pdf.cell(0, 6, f" Chi tiết lỗi #{d_idx + 1}: {VI_DEFECT_MAP.get(d.get('class_name'), d.get('class_name'))}", new_x="LMARGIN", new_y="NEXT")
                    pdf.ln(1)
                    
                defect_crops = cropped_images.get(idx, {}).get(d_idx, {})
                temp_thermal_path = defect_crops.get("thermal", "")
                temp_rgb_path = defect_crops.get("rgb", "")
                crop_success = os.path.exists(temp_thermal_path) and os.path.exists(temp_rgb_path)

                image_y = pdf.get_y() + 1
                if crop_success:
                    IMG_W = 73
                    pdf.image(temp_thermal_path, x=35, y=image_y, w=IMG_W, h=30)
                    pdf.image(temp_rgb_path, x=117, y=image_y, w=IMG_W, h=30)
                    
                    pdf.set_draw_color(226, 232, 240)
                    pdf.set_line_width(0.15)
                    
                    # Chú thích ảnh nhiệt
                    pdf.rect(35, image_y + 30.5, IMG_W, 5, style='D')
                    pdf.set_xy(35, image_y + 30.5)
                    pdf._f("I", 8)
                    pdf.set_text_color(71, 85, 105)
                    pdf.cell(IMG_W, 4, "Ảnh nhiệt phóng to vùng lỗi", align="C")
                    
                    # Chú thích ảnh RGB
                    pdf.rect(117, image_y + 30.5, IMG_W, 5, style='D')
                    pdf.set_xy(117, image_y + 30.5)
                    pdf.cell(IMG_W, 4, "Ảnh quang học bối cảnh", align="C", new_x="LMARGIN", new_y="NEXT")
                    
                    pdf.set_y(image_y + 37)
                else:
                    pdf.ln(2)
                    
                # Bảng thông số chi tiết lỗi
                pdf.ln(1)
                pdf.set_xy(35, pdf.get_y())
                pdf._f("B", 10)
                pdf.set_fill_color(30, 41, 59)
                pdf.set_text_color(255, 255, 255)
                pdf.set_draw_color(226, 232, 240)
                pdf.set_line_width(0.15)
                
                pdf.cell(35, 8.5, "Loại lỗi", 1, 0, "C", fill=True)
                pdf.cell(40, 8.5, "Mức độ & Khuyến nghị", 1, 0, "C", fill=True)
                pdf.cell(25, 8.5, "Hao hụt", 1, 0, "C", fill=True)
                pdf.cell(30, 8.5, "Tỷ lệ diện tích", 1, 0, "C", fill=True)
                pdf.cell(25, 8.5, "Vị trí", 1, 1, "C", fill=True)
                
                raw_cname = d.get("class_name") or d.get("type", "")
                defect_type_str = VI_DEFECT_MAP.get(raw_cname, raw_cname)
                
                severity_str = VI_SEVERITY_MAP.get(d.get("severity", "minor"), "Nhẹ")
                rec_str = translate_rec(d.get("recommendation", "Kiểm tra"))
                severity_rec_str = f"{severity_str}\n({rec_str})"
                
                loss_val = p_detail.get("total_panel_loss", 0.0)
                loss_pct_calc = round((loss_val / panel_power) * 100, 1) if panel_power > 0 else 0.0
                power_loss_str = f"{loss_val:.1f} W\n({loss_pct_calc:.1f}%)"
                
                area_ratio = d.get("area_ratio_percent")
                if area_ratio is not None:
                    area_ratio_str = f"{round(area_ratio, 2)}% diện tích\ntấm pin"
                else:
                    area_ratio_str = "Chưa cập nhật"
                    
                loc_val = d.get("location_in_panel", "center")
                location_str = translate_location(loc_val)
                
                draw_defect_row_local(
                    pdf,
                    [
                        defect_type_str,
                        severity_rec_str,
                        power_loss_str,
                        area_ratio_str,
                        location_str
                    ]
                )
                pdf.ln(5)
                
        return sec_1_page, sec_2_page, sec_3_page, sec_4_page

    @staticmethod
    def generate_inspection_report(batch_id, data_list, output_path):
        """
        Khởi tạo tiến trình tạo báo cáo qua hai bước (Two-pass Rendering).
        """
        # Khớp ảnh Thermal và RGB
        pairs = RegistrationService.match_thermal_rgb("data/raw")
        thermal_to_rgb = {p['thermal']: p['rgb'] for p in pairs}

        batch = None
        if data_list:
            try:
                batch = data_list[0].image.batch
            except Exception:
                pass

        scope_str = batch.scope if (batch and batch.scope) else "Chưa xác định"
        system_capacity = "Chưa xác định"
        supervisor = "Chưa xác định"
        notes = "Không có ghi chú"
        data_type = "UAV thermal image"
        ai_model = "YOLOv8-Solar-M300"
        system_version = "O&M Suite v2.4"

        if scope_str and scope_str.startswith("{"):
            try:
                scope_data = json.loads(scope_str)
                scope_str = scope_data.get("s", "Chưa xác định") or "Chưa xác định"
                system_capacity = scope_data.get("sc", "Chưa xác định") or "Chưa xác định"
                supervisor = scope_data.get("sv", "Chưa xác định") or "Chưa xác định"
                notes = scope_data.get("nt", "Không có ghi chú") or "Không có ghi chú"
                data_type = scope_data.get("dt", "UAV thermal image") or "UAV thermal image"
                ai_model = scope_data.get("am", "YOLOv8-Solar-M300") or "YOLOv8-Solar-M300"
                system_version = scope_data.get("sys", "O&M Suite v2.4") or "O&M Suite v2.4"
            except Exception:
                pass

        project_name = batch.project_name if (batch and batch.project_name) else "Dự án kiểm tra điện mặt trời"
        location     = batch.location    if (batch and batch.location)     else "Chưa xác định"
        scan_time    = batch.scan_time   if (batch and batch.scan_time)    else datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        operator     = batch.operator    if (batch and batch.operator)     else "Đội ngũ kỹ thuật vận hành EPC Solar"
        device       = batch.device      if (batch and batch.device)       else "DJI Matrice 300 RTK + Zenmuse H20T"
        panel_power  = batch.panel_power if (batch and batch.panel_power)  else 600.0

        # Phân loại tấm pin lỗi
        faulty_panels = []
        for p in data_list:
            panel_detail = {}
            is_faulty = False

            if p.defect_type:
                if p.defect_type.startswith("{"):
                    try:
                        panel_detail = json.loads(p.defect_type)
                        review_status = panel_detail.get("review_status", "unreviewed")
                        if review_status == "false_positive":
                            include_in_report = False
                        elif review_status in ["confirmed_defect", "needs_review", "unreviewed"]:
                            include_in_report = True
                        else:
                            include_in_report = panel_detail.get("include_in_report", True)

                        if review_status == "false_positive" or include_in_report is False:
                            is_faulty = False
                        else:
                            is_faulty = len(panel_detail.get("defects", [])) > 0
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

        # Thư mục tạm lưu ảnh đã cắt để chèn vào PDF
        temp_dir = tempfile.mkdtemp()
        
        # ─────────────────────────────────────────
        # Bước tiền xử lý: Cắt toàn bộ ảnh nhiệt zoom lỗi & ảnh RGB bối cảnh
        # ─────────────────────────────────────────
        cropped_images = {}
        for idx, (p, p_detail) in enumerate(faulty_panels):
            defects = p_detail.get("defects", [])
            local_id = p.panel.local_id
            
            thermal_filename = p.image.filename
            rgb_filename = thermal_to_rgb.get(thermal_filename)
            thermal_path = os.path.join("data/results", thermal_filename)
            rgb_path = os.path.join("data/raw", rgb_filename) if rgb_filename else None
            
            cropped_images[idx] = {}
            for d_idx, d in enumerate(defects):
                d_bbox = d.get("bbox") or d.get("box", [])
                d_poly = d.get("polygon", [])
                
                temp_thermal_path = ""
                temp_rgb_path = ""
                crop_success = False
                
                if os.path.exists(thermal_path):
                    try:
                        t_img = cv2.imread(thermal_path)
                        if t_img is not None:
                            h_t, w_t = t_img.shape[:2]
                            
                            # Xác định bbox của lỗi
                            if d_bbox and len(d_bbox) == 4:
                                dx1, dy1, dx2, dy2 = [int(v) for v in d_bbox]
                            elif d_poly and len(d_poly) >= 3:
                                xs = [int(pt[0]) for pt in d_poly]
                                ys = [int(pt[1]) for pt in d_poly]
                                dx1, dy1, dx2, dy2 = min(xs), min(ys), max(xs), max(ys)
                            else:
                                p_bbox = p_detail.get("bbox", [])
                                if p_bbox and len(p_bbox) == 4:
                                    dx1, dy1, dx2, dy2 = [int(v) for v in p_bbox]
                                else:
                                    dx1, dy1, dx2, dy2 = 0, 0, w_t, h_t
                                    
                            dw = dx2 - dx1
                            dh = dy2 - dy1
                            dcx = (dx1 + dx2) // 2
                            dcy = (dy1 + dy2) // 2
                            
                            # Tính kích thước crop (3 lần kích thước lỗi, tối thiểu 150x120)
                            crop_w = int(max(dw * 3, 150))
                            crop_h = int(max(dh * 3, 120))
                            crop_w = min(crop_w, w_t)
                            crop_h = min(crop_h, h_t)
                            
                            x1_c = max(0, dcx - crop_w // 2)
                            y1_c = max(0, dcy - crop_h // 2)
                            x2_c = min(w_t, x1_c + crop_w)
                            y2_c = min(h_t, y1_c + crop_h)
                            
                            if x2_c - x1_c < crop_w:
                                x1_c = max(0, x2_c - crop_w)
                            if y2_c - y1_c < crop_h:
                                y1_c = max(0, y2_c - crop_h)
                                
                            t_drawn = t_img.copy()
                            sev = d.get("severity", "minor").lower()
                            color_map = {
                                "very_minor": (0, 255, 255),
                                "minor": (0, 255, 255),
                                "moderate": (0, 165, 255),
                                "severe": (0, 0, 255),
                                "replace": (0, 0, 180)
                            }
                            bgr_color = color_map.get(sev, (0, 0, 255))
                            
                            # Vẽ viền màu nổi bật lỗi
                            if d_poly and len(d_poly) >= 3:
                                pts = np.array(d_poly, dtype=np.int32)
                                cv2.polylines(t_drawn, [pts], isClosed=True, color=bgr_color, thickness=2)
                            else:
                                cv2.rectangle(t_drawn, (dx1, dy1), (dx2, dy2), bgr_color, 2)
                                
                            crop_t = t_drawn[y1_c:y2_c, x1_c:x2_c]
                            if crop_t is not None and crop_t.size > 0:
                                temp_thermal_path = os.path.join(temp_dir, f"t_{local_id}_{idx}_{d_idx}.jpg")
                                cv2.imwrite(temp_thermal_path, crop_t)
                                
                            # Cắt ảnh RGB bối cảnh (tâm panel, có viền xanh lá)
                            p_bbox = p_detail.get("bbox", [])
                            if p_bbox and len(p_bbox) == 4 and rgb_path and os.path.exists(rgb_path):
                                r_img = cv2.imread(rgb_path)
                                if r_img is not None:
                                    h_r, w_r = r_img.shape[:2]
                                    px1, py1, px2, py2 = [int(v) for v in p_bbox]
                                    pcx = (px1 + px2) // 2
                                    pcy = (py1 + py2) // 2
                                    
                                    scale_x = w_r / w_t
                                    scale_y = h_r / h_t
                                    
                                    crop_w_r = int(w_t * 0.5 * scale_x)
                                    crop_h_r = int(h_t * 0.5 * scale_y)
                                    
                                    x1_r = max(0, int(pcx * scale_x) - crop_w_r // 2)
                                    y1_r = max(0, int(pcy * scale_y) - crop_h_r // 2)
                                    x2_r = min(w_r, x1_r + crop_w_r)
                                    y2_r = min(h_r, y1_r + crop_h_r)
                                    
                                    r_drawn = r_img.copy()
                                    rx1, ry1 = int(px1 * scale_x), int(py1 * scale_y)
                                    rx2, ry2 = int(px2 * scale_x), int(py2 * scale_y)
                                    cv2.rectangle(r_drawn, (rx1, ry1), (rx2, ry2), (0, 255, 0), 2)
                                    
                                    crop_r = r_drawn[y1_r:y2_r, x1_r:x2_r]
                                    if crop_r is not None and crop_r.size > 0:
                                        temp_rgb_path = os.path.join(temp_dir, f"r_{local_id}_{idx}_{d_idx}.jpg")
                                        cv2.imwrite(temp_rgb_path, crop_r)
                                        
                            crop_success = os.path.exists(temp_thermal_path) and os.path.exists(temp_rgb_path)
                    except Exception as e:
                        logger.warning(f"Error cropping for defect {d_idx} on panel {local_id}: {e}")
                        crop_success = False
                        
                if crop_success:
                    cropped_images[idx][d_idx] = {
                        "thermal": temp_thermal_path,
                        "rgb": temp_rgb_path
                    }

        metadata_fields = [
            ("Tên dự án:",       project_name),
            ("Vị trí dự án:",    location),
            ("Thời gian quét:",  scan_time),
            ("Đơn vị thực hiện:", operator),
            ("Thiết bị bay:",    device),
            ("Phạm vi quét:",    scope_str),
            ("Công suất tấm pin:", f"{panel_power} W"),
            ("Công suất hệ thống:", system_capacity),
            ("Người phụ trách:", supervisor),
            ("Loại dữ liệu:",    data_type),
            ("Model AI sử dụng:", ai_model),
            ("Phiên bản hệ thống:", system_version),
            ("Ghi chú:",         notes),
        ]

        # ─────────────────────────────────────────
        # PASS 1: Đo số trang thực tế của các phần
        # ─────────────────────────────────────────
        pdf1 = CustomPDF()
        sec_1, sec_2, sec_3, sec_4 = ReportGenerator._build_pdf_content(
            pdf1, data_list, thermal_to_rgb, metadata_fields, faulty_panels, cropped_images, panel_power, toc_page_numbers=None
        )
        
        toc_page_numbers = {
            'sec_1': sec_1,
            'sec_2': sec_2,
            'sec_3': sec_3,
            'sec_4': sec_4,
        }

        # ─────────────────────────────────────────
        # PASS 2: Xuất file PDF chính thức với số trang chuẩn
        # ─────────────────────────────────────────
        pdf2 = CustomPDF()
        ReportGenerator._build_pdf_content(
            pdf2, data_list, thermal_to_rgb, metadata_fields, faulty_panels, cropped_images, panel_power, toc_page_numbers=toc_page_numbers
        )

        pdf2.output(output_path)

        # Dọn dẹp thư mục tạm
        try:
            shutil.rmtree(temp_dir)
        except Exception:
            pass

        return output_path
