# app/services/report_generator.py
import os
import cv2
import json
import re
import datetime
from fpdf import FPDF
from app.services.registration import RegistrationService

def strip_accents(text):
    """
    Loại bỏ dấu tiếng Việt để tránh lỗi UnicodeEncodeError trong FPDF.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    
    dic = {
        'à': 'a', 'á': 'a', 'ả': 'a', 'ã': 'a', 'ạ': 'a',
        'ă': 'a', 'ằ': 'a', 'ắ': 'a', 'ẳ': 'a', 'ẵ': 'a', 'ặ': 'a',
        'â': 'a', 'ầ': 'a', 'ấ': 'a', 'ẩ': 'a', 'ẫ': 'a', 'ậ': 'a',
        'è': 'e', 'é': 'e', 'ẻ': 'e', 'ẽ': 'e', 'ẹ': 'e',
        'ê': 'e', 'ề': 'e', 'ế': 'e', 'ể': 'e', 'ễ': 'e', 'ệ': 'e',
        'ì': 'i', 'í': 'i', 'ỉ': 'i', 'ĩ': 'i', 'ị': 'i',
        'ò': 'o', 'ó': 'o', 'ỏ': 'o', 'õ': 'o', 'ọ': 'o',
        'ô': 'o', 'ồ': 'o', 'ố': 'o', 'ổ': 'o', 'ỗ': 'o', 'ộ': 'o',
        'ơ': 'o', 'ờ': 'o', 'ớ': 'o', 'ở': 'o', 'ỡ': 'o', 'ợ': 'o',
        'ù': 'u', 'ú': 'u', 'ủ': 'u', 'ũ': 'u', 'ụ': 'u',
        'ư': 'u', 'ừ': 'u', 'ứ': 'u', 'ử': 'u', 'ữ': 'u', 'ự': 'u',
        'ỳ': 'y', 'ý': 'y', 'ỷ': 'y', 'ỹ': 'y', 'ỵ': 'y',
        'đ': 'd',
        'À': 'A', 'Á': 'A', 'Ả': 'A', 'Ã': 'A', 'Ạ': 'A',
        'Ă': 'A', 'Ằ': 'A', 'Ắ': 'A', 'Ẳ': 'A', 'Ẵ': 'A', 'Ặ': 'A',
        'Â': 'A', 'Ầ': 'A', 'Ấ': 'A', 'Ẩ': 'A', 'Ẫ': 'A', 'Ậ': 'A',
        'È': 'E', 'É': 'E', 'Ẻ': 'E', 'Ẽ': 'E', 'Ẹ': 'E',
        'Ê': 'E', 'Ề': 'E', 'Ế': 'E', 'Ể': 'E', 'Ễ': 'E', 'Ệ': 'E',
        'Ì': 'I', 'Í': 'I', 'Ỉ': 'I', 'Ĩ': 'I', 'Ị': 'I',
        'Ò': 'O', 'Ó': 'O', 'Ỏ': 'O', 'Õ': 'O', 'Ọ': 'O',
        'Ô': 'O', 'Ồ': 'O', 'Ố': 'O', 'Ổ': 'O', 'Ỗ': 'O', 'Ộ': 'O',
        'Ơ': 'O', 'Ờ': 'O', 'Ớ': 'O', 'Ở': 'O', 'Ỡ': 'O', 'Ợ': 'O',
        'Ù': 'U', 'Ú': 'U', 'Ủ': 'U', 'Ũ': 'U', 'Ụ': 'U',
        'Ư': 'U', 'Ừ': 'U', 'Ứ': 'U', 'Ử': 'U', 'Ữ': 'U', 'Ự': 'U',
        'Ỳ': 'Y', 'Ý': 'Y', 'Ỷ': 'Y', 'Ỹ': 'Y', 'Ỵ': 'Y',
        'Đ': 'D'
    }
    
    res = []
    for c in text:
        res.append(dic.get(c, c))
    return "".join(res)


class CustomPDF(FPDF):
    def header(self):
        if self.page_no() > 1:
            # Logo công ty ở góc trên bên phải
            logo_path = "data/epc_solar.png"
            if os.path.exists(logo_path):
                self.image(logo_path, x=165, y=6, w=35)
            
            self.set_font("helvetica", "I", 8)
            self.set_text_color(148, 163, 184) # Slate-400
            self.cell(0, 5, strip_accents("SOLAR AI PV INSPECTION REPORT"), ln=False, align="L")
            self.ln(6)
            # vẽ một đường kẻ xám tinh tế
            self.set_draw_color(226, 232, 240) # Slate-200
            self.line(10, 14, 200, 14)
            self.ln(4)

    def footer(self):
        if self.page_no() > 1:
            self.set_y(-15)
            self.set_font("helvetica", "I", 8)
            self.set_text_color(148, 163, 184) # Slate-400
            self.cell(0, 10, strip_accents(f"Page {self.page_no()}"), align="C")


class ReportGenerator:
    @staticmethod
    def generate_inspection_report(batch_id, data_list, output_path):
        """
        Tạo báo cáo kiểm tra tự động chuyên nghiệp dưới dạng PDF.
        Hỗ trợ kẹp song song ảnh nhiệt gán lỗi (annotated thermal) và ảnh quang học (RGB).
        """
        # Sắp xếp các cặp Thermal-RGB
        pairs = RegistrationService.match_thermal_rgb("data/raw")
        thermal_to_rgb = {p['thermal']: p['rgb'] for p in pairs}

        # Trích xuất thông tin đợt kiểm tra từ database
        batch = None
        if data_list:
            try:
                batch = data_list[0].image.batch
            except Exception:
                pass

        project_name = batch.project_name if (batch and batch.project_name) else "Solar PV Inspection Project"
        location = batch.location if (batch and batch.location) else "Unspecified Location"
        scan_time = batch.scan_time if (batch and batch.scan_time) else datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        operator = batch.operator if (batch and batch.operator) else "EPC Solar Operational Engineering Team"
        device = batch.device if (batch and batch.device) else "DJI Matrice 300 RTK + Zenmuse H20T"
        scope = batch.scope if (batch and batch.scope) else "Unspecified Scope"
        panel_power = batch.panel_power if (batch and batch.panel_power) else 600.0

        # Tạo PDF
        pdf = CustomPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        
        # ─────────────────────────────────────────
        # TRANG BÌA (COVER PAGE)
        # ─────────────────────────────────────────
        pdf.add_page()
        
        # Logo lớn ở trang bìa
        logo_path = "data/epc_solar.png"
        if os.path.exists(logo_path):
            pdf.image(logo_path, x=85, y=18, w=40)
            pdf.ln(30)
        else:
            pdf.ln(15)

        # Tiêu đề chính
        pdf.set_font("helvetica", "B", 22)
        pdf.set_text_color(15, 23, 42) # Slate-900
        pdf.cell(0, 12, strip_accents("SOLAR AI INSPECTION REPORT"), ln=True, align="C")
        
        pdf.set_font("helvetica", "B", 12)
        pdf.set_text_color(14, 165, 233) # Sky-500
        pdf.cell(0, 8, strip_accents("AUTOMATED PV ANOMALY DETECTION & DIAGNOSIS"), ln=True, align="C")
        pdf.ln(10)
        
        # Khung viền thông tin đợt kiểm tra
        pdf.set_fill_color(248, 250, 252) # Slate-50
        pdf.set_draw_color(226, 232, 240) # Slate-200
        pdf.rect(20, 78, 170, 95, "DF")
        
        metadata_fields = [
            ("Project Name:", project_name),
            ("Location:", location),
            ("Scan Date/Time:", scan_time),
            ("Scan Agency:", operator),
            ("Drone Device:", device),
            ("Scan Scope:", scope),
            ("Panel Capacity:", f"{panel_power} W"),
        ]
        
        y_offset = 83
        for label, val in metadata_fields:
            pdf.set_xy(30, y_offset)
            pdf.set_font("helvetica", "B", 10)
            pdf.set_text_color(71, 85, 105) # Slate-600
            pdf.cell(45, 7, strip_accents(label))
            pdf.set_font("helvetica", "", 10)
            pdf.set_text_color(15, 23, 42) # Slate-900
            pdf.cell(110, 7, strip_accents(str(val)), ln=True)
            y_offset += 10
            
        # Footer trang bìa
        pdf.set_xy(0, 245)
        pdf.set_font("helvetica", "I", 10)
        pdf.set_text_color(148, 163, 184) # Slate-400
        pdf.cell(0, 10, strip_accents("Confidential - EPC Solar JSC"), align="C", ln=True)
        
        # ─────────────────────────────────────────
        # TRANG 2: TÓM TẮT & THỐNG KÊ CHI TIẾT
        # ─────────────────────────────────────────
        pdf.add_page()
        
        # Header trang
        pdf.set_font("helvetica", "B", 16)
        pdf.set_text_color(15, 23, 42)
        pdf.cell(0, 10, strip_accents("1. Executive Summary"), ln=True)
        pdf.ln(5)
        
        # Tính toán thống kê
        total_panels = len(data_list)
        
        # Đọc dữ liệu chi tiết từ JSON
        faulty_panels = []
        healthy_count = 0
        total_power_loss_w = 0.0
        
        # Phân loại 5 loại lỗi
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
                    except:
                        pass
                elif p.defect_type != "Healthy":
                    # Tương thích định dạng lỗi kiểu cũ dạng chuỗi đơn giản
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
                # Đếm các loại lỗi
                defects = panel_detail.get("defects", [])
                for d in defects:
                    cname = d.get("class_name", "")
                    if cname in stats_defects:
                        stats_defects[cname] += 1
            else:
                healthy_count += 1
                    
        total_faulty = len(faulty_panels)
        health_rate = round((total_panels - total_faulty) / total_panels * 100, 1) if total_panels > 0 else 100.0
        
        # Grid thống kê tóm tắt
        pdf.set_fill_color(248, 250, 252)
        pdf.rect(10, 25, 90, 45, "F")
        pdf.set_xy(15, 30)
        pdf.set_font("helvetica", "B", 10)
        pdf.set_text_color(100, 116, 139)
        pdf.cell(80, 5, strip_accents("TOTAL SCANNED PANELS"), ln=True)
        pdf.set_font("helvetica", "B", 20)
        pdf.set_text_color(14, 165, 233)
        pdf.set_x(15)
        pdf.cell(80, 12, strip_accents(f"{total_panels}"), ln=True)
        pdf.set_font("helvetica", "", 10)
        pdf.set_text_color(71, 85, 105)
        pdf.set_x(15)
        pdf.cell(80, 5, strip_accents(f"Healthy: {total_panels - total_faulty} ({health_rate}%)"), ln=True)

        pdf.set_fill_color(248, 250, 252)
        pdf.rect(110, 25, 90, 45, "F")
        pdf.set_xy(115, 30)
        pdf.set_font("helvetica", "B", 10)
        pdf.set_text_color(100, 116, 139)
        pdf.cell(80, 5, strip_accents("ANOMALOUS / FAULTY PANELS"), ln=True)
        pdf.set_font("helvetica", "B", 20)
        pdf.set_text_color(239, 68, 68)
        pdf.set_x(115)
        pdf.cell(80, 12, strip_accents(f"{total_faulty}"), ln=True)
        pdf.set_font("helvetica", "", 10)
        pdf.set_text_color(71, 85, 105)
        pdf.set_x(115)
        pdf.cell(80, 5, strip_accents(f"Fault Rate: {round(total_faulty / total_panels * 100, 1) if total_panels > 0 else 0.0}%"), ln=True)

        pdf.set_fill_color(248, 250, 252)
        pdf.rect(10, 75, 90, 45, "F")
        pdf.set_xy(15, 80)
        pdf.set_font("helvetica", "B", 10)
        pdf.set_text_color(100, 116, 139)
        pdf.cell(80, 5, strip_accents("TOTAL ESTIMATED POWER LOSS"), ln=True)
        pdf.set_font("helvetica", "B", 20)
        pdf.set_text_color(245, 158, 11)
        pdf.set_x(15)
        pdf.cell(80, 12, strip_accents(f"{round(total_power_loss_w / 1000, 2)} kW"), ln=True)
        pdf.set_font("helvetica", "", 10)
        pdf.set_text_color(71, 85, 105)
        pdf.set_x(15)
        pdf.cell(80, 5, strip_accents(f"Average Loss: {round(total_power_loss_w / total_faulty, 1) if total_faulty > 0 else 0.0} W / fault"), ln=True)

        pdf.set_fill_color(248, 250, 252)
        pdf.rect(110, 75, 90, 45, "F")
        pdf.set_xy(115, 80)
        pdf.set_font("helvetica", "B", 10)
        pdf.set_text_color(100, 116, 139)
        pdf.cell(80, 5, strip_accents("OVERALL SYSTEM HEALTH TIER"), ln=True)
        pdf.set_font("helvetica", "B", 20)
        
        # Xếp hạng sức khỏe
        if health_rate >= 95:
            tier, color = "Tier A (Excellent)", (34, 197, 94)
        elif health_rate >= 85:
            tier, color = "Tier B (Good)", (16, 185, 129)
        elif health_rate >= 75:
            tier, color = "Tier C (Moderate)", (245, 158, 11)
        else:
            tier, color = "Tier D (Poor)", (239, 68, 68)
            
        pdf.set_text_color(*color)
        pdf.set_x(115)
        pdf.cell(80, 12, strip_accents(tier), ln=True)
        pdf.set_font("helvetica", "", 10)
        pdf.set_text_color(71, 85, 105)
        pdf.set_x(115)
        pdf.cell(80, 5, strip_accents("Based on total healthy panel ratio"), ln=True)
        
        pdf.set_xy(10, 130)
        pdf.ln(10)
        
        # Phân loại lỗi
        pdf.set_font("helvetica", "B", 14)
        pdf.set_text_color(15, 23, 42)
        pdf.cell(0, 10, strip_accents("2. Defect Type Distribution Analysis"), ln=True)
        pdf.ln(3)
        
        # Bảng phân loại
        pdf.set_fill_color(15, 23, 42)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("helvetica", "B", 10)
        pdf.cell(80, 8, strip_accents("Defect Class"), 1, 0, "L", True)
        pdf.cell(50, 8, strip_accents("English Term"), 1, 0, "C", True)
        pdf.cell(30, 8, strip_accents("Count"), 1, 0, "C", True)
        pdf.cell(30, 8, strip_accents("Ratio (%)"), 1, 1, "C", True)
        
        pdf.set_text_color(15, 23, 42)
        pdf.set_font("helvetica", "", 10)
        
        defect_mapping_list = [
            ("hotspot_single_cell", "Diem nong don cell", "Single Hotspot", stats_defects["hotspot_single_cell"]),
            ("hotspot_multi_cell", "Diem nong da cell", "Multi Hotspot", stats_defects["hotspot_multi_cell"]),
            ("shading", "Bong che khuat", "Shading", stats_defects["shading"]),
            ("soiling", "Bam bui ban", "Soiling", stats_defects["soiling"]),
            ("crack", "Nut vo vat ly", "Crack", stats_defects["crack"]),
        ]
        
        sum_defects = sum(stats_defects.values())
        
        for index, (key, vi_name, en_name, count) in enumerate(defect_mapping_list):
            pdf.set_fill_color(248, 250, 252) if index % 2 == 0 else pdf.set_fill_color(255, 255, 255)
            pdf.cell(80, 8, strip_accents(f" {vi_name}"), 1, 0, "L", True)
            pdf.cell(50, 8, strip_accents(en_name), 1, 0, "C", True)
            pdf.cell(30, 8, strip_accents(str(count)), 1, 0, "C", True)
            ratio = round(count / sum_defects * 100, 1) if sum_defects > 0 else 0.0
            pdf.cell(30, 8, strip_accents(f"{ratio}%"), 1, 1, "C", True)
            
        # ─────────────────────────────────────────
        # PHẦN 3: DANH SÁCH CHI TIẾT CÁC LỖI
        # ─────────────────────────────────────────
        pdf.add_page()
        pdf.set_font("helvetica", "B", 16)
        pdf.set_text_color(15, 23, 42)
        pdf.cell(0, 10, strip_accents("3. Detailed PV Anomalies Log"), ln=True)
        pdf.ln(5)
        
        # Tạo thư mục tạm để crop ảnh
        temp_dir = "data/temp_crop"
        os.makedirs(temp_dir, exist_ok=True)
        
        # Mapper cho tên lỗi tiếng Việt (không dấu để FPDF an toàn)
        VI_DEFECT_MAP = {
            "hotspot_single_cell": "Diem nong cuc bo (Single)",
            "hotspot_multi_cell": "Diem nong da cum (Multi)",
            "shading": "Bong che khuat (Shading)",
            "soiling": "Bam bui ban (Soiling)",
            "crack": "Nut vo vat ly (Crack)",
        }
        
        VI_SEVERITY_MAP = {
            "very_minor": "Rat nhe",
            "minor": "Nhe",
            "moderate": "Can theo doi",
            "severe": "Uu tien bao tri",
            "replace": "Can thay the"
        }
        
        # Duyệt qua các lỗi
        for idx, (p, p_detail) in enumerate(faulty_panels):
            # Kiểm tra xem có đủ chỗ để in bảng dữ liệu và 2 bức ảnh không (yêu cầu khoảng 75mm)
            if pdf.get_y() + 75 > 280:
                pdf.add_page()
                
            pdf.set_font("helvetica", "B", 11)
            pdf.set_text_color(15, 23, 42)
            pdf.set_fill_color(241, 245, 249) # Slate-100
            
            # Header Panel
            local_id = p.panel.local_id
            row = p_detail.get("row", 0)
            col = p_detail.get("col", 0)
            
            # Đọc tọa độ GPS từ EXIF hoặc ước tính
            lat = p_detail.get("gps_lat")
            lng = p_detail.get("gps_lng")
            if lat is None or lng is None:
                lat = round(10.8231 + row * 0.00015, 6)
                lng = round(106.6297 + col * 0.00025, 6)
            else:
                lat = round(lat, 6)
                lng = round(lng, 6)
            
            pdf.cell(0, 7, strip_accents(f" Anomalous Panel {local_id} (Row {row}, Col {col}) | GPS: {lat}, {lng}"), 1, 1, "L", True)
            
            # Thêm chi tiết panel
            pdf.set_font("helvetica", "", 9)
            pdf.set_text_color(51, 65, 85) # Slate-700
            
            # Cắt ảnh thermal và RGB
            thermal_filename = p.image.filename
            rgb_filename = thermal_to_rgb.get(thermal_filename)
            
            # Đường dẫn ảnh
            thermal_path = os.path.join("data/results", thermal_filename)
            rgb_path = os.path.join("data/raw", rgb_filename) if rgb_filename else None
            
            bbox = p_detail.get("bbox", [])
            
            crop_success = False
            temp_thermal_path = ""
            temp_rgb_path = ""
            
            if bbox and len(bbox) == 4 and os.path.exists(thermal_path):
                try:
                    # 1. Đọc và cắt ảnh Thermal (gán lỗi)
                    t_img = cv2.imread(thermal_path)
                    if t_img is not None:
                        h_t, w_t = t_img.shape[:2]
                        x1, y1, x2, y2 = [int(v) for v in bbox]
                        
                        # Sử dụng mức zoom 50% kích thước ảnh (giống get_panel_image)
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
                            
                        # 2. Đọc và cắt ảnh RGB tương ứng
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
                                    # Vẽ viền xanh lá để đánh dấu panel trên RGB
                                    r_pad_x = int((x1 - x1_t) * scale_x)
                                    r_pad_y = int((y1 - y1_t) * scale_y)
                                    r_w = int((x2 - x1) * scale_x)
                                    r_h = int((y2 - y1) * scale_y)
                                    
                                    cv2.rectangle(
                                        crop_r, 
                                        (r_pad_x, r_pad_y), 
                                        (r_pad_x + r_w, r_pad_y + r_h), 
                                        (0, 255, 0), 
                                        2
                                    )
                                    temp_rgb_path = os.path.join(temp_dir, f"r_{local_id}_{idx}.jpg")
                                    cv2.imwrite(temp_rgb_path, crop_r)
                                    
                        crop_success = os.path.exists(temp_thermal_path) and os.path.exists(temp_rgb_path)
                except Exception as e:
                    crop_success = False

            # In thông tin textual (bên trái)
            text_x = 10
            text_y = pdf.get_y()
            
            pdf.ln(2)
            pdf.set_font("helvetica", "B", 9)
            pdf.cell(40, 5, strip_accents("Defects Found:"))
            pdf.set_font("helvetica", "", 9)
            
            # Gom các loại lỗi
            defects = p_detail.get("defects", [])
            defect_types = []
            confidence_values = []
            relative_positions = []
            
            for d in defects:
                dname = VI_DEFECT_MAP.get(d.get("class_name", ""), str(d.get("class_name", "")))
                defect_types.append(dname)
                confidence_values.append(f"{round(d.get('confidence', 0.0) * 100, 1)}%")
                # Vị trí tương đối
                loc = d.get("location_in_panel", "center")
                u = d.get("relative_position", {}).get("u", 0.5)
                v = d.get("relative_position", {}).get("v", 0.5)
                relative_positions.append(f"{loc} (u:{round(u,2)}, v:{round(v,2)})")
                
            pdf.cell(60, 5, strip_accents(", ".join(defect_types)), ln=True)
            
            pdf.set_font("helvetica", "B", 9)
            pdf.cell(40, 5, strip_accents("Severity & Confidence:"))
            pdf.set_font("helvetica", "", 9)
            sev = VI_SEVERITY_MAP.get(p_detail.get("worst_severity", "minor"), "Nhe")
            pdf.cell(60, 5, strip_accents(f"{sev} | Confidence: {', '.join(confidence_values)}"), ln=True)

            pdf.set_font("helvetica", "B", 9)
            pdf.cell(40, 5, strip_accents("Power Loss & Action:"))
            pdf.set_font("helvetica", "", 9)
            pdf.cell(60, 5, strip_accents(f"{p_detail.get('total_panel_loss', 0.0)} W | {p_detail.get('recommendation', 'Check')}"), ln=True)

            pdf.set_font("helvetica", "B", 9)
            pdf.cell(40, 5, strip_accents("Relative Position:"))
            pdf.set_font("helvetica", "", 9)
            pdf.cell(60, 5, strip_accents(", ".join(relative_positions)), ln=True)

            pdf.set_font("helvetica", "B", 9)
            pdf.cell(40, 5, strip_accents("Raw Thermal Image:"))
            pdf.set_font("helvetica", "", 9)
            pdf.cell(60, 5, strip_accents(f"{thermal_filename}"), ln=True)
            
            pdf.set_font("helvetica", "B", 9)
            pdf.cell(40, 5, strip_accents("Matched RGB Image:"))
            pdf.set_font("helvetica", "", 9)
            pdf.cell(60, 5, strip_accents(f"{rgb_filename or 'Not matched'}"), ln=True)

            # In ảnh kẹp song song nếu crop thành công
            if crop_success:
                # Vị trí đặt ảnh
                curr_y = pdf.get_y() + 2
                
                # In Thermal Crop ở x=110, RGB Crop ở x=155. Kích thước 40x32
                pdf.image(temp_thermal_path, x=110, y=text_y + 2, w=40, h=32)
                pdf.image(temp_rgb_path, x=155, y=text_y + 2, w=40, h=32)
                
                # Nhãn ảnh
                pdf.set_xy(110, text_y + 34)
                pdf.set_font("helvetica", "I", 7)
                pdf.set_text_color(100, 116, 139)
                pdf.cell(40, 4, strip_accents("Thermal (Annotated)"), 0, 0, "C")
                pdf.cell(5, 4, "")
                pdf.cell(40, 4, strip_accents("RGB (Optical)"), 0, 1, "C")
                
                # Reset màu sắc
                pdf.set_text_color(51, 65, 85)
                
                # Di chuyển con trỏ y tới sau vị trí ảnh để tránh ghi đè
                pdf.set_y(max(pdf.get_y(), curr_y + 36))
            else:
                pdf.ln(5)
                
            pdf.ln(6)
            
        # Xóa các file tạm
        try:
            for f in os.listdir(temp_dir):
                os.remove(os.path.join(temp_dir, f))
            os.rmdir(temp_dir)
        except:
            pass
            
        pdf.output(output_path)
        return output_path