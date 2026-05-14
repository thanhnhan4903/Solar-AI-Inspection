# app/services/report_generator.py
from fpdf import FPDF
import datetime

class ReportGenerator:
    @staticmethod
    def generate_inspection_report(batch_id, data_list, output_path):
        """
        Khối 7: Tạo biên bản kiểm tra định kỳ bằng PDF[cite: 1]
        """
        pdf = FPDF()
        pdf.add_page()
        
        # 1. Tiêu đề báo cáo
        pdf.set_font("helvetica", "B", 16)
        pdf.cell(0, 10, "SOLAR PANEL INSPECTION REPORT", ln=True, align="C")
        pdf.set_font("helvetica", "", 10)
        pdf.cell(0, 10, f"Batch ID: {batch_id} | Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True, align="C")
        pdf.ln(10)

        # 2. Bảng thống kê tổng quát
        total_panels = len(data_list)
        defective_panels = [p for p in data_list if p.defect_type != "Healthy"]
        
        pdf.set_font("helvetica", "B", 12)
        pdf.cell(0, 10, "Summary Statistics:", ln=True)
        pdf.set_font("helvetica", "", 10)
        pdf.cell(0, 8, f"- Total Panels Scanned: {total_panels}", ln=True)
        pdf.cell(0, 8, f"- Defective Panels Found: {len(defective_panels)}", ln=True)
        pdf.cell(0, 8, f"- Health Rate: {round((total_panels - len(defective_panels))/total_panels * 100, 2)}%", ln=True)
        pdf.ln(5)

        # 3. Danh sách chi tiết các lỗi nặng (Loss > 20%)
        pdf.set_font("helvetica", "B", 12)
        pdf.cell(0, 10, "Critical Defects List (Action Required):", ln=True)
        
        # Header bảng
        pdf.set_fill_color(200, 200, 200)
        pdf.set_font("helvetica", "B", 9)
        pdf.cell(40, 8, "Panel ID", 1, 0, "C", True)
        pdf.cell(60, 8, "Defect Type", 1, 0, "C", True)
        pdf.cell(40, 8, "Efficiency Loss", 1, 0, "C", True)
        pdf.cell(50, 8, "Image Source", 1, 1, "C", True)

        # Nội dung bảng
        pdf.set_font("helvetica", "", 8)
        for p in defective_panels:
            if p.loss_pct > 0: # Chỉ liệt kê các tấm có lỗi thực tế
                pdf.cell(40, 7, str(p.panel.local_id), 1)
                pdf.cell(60, 7, str(p.defect_type), 1)
                pdf.cell(40, 7, f"{p.loss_pct}%", 1, 0, "C")
                pdf.cell(50, 7, str(p.image.filename), 1, 1)

        pdf.output(output_path)
        return output_path