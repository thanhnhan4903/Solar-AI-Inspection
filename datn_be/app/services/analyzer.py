# app/services/analyzer.py
# SolarAnalyzer — wrapper facade cho pipeline phân tích.
# Phiên bản mới chuyển toàn bộ logic sang panel_geometry.py và defect_logic.py.
# File này giữ lại để không break import ở main.py và các file khác.
from app.services.panel_geometry import assign_row_col_ids
from app.services.defect_logic import assign_defects_to_panels, classify_severity, recommendation_from_severity


class SolarAnalyzer:

    @staticmethod
    def assign_grid_ids(panels):
        """
        Wrapper: Gán hàng/cột cho từng panel.
        local_id format: R01_C03 (row 1, col 3).
        Xem chi tiết tại panel_geometry.assign_row_col_ids().
        """
        return assign_row_col_ids(panels)

    @staticmethod
    def assign_defects(panels, defects):
        """
        Wrapper: Gán defect vào panel bằng overlap area.
        Xem chi tiết tại defect_logic.assign_defects_to_panels().
        """
        return assign_defects_to_panels(panels, defects)

    @staticmethod
    def classify_severity(defect_class: str, area_ratio_percent: float) -> str:
        """Phân loại severity từ area_ratio. Không dùng nhiệt độ."""
        return classify_severity(defect_class, area_ratio_percent)

    @staticmethod
    def recommendation(severity: str) -> str:
        """Sinh recommendation text từ severity."""
        return recommendation_from_severity(severity)