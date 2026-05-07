# app/services/analyzer.py
from shapely.geometry import Polygon

class SolarAnalyzer:
    @staticmethod
    def calculate_metrics(panel_poly, defect_poly, class_name):
        """Tính toán diện tích, trọng tâm và mức độ hao hụt"""
        p_geom = Polygon(panel_poly)
        d_geom = Polygon(defect_poly)
        
        # Lấy trọng tâm (x, y) của tấm pin
        centroid = p_geom.centroid
        
        # Tính hao hụt công suất
        if "multi_cell" in class_name.lower():
            loss = 33.33  # Quy tắc trừ thẳng 1/3
        else:
            # Tỷ lệ diện tích, giới hạn tối đa 100%
            ratio = (d_geom.area / p_geom.area) * 100 if p_geom.area > 0 else 0
            loss = min(ratio, 100)
            
        return round(loss, 2), (round(centroid.x, 2), round(centroid.y, 2))

    @staticmethod
    def assign_grid_ids(panels):
        """
        Đánh số Panel theo thứ tự: trái→phải, trên→dưới.
        Gom các panel có Y gần nhau vào cùng 1 hàng (threshold).
        """
        if not panels:
            return panels

        # Sắp xếp theo Y trước
        sorted_by_y = sorted(panels, key=lambda p: p['y'])

        # Tính threshold = 30% chiều cao trung bình của panel (từ box)
        heights = []
        for p in sorted_by_y:
            box = p.get('box')
            if box and len(box) == 4:
                heights.append(box[3] - box[1])
        row_threshold = (sum(heights) / len(heights)) * 0.3 if heights else 30

        # Gom vào các hàng
        rows = []
        current_row = [sorted_by_y[0]]
        current_y = sorted_by_y[0]['y']

        for p in sorted_by_y[1:]:
            if abs(p['y'] - current_y) <= row_threshold:
                current_row.append(p)
            else:
                rows.append(current_row)
                current_row = [p]
                current_y = p['y']
        rows.append(current_row)

        # Trong mỗi hàng, sắp xếp theo X (trái → phải)
        result = []
        for row in rows:
            row_sorted = sorted(row, key=lambda p: p['x'])
            result.extend(row_sorted)

        # Đánh số thứ tự
        for i, panel in enumerate(result):
            panel['local_id'] = f"Panel_{i+1:02d}"
        return result