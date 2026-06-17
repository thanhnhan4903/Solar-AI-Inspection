# app/services/panel_geometry.py
# Chứa các hàm xử lý hình học cho tấm pin (panel):
#   - Validation panel detection
#   - Refine polygon về tứ giác sạch bằng minAreaRect
#   - Trích xuất bbox, area, center, aspect_ratio từ polygon
#   - Gán hàng/cột (R01_C03) theo vị trí tương đối trong ảnh
#   - Định vị lỗi nằm ở đâu trong panel (upper-left, center, lower-right...)
import cv2
import numpy as np
from typing import List, Optional, Dict, Any, Tuple


# ─────────────────────────────────────────
# 1. VALIDATION PANEL
# ─────────────────────────────────────────

def is_valid_panel_detection(det: Dict[str, Any], median_area: Optional[float] = None) -> bool:
    """
    Kiểm tra một detection class 'panel' có hợp lệ không.
    - Polygon phải có ít nhất 3 điểm
    - Area không quá nhỏ (loại nhiễu)
    - Aspect ratio hợp lý (không dài/hẹp bất thường)
    - Nếu có median_area thì loại panel quá lệch so với median
    """
    poly = det.get("polygon", [])
    area = det.get("area", 0)

    # Cần ít nhất 3 điểm để tạo polygon
    if len(poly) < 3:
        return False

    # Loại diện tích quá nhỏ (< 200 px² thường là nhiễu)
    if area < 200:
        return False

    # Kiểm tra aspect ratio: lấy từ bbox
    bbox = det.get("bbox") or det.get("box")
    if bbox and len(bbox) == 4:
        w = max(bbox[2] - bbox[0], 1)
        h = max(bbox[3] - bbox[1], 1)
        aspect = max(w, h) / min(w, h)
        # Panel tấm pin thường có aspect ratio 1:1 đến 3:1, tối đa ~5:1
        if aspect > 6.0:
            return False

    # So sánh với median area nếu có
    if median_area and median_area > 0:
        ratio = area / median_area
        # Loại panel có diện tích
        if ratio < 0.75 or ratio > 1.25:
            return False

    return True


# ─────────────────────────────────────────
# 2. REFINE PANEL POLYGON (minAreaRect → 4 điểm)
# ─────────────────────────────────────────

def refine_panel_polygon_from_mask(mask_data: np.ndarray, img_shape: Tuple[int, int]) -> Optional[List[List[int]]]:
    """
    Từ binary mask raster của panel:
    1. Morphology close/open 5x5 để điền lỗ nhỏ
    2. Lấy contour lớn nhất
    3. Dùng cv2.minAreaRect → 4 điểm tứ giác sạch
    Trả về list 4 điểm [[x,y], ...] hoặc None nếu thất bại.

    Lý do dùng minAreaRect cho panel: panel là vật thể có hình dạng hình chữ nhật
    (tứ giác xoay), minAreaRect cho ra polygon 4 điểm rất sạch và ổn định.
    """
    img_h, img_w = img_shape[:2]

    # Đảm bảo mask là uint8
    if mask_data.dtype != np.uint8:
        mask_u8 = (mask_data > 0).astype(np.uint8) * 255
    else:
        mask_u8 = mask_data.copy()

    # Resize mask về kích thước ảnh gốc nếu cần
    if mask_u8.shape[:2] != (img_h, img_w):
        mask_u8 = cv2.resize(mask_u8, (img_w, img_h), interpolation=cv2.INTER_NEAREST)

    # Morphology close (điền lỗ nhỏ) rồi open (loại nhiễu nhỏ)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

    # Tìm contour
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Lấy contour lớn nhất
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 100:
        return None

    # Dùng approxPolyDP để trích xuất viền tấm pin thực tế thay vì hình chữ nhật
    epsilon = 0.005 * cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, epsilon, True)
    poly = approx.reshape(-1, 2).tolist()
    
    if len(poly) < 3:
        return None
    return poly


def refine_panel_polygon_from_xy(xy_polygon: List, img_shape: Tuple[int, int]) -> Optional[List[List[int]]]:
    """
    Từ polygon xy của YOLO (result.masks.xy):
    1. Vẽ vào binary mask
    2. Gọi refine_panel_polygon_from_mask
    Dùng khi không có mask raster trực tiếp.
    """
    img_h, img_w = img_shape[:2]
    pts = np.array(xy_polygon, dtype=np.int32)
    if len(pts) < 3:
        return None

    binary = np.zeros((img_h, img_w), dtype=np.uint8)
    cv2.fillPoly(binary, [pts], 255)
    return refine_panel_polygon_from_mask(binary, img_shape)


# ─────────────────────────────────────────
# 3. TRÍCH XUẤT FEATURE TỪ POLYGON
# ─────────────────────────────────────────

def get_polygon_features(points: List) -> Dict[str, Any]:
    """
    Tính bbox, area, center, aspect_ratio từ list điểm polygon.
    Trả về dict với các key: bbox, area, center, aspect_ratio.
    """
    pts = np.array(points, dtype=np.float32)
    if len(pts) < 3:
        return {"bbox": [0, 0, 0, 0], "area": 0, "center": [0, 0], "aspect_ratio": 1.0}

    x_coords = pts[:, 0]
    y_coords = pts[:, 1]

    x1, y1 = float(np.min(x_coords)), float(np.min(y_coords))
    x2, y2 = float(np.max(x_coords)), float(np.max(y_coords))

    w = max(x2 - x1, 1)
    h = max(y2 - y1, 1)

    # Diện tích polygon (Shoelace formula)
    area = float(cv2.contourArea(pts.reshape(-1, 1, 2)))

    cx = float(np.mean(x_coords))
    cy = float(np.mean(y_coords))

    aspect_ratio = round(max(w, h) / min(w, h), 3)

    return {
        "bbox": [round(x1), round(y1), round(x2), round(y2)],
        "area": round(area, 2),
        "center": [round(cx, 2), round(cy, 2)],
        "aspect_ratio": aspect_ratio,
    }


# ─────────────────────────────────────────
# 4. GÁN HÀNG/CỘT (ROW/COL GRID)
# ─────────────────────────────────────────

def _panel_center(panel: Dict[str, Any]) -> Tuple[float, float]:
    """
    Trích xuất tâm panel robust theo thứ tự ưu tiên:
    outer_polygon -> polygon -> inner_polygon -> bbox -> box.
    """
    poly = panel.get("outer_polygon") or panel.get("polygon") or panel.get("inner_polygon")
    if poly and len(poly) >= 3:
        xs = [float(pt[0]) for pt in poly]
        ys = [float(pt[1]) for pt in poly]
        return float(np.mean(xs)), float(np.mean(ys))

    bbox = panel.get("bbox") or panel.get("box")
    if bbox and len(bbox) == 4:
        return (float(bbox[0]) + float(bbox[2])) / 2.0, (float(bbox[1]) + float(bbox[3])) / 2.0

    return 0.0, 0.0


def _cluster_panel_blocks(panels: List[Dict[str, Any]], centers: List[Tuple[float, float]]) -> List[List[int]]:
    """
    Gom cụm panel thành các block/cụm độc lập sử dụng connected components
    dựa trên khoảng cách giữa các tâm panel và median panel pitch.
    """
    n = len(panels)
    if n == 0:
        return []
    if n == 1:
        return [[0]]

    pts = np.array(centers)
    nearest_dists = []
    for i in range(n):
        dists_to_others = [np.linalg.norm(pts[i] - pts[j]) for j in range(n) if i != j]
        if dists_to_others:
            nearest_dists.append(min(dists_to_others))

    median_pitch = float(np.median(nearest_dists)) if nearest_dists else 100.0
    if median_pitch <= 0:
        median_pitch = 100.0

    # Ngưỡng kết nối: 2.2 lần khoảng cách trung vị
    connect_thresh = 2.2 * median_pitch

    # Xây dựng danh sách kề
    adj = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(pts[i] - pts[j])
            if dist <= connect_thresh:
                adj[i].append(j)
                adj[j].append(i)

    # Tìm các thành phần liên thông
    visited = [False] * n
    blocks = []
    for i in range(n):
        if not visited[i]:
            comp = []
            queue = [i]
            visited[i] = True
            while queue:
                curr = queue.pop(0)
                comp.append(curr)
                for neighbor in adj[curr]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)
            blocks.append(comp)

    # Tính centroid của từng block
    block_centroids = []
    for comp in blocks:
        comp_pts = pts[comp]
        block_centroids.append(np.mean(comp_pts, axis=0))

    # Sắp xếp các block từ trên xuống dưới, trái sang phải (cy tăng dần, rồi cx tăng dần)
    sorted_block_indices = sorted(
        range(len(blocks)),
        key=lambda idx: (block_centroids[idx][1], block_centroids[idx][0])
    )
    sorted_blocks = [blocks[idx] for idx in sorted_block_indices]

    return sorted_blocks


def _estimate_block_axes(
    panels: List[Dict[str, Any]],
    block_indices: List[int],
    centers: List[Tuple[float, float]]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ước lượng trục grid local (u_axis, v_axis) cho một block cụ thể.
    """
    u_axis = np.array([1.0, 0.0])
    v_axis = np.array([0.0, -1.0])

    if len(block_indices) < 2:
        return u_axis, v_axis

    # 1. Trích xuất hướng từ cạnh polygon
    angles = []
    for idx in block_indices:
        p = panels[idx]
        poly = p.get("outer_polygon") or p.get("polygon")
        if poly and len(poly) >= 3:
            for k in range(len(poly)):
                pt1 = poly[k]
                pt2 = poly[(k + 1) % len(poly)]
                dx = float(pt2[0] - pt1[0])
                dy = float(pt2[1] - pt1[1])
                dist = np.hypot(dx, dy)
                if dist > 2.0:
                    angle = np.arctan2(dy, dx) % np.pi
                    angles.append(angle)

    dominant_theta = None
    if len(angles) >= 4:
        try:
            # Map góc về [0, pi/2) bằng circular mean của 4 * angle
            angles_arr = np.array(angles)
            mean_sin = np.mean(np.sin(4 * angles_arr))
            mean_cos = np.mean(np.cos(4 * angles_arr))
            mean_4_angle = np.arctan2(mean_sin, mean_cos)
            dominant_theta = (mean_4_angle / 4.0) % (np.pi / 2.0)
        except Exception:
            pass

    # 2. PCA của tọa độ tâm nếu polygon edges không khả dụng
    pca_u, pca_v = None, None
    if len(block_indices) >= 3:
        try:
            block_pts = np.array([centers[idx] for idx in block_indices])
            mean_pt = np.mean(block_pts, axis=0)
            centered = block_pts - mean_pt
            cov = np.cov(centered, rowvar=False)
            if cov.shape == (2, 2) and np.all(np.isfinite(cov)):
                eigenvalues, eigenvectors = np.linalg.eigh(cov)
                sort_idx = np.argsort(eigenvalues)[::-1]
                eigenvectors = eigenvectors[:, sort_idx]
                pca_u = eigenvectors[:, 0]
                pca_v = eigenvectors[:, 1]
        except Exception:
            pass

    # Chọn hướng trục phù hợp
    if dominant_theta is not None:
        v1 = np.array([np.cos(dominant_theta), np.sin(dominant_theta)])
        v2 = np.array([-np.sin(dominant_theta), np.cos(dominant_theta)])
        if abs(v1[0]) > abs(v2[0]):
            u_axis = v1
            v_axis = v2
        else:
            u_axis = v2
            v_axis = v1
    elif pca_u is not None and pca_v is not None:
        if abs(pca_u[0]) > abs(pca_v[0]):
            u_axis = pca_u
            v_axis = pca_v
        else:
            u_axis = pca_v
            v_axis = pca_u

    # Đồng bộ hóa chiều hướng trục
    # u_axis hướng từ trái sang phải
    if u_axis[0] < 0:
        u_axis = -u_axis
    # v_axis hướng từ dưới lên trên (trong hệ pixel y tăng xuống, v_axis[1] phải âm)
    if v_axis[1] > 0:
        v_axis = -v_axis

    return u_axis, v_axis


def _cluster_1d(values: List[float], threshold: float) -> List[float]:
    """
    Gom cụm 1D đơn giản bằng greedy thresholding.
    Trả về danh sách các giá trị trung tâm của cụm.
    """
    if not values:
        return []
    sorted_vals = sorted(values)
    clusters = []
    curr = [sorted_vals[0]]
    for val in sorted_vals[1:]:
        if val - np.mean(curr) <= threshold:
            curr.append(val)
        else:
            clusters.append(curr)
            curr = [val]
    clusters.append(curr)
    return [float(np.mean(c)) for c in clusters]


def _stitch_global_row_col_ids(
    panels: List[Dict[str, Any]],
    centers: List[Tuple[float, float]],
    blocks_meta: List[Dict[str, Any]]
) -> None:
    """
    Nối ghép các block có cùng hướng nghiêng/xoay và pitch tương đồng thành hệ tọa độ global liên tục.
    Sắp xếp theo các lane dọc (vertical lanes) để tránh ảnh hưởng chéo về chỉ số hàng giữa các cột khác nhau.
    """
    n_blocks = len(blocks_meta)
    if n_blocks == 0:
        return

    # Helper tính góc giữa 2 vector đơn vị
    def angle_between(v1, v2):
        dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
        return float(np.arccos(dot) * 180.0 / np.pi)

    # Ma trận liên kết giữa các block (Stitchable blocks)
    stitch_adj = {i: [] for i in range(n_blocks)}
    for i in range(n_blocks):
        for j in range(i + 1, n_blocks):
            bm_i = blocks_meta[i]
            bm_j = blocks_meta[j]

            u_i, v_i = bm_i["u_axis"], bm_i["v_axis"]
            u_j, v_j = bm_j["u_axis"], bm_j["v_axis"]

            angle_u = angle_between(u_i, u_j)
            angle_v = angle_between(v_i, v_j)

            pitch_u_ratio = bm_i["u_pitch"] / bm_j["u_pitch"] if bm_j["u_pitch"] > 0 else 1.0
            pitch_v_ratio = bm_i["v_pitch"] / bm_j["v_pitch"] if bm_j["v_pitch"] > 0 else 1.0

            aligned = (angle_u <= 15.0 or angle_u >= 165.0) and (angle_v <= 15.0 or angle_v >= 165.0)
            pitch_matched = (0.5 <= pitch_u_ratio <= 2.0) and (0.5 <= pitch_v_ratio <= 2.0)

            if aligned and pitch_matched:
                stitch_adj[i].append(j)
                stitch_adj[j].append(i)

    # Gom nhóm Stitch Group liên thông
    visited = [False] * n_blocks
    stitch_groups = []
    for i in range(n_blocks):
        if not visited[i]:
            comp = []
            queue = [i]
            visited[i] = True
            while queue:
                curr = queue.pop(0)
                comp.append(curr)
                for neighbor in stitch_adj[curr]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)
            stitch_groups.append(comp)

    total_stitch_groups = len(stitch_groups)

    # Xử lý gán global_row, global_col cho từng group
    for sg_idx, block_indices_in_sg in enumerate(stitch_groups):
        # 1. Tìm block tham chiếu trong group (block nhiều panel nhất)
        ref_block_idx = max(block_indices_in_sg, key=lambda idx: len(blocks_meta[idx]["panel_indices"]))
        ref_meta = blocks_meta[ref_block_idx]

        global_u_axis = ref_meta["u_axis"]
        global_v_axis = ref_meta["v_axis"]
        global_origin = ref_meta["centroid"]
        global_u_pitch = ref_meta["u_pitch"]
        global_v_pitch = ref_meta["v_pitch"]

        if ref_meta["is_fallback"]:
            other_u_pitches = [blocks_meta[idx]["u_pitch"] for idx in block_indices_in_sg if not blocks_meta[idx]["is_fallback"]]
            other_v_pitches = [blocks_meta[idx]["v_pitch"] for idx in block_indices_in_sg if not blocks_meta[idx]["is_fallback"]]
            if other_u_pitches:
                global_u_pitch = float(np.median(other_u_pitches))
            if other_v_pitches:
                global_v_pitch = float(np.median(other_v_pitches))

        # 2. Chiếu tất cả các tấm pin trong stitch group lên hệ trục global
        for b_idx in block_indices_in_sg:
            b_meta = blocks_meta[b_idx]
            b_us = []
            b_vs = []
            for p_idx in b_meta["panel_indices"]:
                cx, cy = centers[p_idx]
                pt = np.array([cx, cy])
                u = float(np.dot(pt - global_origin, global_u_axis))
                v = float(np.dot(pt - global_origin, global_v_axis))
                panels[p_idx]["_g_u"] = u
                panels[p_idx]["_g_v"] = v
                b_us.append(u)
                b_vs.append(v)
            
            b_meta["g_u_min"] = min(b_us)
            b_meta["g_u_max"] = max(b_us)
            b_meta["g_v_min"] = min(b_vs)
            b_meta["g_v_max"] = max(b_vs)
            b_meta["g_centroid_u"] = float(np.mean(b_us))
            b_meta["g_centroid_v"] = float(np.mean(b_vs))
            b_meta["g_width_u"] = b_meta["g_u_max"] - b_meta["g_u_min"]
            b_meta["g_width_v"] = b_meta["g_v_max"] - b_meta["g_v_min"]
            b_meta["block_row_count"] = max([panels[idx]["block_row"] for idx in b_meta["panel_indices"]])

        # 3. Phân chia các block trong group thành các Vertical Lane
        lane_adj = {idx: [] for idx in block_indices_in_sg}
        for i_idx, b_i in enumerate(block_indices_in_sg):
            for j_idx in range(i_idx + 1, len(block_indices_in_sg)):
                b_j = block_indices_in_sg[j_idx]
                bm_i = blocks_meta[b_i]
                bm_j = blocks_meta[b_j]

                # Overlap ratio along u
                inter_u = max(0.0, min(bm_i["g_u_max"], bm_j["g_u_max"]) - max(bm_i["g_u_min"], bm_j["g_u_min"]))
                min_w = max(1e-6, min(bm_i["g_width_u"], bm_j["g_width_u"]))
                overlap_ratio = inter_u / min_w

                # Centroid distance along u
                dist_u = abs(bm_i["g_centroid_u"] - bm_j["g_centroid_u"])
                max_w = max(bm_i["g_width_u"], bm_j["g_width_u"])

                same_lane = (overlap_ratio >= 0.35) or (dist_u <= 0.75 * max_w)
                if same_lane:
                    lane_adj[b_i].append(b_j)
                    lane_adj[b_j].append(b_i)

        # Gom nhóm các Vertical Lane
        visited_blocks = {idx: False for idx in block_indices_in_sg}
        vertical_lanes = []
        for b_idx in block_indices_in_sg:
            if not visited_blocks[b_idx]:
                comp = []
                queue = [b_idx]
                visited_blocks[b_idx] = True
                while queue:
                    curr = queue.pop(0)
                    comp.append(curr)
                    for neighbor in lane_adj[curr]:
                        if not visited_blocks[neighbor]:
                            visited_blocks[neighbor] = True
                            queue.append(neighbor)
                vertical_lanes.append(comp)

        # Sắp xếp các Vertical Lane từ trái sang phải theo centroid_u trung bình
        lane_centroids_u = []
        for lane in vertical_lanes:
            lane_centroids_u.append(np.mean([blocks_meta[b_idx]["g_centroid_u"] for b_idx in lane]))

        sorted_lane_indices = sorted(range(len(vertical_lanes)), key=lambda idx: lane_centroids_u[idx])
        sorted_vertical_lanes = [vertical_lanes[idx] for idx in sorted_lane_indices]

        # 4. Xử lý gán hàng/cột và offset trong từng lane
        lane_col_offset = 0

        for lane_idx, lane in enumerate(sorted_vertical_lanes):
            lane_id_str = f"L{lane_idx+1:02d}"

            # Sắp xếp các block trong lane từ trên xuống dưới (g_centroid_v giảm dần)
            sorted_blocks_in_lane = sorted(lane, key=lambda idx: blocks_meta[idx]["g_centroid_v"], reverse=True)

            # Chia các block trong lane thành các row group nối tiếp
            row_groups = []
            current_group = [sorted_blocks_in_lane[0]]
            for i in range(1, len(sorted_blocks_in_lane)):
                prev_b = sorted_blocks_in_lane[i-1]
                curr_b = sorted_blocks_in_lane[i]
                meta_prev = blocks_meta[prev_b]
                meta_curr = blocks_meta[curr_b]

                vertical_gap = meta_prev["g_v_min"] - meta_curr["g_v_max"]
                if vertical_gap <= 4.0 * global_v_pitch:
                    current_group.append(curr_b)
                else:
                    row_groups.append(current_group)
                    current_group = [curr_b]
            row_groups.append(current_group)

            # Gán hàng dựa trên từng row group
            # Gom tất cả các u của các panel trong lane này để tìm cột toàn cục của lane
            lane_p_indices = []
            for b_idx in lane:
                lane_p_indices.extend(blocks_meta[b_idx]["panel_indices"])
            
            lane_us = [panels[p_idx]["_g_u"] for p_idx in lane_p_indices]
            col_threshold = 0.55 * global_u_pitch
            lane_u_centers = _cluster_1d(lane_us, col_threshold)
            lane_u_centers = sorted(lane_u_centers)

            # Số cột của lane này
            lane_col_count = len(lane_u_centers)

            # Xử lý từng row group
            for row_group in row_groups:
                # Nếu row group chỉ có 1 block: dùng luôn block_row của panel
                if len(row_group) == 1:
                    b_idx = row_group[0]
                    for p_idx in blocks_meta[b_idx]["panel_indices"]:
                        p = panels[p_idx]
                        u = p["_g_u"]

                        # Tìm cột tương đối trong lane
                        lane_col_idx = int(np.argmin([abs(u - uc) for uc in lane_u_centers]))

                        p["global_row"] = p["block_row"]
                        p["global_col"] = lane_col_offset + (lane_col_idx + 1)
                        p["global_local_id"] = f"R{p['global_row']:02d}_C{p['global_col']:02d}"
                        p["vertical_lane_id"] = lane_id_str

                        # Gán local_id chính cho UI
                        if total_stitch_groups == 1:
                            p["row"] = p["global_row"]
                            p["col"] = p["global_col"]
                            p["local_id"] = p["global_local_id"]
                            p["row_col_method"] = "lane_based_global_stitch"
                        else:
                            p["row"] = p["block_row"]
                            p["col"] = p["block_col"]
                            p["local_id"] = p["block_local_id"]
                            p["row_col_method"] = "block_local_no_vertical_stitch"
                else:
                    # Nếu có nhiều block trong row group, chúng xếp chồng nối tiếp và cần gộp hàng
                    group_p_indices = []
                    for b_idx in row_group:
                        group_p_indices.extend(blocks_meta[b_idx]["panel_indices"])
                    
                    group_vs = [panels[p_idx]["_g_v"] for p_idx in group_p_indices]
                    row_threshold = 0.55 * global_v_pitch
                    group_v_centers = _cluster_1d(group_vs, row_threshold)
                    group_v_centers = sorted(group_v_centers, reverse=True)

                    for p_idx in group_p_indices:
                        p = panels[p_idx]
                        u = p["_g_u"]
                        v = p["_g_v"]

                        lane_col_idx = int(np.argmin([abs(u - uc) for uc in lane_u_centers]))
                        group_row_idx = int(np.argmin([abs(v - vc) for vc in group_v_centers]))

                        p["global_row"] = group_row_idx + 1
                        p["global_col"] = lane_col_offset + (lane_col_idx + 1)
                        p["global_local_id"] = f"R{p['global_row']:02d}_C{p['global_col']:02d}"
                        p["vertical_lane_id"] = lane_id_str

                        # Gán local_id chính cho UI
                        if total_stitch_groups == 1:
                            p["row"] = p["global_row"]
                            p["col"] = p["global_col"]
                            p["local_id"] = p["global_local_id"]
                            p["row_col_method"] = "lane_based_global_stitch"
                        else:
                            p["row"] = p["block_row"]
                            p["col"] = p["block_col"]
                            p["local_id"] = p["block_local_id"]
                            p["row_col_method"] = "block_local_no_vertical_stitch"

            # Tăng lane_col_offset cho lane tiếp theo bên phải
            lane_col_offset += lane_col_count

    # Xóa biến tạm
    for p in panels:
        p.pop("_g_u", None)
        p.pop("_g_v", None)


def assign_row_col_ids(panels: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Gán row, col, local_id (dạng R01_C03) cho từng panel dựa trên vị trí local grid robust.
    Sau đó nối ghép các block có cùng xu hướng nghiêng để tạo thành hệ ID global liên tục.
    """
    if not panels:
        return panels

    # Bước 1: Tính toán tâm panel
    centers = [_panel_center(p) for p in panels]

    # Bước 2: Gom cụm các Block
    blocks = _cluster_panel_blocks(panels, centers)

    blocks_meta = []

    # Bước 3: Xử lý gán hàng cột cho từng block và trích xuất metadata
    for block_idx, block_indices in enumerate(blocks):
        block_id_str = f"B{block_idx+1:02d}"

        # Trường hợp block quá nhỏ (< 3 tấm pin), sử dụng cơ chế gán fallback đơn giản
        if len(block_indices) < 3:
            block_panels = [(idx, panels[idx], centers[idx]) for idx in block_indices]
            # Sắp xếp theo y tăng dần (trên xuống) rồi x tăng dần (trái sang)
            sorted_block_panels = sorted(block_panels, key=lambda x: (x[2][1], x[2][0]))

            if len(sorted_block_panels) == 2:
                p1_idx, p1, c1 = sorted_block_panels[0]
                p2_idx, p2, c2 = sorted_block_panels[1]
                dx = abs(c1[0] - c2[0])
                dy = abs(c1[1] - c2[1])
                if dx > dy:
                    p1["row"] = p2["row"] = 1
                    p1["col"] = 1
                    p2["col"] = 2
                else:
                    p1["row"] = 1
                    p2["row"] = 2
                    p1["col"] = p2["col"] = 1
            else:
                p_idx, p, c = sorted_block_panels[0]
                p["row"] = 1
                p["col"] = 1

            for idx in block_indices:
                p = panels[idx]
                r = p["row"]
                c = p["col"]
                p["block_id"] = block_id_str
                p["block_row"] = r
                p["block_col"] = c
                p["grid_row"] = r
                p["grid_col"] = c
                p["block_local_id"] = f"{block_id_str}_R{r:02d}_C{c:02d}"
                p["local_id"] = p["block_local_id"]
                p["row_col_method"] = "legacy_xy_sort_fallback"

            # Lưu meta cho block fallback
            block_centers = np.array([centers[idx] for idx in block_indices])
            block_centroid = np.mean(block_centers, axis=0)
            blocks_meta.append({
                "block_idx": block_idx,
                "block_id_str": block_id_str,
                "panel_indices": block_indices,
                "centroid": block_centroid,
                "u_axis": np.array([1.0, 0.0]),
                "v_axis": np.array([0.0, -1.0]),
                "u_pitch": 100.0,
                "v_pitch": 50.0,
                "is_fallback": True
            })
            continue

        # Bước 3b: Ước lượng hệ trục local
        u_axis, v_axis = _estimate_block_axes(panels, block_indices, centers)

        # Bước 4: Chiếu lên trục local
        block_centers = np.array([centers[idx] for idx in block_indices])
        block_centroid = np.mean(block_centers, axis=0)

        us = []
        vs = []
        for idx in block_indices:
            cx, cy = centers[idx]
            pt = np.array([cx, cy])
            u = float(np.dot(pt - block_centroid, u_axis))
            v = float(np.dot(pt - block_centroid, v_axis))
            us.append(u)
            vs.append(v)
            panels[idx]["_u"] = u
            panels[idx]["_v"] = v

        # Ước lượng chiều dài/rộng panel để tính ngưỡng pitch
        panel_widths = []
        panel_heights = []
        for idx in block_indices:
            bbox = panels[idx].get("bbox") or panels[idx].get("box", [0, 0, 0, 0])
            if len(bbox) == 4:
                panel_widths.append(float(bbox[2] - bbox[0]))
                panel_heights.append(float(bbox[3] - bbox[1]))
        med_w = float(np.median(panel_widths)) if panel_widths else 100.0
        med_h = float(np.median(panel_heights)) if panel_heights else 50.0

        # Ước lượng pitch
        u_sorted = sorted(us)
        u_diffs = [u_sorted[i+1] - u_sorted[i] for i in range(len(u_sorted)-1)]
        valid_u_diffs = [d for d in u_diffs if d > 0.3 * med_w]
        u_pitch = float(np.median(valid_u_diffs)) if valid_u_diffs else med_w

        v_sorted = sorted(vs)
        v_diffs = [v_sorted[i+1] - v_sorted[i] for i in range(len(v_sorted)-1)]
        valid_v_diffs = [d for d in v_diffs if d > 0.3 * med_h]
        v_pitch = float(np.median(valid_v_diffs)) if valid_v_diffs else med_h

        col_threshold = 0.55 * u_pitch
        row_threshold = 0.55 * v_pitch

        # Bước 6: Gom cụm 1D tọa độ chiếu
        u_centers = _cluster_1d(us, col_threshold)
        v_centers = _cluster_1d(vs, row_threshold)

        # Trục u hướng từ trái sang phải -> sort tăng dần
        u_centers = sorted(u_centers)
        # Trục v hướng từ dưới lên trên -> sort giảm dần (row 1 ở cao nhất)
        v_centers = sorted(v_centers, reverse=True)

        # Bước 7: Ánh xạ panel vào cụm gần nhất
        for idx in block_indices:
            p = panels[idx]
            u = p["_u"]
            v = p["_v"]

            col_idx = int(np.argmin([abs(u - uc) for uc in u_centers]))
            row_idx = int(np.argmin([abs(v - vc) for vc in v_centers]))

            p["row"] = row_idx + 1
            p["col"] = col_idx + 1
            p["block_row"] = row_idx + 1
            p["block_col"] = col_idx + 1
            p["grid_row"] = row_idx + 1
            p["grid_col"] = col_idx + 1
            p["block_id"] = block_id_str
            p["block_local_id"] = f"{block_id_str}_R{row_idx+1:02d}_C{col_idx+1:02d}"
            p["local_id"] = p["block_local_id"]
            p["row_col_method"] = "oriented_grid_projection"

        # Lưu meta cho block
        blocks_meta.append({
            "block_idx": block_idx,
            "block_id_str": block_id_str,
            "panel_indices": block_indices,
            "centroid": block_centroid,
            "u_axis": u_axis,
            "v_axis": v_axis,
            "u_pitch": u_pitch,
            "v_pitch": v_pitch,
            "is_fallback": False
        })

    # Dọn dẹp biến tạm
    for p in panels:
        p.pop("_u", None)
        p.pop("_v", None)

    # Bước 8: Thực hiện Stitch ghép các block song song thành hệ tọa độ global
    try:
        _stitch_global_row_col_ids(panels, centers, blocks_meta)
    except Exception as e:
        for p in panels:
            p["global_row"] = p.get("block_row", 1)
            p["global_col"] = p.get("block_col", 1)
            p["global_local_id"] = p.get("block_local_id", "R01_C01")
            p["local_id"] = p.get("block_local_id", "R01_C01")
            p["row"] = p.get("block_row", 1)
            p["col"] = p.get("block_col", 1)
            p["row_col_method"] = "block_oriented_grid_stitch_fallback"

    return panels


# ─────────────────────────────────────────
# 5. ĐỊNH VỊ LỖI TRONG PANEL
# ─────────────────────────────────────────

def localize_defect_inside_panel(
    defect_center: List[float],
    panel_bbox: List[float],
    panel_polygon: Optional[List[List[float]]] = None
) -> Dict[str, Any]:
    """
    Tính vị trí tương đối của lỗi (defect) bên trong tấm pin (panel).

    Args:
        defect_center: [cx, cy] của defect (pixel coords trong ảnh gốc)
        panel_bbox: [x1, y1, x2, y2] của panel (pixel coords trong ảnh gốc)
        panel_polygon: [[x,y], ...] của panel (pixel coords trong ảnh gốc)

    Returns:
        dict với:
            u (float): 0.0 = trái, 1.0 = phải
            v (float): 0.0 = trên, 1.0 = dưới
            u_long (float): vị trí dọc theo chiều dài của tấm pin (0.0 đến 1.0)
            v_short (float): vị trí dọc theo chiều rộng của tấm pin (0.0 đến 1.0)
            location_in_panel (str): e.g. "upper-left", "middle-center", "lower-right"
    """
    cx, cy = defect_center

    if not panel_polygon or len(panel_polygon) < 3:
        x1, y1, x2, y2 = panel_bbox
        w = max(x2 - x1, 1)
        h = max(y2 - y1, 1)
        u = max(0.0, min(1.0, (cx - x1) / w))
        v = max(0.0, min(1.0, (cy - y1) / h))
        is_landscape = w >= h
        u_long = u if is_landscape else v
        v_short = v if is_landscape else u
    else:
        pts = np.array(panel_polygon, dtype=np.float32)
        rect = cv2.minAreaRect(pts)
        box = cv2.boxPoints(rect)
        
        # ────────────────────────────────────────────────────────
        # CÁCH SỬA: Sắp xếp 4 góc theo thứ tự chuẩn centroid góc quay robust
        # ────────────────────────────────────────────────────────
        cx_box, cy_box = np.mean(box, axis=0)
        angles = np.arctan2(box[:, 1] - cy_box, box[:, 0] - cx_box)
        sorted_box = box[np.argsort(angles)]
        
        # Tìm góc Top-Left để roll mảng
        min_x, min_y = np.min(box, axis=0)
        distances = np.linalg.norm(sorted_box - [min_x, min_y], axis=1)
        tl_idx = np.argmin(distances)
        sorted_box = np.roll(sorted_box, -tl_idx, axis=0)
        
        tl, tr, br, bl = sorted_box[0], sorted_box[1], sorted_box[2], sorted_box[3]
        # ────────────────────────────────────────────────────────

        v_horiz = tr - tl
        v_vert = bl - tl
        
        dot_h = np.dot(v_horiz, v_horiz)
        dot_v = np.dot(v_vert, v_vert)
        
        vc = np.array([cx, cy], dtype=np.float32) - tl
        
        u_proj = np.dot(vc, v_horiz) / dot_h if dot_h > 0 else 0.5
        v_proj = np.dot(vc, v_vert) / dot_v if dot_v > 0 else 0.5
        
        u = max(0.0, min(1.0, float(u_proj)))
        v = max(0.0, min(1.0, float(v_proj)))
        
        if dot_h >= dot_v:
            u_long = u
            v_short = v
        else:
            u_long = v
            v_short = u

    # Phân vùng horizontal
    if u < 0.33:
        h_zone = "left"
    elif u < 0.66:
        h_zone = "center"
    else:
        h_zone = "right"

    # Phân vùng vertical
    if v < 0.33:
        v_zone = "upper"
    elif v < 0.66:
        v_zone = "middle"
    else:
        v_zone = "lower"

    return {
        "u": round(u, 3),
        "v": round(v, 3),
        "u_long": round(u_long, 3),
        "v_short": round(v_short, 3),
        "location_in_panel": f"{v_zone}-{h_zone}",
    }
