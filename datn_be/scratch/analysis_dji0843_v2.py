import json, math, statistics, pathlib, sys

file_path = r"c:/Solar_Inspection_Project/datn_be/data/results/debug_logs/DJI_0843_R_final_polygon_trace.jsonl"
records = []
with open(file_path, encoding='utf-8') as f:
    for line in f:
        try:
            rec = json.loads(line)
            if rec.get('single_column_tracking_candidate'):
                records.append(rec)
        except Exception:
            continue

# Group by block_id
blocks = {}
for r in records:
    bid = r.get('block_id')
    blocks.setdefault(bid, []).append(r)

def bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def area_bbox(bbox):
    x1, y1, x2, y2 = bbox
    return max(0, (x2 - x1)) * max(0, (y2 - y1))

def iou(b1, b2):
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = area_bbox(b1) + area_bbox(b2) - inter
    return inter / union if union else 0

for bid, items in blocks.items():
    cnt = len(items)
    areas = [area_bbox(r['bbox']) for r in items]
    area_mean = sum(areas) / cnt if cnt else 0
    area_cv = (statistics.stdev(areas) / area_mean) if cnt > 1 and area_mean != 0 else 0
    # centers sorted by y
    centers = [bbox_center(r['bbox']) for r in items]
    centers.sort(key=lambda c: c[1])
    # vertical gaps between consecutive centers
    vert_gaps = [abs(centers[i][1] - centers[i-1][1]) for i in range(1, cnt)]
    if vert_gaps:
        vert_mean = sum(vert_gaps) / len(vert_gaps)
        vert_cv = (statistics.stdev(vert_gaps) / vert_mean) if len(vert_gaps) > 1 and vert_mean != 0 else 0
    else:
        vert_mean = vert_cv = 0
    # max IoU overlap among all pairs in block
    max_iou = 0
    for i in range(cnt):
        for j in range(i+1, cnt):
            val = iou(items[i]['bbox'], items[j]['bbox'])
            if val > max_iou:
                max_iou = val
    # conclusion
    if cnt >= 6 and area_cv < 0.2 and vert_cv < 0.3 and max_iou < 0.1:
        concl = 'A'  # suitable for geometry experiment
    elif max_iou > 0.3:
        concl = 'B'  # keep tracking only, geometry risky
    else:
        concl = 'C'  # insufficient data
    # output
    print(f"Block {bid}:")
    print(f"  count = {cnt}")
    print(f"  area mean = {area_mean:.2f}, area cv = {area_cv:.3f}")
    print(f"  centers (sorted by y) = {centers}")
    print(f"  vertical gap mean = {vert_mean:.2f}, cv = {vert_cv:.3f}")
    print(f"  max IoU overlap = {max_iou:.3f}")
    print(f"  conclusion = {concl}\n")
