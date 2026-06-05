import json, sys

path = r'c:/Solar_Inspection_Project/datn_be/data/results/debug_logs/DJI_0843_R_final_polygon_trace.jsonl'
records = []
with open(path, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))

total = len(records)
min_x1 = min(r['bbox'][0] for r in records)
max_x2 = max(r['bbox'][2] for r in records)
min_y1 = min(r['bbox'][1] for r in records)
max_y2 = max(r['bbox'][3] for r in records)

poly_x_vals = []
poly_y_vals = []
for r in records:
    if 'polygon' in r and r['polygon']:
        for pt in r['polygon']:
            poly_x_vals.append(pt[0])
            poly_y_vals.append(pt[1])
    if 'refined_polygon' in r and r['refined_polygon']:
        for pt in r['refined_polygon']:
            poly_x_vals.append(pt[0])
            poly_y_vals.append(pt[1])

if poly_x_vals:
    min_px = min(poly_x_vals)
    max_px = max(poly_x_vals)
    min_py = min(poly_y_vals)
    max_py = max(poly_y_vals)
else:
    min_px = max_px = min_py = max_py = None

out_x1 = sum(1 for r in records if r['bbox'][0] < 0)
out_y1 = sum(1 for r in records if r['bbox'][1] < 0)
out_x2 = sum(1 for r in records if r['bbox'][2] > 640)
out_y2 = sum(1 for r in records if r['bbox'][3] > 512)

first10 = []
for r in records[:10]:
    entry = {
        'raw_idx': r.get('raw_idx'),
        'bbox': r.get('bbox'),
        'polygon': r.get('polygon') if 'polygon' in r else None,
        'refined_polygon': r.get('refined_polygon') if 'refined_polygon' in r else None,
        'final_polygon_source': r.get('final_polygon_source'),
        'final_polygon_stage': r.get('final_polygon_stage'),
        'single_column_tracking_candidate': r.get('single_column_tracking_candidate')
    }
    first10.append(entry)

out = {
    'total_records': total,
    'bbox_min_x1': min_x1,
    'bbox_max_x2': max_x2,
    'bbox_min_y1': min_y1,
    'bbox_max_y2': max_y2,
    'polygon_min_x': min_px,
    'polygon_max_x': max_px,
    'polygon_min_y': min_py,
    'polygon_max_y': max_py,
    'out_of_bounds_counts': {
        'x1_lt_0': out_x1,
        'y1_lt_0': out_y1,
        'x2_gt_640': out_x2,
        'y2_gt_512': out_y2
    },
    'first_10_records': first10
}
print(json.dumps(out, indent=2))
