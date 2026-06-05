import json
target_idxs = {25, 40, 46, 47, 52, 62, 65, 67, 68, 77, 78, 86}
with open('data/results/debug_logs/DJI_0029_R_final_polygon_trace.jsonl', 'r') as f:
    for line in f:
        p = json.loads(line)
        if p['raw_idx'] in target_idxs:
            print(f"raw_idx: {p['raw_idx']}, string_id: {p.get('string_id')}, bbox: {p['bbox']}")
