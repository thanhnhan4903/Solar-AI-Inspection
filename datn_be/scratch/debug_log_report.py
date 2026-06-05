import os, json, glob, pathlib
from collections import Counter

def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue

def analyze_file(filepath):
    records = list(load_jsonl(filepath))
    total = len(records)
    fps_counts = Counter(rec.get('final_polygon_source') for rec in records)
    gg_accept_counts = Counter()
    for rec in records:
        if 'geometry_gate_accept' in rec:
            gg_accept_counts[str(rec['geometry_gate_accept'])] += 1
        else:
            gg_accept_counts['missing'] += 1
    gg_reason_counts = Counter(rec.get('geometry_gate_reason') for rec in records if rec.get('geometry_gate_reason') is not None)
    fp_reason_rej_counts = Counter()
    for rec in records:
        reason = rec.get('final_polygon_reason','')
        if isinstance(reason,str) and reason.startswith('geometry_gate_rejected'):
            fp_reason_rej_counts[reason] += 1
    cnt_string_lattice_middle = sum(1 for rec in records if rec.get('final_polygon_source')=='string_lattice_middle')
    cnt_yolo_original = sum(1 for rec in records if rec.get('final_polygon_source')=='yolo_original')
    near_edge_counts = Counter()
    for rec in records:
        if 'near_edge' in rec:
            near_edge_counts[str(rec['near_edge'])] += 1
    # first 30 matching records
    sample = []
    for rec in records:
        if (rec.get('geometry_gate_accept') is False) or \
           (isinstance(rec.get('final_polygon_reason',''), str) and rec['final_polygon_reason'].startswith('geometry_gate_rejected')) or \
           (rec.get('final_polygon_source')=='string_lattice_middle'):
            sample.append({
                'image_name': rec.get('image_name'),
                'raw_idx': rec.get('raw_idx'),
                'block_id': rec.get('block_id'),
                'bbox': rec.get('bbox'),
                'polygon': rec.get('polygon'),
                'final_polygon_source': rec.get('final_polygon_source'),
                'final_polygon_stage': rec.get('final_polygon_stage'),
                'final_polygon_reason': rec.get('final_polygon_reason'),
                'geometry_gate_accept': rec.get('geometry_gate_accept'),
                'geometry_gate_reason': rec.get('geometry_gate_reason'),
                'iou': rec.get('iou'),
                'center_shift': rec.get('center_shift'),
                'area_ratio': rec.get('area_ratio'),
                'max_overlap': rec.get('max_overlap'),
                'near_edge': rec.get('near_edge')
            })
            if len(sample) >= 30:
                break
    return {
        'total_records': total,
        'final_polygon_source_counts': dict(fps_counts),
        'geometry_gate_accept_counts': dict(gg_accept_counts),
        'geometry_gate_reason_counts': dict(gg_reason_counts),
        'final_polygon_reason_rejected_counts': dict(fp_reason_rej_counts),
        'string_lattice_middle_count': cnt_string_lattice_middle,
        'yolo_original_count': cnt_yolo_original,
        'near_edge_counts': dict(near_edge_counts),
        'sample_records': sample
    }

def main():
    base_dir = pathlib.Path('c:/Solar_Inspection_Project/datn_be/data/results/debug_logs')
    # find *_final_polygon_trace.jsonl files
    final_files = sorted(base_dir.glob('*_final_polygon_trace.jsonl'))
    if not final_files:
        final_files = sorted(base_dir.glob('*_string_lattice_refine.jsonl'))
    # sort by modification time descending
    final_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    # take newest 5 (or all if fewer)
    selected = final_files[:5]
    print(f"Selected {len(selected)} file(s) for analysis:\n")
    for f in selected:
        print(f"File: {f.name}")
        stats = analyze_file(str(f))
        print(f"  Total records: {stats['total_records']}")
        print(f"  final_polygon_source counts: {stats['final_polygon_source_counts']}")
        print(f"  geometry_gate_accept counts: {stats['geometry_gate_accept_counts']}")
        print(f"  geometry_gate_reason counts: {stats['geometry_gate_reason_counts']}")
        print(f"  final_polygon_reason (rejected) counts: {stats['final_polygon_reason_rejected_counts']}")
        print(f"  string_lattice_middle count: {stats['string_lattice_middle_count']}")
        print(f"  yolo_original count: {stats['yolo_original_count']}")
        print(f"  near_edge counts: {stats['near_edge_counts']}")
        print("  First up to 30 matching records:")
        for rec in stats['sample_records']:
            print('    -', rec)
        print('\n')

if __name__ == '__main__':
    main()
