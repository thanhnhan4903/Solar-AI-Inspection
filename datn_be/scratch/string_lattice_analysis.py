import json, pathlib, collections

FIELDS = ['decision', 'reason', 'fallback_reason', 'string_quality', 'pass_ratio']
TARGET_FIELDS = ['use_string_lattice', 'string_quality_failed', 'pass_ratio', 'n_lt', 'n_cols', 'fallback', 'lattice_projection']

def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue

def analyze_file(filepath):
    records = list(load_jsonl(filepath))
    total = len(records)
    # collect optional fields stats
    opt_counts = {field: collections.Counter() for field in FIELDS}
    for rec in records:
        for field in FIELDS:
            if field in rec:
                opt_counts[field][str(rec[field])] += 1
    # count categories
    cnt_final_source = collections.Counter(rec.get('final_polygon_source') for rec in records)
    cnt_stage = collections.Counter(rec.get('final_polygon_stage') for rec in records)
    cnt_reason = collections.Counter(rec.get('final_polygon_reason') for rec in records)
    cnt_string_id = collections.Counter(rec.get('string_id') for rec in records)
    cnt_block_id = collections.Counter(rec.get('block_id') for rec in records)
    # find records with any of TARGET_FIELDS present (truthy)
    related = []
    for rec in records:
        if any(rec.get(key) for key in TARGET_FIELDS):
            related.append(rec)
    # first 30 records where reason/fallback mentions not using lattice
    first30 = []
    for rec in records:
        reason = rec.get('reason') or rec.get('fallback_reason')
        if reason and ('lattice' in str(reason).lower()):
            first30.append(rec)
            if len(first30) >= 30:
                break
    return {
        'total_records': total,
        'optional_field_counts': {k: dict(v) for k, v in opt_counts.items()},
        'final_polygon_source_counts': dict(cnt_final_source),
        'final_polygon_stage_counts': dict(cnt_stage),
        'final_polygon_reason_counts': dict(cnt_reason),
        'string_id_counts': dict(cnt_string_id),
        'block_id_counts': dict(cnt_block_id),
        'related_record_count': len(related),
        'first30_lattice_related': [
            {
                'image_name': rec.get('image_name'),
                'raw_idx': rec.get('raw_idx'),
                'block_id': rec.get('block_id'),
                'string_id': rec.get('string_id'),
                'reason': rec.get('reason'),
                'fallback_reason': rec.get('fallback_reason'),
                'final_polygon_source': rec.get('final_polygon_source'),
                'final_polygon_stage': rec.get('final_polygon_stage'),
                'final_polygon_reason': rec.get('final_polygon_reason'),
                'use_string_lattice': rec.get('use_string_lattice'),
                'string_quality_failed': rec.get('string_quality_failed'),
                'pass_ratio': rec.get('pass_ratio'),
                'n_lt': rec.get('n_lt'),
                'n_cols': rec.get('n_cols'),
                'fallback': rec.get('fallback'),
                'lattice_projection': rec.get('lattice_projection')
            }
            for rec in first30
        ]
    }

def main():
    base = pathlib.Path('c:/Solar_Inspection_Project/datn_be/data/results/debug_logs')
    files = [
        'DJI_0843_R_string_lattice_refine.jsonl',
        'DJI_0845_R_string_lattice_refine.jsonl',
        'DJI_0087_R_string_lattice_refine.jsonl',
        'DJI_0029_R_string_lattice_refine.jsonl'
    ]
    for name in files:
        path = base / name
        if not path.exists():
            print(f"File not found: {name}")
            continue
        stats = analyze_file(str(path))
        print(f"=== {name} ===")
        print(f"Total records: {stats['total_records']}")
        print("Optional field counts:")
        for field, cnt in stats['optional_field_counts'].items():
            if cnt:
                print(f"  {field}: {cnt}")
        print("final_polygon_source counts:", stats['final_polygon_source_counts'])
        print("final_polygon_stage counts:", stats['final_polygon_stage_counts'])
        print("final_polygon_reason counts:", stats['final_polygon_reason_counts'])
        print("string_id counts (non-null):", {k:v for k,v in stats['string_id_counts'].items() if k is not None})
        print("block_id counts (non-null):", {k:v for k,v in stats['block_id_counts'].items() if k is not None})
        print("Related records (any lattice flags):", stats['related_record_count'])
        print("First 30 records with lattice-related reason/fallback:")
        for rec in stats['first30_lattice_related']:
            print('  -', rec)
        print('\n')

if __name__ == '__main__':
    main()
