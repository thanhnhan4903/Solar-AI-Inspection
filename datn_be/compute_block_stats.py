import json, os
from collections import defaultdict

def load_log(path):
    entries = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    return entries

def compute_stats(image_name):
    string_log = os.path.join('data', 'results', 'debug_logs', f"{image_name}_string_lattice_refine.jsonl")
    entries = load_log(string_log)
    # group by block_id (string_id)
    blocks = defaultdict(list)
    block_info = {}
    for e in entries:
        if e.get('record_type') == 'string':
            block_id = e['string_id']
            block_info[block_id] = e
        elif e.get('record_type') == 'panel':
            block_id = e['string_id']
            blocks[block_id].append(e)
    result = []
    for b_id, panels in blocks.items():
        info = block_info.get(b_id, {})
        decision = info.get('decision')
        n = info.get('n_panels')
        n_cols = info.get('n_cols')
        n_rows = info.get('n_rows')
        pitch_cv = info.get('gap_cv')  # gap_cv corresponds to pitch variance
        rail_width_cv = info.get('rail_low_shift')  # placeholder; not directly logged
        # median original coverage for middle panels (ignore missing values)
        middle_cov = [p.get('original_coverage_ratio') for p in panels
                      if p.get('panel_role') == 'middle' and p.get('original_coverage_ratio') is not None]
        median_middle_cov = sorted(middle_cov)[len(middle_cov)//2] if middle_cov else None
        # max inward cut ratio for middle panels (ignore missing)
        cuts = [p.get('max_inward_cut_ratio') for p in panels
                if p.get('panel_role') == 'middle' and p.get('max_inward_cut_ratio') is not None]
        max_cut = max(cuts) if cuts else None
        # max overlap among all panels in the block
        overlaps = [p.get('max_overlap') for p in panels if p.get('max_overlap') is not None]
        max_overlap = max(overlaps) if overlaps else None
        fallback_reason = info.get('fallback_reason')
        # final geometry source counts per block
        src_counts = defaultdict(int)
        for p in panels:
            src = p.get('final_polygon_source')
            if src:
                src_counts[src] += 1
        result.append({
            'block_id': b_id,
            'decision': decision,
            'n': n,
            'n_cols': n_cols,
            'n_rows': n_rows,
            'pitch_cv': pitch_cv,
            'rail_width_cv': rail_width_cv,
            'median_middle_original_coverage': median_middle_cov,
            'max_middle_inward_cut_ratio': max_cut,
            'max_overlap': max_overlap,
            'fallback_reason': fallback_reason,
            'final_geometry_source_counts': dict(src_counts)
        })
    return result

if __name__ == '__main__':
    for img in ['DJI_0845_R', 'DJI_0843_R']:
        stats = compute_stats(img)
        print(f"=== {img} ===")
        for blk in stats:
            print(blk)
