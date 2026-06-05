import json
from pathlib import Path
from collections import Counter

BAD_RAW_IDX = {25, 40, 46, 47, 52, 62, 65, 67, 68, 77, 78, 86}
DEBUG_DIR = Path("data/results/debug_logs")

# ── 1. DJI_0029_R 12-panel table ──────────────────────────────────────────────
trace_path = DEBUG_DIR / "DJI_0029_R_final_polygon_trace.jsonl"
print("\n" + "=" * 90)
print("DJI_0029_R — 12 bad panels rescue report")
print("=" * 90)

rows = {}
with open(trace_path, encoding="utf-8") as f:
    for line in f:
        p = json.loads(line)
        if p.get("raw_idx") in BAD_RAW_IDX:
            rows[p["raw_idx"]] = p

header = (f"{'idx':>5}  {'final_polygon_reason':<34}  {'rescue':>6}  "
          f"{'row_src':<6}  {'global':>6}  {'IoU':>6}  {'row_d':>6}  {'col_d':>6}  {'reject_reason'}")
print(header)
print("-" * len(header))

for ridx in sorted(BAD_RAW_IDX):
    p = rows.get(ridx)
    if not p:
        print(f"{ridx:>5}  NOT IN TRACE")
        continue
    print(
        f"{ridx:>5}  "
        f"{str(p.get('final_polygon_reason','')):<34}  "
        f"{str(p.get('line_consensus_rescue_accept','')):>6}  "
        f"{str(p.get('line_consensus_row_source','')):.<6}  "
        f"{str(p.get('line_consensus_global_row_used','')):>6}  "
        f"{p.get('line_consensus_iou', 0.0):>6.4f}  "
        f"{p.get('line_consensus_row_dist', 0.0) or 0.0:>6.2f}  "
        f"{p.get('line_consensus_col_dist', 0.0) or 0.0:>6.2f}  "
        f"{str(p.get('line_consensus_rescue_reject_reason',''))}"
    )

# ── 2. Source counts regression ─────────────────────────────────────────────
for img in ["DJI_0843_R", "DJI_0845_R", "DJI_0087_R"]:
    trace = DEBUG_DIR / f"{img}_final_polygon_trace.jsonl"
    print(f"\n{'=' * 60}")
    print(f"{img} — final_polygon_source counts")
    print(f"{'=' * 60}")
    if not trace.exists():
        print(f"  TRACE NOT FOUND")
        continue
    counts = Counter()
    with open(trace, encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            counts[p.get("final_polygon_source", "unknown")] += 1
    for src, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {src:<40} {cnt}")

print("\nDone.")
