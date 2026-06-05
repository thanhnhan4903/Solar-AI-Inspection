"""
Export overlay for DJI_0029_R showing rescue result.
Green = line_consensus_rescue accept
Yellow = rescue reject (kept old polygon)
Red = yolo_original (no rescue)
Blue = other good panels
"""
import json, cv2, numpy as np
from pathlib import Path

IMAGE_PATH  = Path("data/precalib/DJI_0029_R.JPG")
TRACE_PATH  = Path("data/results/debug_logs/DJI_0029_R_final_polygon_trace.jsonl")
OUT_PATH    = Path("data/results/debug/debug_DJI_0029_R_line_consensus_rescue_overlay.JPG")
BAD_RAW_IDX = {25, 40, 46, 47, 52, 62, 65, 67, 68, 77, 78, 86}

img = cv2.imread(str(IMAGE_PATH))
assert img is not None, f"Image not found: {IMAGE_PATH}"
overlay = img.copy()

panels = []
with open(TRACE_PATH, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            panels.append(json.loads(line))

for p in panels:
    poly = p.get("polygon")
    if not poly or len(poly) < 3:
        continue

    pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
    ridx = p.get("raw_idx")
    reason = p.get("final_polygon_reason", "")
    rescue_accept = p.get("line_consensus_rescue_accept")

    if ridx in BAD_RAW_IDX:
        if rescue_accept is True:
            color = (0, 220, 0)       # bright green — rescued
            thickness = 2
        else:
            color = (0, 165, 255)     # orange — rescue rejected, kept old
            thickness = 2
    elif reason == "line_consensus_rescue":
        color = (0, 200, 100)         # teal — rescued (not in bad list)
        thickness = 1
    elif p.get("final_polygon_source") == "yolo_original":
        color = (0, 0, 200)           # red — yolo original
        thickness = 1
    else:
        color = (180, 180, 180)       # grey — other
        thickness = 1

    cv2.polylines(overlay, [pts], isClosed=True, color=color, thickness=thickness)

    # Label bad panels
    if ridx in BAD_RAW_IDX:
        cx = int(np.mean([pt[0] for pt in poly]))
        cy = int(np.mean([pt[1] for pt in poly]))
        label = f"{ridx}"
        cv2.putText(overlay, label, (cx-8, cy+4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1, cv2.LINE_AA)

# Legend
legend = [
    ((0, 220, 0),   "Rescue ACCEPT (bad panels)"),
    ((0, 165, 255), "Rescue REJECT (kept old polygon)"),
    ((0, 0, 200),   "yolo_original"),
    ((180, 180, 180), "Other panels"),
]
for i, (c, label) in enumerate(legend):
    y = 16 + i * 16
    cv2.rectangle(overlay, (6, y-10), (18, y+2), c, -1)
    cv2.putText(overlay, label, (22, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, c, 1, cv2.LINE_AA)

# Blend
result = cv2.addWeighted(overlay, 0.85, img, 0.15, 0)
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(OUT_PATH), result, [cv2.IMWRITE_JPEG_QUALITY, 92])
print(f"Saved: {OUT_PATH}")
print(f"  Green (rescue accept): {sum(1 for p in panels if p.get('raw_idx') in BAD_RAW_IDX and p.get('line_consensus_rescue_accept') is True)}")
print(f"  Orange (rescue reject): {sum(1 for p in panels if p.get('raw_idx') in BAD_RAW_IDX and p.get('line_consensus_rescue_accept') is False)}")
