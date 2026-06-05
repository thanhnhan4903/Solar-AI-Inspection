"""
Diagnostic: compare production helper vs prototype for raw_idx 46, 52, 77.
Key hypotheses:
  - Production uses LOCAL row voting (block_id==0 panels only)
  - Prototype uses GLOBAL row voting (all good panels)
  - This may cause different row lines -> different proposed_center -> different IoU / col_dist
"""
import json, math, collections
import numpy as np
from pathlib import Path

try:
    from shapely.geometry import Polygon as ShapelyPolygon
    SHAPELY = True
except ImportError:
    SHAPELY = False

TRACE  = Path("data/results/debug_logs/DJI_0029_R_final_polygon_trace.jsonl")
TARGET_IDX = {46, 52, 77}
ALL_12     = {25, 40, 46, 47, 52, 62, 65, 67, 68, 77, 78, 86}

# ── load all panels ──────────────────────────────────────────────────────────
panels = []
with open(TRACE, encoding="utf-8") as f:
    for ln in f:
        ln = ln.strip()
        if ln:
            panels.append(json.loads(ln))

# ── define "good" as production does (final_polygon_source clean, not endpoint, not clamped) ──
def is_good_prod(p):
    src = p.get("final_polygon_source", "")
    if src not in {"string_lattice_middle", "middle_locked_endpoint", "line_consensus_rescue"}:
        return False
    if p.get("is_endpoint") or p.get("is_endpoint_panel"):
        return False
    if p.get("final_polygon_clamped"):
        return False
    if p.get("near_edge"):
        return False
    return True

# ── define "good" as prototype does ──────────────────────────────────────────
def compute_bbox_stats(panels_list):
    ws = [p["bbox"][2]-p["bbox"][0] for p in panels_list]
    hs = [p["bbox"][3]-p["bbox"][1] for p in panels_list]
    areas = [w*h for w,h in zip(ws,hs)]
    aspects = [w/h if h>0 else 1.0 for w,h in zip(ws,hs)]
    return (float(np.median(ws)), float(np.median(hs)),
            float(np.median(areas)), float(np.median(aspects)))

gmw, gmh, gma, gmaspect = compute_bbox_stats(panels)

def is_good_proto(p):
    src = p.get("final_polygon_source", "")
    if src in {"string_lattice_middle", "middle_locked_endpoint"}:
        return True
    if src == "yolo_original":
        w = p["bbox"][2]-p["bbox"][0]; h = p["bbox"][3]-p["bbox"][1]
        area = w*h; asp = w/h if h>0 else 1.0
        if (0.7*gmaspect <= asp <= 1.3*gmaspect and
            0.65*gma <= area <= 1.35*gma and
            not p.get("final_polygon_clamped") and
            not p.get("near_edge")):
            return True
    return False

good_prod_all  = [p for p in panels if is_good_prod(p)]
good_proto_all = [p for p in panels if is_good_proto(p)]

good_prod_b0   = [p for p in good_prod_all  if p.get("block_id") == 0]
good_proto_b0  = [p for p in good_proto_all if p.get("block_id") == 0]

print(f"\n{'='*70}")
print(f"GOOD PANELS COMPARISON")
print(f"{'='*70}")
print(f"  Production  all:    {len(good_prod_all):3d}   block_id==0: {len(good_prod_b0):3d}")
print(f"  Prototype   all:    {len(good_proto_all):3d}   block_id==0: {len(good_proto_b0):3d}")

# ── orientation-aware median geometry ────────────────────────────────────────
def median_geometry(good_list):
    widths, heights, angles = [], [], []
    for p in good_list:
        poly = p.get("polygon")
        if not poly or len(poly) < 4:
            continue
        pts = np.array(poly, dtype=np.float32)
        edges = []
        for i in range(len(pts)):
            pA, pB = pts[i], pts[(i+1)%len(pts)]
            edges.append((float(np.linalg.norm(pB-pA)), pA, pB))
        edges.sort(key=lambda x: x[0])
        heights.append((edges[0][0]+edges[1][0])/2.0)
        widths.append((edges[2][0]+edges[3][0])/2.0)
        dx = edges[3][2][0]-edges[3][1][0]
        dy = edges[3][2][1]-edges[3][1][1]
        ang = math.atan2(dy,dx)*180/math.pi
        if ang > 90: ang -= 180
        elif ang < -90: ang += 180
        if ang > 45: ang -= 90
        elif ang < -45: ang += 90
        angles.append(ang)
    if len(widths) >= 3:
        return float(np.median(widths)), float(np.median(heights)), float(np.median(angles))
    return None, None, None

mw_prod, mh_prod, ma_prod = median_geometry(good_prod_b0)
mw_proto, mh_proto, ma_proto = median_geometry(good_proto_b0)

print(f"\n{'='*70}")
print(f"MEDIAN GEOMETRY COMPARISON  (from block_id==0 good panels)")
print(f"{'='*70}")
print(f"  {'':20s}  {'width':>8}  {'height':>8}  {'angle':>8}")
print(f"  {'Production':20s}  {mw_prod or 0:8.2f}  {mh_prod or 0:8.2f}  {ma_prod or 0:8.2f}")
print(f"  {'Prototype':20s}  {mw_proto or 0:8.2f}  {mh_proto or 0:8.2f}  {ma_proto or 0:8.2f}")

# ── run voting ────────────────────────────────────────────────────────────────
def run_voting(panel_list, dominant_angle, is_perpendicular, tol):
    candidates = []
    for p in panel_list:
        cx, cy = p["center"]
        # project center onto the consensus axis
        r = math.radians(dominant_angle)
        if is_perpendicular:
            proj = cx * math.sin(r) - cy * math.cos(r)
        else:
            proj = -cx * math.sin(r) + cy * math.cos(r)
        added = False
        for c in candidates:
            angle_diff = abs(dominant_angle - c["angle"])
            if angle_diff > 180: angle_diff = 360 - angle_diff
            if angle_diff <= 2.0 and abs(proj - c["mean_distance"]) <= tol:
                c["support_count"] += 1
                c["projections"].append(proj)
                c["angles"].append(dominant_angle)
                c["mean_distance"] = float(np.mean(c["projections"]))
                c["angle"] = float(np.mean(c["angles"]))
                added = True
                break
        if not added:
            candidates.append({
                "angle": dominant_angle, "projection": proj,
                "mean_distance": proj, "support_count": 1,
                "projections": [proj], "angles": [dominant_angle],
            })
    for c in candidates:
        c["projection"] = c["mean_distance"]
    return candidates

def run_nms(candidates, is_perpendicular):
    candidates.sort(key=lambda x: (-x["support_count"], x["mean_distance"]))
    kept = []
    for c in candidates:
        dup = False
        t1, p1 = c["angle"], c["projection"]
        r1 = math.radians(t1)
        ref1 = (p1+256*math.cos(r1))/math.sin(r1) if is_perpendicular else (p1+320*math.sin(r1))/math.cos(r1) if math.cos(r1)!=0 else 0
        for k in kept:
            t2, p2 = k["angle"], k["projection"]
            r2 = math.radians(t2)
            ref2 = (p2+256*math.cos(r2))/math.sin(r2) if is_perpendicular else (p2+320*math.sin(r2))/math.cos(r2) if math.cos(r2)!=0 else 0
            ad = abs(t1-t2); ad = 360-ad if ad>180 else ad
            if ad <= 2.0 and abs(ref1-ref2) <= 10.0:
                dup = True; break
        if not dup:
            kept.append(c)
    return kept

# ── per-scenario voting ───────────────────────────────────────────────────────
scenarios = {
    "prod_local_row+local_col": {
        "row_panels": good_prod_b0, "col_panels": good_prod_b0,
        "mw": mw_prod, "mh": mh_prod, "ma": ma_prod,
        "label": "Production (local row + local col, block0 only)",
    },
    "proto_global_row+local_col": {
        "row_panels": good_proto_all, "col_panels": good_proto_b0,
        "mw": mw_proto, "mh": mh_proto, "ma": ma_proto,
        "label": "Prototype (global row + local col block0)",
    },
}

iou_thresh     = 0.40
row_dist_thresh = 8.0
col_dist_thresh = 8.0

def compute_proposal(p, row_lines, col_lines, m_w, m_h, m_a):
    cx, cy = p["center"]
    min_dist_A, best_A = float("inf"), None
    for line in row_lines:
        if line["support_count"] < 4: continue
        r = math.radians(line["angle"])
        d = abs(-cx*math.sin(r) + cy*math.cos(r) - line["projection"])
        if d < min_dist_A:
            min_dist_A, best_A = d, line
    min_dist_B, best_B = float("inf"), None
    for line in col_lines:
        if line["support_count"] < 4: continue
        r = math.radians(line["angle"])
        d = abs(cx*math.sin(r) - cy*math.cos(r) - line["projection"])
        if d < min_dist_B:
            min_dist_B, best_B = d, line

    proposed_center = (cx, cy)
    if best_A and best_B:
        rA = math.radians(best_A["angle"]); rB = math.radians(best_B["angle"])
        a1=-math.sin(rA); b1=math.cos(rA); c1=best_A["projection"]
        a2=math.sin(rB);  b2=-math.cos(rB); c2=best_B["projection"]
        D = a1*b2 - a2*b1
        if abs(D) > 1e-5:
            px = (c1*b2 - c2*b1)/D
            py = (a1*c2 - a2*c1)/D
            proposed_center = (px, py)

    pcx, pcy = proposed_center
    rad = math.radians(m_a); ca=math.cos(rad); sa=math.sin(rad)
    hw=m_w/2; hh=m_h/2
    poly = [[-hw,-hh],[hw,-hh],[hw,hh],[-hw,hh]]
    prop = [[lx*ca-ly*sa+pcx, lx*sa+ly*ca+pcy] for lx,ly in poly]
    prop = np.array(prop, dtype=np.float32)

    px1,py1 = float(np.min(prop[:,0])),float(np.min(prop[:,1]))
    px2,py2 = float(np.max(prop[:,0])),float(np.max(prop[:,1]))

    ob = p["bbox"]; obw=ob[2]-ob[0]; obh=ob[3]-ob[1]; ob_area=obw*obh
    ix1=max(ob[0],px1); iy1=max(ob[1],py1)
    ix2=min(ob[2],px2); iy2=min(ob[3],py2)
    inter=max(0,(ix2-ix1))*max(0,(iy2-iy1))
    prop_area=(px2-px1)*(py2-py1)
    union=ob_area+prop_area-inter
    iou=inter/union if union>0 else 0.0
    shift=math.sqrt((cx-pcx)**2+(cy-pcy)**2)
    area_ratio=prop_area/max(ob_area,1)

    return {
        "proposed_center": proposed_center,
        "prop_bbox": [px1,py1,px2,py2],
        "iou": iou, "center_shift": shift, "area_ratio": area_ratio,
        "min_dist_A": min_dist_A, "min_dist_B": min_dist_B,
        "support_A": best_A["support_count"] if best_A else 0,
        "support_B": best_B["support_count"] if best_B else 0,
        "row_line_proj": best_A["projection"] if best_A else None,
        "col_line_proj": best_B["projection"] if best_B else None,
    }

# ── run both scenarios for 3 target panels ───────────────────────────────────
target_panels = [p for p in panels if p.get("raw_idx") in TARGET_IDX]

for sc_key, sc in scenarios.items():
    mw = sc["mw"] or gmw; mh = sc["mh"] or gmh; ma = sc["ma"] or 0.0
    tol_A = 0.20*mh; tol_B = 0.15*mw
    row_lines = run_nms(run_voting(sc["row_panels"], ma, False, tol_A), False)
    col_lines = run_nms(run_voting(sc["col_panels"], ma, True,  tol_B), True)
    sc["row_lines"] = row_lines
    sc["col_lines"] = col_lines
    sc["mw_eff"] = mw; sc["mh_eff"] = mh; sc["ma_eff"] = ma
    print(f"\n  [{sc_key}] row_lines={len(row_lines)}  col_lines={len(col_lines)}  median_w={mw:.1f} h={mh:.1f} a={ma:.2f}")

print(f"\n{'='*70}")
print(f"PER-PANEL COMPARISON FOR raw_idx {{46, 52, 77}}")
print(f"{'='*70}")

for tp in sorted(target_panels, key=lambda x: x["raw_idx"]):
    rid = tp["raw_idx"]
    print(f"\n--- raw_idx={rid} ---")
    print(f"  bbox:          {tp['bbox']}")
    print(f"  center:        {tp['center']}")
    print(f"  block_id:      {tp.get('block_id')}  string_id: {tp.get('string_id')}")
    print(f"  relaxed_accept:{tp.get('geometry_gate_relaxed_accept')}  final_reason: {tp.get('final_polygon_reason')}")
    print()

    for sc_key, sc in scenarios.items():
        res = compute_proposal(tp, sc["row_lines"], sc["col_lines"],
                               sc["mw_eff"], sc["mh_eff"], sc["ma_eff"])
        iou_ok = res["iou"] >= iou_thresh
        row_ok = res["min_dist_A"] <= row_dist_thresh
        col_ok = res["min_dist_B"] <= col_dist_thresh
        verdict = "PASS" if (iou_ok and row_ok and col_ok) else "FAIL"
        print(f"  [{sc_key}]")
        print(f"    proposed_center: ({res['proposed_center'][0]:.1f}, {res['proposed_center'][1]:.1f})")
        print(f"    prop_bbox:       [{', '.join(f'{v:.1f}' for v in res['prop_bbox'])}]")
        print(f"    row_dist={res['min_dist_A']:.2f} (supp={res['support_A']}, proj={res['row_line_proj']})  OK={row_ok}")
        print(f"    col_dist={res['min_dist_B']:.2f} (supp={res['support_B']}, proj={res['col_line_proj']})  OK={col_ok}")
        print(f"    IoU={res['iou']:.4f}  shift={res['center_shift']:.2f}  area_ratio={res['area_ratio']:.3f}  => {verdict}")

    # Production trace values for reference
    print(f"  [PRODUCTION trace actual]")
    print(f"    row_dist={tp.get('line_consensus_row_dist'):.4f}  col_dist={tp.get('line_consensus_col_dist'):.4f}")
    print(f"    IoU={tp.get('line_consensus_iou'):.4f}  shift={tp.get('line_consensus_center_shift'):.4f}")
    print(f"    reject_reason: {tp.get('line_consensus_rescue_reject_reason')}")

print(f"\n{'='*70}")
print("CONCLUSIONS")
print(f"{'='*70}")
for tp in sorted(target_panels, key=lambda x: x["raw_idx"]):
    rid = tp["raw_idx"]
    reason = tp.get("line_consensus_rescue_reject_reason")
    prod_row = tp.get("line_consensus_row_dist", 0)
    prod_col = tp.get("line_consensus_col_dist", 0)
    prod_iou = tp.get("line_consensus_iou", 0)
    print(f"\n  raw_idx={rid}  reason={reason}")
    if reason == "low_iou":
        print(f"    prod IoU={prod_iou:.4f} < 0.40 threshold")
        print(f"    => proposed center likely shifted far from current bbox center")
        print(f"    => compare prod vs proto proposed_center above to see if global row voting fixes it")
    elif reason == "large_col_dist":
        print(f"    prod col_dist={prod_col:.4f} > 8.0 threshold")
        print(f"    => no column consensus line close enough to this panel's center")
        print(f"    => check if panel is isolated from good panels in column direction")
