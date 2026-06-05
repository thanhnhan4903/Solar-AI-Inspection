"""
Diagnostic: deep-compare production helper vs prototype for raw_idx 52 and 77.
Recomputes all candidates and explains reject reason.
"""
import json, math
import numpy as np
from pathlib import Path

try:
    from shapely.geometry import Polygon as ShapelyPoly
    SHAPELY = True
except ImportError:
    SHAPELY = False

TRACE  = Path("data/results/debug_logs/DJI_0029_R_final_polygon_trace.jsonl")
TARGET = {52, 77}

# ── load panels ───────────────────────────────────────────────────────────────
panels = []
with open(TRACE, encoding="utf-8") as f:
    for ln in f:
        ln = ln.strip()
        if ln:
            panels.append(json.loads(ln))

# ── good panel definitions ────────────────────────────────────────────────────
def is_good_prod(p):
    src = p.get("final_polygon_source", "")
    return (src in {"string_lattice_middle", "middle_locked_endpoint"}
            and not p.get("final_polygon_clamped")
            and not p.get("near_edge")
            and not (p.get("is_endpoint") or p.get("is_endpoint_panel")))

all_ws   = [p["bbox"][2]-p["bbox"][0] for p in panels]
all_hs   = [p["bbox"][3]-p["bbox"][1] for p in panels]
all_a    = [w*h for w,h in zip(all_ws,all_hs)]
all_asp  = [w/h if h>0 else 1.0 for w,h in zip(all_ws,all_hs)]
gmw,gmh  = float(np.median(all_ws)), float(np.median(all_hs))
gma,gasp = float(np.median(all_a)), float(np.median(all_asp))

def is_good_proto(p):
    src = p.get("final_polygon_source","")
    if src in {"string_lattice_middle","middle_locked_endpoint"}:
        return True
    if src == "yolo_original":
        w=p["bbox"][2]-p["bbox"][0]; h=p["bbox"][3]-p["bbox"][1]
        area=w*h; asp=w/h if h>0 else 1.0
        if (0.7*gasp<=asp<=1.3*gasp and 0.65*gma<=area<=1.35*gma
                and not p.get("final_polygon_clamped") and not p.get("near_edge")):
            return True
    return False

good_prod_all  = [p for p in panels if is_good_prod(p)]
good_proto_all = [p for p in panels if is_good_proto(p)]
good_prod_b0   = [p for p in good_prod_all  if p.get("block_id")==0]
good_proto_b0  = [p for p in good_proto_all if p.get("block_id")==0]

# ── median geometry ───────────────────────────────────────────────────────────
def median_geom(glist):
    ws,hs,angs=[],[],[]
    for p in glist:
        poly=p.get("polygon")
        if not poly or len(poly)<4: continue
        pts=np.array(poly,dtype=np.float32)
        edges=[(float(np.linalg.norm(pts[(i+1)%len(pts)]-pts[i])),pts[i],pts[(i+1)%len(pts)])
               for i in range(len(pts))]
        edges.sort(key=lambda x:x[0])
        hs.append((edges[0][0]+edges[1][0])/2)
        ws.append((edges[2][0]+edges[3][0])/2)
        dx=edges[3][2][0]-edges[3][1][0]; dy=edges[3][2][1]-edges[3][1][1]
        a=math.atan2(dy,dx)*180/math.pi
        if a>90: a-=180
        elif a<-90: a+=180
        if a>45: a-=90
        elif a<-45: a+=90
        angs.append(a)
    if len(ws)>=3:
        return float(np.median(ws)),float(np.median(hs)),float(np.median(angs))
    return None,None,None

mw_p,mh_p,ma_p = median_geom(good_prod_b0)
mw_t,mh_t,ma_t = median_geom(good_proto_b0)

print(f"\n{'='*70}")
print("GOOD PANELS COMPARISON")
print(f"{'='*70}")
print(f"  Production  all={len(good_prod_all):3d}  b0={len(good_prod_b0):3d}  w={mw_p:.1f} h={mh_p:.1f} a={ma_p:.2f}")
print(f"  Prototype   all={len(good_proto_all):3d}  b0={len(good_proto_b0):3d}  w={mw_t:.1f} h={mh_t:.1f} a={ma_t:.2f}")

# ── voting ────────────────────────────────────────────────────────────────────
def run_voting(plist, dom_a, is_perp, tol):
    cands=[]
    base=dom_a+90 if is_perp else dom_a
    for theta in np.arange(base-10,base+10.01,0.25):
        tr=math.radians(theta)
        projs=[]
        for i,gp in enumerate(plist):
            cx,cy=gp["center"]
            proj=(cx*math.sin(tr)-cy*math.cos(tr)) if is_perp else (-cx*math.sin(tr)+cy*math.cos(tr))
            projs.append((i,proj))
        projs.sort(key=lambda x:x[1])
        cur=[]
        for i,v in projs:
            if not cur: cur.append((i,v))
            elif v-cur[-1][1]<=tol: cur.append((i,v))
            else:
                cands.append(cur); cur=[(i,v)]
        if cur: cands.append(cur)
    res=[]
    for c in cands:
        idxs=[i for i,_ in c]; pvs=[v for _,v in c]
        pmean=float(np.mean(pvs))
        res.append({"angle":float(base),"projection":pmean,"support_count":len(c),
                    "mean_distance":float(np.mean([abs(v-pmean) for v in pvs])),
                    "raw_idxs":[plist[i].get("raw_idx") for i in idxs]})
    return res

def run_nms(cands,is_perp):
    cands.sort(key=lambda x:(-x["support_count"],x["mean_distance"]))
    kept=[]
    for c in cands:
        t1,p1=c["angle"],c["projection"]
        r1=math.radians(t1)
        ref1=(p1+256*math.cos(r1))/math.sin(r1) if is_perp else (p1+320*math.sin(r1))/math.cos(r1) if math.cos(r1)!=0 else 0
        dup=False
        for k in kept:
            t2,p2=k["angle"],k["projection"]
            r2=math.radians(t2)
            ref2=(p2+256*math.cos(r2))/math.sin(r2) if is_perp else (p2+320*math.sin(r2))/math.cos(r2) if math.cos(r2)!=0 else 0
            ad=abs(t1-t2); ad=360-ad if ad>180 else ad
            if ad<=2.0 and abs(ref1-ref2)<=10.0: dup=True; break
        if not dup: kept.append(c)
    return kept

# ── compute proposal ─────────────────────────────────────────────────────────
def compute_proposal(p, row_lines, col_lines, m_w, m_h, m_a, local_good,
                     col_top=1):
    """col_top: how many top col lines to try (1=best only, 3=top-3)"""
    cx,cy=p["center"]

    # Find top-N col lines
    col_cands=[]
    for line in col_lines:
        if line["support_count"]<4: continue
        r=math.radians(line["angle"])
        d=abs(cx*math.sin(r)-cy*math.cos(r)-line["projection"])
        col_cands.append((d,line))
    col_cands.sort(key=lambda x:x[0])

    # Find best row line
    min_A=float("inf"); best_A=None
    for line in row_lines:
        if line["support_count"]<4: continue
        r=math.radians(line["angle"])
        d=abs(-cx*math.sin(r)+cy*math.cos(r)-line["projection"])
        if d<min_A: min_A=d; best_A=line

    results=[]
    for col_rank,(min_B_val,best_B) in enumerate(col_cands[:col_top], 1):
        if not best_A:
            results.append({"col_rank":col_rank,"pass":False,"reason":"no_row_line",
                            "min_A":float("inf"),"min_B":min_B_val,
                            "support_A":0,"support_B":best_B["support_count"],
                            "col_proj":best_B["projection"],"col_raw_idxs":best_B.get("raw_idxs",[])})
            continue

        # Intersection
        rA=math.radians(best_A["angle"]); rB=math.radians(best_B["angle"])
        a1,b1,c1=-math.sin(rA),math.cos(rA),best_A["projection"]
        a2,b2,c2=math.sin(rB),-math.cos(rB),best_B["projection"]
        D=a1*b2-a2*b1
        epx,epy=((c1*b2-c2*b1)/D,(a1*c2-a2*c1)/D) if abs(D)>1e-5 else (cx,cy)

        # Polygon
        rad=math.radians(m_a); ca=math.cos(rad); sa=math.sin(rad)
        hw,hh=m_w/2,m_h/2
        pp=np.array([[lx*ca-ly*sa+epx,lx*sa+ly*ca+epy]
                     for lx,ly in [[-hw,-hh],[hw,-hh],[hw,hh],[-hw,hh]]],dtype=np.float32)
        px1,py1=float(np.min(pp[:,0])),float(np.min(pp[:,1]))
        px2,py2=float(np.max(pp[:,0])),float(np.max(pp[:,1]))

        ob=p.get("original_yolo_bbox") or p["bbox"]
        obw,obh=ob[2]-ob[0],ob[3]-ob[1]; ob_a=obw*obh
        ix1,iy1=max(ob[0],px1),max(ob[1],py1)
        ix2,iy2=min(ob[2],px2),min(ob[3],py2)
        inter=(ix2-ix1)*(iy2-iy1) if ix2>ix1 and iy2>iy1 else 0.0
        prop_a=(px2-px1)*(py2-py1)
        union=ob_a+prop_a-inter
        iou=inter/union if union>0 else 0.0
        shift=math.sqrt((cx-epx)**2+(cy-epy)**2)
        aratio=prop_a/max(ob_a,1)
        inside=(0<=px1 and px2<=640 and 0<=py1 and py2<=512)

        # overlap
        max_ov=0.0; ov_approx=False
        if SHAPELY:
            try:
                sp=ShapelyPoly(pp)
                if sp.is_valid:
                    for gp in local_good:
                        if gp.get("raw_idx")==p.get("raw_idx"): continue
                        gp2=ShapelyPoly(gp["polygon"])
                        if gp2.is_valid:
                            ia2=sp.intersection(gp2).area; ma2=min(sp.area,gp2.area)
                            if ma2>0: max_ov=max(max_ov,ia2/ma2)
                        else: ov_approx=True
                else: ov_approx=True
            except: ov_approx=True
        if not SHAPELY or ov_approx:
            for gp in local_good:
                if gp.get("raw_idx")==p.get("raw_idx"): continue
                gx1,gy1,gx2,gy2=gp["bbox"]
                bx1,by1=max(px1,gx1),max(py1,gy1)
                bx2,by2=min(px2,gx2),min(py2,gy2)
                if bx2>bx1 and by2>by1:
                    bi=(bx2-bx1)*(by2-by1); bg=(gx2-gx1)*(gy2-gy1); bp=(px2-px1)*(py2-py1)
                    bu=bg+bp-bi; max_ov=max(max_ov,bi/bu if bu>0 else 0.0)

        cA=min_A<=8.0; cB=min_B_val<=8.0; cI=iou>=0.40
        cS=shift<=0.35*obh; cAr=0.80<=aratio<=1.20
        cIn=inside; cOv=max_ov<=0.03
        passed=cA and cB and cI and cS and cAr and cIn and cOv
        reason="ok"
        if not passed:
            if not cA: reason="large_row_dist"
            elif not cB: reason="large_col_dist"
            elif not cI: reason="low_iou"
            elif not cS: reason="large_center_shift"
            elif not cAr: reason="area_ratio_out"
            elif not cIn: reason="outside_image"
            elif not cOv: reason="max_overlap_high"

        results.append({
            "col_rank": col_rank,
            "pass": passed, "reason": reason,
            "min_A": min_A, "min_B": min_B_val,
            "support_A": best_A["support_count"],
            "support_B": best_B["support_count"],
            "row_proj": best_A["projection"],
            "col_proj": best_B["projection"],
            "row_raw_idxs": best_A.get("raw_idxs",[]),
            "col_raw_idxs": best_B.get("raw_idxs",[]),
            "proposed_center": (epx,epy),
            "prop_bbox": [px1,py1,px2,py2],
            "iou": iou, "center_shift": shift, "area_ratio": aratio,
            "max_overlap": max_ov, "overlap_approx": ov_approx,
            "ob_h": obh,
        })
    if not results:
        results.append({"col_rank":0,"pass":False,"reason":"no_col_line",
                        "min_A":min_A if best_A else float("inf"),"min_B":float("inf"),
                        "support_A":best_A["support_count"] if best_A else 0,"support_B":0})
    return results

# ── pre-compute lines for both scenarios ─────────────────────────────────────
def build_lines(good_list, dom_a, tol_A, tol_B):
    row=run_nms(run_voting(good_list, dom_a, False, tol_A), False)
    col=run_nms(run_voting(good_list, dom_a, True,  tol_B), True)
    return row, col

tol_A_prod=0.20*(mh_p or gmh); tol_B_prod=0.15*(mw_p or gmw)
tol_A_prot=0.20*(mh_t or gmh); tol_B_prot=0.15*(mw_t or gmw)

row_prod_local,  col_prod_local  = build_lines(good_prod_b0,   ma_p or 0, tol_A_prod, tol_B_prod)
row_prod_global, _               = build_lines(good_prod_all,  ma_p or 0, tol_A_prod, tol_B_prod)
row_prot_global, col_prot_local  = build_lines(good_proto_all, ma_t or 0, tol_A_prot, tol_B_prot), \
                                   build_lines(good_proto_b0,  ma_t or 0, tol_A_prot, tol_B_prot)[1]
# fix tuple
row_prot_global = build_lines(good_proto_all, ma_t or 0, tol_A_prot, tol_B_prot)[0]

print(f"\n  Row lines: prod_local={len(row_prod_local)}  prod_global={len(row_prod_global)}  proto_global={len(row_prot_global)}")
print(f"  Col lines: prod_local={len(col_prod_local)}  proto_local={len(col_prot_local)}")

# ── per panel analysis ────────────────────────────────────────────────────────
target_panels = [p for p in panels if p.get("raw_idx") in TARGET]

for tp in sorted(target_panels, key=lambda x: x["raw_idx"]):
    rid = tp["raw_idx"]
    print(f"\n{'='*70}")
    print(f"RAW_IDX = {rid}")
    print(f"{'='*70}")

    # A. Production trace
    print("\n[A] PRODUCTION TRACE:")
    fields_A = ["raw_idx","original_yolo_bbox","bbox","polygon","final_polygon_reason",
                "geometry_gate_relaxed_accept","line_consensus_rescue_accept",
                "line_consensus_rescue_reject_reason","line_consensus_row_source",
                "line_consensus_global_row_used","line_consensus_row_support",
                "line_consensus_col_support","line_consensus_row_dist","line_consensus_col_dist",
                "line_consensus_iou","line_consensus_center_shift","line_consensus_area_ratio",
                "line_consensus_max_overlap"]
    for k in fields_A:
        v = tp.get(k, "N/A")
        print(f"  {k}: {v}")

    # B. Recompute candidates
    print("\n[B] RECOMPUTED CANDIDATES:")
    mw=mw_p or gmw; mh=mh_p or gmh; ma=ma_p or 0.0

    scenarios = [
        ("prod_local_row  + local_col",  row_prod_local,  col_prod_local,  "local",  mw,mh,ma, good_prod_b0),
        ("prod_global_row + local_col",  row_prod_global, col_prod_local,  "global", mw,mh,ma, good_prod_b0),
        ("proto_global_row+ proto_col",  row_prot_global, col_prot_local,  "proto",
         mw_t or gmw, mh_t or gmh, ma_t or 0.0, good_proto_b0),
    ]

    for sc_label, rlines, clines, src_tag, sc_mw, sc_mh, sc_ma, lgood in scenarios:
        # Top-3 col candidates for prod scenarios, top-1 for proto
        top = 3 if "proto" not in src_tag else 1
        results = compute_proposal(tp, rlines, clines, sc_mw, sc_mh, sc_ma, lgood, col_top=top)
        print(f"\n  [{sc_label}]  row_lines={len(rlines)}  col_lines={len(clines)}")
        for res in results:
            rank = res["col_rank"]
            tag = f"col_rank={rank}" if top > 1 else "best"
            passed = "PASS" if res["pass"] else f"FAIL:{res['reason']}"
            pcx = res.get("proposed_center", (0,0))
            print(f"    {tag}:  row_d={res['min_A']:.2f}(sup={res['support_A']})  "
                  f"col_d={res['min_B']:.2f}(sup={res['support_B']})  "
                  f"IoU={res.get('iou',0):.4f}  shift={res.get('center_shift',0):.2f}  "
                  f"aratio={res.get('area_ratio',0):.3f}  ov={res.get('max_overlap',0):.4f}  "
                  f"=> {passed}")
            if "proposed_center" in res:
                print(f"           center=({pcx[0]:.1f},{pcx[1]:.1f})  bbox={[round(v,1) for v in res.get('prop_bbox',[])]}  "
                      f"col_proj={res.get('col_proj',0):.1f}  row_proj={res.get('row_proj',0):.1f}")
            if top > 1 and "col_raw_idxs" in res:
                print(f"           col_supporting_raw_idxs={res['col_raw_idxs'][:8]}")

    # Summary
    print(f"\n[CONCLUSION for raw_idx={rid}]")
    cx,cy=tp["center"]
    print(f"  Current center:      ({cx:.1f},{cy:.1f})")
    print(f"  Block_id:            {tp.get('block_id')}  string_id: {tp.get('string_id')}")
    print(f"  Production reject:   {tp.get('line_consensus_rescue_reject_reason')}")
    print(f"  Global row used:     {tp.get('line_consensus_global_row_used')}")

print("\n\nDone.")
