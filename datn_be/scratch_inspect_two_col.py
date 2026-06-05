import os
import json
import numpy as np
import cv2
from app.services.ai_engine import AIEngine, PANEL_CONF_THRESHOLD, DEFECT_CONF_THRESHOLD
from app.services.panel_processor import (
    refine_panels_by_string_lattice,
    refine_panels_by_two_column_block_lattice,
    refine_panel_contour,
    get_polygon_features,
)

def build_raw_panels(result, orig_img):
    """Replicate the raw panel extraction part of process_yolo_predictions.
    Returns list of panel dicts before any refinement.
    """
    masks_xy = result.masks.xy
    raw_panels = []
    for i in range(len(result.boxes)):
        class_id = int(result.boxes.cls[i])
        class_name = result.names[class_id]
        confidence = float(result.boxes.conf[i])
        bbox = result.boxes.xyxy[i].tolist()
        xy_polygon = masks_xy[i]
        if class_name.lower() != "panel":
            continue
        if confidence < PANEL_CONF_THRESHOLD or len(xy_polygon) < 3:
            continue
        image_path = getattr(result, "path", "unknown_image.jpg")
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        panel_poly = refine_panel_contour(
            orig_img, bbox, xy_polygon, idx=i, conf=confidence, image_name=image_name
        )
        features = get_polygon_features(panel_poly)
        panel_dict = {
            "class_name": class_name,
            "confidence": round(confidence, 4),
            "bbox": [round(v) for v in features["bbox"]],
            "polygon": panel_poly,
            "area": features["area"],
            "center": features["center"],
            "aspect_ratio": features["aspect_ratio"],
            "category": "panel",
            "box": [round(v) for v in bbox],
            "raw_yolo_poly": xy_polygon.tolist() if xy_polygon is not None else [],
            "raw_idx": i,
            "raw_conf": confidence,
            "final_polygon_source": "yolo_original",
            "final_polygon_stage": "yolo_refinement",
            "final_polygon_reason": "yolo_detection_contour",
        }
        raw_panels.append(panel_dict)
    return raw_panels

def polygon_stats(poly):
    pts = np.array(poly, dtype=np.float32)
    area = float(cv2.contourArea(pts.reshape(-1, 1, 2))) if pts.shape[0] >= 3 else 0.0
    centroid = pts.mean(axis=0).tolist()
    first4 = pts[:4]
    rounded = [list(map(int, np.round(pt))) for pt in first4]
    hash_str = "-".join([f"{x}_{y}" for x, y in rounded])
    return area, centroid, hash_str

def main():
    engine = AIEngine("weights/best.pt")
    img_path = "data/precalib/DJI_0843_R.JPG"
    result = engine.model.predict(
        source=img_path,
        conf=min(PANEL_CONF_THRESHOLD, DEFECT_CONF_THRESHOLD),
        save=False,
        verbose=False,
    )[0]
    orig_img = result.orig_img
    # Stage 1: raw YOLO panels
    raw = build_raw_panels(result, orig_img)
    # Stage 2: after string lattice refinement
    string_refined = refine_panels_by_string_lattice(
        panels=raw, image_shape=orig_img.shape, image_path=img_path
    )
    # Stage 3: after two‑column refinement
    two_col_refined = refine_panels_by_two_column_block_lattice(
        panels=string_refined, image_shape=orig_img.shape, image_path=img_path
    )
    # pick a panel where two‑col decided to use lattice (block_id may be present)
    target = None
    for p in two_col_refined:
        if p.get("two_col_decision") == "use_two_col_lattice":
            target = p
            break
    if not target:
        print("No panel with two_col_decision=use_two_col_lattice found.")
        return
    raw_idx = target["raw_idx"]
    stage_raw = next(p for p in raw if p["raw_idx"] == raw_idx)
    stage_string = next(p for p in string_refined if p["raw_idx"] == raw_idx)
    stage_two = target
    # compute stats
    def fmt(p):
        src = p.get("final_polygon_source", "?")
        area, cent, hsh = polygon_stats(p["polygon"])
        return src, f"{area:.2f}", f"{cent[0]:.1f},{cent[1]:.1f}", hsh
    rows = []
    prev_src = None
    for label, panel in [("YOLO", stage_raw), ("String", stage_string), ("TwoCol", stage_two)]:
        src, area, cent, hsh = fmt(panel)
        changed = "yes" if prev_src is not None and src != prev_src else "no"
        rows.append([label, src, area, cent, hsh, changed])
        prev_src = src
    # output table
    print("| Stage | Source | Area | Centroid | First4Hash | ChangedFromPrev |")
    print("|-------|--------|------|----------|------------|-----------------|")
    for r in rows:
        print(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} |")
    # conclusions
    if all(r[5] == "no" for r in rows[1:]):
        conclusion = "A) two_col does not change polygon"
    elif rows[2][5] == "yes" and rows[2][1] == rows[1][1]:
        conclusion = "B) two_col changes polygon but source not updated"
    else:
        conclusion = "C) two_col changes polygon then later overwritten"
    print("\nConclusion:", conclusion)

if __name__ == "__main__":
    main()
