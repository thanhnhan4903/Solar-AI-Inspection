import sys
sys.path.insert(0, '.')

from app.services.ai_engine import AIEngine, PANEL_CONF_THRESHOLD, DEFECT_CONF_THRESHOLD
from app.services.panel_processor import process_yolo_predictions
from shapely.geometry import Polygon as SPolygon
from shapely.validation import make_valid
from app.services.defect_logic import DEFECT_PANEL_OVERLAP_THRESHOLD

print(f"Thresholds: panel={PANEL_CONF_THRESHOLD}, defect={DEFECT_CONF_THRESHOLD}, overlap={DEFECT_PANEL_OVERLAP_THRESHOLD}")

engine = AIEngine("weights/best.pt")

for img_path in ["data/precalib/DJI_0953.JPG", "data/precalib/DJI_0957.JPG"]:
    print(f"\n=== {img_path} ===")
    global_conf = min(PANEL_CONF_THRESHOLD, DEFECT_CONF_THRESHOLD)
    results = engine.model.predict(source=img_path, conf=global_conf, save=False, verbose=False)
    result = results[0]

    detections = process_yolo_predictions(
        result=result,
        orig_img=result.orig_img,
        panel_conf=PANEL_CONF_THRESHOLD,
        defect_conf=DEFECT_CONF_THRESHOLD,
        panel_class_name="panel"
    )

    panels = [d for d in detections if d["category"] == "panel"]
    defects = [d for d in detections if d["category"] == "defect"]
    print(f"  process_yolo_predictions: {len(panels)} panels, {len(defects)} defects")

    for d in defects:
        cls = d["class_name"]
        poly = d["polygon"]
        print(f"\n  Defect [{cls}]: {len(poly)} points")

        if len(poly) < 3:
            print("    -> Skip: too few points")
            continue

        try:
            d_geom = SPolygon(poly)
            if not d_geom.is_valid:
                d_geom = make_valid(d_geom)
            print(f"    -> d_geom area={d_geom.area:.1f}, type={d_geom.geom_type}, valid={d_geom.is_valid}")
        except Exception as e:
            print(f"    -> FAILED to build defect geom: {e}")
            continue

        if d_geom.area == 0:
            print("    -> ZERO AREA! Will be skipped.")
            continue

        best_ratio = 0
        for p in panels:
            p_poly = p.get("polygon", [])
            if len(p_poly) < 3:
                continue
            try:
                p_geom = SPolygon(p_poly)
                if not p_geom.is_valid:
                    p_geom = make_valid(p_geom)
                inter = p_geom.intersection(d_geom)
                ratio = inter.area / d_geom.area if d_geom.area > 0 else 0
                if ratio > best_ratio:
                    best_ratio = ratio
            except Exception as ex:
                print(f"      -> intersection error: {ex}")

        print(f"    -> Best overlap={best_ratio:.6f}, threshold={DEFECT_PANEL_OVERLAP_THRESHOLD}, assigned={best_ratio >= DEFECT_PANEL_OVERLAP_THRESHOLD}")
