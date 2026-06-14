#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
yolo_panel_mask_polygon_export.py

Purpose
-------
Run YOLO on thermal PV images and export ONLY panel/module detections as
panel_refine-style logs with bbox, mask polygon, OBB, and pixel-scale metrics.

This script is intended to run immediately after YOLO/refine so we can decide
whether an image should go through the v34-style small/UAV grid route or the
v42-style close-up route.

Important
---------
- This script does NOT export defect detections.
- This script does NOT use YOLO polygon as final geometry.
- YOLO mask/polygon is exported only as prior data: scale, orientation, and
  search region for later line-snap.

Default working resolution is 640 x 512, as requested. Images are resized to
that resolution before YOLO inference, so all exported coordinates and areas
are in the 640x512 coordinate system unless you change --work-width/--work-height.

Example
-------
python yolo_panel_mask_polygon_export.py ^
  --model runs/segment/train/weights/best.pt ^
  --images "D:\Image\B5\B5\test" ^
  --out "C:\Solar_Inspection_Project\datn_be\data\results\debug_logs_panelmask" ^
  --panel-class-names panel module solar_panel pv_panel

If your model uses class IDs instead of useful names:
python yolo_panel_mask_polygon_export.py ^
  --model best.pt ^
  --images "D:\Image\B5\B5\test\DJI_0953.JPG" ^
  --out debug_logs_panelmask ^
  --panel-class-ids 0
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except Exception as exc:  # pragma: no cover
    YOLO = None
    YOLO_IMPORT_ERROR = exc
else:
    YOLO_IMPORT_ERROR = None


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_PANEL_NAMES = {
    "panel", "panels", "module", "modules", "pv_panel", "solar_panel",
    "pv module", "pv_module", "solar module", "solar_module", "photovoltaic_panel",
}
DEFAULT_EXCLUDE_NAMES = {
    "hotspot", "hot spot", "fault", "defect", "cell", "diode", "soiling",
    "shadow", "shading", "crack", "cracking", "vegetation", "offline",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export YOLO panel mask/polygon/bbox metrics at a fixed working resolution."
    )
    p.add_argument("--model", required=True, help="Path to YOLO detection/segmentation model, e.g. best.pt")
    p.add_argument("--images", nargs="+", required=True, help="Image file(s) or folder(s).")
    p.add_argument("--out", required=True, help="Output folder for *_panel_refine.jsonl, summary CSV, and debug images.")
    p.add_argument("--work-width", type=int, default=640, help="Working image width. Default: 640")
    p.add_argument("--work-height", type=int, default=512, help="Working image height. Default: 512")
    p.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size. Default: 640")
    p.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold. Default: 0.25")
    p.add_argument("--iou", type=float, default=0.50, help="YOLO NMS IoU threshold. Default: 0.50")
    p.add_argument(
        "--panel-class-names",
        nargs="*",
        default=sorted(DEFAULT_PANEL_NAMES),
        help="Accepted panel class names. Case-insensitive. If omitted, uses common panel/module names.",
    )
    p.add_argument(
        "--panel-class-ids",
        nargs="*",
        type=int,
        default=None,
        help="Accepted panel class IDs. Use this if your class names are not useful.",
    )
    p.add_argument(
        "--exclude-class-names",
        nargs="*",
        default=sorted(DEFAULT_EXCLUDE_NAMES),
        help="Class names to always ignore, useful when one model detects panels and defects.",
    )
    p.add_argument("--max-det", type=int, default=300, help="YOLO max detections per image. Default: 300")
    p.add_argument("--save-debug", action="store_true", help="Save debug overlay images.")
    p.add_argument("--no-jsonl", action="store_true", help="Do not write per-image JSONL files.")
    p.add_argument("--min-aspect", type=float, default=1.25, help="Optional rough panel aspect filter lower bound.")
    p.add_argument("--max-aspect", type=float, default=3.20, help="Optional rough panel aspect filter upper bound.")
    p.add_argument(
        "--no-aspect-filter",
        action="store_true",
        help="Disable aspect-ratio filtering after class selection.",
    )
    return p.parse_args()


def iter_images(paths: Sequence[str]) -> List[Path]:
    out: List[Path] = []
    for s in paths:
        p = Path(s)
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            out.append(p)
        elif p.is_dir():
            for q in sorted(p.rglob("*")):
                if q.is_file() and q.suffix.lower() in IMAGE_EXTS:
                    out.append(q)
        else:
            print(f"[WARN] image path not found or unsupported: {p}")
    # de-duplicate while preserving order
    seen = set()
    unique: List[Path] = []
    for p in out:
        key = str(p.resolve()).lower()
        if key not in seen:
            unique.append(p)
            seen.add(key)
    return unique


def normalize_name(s: str) -> str:
    return str(s).strip().lower().replace("-", "_")


def get_class_name(names: Any, cls_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(cls_id, cls_id))
    if isinstance(names, (list, tuple)) and 0 <= cls_id < len(names):
        return str(names[cls_id])
    return str(cls_id)


def should_keep_class(
    cls_id: int,
    cls_name: str,
    panel_ids: Optional[Sequence[int]],
    panel_names: set[str],
    exclude_names: set[str],
) -> bool:
    n = normalize_name(cls_name)
    n_space = n.replace("_", " ")
    if panel_ids is not None:
        return int(cls_id) in set(int(x) for x in panel_ids)
    if n in exclude_names or n_space in exclude_names:
        return False
    if n in panel_names or n_space in panel_names:
        return True
    # Safe convenience: accept names containing both pv/solar and panel/module.
    if ("panel" in n or "module" in n) and not any(bad in n for bad in exclude_names):
        return True
    return False


def contour_from_mask(mask_u8: np.ndarray) -> Optional[np.ndarray]:
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    c = contours[0]
    if cv2.contourArea(c) <= 1:
        return None
    return c


def approx_polygon_from_contour(contour: np.ndarray, epsilon_frac: float = 0.008) -> np.ndarray:
    peri = cv2.arcLength(contour, True)
    eps = max(1.0, float(epsilon_frac) * peri)
    approx = cv2.approxPolyDP(contour, eps, True)
    return approx.reshape(-1, 2).astype(float)


def obb_from_points(points: np.ndarray) -> Tuple[List[List[float]], float, float, float]:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    rect = cv2.minAreaRect(pts)
    box = cv2.boxPoints(rect).astype(float)
    (cx, cy), (w, h), angle = rect
    # Normalize angle for long-side orientation.
    long_side = max(float(w), float(h))
    short_side = min(float(w), float(h))
    theta = float(angle)
    if w < h:
        theta += 90.0
    # Map roughly to [-90, 90)
    while theta >= 90.0:
        theta -= 180.0
    while theta < -90.0:
        theta += 180.0
    return box.tolist(), float(theta), float(long_side), float(short_side)


def polygon_area(poly: Sequence[Sequence[float]]) -> float:
    if not poly or len(poly) < 3:
        return 0.0
    return abs(float(cv2.contourArea(np.asarray(poly, dtype=np.float32))))


def bbox_polygon(x1: float, y1: float, x2: float, y2: float) -> List[List[float]]:
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def route_from_metrics(med_long: float, med_short: float, med_area: float, med_frac: float, count: int) -> str:
    # Conservative router discussed in the conversation.
    if (
        med_long >= 130.0
        and med_short >= 60.0
        and med_area >= 8000.0
        and med_frac >= 0.035
        and count <= 25
    ):
        return "local_closeup_v42_candidate"
    return "core_v34_candidate"


def summarize_image(image_stem: str, rows: List[Dict[str, Any]], image_w: int, image_h: int) -> Dict[str, Any]:
    image_area = float(image_w * image_h)
    if rows:
        longs = np.array([max(r["bbox_w"], r["bbox_h"]) for r in rows], dtype=float)
        shorts = np.array([min(r["bbox_w"], r["bbox_h"]) for r in rows], dtype=float)
        areas = np.array([r["bbox_area"] for r in rows], dtype=float)
        fracs = areas / max(image_area, 1.0)
        aspects = np.array([r["aspect"] for r in rows], dtype=float)
        med_long = float(np.median(longs))
        med_short = float(np.median(shorts))
        med_area = float(np.median(areas))
        med_frac = float(np.median(fracs))
        med_aspect = float(np.median(aspects))
    else:
        med_long = med_short = med_area = med_frac = med_aspect = 0.0
    route = route_from_metrics(med_long, med_short, med_area, med_frac, len(rows))
    return {
        "image": image_stem,
        "image_w": image_w,
        "image_h": image_h,
        "image_area": image_w * image_h,
        "panel_count": len(rows),
        "median_long_side_px": round(med_long, 3),
        "median_short_side_px": round(med_short, 3),
        "median_bbox_area_px2": round(med_area, 3),
        "median_bbox_area_frac": round(med_frac, 6),
        "median_aspect": round(med_aspect, 3),
        "recommended_geometry_route": route,
    }


def draw_debug(img: np.ndarray, rows: List[Dict[str, Any]]) -> np.ndarray:
    vis = img.copy()
    for i, r in enumerate(rows):
        poly = r.get("mask_polygon") or r.get("bbox_polygon")
        if poly and len(poly) >= 2:
            pts = np.asarray(poly, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(vis, [pts], True, (0, 255, 255), 1, cv2.LINE_AA)
        obb = r.get("obb")
        if obb:
            pts = np.asarray(obb, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(vis, [pts], True, (255, 255, 0), 1, cv2.LINE_AA)
        x1, y1, x2, y2 = [int(round(v)) for v in r["bbox"]]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(vis, f"{i}:{r['bbox_w']:.0f}x{r['bbox_h']:.0f}", (x1, max(10, y1 - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1, cv2.LINE_AA)
    return vis


def process_image(
    model: Any,
    image_path: Path,
    out_dir: Path,
    args: argparse.Namespace,
    panel_names: set[str],
    exclude_names: set[str],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Cannot read image: {image_path}")
    work_w, work_h = int(args.work_width), int(args.work_height)
    img = cv2.resize(bgr, (work_w, work_h), interpolation=cv2.INTER_AREA)
    image_area = float(work_w * work_h)

    results = model.predict(
        source=img,
        imgsz=int(args.imgsz),
        conf=float(args.conf),
        iou=float(args.iou),
        max_det=int(args.max_det),
        verbose=False,
    )
    if not results:
        return summarize_image(image_path.stem, [], work_w, work_h), []
    res = results[0]
    names = getattr(res, "names", getattr(model, "names", {}))
    boxes = getattr(res, "boxes", None)
    masks = getattr(res, "masks", None)
    rows: List[Dict[str, Any]] = []

    if boxes is None or len(boxes) == 0:
        return summarize_image(image_path.stem, [], work_w, work_h), []

    # Extract masks as numpy arrays if available.
    mask_arrays: Optional[np.ndarray] = None
    if masks is not None and getattr(masks, "data", None) is not None:
        try:
            mask_arrays = masks.data.detach().cpu().numpy()
        except Exception:
            mask_arrays = None

    xyxy = boxes.xyxy.detach().cpu().numpy()
    confs = boxes.conf.detach().cpu().numpy() if boxes.conf is not None else np.ones((len(xyxy),), dtype=float)
    clss = boxes.cls.detach().cpu().numpy().astype(int) if boxes.cls is not None else np.zeros((len(xyxy),), dtype=int)

    for i, (box, conf, cls_id) in enumerate(zip(xyxy, confs, clss)):
        cls_name = get_class_name(names, int(cls_id))
        if not should_keep_class(int(cls_id), cls_name, args.panel_class_ids, panel_names, exclude_names):
            continue
        x1, y1, x2, y2 = [float(v) for v in box]
        x1 = max(0.0, min(x1, work_w - 1.0))
        x2 = max(0.0, min(x2, work_w - 1.0))
        y1 = max(0.0, min(y1, work_h - 1.0))
        y2 = max(0.0, min(y2, work_h - 1.0))
        bw = max(0.0, x2 - x1)
        bh = max(0.0, y2 - y1)
        if bw < 2 or bh < 2:
            continue
        long_side = max(bw, bh)
        short_side = min(bw, bh)
        aspect = long_side / max(short_side, 1e-6)
        if not args.no_aspect_filter and not (args.min_aspect <= aspect <= args.max_aspect):
            continue

        bbox_area = float(bw * bh)
        row: Dict[str, Any] = {
            "image": image_path.stem,
            "idx": len(rows),
            "source_index": int(i),
            "class_id": int(cls_id),
            "class_name": cls_name,
            "confidence": round(float(conf), 6),
            "bbox": [round(x1, 3), round(y1, 3), round(x2, 3), round(y2, 3)],
            "bbox_w": round(bw, 3),
            "bbox_h": round(bh, 3),
            "bbox_area": round(bbox_area, 3),
            "bbox_area_frac": round(bbox_area / max(image_area, 1.0), 6),
            "aspect": round(aspect, 4),
            "bbox_polygon": bbox_polygon(round(x1, 3), round(y1, 3), round(x2, 3), round(y2, 3)),
            "has_mask": False,
            "mask_area": None,
            "mask_area_frac": None,
            "mask_polygon": None,
            "mask_polygon_area": None,
            "mask_polygon_area_frac": None,
            "obb": None,
            "obb_angle_deg": None,
            "obb_long_side": None,
            "obb_short_side": None,
        }

        # Prefer true YOLO segmentation mask if available.
        if mask_arrays is not None and i < len(mask_arrays):
            m = mask_arrays[i]
            if m.shape[:2] != (work_h, work_w):
                m = cv2.resize(m.astype(np.float32), (work_w, work_h), interpolation=cv2.INTER_LINEAR)
            mask_u8 = (m > 0.5).astype(np.uint8) * 255
            c = contour_from_mask(mask_u8)
            if c is not None:
                approx = approx_polygon_from_contour(c)
                mask_area = int(np.count_nonzero(mask_u8))
                poly = [[round(float(x), 3), round(float(y), 3)] for x, y in approx.tolist()]
                obb, angle, olong, oshort = obb_from_points(approx)
                row.update({
                    "has_mask": True,
                    "mask_area": int(mask_area),
                    "mask_area_frac": round(float(mask_area) / max(image_area, 1.0), 6),
                    "mask_polygon": poly,
                    "mask_polygon_area": round(polygon_area(poly), 3),
                    "mask_polygon_area_frac": round(polygon_area(poly) / max(image_area, 1.0), 6),
                    "obb": [[round(float(x), 3), round(float(y), 3)] for x, y in obb],
                    "obb_angle_deg": round(float(angle), 4),
                    "obb_long_side": round(float(olong), 3),
                    "obb_short_side": round(float(oshort), 3),
                })
        if row["obb"] is None:
            # Fallback OBB from bbox polygon. Angle will be 0/90-ish; useful mainly as placeholder.
            bp = np.asarray(row["bbox_polygon"], dtype=np.float32)
            obb, angle, olong, oshort = obb_from_points(bp)
            row.update({
                "obb": [[round(float(x), 3), round(float(y), 3)] for x, y in obb],
                "obb_angle_deg": round(float(angle), 4),
                "obb_long_side": round(float(olong), 3),
                "obb_short_side": round(float(oshort), 3),
            })
        rows.append(row)

    # Sort top-to-bottom then left-to-right to be stable.
    rows.sort(key=lambda r: ((r["bbox"][1] + r["bbox"][3]) * 0.5, (r["bbox"][0] + r["bbox"][2]) * 0.5))
    for j, r in enumerate(rows):
        r["idx"] = j

    summary = summarize_image(image_path.stem, rows, work_w, work_h)

    if not args.no_jsonl:
        jsonl_path = out_dir / f"{image_path.stem}_panel_refine.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        summary_path = out_dir / f"{image_path.stem}_panel_summary.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.save_debug:
        debug = draw_debug(img, rows)
        cv2.imwrite(str(out_dir / f"debug_{image_path.stem}_yolo_panel_mask.JPG"), debug)

    return summary, rows


def main() -> None:
    args = parse_args()
    if YOLO is None:
        raise RuntimeError(f"Cannot import ultralytics.YOLO: {YOLO_IMPORT_ERROR}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    image_paths = iter_images(args.images)
    if not image_paths:
        raise RuntimeError("No images found.")

    panel_names = {normalize_name(x) for x in args.panel_class_names}
    panel_names |= {x.replace("_", " ") for x in list(panel_names)}
    exclude_names = {normalize_name(x) for x in args.exclude_class_names}
    exclude_names |= {x.replace("_", " ") for x in list(exclude_names)}

    print(f"[LOAD_MODEL] {args.model}")
    model = YOLO(args.model)
    print(f"[IMAGES] n={len(image_paths)} work={args.work_width}x{args.work_height} out={out_dir}")
    print(f"[CLASSES] panel_ids={args.panel_class_ids} panel_names={sorted(panel_names)}")

    all_summaries: List[Dict[str, Any]] = []
    all_rows_flat: List[Dict[str, Any]] = []

    for p in image_paths:
        try:
            summary, rows = process_image(model, p, out_dir, args, panel_names, exclude_names)
            all_summaries.append(summary)
            for r in rows:
                flat = {k: v for k, v in r.items() if k not in {"bbox_polygon", "mask_polygon", "obb"}}
                flat["file"] = str(p)
                all_rows_flat.append(flat)
            print(
                "[IMAGE_SUMMARY] "
                f"image={p.stem} panels={summary['panel_count']} "
                f"med_long={summary['median_long_side_px']} med_short={summary['median_short_side_px']} "
                f"med_area={summary['median_bbox_area_px2']} "
                f"med_frac={summary['median_bbox_area_frac']:.4f} "
                f"route={summary['recommended_geometry_route']}"
            )
        except Exception as exc:
            print(f"[ERROR] image={p} {type(exc).__name__}: {exc}")

    # Write image-level summary CSV.
    csv_path = out_dir / "panel_scale_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "image", "image_w", "image_h", "image_area", "panel_count",
            "median_long_side_px", "median_short_side_px", "median_bbox_area_px2",
            "median_bbox_area_frac", "median_aspect", "recommended_geometry_route",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_summaries)

    # Write panel-level flat CSV.
    panel_csv_path = out_dir / "panel_instances_metrics.csv"
    with panel_csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "file", "image", "idx", "source_index", "class_id", "class_name", "confidence",
            "bbox", "bbox_w", "bbox_h", "bbox_area", "bbox_area_frac", "aspect",
            "has_mask", "mask_area", "mask_area_frac", "mask_polygon_area", "mask_polygon_area_frac",
            "obb_angle_deg", "obb_long_side", "obb_short_side",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows_flat)

    print(f"[DONE] images={len(all_summaries)} panel_instances={len(all_rows_flat)}")
    print(f"[OUT] {csv_path}")
    print(f"[OUT] {panel_csv_path}")


if __name__ == "__main__":
    main()
