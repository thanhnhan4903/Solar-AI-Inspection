"""
app/services/pv_panel_snapper.py
=================================
Adapter service: kết nối backend cũ với v62 line-snap engine.

Nhận image_path + yolo_panels -> gọi v62 process_image_for_backend() ->
trả về danh sách panels_snapped với outer_polygon và inner_polygon.

Quy tắc:
  - Không thay đổi thuật toán line-snap trong v62.
  - Không crash backend khi v62 thất bại (có fallback).
  - Trả về panel dict tương thích với schema backend cũ.
"""
from __future__ import annotations

import os
import logging
import uuid
import json
import tempfile
import shutil
# importlib.util removed
from pathlib import Path
from typing import List, Dict, Any, Optional

import cv2
import numpy as np

logger = logging.getLogger("solar_ai")

# ─────────────────────────────────────────────────────────────────────────────
# Import v62 geometry engine
# ─────────────────────────────────────────────────────────────────────────────
from app.services.pv_geometry_engine import process_image_for_backend


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _poly_to_list(poly) -> List[List[float]]:
    """Convert numpy array or nested list to plain Python list of [x, y]."""
    if poly is None:
        return []
    if isinstance(poly, np.ndarray):
        return [[float(pt[0]), float(pt[1])] for pt in poly]
    return [[float(pt[0]), float(pt[1])] for pt in poly]


def _poly_bbox(poly: List[List[float]]) -> List[float]:
    """Compute [x1, y1, x2, y2] bbox from polygon points."""
    if not poly:
        return [0.0, 0.0, 0.0, 0.0]
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return [min(xs), min(ys), max(xs), max(ys)]


def _poly_center(poly: List[List[float]]) -> List[float]:
    """Compute [cx, cy] center from polygon points."""
    if not poly:
        return [0.0, 0.0]
    cx = float(np.mean([p[0] for p in poly]))
    cy = float(np.mean([p[1] for p in poly]))
    return [cx, cy]


def _poly_area(poly: List[List[float]]) -> float:
    """Compute polygon area via cv2.contourArea."""
    if len(poly) < 3:
        return 0.0
    try:
        pts = np.array(poly, dtype=np.float32).reshape(-1, 1, 2)
        return float(cv2.contourArea(pts))
    except Exception:
        return 0.0


def _write_panel_log_jsonl(yolo_panels: List[Dict[str, Any]], out_path: str) -> None:
    """Write yolo_panels list as JSONL so v62 can load them as prior detections."""
    with open(out_path, "w", encoding="utf-8") as f:
        for p in yolo_panels:
            # v62 load_panels_from_logs() expects field "polygon" (4-point) and "bbox"
            bbox = p.get("bbox") or p.get("box", [0, 0, 0, 0])
            polygon = p.get("polygon") or p.get("raw_yolo_poly", [])
            # Ensure polygon is a plain list (not numpy)
            if hasattr(polygon, "tolist"):
                polygon = polygon.tolist()

            record = {
                "bbox": [float(v) for v in bbox[:4]] if len(bbox) >= 4 else [0.0, 0.0, 0.0, 0.0],
                "polygon": [[float(pt[0]), float(pt[1])] for pt in polygon] if polygon else [],
                "refined_polygon": [[float(pt[0]), float(pt[1])] for pt in polygon] if polygon else [],
                "confidence": float(p.get("confidence", p.get("raw_conf", 0.9))),
                "class_name": "panel",
                "category": "panel",
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main API
# ─────────────────────────────────────────────────────────────────────────────

def snap_panels_with_v61(
    image_path: str,
    yolo_panels: List[Dict[str, Any]],
    output_dir: str,
    image_stem: Optional[str] = None,
    debug: bool = False,
) -> List[Dict[str, Any]]:
    """
    Chạy thuật toán line-snap v62 và trả về danh sách panels đã snap.

    Args:
        image_path:   Đường dẫn tới file ảnh thermal.
        yolo_panels:  Danh sách panel dict từ YOLO (chỉ dùng làm detection prior).
        output_dir:   Thư mục để ghi file debug (nếu debug=True).
        image_stem:   Tên stem của ảnh (dùng cho file log). Mặc định tự suy ra.
        debug:        Nếu True, ghi ảnh debug line_snap và calc_inner_polygon.

    Returns:
        Danh sách panel dict với các field:
          - "polygon": outer_polygon (để tương thích backend cũ)
          - "outer_polygon": outer_polygon từ line-snap
          - "inner_polygon": inner_polygon (3px inset từ outer)
          - "calc_polygon": inner_polygon (alias)
          - "bbox": [x1, y1, x2, y2]
          - "center": [cx, cy]
          - "area": inner_area
          - "outer_area": float
          - "inner_area": float
          - "geometry_source": "v61_line_snap" | "yolo_fallback"
          - "valid": bool
          - "category": "panel"
          - "class_name": "panel"
          - "confidence": float

        Trả về [] nếu v62 thất bại (caller sẽ fallback về yolo_panels).
    """
    if not yolo_panels:
        logger.warning("[PV_SNAPPER] No yolo_panels provided, skipping v62 snap.")
        return []

    # Using statically imported process_image_for_backend

    if not os.path.exists(image_path):
        logger.warning(f"[PV_SNAPPER] Image not found: {image_path}")
        return []

    stem = image_stem or Path(image_path).stem
    os.makedirs(output_dir, exist_ok=True)

    # Write panel prior log (v62 reads this as its panel detection input)
    log_dir = os.path.join(output_dir, "debug_logs")
    os.makedirs(log_dir, exist_ok=True)
    panel_log_path = os.path.join(log_dir, f"{stem}_panel_refine.jsonl")

    try:
        _write_panel_log_jsonl(yolo_panels, panel_log_path)
    except Exception as e:
        logger.error(f"[PV_SNAPPER] Failed to write panel log: {e}")
        return []

    # Call v62 engine
    try:
        result = process_image_for_backend(
            image_path=image_path,
            panel_log_path=panel_log_path,
            output_dir=output_dir,
            debug=debug,
            log_level="quiet",
        )
    except Exception as e:
        logger.exception(f"[PV_SNAPPER] v62 process_image_for_backend raised: {e}")
        return []

    if not result.get("ok", False):
        err = result.get("error", "unknown error")
        logger.warning(f"[PV_SNAPPER] v62 returned ok=False: {err}")
        return []

    raw_panels = result.get("panels", [])
    if not raw_panels:
        logger.warning(f"[PV_SNAPPER] v62 returned 0 panels for stem={stem}.")
        return []

    # Convert to backend-compatible panel dicts
    snapped: List[Dict[str, Any]] = []
    for p in raw_panels:
        # V62 area/geometry filter đã quyết định panel nào hợp lệ.
        # Không trả panel valid=False về backend/frontend.
        if p.get("valid") is False:
            continue

        outer = _poly_to_list(p.get("outer_polygon", []))
        inner = _poly_to_list(p.get("inner_polygon", []) or outer)

        if not outer or len(outer) < 3:
            continue

        outer_area = p.get("outer_area") or _poly_area(outer)
        inner_area = p.get("inner_area") or _poly_area(inner)
        bbox = _poly_bbox(outer)
        center = _poly_center(outer)

        snap_dict: Dict[str, Any] = {
            # Backward-compat: backend reads panel["polygon"]
            "polygon": outer,
            "outer_polygon": outer,
            "inner_polygon": inner,
            "calc_polygon": inner,

            "bbox": [round(v, 1) for v in bbox],
            "box": [round(v, 1) for v in bbox],
            "center": [round(center[0], 2), round(center[1], 2)],

            # area = inner_area so downstream metric calcs use safe inner region
            "area": round(inner_area, 2),
            "outer_area": round(outer_area, 2),
            "inner_area": round(inner_area, 2),

            "geometry_source": "v61_line_snap",
            "snap_source": p.get("source", "line_snap"),
            "snap_engine": p.get("engine", result.get("engine", "v62")),
            "valid": bool(p.get("valid", True)),
            "filter_reason": p.get("filter_reason", ""),

            "category": "panel",
            "class_name": "panel",
            "confidence": 1.0,  # geometry is certain, overrides YOLO conf

            # Placeholder fields for downstream compat
            "defects": [],
            "id": str(uuid.uuid4()),

            # v62 metadata
            "snap_block_id": None,
            "snap_row": None,
            "snap_col": None,
        }
        snapped.append(snap_dict)

    summary = result.get("summary", {})
    logger.info(
        f"[PV_SNAPPER] stem={stem} route={result.get('route')} engine={result.get('engine')} "
        f"n_total={summary.get('n_total', len(raw_panels))} "
        f"n_valid={summary.get('n_valid')} "
        f"n_returned={len(snapped)}"
    )

    try:
        n_valid = int(summary.get("n_valid", -1))
        if n_valid >= 0 and len(snapped) != n_valid:
            logger.warning(
                f"[PV_SNAPPER] returned panels mismatch: n_valid={n_valid}, n_returned={len(snapped)}"
            )
    except Exception:
        pass

    return snapped
