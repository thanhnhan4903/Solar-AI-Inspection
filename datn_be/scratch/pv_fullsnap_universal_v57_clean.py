#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pv_fullsnap_universal_v57_clean.py

Clean unified PV panel geometry runner for Drone Solar.

Purpose
-------
V57 is a maintainable refactor direction after v55/v56:
- one file;
- no embedded _ENGINE_V34_LINES / _ENGINE_V42_LINES;
- no subprocess;
- no temporary engine materialization;
- one unified pipeline with a single route if:
      rep_area_frac > 0.025  -> large_local
      rep_area_frac <= 0.025 -> small_core

Important geometry contract
---------------------------
YOLO/panel logs are used as priors only: bbox/polygon gives search regions,
scale routing, and block layout. Final panel polygons are built from fitted
shared rails and fitted row lines, then intersected as:
    TL = top row line    ∩ left rail
    TR = top row line    ∩ right rail
    BR = bottom row line ∩ right rail
    BL = bottom row line ∩ left rail

The inner/calc polygon is a geometric inset from the final snapped polygon.
Debug outputs are kept compatible with the existing backend:
    debug_<stem>_line_snap.JPG
    debug_<stem>_calc_inner_polygon.JPG

Notes
-----
This is intentionally a clean baseline. It keeps the core ideas of v34/v42,
not every legacy fallback/debug branch. If one edge case regresses, add a
small named guard to this file instead of re-embedding old engines.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# OpenCV uses BGR.
COLOR_RAIL = (255, 0, 0)       # blue
COLOR_ROW = (255, 0, 255)      # magenta
COLOR_OUTER = (0, 255, 255)    # yellow
COLOR_INNER = (255, 255, 0)    # cyan


@dataclass
class GeometryConfig:
    route_area_frac_threshold: float = 0.025
    representative_top_area_ratio: float = 0.40
    visual_thickness: int = 1
    inner_panel_margin_px: float = 2.0

    # small_core / v34 essence
    small_row_x_step_px: float = 0.85
    small_row_dy_step_px: float = 0.5
    small_row_search_radius_px: float = 6.0
    small_rail_y_step_px: float = 3.0
    small_rail_search_radius_px: float = 7.0
    small_max_row_angle_deg: float = 4.0

    # large_local / v42 essence
    large_row_x_step_px: float = 2.0
    large_row_dy_step_px: float = 0.75
    large_row_search_radius_px: float = 4.0
    large_rail_y_step_px: float = 4.0
    large_rail_search_radius_px: float = 5.0
    large_max_row_angle_deg: float = 3.0

    # block grouping
    x_column_gap_factor: float = 1.35
    y_row_merge_factor: float = 0.55
    block_gap_factor: float = 1.70

    # validation / fallback guards
    min_panel_area_px: float = 20.0
    max_center_shift_ratio: float = 0.35
    max_outer_expand_px: float = 5.0


@dataclass
class PanelPrior:
    idx: int
    bbox: Tuple[float, float, float, float]
    polygon: Optional[np.ndarray] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def cx(self) -> float:
        x1, _, x2, _ = self.bbox
        return 0.5 * (x1 + x2)

    @property
    def cy(self) -> float:
        _, y1, _, y2 = self.bbox
        return 0.5 * (y1 + y2)

    @property
    def w(self) -> float:
        x1, _, x2, _ = self.bbox
        return max(0.0, x2 - x1)

    @property
    def h(self) -> float:
        _, y1, _, y2 = self.bbox
        return max(0.0, y2 - y1)

    @property
    def area(self) -> float:
        return self.w * self.h


@dataclass
class LineYC:
    """Line as y = m*x + c."""
    m: float
    c: float
    source: str = "unknown"
    support: int = 0
    residual: float = 0.0

    def y_at(self, x: float) -> float:
        return float(self.m * x + self.c)


@dataclass
class RailXY:
    """Rail as polyline points [(x,y), ...], generally near-vertical."""
    pts: List[Tuple[float, float]]
    source: str = "unknown"
    support: int = 0
    residual: float = 0.0


@dataclass
class PanelGeometry:
    idx: int
    row: int
    col: int
    route: str
    outer_polygon: np.ndarray
    inner_polygon: np.ndarray
    source: str
    valid: bool = True


@dataclass
class BlockGeometry:
    block_id: int
    route: str
    columns: List[List[PanelPrior]]
    rows: List[List[PanelPrior]]
    rails: List[RailXY]
    row_lines: List[LineYC]
    panels: List[PanelGeometry]


# ---------------------------------------------------------------------------
# IO / parsing
# ---------------------------------------------------------------------------

def read_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    rows.append(obj)
        return rows

    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        for key in ("panels", "detections", "items", "records"):
            val = data.get(key)
            if isinstance(val, list):
                return [x for x in val if isinstance(x, dict)]
        return [data]
    return []


def bbox_from_record(p: Dict[str, Any]) -> Optional[Tuple[float, float, float, float]]:
    bbox = p.get("bbox") or p.get("xyxy") or p.get("box")
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        try:
            x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
            if x2 > x1 and y2 > y1:
                return x1, y1, x2, y2
        except Exception:
            return None
    return None


def polygon_from_record(p: Dict[str, Any]) -> Optional[np.ndarray]:
    for key in ("refined_polygon", "polygon", "raw_yolo_poly", "mask_polygon", "obb"):
        poly = p.get(key)
        if isinstance(poly, (list, tuple)) and len(poly) >= 3:
            try:
                arr = np.array(poly, dtype=np.float32)
                if arr.ndim == 2 and arr.shape[1] >= 2:
                    if len(arr) == 4:
                        return sort_panel_corners(arr[:, :2])
                    rect = cv2.minAreaRect(arr[:, :2])
                    return sort_panel_corners(cv2.boxPoints(rect))
            except Exception:
                pass
    return None


def load_panel_priors(panel_log: Path) -> List[PanelPrior]:
    records = read_json_or_jsonl(panel_log)
    panels: List[PanelPrior] = []
    for idx, rec in enumerate(records):
        b = bbox_from_record(rec)
        if b is None:
            continue
        panels.append(PanelPrior(idx=idx, bbox=b, polygon=polygon_from_record(rec), raw=rec))
    if not panels:
        raise ValueError(f"No valid panel bbox loaded from {panel_log}")
    return panels


def find_panel_log(logs_dir: Path, stem: str) -> Optional[Path]:
    candidates = [
        logs_dir / f"{stem}_panel_refine.jsonl",
        logs_dir / f"{stem}_panel_refine.json",
        logs_dir / f"raw_{stem}_panel_refine.jsonl",
        logs_dir / f"raw_{stem}_panel_refine.json",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def iter_images(dataset: Path, skip_raw_duplicates: bool = True) -> Iterable[Path]:
    if dataset.is_file() and dataset.suffix.lower() in IMAGE_EXTS:
        yield dataset
        return
    for p in sorted(dataset.iterdir()):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
            continue
        if skip_raw_duplicates and p.stem.lower().startswith("raw_"):
            continue
        yield p


# ---------------------------------------------------------------------------
# Common geometry helpers
# ---------------------------------------------------------------------------

def sort_panel_corners(poly: Sequence[Sequence[float]]) -> np.ndarray:
    pts = np.array(poly, dtype=np.float32)[:, :2]
    if pts.shape[0] != 4:
        rect = cv2.minAreaRect(pts)
        pts = cv2.boxPoints(rect).astype(np.float32)
    center = np.mean(pts, axis=0)
    ang = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    pts = pts[np.argsort(ang)]
    # After angle sort: split top/bottom by y then left/right by x.
    top_idx = np.argsort(pts[:, 1])[:2]
    bot_idx = np.argsort(pts[:, 1])[2:]
    top = pts[top_idx][np.argsort(pts[top_idx][:, 0])]
    bot = pts[bot_idx][np.argsort(pts[bot_idx][:, 0])]
    return np.array([top[0], top[1], bot[1], bot[0]], dtype=np.float32)


def clamp_slope(m: float, max_abs_angle_deg: float) -> float:
    ang = math.degrees(math.atan(float(m)))
    lim = float(max_abs_angle_deg)
    if abs(ang) <= lim:
        return float(m)
    return math.tan(math.radians(lim if ang > 0 else -lim))


def robust_fit_y_from_x(points: Sequence[Tuple[float, float]], fallback_m: float = 0.0, fallback_c: float = 0.0) -> Tuple[float, float, float]:
    if len(points) < 2:
        return float(fallback_m), float(fallback_c), 999.0
    arr = np.array(points, dtype=np.float32)
    xs, ys = arr[:, 0], arr[:, 1]
    try:
        m, c = np.polyfit(xs, ys, 1)
        for _ in range(2):
            res = ys - (m * xs + c)
            med = np.median(res)
            mad = np.median(np.abs(res - med)) + 1e-6
            keep = np.abs(res - med) <= max(1.5, 2.5 * mad)
            if int(np.sum(keep)) < 2 or int(np.sum(keep)) == len(xs):
                break
            xs, ys = xs[keep], ys[keep]
            m, c = np.polyfit(xs, ys, 1)
        residual = float(np.std(ys - (m * xs + c))) if len(xs) else 999.0
        return float(m), float(c), residual
    except Exception:
        return float(fallback_m), float(fallback_c), 999.0


def robust_fit_x_from_y(points: Sequence[Tuple[float, float]], fallback_x: float = 0.0) -> Tuple[float, float, float]:
    # Fit x = a*y + b. Returns a,b,residual.
    if len(points) < 2:
        return 0.0, float(fallback_x), 999.0
    arr = np.array(points, dtype=np.float32)
    xs, ys = arr[:, 0], arr[:, 1]
    try:
        a, b = np.polyfit(ys, xs, 1)
        for _ in range(2):
            res = xs - (a * ys + b)
            med = np.median(res)
            mad = np.median(np.abs(res - med)) + 1e-6
            keep = np.abs(res - med) <= max(1.5, 2.5 * mad)
            if int(np.sum(keep)) < 2 or int(np.sum(keep)) == len(xs):
                break
            xs, ys = xs[keep], ys[keep]
            a, b = np.polyfit(ys, xs, 1)
        residual = float(np.std(xs - (a * ys + b))) if len(xs) else 999.0
        return float(a), float(b), residual
    except Exception:
        return 0.0, float(fallback_x), 999.0


def make_rail_from_xy_line(a: float, b: float, y_min: float, y_max: float, n: int = 15) -> List[Tuple[float, float]]:
    ys = np.linspace(float(y_min), float(y_max), max(2, int(n)))
    return [(float(a * y + b), float(y)) for y in ys]


def get_boundary_x_at_y(rail: RailXY | List[Tuple[float, float]], y_val: float) -> float:
    pts = rail.pts if isinstance(rail, RailXY) else rail
    if not pts:
        return 0.0
    if len(pts) == 1:
        return float(pts[0][0])
    pts2 = sorted(pts, key=lambda p: p[1])
    y = float(y_val)
    if y <= pts2[0][1]:
        return float(pts2[0][0])
    if y >= pts2[-1][1]:
        return float(pts2[-1][0])
    for p0, p1 in zip(pts2[:-1], pts2[1:]):
        x0, y0 = p0
        x1, y1 = p1
        if y0 <= y <= y1:
            if abs(y1 - y0) < 1e-6:
                return float(x0)
            t = (y - y0) / (y1 - y0)
            return float(x0 + t * (x1 - x0))
    return float(pts2[0][0])


def intersect_line_with_rail(line: LineYC, rail: RailXY) -> Tuple[float, float]:
    # Solve intersection with rail polyline. Fallback to x at y around line center.
    pts = rail.pts
    for p0, p1 in zip(pts[:-1], pts[1:]):
        x0, y0 = p0
        x1, y1 = p1
        dx = x1 - x0
        dy = y1 - y0
        denom = dy - line.m * dx
        if abs(denom) < 1e-6:
            continue
        t = (line.m * x0 + line.c - y0) / denom
        if 0.0 <= t <= 1.0:
            x = x0 + t * dx
            y = y0 + t * dy
            return float(x), float(y)
    # Fallback: evaluate at the midpoint y of rail.
    y_ref = float(np.mean([p[1] for p in pts])) if pts else 0.0
    x = get_boundary_x_at_y(rail, y_ref)
    return float(x), float(line.y_at(x))


def polygon_area(poly: np.ndarray) -> float:
    return float(abs(cv2.contourArea(np.array(poly, dtype=np.float32))))


def validate_panel_polygon(poly: np.ndarray, prior: Optional[PanelPrior], cfg: GeometryConfig) -> bool:
    poly = np.array(poly, dtype=np.float32)
    if poly.shape != (4, 2):
        return False
    if polygon_area(poly) < cfg.min_panel_area_px:
        return False
    if not cv2.isContourConvex(poly.astype(np.int32)):
        return False
    if prior is not None:
        pc = np.mean(poly, axis=0)
        dx = abs(float(pc[0]) - prior.cx)
        dy = abs(float(pc[1]) - prior.cy)
        if dx > cfg.max_center_shift_ratio * max(prior.w, 1.0) + 8:
            return False
        if dy > cfg.max_center_shift_ratio * max(prior.h, 1.0) + 8:
            return False
    return True


def inset_quad_polygon(poly: np.ndarray, margin_px: float) -> Tuple[np.ndarray, str, bool]:
    poly_f = np.array(poly, dtype=np.float32)
    if poly_f.shape != (4, 2):
        return poly_f, "invalid_shape", False
    if margin_px <= 0:
        return poly_f.copy(), "no_inset", True

    centroid = np.mean(poly_f, axis=0)
    angles = np.arctan2(poly_f[:, 1] - centroid[1], poly_f[:, 0] - centroid[0])
    order = np.argsort(angles)
    q = poly_f[order]
    lines = []
    ok = True
    for i in range(4):
        p_i = q[i]
        p_j = q[(i + 1) % 4]
        dx, dy = float(p_j[0] - p_i[0]), float(p_j[1] - p_i[1])
        length = math.hypot(dx, dy)
        if length < 1e-6:
            ok = False
            break
        nx, ny = -dy / length, dx / length
        mid = 0.5 * (p_i + p_j)
        if nx * (centroid[0] - mid[0]) + ny * (centroid[1] - mid[1]) < 0:
            nx, ny = -nx, -ny
        d = nx * p_i[0] + ny * p_i[1] + float(margin_px)
        lines.append((nx, ny, d))

    inner_q = []
    if ok:
        for i in range(4):
            nx1, ny1, d1 = lines[(i - 1) % 4]
            nx2, ny2, d2 = lines[i]
            det = nx1 * ny2 - ny1 * nx2
            if abs(det) < 1e-6:
                ok = False
                break
            x = (d1 * ny2 - ny1 * d2) / det
            y = (nx1 * d2 - d1 * nx2) / det
            if not np.isfinite(x) or not np.isfinite(y):
                ok = False
                break
            inner_q.append([x, y])

    if ok and len(inner_q) == 4:
        inner_sorted = np.array(inner_q, dtype=np.float32)
        if polygon_area(inner_sorted) > 1 and cv2.isContourConvex(inner_sorted.astype(np.int32)):
            inner = np.zeros_like(poly_f)
            for orig_idx in range(4):
                pos = int(np.where(order == orig_idx)[0][0])
                inner[orig_idx] = inner_sorted[pos]
            return inner, "geometric_inset", True

    # Fallback: centroid scale. This keeps an inner shape even when lines are nearly parallel.
    dists = np.linalg.norm(poly_f - centroid, axis=1)
    scale = max(0.0, 1.0 - float(margin_px) / max(float(np.mean(dists)), 1.0))
    inner = centroid + scale * (poly_f - centroid)
    valid = polygon_area(inner) > 1 and cv2.isContourConvex(inner.astype(np.int32))
    return inner.astype(np.float32), "centroid_fallback", bool(valid)


# ---------------------------------------------------------------------------
# Routing / grouping
# ---------------------------------------------------------------------------

def compute_route_metrics(image_shape: Tuple[int, int, int], panels: List[PanelPrior], top_ratio: float) -> Dict[str, Any]:
    h, w = image_shape[:2]
    image_area = float(max(1, w * h))
    positive = [p for p in panels if p.area > 1]
    if not positive:
        raise ValueError("No positive-area panels for routing")
    sorted_panels = sorted(positive, key=lambda p: p.area, reverse=True)
    k = max(1, int(math.ceil(len(sorted_panels) * float(top_ratio))))
    top = sorted_panels[:k]
    rep_area = float(np.median([p.area for p in top]))
    rep_w = float(np.median([p.w for p in top]))
    rep_h = float(np.median([p.h for p in top]))
    return {
        "image_w": w,
        "image_h": h,
        "image_area": image_area,
        "panel_count": len(positive),
        "top_k": k,
        "rep_w": rep_w,
        "rep_h": rep_h,
        "rep_area": rep_area,
        "rep_area_frac": rep_area / image_area,
    }


def cluster_1d(items: List[PanelPrior], attr: str, threshold: float) -> List[List[PanelPrior]]:
    if not items:
        return []
    key = (lambda p: p.cx) if attr == "x" else (lambda p: p.cy)
    ordered = sorted(items, key=key)
    clusters: List[List[PanelPrior]] = [[ordered[0]]]
    for p in ordered[1:]:
        curr_center = float(np.mean([key(q) for q in clusters[-1]]))
        if abs(key(p) - curr_center) <= threshold:
            clusters[-1].append(p)
        else:
            clusters.append([p])
    return clusters


def split_columns_into_blocks(columns: List[List[PanelPrior]], median_w: float, cfg: GeometryConfig) -> List[List[List[PanelPrior]]]:
    if not columns:
        return []
    col_centers = [float(np.mean([p.cx for p in col])) for col in columns]
    blocks: List[List[List[PanelPrior]]] = [[columns[0]]]
    threshold = max(10.0, cfg.block_gap_factor * median_w)
    for i in range(1, len(columns)):
        gap = col_centers[i] - col_centers[i - 1]
        if gap > threshold:
            blocks.append([columns[i]])
        else:
            blocks[-1].append(columns[i])
    return blocks


def group_into_blocks(panels: List[PanelPrior], cfg: GeometryConfig) -> List[List[List[PanelPrior]]]:
    median_w = float(np.median([p.w for p in panels])) if panels else 40.0
    median_h = float(np.median([p.h for p in panels])) if panels else 30.0
    # First cluster x centers into physical columns.
    col_thresh = max(10.0, cfg.x_column_gap_factor * median_w * 0.55)
    columns = cluster_1d(panels, "x", col_thresh)
    for col in columns:
        col.sort(key=lambda p: p.cy)
    columns.sort(key=lambda col: float(np.mean([p.cx for p in col])))
    # Then group adjacent columns into blocks/strings.
    return split_columns_into_blocks(columns, median_w, cfg)


def rows_for_block(block_cols: List[List[PanelPrior]], cfg: GeometryConfig) -> List[List[PanelPrior]]:
    panels = [p for col in block_cols for p in col]
    median_h = float(np.median([p.h for p in panels])) if panels else 30.0
    row_thresh = max(8.0, cfg.y_row_merge_factor * median_h)
    rows = cluster_1d(panels, "y", row_thresh)
    for row in rows:
        row.sort(key=lambda p: p.cx)
    rows.sort(key=lambda row: float(np.mean([p.cy for p in row])))
    return rows


# ---------------------------------------------------------------------------
# Thermal snapping core
# ---------------------------------------------------------------------------

def sample_gray_bilinear(gray: np.ndarray, x: float, y: float) -> float:
    h, w = gray.shape[:2]
    if x < 0 or y < 0 or x >= w - 1 or y >= h - 1:
        xi = int(max(0, min(w - 1, round(x))))
        yi = int(max(0, min(h - 1, round(y))))
        return float(gray[yi, xi])
    x0 = int(math.floor(x)); y0 = int(math.floor(y))
    dx = float(x - x0); dy = float(y - y0)
    v00 = float(gray[y0, x0]); v10 = float(gray[y0, x0 + 1])
    v01 = float(gray[y0 + 1, x0]); v11 = float(gray[y0 + 1, x0 + 1])
    return (1-dx)*(1-dy)*v00 + dx*(1-dy)*v10 + (1-dx)*dy*v01 + dx*dy*v11


def prior_boundary_xs(block_cols: List[List[PanelPrior]]) -> List[float]:
    cols = block_cols
    xs: List[float] = []
    if not cols:
        return xs
    # Left boundary from left edges of first column.
    xs.append(float(np.median([p.bbox[0] for p in cols[0]])))
    # Internal boundaries from adjacent col gap midpoint.
    for c0, c1 in zip(cols[:-1], cols[1:]):
        right0 = float(np.median([p.bbox[2] for p in c0]))
        left1 = float(np.median([p.bbox[0] for p in c1]))
        xs.append(0.5 * (right0 + left1))
    # Right boundary from right edges of last column.
    xs.append(float(np.median([p.bbox[2] for p in cols[-1]])))
    return xs


def prior_boundary_ys(rows: List[List[PanelPrior]]) -> List[float]:
    if not rows:
        return []
    ys = [float(np.median([p.bbox[1] for p in rows[0]]))]
    for r0, r1 in zip(rows[:-1], rows[1:]):
        bot0 = float(np.median([p.bbox[3] for p in r0]))
        top1 = float(np.median([p.bbox[1] for p in r1]))
        ys.append(0.5 * (bot0 + top1))
    ys.append(float(np.median([p.bbox[3] for p in rows[-1]])))
    return ys


def fit_rail_vertical_valley(gray: np.ndarray, x_prior: float, y_min: float, y_max: float, cfg: GeometryConfig, route: str) -> RailXY:
    h, w = gray.shape[:2]
    step = cfg.small_rail_y_step_px if route == "small_core" else cfg.large_rail_y_step_px
    radius = cfg.small_rail_search_radius_px if route == "small_core" else cfg.large_rail_search_radius_px
    y0 = max(0.0, float(y_min)); y1 = min(float(h - 1), float(y_max))
    if y1 <= y0 + 2:
        pts = [(x_prior, y0), (x_prior, y1)]
        return RailXY(pts=pts, source="prior_vertical_degenerate")
    ys = np.linspace(y0, y1, max(12, int((y1 - y0) / max(step, 0.5))))
    pts: List[Tuple[float, float]] = []
    for yy in ys:
        best = None
        for dx in np.arange(-radius, radius + 1e-6, 0.5):
            xx = float(x_prior + dx)
            if xx < 2 or xx >= w - 2:
                continue
            center = sample_gray_bilinear(gray, xx, yy)
            l2 = sample_gray_bilinear(gray, xx - 2.0, yy)
            r2 = sample_gray_bilinear(gray, xx + 2.0, yy)
            l4 = sample_gray_bilinear(gray, xx - 4.0, yy)
            r4 = sample_gray_bilinear(gray, xx + 4.0, yy)
            side = float(np.median([l2, r2, l4, r4]))
            contrast = max(0.0, side - center)
            score = center - 0.75 * contrast + 0.12 * abs(dx)
            if best is None or score < best[0]:
                best = (score, xx, center, contrast)
        if best is None:
            continue
        _, xx, center, contrast = best
        if center < 170.0 or contrast >= 3.0:
            pts.append((float(xx), float(yy)))
    min_support = max(8, int(0.20 * len(ys)))
    if len(pts) >= min_support:
        a, b, residual = robust_fit_x_from_y(pts, fallback_x=x_prior)
        # Clamp center shift to avoid background pulling outer rails.
        y_mid = 0.5 * (y0 + y1)
        shift = (a * y_mid + b) - x_prior
        max_shift = 4.0 if route == "small_core" else 6.0
        if abs(shift) > max_shift:
            b += math.copysign(max_shift - abs(shift), shift)
        rail_pts = make_rail_from_xy_line(a, b, y0, y1, n=15)
        return RailXY(pts=rail_pts, source="dark_vertical_valley", support=len(pts), residual=residual)
    return RailXY(pts=[(float(x_prior), y) for y in np.linspace(y0, y1, 15)], source="prior_vertical", support=len(pts), residual=999.0)


def fit_row_dark_valley(gray: np.ndarray, y_prior: float, x_min: float, x_max: float, cfg: GeometryConfig, route: str) -> LineYC:
    h, w = gray.shape[:2]
    x0 = max(0.0, float(x_min)); x1 = min(float(w - 1), float(x_max))
    if x1 <= x0 + 4:
        return LineYC(0.0, float(y_prior), "prior_horizontal_degenerate")
    x_step = cfg.small_row_x_step_px if route == "small_core" else cfg.large_row_x_step_px
    dy_step = cfg.small_row_dy_step_px if route == "small_core" else cfg.large_row_dy_step_px
    radius = cfg.small_row_search_radius_px if route == "small_core" else cfg.large_row_search_radius_px
    max_angle = cfg.small_max_row_angle_deg if route == "small_core" else cfg.large_max_row_angle_deg
    xs = np.linspace(x0, x1, max(30, int((x1 - x0) / max(x_step, 0.5))))
    pts: List[Tuple[float, float]] = []
    for xx in xs:
        best = None
        for dy in np.arange(-radius, radius + 1e-6, dy_step):
            yy = float(y_prior + dy)
            if yy < 3 or yy >= h - 3:
                continue
            center = sample_gray_bilinear(gray, xx, yy)
            u2 = sample_gray_bilinear(gray, xx, yy - 2.0)
            d2 = sample_gray_bilinear(gray, xx, yy + 2.0)
            u4 = sample_gray_bilinear(gray, xx, yy - 4.0)
            d4 = sample_gray_bilinear(gray, xx, yy + 4.0)
            side = float(np.median([u2, d2, u4, d4]))
            contrast = max(0.0, side - center)
            score = center - 0.80 * contrast + 0.20 * abs(float(dy))
            if best is None or score < best[0]:
                best = (score, yy, center, contrast)
        if best is None:
            continue
        _, yy, center, contrast = best
        if center < 170.0 or contrast >= 3.0:
            pts.append((float(xx), float(yy)))
    if len(pts) >= max(8, int(0.20 * len(xs))):
        m, c, residual = robust_fit_y_from_x(pts, fallback_m=0.0, fallback_c=y_prior)
        m_clamped = clamp_slope(m, max_angle)
        if abs(m_clamped - m) > 1e-9:
            # Preserve y at row center.
            x_mid = 0.5 * (x0 + x1)
            y_mid = m * x_mid + c
            c = y_mid - m_clamped * x_mid
            m = m_clamped
        # Clamp center shift from prior.
        x_mid = 0.5 * (x0 + x1)
        shift = (m * x_mid + c) - y_prior
        max_shift = 4.0 if route == "small_core" else 5.0
        if abs(shift) > max_shift:
            c += math.copysign(max_shift - abs(shift), shift)
        return LineYC(float(m), float(c), "dark_horizontal_valley", support=len(pts), residual=residual)
    return LineYC(0.0, float(y_prior), "prior_horizontal", support=len(pts), residual=999.0)


# ---------------------------------------------------------------------------
# Unified engine routes
# ---------------------------------------------------------------------------

def build_block_geometry(gray: np.ndarray, block_cols: List[List[PanelPrior]], route: str, block_id: int, cfg: GeometryConfig) -> BlockGeometry:
    rows = rows_for_block(block_cols, cfg)
    panels_flat = [p for col in block_cols for p in col]
    if not panels_flat or not rows:
        return BlockGeometry(block_id, route, block_cols, rows, [], [], [])

    x_priors = prior_boundary_xs(block_cols)
    y_priors = prior_boundary_ys(rows)

    # Extents come from priors plus a small guard. They are search limits, not final geometry.
    x_min = min(p.bbox[0] for p in panels_flat)
    x_max = max(p.bbox[2] for p in panels_flat)
    y_min = min(p.bbox[1] for p in panels_flat)
    y_max = max(p.bbox[3] for p in panels_flat)
    med_w = float(np.median([p.w for p in panels_flat]))
    med_h = float(np.median([p.h for p in panels_flat]))
    x_margin = min(8.0, 0.18 * med_w)
    y_margin = min(8.0, 0.18 * med_h)

    rails = [fit_rail_vertical_valley(gray, xp, y_min - y_margin, y_max + y_margin, cfg, route) for xp in x_priors]

    # Row span uses the fitted outer rails at prior y. We fit rows across the block width.
    row_lines: List[LineYC] = []
    for yp in y_priors:
        x_left = get_boundary_x_at_y(rails[0], yp) + 3.0
        x_right = get_boundary_x_at_y(rails[-1], yp) - 3.0
        row_lines.append(fit_row_dark_valley(gray, yp, x_left, x_right, cfg, route))

    # Monotonic row guard: prevent collapse. Preserve slopes, adjust intercept by pitch.
    if len(row_lines) >= 2:
        centers = []
        x_mid = 0.5 * (x_min + x_max)
        for ln in row_lines:
            centers.append(ln.y_at(x_mid))
        diffs = [centers[i] - centers[i-1] for i in range(1, len(centers)) if centers[i] > centers[i-1]]
        pitch = float(np.median(diffs)) if diffs else max(8.0, med_h)
        min_pitch = max(4.0, 0.45 * pitch)
        for i in range(1, len(row_lines)):
            prev_y = row_lines[i-1].y_at(x_mid)
            curr_y = row_lines[i].y_at(x_mid)
            if curr_y <= prev_y + min_pitch:
                target_y = prev_y + pitch
                row_lines[i].c += target_y - curr_y
                row_lines[i].source += "+pitch_guard"

    panels: List[PanelGeometry] = []
    # Create lookup by nearest prior center for each row/col cell.
    for r_idx in range(max(0, len(row_lines) - 1)):
        for c_idx in range(max(0, len(rails) - 1)):
            # Find nearest prior panel to this cell, if any.
            prior = nearest_panel_for_cell(block_cols, rows, r_idx, c_idx)
            left_rail = rails[c_idx]
            right_rail = rails[c_idx + 1]
            top_line = row_lines[r_idx]
            bot_line = row_lines[r_idx + 1]
            tl = intersect_line_with_rail(top_line, left_rail)
            tr = intersect_line_with_rail(top_line, right_rail)
            br = intersect_line_with_rail(bot_line, right_rail)
            bl = intersect_line_with_rail(bot_line, left_rail)
            outer = np.array([tl, tr, br, bl], dtype=np.float32)
            if not validate_panel_polygon(outer, prior, cfg):
                # Conservative fallback still comes from grid priors, not direct YOLO polygon.
                # Use the corresponding prior bbox only to re-anchor a rectangular cell if the
                # fitted intersections degenerate.
                if prior is None:
                    continue
                x1, y1, x2, y2 = prior.bbox
                outer = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
                source = f"{route}_prior_cell_fallback"
            else:
                source = f"{route}_rail_row_intersection"
            inner, _, valid_inner = inset_quad_polygon(outer, cfg.inner_panel_margin_px)
            if not valid_inner:
                inner = outer.copy()
            panels.append(PanelGeometry(
                idx=len(panels), row=r_idx, col=c_idx, route=route,
                outer_polygon=outer, inner_polygon=inner, source=source, valid=True
            ))

    return BlockGeometry(block_id, route, block_cols, rows, rails, row_lines, panels)


def nearest_panel_for_cell(block_cols: List[List[PanelPrior]], rows: List[List[PanelPrior]], r_idx: int, c_idx: int) -> Optional[PanelPrior]:
    if c_idx >= len(block_cols) or r_idx >= len(rows):
        return None
    col_set = set(p.idx for p in block_cols[c_idx])
    row_set = set(p.idx for p in rows[r_idx])
    both = [p for p in block_cols[c_idx] if p.idx in row_set]
    if both:
        return both[0]
    # Fallback by nearest y in column.
    col = block_cols[c_idx]
    if not col:
        return None
    if r_idx < len(col):
        return col[r_idx]
    return min(col, key=lambda p: abs(p.cy - np.mean([q.cy for q in rows[r_idx]]))) if rows and r_idx < len(rows) else None


def run_unified_geometry(image_bgr: np.ndarray, panels: List[PanelPrior], route: str, cfg: GeometryConfig) -> List[BlockGeometry]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blocks_cols = group_into_blocks(panels, cfg)
    results: List[BlockGeometry] = []
    for i, block_cols in enumerate(blocks_cols, start=1):
        try:
            bg = build_block_geometry(gray, block_cols, route, i, cfg)
            if bg.panels:
                results.append(bg)
        except Exception as e:
            print(f"[BLOCK_ERROR] block={i} route={route} type={type(e).__name__} message={e}")
    return results


# ---------------------------------------------------------------------------
# Drawing / export
# ---------------------------------------------------------------------------

def draw_polyline(img: np.ndarray, pts: Sequence[Tuple[float, float]], color: Tuple[int, int, int], thickness: int = 1) -> None:
    if len(pts) < 2:
        return
    for p0, p1 in zip(pts[:-1], pts[1:]):
        cv2.line(img, (int(round(p0[0])), int(round(p0[1]))), (int(round(p1[0])), int(round(p1[1]))), color, thickness, cv2.LINE_AA)


def draw_line_snap(image_bgr: np.ndarray, blocks: List[BlockGeometry], cfg: GeometryConfig) -> np.ndarray:
    out = image_bgr.copy()
    for block in blocks:
        for rail in block.rails:
            draw_polyline(out, rail.pts, COLOR_RAIL, max(1, cfg.visual_thickness + 1))
        for row in block.row_lines:
            if not block.rails:
                continue
            x0 = get_boundary_x_at_y(block.rails[0], row.y_at(0))
            x1 = get_boundary_x_at_y(block.rails[-1], row.y_at(image_bgr.shape[1] - 1))
            # Better endpoints: intersect with outer rails.
            p0 = intersect_line_with_rail(row, block.rails[0])
            p1 = intersect_line_with_rail(row, block.rails[-1])
            cv2.line(out, (int(round(p0[0])), int(round(p0[1]))), (int(round(p1[0])), int(round(p1[1]))), COLOR_ROW, cfg.visual_thickness, cv2.LINE_AA)
    return out


def draw_calc_inner(image_bgr: np.ndarray, blocks: List[BlockGeometry], cfg: GeometryConfig) -> np.ndarray:
    out = image_bgr.copy()
    for block in blocks:
        for pg in block.panels:
            outer = np.array(pg.outer_polygon, dtype=np.int32)
            inner = np.array(pg.inner_polygon, dtype=np.int32)
            cv2.polylines(out, [outer], True, COLOR_OUTER, cfg.visual_thickness, cv2.LINE_AA)
            cv2.polylines(out, [inner], True, COLOR_INNER, max(1, cfg.visual_thickness + 1), cv2.LINE_AA)
    return out


def save_geometry_json(out_dir: Path, stem: str, route: str, metrics: Dict[str, Any], blocks: List[BlockGeometry]) -> None:
    rows: List[Dict[str, Any]] = []
    for b in blocks:
        for p in b.panels:
            rows.append({
                "panel_index": p.idx,
                "block_id": b.block_id,
                "row": p.row,
                "col": p.col,
                "route": p.route,
                "source": p.source,
                "valid": p.valid,
                "outer_polygon": np.asarray(p.outer_polygon, dtype=float).round(3).tolist(),
                "inner_polygon": np.asarray(p.inner_polygon, dtype=float).round(3).tolist(),
                "outer_area_px": round(polygon_area(p.outer_polygon), 3),
                "inner_area_px": round(polygon_area(p.inner_polygon), 3),
            })
    data = {
        "stem": stem,
        "route": route,
        "metrics": metrics,
        "panels": rows,
    }
    (out_dir / f"{stem}_geometry_v57.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def output_paths(out_dir: Path, stem: str) -> Tuple[Path, Path]:
    return out_dir / f"debug_{stem}_line_snap.JPG", out_dir / f"debug_{stem}_calc_inner_polygon.JPG"


# ---------------------------------------------------------------------------
# Runner / CLI
# ---------------------------------------------------------------------------

def process_one_image(image_path: Path, logs_dir: Path, out_dir: Path, cfg: GeometryConfig, geometry_mode: str = "auto", export_json: bool = True) -> bool:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        print(f"[ERROR] stem={image_path.stem} reason=cannot_read_image path={image_path}")
        return False
    panel_log = find_panel_log(logs_dir, image_path.stem)
    if panel_log is None:
        print(f"[SKIP] stem={image_path.stem} reason=missing_panel_log")
        return False
    try:
        panels = load_panel_priors(panel_log)
        metrics = compute_route_metrics(image.shape, panels, cfg.representative_top_area_ratio)
    except Exception as e:
        print(f"[ERROR] stem={image_path.stem} reason=load_or_route_failed type={type(e).__name__} message={e}")
        return False

    if geometry_mode == "small_core":
        route = "small_core"
        reason = "forced_small_core"
    elif geometry_mode == "large_local":
        route = "large_local"
        reason = "forced_large_local"
    else:
        route = "large_local" if metrics["rep_area_frac"] > cfg.route_area_frac_threshold else "small_core"
        reason = f"rep_area_frac>{cfg.route_area_frac_threshold}" if route == "large_local" else f"rep_area_frac<={cfg.route_area_frac_threshold}"

    print(
        f"[V57_ROUTE_METRICS] stem={image_path.stem} decision={route} reason={reason} "
        f"n={metrics['panel_count']} top_k={metrics['top_k']} "
        f"rep_w={metrics['rep_w']:.1f} rep_h={metrics['rep_h']:.1f} "
        f"rep_area={metrics['rep_area']:.1f} rep_area_frac={metrics['rep_area_frac']:.5f} "
        f"image={metrics['image_w']}x{metrics['image_h']} log={panel_log.name}"
    )

    blocks = run_unified_geometry(image, panels, route, cfg)
    if not blocks or not any(b.panels for b in blocks):
        print(f"[ERROR] stem={image_path.stem} route={route} reason=no_panel_geometry_built")
        return False

    out_dir.mkdir(parents=True, exist_ok=True)
    line_path, calc_path = output_paths(out_dir, image_path.stem)
    line_img = draw_line_snap(image, blocks, cfg)
    calc_img = draw_calc_inner(image, blocks, cfg)
    cv2.imwrite(str(line_path), line_img)
    cv2.imwrite(str(calc_path), calc_img)
    if export_json:
        save_geometry_json(out_dir, image_path.stem, route, metrics, blocks)

    n_panels = sum(len(b.panels) for b in blocks)
    print(f"[DONE] stem={image_path.stem} route={route} panels={n_panels} outputs={line_path.name},{calc_path.name}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="V57 clean unified PV geometry engine: no embedded engines, no subprocess.")
    ap.add_argument("--dataset", required=True, help="Image folder or single image path")
    ap.add_argument("--logs", required=True, help="Folder containing *_panel_refine.json/jsonl")
    ap.add_argument("--out", required=True, help="Output debug folder")
    ap.add_argument("--geometry-mode", choices=["auto", "small_core", "large_local"], default="auto", help="Force route or auto")
    ap.add_argument("--route-threshold", type=float, default=0.025, help="rep_area_frac threshold. Default: 0.025")
    ap.add_argument("--top-frac", type=float, default=0.40, help="Top area fraction for representative panel. Default: 0.40")
    ap.add_argument("--inner-margin", type=float, default=2.0, help="Geometric inner polygon margin in pixels. Float allowed.")
    ap.add_argument("--visual-thickness", type=int, default=1, help="Debug polygon/line thickness")
    ap.add_argument("--no-skip-raw", action="store_true", help="Do not skip raw_* images")
    ap.add_argument("--no-json", action="store_true", help="Do not export <stem>_geometry_v57.json")
    args = ap.parse_args()

    cfg = GeometryConfig(
        route_area_frac_threshold=float(args.route_threshold),
        representative_top_area_ratio=float(args.top_frac),
        inner_panel_margin_px=float(args.inner_margin),
        visual_thickness=max(1, int(args.visual_thickness)),
    )

    dataset = Path(args.dataset)
    logs_dir = Path(args.logs)
    out_dir = Path(args.out)
    total = ok = skipped_or_failed = 0
    for img_path in iter_images(dataset, skip_raw_duplicates=not args.no_skip_raw):
        total += 1
        if process_one_image(img_path, logs_dir, out_dir, cfg, geometry_mode=args.geometry_mode, export_json=not args.no_json):
            ok += 1
        else:
            skipped_or_failed += 1
    print(f"[BATCH_SUMMARY] ok={ok} failed_or_skipped={skipped_or_failed} total={total} out={out_dir}")
    return 0 if ok > 0 and skipped_or_failed == 0 else (0 if ok > 0 else 1)


if __name__ == "__main__":
    raise SystemExit(main())
