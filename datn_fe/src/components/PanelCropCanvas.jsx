import React, { useState, useEffect, useRef } from 'react';

const BASE_URL = 'http://127.0.0.1:8000';

function getDisplayPolygon(d) {
    if (!d) return null;
    if (Array.isArray(d.display_polygon) && d.display_polygon.length >= 3) return d.display_polygon;
    if (Array.isArray(d.polygon) && d.polygon.length >= 3) return d.polygon;
    if (Array.isArray(d.analysis_polygon) && d.analysis_polygon.length >= 3) return d.analysis_polygon;
    if (d.bbox && d.bbox.length === 4) {
        const [x1, y1, x2, y2] = d.bbox;
        return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]];
    }
    return null;
}

function getDefectColor(className) {
    const name = String(className || "").toLowerCase();
    if (name.includes("hotspot_single") || name.includes("single_cell") || name.includes("single-cell") || name.includes("hotspot") || name.includes("hot")) {
        return "#ff3b30";
    }
    if (name.includes("hotspot_multi") || name.includes("multi_cell") || name.includes("multicell") || name.includes("multi-cell")) {
        return "#ff2d55";
    }
    if (name.includes("crack") || name.includes("nut")) {
        return "#f59e0b";
    }
    if (name.includes("shading") || name.includes("shadow") || name.includes("shade") || name.includes("soil") || name.includes("soiling") || name.includes("dirt")) {
        return "#8b5cf6";
    }
    return "#ef4444";
}

export function PanelCropCanvas({ item, imgWidth = 640, imgHeight = 512, canvasWidth = 480 }) {
    const canvasRef = useRef(null);
    const [imgLoaded, setImgLoaded] = useState(false);
    const PADDING = 36;

    // Compute crop box from panel polygon or bbox
    const panelPoly = item?.panel?.outer_polygon || item?.panel?.polygon || item?.polygon || [];
    const panelBbox = item?.panel?.bbox || item?.panel?.box || item?.bbox || [];

    let px1 = Infinity, py1 = Infinity, px2 = -Infinity, py2 = -Infinity;
    if (panelPoly.length >= 3) {
        panelPoly.forEach(([x, y]) => {
            px1 = Math.min(px1, x); py1 = Math.min(py1, y);
            px2 = Math.max(px2, x); py2 = Math.max(py2, y);
        });
    } else if (panelBbox.length === 4) {
        [px1, py1, px2, py2] = panelBbox;
    }

    if (!isFinite(px1)) { px1 = 0; py1 = 0; px2 = imgWidth; py2 = imgHeight; }

    const cropX1 = Math.max(0, px1 - PADDING);
    const cropY1 = Math.max(0, py1 - PADDING);
    const cropX2 = Math.min(imgWidth, px2 + PADDING);
    const cropY2 = Math.min(imgHeight, py2 + PADDING);
    const cropW = cropX2 - cropX1;
    const cropH = cropY2 - cropY1;

    const CANVAS_W = canvasWidth;
    const CANVAS_H = Math.round((cropH / cropW) * CANVAS_W) || 300;
    const scale = CANVAS_W / cropW;

    const imgUrl = item?.image_url
        ? `${BASE_URL}${item.image_url}`
        : item?.annotated_image_url ? `${BASE_URL}${item.annotated_image_url}` : null;

    const isRgb = imgUrl && (imgUrl.includes('/data/raw/') || imgUrl.includes('/raw/'));

    useEffect(() => {
        setImgLoaded(false);
    }, [imgUrl]);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas || !imgLoaded) return;
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, CANVAS_W, CANVAS_H);

        // Draw cropped image
        const img = new Image();
        img.onload = () => {
            let rCropX1 = cropX1;
            let rCropY1 = cropY1;
            let rCropW = cropW;
            let rCropH = cropH;

            const w_r = img.naturalWidth;
            const h_r = img.naturalHeight;

            if (isRgb && w_r && h_r) {
                // Calculate zoomed RGB cropping box (centered at panel, zoomed by 1.5x with 20% slack, so zoomed by 1.25x)
                const cx_raw = ((cropX1 + cropX2) / 2) * (w_r / imgWidth);
                const cy_raw = ((cropY1 + cropY2) / 2) * (h_r / imgHeight);
                const w_raw = (cropX2 - cropX1) * (w_r / imgWidth);
                const h_raw = (cropY2 - cropY1) * (h_r / imgHeight);

                const w_raw_zoomed = w_raw / 1.25;
                const h_raw_zoomed = h_raw / 1.25;

                rCropX1 = Math.max(0, cx_raw - w_raw_zoomed / 2);
                rCropY1 = Math.max(0, cy_raw - h_raw_zoomed / 2);
                rCropW = Math.min(w_r - rCropX1, w_raw_zoomed);
                rCropH = Math.min(h_r - rCropY1, h_raw_zoomed);
            }

            ctx.drawImage(img, rCropX1, rCropY1, rCropW, rCropH, 0, 0, CANVAS_W, CANVAS_H);

            const toCanvas = (x, y) => {
                if (isRgb && w_r && h_r) {
                    const x_raw = x * (w_r / imgWidth);
                    const y_raw = y * (h_r / imgHeight);
                    return [
                        (x_raw - rCropX1) * (CANVAS_W / rCropW),
                        (y_raw - rCropY1) * (CANVAS_H / rCropH)
                    ];
                } else {
                    return [(x - cropX1) * scale, (y - cropY1) * scale];
                }
            };

            // Draw panel outline (only for thermal images, not RGB)
            if (!isRgb && panelPoly.length >= 3) {
                ctx.beginPath();
                const [fx, fy] = toCanvas(...panelPoly[0]);
                ctx.moveTo(fx, fy);
                panelPoly.forEach(([x, y]) => { const [cx, cy] = toCanvas(x, y); ctx.lineTo(cx, cy); });
                ctx.closePath();
                ctx.strokeStyle = 'rgba(56,189,248,0.85)';
                ctx.lineWidth = 2;
                ctx.stroke();
            }

            // Draw defect polygons
            (item?.defects || []).forEach((d) => {
                const poly = getDisplayPolygon(d);
                if (!poly || poly.length < 3) return;

                const color = getDefectColor(d.class_name || d.type);
                ctx.beginPath();
                const [fx, fy] = toCanvas(...poly[0]);
                ctx.moveTo(fx, fy);
                poly.forEach(([x, y]) => { const [cx, cy] = toCanvas(x, y); ctx.lineTo(cx, cy); });
                ctx.closePath();
                ctx.strokeStyle = color;
                ctx.lineWidth = 2.5;
                ctx.stroke();
            });

            // Panel ID label
            const localId = item?.local_id || item?.panel?.local_id || '';
            if (localId) {
                ctx.beginPath();
                ctx.fillStyle = 'rgba(15,23,42,0.82)';
                ctx.roundRect?.(6, 6, 100, 22, 6) || ctx.fillRect(6, 6, 100, 22);
                ctx.fill();
                ctx.fillStyle = '#38bdf8';
                ctx.font = 'bold 12px sans-serif';
                ctx.fillText(`Panel ${localId}`, 12, 21);
            }
        };
        img.src = imgUrl;
    // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [imgLoaded, imgUrl, item, cropX1, cropY1, cropW, cropH, CANVAS_W, CANVAS_H, imgWidth, imgHeight, isRgb]);

    if (!imgUrl) {
        return (
            <div style={{ width: "100%", height: "100%", display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#64748b', fontSize: 13, backgroundColor: '#090d16' }}>
                Không có ảnh nguồn
            </div>
        );
    }

    return (
        <div style={{ position: 'relative', display: 'flex', width: '100%', height: '100%', alignItems: 'center', justifyContent: 'center' }}>
            {/* Hidden img to detect load */}
            <img
                src={imgUrl}
                alt=""
                style={{ display: 'none' }}
                onLoad={() => setImgLoaded(true)}
                onError={() => setImgLoaded(true)}
            />
            <canvas
                ref={canvasRef}
                width={CANVAS_W}
                height={CANVAS_H}
                style={{ display: 'block', maxWidth: '100%', maxHeight: '100%', objectFit: 'contain' }}
            />
        </div>
    );
}
