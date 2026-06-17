import React, { useState, useMemo, useEffect } from 'react';
import {
    MapContainer,
    ImageOverlay,
    Polygon,
    Tooltip,
    useMap
} from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import {
    Battery, Wifi, Compass, Navigation, Activity, Search, Thermometer, Map as MapIcon, X, Maximize2, ShieldAlert, ChevronLeft, ChevronRight
} from 'lucide-react';
import DefectReviewModal from '../../components/DefectReviewModal';
import { normalizePanel } from '../../utils/inspectionData';

import './UnifiedStyles.css';

const STATUS_COLORS = {
    "healthy": "#22c55e",         // Green
    "faulty": "#ef4444",         // Red
    "Healthy": "#22c55e",
    "normal": "#22c55e",
    "hotspot_single": "#ff3b30",
    "hotspot_multi": "#ff2d55",
    "crack": "#f59e0b",
    "shading": "#8b5cf6",
};

function getDefectGroup(className) {
  const name = String(className || "").toLowerCase();

  if (
    name.includes("hotspot_single") ||
    name.includes("single_cell") ||
    name.includes("single-cell")
  ) {
    return "hotspot_single";
  }

  if (
    name.includes("hotspot_multi") ||
    name.includes("multi_cell") ||
    name.includes("multicell") ||
    name.includes("multi-cell")
  ) {
    return "hotspot_multi";
  }

  if (name.includes("hotspot") || name.includes("hot")) {
    return "hotspot_single";
  }

  if (name.includes("crack") || name.includes("nut")) {
    return "crack";
  }

  if (
    name.includes("shading") ||
    name.includes("shadow") ||
    name.includes("shade") ||
    name.includes("soil") ||
    name.includes("soiling") ||
    name.includes("dirt")
  ) {
    return "shading";
  }

  return name || "unknown";
}

function getDefectColorByGroup(group) {
  switch (group) {
    case "hotspot_single":
      return "#ff3b30";
    case "hotspot_multi":
      return "#ff2d55";
    case "crack":
      return "#f59e0b";
    case "shading":
      return "#8b5cf6";
    default:
      return "#ff3b30";
  }
}

function panelHasDefectGroup(panel, group) {
  const defects = panel.defects || [];

  if (defects.some((d) => getDefectGroup(d.class_name) === group)) {
    return true;
  }

  if (getDefectGroup(panel.main_defect_class) === group) {
    return true;
  }

  if (getDefectGroup(panel.defect_type) === group) {
    return true;
  }

  return false;
}

// ✅ Đổi sang precalib/ — polygon được tính trên ảnh precalib,
// nên overlay phải dùng cùng ảnh để không bị lệch.
const IMAGE_BASE_URL = "http://127.0.0.1:8000/data/precalib/";
const RAW_IMAGE_BASE_URL = "http://127.0.0.1:8000/data/raw/";

// FitBounds component
function FitBounds({ gridData, focusTarget }) {
    const map = useMap();
    useEffect(() => {
        if (gridData && gridData.length > 0) {
            if (focusTarget) {
                const targetImg = gridData.find(img => img.filename === focusTarget);
                if (targetImg) {
                    const [[bY, bX], [tY, tX]] = targetImg.bounds;
                    map.fitBounds([[bY, bX], [tY, tX]], { padding: [50, 50], maxZoom: 0 });
                    return;
                }
            }
            let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
            gridData.forEach(img => {
                const [[bY, bX], [tY, tX]] = img.bounds;
                if (bX < minX) minX = bX;
                if (bY < minY) minY = bY;
                if (tX > maxX) maxX = tX;
                if (tY > maxY) maxY = tY;
            });
            map.fitBounds([[minY, minX], [maxY, maxX]], { padding: [50, 50] });
        }
    }, [gridData, map, focusTarget]);
    return null;
}

/**
 * Chuyển polygon pixel-space sang tọa độ leaflet CRS.Simple.
 * Panel polygon từ backend là [[x,y], ...] (pixel coords trong ảnh gốc).
 * Leaflet CRS.Simple dùng [lat, lng] = [y_map, x_map].
 *
 * xOffset, yOffset: vị trí ảnh trong grid map ảo.
 * Trục y leaflet: dương lên trên → yOffset là số âm (row * -imgH).
 */
function pixelPolyToLeafletScaled(poly, layout) {
  if (!poly || !Array.isArray(poly)) return [];

  const srcW = Number(layout.srcW || 640);
  const srcH = Number(layout.srcH || 512);
  const dstW = Number(layout.w);
  const dstH = Number(layout.h);
  const offX = Number(layout.x);
  const offY = Number(layout.y);

  return poly.map(([x, y]) => {
    const px = Number(x);
    const py = Number(y);

    const mapX = offX + (px / srcW) * dstW;

    // QUAN TRỌNG:
    // Pixel y=0 là mép trên ảnh.
    // Leaflet bounds [offY, offY + dstH] có offY là mép dưới.
    // Vì vậy phải lật Y:
    const mapY = offY + dstH - (py / srcH) * dstH;

    return [mapY, mapX];
  });
}

const getPanelDrawPolygon = (panel) => {
    if (panel?.outer_polygon && panel.outer_polygon.length >= 3) {
        return panel.outer_polygon;
    }
    if (panel?.polygon && panel.polygon.length >= 3) {
        return panel.polygon;
    }
    return null;
};

const polygonFromBBox = (bbox) => {
    if (!bbox || bbox.length < 4) return null;
    const bx1 = Number(bbox[0]) || 0;
    const by1 = Number(bbox[1]) || 0;
    const bx2 = Number(bbox[2]) || 0;
    const by2 = Number(bbox[3]) || 0;
    return [[bx1, by1], [bx2, by1], [bx2, by2], [bx1, by2]];
};

const getDefectColor = (className) => {
    return getDefectColorByGroup(getDefectGroup(className));
};

function translateLocationInPanel(location) {
  const value = String(location || "").toLowerCase();

  const map = {
    "upper-left": "Góc trên bên trái",
    "upper-center": "Phía trên giữa",
    "upper-right": "Góc trên bên phải",
    "middle-left": "Giữa bên trái",
    "middle-center": "Trung tâm tấm",
    "middle-right": "Giữa bên phải",
    "lower-left": "Góc dưới bên trái",
    "lower-center": "Phía dưới giữa",
    "lower-right": "Góc dưới bên phải",
  };

  return map[value] || location || "Không xác định";
}

function translateSeverity(value) {
  const v = String(value || "").toLowerCase();

  const map = {
    "critical": "Rất nghiêm trọng",
    "severe": "Nghiêm trọng",
    "high": "Cao",
    "medium": "Trung bình",
    "low": "Thấp",
    "normal": "Bình thường",
    "replace": "Cần thay thế",
    "monitor": "Theo dõi",
    "needs_review": "Cần xem xét",
    "healthy": "Bình thường",
  };

  return map[v] || value || "Không xác định";
}

function getImageRegionFromCartesian(x, y, width, height) {
  if (!Number.isFinite(x) || !Number.isFinite(y) || !width || !height) {
    return "Không xác định";
  }

  const horizontal =
    x < width / 3 ? "bên trái" :
    x > (2 * width) / 3 ? "bên phải" :
    "ở giữa";

  const vertical =
    y < height / 3 ? "phía dưới" :
    y > (2 * height) / 3 ? "phía trên" :
    "giữa";

  if (vertical === "giữa" && horizontal === "ở giữa") {
    return "Trung tâm ảnh";
  }

  if (vertical === "giữa") {
    return `Giữa ${horizontal}`;
  }

  if (horizontal === "ở giữa") {
    return `${vertical} giữa`;
  }

  return `${vertical} ${horizontal}`;
}

function formatConfidence(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return "Chưa có dữ liệu";
  const pct = n <= 1 ? n * 100 : n;
  return `${pct.toFixed(1)}%`;
}



export default function UnifiedDashboard({ data, panelPower = 600, focusTarget, batchId, onRefresh }) {
    const [isSidebarOpen, setIsSidebarOpen] = useState(true);
    const [isReviewOpen, setIsReviewOpen] = useState(false);
    const PANEL_RATED_POWER_W = 400; // Công suất định mức tấm pin 400W
    const translateDefect = (cls) => {
        if (!cls) return "Điểm bất thường";
        const c = cls.toLowerCase();
        if (c.includes("hotspot_single") || c.includes("single_cell") || c.includes("single-cell")) return "Hotspot (Đơn)";
        if (c.includes("hotspot_multi") || c.includes("multi_cell") || c.includes("multicell") || c.includes("multi-cell")) return "Hotspot (Đa)";
        if (c.includes("hotspot") || c.includes("hot")) return "Hotspot (Đơn)";
        if (c.includes("crack") || c.includes("nut")) return "Nứt (Crack)";
        if (
            c.includes("shading") ||
            c.includes("shadow") ||
            c.includes("shade") ||
            c.includes("soil") ||
            c.includes("soiling") ||
            c.includes("dirt")
        ) {
            return "Che bóng";
        }
        return cls;
    };
    const [viewMode, setViewMode] = useState('monitor');
    const [searchQuery, setSearchQuery] = useState('');
    const [statusFilter, setStatusFilter] = useState('all');
    const [showHeatmap, setShowHeatmap] = useState(true);
    const [hoveredPanel, setHoveredPanel] = useState(null);
    const [clickedPanel, setClickedPanel] = useState(null);
    const activePanel = clickedPanel || hoveredPanel;
    const showDrawer = !!activePanel;

    // Calculate Grid Mapping
    const gridData = useMemo(() => {
        if (!data || data.length === 0) return [];
        const IMAGES_PER_ROW = 5;
        const PADDING = 100;

        return data.map((img, index) => {
            const col = index % IMAGES_PER_ROW;
            const row = Math.floor(index / IMAGES_PER_ROW);

            const imgW = img.image_width || 640;
            const imgH = img.image_height || 512;

            const xOffset = col * (imgW + PADDING);
            const yOffset = -(row * (imgH + PADDING));

            const layout = {
                x: xOffset,
                y: yOffset,
                w: imgW,
                h: imgH,
                srcW: img.image_width || 640,
                srcH: img.image_height || 512,
            };

            const bounds = [
                [layout.y, layout.x],
                [layout.y + layout.h, layout.x + layout.w],
            ];

            const mappedPanels = img.panels.map(p => {
                const normalizedP = normalizePanel(p);
                let srcPoly = getPanelDrawPolygon(normalizedP);
                if (!srcPoly) {
                    srcPoly = polygonFromBBox(normalizedP.bbox || normalizedP.box);
                }

                const leafletPolygon = srcPoly ? pixelPolyToLeafletScaled(srcPoly, layout) : [];

                const bbox = normalizedP.bbox || normalizedP.box || [0, 0, 0, 0];
                const boxW = bbox[2] - bbox[0];
                const boxH = bbox[3] - bbox[1];

                const mappedDefects = (normalizedP.defects || [])
                  .map((d, idx) => {
                    const rawPoly = d.display_polygon || d.polygon;
                    if (!rawPoly || !Array.isArray(rawPoly) || rawPoly.length < 3) {
                      return null;
                    }

                    const cleanPoly = rawPoly
                      .map((pt) => Array.isArray(pt) ? pt : null)
                      .filter((pt) =>
                        pt &&
                        pt.length >= 2 &&
                        Number.isFinite(Number(pt[0])) &&
                        Number.isFinite(Number(pt[1]))
                      )
                      .map((pt) => [Number(pt[0]), Number(pt[1])]);

                    if (cleanPoly.length < 3) return null;

                    return {
                      ...d,
                      defect_index: idx,
                      raw_polygon_px: cleanPoly,
                      polygon: pixelPolyToLeafletScaled(cleanPoly, layout),
                    };
                  })
                  .filter(Boolean);

                return {
                    ...normalizedP,
                    source_rgb: img.rgb_image,
                    source_thermal: img.filename,
                    imgW, imgH,
                    boxW,
                    boxH,
                    status: normalizedP.status,
                    polygon: leafletPolygon,
                    bbox_px: bbox,
                    mappedDefects,
                };
            });

            return { ...img, bounds, mappedPanels };
        });
    }, [data]);

    const allPanels = useMemo(() => gridData.flatMap(img => img.mappedPanels), [gridData]);

    const filteredPanels = useMemo(() => {
        return allPanels.filter(panel => {
            const lid = panel.local_id || "";
            const matchesSearch = lid.toLowerCase().includes(searchQuery.toLowerCase());
 
            let matchesStatus = false;
            if (statusFilter === 'all') {
                matchesStatus = true;
            } else if (statusFilter === 'normal') {
                const defects = panel.defects || [];
                matchesStatus = defects.length === 0;
            } else {
                matchesStatus = panelHasDefectGroup(panel, statusFilter);
            }
 
            return matchesSearch && matchesStatus;
        });
    }, [allPanels, searchQuery, statusFilter]);

    const stats = useMemo(() => {
        const counts = { Total: allPanels.length, Healthy: 0, Issues: 0 };
        allPanels.forEach(p => {
            if (p.status === "healthy") counts.Healthy++;
            else counts.Issues++;
        });
        return counts;
    }, [allPanels]);

    useEffect(() => {
        const handler = () => {
            if (onRefresh) onRefresh();
        };
        window.addEventListener('review-sync-completed', handler);
        return () => window.removeEventListener('review-sync-completed', handler);
    }, [onRefresh]);

    return (
        <div className="unified-container" style={{ position: "relative" }}>
            {/* CRS.Simple Map Engine */}
            <MapContainer
                crs={L.CRS.Simple}
                center={[0, 0]}
                zoom={-1}
                minZoom={-3}
                maxZoom={2}
                scrollWheelZoom={true}
                zoomControl={false}
                style={{ width: '100%', height: '100%', background: '#030712' }}
            >
                <FitBounds gridData={gridData} focusTarget={focusTarget} />

                {gridData.map((img, i) => (
                    <React.Fragment key={i}>
                        {showHeatmap && (
                            <ImageOverlay
                                url={`${IMAGE_BASE_URL}${img.filename}`}
                                bounds={img.bounds}
                            />
                        )}
                        {focusTarget === img.filename && (
                            <div className="focus-rectangle">
                                <Polygon
                                    positions={[
                                        [img.bounds[0][0], img.bounds[0][1]],
                                        [img.bounds[0][0], img.bounds[1][1]],
                                        [img.bounds[1][0], img.bounds[1][1]],
                                        [img.bounds[1][0], img.bounds[0][1]]
                                    ]}
                                    pathOptions={{ color: '#0ea5e9', weight: 8, fill: false, dashArray: "20, 20" }}
                                />
                            </div>
                        )}
                    </React.Fragment>
                ))}

                {filteredPanels.map((p, i) => {
                    const hasDefect = (p.defects || []).length > 0;
                    const isHealthy = !hasDefect;
                    const panelStrokeColor = hasDefect ? "#ef4444" : "#22c55e";

                    return (
                        <Polygon
                            key={i}
                            positions={p.polygon}
                            pathOptions={{
                                color: panelStrokeColor,
                                fillColor: panelStrokeColor,
                                fillOpacity: hasDefect ? 0.08 : 0.03,
                                weight: hasDefect ? 3 : 2.5,
                                opacity: 1,
                                lineCap: "round",
                                lineJoin: "round",
                            }}
                            eventHandlers={{
                                mouseover: () => setHoveredPanel(p),
                                mouseout: () => setHoveredPanel(null),
                                click: (e) => {
                                    if (e?.originalEvent) {
                                        e.originalEvent.stopPropagation();
                                    }
                                    setClickedPanel(p);
                                }
                            }}
                        >
                        <Tooltip sticky className="bg-slate-900 border-none text-white shadow-xl rounded-lg">
                            <div className="text-sm font-bold text-sky-400">{p.local_id}</div>
                            <div className="text-xs">
                                {isHealthy
                                    ? "Bình thường"
                                    : `${translateDefect(p.main_defect_class)} (-${Number(p.total_panel_loss || 0).toFixed(1)} W)`
                                }
                            </div>
                            {p.worst_severity && !isHealthy && (
                                <div className="text-xs text-orange-300">Mức độ: {p.worst_severity}</div>
                            )}
                            {p.review_label && (
                                <div className="text-xs text-emerald-400 font-bold" style={{ marginTop: 2 }}>
                                    Duyệt: {p.review_label}
                                </div>
                            )}
                            {p.geometry_source && (
                                <div className="text-xs text-slate-400" style={{ marginTop: 2, fontStyle: 'italic' }}>
                                    Polygon: {p.geometry_source === 'v61_line_snap' ? '✓ Line-snap V61' : `⚠ ${p.geometry_source || 'unknown'}`}
                                </div>
                            )}
                        </Tooltip>
                        </Polygon>
                    );
                })}

                {filteredPanels.map((p, i) => {
                    return (p.mappedDefects || []).map((d, di) => {
                        const defectGroup = getDefectGroup(d.class_name);
                        const defectColor = getDefectColorByGroup(defectGroup);
                        const isNeedsReview = d.thermal_validation_status === "needs_review"
                            || d.thermal_validation_status === "suspect_false_positive"
                            || d.thermal_validation_status === "class_mismatch";

                        const dashPatterns = [
                            "6, 4",
                            "2, 4",
                            "10, 4",
                            "1, 6",
                            "12, 3",
                        ];

                        const dashArray = isNeedsReview
                            ? "6, 6"
                            : dashPatterns[d.defect_index % dashPatterns.length];

                        return (
                            <Polygon
                                key={`defect-${i}-${di}`}
                                positions={d.polygon}
                                pathOptions={{
                                    color: defectColor,
                                    fillColor: defectColor,
                                    fillOpacity: 0.06,
                                    weight: 3,
                                    opacity: 1,
                                    dashArray,
                                    lineCap: "round",
                                    lineJoin: "round",
                                }}
                                eventHandlers={{
                                    click: () => {
                                        setClickedPanel(p);
                                    }
                                }}
                            >
                                <Tooltip sticky className="bg-slate-900 border-none text-white shadow-xl rounded-lg">
                                    <div className="text-sm font-bold text-rose-400">{d.class_name || "unknown"}</div>
                                    <div className="text-xs">Panel ID: {p.local_id}</div>
                                    <div className="text-xs">Độ tin cậy nhận diện lỗi YOLO: {d.confidence != null ? formatConfidence(d.confidence) : 'Chưa có dữ liệu'}</div>
                                    <div className="text-xs">Vị trí: {translateLocationInPanel(d.location_in_panel)}</div>
                                    <div className="text-xs">Mức độ: {translateSeverity(d.severity) || 'N/A'}</div>
                                    <div className="text-xs font-semibold">Tỷ lệ vùng lỗi: {d.area_ratio_percent != null ? `${d.area_ratio_percent.toFixed(2)}% diện tích tấm` : 'Chưa có dữ liệu'}</div>
                                    {d.thermal_validation_score != null && (
                                        <div className="text-xs text-orange-300 font-semibold">Điểm kiểm chứng: {(d.thermal_validation_score * 100).toFixed(1)}%</div>
                                    )}
                                </Tooltip>
                            </Polygon>
                        );
                    });
                })}
            </MapContainer>

            {/* Color Legend */}
            <div style={{
                position: 'absolute',
                bottom: 24,
                right: showDrawer ? 520 : 24,
                background: 'rgba(15, 23, 42, 0.88)',
                border: '1px solid rgba(148, 163, 184, 0.25)',
                borderRadius: 12,
                padding: '10px 12px',
                fontSize: 12,
                color: '#e5e7eb',
                zIndex: 1000,
                backdropFilter: 'blur(8px)',
                boxShadow: '0 4px 20px rgba(0, 0, 0, 0.4)',
                transition: 'right 0.4s cubic-bezier(0.16, 1, 0.3, 1)',
                display: 'flex',
                flexDirection: 'column',
                gap: 6,
                minWidth: 160
            }}>
                <div style={{ fontWeight: 700, color: '#f8fafc', borderBottom: '1px solid rgba(255,255,255,0.1)', paddingBottom: 4, marginBottom: 2 }}>
                    CHÚ GIẢI BẢN ĐỒ
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <div style={{ width: 12, height: 12, borderRadius: 3, backgroundColor: '#22c55e', border: '1px solid rgba(255,255,255,0.2)' }} />
                    <span>Panel bình thường</span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <div style={{ width: 12, height: 12, borderRadius: 3, backgroundColor: '#ef4444', border: '1px solid rgba(255,255,255,0.2)' }} />
                    <span>Panel có lỗi</span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <div style={{ width: 12, height: 4, borderRadius: 2, backgroundColor: '#ff3b30' }} />
                    <span>hotspot_single_cell</span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <div style={{ width: 12, height: 4, borderRadius: 2, backgroundColor: '#ff2d55' }} />
                    <span>hotspot_multi_cell</span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <div style={{ width: 12, height: 4, borderRadius: 2, backgroundColor: '#f59e0b' }} />
                    <span>crack</span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <div style={{ width: 12, height: 4, borderRadius: 2, backgroundColor: '#8b5cf6' }} />
                    <span>shading</span>
                </div>
            </div>

            {/* Hover/Click Control Drawer */}
            {(() => {
                const activePanel = clickedPanel || hoveredPanel;
                if (!activePanel) return null;

                const [x1, y1, x2, y2] = activePanel.bbox_px || [0, 0, 0, 0];
                const boxW = x2 - x1;
                const boxH = y2 - y1;

                // 1. Thermal Crop: Thêm 50% padding xung quanh tấm pin để hiện bối cảnh lân cận vừa phải
                const padX_t = Math.round(boxW * 0.5);
                const padY_t = Math.round(boxH * 0.5);

                const cropX1_t = Math.max(0, x1 - padX_t);
                const cropY1_t = Math.max(0, y1 - padY_t);
                const cropX2_t = Math.min(activePanel.imgW, x2 + padX_t);
                const cropY2_t = Math.min(activePanel.imgH, y2 + padY_t);

                const cropW_t = cropX2_t - cropX1_t;
                const cropH_t = cropY2_t - cropY1_t;

                const scale_t = Math.min(420 / Math.max(cropW_t, 1), 160 / Math.max(cropH_t, 1));

                // 2. RGB Crop: Thêm 150% padding xung quanh tấm pin để hiện bối cảnh rất rộng
                const padX_r = Math.round(boxW * 1.5);
                const padY_r = Math.round(boxH * 1.5);

                const cropX1_r = Math.max(0, x1 - padX_r);
                const cropY1_r = Math.max(0, y1 - padY_r);
                const cropX2_r = Math.min(activePanel.imgW, x2 + padX_r);
                const cropY2_r = Math.min(activePanel.imgH, y2 + padY_r);

                const cropW_r = cropX2_r - cropX1_r;
                const cropH_r = cropY2_r - cropY1_r;

                const scale_r = Math.min(420 / Math.max(cropW_r, 1), 160 / Math.max(cropH_r, 1));

                // Định nghĩa màu sắc & nhãn tùy theo mức độ nghiêm trọng
                const hasDefect = (activePanel.defects || []).length > 0;
                const severity = hasDefect ? (activePanel.worst_severity || "Lỗi") : "HEALTHY";
                const isSevere = hasDefect && (severity.toLowerCase() === "severe" || severity.toLowerCase() === "high");
                const themeColor = hasDefect ? (isSevere ? "#ef4444" : "#f59e0b") : "#22c55e";
                const glowShadow = hasDefect 
                    ? (isSevere 
                        ? "0 0 15px rgba(239, 68, 68, 0.4)" 
                        : "0 0 15px rgba(245, 158, 11, 0.4)")
                    : "0 0 15px rgba(34, 197, 94, 0.4)";

                return (
                    <div 
                        className="unified-overlay glass-panel"
                        style={{
                            position: "absolute", top: 24, right: 24, bottom: 24, zIndex: 1000,
                            width: 480, padding: "24px 20px", display: "flex", flexDirection: "column",
                            pointerEvents: clickedPanel ? "auto" : "none", 
                            overflowY: "auto", overflowX: "hidden",
                            border: clickedPanel ? "1px solid rgba(14, 165, 233, 0.6)" : "1px solid rgba(14, 165, 233, 0.35)",
                            boxShadow: "-10px 0 35px rgba(0, 0, 0, 0.65)",
                            background: "rgba(10, 15, 25, 0.92)",
                            backdropFilter: "blur(20px)",
                            borderRadius: 16
                        }}
                    >
                        {/* Toolbar nút chức năng */}
                        <div className="panel-detail-toolbar" style={{ display: "flex", justifyContent: "space-between", marginBottom: 16 }}>
                             <div style={{ display: "flex", gap: 8 }}>
                                 {batchId && (
                                     <button
                                         onClick={() => setIsReviewOpen(true)}
                                         style={{
                                             background: "rgba(239, 68, 68, 0.95)",
                                             border: "1px solid rgba(255,255,255,0.15)",
                                             borderRadius: 8,
                                             padding: "6px 12px",
                                             display: "flex",
                                             alignItems: "center",
                                             gap: 6,
                                             color: "#fff",
                                             fontSize: 12,
                                             fontWeight: 700,
                                             cursor: "pointer",
                                             boxShadow: "0 2px 8px rgba(239,68,68,0.2)"
                                         }}
                                     >
                                         <ShieldAlert size={14} color="#fff" /> Duyệt lỗi phát hiện
                                     </button>
                                 )}
                                 <div style={{
                                     background: "rgba(255, 255, 255, 0.08)",
                                     border: "1px solid rgba(255,255,255,0.1)",
                                     borderRadius: 8,
                                     padding: "6px 12px",
                                     display: "flex",
                                     alignItems: "center",
                                     gap: 6,
                                     color: "#fff",
                                     fontSize: 12,
                                     fontWeight: 600
                                 }}>
                                     <Maximize2 size={14} color="#0EA5E9" /> Bản đồ toàn cảnh
                                 </div>
                             </div>
                             {clickedPanel && (
                                 <button 
                                     onClick={(e) => {
                                         e.stopPropagation();
                                         setClickedPanel(null);
                                         setHoveredPanel(null);
                                     }}
                                     style={{
                                         background: "rgba(255,255,255,0.08)",
                                         border: "1px solid rgba(255,255,255,0.15)",
                                         borderRadius: 8,
                                         padding: 6,
                                         cursor: "pointer",
                                         display: "flex",
                                         alignItems: "center",
                                         justifyContent: "center",
                                         color: "#94A3B8",
                                         transition: "all 0.15s"
                                     }}
                                     onMouseEnter={e => { e.currentTarget.style.background = "rgba(239, 68, 68, 0.2)"; e.currentTarget.style.color = "#FF8A8A"; }}
                                     onMouseLeave={e => { e.currentTarget.style.background = "rgba(255,255,255,0.08)"; e.currentTarget.style.color = "#94A3B8"; }}
                                 >
                                     <X size={15} />
                                 </button>
                             )}
                         </div>

                        {/* Header của Control Panel */}
                        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", borderBottom: "1px solid rgba(255,255,255,0.08)", paddingBottom: 16, marginBottom: 18 }}>
                            <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                                <div style={{ 
                                    width: 10, height: 10, borderRadius: "50%", 
                                    background: themeColor, boxShadow: glowShadow,
                                    animation: "pulse 1.8s infinite"
                                }} />
                                <span style={{ color: "#F8FAFC", fontWeight: 800, fontSize: 18, letterSpacing: "0.5px" }}>
                                    THÔNG TIN TẤM PIN: {activePanel.local_id}
                                </span>
                            </div>
                            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                                {activePanel.review_status && (
                                    <span style={{ 
                                        color: activePanel.review_status === 'confirmed_defect' ? '#fca5a5' : 
                                               activePanel.review_status === 'needs_review' ? '#fde047' : 
                                               activePanel.review_status === 'false_positive' ? '#94a3b8' : '#e2e8f0',
                                        background: activePanel.review_status === 'confirmed_defect' ? 'rgba(239, 68, 68, 0.15)' : 
                                                    activePanel.review_status === 'needs_review' ? 'rgba(245, 158, 11, 0.15)' : 
                                                    activePanel.review_status === 'false_positive' ? 'rgba(148, 163, 184, 0.15)' : 'rgba(255,255,255,0.05)',
                                        fontWeight: 700, fontSize: 11, padding: "4px 10px", 
                                        borderRadius: 6, textTransform: "uppercase",
                                        border: '1px solid ' + (
                                            activePanel.review_status === 'confirmed_defect' ? 'rgba(239, 68, 68, 0.3)' : 
                                            activePanel.review_status === 'needs_review' ? 'rgba(245, 158, 11, 0.3)' : 
                                            activePanel.review_status === 'false_positive' ? 'rgba(148, 163, 184, 0.3)' : 'rgba(255,255,255,0.1)'
                                        ),
                                        letterSpacing: "0.5px"
                                    }}>
                                        {activePanel.review_status === 'confirmed_defect' ? 'Đã xác nhận' :
                                         activePanel.review_status === 'needs_review' ? 'Cần kiểm tra lại' :
                                         activePanel.review_status === 'false_positive' ? 'Bỏ qua' :
                                         activePanel.review_label || 'Chưa duyệt'}
                                    </span>
                                )}
                                <span style={{ 
                                    color: "#FFFFFF", background: themeColor, 
                                    fontWeight: 700, fontSize: 11, padding: "4px 10px", 
                                    borderRadius: 6, textTransform: "uppercase",
                                    boxShadow: glowShadow, letterSpacing: "0.5px"
                                }}>
                                    {translateSeverity(severity)}
                                </span>
                            </div>
                        </div>

                        {/* Lượng thất thoát sản lượng dạng SCADA Banner */}
                        <div style={{
                            background: hasDefect 
                                ? "linear-gradient(90deg, rgba(239, 68, 68, 0.12), rgba(15, 23, 42, 0.2))"
                                : "linear-gradient(90deg, rgba(34, 197, 94, 0.12), rgba(15, 23, 42, 0.2))",
                            border: `1px solid ${themeColor}50`,
                            borderRadius: 12, padding: "16px 20px", marginBottom: 20,
                            boxShadow: `inset 0 0 15px ${themeColor}15`,
                            display: "flex", flexDirection: "column", gap: 4
                        }}>
                            <span style={{ color: "#94A3B8", fontSize: 11, fontWeight: 700, textTransform: "uppercase", letterSpacing: "1px" }}>
                                Ước tính sản lượng thất thoát
                            </span>
                            <div style={{ display: "flex", alignItems: "baseline", gap: 8 }}>
                                <span style={{ color: themeColor, fontSize: 32, fontWeight: 900, textShadow: glowShadow, letterSpacing: "-0.5px" }}>
                                    -{Number(activePanel.total_panel_loss || 0).toFixed(1)} W
                                </span>
                                <span style={{ color: "#cbd5e1", fontSize: 14, fontWeight: 500 }}>
                                    / {Number(panelPower).toFixed(1)} W định mức
                                </span>
                            </div>
                        </div>

                        {/* Danh sách lỗi phát hiện */}
                        <div style={{ marginBottom: 20 }}>
                            <h3 style={{ color: "#38bdf8", fontSize: 12, fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.5px", marginBottom: 10, display: "flex", alignItems: "center", gap: 6 }}>
                                <Activity size={14} /> Danh sách lỗi phát hiện
                            </h3>
                            {activePanel.defects && activePanel.defects.length > 0 ? (
                                <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                                    {(() => {
                                        const groupedMap = {};
                                        (activePanel.defects || []).forEach(d => {
                                            const cls = d.class_name || "Unknown";
                                            if (!groupedMap[cls]) {
                                                groupedMap[cls] = {
                                                    class_name: cls,
                                                    area_ratio_percent: 0,
                                                    locations: new Set(),
                                                    severities: new Set(),
                                                    confidences: [],
                                                };
                                            }
                                            groupedMap[cls].area_ratio_percent += d.area_ratio_percent || d.loss_pct || 0;
                                            if (d.location_in_panel) {
                                                groupedMap[cls].locations.add(d.location_in_panel);
                                            }
                                            if (d.severity) {
                                                groupedMap[cls].severities.add(d.severity);
                                            }
                                            if (d.confidence != null) {
                                                groupedMap[cls].confidences.push(d.confidence);
                                            }
                                        });

                                        const groupedList = Object.values(groupedMap).map(grouped => {
                                            const locationList = Array.from(grouped.locations);
                                            const location_in_panel = locationList.length > 0 ? locationList.join(", ") : "Trung tâm";
                                            
                                            const severity_order = ["very_minor", "minor", "moderate", "severe", "replace"];
                                            const severityList = Array.from(grouped.severities);
                                            const severity = severityList.length > 0
                                                ? severityList.reduce((worst, current) => {
                                                    return severity_order.indexOf(current) > severity_order.indexOf(worst) ? current : worst;
                                                  }, "very_minor")
                                                : "Trung bình";

                                            const avgConfidence = grouped.confidences.length > 0
                                                ? (grouped.confidences.reduce((a, b) => a + b, 0) / grouped.confidences.length)
                                                : null;

                                            return {
                                                class_name: grouped.class_name,
                                                area_ratio_percent: grouped.area_ratio_percent,
                                                location_in_panel,
                                                severity,
                                                confidence: avgConfidence,
                                            };
                                        });

                                        return groupedList.map((d, i) => {
                                            const ratio = d.area_ratio_percent;
                                            const defectGroup = getDefectGroup(d.class_name);
                                            const defectColor = getDefectColorByGroup(defectGroup);
                                            return (
                                                <div key={i} style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.05)", borderRadius: 10, padding: 12 }}>
                                                    <div style={{ display: "flex", justifyContent: "space-between", fontSize: 13, fontWeight: 600, marginBottom: 4 }}>
                                                        <span style={{ color: defectColor }}>{d.class_name || "unknown"}</span>
                                                        <span style={{ color: "#E2E8F0" }}>Tỷ lệ vùng lỗi: {ratio > 0 ? `${ratio.toFixed(2)}% diện tích tấm` : 'Chưa có dữ liệu'}</span>
                                                    </div>
                                                    <div style={{ display: "flex", flexDirection: "column", gap: 8, marginTop: 10 }}>
                                                        <div>
                                                            <span style={{ fontSize: 12, fontWeight: 700, color: "#94A3B8" }}>Vị trí: </span>
                                                            <span className="px-2 py-1 rounded-md bg-cyan-500/15 text-cyan-300 border border-cyan-500/30 text-xs font-bold">
                                                                {translateLocationInPanel(d.location_in_panel)}
                                                            </span>
                                                        </div>
                                                        <div>
                                                            <span style={{ fontSize: 12, fontWeight: 700, color: "#94A3B8" }}>Mức độ: </span>
                                                            {(() => {
                                                                const sev = String(d.severity || "").toLowerCase();
                                                                const isHighSev = ["critical", "severe", "high", "replace"].includes(sev);
                                                                const sevColor = isHighSev ? "#ef4444" : "#f59e0b";
                                                                return (
                                                                    <span 
                                                                        className="px-2 py-1 rounded-md text-xs font-bold" 
                                                                        style={{ 
                                                                            color: sevColor, 
                                                                            backgroundColor: `${sevColor}20`, 
                                                                            border: `1px solid ${sevColor}50` 
                                                                        }}
                                                                    >
                                                                        {translateSeverity(d.severity || "Trung bình")}
                                                                    </span>
                                                                );
                                                            })()}
                                                        </div>
                                                        {d.confidence != null && (
                                                            <div style={{ fontSize: 12, fontWeight: 700, color: "#94A3B8", marginTop: 2 }}>
                                                                Độ tin cậy nhận diện lỗi YOLO: <span style={{ color: "#38bdf8" }}>{formatConfidence(d.confidence)}</span>
                                                            </div>
                                                        )}
                                                    </div>
                                                    <div style={{ width: "100%", height: 6, background: "rgba(255,255,255,0.06)", borderRadius: 3, marginTop: 12, overflow: "hidden" }}>
                                                        <div style={{ width: `${Math.min(100, ratio)}%`, height: "100%", background: themeColor, borderRadius: 3 }} />
                                                    </div>
                                                </div>
                                            );
                                        });
                                    })()}
                                </div>
                            ) : (
                                <div style={{ fontSize: 13, color: "#94A3B8", fontStyle: "italic", background: "rgba(255,255,255,0.02)", padding: 10, borderRadius: 8, textAlign: "center" }}>
                                    Không ghi nhận thông số lỗi chi tiết.
                                </div>
                            )}
                        </div>

                        {/* Trạng thái xử lý O&M */}
                        <div style={{ marginBottom: 20 }}>
                            <h3 style={{ color: "#38bdf8", fontSize: 12, fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.5px", marginBottom: 10, display: "flex", alignItems: "center", gap: 6 }}>
                                <Activity size={14} /> Trạng thái xử lý (O&M)
                            </h3>
                            {hasDefect ? (
                                <div style={{
                                    background: "rgba(255,255,255,0.03)",
                                    border: "1px solid rgba(255,255,255,0.05)",
                                    borderRadius: 10,
                                    padding: 12,
                                    fontSize: 12,
                                    display: "flex",
                                    flexDirection: "column",
                                    gap: 8,
                                    color: "#cbd5e1"
                                }}>
                                    <div>
                                        <span style={{ color: "#94a3b8" }}>Trạng thái duyệt: </span>
                                        <span style={{
                                            fontWeight: 700,
                                            color: activePanel.review_status === 'confirmed_defect' ? '#fca5a5' :
                                                   activePanel.review_status === 'needs_review' ? '#fbbf24' :
                                                   activePanel.review_status === 'false_positive' ? '#94a3b8' : '#cbd5e1'
                                        }}>
                                            {activePanel.review_status === 'confirmed_defect' ? 'Đã xác nhận' :
                                             activePanel.review_status === 'needs_review' ? 'Cần kiểm tra lại' :
                                             activePanel.review_status === 'false_positive' ? 'Bỏ qua' : 'Chưa duyệt'}
                                        </span>
                                    </div>
                                    <div>
                                        <span style={{ color: "#94a3b8" }}>Ưu tiên xử lý: </span>
                                        <span style={{
                                            fontWeight: 700,
                                            color: activePanel.maintenance_priority === 'urgent' ? '#ef4444' :
                                                   activePanel.maintenance_priority === 'high' ? '#f97316' :
                                                   activePanel.maintenance_priority === 'medium' ? '#fbbf24' : '#10b981'
                                        }}>
                                            {activePanel.maintenance_priority === 'urgent' ? 'Khẩn cấp' :
                                             activePanel.maintenance_priority === 'high' ? 'Cao' :
                                             activePanel.maintenance_priority === 'medium' ? 'Trung bình' :
                                             activePanel.maintenance_priority === 'low' ? 'Thấp' : 'Trung bình'}
                                        </span>
                                    </div>
                                    <div>
                                        <span style={{ color: "#94a3b8" }}>Người duyệt: </span>
                                        <span style={{ fontWeight: 600 }}>{activePanel.reviewer_name || "Chưa cập nhật"}</span>
                                    </div>
                                    <div>
                                        <span style={{ color: "#94a3b8" }}>Thời gian duyệt: </span>
                                        <span style={{ fontWeight: 600 }}>{activePanel.reviewed_at || "Chưa cập nhật"}</span>
                                    </div>
                                    <div>
                                        <span style={{ color: "#94a3b8" }}>Ghi chú: </span>
                                        <span style={{ fontStyle: activePanel.review_note ? "normal" : "italic" }}>
                                            {activePanel.review_note || "Không có ghi chú"}
                                        </span>
                                    </div>
                                </div>
                            ) : (
                                <div style={{ fontSize: 13, color: "#94A3B8", fontStyle: "italic", background: "rgba(255,255,255,0.02)", padding: 10, borderRadius: 8, textAlign: "center" }}>
                                    Không có lỗi cần xử lý.
                                </div>
                            )}
                        </div>

                        {/* Kiểm chứng nhiệt tương đối */}
                        {activePanel.defects && activePanel.defects.length > 0 && activePanel.defects.some(d => d.thermal_validation_status) && (
                            <div style={{ marginBottom: 20 }}>
                                <h3 style={{ color: "#38bdf8", fontSize: 12, fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.5px", marginBottom: 10, display: "flex", alignItems: "center", gap: 6 }}>
                                    <Thermometer size={14} /> Kiểm chứng nhiệt tương đối
                                </h3>
                                <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                                    {activePanel.defects.filter(d => d.thermal_validation_status).map((d, idx) => {
                                        const tvStatus = d.thermal_validation_status;
                                        if (tvStatus === 'not_run') return null;

                                        const statusMeta = {
                                            confirmed_by_relative_thermal: { label: 'Đã xác nhận bằng tương phản nhiệt', color: '#10b981', bg: 'rgba(16,185,129,0.08)', icon: '✔' },
                                            needs_review:                  { label: 'Cần xem xét',                       color: '#f59e0b', bg: 'rgba(245,158,11,0.08)',  icon: '⚠' },
                                            class_mismatch:                { label: 'Nghi ngờ sai loại lỗi',             color: '#f97316', bg: 'rgba(249,115,22,0.08)',  icon: '⚠' },
                                            suspect_false_positive:        { label: 'Nghi ngờ nhận nhầm',                color: '#ef4444', bg: 'rgba(239,68,68,0.08)',   icon: '✗' },
                                            insufficient_pixels:           { label: 'Không đủ dữ liệu ảnh',             color: '#64748b', bg: 'rgba(100,116,139,0.08)', icon: '?' },
                                            not_run:                       { label: 'Chưa chạy',                        color: '#64748b', bg: 'rgba(100,116,139,0.08)', icon: '-' },
                                        };
                                        const sm = statusMeta[tvStatus] || statusMeta.not_run;
                                        const fmt = (v) => v != null ? Number(v).toFixed(3) : 'N/A';

                                        return (
                                            <div key={idx} style={{
                                                padding: '12px 14px', backgroundColor: sm.bg,
                                                border: `1px solid ${sm.color}30`, borderRadius: 12
                                            }}>
                                                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                                                    <span style={{ color: '#cbd5e1', fontSize: 12, fontWeight: 700 }}>Lỗi: {translateDefect(d.class_name)}</span>
                                                    <span style={{ color: sm.color, fontSize: 11, fontWeight: 700, display: 'flex', alignItems: 'center', gap: 4 }}>
                                                        {sm.icon} {sm.label}
                                                    </span>
                                                </div>

                                                <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1fr', gap: '4px 12px', fontSize: 11 }}>
                                                    <div style={{ color: '#64748b' }}>Điểm kiểm chứng</div>
                                                    <div style={{ color: sm.color, fontWeight: 700 }}>
                                                        {d.thermal_validation_score != null ? (d.thermal_validation_score * 100).toFixed(0) + '%' : 'N/A'}
                                                    </div>

                                                    <div style={{ color: '#64748b' }}>Chênh lệch nhiệt (Δhot)</div>
                                                    <div style={{ color: d.relative_hot_delta >= 0.12 ? '#f87171' : '#cbd5e1', fontWeight: 600 }}>
                                                        {fmt(d.relative_hot_delta)}
                                                    </div>

                                                    <div style={{ color: '#64748b' }}>Khử màu lam (Blue suppression)</div>
                                                    <div style={{ color: d.blue_suppression >= 0.06 ? '#fb923c' : '#cbd5e1', fontWeight: 600 }}>
                                                        {fmt(d.blue_suppression)}
                                                    </div>

                                                    <div style={{ color: '#64748b' }}>Độ không đồng nhất (TNI)</div>
                                                    <div style={{ color: d.tni >= 0.12 ? '#facc15' : '#cbd5e1', fontWeight: 600 }}>
                                                        {fmt(d.tni)}
                                                    </div>

                                                    <div style={{ color: '#64748b' }}>Đề xuất / Phân loại tự động</div>
                                                    <div style={{ color: '#38bdf8', fontWeight: 600 }}>
                                                        {d.rule_class ? `${translateDefect(d.rule_class)} (${d.suggested_review_status === 'confirmed_defect' ? 'Xác nhận' : 'Xem xét'})` : 'N/A'}
                                                    </div>
                                                </div>
                                            </div>
                                        );
                                    })}
                                </div>
                            </div>
                        )}

                        {/* So sánh Side-by-Side xếp dọc */}
                        <div style={{ display: "flex", flexDirection: "column", gap: 16, marginBottom: 20 }}>
                            {/* Thermal Crop */}
                            <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                                <span style={{ color: "#94A3B8", fontSize: 11, fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.5px" }}>
                                    Ảnh hồng ngoại (Nhiệt - Zoom cận)
                                </span>
                                <div style={{
                                    width: "100%", height: 180, background: "#05070c", borderRadius: 10, overflow: "hidden",
                                    display: "flex", alignItems: "center", justifyContent: "center",
                                    border: "1px solid rgba(239, 68, 68, 0.25)", position: "relative"
                                }}>
                                    <div style={{
                                        width: cropW_t, height: cropH_t,
                                        overflow: "hidden", position: "absolute",
                                        transform: `scale(${scale_t})`,
                                        transformOrigin: "center center"
                                    }}>
                                        <img
                                            src={`${IMAGE_BASE_URL}${activePanel.source_thermal}`}
                                            style={{ position: "absolute", left: -cropX1_t, top: -cropY1_t, maxWidth: "none" }}
                                        />
                                        <div style={{
                                            position: "absolute",
                                            left: x1 - cropX1_t,
                                            top: y1 - cropY1_t,
                                            width: boxW,
                                            height: boxH,
                                            border: "2px solid #ef4444"
                                        }} />
                                    </div>
                                </div>
                            </div>

                            {/* RGB Crop */}
                            <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                                <span style={{ color: "#94A3B8", fontSize: 11, fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.5px" }}>
                                    Ảnh thực tế (RGB - Bối cảnh rộng)
                                </span>
                                <div style={{
                                    width: "100%", height: 180, background: "#05070c", borderRadius: 10, overflow: "hidden",
                                    display: "flex", alignItems: "center", justifyContent: "center",
                                    border: "1px solid rgba(14, 165, 233, 0.25)", position: "relative"
                                }}>
                                    {activePanel.source_rgb ? (
                                        <div style={{
                                            width: cropW_r, height: cropH_r,
                                            overflow: "hidden", position: "absolute",
                                            transform: `scale(${scale_r})`,
                                            transformOrigin: "center center"
                                        }}>
                                            <img
                                                src={`${RAW_IMAGE_BASE_URL}${activePanel.source_rgb}`}
                                                style={{
                                                    position: "absolute",
                                                    left: -cropX1_r,
                                                    top: -cropY1_r,
                                                    width: activePanel.imgW,
                                                    height: activePanel.imgH,
                                                    maxWidth: "none"
                                                }}
                                            />
                                        </div>
                                    ) : (
                                        <span style={{ color: "#64748B", fontSize: 11 }}>Không có ảnh RGB</span>
                                    )}
                                </div>
                            </div>
                        </div>

                        {/* Tech specs/Metadata */}
                        <div style={{ marginTop: "auto", borderTop: "1px solid rgba(255,255,255,0.08)", paddingTop: 16 }}>
                            <h4 style={{ color: "#e2e8f0", fontSize: 11, fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.5px", marginBottom: 8, display: "flex", alignItems: "center", gap: 6 }}>
                                <Battery size={14} color="#0EA5E9" /> THÔNG TIN TẤM PIN
                            </h4>
                            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.04)", borderRadius: 10, padding: 10 }}>
                                <div style={{ fontSize: 11 }}>
                                    <div style={{ color: "#64748B" }}>Tọa độ tâm (chuẩn hóa)</div>
                                    <div style={{ color: "#E2E8F0", fontWeight: 600, marginTop: 2 }}>
                                        x: {(((x1 + x2) / 2) / (activePanel.imgW || 640)).toFixed(3)}, y: {(1 - ((y1 + y2) / 2) / (activePanel.imgH || 512)).toFixed(3)}
                                    </div>
                                    <div style={{ color: "#64748B", fontSize: 9, marginTop: 2 }}>
                                        Hệ Descartes chuẩn hóa [0,1]
                                    </div>
                                    <div style={{ color: "#38bdf8", fontWeight: 600, marginTop: 2 }}>
                                        Khu vực ảnh: {getImageRegionFromCartesian(
                                            Math.round((x1 + x2) / 2),
                                            activePanel.imgH ? (activePanel.imgH - Math.round((y1 + y2) / 2)) : Math.round((y1 + y2) / 2),
                                            activePanel.imgW || 640,
                                            activePanel.imgH || 512
                                        )}
                                    </div>
                                </div>
                                <div style={{ fontSize: 11 }}>
                                    <div style={{ color: "#64748B" }}>Kích thước nguồn</div>
                                    <div style={{ color: "#E2E8F0", fontWeight: 600, marginTop: 2 }}>
                                        {activePanel.imgW} x {activePanel.imgH} px
                                    </div>
                                </div>
                                {activePanel.yolo_confidence != null && (
                                    <div style={{ fontSize: 11, gridColumn: "span 2" }}>
                                        <div style={{ color: "#64748B" }}>Độ tin cậy nhận diện panel (YOLO)</div>
                                        <div style={{ color: "#38bdf8", fontWeight: 600, marginTop: 2 }}>
                                            {(activePanel.yolo_confidence * 100).toFixed(1)}%
                                        </div>
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>
                );
            })()}

            {/* Left Sidebar */}
            {viewMode === 'monitor' && (
                <div 
                    className="unified-overlay sidebar-left glass-panel" 
                    style={{ 
                        zIndex: 1000,
                        transform: isSidebarOpen ? 'translateX(0)' : 'translateX(calc(-100% - 24px))',
                        transition: 'transform 0.4s cubic-bezier(0.16, 1, 0.3, 1)'
                    }}
                >
                    {/* Toggle Button */}
                    <button 
                        onClick={() => setIsSidebarOpen(!isSidebarOpen)}
                        style={{
                            position: 'absolute',
                            right: -32,
                            top: 24,
                            width: 32,
                            height: 48,
                            background: 'rgba(10, 15, 25, 0.95)',
                            border: '1px solid rgba(255, 255, 255, 0.12)',
                            borderLeft: 'none',
                            borderRadius: '0 8px 8px 0',
                            color: '#38bdf8',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            cursor: 'pointer',
                            zIndex: 1001,
                            backdropFilter: 'blur(16px)',
                            boxShadow: '12px 0 24px rgba(0, 0, 0, 0.4)'
                        }}
                    >
                        {isSidebarOpen ? <ChevronLeft size={20} /> : <ChevronRight size={20} />}
                    </button>

                    <h2 className="text-xl font-bold mb-6 flex items-center gap-2 text-white">
                        <MapIcon className="text-sky-400" size={24} />
                        Bản đồ toàn cảnh
                    </h2>

                    <div className="relative mb-6">
                        <Search className="absolute left-3 top-2.5 text-slate-500" size={16} />
                        <input
                            type="text"
                            placeholder="Tìm kiếm ID (vd. R01_C03)..."
                            className="w-full bg-slate-800/50 border border-slate-700 rounded-lg py-2 pl-10 pr-4 text-sm focus:outline-none focus:ring-2 focus:ring-sky-500 text-white"
                            value={searchQuery}
                            onChange={(e) => setSearchQuery(e.target.value)}
                        />
                    </div>

                    <div className="space-y-4">
                        <div>
                            <label className="text-[10px] text-slate-400 uppercase font-bold mb-2 block">Lọc theo loại lỗi</label>
                            <div className="grid grid-cols-2 gap-2">
                                {[
                                    { key: "all", label: "Tất cả" },
                                    { key: "normal", label: "Bình thường", color: "#22c55e" },
                                    { key: "hotspot_single", label: "hotspot_single_cell", color: "#ff3b30" },
                                    { key: "hotspot_multi", label: "hotspot_multi_cell", color: "#ff2d55" },
                                    { key: "crack", label: "crack", color: "#f59e0b" },
                                    { key: "shading", label: "shading", color: "#8b5cf6" },
                                ].map(item => (
                                    <button
                                        key={item.key}
                                        onClick={() => setStatusFilter(item.key)}
                                        className={`text-xs p-2 rounded-lg border flex items-center gap-2 transition-all ${
                                            statusFilter === item.key
                                                ? 'text-white font-semibold'
                                                : 'bg-slate-800/50 border-slate-700 text-slate-400 hover:text-slate-300'
                                        }`}
                                        style={
                                            statusFilter === item.key
                                                ? {
                                                      backgroundColor: item.color ? `${item.color}20` : 'rgba(14, 165, 233, 0.2)',
                                                      borderColor: item.color || '#38bdf8',
                                                      boxShadow: item.color ? `0 0 8px ${item.color}40` : 'none',
                                                  }
                                                : {}
                                        }
                                    >
                                        {item.color && (
                                            <div className="w-2 h-2 rounded-full flex-shrink-0" style={{ background: item.color }} />
                                        )}
                                        <span className="truncate">{item.label}</span>
                                    </button>
                                ))}
                            </div>
                        </div>


                    </div>

                    <div className="mt-8 pt-6 border-t border-slate-700/50">
                        <div className="grid grid-cols-2 gap-4">
                            <div className="p-3 bg-slate-800/50 rounded-xl text-center border border-slate-700/50">
                                <div className="text-[10px] text-slate-400 uppercase">Bình thường</div>
                                <div className="text-xl font-bold text-emerald-400">{stats.Healthy}</div>
                            </div>
                            <div className="p-3 bg-slate-800/50 rounded-xl text-center border border-slate-700/50">
                                <div className="text-[10px] text-slate-400 uppercase">Phát hiện lỗi</div>
                                <div className="text-xl font-bold text-rose-400">{stats.Issues}</div>
                            </div>
                        </div>
                    </div>

                    {!data || data.length === 0 ? (
                        <div style={{ marginTop: 20, padding: 16, background: "rgba(239, 68, 68, 0.1)", border: "1px solid rgba(239, 68, 68, 0.3)", borderRadius: 12, color: "#FCA5A5", fontSize: 13, textAlign: "center" }}>
                            Không có dữ liệu. Vui lòng tải ảnh drone và chạy phân tích AI ở Trang chủ!
                        </div>
                    ) : null}
                </div>
            )}

            {/* Top right floating info */}
            {!showDrawer && (
                <div style={{ position: "absolute", top: 24, right: 24, zIndex: 1000, display: "flex", gap: 12 }}>
                    {batchId && (
                        <button
                            onClick={() => setIsReviewOpen(true)}
                            style={{
                                background: "rgba(239, 68, 68, 0.95)",
                                backdropFilter: "blur(12px)",
                                border: "1px solid rgba(255,255,255,0.15)",
                                borderRadius: 12,
                                padding: "8px 16px",
                                display: "flex",
                                alignItems: "center",
                                gap: 8,
                                color: "#fff",
                                fontSize: 13,
                                fontWeight: 700,
                                cursor: "pointer",
                                boxShadow: "0 4px 12px rgba(239,68,68,0.25)",
                                transition: "all 0.2s"
                            }}
                            onMouseEnter={(e) => e.currentTarget.style.transform = "translateY(-1px)"}
                            onMouseLeave={(e) => e.currentTarget.style.transform = "none"}
                        >
                            <ShieldAlert size={16} color="#fff" /> Duyệt lỗi phát hiện
                        </button>
                    )}
                    <div style={{ background: "rgba(15, 23, 42, 0.8)", backdropFilter: "blur(12px)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 12, padding: "8px 16px", display: "flex", alignItems: "center", gap: 8, color: "#fff", fontSize: 13, fontWeight: 600 }}>
                        <Maximize2 size={16} color="#0EA5E9" /> Bản đồ toàn cảnh
                    </div>
                </div>
            )}

            <DefectReviewModal 
                isOpen={isReviewOpen} 
                onClose={() => setIsReviewOpen(false)} 
                batchId={batchId} 
                onRefresh={onRefresh} 
            />
        </div>
    );
}
