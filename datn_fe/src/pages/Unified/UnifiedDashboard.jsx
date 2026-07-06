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
const IMAGE_BASE_URL = "/data/precalib/";
const RAW_IMAGE_BASE_URL = "/data/raw/";

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
    // Clamp tọa độ pixel vào trong kích thước ảnh gốc
    // để tránh polygon vượt ra ngoài viền của ImageOverlay.
    const px = Math.max(0, Math.min(Number(x), srcW));
    const py = Math.max(0, Math.min(Number(y), srcH));

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
  if (!location) return "Không xác định";
  
  const map = {
    "upper-left": "Góc trên trái",
    "upper-center": "Trên giữa",
    "upper-right": "Góc trên phải",
    "middle-left": "Giữa trái",
    "middle-center": "Trung tâm",
    "middle-right": "Giữa phải",
    "lower-left": "Góc dưới trái",
    "lower-center": "Dưới giữa",
    "lower-right": "Góc dưới phải",
    "bottom-left": "Góc dưới trái",
    "bottom-center": "Dưới giữa",
    "bottom-right": "Góc dưới phải",
    "bottom": "Dưới",
    "top": "Trên",
  };

  const getOrder = (key) => {
    let v = 2, u = 1;
    if (key.includes('upper') || key.includes('top')) v = 0;
    else if (key.includes('middle')) v = 1;
    else if (key.includes('lower') || key.includes('bottom')) v = 2;
    
    if (key.includes('left')) u = 0;
    else if (key.includes('center')) u = 1;
    else if (key.includes('right')) u = 2;
    
    return v * 10 + u;
  };

  // Tách các vị trí, loại bỏ khoảng trắng dư thừa, chuyển thành chữ thường, 
  // lọc các giá trị rỗng, sort theo order rồi mới dịch và nối lại.
  const parts = location.split(',')
    .map(p => p.trim().toLowerCase())
    .filter(p => p.length > 0);

  // Chỉ sort nếu nó chứa các từ khoá chỉ vị trí
  const hasKeywords = parts.some(p => Object.keys(map).includes(p));
  
  if (hasKeywords) {
    parts.sort((a, b) => getOrder(a) - getOrder(b));
  }

  // Loại bỏ các vị trí trùng lặp (nếu có do gom nhóm)
  const uniqueParts = [...new Set(parts)];

  return uniqueParts.map(key => map[key] || key).join(', ');
}

function translateSeverity(value) {
  const v = String(value || "").toLowerCase();

  const map = {
    "healthy": "Bình thường",
    "level_1_monitoring": "Mức 1 – Theo dõi",
    "level_2_inspection": "Mức 2 – Cần kiểm tra",
    "level_3_priority": "Mức 3 – Ưu tiên xử lý",
    "recheck_required": "Cần chụp lại",
  };

  return map[v] || "Chưa phân loại";
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

const isValidPolygon = (poly) => {
  return Array.isArray(poly) && poly.length >= 3;
};

const SHOW_RAW_YOLO_POLYGON = false;

const getDefectRenderPolygon = (d) => {
  const isValidPolygon = (p) => Array.isArray(p) && p.length >= 3;

  if (isValidPolygon(d.display_polygon)) {
    return { polygon: d.display_polygon, source: "display_polygon", sourceLabel: "Polygon đã xử lý" };
  }
  if (isValidPolygon(d.polygon)) {
    return { polygon: d.polygon, source: "polygon", sourceLabel: "Polygon đã xử lý" };
  }
  if (isValidPolygon(d.analysis_polygon)) {
    return { polygon: d.analysis_polygon, source: "analysis_polygon", sourceLabel: "Polygon phân tích" };
  }

  if (SHOW_RAW_YOLO_POLYGON) {
    if (isValidPolygon(d.yolo_polygon)) {
      return { polygon: d.yolo_polygon, source: "yolo_polygon", sourceLabel: "YOLO thô - debug" };
    }
    if (isValidPolygon(d.raw_yolo_polygon)) {
      return { polygon: d.raw_yolo_polygon, source: "raw_yolo_polygon", sourceLabel: "YOLO gốc - debug" };
    }
  }

  const bboxPoly = polygonFromBBox(d.bbox || d.box);
  if (isValidPolygon(bboxPoly)) {
    return { polygon: bboxPoly, source: "bbox", sourceLabel: "BBox fallback" };
  }

  return { polygon: null, source: "none", sourceLabel: "Không có polygon hợp lệ" };
};

export default function UnifiedDashboard({ data, panelPower = 600, focusTarget, batchId, onRefresh }) {
    const [isSidebarOpen, setIsSidebarOpen] = useState(true);
    const [isReviewOpen, setIsReviewOpen] = useState(false);
    const PANEL_RATED_POWER_W = 400; // Công suất định mức tấm pin 400W
    const translateDefect = (cls) => {
        if (!cls) return "unknown";
        const c = cls.toLowerCase();
        if (c.includes("hotspot_single") || c.includes("single_cell") || c.includes("single-cell")) return "hotspot_single_cell";
        if (c.includes("hotspot_multi") || c.includes("multi_cell") || c.includes("multicell") || c.includes("multi-cell")) return "hotspot_multi_cell";
        if (c.includes("hotspot") || c.includes("hot")) return "hotspot_single_cell";
        if (c.includes("crack") || c.includes("nut")) return "crack";
        if (
            c.includes("shading") ||
            c.includes("shadow") ||
            c.includes("shade") ||
            c.includes("soil") ||
            c.includes("soiling") ||
            c.includes("dirt")
        ) {
            return "shading";
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
                    const renderInfo = getDefectRenderPolygon(d);
                    const rawPoly = renderInfo.polygon;

                    if (!rawPoly) return null;

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
                      render_source: renderInfo.source,
                      render_source_label: renderInfo.sourceLabel,
                    };
                  })
                  .filter(Boolean);

                return {
                    ...normalizedP,
                    source_rgb: img.rgb_image,
                    source_thermal: img.filename,
                    quality_status: img.quality_status || "ok",
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
                        <Tooltip sticky className="bg-slate-900 border-none text-white shadow-xl rounded-lg px-3 py-2">
                            <div className="text-sm font-bold text-sky-400 mb-1">Tấm pin: {p.local_id}</div>
                            {isHealthy ? (
                                <div className="text-xs text-emerald-400 font-semibold">Bình thường</div>
                            ) : (
                                <>
                                    <div className="text-xs text-rose-400 font-semibold mb-1">
                                        Tấm pin có lỗi
                                    </div>
                                    <div className="text-xs text-slate-300">
                                        Giảm phát ước tính: <span className="text-rose-400 font-bold">{Number(p.total_panel_loss || 0).toFixed(1)} W</span>
                                    </div>
                                    {p.worst_severity && (
                                        <div className="text-xs text-orange-300 mt-1">
                                            Mức độ: {translateSeverity(p.worst_severity)}
                                        </div>
                                    )}
                                </>
                            )}
                        </Tooltip>
                        </Polygon>
                    );
                })}

                {filteredPanels.map((p, i) => {
                    return (p.mappedDefects || []).map((d, di) => {
                        const defectGroup = getDefectGroup(d.class_name);
                        const defectColor = getDefectColorByGroup(defectGroup);
                        const isNeedsReview = d.thermal_validation_status === "weak_relative_contrast"
                            || d.thermal_validation_status === "thermal_mismatch_need_review"
                            || d.thermal_validation_status === "cooler_than_panel_background_recheck"
                            || d.thermal_validation_status === "shading_need_recheck"
                            || d.thermal_validation_status === "shape_based_detection_review";

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
                                    lineCap: "round",
                                    lineJoin: "round",
                                    smoothFactor: 2, // Làm mượt các điểm ảnh lởm chởm
                                }}
                                eventHandlers={{
                                    click: () => {
                                        setClickedPanel(p);
                                    }
                                }}
                            >
                                <Tooltip sticky className="bg-slate-900 border-none text-white shadow-xl rounded-lg px-3 py-2 flex flex-col gap-1">
                                    <div className="text-sm font-bold text-rose-400 mb-1">{translateDefect(d.class_name)}</div>
                                    <div className="text-xs text-slate-300">Tấm pin: <span className="text-sky-400 font-semibold">{p.local_id}</span></div>
                                    <div className="text-xs text-slate-300">Độ tin cậy YOLO: <span className="text-white font-semibold">{d.confidence != null ? formatConfidence(d.confidence) : 'N/A'}</span></div>
                                    <div className="text-xs text-slate-300">Vị trí: <span className="text-white font-semibold">{translateLocationInPanel(d.location_in_panel)}</span></div>

                                    {d.yolo_display_area_retention != null && d.yolo_display_area_retention < 1.0 && (
                                        <div className="text-xs text-slate-400">Độ giữ diện tích: <span className="text-white font-semibold">{(d.yolo_display_area_retention * 100).toFixed(1)}%</span></div>
                                    )}
                                    <div className="text-xs text-slate-300">Diện tích lỗi: <span className="text-white font-semibold">{d.area_ratio_percent != null ? `${d.area_ratio_percent.toFixed(2)}%` : 'N/A'}</span></div>
                                    {d.relative_thermal_delta != null && (
                                        <div className="text-xs text-orange-300 mt-1 font-semibold">
                                            Tương phản nhiệt: {d.relative_thermal_delta > 0 ? `+${d.relative_thermal_delta.toFixed(3)}` : d.relative_thermal_delta.toFixed(3)}
                                        </div>
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
                    <span>hotspot single cell</span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <div style={{ width: 12, height: 4, borderRadius: 2, backgroundColor: '#ff2d55' }} />
                    <span>hotspot multi_cell</span>
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

                // 2. RGB Crop: Căn giữa theo ảnh nhiệt và áp dụng zoom 1.25x (nới 20% so với 1.5x)
                const cropW_r = cropW_t / 1.25;
                const cropH_r = cropH_t / 1.25;

                const cx_t = (cropX1_t + cropX2_t) / 2;
                const cy_t = (cropY1_t + cropY2_t) / 2;

                const cropX1_r = Math.max(0, cx_t - cropW_r / 2);
                const cropY1_r = Math.max(0, cy_t - cropH_r / 2);
                const cropX2_r = Math.min(activePanel.imgW || 640, cx_t + cropW_r / 2);
                const cropY2_r = Math.min(activePanel.imgH || 512, cy_t + cropH_r / 2);

                const scale_r = Math.min(420 / Math.max(cropW_r, 1), 160 / Math.max(cropH_r, 1));

                // Định nghĩa màu sắc & nhãn tùy theo mức độ nghiêm trọng
                const hasDefect = (activePanel.defects || []).length > 0;
                const severity = hasDefect ? (activePanel.worst_severity || "healthy") : "healthy";
                const isSevere = hasDefect && severity.toLowerCase() === "level_3_priority";
                const isWarning = hasDefect && severity.toLowerCase() === "level_2_inspection";
                const themeColor = hasDefect ? (isSevere ? "#ef4444" : isWarning ? "#f59e0b" : "#3b82f6") : "#22c55e";
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
                            <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                                <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
                                    <div style={{ 
                                        width: 10, height: 10, borderRadius: "50%", 
                                        background: themeColor, boxShadow: glowShadow,
                                        animation: "pulse 1.8s infinite"
                                    }} />
                                    <span style={{ color: "#F8FAFC", fontWeight: 800, fontSize: 18, letterSpacing: "0.5px" }}>
                                        THÔNG TIN TẤM PIN: {activePanel.local_id}
                                    </span>
                                    {activePanel.quality_status === "warning" && (
                                        <span style={{
                                            background: "rgba(245, 158, 11, 0.15)",
                                            color: "#F59E0B",
                                            border: "1px solid rgba(245, 158, 11, 0.3)",
                                            padding: "2px 8px",
                                            borderRadius: 6,
                                            fontSize: 10,
                                            fontWeight: 700,
                                            letterSpacing: "0.5px",
                                            boxShadow: "0 0 10px rgba(245, 158, 11, 0.1)"
                                        }}>
                                            CẢNH BÁO CHẤT LƯỢNG
                                        </span>
                                    )}
                                </div>
                                <div style={{ color: "#94a3b8", fontSize: 12, marginLeft: 20, display: "flex", alignItems: "center", gap: 6 }}>
                                    Ảnh: {activePanel.source_thermal || activePanel.filename || "Không xác định"}
                                    {activePanel.quality_status === "warning" && (
                                        <span style={{ color: "#F59E0B", display: "inline-flex", alignItems: "center", gap: 3 }} title="Ảnh chụp có cảnh báo chất lượng (mờ/nhiễu/tương phản thấp)">
                                            ⚠️ (Ảnh Cảnh Báo)
                                        </span>
                                    )}
                                </div>
                            </div>
                            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
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
                                            
                                            const severity_order = ["healthy", "recheck_required", "level_1_monitoring", "level_2_inspection", "level_3_priority"];
                                            const severityList = Array.from(grouped.severities);
                                            const severity = severityList.length > 0
                                                ? severityList.reduce((worst, current) => {
                                                    return severity_order.indexOf(current) > severity_order.indexOf(worst) ? current : worst;
                                                  }, "healthy")
                                                : "healthy";

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
                                                                const isHighSev = sev === "level_3_priority";
                                                                const isMedSev = sev === "level_2_inspection";
                                                                const sevColor = isHighSev ? "#ef4444" : (isMedSev ? "#f59e0b" : "#3b82f6");
                                                                return (
                                                                    <span 
                                                                        className="px-2 py-1 rounded-md text-xs font-bold" 
                                                                        style={{ 
                                                                            color: sevColor, 
                                                                            backgroundColor: `${sevColor}20`, 
                                                                            border: `1px solid ${sevColor}50` 
                                                                        }}
                                                                    >
                                                                        {translateSeverity(d.severity || "healthy")}
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
                                    {activePanel.defects
                                        .filter(d => d.thermal_validation_status)
                                        .sort((a, b) => {
                                            const vA = a.relative_position?.v ?? 0;
                                            const vB = b.relative_position?.v ?? 0;
                                            
                                            const getZone = (v) => {
                                                if (v < 0.333) return 0;
                                                if (v < 0.666) return 1;
                                                return 2;
                                            };
                                            
                                            const zoneA = getZone(vA);
                                            const zoneB = getZone(vB);
                                            
                                            if (zoneA !== zoneB) return zoneA - zoneB; // Từ trên xuống theo 3 zone
                                            
                                            const uA = a.relative_position?.u ?? 0;
                                            const uB = b.relative_position?.u ?? 0;
                                            return uA - uB; // Từ trái qua phải
                                        })
                                        .map((d, idx) => {
                                        const tvStatus = d.thermal_validation_status;
                                        if (tvStatus === 'not_run') return null;

                                        const getTvMeta = (status, delta) => {
                                            const formatDelta = (val) => {
                                                if (val == null) return "Chưa có dữ liệu";
                                                const num = Number(val);
                                                return num > 0 ? `+${num.toFixed(3)}` : num.toFixed(3);
                                            };
                                            
                                            let deltaText = "Chưa có dữ liệu";
                                            if (delta != null) {
                                                if (delta > 0.05) deltaText = "Vùng lỗi sáng/nóng hơn nền panel";
                                                else if (delta < -0.05) deltaText = "Vùng lỗi tối/lạnh hơn nền panel";
                                                else deltaText = "Tương phản nhiệt tương đối chưa rõ";
                                            }

                                            let label = "Chưa có dữ liệu";
                                            let color = "#94a3b8";
                                            let bg = "rgba(148,163,184,0.08)";
                                            let icon = "❓";

                                            if (status === "hotter_than_panel_background") {
                                                label = "Vùng lỗi nóng hơn nền panel";
                                                color = "#f87171"; bg = "rgba(248,113,113,0.08)"; icon = "🔥";
                                            } else if (status === "weak_relative_contrast") {
                                                label = "Tương phản nhiệt chưa rõ";
                                                color = "#f59e0b"; bg = "rgba(245,158,11,0.08)"; icon = "⚠️";
                                            } else if (status === "thermal_mismatch_need_review") {
                                                label = "Không khớp đặc trưng nhiệt, cần xem xét";
                                                color = "#f97316"; bg = "rgba(249,115,22,0.08)"; icon = "⚠️";
                                            } else if (status === "cooler_than_panel_background_recheck") {
                                                label = "Vùng bị che/tối hơn nền panel, cần chụp lại";
                                                color = "#38bdf8"; bg = "rgba(56,189,248,0.08)"; icon = "❄️";
                                            } else if (status === "shading_need_recheck") {
                                                label = "Che bóng, cần chụp lại";
                                                color = "#818cf8"; bg = "rgba(129,140,248,0.08)"; icon = "☁️";
                                            } else if (status === "shape_based_detection_review") {
                                                label = "Lỗi dựa hình thái, cần duyệt";
                                                color = "#c084fc"; bg = "rgba(192,132,252,0.08)"; icon = "🔍";
                                            } else if (status === "not_run") {
                                                label = "Chưa kiểm chứng";
                                            }

                                            return { label, color, bg, icon, formatDelta, deltaText };
                                        };

                                        const sm = getTvMeta(tvStatus, d.relative_thermal_delta);

                                        let defectLabel = translateDefect(d.class_name);
                                        if (d.class_name && d.class_name.toLowerCase().includes('shad')) {
                                            defectLabel = "Che bóng / Cần chụp lại";
                                        }

                                        return (
                                            <div key={idx} style={{
                                                padding: '12px 14px', backgroundColor: sm.bg,
                                                border: `1px solid ${sm.color}30`, borderRadius: 12
                                            }}>
                                                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 8 }}>
                                                    <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                                                        <span style={{ color: '#cbd5e1', fontSize: 12, fontWeight: 700 }}>Lỗi: {defectLabel}</span>
                                                        <span style={{ color: '#94a3b8', fontSize: 11 }}>Vị trí: {translateLocationInPanel(d.location_in_panel)}</span>
                                                    </div>
                                                    <span style={{ color: sm.color, fontSize: 11, fontWeight: 700, display: 'flex', alignItems: 'center', gap: 4, textAlign: 'right' }}>
                                                        {sm.icon} {sm.label}
                                                    </span>
                                                </div>

                                                <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1fr', gap: '4px 12px', fontSize: 11 }}>
                                                    <div style={{ color: '#64748b' }}>Tương phản nhiệt tương đối</div>
                                                    <div style={{ color: sm.color, fontWeight: 700 }}>
                                                        {sm.formatDelta(d.relative_thermal_delta)}
                                                    </div>

                                                    <div style={{ color: '#64748b' }}>Kết luận</div>
                                                    <div style={{ color: '#cbd5e1', fontWeight: 600 }}>
                                                        {sm.deltaText}
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
                                    { key: "hotspot_single", label: "hotspot single cell", color: "#ff3b30" },
                                    { key: "hotspot_multi", label: "hotspot multi_cell", color: "#ff2d55" },
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
