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
    Battery, Wifi, Compass, Navigation, Activity, Search, Thermometer, Map as MapIcon, X, Maximize2
} from 'lucide-react';

import './UnifiedStyles.css';

const STATUS_COLORS = {
    "healthy": "#10b981",         // Green
    "faulty": "#ef4444",         // Red
    // Legacy keys (từ code cũ)
    "Healthy": "#10b981",
    "Hotspot (Single)": "#ff0000ff",
    "Hotspot (Multi)": "#ff0000ff",
    "Crack": "#8b5cf6",
    "Soiling": "#f59e0b",
};

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
function pixelPolyToLeaflet(polygon, xOffset, yOffset) {
    if (!polygon) return [];
    return polygon.map(([px, py]) => {
        const x = Number(px) || 0;
        const y = Number(py) || 0;
        return [yOffset - y, xOffset + x];
    });
}

export default function UnifiedDashboard({ data, panelPower = 600, focusTarget }) {
    const PANEL_RATED_POWER_W = 400; // Công suất định mức tấm pin 400W
    const translateDefect = (cls) => {
        if (!cls) return "Điểm bất thường";
        const c = cls.toLowerCase();
        if (c.includes("hotspot_single_cell")) return "Hotspot (Đơn cell)";
        if (c.includes("hotspot_multi_cell")) return "Hotspot (Đa cell)";
        if (c.includes("hotspot")) return "Điểm nóng (Hotspot)";
        if (c.includes("crack")) return "Vết nứt (Crack)";
        if (c.includes("soil")) return "Bám bẩn (Soiling)";
        return cls;
    };
    const [viewMode, setViewMode] = useState('monitor');
    const [searchQuery, setSearchQuery] = useState('');
    const [statusFilter, setStatusFilter] = useState('All');
    const [showHeatmap, setShowHeatmap] = useState(true);
    const [hoveredPanel, setHoveredPanel] = useState(null);
    const [clickedPanel, setClickedPanel] = useState(null);

    // Calculate Grid Mapping
    const gridData = useMemo(() => {
        if (!data || data.length === 0) return [];
        const IMAGES_PER_ROW = 4;
        const PADDING = 100;

        return data.map((img, index) => {
            const col = index % IMAGES_PER_ROW;
            const row = Math.floor(index / IMAGES_PER_ROW);

            const imgW = img.image_width || 640;
            const imgH = img.image_height || 512;

            const xOffset = col * (imgW + PADDING);
            const yOffset = -(row * (imgH + PADDING));

            const bounds = [[yOffset - imgH, xOffset], [yOffset, xOffset + imgW]];

            const mappedPanels = img.panels.map(p => {
                // ✅ Ưu tiên dùng polygon thật từ backend (đã refine bằng minAreaRect)
                // Fallback về bbox chỉ khi không có polygon
                let leafletPolygon;
                if (p.polygon && p.polygon.length >= 3) {
                    leafletPolygon = pixelPolyToLeaflet(p.polygon, xOffset, yOffset);
                } else {
                    // Fallback: tạo polygon từ bbox với phòng vệ tối đa
                    const bbox = p.bbox && p.bbox.length === 4 ? p.bbox : (p.box && p.box.length === 4 ? p.box : [0, 0, 0, 0]);
                    const bx1 = Number(bbox[0]) || 0;
                    const by1 = Number(bbox[1]) || 0;
                    const bx2 = Number(bbox[2]) || 0;
                    const by2 = Number(bbox[3]) || 0;
                    leafletPolygon = [
                        [yOffset - by1, xOffset + bx1],
                        [yOffset - by1, xOffset + bx2],
                        [yOffset - by2, xOffset + bx2],
                        [yOffset - by2, xOffset + bx1],
                    ];
                }

                // Status: dùng field mới 'status' (faulty/healthy), fallback sang cũ
                const status = p.status || (p.total_panel_loss > 0 ? "faulty" : "healthy");
                const bbox = p.bbox || p.box || [0, 0, 0, 0];
                const boxW = bbox[2] - bbox[0];
                const boxH = bbox[3] - bbox[1];

                return {
                    ...p,
                    source_rgb: img.rgb_image,
                    source_thermal: img.filename,
                    imgW, imgH,
                    boxW,
                    boxH,
                    status,
                    polygon: leafletPolygon,
                    // Giữ bbox gốc pixel để dùng cho hover crop
                    bbox_px: bbox,
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

            // Phân loại theo status mới
            let panelCat = "Healthy";
            if (panel.status === "faulty") {
                // Phân loại chi tiết theo defect chính
                const mainClass = panel.main_defect_class || "";
                if (mainClass.includes("hotspot_single_cell")) panelCat = "Hotspot (Single)";
                else if (mainClass.includes("hotspot_multi_cell")) panelCat = "Hotspot (Multi)";
                else if (mainClass.includes("hotspot")) panelCat = "Hotspot (Single)";
                else if (mainClass.includes("crack")) panelCat = "Crack";
                else if (mainClass.includes("soil")) panelCat = "Soiling";
                else panelCat = "Hotspot (Single)"; // default faulty
            }

            const matchesStatus = statusFilter === 'All' ||
                (statusFilter === 'Healthy' && panelCat === 'Healthy') ||
                (statusFilter !== 'Healthy' && panelCat === statusFilter);
            return matchesSearch && matchesStatus;
        });
    }, [allPanels, searchQuery, statusFilter]);

    const stats = useMemo(() => {
        const counts = { Total: allPanels.length, Healthy: 0, Issues: 0 };
        allPanels.forEach(p => {
            if (p.status === "healthy" || p.status === "Healthy") counts.Healthy++;
            else counts.Issues++;
        });
        return counts;
    }, [allPanels]);

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
                    const isHealthy = p.status === "healthy" || p.status === "Healthy";
                    let color = STATUS_COLORS["healthy"];
                    if (!isHealthy) {
                        const mainClass = p.main_defect_class || "";
                        if (mainClass.includes("crack")) color = STATUS_COLORS["Crack"];
                        else if (mainClass.includes("soil")) color = STATUS_COLORS["Soiling"];
                        else color = STATUS_COLORS["faulty"];
                    }

                    return (
                        <Polygon
                            key={i}
                            positions={p.polygon}
                            pathOptions={{
                                color: color,
                                fillColor: color,
                                fillOpacity: isHealthy ? 0.25 : 0.45,
                                weight: isHealthy ? 2 : 3
                            }}
                            eventHandlers={{
                                mouseover: () => setHoveredPanel(p),
                                mouseout: () => setHoveredPanel(null),
                                click: () => {
                                    if (!isHealthy) {
                                        setClickedPanel(p);
                                    }
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
                            </Tooltip>
                        </Polygon>
                    );
                })}
            </MapContainer>

            {/* Hover/Click Control Drawer */}
            {(() => {
                const activePanel = clickedPanel || hoveredPanel;
                if (!activePanel || (activePanel.status !== "faulty" && activePanel.status !== "Healthy" && activePanel.status !== "healthy")) return null;

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
                const severity = activePanel.worst_severity || "Lỗi";
                const isSevere = severity.toLowerCase() === "severe" || severity.toLowerCase() === "high";
                const themeColor = isSevere ? "#ef4444" : "#f59e0b";
                const glowShadow = isSevere 
                    ? "0 0 15px rgba(239, 68, 68, 0.4)" 
                    : "0 0 15px rgba(245, 158, 11, 0.4)";

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
                        {/* Header của Control Panel */}
                        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", borderBottom: "1px solid rgba(255,255,255,0.08)", paddingBottom: 16, marginBottom: 18 }}>
                            <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                                <div style={{ 
                                    width: 10, height: 10, borderRadius: "50%", 
                                    background: themeColor, boxShadow: glowShadow,
                                    animation: "pulse 1.8s infinite"
                                }} />
                                <span style={{ color: "#F8FAFC", fontWeight: 800, fontSize: 18, letterSpacing: "0.5px" }}>
                                    PANEL DETECTOR: {activePanel.local_id}
                                </span>
                            </div>
                            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                                <span style={{ 
                                    color: "#FFFFFF", background: themeColor, 
                                    fontWeight: 700, fontSize: 11, padding: "4px 10px", 
                                    borderRadius: 6, textTransform: "uppercase",
                                    boxShadow: glowShadow, letterSpacing: "0.5px"
                                }}>
                                    {severity}
                                </span>
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
                                            borderRadius: 6,
                                            padding: 4,
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
                        </div>

                        {/* Lượng thất thoát sản lượng dạng SCADA Banner */}
                        <div style={{
                            background: "linear-gradient(90deg, rgba(239, 68, 68, 0.12), rgba(15, 23, 42, 0.2))",
                            border: `1px solid rgba(${isSevere ? "239, 68, 68" : "245, 158, 11"}, 0.3)`,
                            borderRadius: 12, padding: "16px 20px", marginBottom: 20,
                            boxShadow: `inset 0 0 15px rgba(${isSevere ? "239, 68, 68" : "245, 158, 11"}, 0.05)`,
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
                            <span style={{ color: "#64748b", fontSize: 10, fontStyle: "italic", marginTop: 2 }}>
                                * Dựa trên thuật toán phân tích 3 phần (3-part division logic) ở defect_logic.py
                            </span>
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
                                                };
                                            }
                                            groupedMap[cls].area_ratio_percent += d.area_ratio_percent || d.loss_pct || 0;
                                            if (d.location_in_panel) {
                                                groupedMap[cls].locations.add(d.location_in_panel);
                                            }
                                            if (d.severity) {
                                                groupedMap[cls].severities.add(d.severity);
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

                                            return {
                                                class_name: grouped.class_name,
                                                area_ratio_percent: grouped.area_ratio_percent,
                                                location_in_panel,
                                                severity,
                                            };
                                        });

                                        return groupedList.map((d, i) => {
                                            const ratio = d.area_ratio_percent;
                                            return (
                                                <div key={i} style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.05)", borderRadius: 10, padding: 10 }}>
                                                    <div style={{ display: "flex", justifyContent: "space-between", fontSize: 13, fontWeight: 600, marginBottom: 4 }}>
                                                        <span style={{ color: "#EF4444" }}>{translateDefect(d.class_name)}</span>
                                                        <span style={{ color: "#E2E8F0" }}>Diện tích: {ratio.toFixed(2)}% tấm pin</span>
                                                    </div>
                                                    <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, color: "#94A3B8" }}>
                                                        <span>Vị trí: {d.location_in_panel}</span>
                                                        <span>Mức độ: {d.severity || "Trung bình"}</span>
                                                    </div>
                                                    <div style={{ width: "100%", height: 6, background: "rgba(255,255,255,0.06)", borderRadius: 3, marginTop: 8, overflow: "hidden" }}>
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
                                <Battery size={14} color="#0EA5E9" /> Thông số kỹ thuật phân tích
                            </h4>
                            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.04)", borderRadius: 10, padding: 10 }}>
                                <div style={{ fontSize: 11 }}>
                                    <div style={{ color: "#64748B" }}>Tọa độ trung tâm</div>
                                    <div style={{ color: "#E2E8F0", fontWeight: 600, marginTop: 2 }}>
                                        X: {Math.round((x1 + x2) / 2)}, Y: {Math.round((y1 + y2) / 2)}
                                    </div>
                                </div>
                                <div style={{ fontSize: 11 }}>
                                    <div style={{ color: "#64748B" }}>Độ tin cậy AI</div>
                                    <div style={{ color: "#38bdf8", fontWeight: 600, marginTop: 2 }}>
                                        {((activePanel.confidence || 0.85) * 100).toFixed(1)}% (SegYOLO)
                                    </div>
                                </div>
                                <div style={{ fontSize: 11 }}>
                                    <div style={{ color: "#64748B" }}>Kích thước nguồn</div>
                                    <div style={{ color: "#E2E8F0", fontWeight: 600, marginTop: 2 }}>
                                        {activePanel.imgW} x {activePanel.imgH} px
                                    </div>
                                </div>
                                <div style={{ fontSize: 11 }}>
                                    <div style={{ color: "#64748B" }}>Loại thiết bị</div>
                                    <div style={{ color: "#E2E8F0", fontWeight: 600, marginTop: 2 }}>
                                        Solar Panel (Monocrystal)
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                );
            })()}

            {/* Left Sidebar */}
            {viewMode === 'monitor' && (
                <div className="unified-overlay sidebar-left glass-panel" style={{ zIndex: 1000 }}>
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
                            <label className="text-[10px] text-slate-400 uppercase font-bold mb-2 block">Lọc theo trạng thái</label>
                            <div className="grid grid-cols-2 gap-2">
                                <button onClick={() => setStatusFilter('All')} className={`text-xs p-2 rounded-lg border ${statusFilter === 'All' ? 'bg-sky-600 border-sky-500 text-white' : 'bg-slate-800/50 border-slate-700 text-slate-400'}`}>Tất cả</button>
                                {["Healthy", "Hotspot (Single)", "Hotspot (Multi)", "Crack", "Soiling"].map(s => (
                                    <button key={s} onClick={() => setStatusFilter(s)} className={`text-xs p-2 rounded-lg border flex items-center gap-2 ${statusFilter === s ? 'bg-sky-600 border-sky-500 text-white' : 'bg-slate-800/50 border-slate-700 text-slate-400'}`}>
                                        <div className="w-2 h-2 rounded-full flex-shrink-0" style={{ background: STATUS_COLORS[s] || "#ccc" }} />
                                        <span className="truncate">{s === 'Healthy' ? 'Bình thường' : s === 'Hotspot (Single)' ? 'Hotspot (Đơn)' : s === 'Hotspot (Multi)' ? 'Hotspot (Đa)' : s === 'Crack' ? 'Nứt (Crack)' : 'Bám bẩn'}</span>
                                    </button>
                                ))}
                            </div>
                        </div>

                        <button
                            onClick={() => setShowHeatmap(!showHeatmap)}
                            className={`w-full flex items-center justify-between p-3 rounded-lg border transition ${showHeatmap ? 'bg-orange-600/20 border-orange-500/50 text-orange-400' : 'bg-slate-800/50 border-slate-700 text-slate-400'}`}
                        >
                            <div className="flex items-center gap-3">
                                <Thermometer size={18} />
                                <span className="text-sm font-bold">Hiển thị lớp phủ nhiệt</span>
                            </div>
                            <div className={`w-8 h-4 rounded-full relative ${showHeatmap ? 'bg-orange-500' : 'bg-slate-600'}`}>
                                <div className={`absolute top-0.5 w-3 h-3 bg-white rounded-full transition-all ${showHeatmap ? 'right-0.5' : 'left-0.5'}`} />
                            </div>
                        </button>
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
            <div style={{ position: "absolute", top: 24, right: 24, zIndex: 1000, display: "flex", gap: 12 }}>
                <div style={{ background: "rgba(15, 23, 42, 0.8)", backdropFilter: "blur(12px)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 12, padding: "8px 16px", display: "flex", alignItems: "center", gap: 8, color: "#fff", fontSize: 13, fontWeight: 600 }}>
                    <Maximize2 size={16} color="#0EA5E9" /> Bản đồ toàn cảnh
                </div>
            </div>
        </div>
    );
}
