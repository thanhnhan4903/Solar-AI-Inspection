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
    "Healthy": "#10b981",    // Green
    "Hotspot": "#ef4444",    // Red
    "Crack": "#8b5cf6",      // Purple
    "Soiling": "#f59e0b"     // Orange
};

const IMAGE_BASE_URL = "http://127.0.0.1:8000/data/raw/";

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

export default function UnifiedDashboard({ data, focusTarget }) {
    const [viewMode, setViewMode] = useState('monitor'); // 'monitor' or 'mission'
    const [searchQuery, setSearchQuery] = useState('');
    const [statusFilter, setStatusFilter] = useState('All');
    const [showHeatmap, setShowHeatmap] = useState(true);
    
    // Hover State for Side-by-Side viewer
    const [hoveredPanel, setHoveredPanel] = useState(null);

    // Calculate Grid Mapping
    const gridData = useMemo(() => {
        if (!data || data.length === 0) return [];
        const IMAGES_PER_ROW = 4;
        const PADDING = 100; // pixels between images

        return data.map((img, index) => {
            const col = index % IMAGES_PER_ROW;
            const row = Math.floor(index / IMAGES_PER_ROW);
            
            const imgW = img.image_width || 640;
            const imgH = img.image_height || 512;
            
            const xOffset = col * (imgW + PADDING);
            const yOffset = - (row * (imgH + PADDING)); // negative down
            
            const bounds = [[yOffset - imgH, xOffset], [yOffset, xOffset + imgW]];
            
            const mappedPanels = img.panels.map(p => {
                const [x1, y1, x2, y2] = p.box;
                const map_x1 = xOffset + x1;
                const map_y1 = yOffset - y1; 
                const map_x2 = xOffset + x2;
                const map_y2 = yOffset - y2; 
                
                const status = p.total_panel_loss > 0 ? (p.defects.length > 0 ? p.defects[0].type : "Hotspot") : "Healthy";
                
                return {
                    ...p,
                    source_rgb: img.rgb_image,
                    source_thermal: img.filename,
                    imgW, imgH,
                    boxW: x2 - x1,
                    boxH: y2 - y1,
                    status,
                    polygon: [[map_y1, map_x1], [map_y1, map_x2], [map_y2, map_x2], [map_y2, map_x1]]
                };
            });
            
            return { ...img, bounds, mappedPanels };
        });
    }, [data]);

    const allPanels = useMemo(() => gridData.flatMap(img => img.mappedPanels), [gridData]);

    const filteredPanels = useMemo(() => {
        return allPanels.filter(panel => {
            const matchesSearch = panel.local_id.toLowerCase().includes(searchQuery.toLowerCase());
            let panelCat = panel.status.toLowerCase().includes('hotspot') ? 'Hotspot' : 
                           (panel.status.toLowerCase().includes('crack') ? 'Crack' : 
                           (panel.status.toLowerCase().includes('soil') ? 'Soiling' : 'Healthy'));
            const matchesStatus = statusFilter === 'All' || panelCat === statusFilter;
            return matchesSearch && matchesStatus;
        });
    }, [allPanels, searchQuery, statusFilter]);

    const stats = useMemo(() => {
        const counts = { Total: allPanels.length, Healthy: 0, Issues: 0 };
        allPanels.forEach(p => {
            if (p.status === "Healthy") counts.Healthy++;
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
                style={{ width: '100%', height: '100%', background: '#030712' }} // Dark cyber background
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
                                {/* Use simple custom logic to draw a border since react-leaflet Rectangle might need separate import */}
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
                    let color = STATUS_COLORS["Healthy"];
                    if (p.status.toLowerCase().includes("hotspot")) color = STATUS_COLORS["Hotspot"];
                    else if (p.status.toLowerCase().includes("crack")) color = STATUS_COLORS["Crack"];
                    else if (p.status.toLowerCase().includes("soil")) color = STATUS_COLORS["Soiling"];

                    return (
                        <Polygon
                            key={i}
                            positions={p.polygon}
                            pathOptions={{
                                color: color,
                                fillColor: color,
                                fillOpacity: p.status === "Healthy" ? 0.1 : 0.4,
                                weight: 2
                            }}
                            eventHandlers={{
                                mouseover: () => setHoveredPanel(p),
                                mouseout: () => setHoveredPanel(null)
                            }}
                        >
                            <Tooltip sticky className="bg-slate-900 border-none text-white shadow-xl rounded-lg">
                                <div className="text-sm font-bold text-sky-400">{p.local_id}</div>
                                <div className="text-xs">{p.status} {p.status !== "Healthy" && `(${p.total_panel_loss.toFixed(1)}%)`}</div>
                            </Tooltip>
                        </Polygon>
                    );
                })}
            </MapContainer>

            {/* Hover Side-by-Side Viewer */}
            {hoveredPanel && hoveredPanel.status !== "Healthy" && (
                <div style={{
                    position: "absolute", bottom: 40, right: 40, zIndex: 1000,
                    background: "rgba(15, 23, 42, 0.9)", backdropFilter: "blur(20px)",
                    borderRadius: 16, padding: 20, border: "1px solid rgba(14, 165, 233, 0.3)",
                    boxShadow: "0 20px 40px rgba(0,0,0,0.5)", width: 500, pointerEvents: "none"
                }}>
                    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 12 }}>
                        <span style={{ color: "#F8FAFC", fontWeight: 700, fontSize: 16 }}>{hoveredPanel.local_id}</span>
                        <span style={{ color: "#EF4444", fontWeight: 700, fontSize: 14 }}>{hoveredPanel.status}</span>
                    </div>
                    
                    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
                        {/* Thermal Crop */}
                        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                            <span style={{ color: "#94A3B8", fontSize: 12, fontWeight: 600 }}>Ảnh Nhiệt (Thermal)</span>
                            <div style={{ 
                                width: "100%", height: 180, background: "#000", borderRadius: 8, overflow: "hidden", position: "relative",
                                display: "flex", alignItems: "center", justifyContent: "center"
                            }}>
                                <div style={{
                                    width: hoveredPanel.boxW, height: hoveredPanel.boxH,
                                    overflow: "hidden", position: "relative",
                                    transform: `scale(${Math.min(220/hoveredPanel.boxW, 180/hoveredPanel.boxH)})`,
                                    transformOrigin: "center center"
                                }}>
                                    <img 
                                        src={`${IMAGE_BASE_URL}${hoveredPanel.source_thermal}`} 
                                        style={{ position: "absolute", left: -hoveredPanel.box[0], top: -hoveredPanel.box[1], maxWidth: "none" }}
                                    />
                                    <div style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%", border: "2px solid #EF4444" }} />
                                </div>
                            </div>
                        </div>

                        {/* RGB Crop */}
                        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                            <span style={{ color: "#94A3B8", fontSize: 12, fontWeight: 600 }}>Ảnh Quang (RGB)</span>
                            <div style={{ 
                                width: "100%", height: 180, background: "#000", borderRadius: 8, overflow: "hidden", position: "relative",
                                display: "flex", alignItems: "center", justifyContent: "center"
                            }}>
                                {hoveredPanel.source_rgb ? (
                                    <div style={{
                                        width: hoveredPanel.boxW, height: hoveredPanel.boxH,
                                        overflow: "hidden", position: "relative",
                                        transform: `scale(${Math.min(220/hoveredPanel.boxW, 180/hoveredPanel.boxH)})`,
                                        transformOrigin: "center center"
                                    }}>
                                        <img 
                                            src={`${IMAGE_BASE_URL}${hoveredPanel.source_rgb}`} 
                                            style={{ position: "absolute", left: -hoveredPanel.box[0], top: -hoveredPanel.box[1], maxWidth: "none" }}
                                        />
                                        <div style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%", border: "2px solid #0EA5E9" }} />
                                    </div>
                                ) : (
                                    <span style={{ color: "#64748B", fontSize: 12 }}>Không tìm thấy ảnh RGB</span>
                                )}
                            </div>
                        </div>
                    </div>
                </div>
            )}

            {/* Left Sidebar - GIS Controls */}
            {viewMode === 'monitor' && (
                <div className="unified-overlay sidebar-left glass-panel" style={{ zIndex: 1000 }}>
                    <h2 className="text-xl font-bold mb-6 flex items-center gap-2 text-white">
                        <MapIcon className="text-sky-400" size={24} />
                        Solar Farm Virtual Map
                    </h2>

                    <div className="relative mb-6">
                        <Search className="absolute left-3 top-2.5 text-slate-500" size={16} />
                        <input 
                            type="text" 
                            placeholder="Tìm ID (VD: Panel_01)..." 
                            className="w-full bg-slate-800/50 border border-slate-700 rounded-lg py-2 pl-10 pr-4 text-sm focus:outline-none focus:ring-2 focus:ring-sky-500 text-white"
                            value={searchQuery}
                            onChange={(e) => setSearchQuery(e.target.value)}
                        />
                    </div>

                    <div className="space-y-4">
                        <div>
                            <label className="text-[10px] text-slate-400 uppercase font-bold mb-2 block">Lọc trạng thái</label>
                            <div className="grid grid-cols-2 gap-2">
                                <button onClick={() => setStatusFilter('All')} className={`text-xs p-2 rounded-lg border ${statusFilter === 'All' ? 'bg-sky-600 border-sky-500 text-white' : 'bg-slate-800/50 border-slate-700 text-slate-400'}`}>Tất cả</button>
                                {["Healthy", "Hotspot", "Crack", "Soiling"].map(s => (
                                    <button key={s} onClick={() => setStatusFilter(s)} className={`text-xs p-2 rounded-lg border flex items-center gap-2 ${statusFilter === s ? 'bg-sky-600 border-sky-500 text-white' : 'bg-slate-800/50 border-slate-700 text-slate-400'}`}>
                                        <div className="w-2 h-2 rounded-full" style={{ background: STATUS_COLORS[s] || "#ccc" }} /> {s}
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
                                <span className="text-sm font-bold">Hiển thị lớp ảnh Nhiệt</span>
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
                                <div className="text-[10px] text-slate-400 uppercase">Lỗi phát hiện</div>
                                <div className="text-xl font-bold text-rose-400">{stats.Issues}</div>
                            </div>
                        </div>
                    </div>
                    
                    {!data || data.length === 0 ? (
                        <div style={{ marginTop: 20, padding: 16, background: "rgba(239, 68, 68, 0.1)", border: "1px solid rgba(239, 68, 68, 0.3)", borderRadius: 12, color: "#FCA5A5", fontSize: 13, textAlign: "center" }}>
                            Chưa có dữ liệu. Vui lòng tải dữ liệu và phân tích ở màn hình Dashboard trước!
                        </div>
                    ) : null}
                </div>
            )}
            
            {/* Top right floating info */}
            <div style={{ position: "absolute", top: 24, right: 24, zIndex: 1000, display: "flex", gap: 12 }}>
                <div style={{ background: "rgba(15, 23, 42, 0.8)", backdropFilter: "blur(12px)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 12, padding: "8px 16px", display: "flex", alignItems: "center", gap: 8, color: "#fff", fontSize: 13, fontWeight: 600 }}>
                    <Maximize2 size={16} color="#0EA5E9" /> Chế độ Virtual Map
                </div>
            </div>
        </div>
    );
}
