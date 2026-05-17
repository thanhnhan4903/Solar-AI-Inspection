import React, { useState } from "react";
import { ArrowLeft, ArrowRight, MapPin, Activity, AlertTriangle, Percent, LayoutGrid, Info, Map as MapIcon, Target } from "lucide-react";
import { colors } from "../../constants/theme";

const IMAGE_BASE_API = "http://127.0.0.1:8000/data/precalib/";

const SEVERITY_COLORS = {
    "very_minor": "#94a3b8",
    "minor": "#f59e0b",
    "moderate": "#f97316",
    "severe": "#ef4444",
    "replace": "#7c3aed",
};

const DEFECT_STROKE_COLORS = {
    "hotspot_single_cell": "#f97316",
    "hotspot_multi_cell": "#ef4444",
    "shading": "#06b6d4",
    "soiling": "#eab308",
    "crack": "#a855f7",
};

// Lấy tên file không có extension
function getImageDisplayName(filename) {
    if (!filename) return "";
    return filename.replace(/\.[^/.]+$/, "");
}

// Lấy số thứ tự panel từ local_id (R02_C03 → số index)
function getPanelIndex(panels, localId) {
    if (!panels) return null;
    const idx = panels.findIndex(p => p.local_id === localId);
    return idx >= 0 ? idx + 1 : null;
}

export default function PanelDetail({ panel: image, data, onSelect, onBack, onViewOnMap }) {
    if (!image) return null;

    const [hoveredPanel, setHoveredPanel] = useState(null);
    const [hoveredDefect, setHoveredDefect] = useState(null);

    const imgW = image.image_width || 640;
    const imgH = image.image_height || 512;

    const displayName = getImageDisplayName(image.filename);

    const handleNext = () => {
        if (!data || data.length === 0) return;
        const currentIndex = data.findIndex(img => img.filename === image.filename);
        if (currentIndex !== -1) {
            const nextIndex = (currentIndex + 1) % data.length;
            onSelect({
                ...data[nextIndex],
                id: `Image ${nextIndex + 1}`,
                status: data[nextIndex].panels.filter(p => (p.status === "faulty" || p.total_panel_loss > 0)).length > 0 ? "defective" : "healthy",
                faulty_count: data[nextIndex].panels.filter(p => (p.status === "faulty" || p.total_panel_loss > 0)).length
            });
        }
    };

    const handlePrev = () => {
        if (!data || data.length === 0) return;
        const currentIndex = data.findIndex(img => img.filename === image.filename);
        if (currentIndex !== -1) {
            const prevIndex = (currentIndex - 1 + data.length) % data.length;
            onSelect({
                ...data[prevIndex],
                id: `Image ${prevIndex + 1}`,
                status: data[prevIndex].panels.filter(p => (p.status === "faulty" || p.total_panel_loss > 0)).length > 0 ? "defective" : "healthy",
                faulty_count: data[prevIndex].panels.filter(p => (p.status === "faulty" || p.total_panel_loss > 0)).length
            });
        }
    };

    const isPanelFaulty = (p) => p.status === "faulty" || p.total_panel_loss > 0;

    const panelIndex = hoveredPanel ? getPanelIndex(image.panels, hoveredPanel.local_id) : null;

    return (
        <div style={{ background: "#fff", borderRadius: 20, padding: 24, display: "flex", flexDirection: "column", height: "100%" }}>
            {/* Header */}
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
                <button onClick={onBack} style={{ display: "flex", alignItems: "center", gap: 8, border: "none", background: "none", color: colors.primary, cursor: "pointer", fontWeight: 600 }}>
                    <ArrowLeft size={18} /> Back
                </button>
                <div style={{ display: "flex", gap: 24 }}>
                    <button onClick={handlePrev} style={{ display: "flex", alignItems: "center", border: "none", background: "none", color: "#1E293B", cursor: "pointer", padding: 0 }}>
                        <ArrowLeft size={28} />
                    </button>
                    <button onClick={handleNext} style={{ display: "flex", alignItems: "center", border: "none", background: "none", color: "#1E293B", cursor: "pointer", padding: 0 }}>
                        <ArrowRight size={28} />
                    </button>
                </div>
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "1.2fr 1fr", gap: 32, flex: 1 }}>
                {/* TRÁI */}
                <div>
                    {/* Tên tấm hình */}
                    <div style={{
                        display: "inline-block",
                        border: "2px solid #1E293B",
                        borderRadius: 6,
                        padding: "4px 16px",
                        marginBottom: 12,
                        fontWeight: 700,
                        fontSize: 15,
                        color: "#1E293B",
                        letterSpacing: 0.5,
                    }}>
                        {image.id} – {displayName}
                    </div>

                    <div style={{ borderRadius: 12, overflow: "hidden", background: "#000", position: "relative", minHeight: 400 }}>
                        <img
                            src={`${IMAGE_BASE_API}${image.filename}`}
                            style={{ width: "100%", display: "block", objectFit: "contain" }}
                            alt={image.id}
                        />
                        <svg
                            style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%" }}
                            viewBox={`0 0 ${imgW} ${imgH}`}
                            preserveAspectRatio="xMidYMid meet"
                        >
                            {image.panels && image.panels.map((p, i) => {
                                const isHovered = hoveredPanel?.local_id === p.local_id;
                                const faulty = isPanelFaulty(p);
                                const strokeColor = faulty ? "#EF4444" : "#10B981";
                                const fillColor = isHovered
                                    ? (faulty ? "rgba(239,68,68,0.35)" : "rgba(16,185,129,0.35)")
                                    : (faulty ? "rgba(239,68,68,0.12)" : "transparent");

                                return (
                                    <g key={i}>
                                        {p.polygon && p.polygon.length >= 3 ? (
                                            <polygon
                                                points={p.polygon.map(pt => Array.isArray(pt) ? pt.join(',') : pt).join(' ')}
                                                fill={fillColor}
                                                stroke={strokeColor}
                                                strokeWidth={isHovered ? 2.5 : 1.5}
                                                onMouseEnter={() => { setHoveredPanel(p); setHoveredDefect(null); }}
                                                onMouseLeave={() => { setHoveredPanel(null); setHoveredDefect(null); }}
                                                style={{ cursor: "pointer", transition: "all 0.2s" }}
                                            />
                                        ) : (
                                            p.bbox && (
                                                <rect
                                                    x={(p.bbox || p.box)[0]} y={(p.bbox || p.box)[1]}
                                                    width={(p.bbox || p.box)[2] - (p.bbox || p.box)[0]}
                                                    height={(p.bbox || p.box)[3] - (p.bbox || p.box)[1]}
                                                    fill={fillColor}
                                                    stroke={strokeColor}
                                                    strokeWidth={isHovered ? 2.5 : 1.5}
                                                    onMouseEnter={() => { setHoveredPanel(p); setHoveredDefect(null); }}
                                                    onMouseLeave={() => { setHoveredPanel(null); setHoveredDefect(null); }}
                                                    style={{ cursor: "pointer", transition: "all 0.2s" }}
                                                />
                                            )
                                        )}

                                        {p.center && (
                                            <text
                                                x={p.center[0]}
                                                y={p.center[1]}
                                                textAnchor="middle"
                                                dominantBaseline="middle"
                                                fontSize={10}
                                                fill={strokeColor}
                                                fontWeight="bold"
                                                style={{ pointerEvents: "none", userSelect: "none" }}
                                            >
                                                {p.local_id || ""}
                                            </text>
                                        )}

                                        {faulty && p.defects && p.defects.map((d, di) => {
                                            const defectStroke = DEFECT_STROKE_COLORS[d.class_name] || "#f97316";
                                            const isDefectHovered = hoveredDefect?.class_name === d.class_name
                                                && hoveredDefect?.location_in_panel === d.location_in_panel
                                                && hoveredPanel?.local_id === p.local_id;

                                            return d.polygon && d.polygon.length >= 3 ? (
                                                <polygon
                                                    key={di}
                                                    points={d.polygon.map(pt => Array.isArray(pt) ? pt.join(',') : pt).join(' ')}
                                                    fill={`${defectStroke}55`}
                                                    stroke={defectStroke}
                                                    strokeWidth={isDefectHovered ? 2.5 : 1.5}
                                                    strokeDasharray={isDefectHovered ? "none" : "4 2"}
                                                    onMouseEnter={() => { setHoveredPanel(p); setHoveredDefect(d); }}
                                                    onMouseLeave={() => { setHoveredPanel(null); setHoveredDefect(null); }}
                                                    style={{ cursor: "crosshair", transition: "all 0.2s" }}
                                                />
                                            ) : null;
                                        })}
                                    </g>
                                );
                            })}
                        </svg>
                    </div>
                    <p style={{ fontSize: 13, color: "#64748b", marginTop: 12, display: "flex", gap: 6, alignItems: "center" }}>
                        <Info size={14} /> <i>Hover over a panel rectangle or polygon to view detailed diagnostics.</i>
                    </p>
                </div>

                {/* PHẢI */}
                <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
                    <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: 16 }}>
                        {!hoveredPanel ? (
                            // MẶC ĐỊNH
                            <>
                                <div style={{ padding: 20, background: "#f8fafc", borderRadius: 16, border: "1px solid #e2e8f0" }}>
                                    <h3 style={{ margin: "0 0 16px 0", color: "#1E293B" }}>Overview {image.id}</h3>
                                    <div style={{ display: "flex", gap: 20 }}>
                                        <div style={{ flex: 1 }}>
                                            <p style={{ margin: "0 0 8px 0", color: "#64748B", fontSize: 13, fontWeight: 600 }}>TOTAL PANELS</p>
                                            <p style={{ margin: 0, fontSize: 24, fontWeight: "bold", color: colors.primary }}>{image.total_panels}</p>
                                        </div>
                                        <div style={{ flex: 1 }}>
                                            <p style={{ margin: "0 0 8px 0", color: "#64748B", fontSize: 13, fontWeight: 600 }}>FAULTY PANELS</p>
                                            <p style={{ margin: 0, fontSize: 24, fontWeight: "bold", color: image.faulty_count > 0 ? colors.danger : colors.success }}>{image.faulty_count}</p>
                                        </div>
                                    </div>
                                </div>
                                {image.faulty_count > 0 && (
                                    <div style={{ padding: 20, background: "#FFF5F5", borderRadius: 16, border: "1px solid #FED7D7" }}>
                                        <h4 style={{ margin: "0 0 10px 0", display: "flex", alignItems: "center", gap: 8, color: colors.error }}>
                                            <AlertTriangle size={18} /> Warning
                                        </h4>
                                        <p style={{ margin: 0, color: "#9B2C2C", fontSize: 14 }}>
                                            Detected <b>{image.faulty_count}</b> faulty panels. Hover over red bounding boxes to investigate.
                                        </p>
                                    </div>
                                )}
                            </>
                        ) : hoveredDefect ? (
                            // HOVER DEFECT
                            <>
                                <div style={{ display: "flex", alignItems: "center", gap: 12, paddingBottom: 16, borderBottom: "1px solid #e2e8f0" }}>
                                    <Target size={24} color={DEFECT_STROKE_COLORS[hoveredDefect.class_name] || "#f97316"} />
                                    <div>
                                        <h3 style={{ margin: 0, color: "#1E293B", fontSize: 15 }}>
                                            {hoveredDefect.class_name?.replace(/_/g, " ")}
                                        </h3>
                                        <p style={{ margin: 0, fontSize: 12, color: "#64748B" }}>
                                            Trong {hoveredPanel.local_id} · {hoveredDefect.location_in_panel}
                                        </p>
                                    </div>
                                </div>

                                <div style={{
                                    padding: "10px 16px", borderRadius: 12,
                                    background: `${SEVERITY_COLORS[hoveredDefect.severity] || "#94a3b8"}22`,
                                    border: `1px solid ${SEVERITY_COLORS[hoveredDefect.severity] || "#94a3b8"}66`,
                                    display: "flex", justifyContent: "space-between", alignItems: "center"
                                }}>
                                    <span style={{ fontWeight: 700, color: SEVERITY_COLORS[hoveredDefect.severity] || "#94a3b8", textTransform: "capitalize" }}>
                                        {hoveredDefect.severity}
                                    </span>
                                    <span style={{ fontSize: 13, color: "#64748B" }}>{hoveredDefect.recommendation}</span>
                                </div>

                                <div style={{ padding: 16, border: "1px solid #e2e8f0", borderRadius: 16, display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
                                    <div>
                                        <p style={{ margin: "0 0 4px 0", fontSize: 11, color: "#94A3B8", fontWeight: 600, textTransform: "uppercase" }}>Defect Area</p>
                                        <p style={{ margin: 0, fontSize: 18, fontWeight: "bold", color: "#1E293B" }}>
                                            {(hoveredDefect.area_ratio_percent || 0).toFixed(3)}%
                                        </p>
                                    </div>
                                    <div>
                                        <p style={{ margin: "0 0 4px 0", fontSize: 11, color: "#94A3B8", fontWeight: 600, textTransform: "uppercase" }}>Overlap</p>
                                        <p style={{ margin: 0, fontSize: 18, fontWeight: "bold", color: "#1E293B" }}>
                                            {((hoveredDefect.overlap_ratio || 0) * 100).toFixed(1)}%
                                        </p>
                                    </div>
                                    <div>
                                        <p style={{ margin: "0 0 4px 0", fontSize: 11, color: "#94A3B8", fontWeight: 600, textTransform: "uppercase" }}>Position (u,v)</p>
                                        <p style={{ margin: 0, fontSize: 13, color: "#475569" }}>
                                            {hoveredDefect.relative_position
                                                ? `(${hoveredDefect.relative_position.u?.toFixed(2)}, ${hoveredDefect.relative_position.v?.toFixed(2)})`
                                                : "—"
                                            }
                                        </p>
                                    </div>
                                    <div>
                                        <p style={{ margin: "0 0 4px 0", fontSize: 11, color: "#94A3B8", fontWeight: 600, textTransform: "uppercase" }}>Confidence</p>
                                        <p style={{ margin: 0, fontSize: 13, color: "#475569" }}>
                                            {((hoveredDefect.confidence || 0) * 100).toFixed(0)}%
                                        </p>
                                    </div>
                                </div>
                            </>
                        ) : (
                            // HOVER PANEL
                            <>
                                {/* Header: Chi tiết Tấm số N + ID box */}
                                <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", paddingBottom: 16, borderBottom: "1px solid #e2e8f0" }}>
                                    <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
                                        <LayoutGrid size={24} color={colors.primary} />
                                        <div>
                                            <h3 style={{ margin: 0, color: "#1E293B" }}>
                                                Panel #{panelIndex} Details
                                            </h3>
                                            {hoveredPanel.row && (
                                                <p style={{ margin: 0, fontSize: 12, color: "#64748B" }}>
                                                    Row {hoveredPanel.row}, Col {hoveredPanel.col}
                                                </p>
                                            )}
                                        </div>
                                    </div>
                                    {/* ID TẤM HÌNH box */}
                                    <div style={{
                                        border: "2px solid #1E293B",
                                        borderRadius: 6,
                                        padding: "6px 14px",
                                        fontWeight: 700,
                                        fontSize: 13,
                                        color: "#1E293B",
                                        letterSpacing: 0.5,
                                        whiteSpace: "nowrap"
                                    }}>
                                        {hoveredPanel.local_id}
                                    </div>
                                </div>

                                {/* Tình trạng */}
                                <div style={{ padding: 20, background: isPanelFaulty(hoveredPanel) ? "#FFF5F5" : "#F0FDF4", borderRadius: 16, border: `1px solid ${isPanelFaulty(hoveredPanel) ? '#FED7D7' : '#BBF7D0'}` }}>
                                    <h4 style={{ margin: "0 0 10px 0", display: "flex", alignItems: "center", gap: 8, color: isPanelFaulty(hoveredPanel) ? colors.error : colors.success }}>
                                        <Activity size={18} /> Status
                                    </h4>
                                    <div style={{ fontSize: 22, fontWeight: "bold", color: isPanelFaulty(hoveredPanel) ? colors.error : colors.success }}>
                                        {isPanelFaulty(hoveredPanel)
                                            ? `${hoveredPanel.worst_severity || "Faulty"} — ${(hoveredPanel.total_defect_area_ratio_percent || hoveredPanel.total_panel_loss || 0).toFixed(2)}%`
                                            : "Healthy"
                                        }
                                    </div>
                                    {isPanelFaulty(hoveredPanel) && hoveredPanel.recommendation && (
                                        <p style={{ margin: "8px 0 0 0", fontSize: 13, color: "#9B2C2C" }}>
                                            → {hoveredPanel.recommendation}
                                        </p>
                                    )}
                                </div>

                                {/* Tọa độ */}
                                <div style={{ padding: 16, border: "1px solid #e2e8f0", borderRadius: 16 }}>
                                    <h4 style={{ margin: "0 0 10px 0", display: "flex", alignItems: "center", gap: 8, fontSize: 14 }}>
                                        <MapPin size={16} /> Bounding Box Coordinates
                                    </h4>
                                    <p style={{ margin: "4px 0", fontSize: 13 }}>
                                        X1: <b>{Math.round((hoveredPanel.bbox || hoveredPanel.box || [])[0] || 0)} px</b> | Y1: <b>{Math.round((hoveredPanel.bbox || hoveredPanel.box || [])[1] || 0)} px</b>
                                    </p>
                                    <p style={{ margin: "4px 0", fontSize: 13 }}>
                                        X2: <b>{Math.round((hoveredPanel.bbox || hoveredPanel.box || [])[2] || 0)} px</b> | Y2: <b>{Math.round((hoveredPanel.bbox || hoveredPanel.box || [])[3] || 0)} px</b>
                                    </p>
                                    <p style={{ margin: "12px 0 0 0", fontSize: 13, color: "#718096" }}>
                                        <Percent size={13} style={{ verticalAlign: "middle", marginRight: 4 }} />
                                        AI Confidence: <b>{((hoveredPanel.confidence || 0) * 100).toFixed(0)}%</b>
                                    </p>
                                </div>

                                {/* Danh sách lỗi */}
                                {isPanelFaulty(hoveredPanel) && hoveredPanel.defects && hoveredPanel.defects.length > 0 && (
                                    <div style={{ padding: 16, background: "#FFF5F5", border: "1px solid #FED7D7", borderRadius: 16 }}>
                                        <h4 style={{ margin: "0 0 10px 0", color: colors.error, display: "flex", alignItems: "center", gap: 8, fontSize: 14 }}>
                                            <AlertTriangle size={16} /> Anomalies ({hoveredPanel.defects.length}) — hover over a region to view details
                                        </h4>
                                        {hoveredPanel.defects.map((d, i) => (
                                            <div key={i} style={{
                                                display: "flex", justifyContent: "space-between",
                                                padding: "8px 0",
                                                borderBottom: i < hoveredPanel.defects.length - 1 ? "1px solid #FED7D7" : "none",
                                                alignItems: "center"
                                            }}>
                                                <div style={{ display: "flex", flexDirection: "column" }}>
                                                    <span style={{ fontWeight: 600, color: "#C53030", fontSize: 13, textTransform: "capitalize" }}>
                                                        {d.class_name?.replace(/_/g, " ") || d.type}
                                                    </span>
                                                    <span style={{ fontSize: 11, color: "#9B2C2C" }}>
                                                        {d.location_in_panel} · {d.severity}
                                                    </span>
                                                </div>
                                                <span style={{
                                                    color: SEVERITY_COLORS[d.severity] || "#E53E3E",
                                                    fontWeight: 600, fontSize: 13
                                                }}>
                                                    {(d.area_ratio_percent || d.loss || 0).toFixed(3)}%
                                                </span>
                                            </div>
                                        ))}
                                    </div>
                                )}
                            </>
                        )}
                    </div>

                    <button
                        onClick={() => onViewOnMap && onViewOnMap(image)}
                        style={{
                            marginTop: "auto",
                            padding: "16px",
                            border: "2px solid #1E293B",
                            background: "transparent",
                            color: "#1E293B",
                            fontWeight: 700,
                            fontSize: 14,
                            cursor: "pointer",
                            textTransform: "uppercase",
                            display: "flex",
                            justifyContent: "center",
                            alignItems: "center",
                            gap: 8,
                            transition: "all 0.2s"
                        }}
                        onMouseEnter={e => { e.currentTarget.style.background = "#1E293B"; e.currentTarget.style.color = "#fff"; }}
                        onMouseLeave={e => { e.currentTarget.style.background = "transparent"; e.currentTarget.style.color = "#1E293B"; }}
                    >
                        <MapIcon size={18} /> View on Map
                    </button>
                </div>
            </div>
        </div>
    );
}