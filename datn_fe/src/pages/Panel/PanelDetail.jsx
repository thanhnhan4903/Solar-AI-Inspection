import React, { useState, useEffect, useRef } from "react";
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
    "hotspot_single_cell": "#ec0a0aff",
    "hotspot_multi_cell": "#da0606ff",
    "shading": "#06b6d4",
    "soiling": "#eab308",
    "crack": "#a855f7",
};

const LOCATION_MAP = {
    "upper": "Trên",
    "middle": "Giữa",
    "lower": "Dưới",
    "upper-left": "Trên - Trái",
    "upper-right": "Trên - Phải",
    "upper-center": "Trên - Giữa",
    "middle-left": "Giữa - Trái",
    "middle-right": "Giữa - Phải",
    "middle-center": "Chính Giữa",
    "lower-left": "Dưới - Trái",
    "lower-right": "Dưới - Phải",
    "lower-center": "Dưới - Giữa",
    "center": "Giữa",
    "left": "Trái",
    "right": "Phải"
};

const DEFECT_NAME_MAP = {
    "hotspot_single_cell": "Điểm nóng cục bộ",
    "hotspot_multi_cell": "Điểm nóng đa cụm",
    "shading": "Che bóng",
    "soiling": "Bụi bẩn",
    "crack": "Nứt vỡ",
    "diode": "Lỗi Diode",
    "vegetation": "Cây cỏ che khuất",
    "panel_failure": "Hỏng tấm pin",
    "hotspot": "Điểm nóng",
    "pid": "Suy thoái PID",
    "delamination": "Bong tróc",
    "glass_breakage": "Vỡ kính"
};

const SEVERITY_MAP = {
    "very_minor": "Rất nhẹ",
    "minor": "Nhẹ",
    "moderate": "Cần theo dõi",
    "severe": "Ưu tiên bảo trì",
    "replace": "Cần thay thế"
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

// Kiểm tra tấm pin có lỗi hay không
function isPanelFaulty(p) {
    if (!p) return false;
    return p.status === "faulty" || p.total_panel_loss > 0;
}

export default function PanelDetail({ panel: image, data, onSelect, onBack, onViewOnMap }) {
    if (!image) return null;

    const [hoveredPanel, setHoveredPanel] = useState(null);
    const [hoveredDefect, setHoveredDefect] = useState(null);
    const [selectedFaultyPanel, setSelectedFaultyPanel] = useState(null);
    const [modalHoveredDefect, setModalHoveredDefect] = useState(null);

    const [zoomParams, setZoomParams] = useState({ scale: 1, originX: 50, originY: 50, translateX: 0, translateY: 0 });
    const [dragState, setDragState] = useState({ isDragging: false, startX: 0, startY: 0 });
    const imageContainerRef = useRef(null);

    useEffect(() => {
        const div = imageContainerRef.current;
        if (!div) return;
        const onWheel = (e) => {
            if (e.ctrlKey) {
                e.preventDefault();
                const zoomSpeed = 0.15;
                setZoomParams(prev => {
                    const newScale = e.deltaY < 0 ? prev.scale + zoomSpeed : prev.scale - zoomSpeed;
                    const clampedScale = Math.max(1, Math.min(newScale, 8));
                    if (prev.scale === 1 && clampedScale > 1) {
                        const rect = div.getBoundingClientRect();
                        const x = ((e.clientX - rect.left) / rect.width) * 100;
                        const y = ((e.clientY - rect.top) / rect.height) * 100;
                        return { scale: clampedScale, originX: x, originY: y, translateX: 0, translateY: 0 };
                    } else if (clampedScale === 1) {
                        return { scale: 1, originX: 50, originY: 50, translateX: 0, translateY: 0 };
                    }
                    return { ...prev, scale: clampedScale };
                });
            }
        };
        div.addEventListener("wheel", onWheel, { passive: false });
        return () => div.removeEventListener("wheel", onWheel);
    }, []);

    // Đồng bộ hóa tấm pin lỗi đang được zoom khi người dùng chuyển sang ảnh khác
    useEffect(() => {
        if (selectedFaultyPanel) {
            const matchingPanel = image.panels?.find(p => p.local_id === selectedFaultyPanel.local_id);
            // Chỉ giữ modal zoom mở nếu tìm thấy tấm pin tương ứng và tấm pin đó có lỗi
            if (matchingPanel && isPanelFaulty(matchingPanel)) {
                setSelectedFaultyPanel(matchingPanel);
            } else {
                setSelectedFaultyPanel(null);
            }
            setModalHoveredDefect(null);
        }
    }, [image]);

    useEffect(() => {
        const handleKeyDown = (e) => {
            if (e.key === "Escape") {
                if (selectedFaultyPanel) {
                    setSelectedFaultyPanel(null);
                    setModalHoveredDefect(null);
                } else {
                    onBack();
                }
                return;
            }

            if (!data || data.length === 0 || !image) return;
            const idx = data.findIndex(img => img.filename === image.filename);
            if (idx === -1) return;

            if (e.key === "ArrowLeft" && idx > 0) {
                onSelect({
                    ...data[idx - 1],
                    id: `Image ${idx}`,
                    status: data[idx - 1].panels.filter(p => (p.status === "faulty" || p.total_panel_loss > 0)).length > 0 ? "defective" : "healthy",
                    faulty_count: data[idx - 1].panels.filter(p => (p.status === "faulty" || p.total_panel_loss > 0)).length
                });
            } else if (e.key === "ArrowRight" && idx < data.length - 1) {
                onSelect({
                    ...data[idx + 1],
                    id: `Image ${idx + 2}`,
                    status: data[idx + 1].panels.filter(p => (p.status === "faulty" || p.total_panel_loss > 0)).length > 0 ? "defective" : "healthy",
                    faulty_count: data[idx + 1].panels.filter(p => (p.status === "faulty" || p.total_panel_loss > 0)).length
                });
            }
        };
        window.addEventListener("keydown", handleKeyDown);
        return () => window.removeEventListener("keydown", handleKeyDown);
    }, [data, image, onSelect, selectedFaultyPanel, onBack]);

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

    const panelIndex = hoveredPanel ? getPanelIndex(image.panels, hoveredPanel.local_id) : null;

    return (
        <div style={{ background: "#fff", borderRadius: 20, padding: 24, display: "flex", flexDirection: "column", height: "100%" }}>
            {/* Header */}
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
                <button onClick={onBack} style={{ display: "flex", alignItems: "center", gap: 8, border: "none", background: "none", color: colors.primary, cursor: "pointer", fontWeight: 600 }}>
                    <ArrowLeft size={18} /> Quay lại
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

                    <div 
                        ref={imageContainerRef} 
                        style={{ 
                            borderRadius: 12, overflow: "hidden", background: "#000", position: "relative", 
                            width: "100%", aspectRatio: "5/4",
                            cursor: zoomParams.scale > 1 ? (dragState.isDragging ? "grabbing" : "grab") : "default"
                        }}
                        onMouseDown={(e) => {
                            if (zoomParams.scale > 1) {
                                setDragState({ isDragging: true, startX: e.clientX, startY: e.clientY });
                                e.preventDefault();
                            }
                        }}
                        onMouseMove={(e) => {
                            if (dragState.isDragging) {
                                const dx = e.clientX - dragState.startX;
                                const dy = e.clientY - dragState.startY;
                                setZoomParams(prev => ({
                                    ...prev,
                                    translateX: prev.translateX + dx / prev.scale,
                                    translateY: prev.translateY + dy / prev.scale
                                }));
                                setDragState({ isDragging: true, startX: e.clientX, startY: e.clientY });
                            }
                        }}
                        onMouseUp={() => setDragState(prev => ({ ...prev, isDragging: false }))}
                        onMouseLeave={() => setDragState(prev => ({ ...prev, isDragging: false }))}
                    >
                        <div style={{
                            width: "100%", height: "100%", position: "absolute", top: 0, left: 0,
                            transform: `scale(${zoomParams.scale}) translate(${zoomParams.translateX}px, ${zoomParams.translateY}px)`,
                            transformOrigin: `${zoomParams.originX}% ${zoomParams.originY}%`,
                            transition: dragState.isDragging ? "none" : "transform 0.1s ease-out"
                        }}>
                            <img
                                src={`${IMAGE_BASE_API}${image.filename}`}
                                style={{ width: "100%", height: "100%", display: "block", objectFit: "contain" }}
                                alt={image.id}
                            />
                            <svg
                                style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%" }}
                                viewBox={`0 0 ${imgW} ${imgH}`}
                                preserveAspectRatio="xMidYMid meet"
                            >
                            {image.panels && image.panels.map((p, i) => {
                                const faulty = isPanelFaulty(p);
                                if (!faulty) return null; // Chỉ hiện tấm pin bị lỗi

                                const isHovered = hoveredPanel?.local_id === p.local_id;
                                const strokeColor = faulty ? "#EF4444" : "#22c55e";
                                const fillColor = isHovered
                                    ? (faulty ? "rgba(239,68,68,0.35)" : "rgba(34,197,94,0.25)")
                                    : (faulty ? "rgba(239,68,68,0.12)" : "rgba(34,197,94,0.08)");

                                return (
                                    <g key={i} onClick={() => setSelectedFaultyPanel(p)}>
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
                                            const defectStroke = "#00E5FF"; // Màu xanh dương lơ (cyan) để nổi bật trên nền nhiệt
                                            const isDefectHovered = hoveredDefect?.class_name === d.class_name
                                                && hoveredDefect?.location_in_panel === d.location_in_panel
                                                && hoveredPanel?.local_id === p.local_id;

                                            return d.polygon && d.polygon.length >= 3 ? (
                                                <polygon
                                                    key={di}
                                                    points={d.polygon.map(pt => Array.isArray(pt) ? pt.join(',') : pt).join(' ')}
                                                    fill={isDefectHovered ? "rgba(0, 229, 255, 0.2)" : "transparent"} 
                                                    stroke={defectStroke}
                                                    strokeWidth={isDefectHovered ? 3.5 : 2}
                                                    strokeDasharray="none" // Bỏ viền đứt nét để liền mạch, dễ nhìn hơn
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
                    </div>
                    <p style={{ fontSize: 13, color: "#64748b", marginTop: 12, display: "flex", gap: 6, alignItems: "center" }}>
                        <Info size={14} /> <i>Rê chuột vào viền đỏ để xem chi tiết, click để phóng to tấm pin, Ctrl + Lăn chuột để zoom ảnh.</i>
                    </p>
                </div>

                {/* PHẢI */}
                <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
                    <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: 16 }}>
                        {!hoveredPanel ? (
                            // MẶC ĐỊNH
                            <>
                                <div style={{ padding: 20, background: "#f8fafc", borderRadius: 16, border: "1px solid #e2e8f0" }}>
                                    <h3 style={{ margin: "0 0 16px 0", color: "#1E293B" }}>Tổng quan {image.id}</h3>
                                    <div style={{ display: "flex", gap: 20 }}>
                                        <div style={{ flex: 1 }}>
                                            <p style={{ margin: "0 0 8px 0", color: "#64748B", fontSize: 13, fontWeight: 600 }}>TỔNG SỐ TẤM PIN</p>
                                            <p style={{ margin: 0, fontSize: 24, fontWeight: "bold", color: colors.primary }}>{image.total_panels}</p>
                                        </div>
                                        <div style={{ flex: 1 }}>
                                            <p style={{ margin: "0 0 8px 0", color: "#64748B", fontSize: 13, fontWeight: 600 }}>SỐ LƯỢNG LỖI</p>
                                            <p style={{ margin: 0, fontSize: 24, fontWeight: "bold", color: image.faulty_count > 0 ? colors.danger : colors.success }}>{image.faulty_count}</p>
                                        </div>
                                    </div>
                                </div>
                                {image.faulty_count > 0 && (
                                    <div style={{ padding: 20, background: "#FFF5F5", borderRadius: 16, border: "1px solid #FED7D7" }}>
                                        <h4 style={{ margin: "0 0 10px 0", display: "flex", alignItems: "center", gap: 8, color: colors.error }}>
                                            <AlertTriangle size={18} /> Cảnh báo
                                        </h4>
                                        <p style={{ margin: 0, color: "#9B2C2C", fontSize: 14 }}>
                                            Phát hiện <b>{image.faulty_count}</b> tấm pin lỗi. Rê chuột vào khung màu đỏ để xem chi tiết.
                                        </p>
                                    </div>
                                )}
                            </>
                        ) : hoveredDefect ? (
                            // HOVER DEFECT
                            <>
                                <div style={{ display: "flex", alignItems: "center", gap: 16, paddingBottom: 16, borderBottom: "1px solid #e2e8f0" }}>
                                    <Target size={36} color="#00E5FF" />
                                    <div>
                                        <h3 style={{ margin: 0, color: "#1E293B", fontSize: 20 }}>
                                            {DEFECT_NAME_MAP[hoveredDefect?.class_name] || (typeof hoveredDefect?.class_name === 'string' ? hoveredDefect.class_name.replace(/_/g, " ") : String(hoveredDefect?.class_name || ""))}
                                        </h3>
                                        <p style={{ margin: "6px 0 0 0", fontSize: 16, color: "#475569", fontWeight: 500 }}>
                                            Tấm pin: <b>{hoveredPanel?.local_id || "N/A"}</b> &nbsp;|&nbsp; Vị trí: <b style={{color: "#0284c7"}}>{LOCATION_MAP[hoveredDefect?.location_in_panel] || hoveredDefect?.location_in_panel || "N/A"}</b>
                                        </p>
                                    </div>
                                </div>

                                <div style={{
                                    padding: "10px 16px", borderRadius: 12,
                                    background: `${SEVERITY_COLORS[hoveredDefect?.severity] || "#94a3b8"}22`,
                                    border: `1px solid ${SEVERITY_COLORS[hoveredDefect?.severity] || "#94a3b8"}66`,
                                    display: "flex", justifyContent: "space-between", alignItems: "center"
                                }}>
                                    <span style={{ fontWeight: 700, color: SEVERITY_COLORS[hoveredDefect?.severity] || "#94a3b8", textTransform: "capitalize" }}>
                                        {SEVERITY_MAP[hoveredDefect?.severity] || hoveredDefect?.severity || "Không rõ"}
                                    </span>
                                    <span style={{ fontSize: 13, color: "#64748B" }}>{hoveredDefect?.recommendation || ""}</span>
                                </div>

                                <div style={{ padding: 16, border: "1px solid #e2e8f0", borderRadius: 16, display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 12 }}>
                                    <div>
                                        <p style={{ margin: "0 0 4px 0", fontSize: 11, color: "#94A3B8", fontWeight: 600, textTransform: "uppercase" }}>Tổn thất dự kiến</p>
                                        <p style={{ margin: 0, fontSize: 18, fontWeight: "bold", color: "#1E293B" }}>
                                            {hoveredDefect?.class_name === "crack" ? "100%" : "33.3%"} / vùng
                                        </p>
                                    </div>
                                    <div>
                                        <p style={{ margin: "0 0 4px 0", fontSize: 11, color: "#94A3B8", fontWeight: 600, textTransform: "uppercase" }}>Vị trí (u,v)</p>
                                        <p style={{ margin: 0, fontSize: 15, color: "#475569", fontWeight: 500 }}>
                                            {hoveredDefect?.relative_position && typeof hoveredDefect.relative_position === 'object' && hoveredDefect.relative_position.u !== undefined
                                                ? `(${Number(hoveredDefect.relative_position.u || 0).toFixed(2)}, ${Number(hoveredDefect.relative_position.v || 0).toFixed(2)})`
                                                : "—"
                                            }
                                        </p>
                                    </div>
                                    <div>
                                        <p style={{ margin: "0 0 4px 0", fontSize: 11, color: "#94A3B8", fontWeight: 600, textTransform: "uppercase" }}>Độ tin cậy AI</p>
                                        <p style={{ margin: 0, fontSize: 15, color: "#475569", fontWeight: 500 }}>
                                            {((Number(hoveredDefect?.confidence) || 0) * 100).toFixed(0)}%
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
                                                Tấm pin {hoveredPanel.local_id}
                                            </h3>
                                            {hoveredPanel.row && (
                                                <p style={{ margin: 0, fontSize: 12, color: "#64748B" }}>
                                                    Hàng {hoveredPanel.row} · Cột {hoveredPanel.col}
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

                                {/* Đánh giá hiệu suất */}
                                <div style={{ padding: 20, background: isPanelFaulty(hoveredPanel) ? "#FFF5F5" : "#F0FDF4", borderRadius: 16, border: `1px solid ${isPanelFaulty(hoveredPanel) ? '#FED7D7' : '#BBF7D0'}` }}>
                                    <h4 style={{ margin: "0 0 10px 0", display: "flex", alignItems: "center", gap: 8, color: isPanelFaulty(hoveredPanel) ? colors.error : colors.success }}>
                                        <Activity size={18} /> Đánh giá hiệu suất
                                    </h4>
                                    <div style={{ fontSize: 22, fontWeight: "bold", color: isPanelFaulty(hoveredPanel) ? colors.error : colors.success }}>
                                        {isPanelFaulty(hoveredPanel)
                                            ? `Hao hụt: ${(hoveredPanel.total_panel_loss || 0).toFixed(0)} W`
                                            : "Hoạt động tối ưu (600 W)"
                                        }
                                    </div>
                                    {isPanelFaulty(hoveredPanel) && hoveredPanel.recommendation && (
                                        <p style={{ margin: "8px 0 0 0", fontSize: 13, color: "#9B2C2C", fontWeight: 600 }}>
                                            Khuyến nghị: {hoveredPanel.recommendation}
                                        </p>
                                    )}
                                </div>

                                {/* Tọa độ */}
                                <div style={{ padding: 16, border: "1px solid #e2e8f0", borderRadius: 16 }}>
                                    <h4 style={{ margin: "0 0 10px 0", display: "flex", alignItems: "center", gap: 8, fontSize: 14 }}>
                                        <MapPin size={16} /> Tọa độ khung (Bounding Box)
                                    </h4>
                                    <p style={{ margin: "4px 0", fontSize: 13 }}>
                                        X1: <b>{Math.round((hoveredPanel.bbox || hoveredPanel.box || [])[0] || 0)} px</b> | Y1: <b>{Math.round((hoveredPanel.bbox || hoveredPanel.box || [])[1] || 0)} px</b>
                                    </p>
                                    <p style={{ margin: "4px 0", fontSize: 13 }}>
                                        X2: <b>{Math.round((hoveredPanel.bbox || hoveredPanel.box || [])[2] || 0)} px</b> | Y2: <b>{Math.round((hoveredPanel.bbox || hoveredPanel.box || [])[3] || 0)} px</b>
                                    </p>
                                    <p style={{ margin: "12px 0 0 0", fontSize: 13, color: "#718096" }}>
                                        <Percent size={13} style={{ verticalAlign: "middle", marginRight: 4 }} />
                                        Độ tin cậy AI: <b>{((hoveredPanel.confidence || 0) * 100).toFixed(0)}%</b>
                                    </p>
                                </div>

                                {/* Danh sách lỗi */}
                                {isPanelFaulty(hoveredPanel) && hoveredPanel.defects && hoveredPanel.defects.length > 0 && (
                                    <div style={{ padding: 16, background: "#FFF5F5", border: "1px solid #FED7D7", borderRadius: 16 }}>
                                        <h4 style={{ margin: "0 0 10px 0", color: colors.error, display: "flex", alignItems: "center", gap: 8, fontSize: 14 }}>
                                            <AlertTriangle size={16} /> Điểm bất thường ({hoveredPanel.defects.length}) — Rê chuột vào lỗi để xem
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
                                                        {DEFECT_NAME_MAP[d.class_name] || d.class_name?.replace(/_/g, " ") || d.type}
                                                    </span>
                                                    <span style={{ fontSize: 11, color: "#9B2C2C" }}>
                                                        {LOCATION_MAP[d.location_in_panel] || d.location_in_panel} · {SEVERITY_MAP[d.severity] || d.severity}
                                                    </span>
                                                </div>
                                                <span style={{
                                                    color: SEVERITY_COLORS[d.severity] || "#E53E3E",
                                                    fontWeight: 600, fontSize: 13
                                                }}>
                                                    AI: {((d.confidence || 0) * 100).toFixed(0)}%
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
                        <MapIcon size={18} /> Xem trên bản đồ
                    </button>
                </div>
            </div>

            {/* Modal Zoom Tấm Pin Lỗi */}
            {selectedFaultyPanel && (() => {
                const pBox = selectedFaultyPanel.bbox || selectedFaultyPanel.box;
                if (!pBox) return null;
                const pad = 30; // padding để khung hình rộng hơn một chút so với tấm pin
                const vBox = `${Math.max(0, pBox[0] - pad)} ${Math.max(0, pBox[1] - pad)} ${pBox[2] - pBox[0] + pad*2} ${pBox[3] - pBox[1] + pad*2}`;
                
                return (
                    <div 
                        style={{
                            position: "fixed", top: 0, left: 0, width: "100vw", height: "100vh",
                            background: "rgba(15, 23, 42, 0.95)", zIndex: 9999,
                            display: "flex", justifyContent: "center", alignItems: "center",
                            padding: 40, gap: 32
                        }}
                        onClick={() => { setSelectedFaultyPanel(null); setModalHoveredDefect(null); }}
                    >
                        {/* LEFT: ZOOMED IMAGE */}
                        <div style={{ flex: 1, height: "100%", display: "flex", flexDirection: "column", justifyContent: "center", alignItems: "center" }}>
                            <h2 style={{ color: "#fff", marginBottom: 20, alignSelf: "flex-start" }}>
                                Cận cảnh lỗi - Tấm pin {selectedFaultyPanel.local_id}
                            </h2>
                            <div 
                                style={{ 
                                    width: "100%", maxHeight: "80vh", aspectRatio: "5/4",
                                    background: "#000", borderRadius: 16, overflow: "hidden",
                                    boxShadow: "0 25px 50px -12px rgba(0, 0, 0, 0.5)",
                                    border: "1px solid #334155"
                                }} 
                                onClick={e => e.stopPropagation()}
                            >
                                <svg width="100%" height="100%" viewBox={vBox} preserveAspectRatio="xMidYMid meet">
                                    <image href={`${IMAGE_BASE_API}${image.filename}`} width={imgW} height={imgH} />
                                    <polygon
                                        points={selectedFaultyPanel.polygon?.map(pt => Array.isArray(pt) ? pt.join(',') : pt).join(' ')}
                                        fill="transparent"
                                        stroke="#EF4444"
                                        strokeWidth={2}
                                    />
                                    {selectedFaultyPanel.defects && selectedFaultyPanel.defects.map((d, di) => {
                                        const isDefectHovered = modalHoveredDefect?.class_name === d.class_name
                                                && modalHoveredDefect?.location_in_panel === d.location_in_panel;
                                        return d.polygon && d.polygon.length >= 3 ? (
                                            <polygon
                                                key={di}
                                                points={d.polygon.map(pt => Array.isArray(pt) ? pt.join(',') : pt).join(' ')}
                                                fill={isDefectHovered ? "rgba(0, 229, 255, 0.2)" : "transparent"}
                                                stroke="#00E5FF"
                                                strokeWidth={isDefectHovered ? 3.5 : 2}
                                                style={{ cursor: "pointer", transition: "all 0.2s" }}
                                                onMouseEnter={() => setModalHoveredDefect(d)}
                                                onMouseLeave={() => setModalHoveredDefect(null)}
                                            />
                                        ) : null
                                    })}
                                </svg>
                            </div>
                            <p style={{ color: "#94A3B8", marginTop: 20, alignSelf: "flex-start" }}>Nhấn ESC hoặc click ra ngoài để đóng</p>
                        </div>
                        
                        {/* RIGHT: DEFECT DETAILS */}
                        <div 
                            style={{ 
                                width: 420, background: "#fff", borderRadius: 20, padding: 24, 
                                alignSelf: "center", maxHeight: "80vh", overflowY: "auto",
                                display: "flex", flexDirection: "column", gap: 16
                            }} 
                            onClick={e => e.stopPropagation()}
                        >
                            {!modalHoveredDefect ? (
                                <div style={{ textAlign: "center", color: "#64748b", margin: "auto 0" }}>
                                    <Info size={40} style={{ opacity: 0.5, marginBottom: 16, display: "inline-block" }} />
                                    <p style={{ margin: 0, fontSize: 16 }}>Rê chuột vào viền lỗi (màu xanh cyan) trên ảnh bên cạnh để xem chi tiết.</p>
                                </div>
                            ) : (
                                <>
                                    <div style={{ display: "flex", alignItems: "center", gap: 16, paddingBottom: 16, borderBottom: "1px solid #e2e8f0" }}>
                                        <Target size={36} color="#00E5FF" />
                                        <div>
                                            <h3 style={{ margin: 0, color: "#1E293B", fontSize: 20 }}>
                                                {DEFECT_NAME_MAP[modalHoveredDefect?.class_name] || (typeof modalHoveredDefect?.class_name === 'string' ? modalHoveredDefect.class_name.replace(/_/g, " ") : String(modalHoveredDefect?.class_name || ""))}
                                            </h3>
                                            <p style={{ margin: "6px 0 0 0", fontSize: 15, color: "#475569", fontWeight: 500 }}>
                                                Tấm pin: <b>{selectedFaultyPanel.local_id}</b> &nbsp;|&nbsp; Vị trí: <b style={{color: "#0284c7"}}>{LOCATION_MAP[modalHoveredDefect?.location_in_panel] || modalHoveredDefect?.location_in_panel}</b>
                                            </p>
                                        </div>
                                    </div>

                                    <div style={{
                                        padding: "12px 16px", borderRadius: 12,
                                        background: `${SEVERITY_COLORS[modalHoveredDefect?.severity] || "#94a3b8"}22`,
                                        border: `1px solid ${SEVERITY_COLORS[modalHoveredDefect?.severity] || "#94a3b8"}66`,
                                        display: "flex", justifyContent: "space-between", alignItems: "center"
                                    }}>
                                        <span style={{ fontWeight: 700, color: SEVERITY_COLORS[modalHoveredDefect?.severity] || "#94a3b8", textTransform: "capitalize" }}>
                                            {SEVERITY_MAP[modalHoveredDefect?.severity] || modalHoveredDefect?.severity}
                                        </span>
                                        <span style={{ fontSize: 13, color: "#64748B" }}>{modalHoveredDefect?.recommendation || "Cần kiểm tra"}</span>
                                    </div>

                                    <div style={{ padding: 16, border: "1px solid #e2e8f0", borderRadius: 16, display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
                                        <div>
                                            <p style={{ margin: "0 0 4px 0", fontSize: 11, color: "#94A3B8", fontWeight: 600, textTransform: "uppercase" }}>Tổn thất dự kiến</p>
                                            <p style={{ margin: 0, fontSize: 16, fontWeight: "bold", color: "#1E293B" }}>
                                                {modalHoveredDefect?.class_name === "crack" ? "100%" : "33.3%"} / vùng
                                            </p>
                                        </div>
                                        <div>
                                            <p style={{ margin: "0 0 4px 0", fontSize: 11, color: "#94A3B8", fontWeight: 600, textTransform: "uppercase" }}>Độ tin cậy AI</p>
                                            <p style={{ margin: 0, fontSize: 16, color: "#475569", fontWeight: 500 }}>
                                                {((Number(modalHoveredDefect?.confidence) || 0) * 100).toFixed(0)}%
                                            </p>
                                        </div>
                                    </div>
                                </>
                            )}
                        </div>
                    </div>
                );
            })()}
        </div>
    );
}