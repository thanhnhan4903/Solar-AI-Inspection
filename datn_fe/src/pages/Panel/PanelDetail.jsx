import React, { useState } from "react";
import { ArrowLeft, ArrowRight, MapPin, Activity, AlertTriangle, Percent, LayoutGrid, Info, Map as MapIcon } from "lucide-react";
import { colors } from "../../constants/theme";

const IMAGE_BASE_API = "http://127.0.0.1:8000/data/precalib/";

export default function PanelDetail({ panel: image, data, onSelect, onBack, onViewOnMap }) {
    if (!image) return null;

    const [hoveredPanel, setHoveredPanel] = useState(null);

    const imgW = image.image_width || 640;
    const imgH = image.image_height || 512;

    const handleNext = () => {
        if (!data || data.length === 0) return;
        const currentIndex = data.findIndex(img => img.filename === image.filename);
        if (currentIndex !== -1) {
            const nextIndex = (currentIndex + 1) % data.length;
            onSelect({
                ...data[nextIndex],
                id: `Hình ${nextIndex + 1}`,
                status: data[nextIndex].panels.filter(p => p.total_panel_loss > 0).length > 0 ? "defective" : "healthy",
                faulty_count: data[nextIndex].panels.filter(p => p.total_panel_loss > 0).length
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
                id: `Hình ${prevIndex + 1}`,
                status: data[prevIndex].panels.filter(p => p.total_panel_loss > 0).length > 0 ? "defective" : "healthy",
                faulty_count: data[prevIndex].panels.filter(p => p.total_panel_loss > 0).length
            });
        }
    };

    return (
        <div style={{ background: "#fff", borderRadius: 20, padding: 24, display: "flex", flexDirection: "column", height: "100%" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 20 }}>
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
                
                {/* TRÁI: Khu vực xem ảnh toàn cảnh & Tương tác */}
                <div>
                    <h2 style={{ marginBottom: 16, marginTop: 0 }}>{image.id} - Bản đồ chi tiết</h2>
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
                                const isFaulty = p.total_panel_loss > 0;
                                
                                const strokeColor = isFaulty ? "#EF4444" : "#10B981"; // Red if faulty, Green if healthy
                                const fillColor = isHovered 
                                    ? (isFaulty ? "rgba(239, 68, 68, 0.4)" : "rgba(16, 185, 129, 0.4)")
                                    : (isFaulty ? "rgba(239, 68, 68, 0.15)" : "transparent");

                                return p.polygon ? (
                                    <polygon 
                                        key={i}
                                        points={p.polygon.map(pt => pt.join(',')).join(' ')}
                                        fill={fillColor}
                                        stroke={strokeColor}
                                        strokeWidth={isHovered ? 3 : 1.5}
                                        onMouseEnter={() => setHoveredPanel(p)}
                                        onMouseLeave={() => setHoveredPanel(null)}
                                        style={{ cursor: "pointer", transition: "all 0.2s" }}
                                    />
                                ) : (
                                    p.box && (
                                        <rect
                                            key={i}
                                            x={p.box[0]} y={p.box[1]}
                                            width={p.box[2] - p.box[0]} height={p.box[3] - p.box[1]}
                                            fill={fillColor}
                                            stroke={strokeColor}
                                            strokeWidth={isHovered ? 3 : 1.5}
                                            onMouseEnter={() => setHoveredPanel(p)}
                                            onMouseLeave={() => setHoveredPanel(null)}
                                            style={{ cursor: "pointer", transition: "all 0.2s" }}
                                        />
                                    )
                                );
                            })}
                        </svg>
                    </div>
                    <p style={{ fontSize: 13, color: "#64748b", marginTop: 12, display: "flex", gap: 6, alignItems: "center" }}>
                        <Info size={14} /> <i>Rê chuột vào từng khung chữ nhật/đa giác trên ảnh để xem thông tin chi tiết của tấm pin đó.</i>
                    </p>
                </div>

                {/* PHẢI: Bảng thông tin thay đổi động */}
                <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
                    <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: 16 }}>
                        {!hoveredPanel ? (
                            // TRẠNG THÁI MẶC ĐỊNH (Khi chưa hover)
                            <>
                                <div style={{ padding: 20, background: "#f8fafc", borderRadius: 16, border: "1px solid #e2e8f0" }}>
                                    <h3 style={{ margin: "0 0 16px 0", color: "#1E293B" }}>Tổng quan {image.id}</h3>
                                    <div style={{ display: "flex", gap: 20 }}>
                                        <div style={{ flex: 1 }}>
                                            <p style={{ margin: "0 0 8px 0", color: "#64748B", fontSize: 13, fontWeight: 600 }}>TỔNG SỐ TẤM PIN</p>
                                            <p style={{ margin: 0, fontSize: 24, fontWeight: "bold", color: colors.primary }}>{image.total_panels}</p>
                                        </div>
                                        <div style={{ flex: 1 }}>
                                            <p style={{ margin: "0 0 8px 0", color: "#64748B", fontSize: 13, fontWeight: 600 }}>TẤM BỊ LỖI</p>
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
                                            Phát hiện <b>{image.faulty_count}</b> tấm pin có dấu hiệu hư hỏng. Hãy rê chuột vào các khung màu đỏ trên ảnh để kiểm tra.
                                        </p>
                                    </div>
                                )}
                            </>
                        ) : (
                            // TRẠNG THÁI HOVER (Hiển thị chi tiết tấm pin)
                            <>
                                <div style={{ display: "flex", alignItems: "center", gap: 12, paddingBottom: 16, borderBottom: "1px solid #e2e8f0" }}>
                                    <LayoutGrid size={24} color={colors.primary} />
                                    <h3 style={{ margin: 0, color: "#1E293B" }}>Chi tiết {hoveredPanel.local_id.replace("Panel_", "Tấm số ")}</h3>
                                </div>

                                {/* Tình trạng */}
                                <div style={{ padding: 20, background: hoveredPanel.total_panel_loss > 0 ? "#FFF5F5" : "#F0FDF4", borderRadius: 16, border: `1px solid ${hoveredPanel.total_panel_loss > 0 ? '#FED7D7' : '#BBF7D0'}` }}>
                                    <h4 style={{ margin: "0 0 10px 0", display: "flex", alignItems: "center", gap: 8, color: hoveredPanel.total_panel_loss > 0 ? colors.error : colors.success }}>
                                        <Activity size={18} /> Tình trạng
                                    </h4>
                                    <div style={{ fontSize: 24, fontWeight: "bold", color: hoveredPanel.total_panel_loss > 0 ? colors.error : colors.success }}>
                                        {hoveredPanel.total_panel_loss > 0 ? `Lỗi: ${hoveredPanel.total_panel_loss.toFixed(1)}%` : "Bình thường"}
                                    </div>
                                </div>

                                {/* Tọa độ bounding box */}
                                <div style={{ padding: 20, border: "1px solid #e2e8f0", borderRadius: 16 }}>
                                    <h4 style={{ margin: "0 0 10px 0", display: "flex", alignItems: "center", gap: 8 }}><MapPin size={18}/> Tọa độ bounding box</h4>
                                    <p style={{ margin: "4px 0" }}>X1: <b>{hoveredPanel.box ? Math.round(hoveredPanel.box[0]) : 0}</b> px | Y1: <b>{hoveredPanel.box ? Math.round(hoveredPanel.box[1]) : 0}</b> px</p>
                                    <p style={{ margin: "4px 0" }}>X2: <b>{hoveredPanel.box ? Math.round(hoveredPanel.box[2]) : 0}</b> px | Y2: <b>{hoveredPanel.box ? Math.round(hoveredPanel.box[3]) : 0}</b> px</p>
                                    {hoveredPanel.confidence && (
                                        <p style={{ marginTop: 12, color: "#718096", margin: "12px 0 0 0" }}>
                                            <Percent size={14} style={{ verticalAlign: "middle", marginRight: 4 }}/> Độ tin cậy AI: <b>{(hoveredPanel.confidence * 100).toFixed(0)}%</b>
                                        </p>
                                    )}
                                </div>

                                {/* Danh sách lỗi phát hiện */}
                                {hoveredPanel.total_panel_loss > 0 && hoveredPanel.defects && hoveredPanel.defects.length > 0 && (
                                    <div style={{ padding: 20, background: "#FFF5F5", border: "1px solid #FED7D7", borderRadius: 16 }}>
                                        <h4 style={{ margin: "0 0 12px 0", color: colors.error, display: "flex", alignItems: "center", gap: 8 }}>
                                            <AlertTriangle size={18}/> Lỗi phát hiện ({hoveredPanel.defects.length})
                                        </h4>
                                        {hoveredPanel.defects.map((d, i) => (
                                            <div key={i} style={{ display: "flex", justifyContent: "space-between", padding: "10px 0", borderBottom: i < hoveredPanel.defects.length - 1 ? "1px solid #FED7D7" : "none" }}>
                                                <span style={{ fontWeight: 600, color: "#C53030", textTransform: "capitalize" }}>{d.type}</span>
                                                <span style={{ color: "#E53E3E", fontWeight: 600 }}>-{d.loss.toFixed(2)}%</span>
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