import React from "react";
import { ArrowLeft, MapPin, Activity, AlertTriangle, Percent } from "lucide-react";
import { colors } from "../../constants/theme";

// Gọi API lấy ảnh cắt 50% có khung xanh
const PANEL_IMAGE_API = "http://127.0.0.1:8000/api/v1/panel-image";

export default function PanelDetail({ panel, onBack }) {
    if (!panel) return null;

    const hasDefects = panel.total_panel_loss > 0;

    // Tính toán lại tọa độ SVG cho ảnh cắt 50%
    const imgW = panel.imgW || 640;
    const imgH = panel.imgH || 512;
    const cropW = Math.floor(imgW * 0.5);
    const cropH = Math.floor(imgH * 0.5);
    
    let cx1 = 0, cy1 = 0;
    if (panel.box) {
        const cx = Math.floor((panel.box[0] + panel.box[2]) / 2);
        const cy = Math.floor((panel.box[1] + panel.box[3]) / 2);
        cx1 = Math.max(0, cx - Math.floor(cropW / 2));
        cy1 = Math.max(0, cy - Math.floor(cropH / 2));
        
        let cx2 = Math.min(imgW, cx + Math.floor(cropW / 2));
        let cy2 = Math.min(imgH, cy + Math.floor(cropH / 2));
        
        if (cx2 - cx1 < cropW) {
            if (cx1 === 0) cx2 = Math.min(imgW, cx1 + cropW);
            if (cx2 === imgW) cx1 = Math.max(0, cx2 - cropW);
        }
        if (cy2 - cy1 < cropH) {
            if (cy1 === 0) cy2 = Math.min(imgH, cy1 + cropH);
            if (cy2 === imgH) cy1 = Math.max(0, cy2 - cropH);
        }
    }

    return (
        <div style={{ background: "#fff", borderRadius: 20, padding: 24 }}>
            <button onClick={onBack} style={{ display: "flex", alignItems: "center", gap: 8, border: "none", background: "none", color: colors.primary, cursor: "pointer", fontWeight: 600, marginBottom: 20 }}>
                <ArrowLeft size={18} /> Quay lại
            </button>
            <div style={{ display: "grid", gridTemplateColumns: "1.2fr 1fr", gap: 32 }}>
                <div>
                    <h2 style={{ marginBottom: 16 }}>{panel.local_id} - Ảnh phân tích chi tiết</h2>
                    <div style={{ borderRadius: 12, overflow: "hidden", background: "#000", position: "relative" }}>
                        {/* Ảnh đã cắt từ Backend */}
                        <img 
                            src={`${PANEL_IMAGE_API}?filename=${panel.sourceFile}&x1=${panel.box[0]}&y1=${panel.box[1]}&x2=${panel.box[2]}&y2=${panel.box[3]}${panel.polygon ? `&polygon=${panel.polygon.flat().join(',')}` : ''}`} 
                            style={{ width: "100%", display: "block" }} 
                            alt="AI Result" 
                        />
                        {/* SVG overlay: chỉ cần vẽ các lỗi (khung xanh đã được vẽ từ Backend) */}
                        {panel.box && panel.imgW && panel.imgH && (
                            <svg 
                                style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%", pointerEvents: "none" }}
                                viewBox={`0 0 ${cropW} ${cropH}`}
                                preserveAspectRatio="xMidYMid meet"
                            >
                                {/* Khung đỏ cho từng lỗi */}
                                {panel.defects && panel.defects.map((d, i) => (
                                    d.box && (
                                        <g key={i}>
                                            <rect 
                                                x={d.box[0] - cx1} y={d.box[1] - cy1} 
                                                width={d.box[2] - d.box[0]} height={d.box[3] - d.box[1]}
                                                fill="rgba(255,0,0,0.25)" stroke="#FF3333" strokeWidth="2"
                                            />
                                            <text x={d.box[0] - cx1 + 2} y={d.box[1] - cy1 - 4} fill="#FF3333" fontSize="11" fontWeight="bold">
                                                {d.type}
                                            </text>
                                        </g>
                                    )
                                ))}
                            </svg>
                        )}
                    </div>
                </div>
                <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
                    {/* Tình trạng */}
                    <div style={{ padding: 20, background: "#f8fafc", borderRadius: 16 }}>
                        <h4 style={{ margin: "0 0 10px 0", display: "flex", alignItems: "center", gap: 8 }}><Activity size={18}/> Tình trạng</h4>
                        <div style={{ fontSize: 24, fontWeight: "bold", color: hasDefects ? colors.error : colors.success }}>
                            {hasDefects ? `Lỗi: ${panel.total_panel_loss.toFixed(1)}%` : "Bình thường"}
                        </div>
                    </div>
                    {/* Tọa độ bounding box */}
                    <div style={{ padding: 20, border: "1px solid #e2e8f0", borderRadius: 16 }}>
                        <h4 style={{ margin: "0 0 10px 0", display: "flex", alignItems: "center", gap: 8 }}><MapPin size={18}/> Tọa độ bounding box</h4>
                        <p>X1: <b>{panel.box ? Math.round(panel.box[0]) : 0}</b> px | Y1: <b>{panel.box ? Math.round(panel.box[1]) : 0}</b> px</p>
                        <p>X2: <b>{panel.box ? Math.round(panel.box[2]) : 0}</b> px | Y2: <b>{panel.box ? Math.round(panel.box[3]) : 0}</b> px</p>
                        {panel.confidence && (
                            <p style={{ marginTop: 8, color: "#718096" }}>
                                <Percent size={14} style={{ verticalAlign: "middle" }}/> Confidence: <b>{(panel.confidence * 100).toFixed(0)}%</b>
                            </p>
                        )}
                    </div>
                    {/* Danh sách lỗi phát hiện */}
                    {hasDefects && panel.defects && panel.defects.length > 0 && (
                        <div style={{ padding: 20, background: "#FFF5F5", border: "1px solid #FED7D7", borderRadius: 16 }}>
                            <h4 style={{ margin: "0 0 10px 0", color: colors.error, display: "flex", alignItems: "center", gap: 8 }}><AlertTriangle size={18}/> Lỗi phát hiện ({panel.defects.length})</h4>
                            {panel.defects.map((d, i) => (
                                <div key={i} style={{ display: "flex", justifyContent: "space-between", padding: "8px 0", borderBottom: i < panel.defects.length - 1 ? "1px solid #FED7D7" : "none" }}>
                                    <span style={{ fontWeight: 600, color: "#C53030" }}>{d.type}</span>
                                    <span style={{ color: "#E53E3E", fontWeight: 600 }}>-{d.loss.toFixed(2)}%</span>
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}