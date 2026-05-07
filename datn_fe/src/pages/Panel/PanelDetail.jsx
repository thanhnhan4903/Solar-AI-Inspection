import React from "react";
import { ArrowLeft, MapPin, Activity, AlertTriangle, Percent } from "lucide-react";
import { colors } from "../../constants/theme";

// Ảnh từ results (đã qua YOLO)
const IMAGE_URL = "http://127.0.0.1:8000/data/results/";

export default function PanelDetail({ panel, onBack }) {
    if (!panel) return null;

    const hasDefects = panel.total_panel_loss > 0;

    return (
        <div style={{ background: "#fff", borderRadius: 20, padding: 24 }}>
            <button onClick={onBack} style={{ display: "flex", alignItems: "center", gap: 8, border: "none", background: "none", color: colors.primary, cursor: "pointer", fontWeight: 600, marginBottom: 20 }}>
                <ArrowLeft size={18} /> Quay lại
            </button>
            <div style={{ display: "grid", gridTemplateColumns: "1.2fr 1fr", gap: 32 }}>
                <div>
                    <h2 style={{ marginBottom: 16 }}>{panel.local_id} - Ảnh phân tích chi tiết</h2>
                    <div style={{ borderRadius: 12, overflow: "hidden", background: "#000", position: "relative" }}>
                        {/* Ảnh YOLO annotated từ results */}
                        <img 
                            src={`${IMAGE_URL}${panel.sourceFile}?t=${Date.now()}`} 
                            style={{ width: "100%", display: "block" }} 
                            alt="AI Result" 
                        />
                        {/* SVG overlay: khoanh vàng panel + đỏ cho lỗi */}
                        {panel.box && panel.imgW && panel.imgH && (
                            <svg 
                                style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%", pointerEvents: "none" }}
                                viewBox={`0 0 ${panel.imgW} ${panel.imgH}`}
                                preserveAspectRatio="xMidYMid meet"
                            >
                                {/* Khung vàng cho panel đang xem */}
                                <rect 
                                    x={panel.box[0]} y={panel.box[1]} 
                                    width={panel.box[2] - panel.box[0]} height={panel.box[3] - panel.box[1]}
                                    fill="rgba(255,255,0,0.15)" stroke="#FFD700" strokeWidth="3"
                                />
                                <text x={panel.box[0] + 4} y={panel.box[1] - 6} fill="#FFD700" fontSize="14" fontWeight="bold">
                                    {panel.local_id}
                                </text>
                                {/* Khung đỏ cho từng lỗi */}
                                {panel.defects && panel.defects.map((d, i) => (
                                    d.box && (
                                        <g key={i}>
                                            <rect 
                                                x={d.box[0]} y={d.box[1]} 
                                                width={d.box[2] - d.box[0]} height={d.box[3] - d.box[1]}
                                                fill="rgba(255,0,0,0.25)" stroke="#FF3333" strokeWidth="2"
                                            />
                                            <text x={d.box[0] + 2} y={d.box[1] - 4} fill="#FF3333" fontSize="11" fontWeight="bold">
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