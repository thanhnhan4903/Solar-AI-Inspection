import React from "react";
import { colors } from "../../constants/theme";
import { PageHeader } from "../../components/layout/PageHeader";
import { ActionButton } from "../../components/ui/ActionButton";
import { BadgePill } from "../../components/ui/BadgePill";

// Ảnh từ results (đã qua YOLO, có annotation sẵn)
const IMAGE_URL = "http://127.0.0.1:8000/data/results/";

export default function PanelPage({ data, onSelect, onNavigate }) {
    if (!data || data.length === 0) return <div style={{ textAlign: "center", padding: 100 }}><ActionButton onClick={() => onNavigate("home")}>Vui lòng chạy AI tại Dashboard trước</ActionButton></div>;

    const allPanels = data.flatMap(img => img.panels.map(p => ({ 
        ...p, 
        sourceFile: img.filename,
        imgW: img.image_width,
        imgH: img.image_height,
        status: p.total_panel_loss > 0 ? "hotspot" : "healthy" 
    })));

    return (
        <div>
            <PageHeader title="Panel Management" subtitle={`${allPanels.length} panels detected`} />
            <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 16 }}>
                {allPanels.map((p, index) => (
                    <div key={index} onClick={() => onSelect(p)} style={{ background: "#fff", borderRadius: 16, border: "1px solid #E2E8F0", overflow: "hidden", cursor: "pointer", transition: "transform 0.2s, box-shadow 0.2s" }}
                        onMouseEnter={e => { e.currentTarget.style.transform = "translateY(-2px)"; e.currentTarget.style.boxShadow = "0 8px 25px rgba(0,0,0,0.1)"; }}
                        onMouseLeave={e => { e.currentTarget.style.transform = "none"; e.currentTarget.style.boxShadow = "none"; }}
                    >
                        <div style={{ position: "relative", width: "100%", paddingTop: "75%", background: "#000" }}>
                            {/* Ảnh YOLO annotated từ results */}
                            <img 
                                src={`${IMAGE_URL}${p.sourceFile}?t=${Date.now()}`} 
                                style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%", objectFit: "contain" }} 
                                alt={p.local_id} 
                            />
                            {/* SVG overlay: khoanh vàng panel này */}
                            {p.box && p.imgW && p.imgH && (
                                <svg 
                                    style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%", pointerEvents: "none" }}
                                    viewBox={`0 0 ${p.imgW} ${p.imgH}`}
                                    preserveAspectRatio="xMidYMid meet"
                                >
                                    <rect 
                                        x={p.box[0]} y={p.box[1]} 
                                        width={p.box[2] - p.box[0]} height={p.box[3] - p.box[1]}
                                        fill="rgba(255,255,0,0.18)" stroke="#FFD700" strokeWidth="3"
                                    />
                                </svg>
                            )}
                        </div>
                        <div style={{ padding: 12, display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                            <span style={{ fontWeight: 600, fontSize: 13 }}>{p.local_id}</span>
                            <BadgePill type={p.status} />
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
}