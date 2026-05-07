import React from "react";
import { Zap } from "lucide-react";
import { SolarCard, CardHeader } from "./SolarCard";

export function ImageWithMask({ title, subtitle, type, panelId, masks = [] }) {
    const isThermal = type === "thermal";
    
    return (
        <SolarCard style={{ height: "fit-content" }}>
            <CardHeader title={title} subtitle={subtitle} />
            <div style={{
                height: 320,
                background: isThermal 
                    ? "linear-gradient(135deg, #4c1d95, #7c3aed, #db2777, #ea580c, #facc15)"
                    : "linear-gradient(135deg,#E0F2FE,#DBEAFE)",
                borderRadius: 14,
                position: "relative",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                border: "1px solid #E2E8F0",
                overflow: "hidden"
            }}>
                {/* Background Decoration for placeholders */}
                {!isThermal && (
                    <div style={{ textAlign: "center", color: "#94A3B8", fontSize: 14 }}>
                        <p style={{ margin: 0 }}>Panel #{panelId}</p>
                        <p style={{ margin: 0, fontSize: 12, opacity: 0.7 }}>RGB View</p>
                    </div>
                )}
                {isThermal && (
                    <>
                        <div style={{ 
                            position: "absolute", inset: 0, 
                            background: "radial-gradient(circle at 30% 40%, rgba(255,255,255,0.2) 0%, transparent 60%)",
                            pointerEvents: "none"
                        }} />
                        <span style={{ color: "#fff", fontWeight: 600, fontSize: 15, display: "flex", alignItems: "center", gap: 8, textShadow: "0 2px 4px rgba(0,0,0,0.3)" }}>
                            <Zap size={18} fill="#fff" /> Thermal View
                        </span>
                    </>
                )}

                {/* SVG Overlay for Polygon Masks */}
                <svg 
                    viewBox="0 0 100 100" 
                    preserveAspectRatio="none"
                    style={{
                        position: "absolute",
                        top: 0,
                        left: 0,
                        width: "100%",
                        height: "100%",
                        pointerEvents: "none"
                    }}
                >
                    {masks.map((mask, idx) => (
                        <g key={idx}>
                            <polygon
                                points={mask.points}
                                fill={isThermal ? "rgba(255,255,255,0.2)" : `${mask.color}20`}
                                stroke={isThermal ? "#fff" : mask.color}
                                strokeWidth="0.8"
                                style={{ 
                                    vectorEffect: "non-scaling-stroke",
                                    filter: isThermal ? "drop-shadow(0 0 2px rgba(255,255,255,0.5))" : `drop-shadow(0 0 2px ${mask.color}50)`
                                }}
                            />
                            {/* Label for the mask */}
                            <foreignObject x={mask.points.split(" ")[0].split(",")[0]} y={parseInt(mask.points.split(" ")[0].split(",")[1]) - 8} width="50" height="20">
                                <div style={{ 
                                    fontSize: "3px", 
                                    fontWeight: 700, 
                                    color: isThermal ? "#fff" : mask.color,
                                    textTransform: "uppercase",
                                    background: isThermal ? "rgba(0,0,0,0.3)" : "rgba(255,255,255,0.8)",
                                    padding: "1px 2px",
                                    borderRadius: "1px",
                                    display: "inline-block"
                                }}>
                                    {mask.label}
                                </div>
                            </foreignObject>
                        </g>
                    ))}
                </svg>
            </div>
        </SolarCard>
    );
}
