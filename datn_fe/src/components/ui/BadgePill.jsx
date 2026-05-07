import React from "react";

export const BADGE_CFG = {
    hotspot: { bg: "#FEE2E2", color: "#DC2626", label: "Hotspot" },
    soiling: { bg: "#FEF3C7", color: "#D97706", label: "Soiling" },
    healthy: { bg: "#D1FAE5", color: "#059669", label: "Healthy" },
    crack: { bg: "#EDE9FE", color: "#7C3AED", label: "Crack" },
};

export function BadgePill({ type }) {
    const cfg = BADGE_CFG[type] || BADGE_CFG.healthy;
    return (
        <span style={{
            display: "inline-block", padding: "3px 10px", borderRadius: 20,
            background: cfg.bg, color: cfg.color, fontSize: 11, fontWeight: 600,
        }}>{cfg.label}</span>
    );
}

// Backward compat
export function Badge({ type }) { return <BadgePill type={type} />; }
