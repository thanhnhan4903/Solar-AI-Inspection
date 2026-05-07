import React from "react";
import { colors } from "../../constants/theme";

export function ActionButton({ children, onClick, icon, variant, style }) {
    const isPrimary = variant !== "secondary";
    return (
        <button onClick={onClick} style={{
            display: "inline-flex", alignItems: "center", gap: 7,
            padding: "9px 18px", borderRadius: 10, border: isPrimary ? "none" : "1px solid #E2E8F0",
            background: isPrimary ? `linear-gradient(90deg,${colors.primary},#3B82F6)` : "#fff",
            color: isPrimary ? "#fff" : "#475569",
            fontWeight: 600, fontSize: 13, cursor: "pointer", transition: "opacity .15s, transform .1s",
            ...style,
        }}
            onMouseEnter={e => { e.currentTarget.style.opacity = ".88"; e.currentTarget.style.transform = "translateY(-1px)"; }}
            onMouseLeave={e => { e.currentTarget.style.opacity = "1"; e.currentTarget.style.transform = "none"; }}
        >
            {icon}{children}
        </button>
    );
}
