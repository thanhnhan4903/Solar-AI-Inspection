import React from "react";

export function SolarCard({ children, style }) {
    return (
        <div style={{
            background: "#fff", borderRadius: 16, padding: "20px 22px",
            border: "1px solid #E2E8F0", boxShadow: "0 1px 3px rgba(0,0,0,.04)",
            ...style,
        }}>
            {children}
        </div>
    );
}

export function CardHeader({ title, subtitle }) {
    return (
        <div style={{ marginBottom: 16 }}>
            <h3 style={{ fontSize: 15, fontWeight: 600, color: "#0F172A", margin: 0 }}>{title}</h3>
            {subtitle && <p style={{ fontSize: 12, color: "#94A3B8", marginTop: 2 }}>{subtitle}</p>}
        </div>
    );
}
