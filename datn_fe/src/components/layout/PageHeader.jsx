import React from "react";

export function PageHeader({ title, subtitle }) {
    return (
        <div style={{ marginBottom: 28 }}>
            <h1 style={{ fontSize: 24, fontWeight: 700, color: "#0F172A", margin: 0 }}>{title}</h1>
            {subtitle && <p style={{ color: "#64748B", fontSize: 14, marginTop: 4 }}>{subtitle}</p>}
        </div>
    );
}
