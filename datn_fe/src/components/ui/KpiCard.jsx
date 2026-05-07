import React from "react";
import { colors } from "../../constants/theme";

export function KpiCard({ icon, label, value, accent, delta, up }) {
    return (
        <div style={{
            background: "#fff", borderRadius: 16, padding: "18px 20px",
            border: "1px solid #E2E8F0", borderTop: `3px solid ${accent}`,
            boxShadow: "0 1px 3px rgba(0,0,0,.04)",
        }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
                <div>
                    <p style={{ fontSize: 12, color: "#94A3B8", fontWeight: 500, marginBottom: 8, textTransform: "uppercase", letterSpacing: ".5px" }}>{label}</p>
                    <p style={{ fontSize: 26, fontWeight: 700, color: "#0F172A", lineHeight: 1 }}>{value}</p>
                    {delta && (
                        <p style={{ fontSize: 12, color: up === false ? colors.danger : up ? colors.success : "#94A3B8", marginTop: 6, fontWeight: 500 }}>
                            {up === true ? "▲ " : up === false ? "▼ " : ""}{delta}
                        </p>
                    )}
                </div>
                <div style={{
                    width: 40, height: 40, borderRadius: 12,
                    background: `${accent}15`, display: "flex", alignItems: "center", justifyContent: "center",
                    color: accent,
                }}>{icon}</div>
            </div>
        </div>
    );
}

// Backward compat
export function StatCard({ title, value }) {
    return <KpiCard label={title} value={value} accent={colors.primary} />;
}
