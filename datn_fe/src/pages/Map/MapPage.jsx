import React, { useState, useEffect } from "react";
import { colors } from "../../constants/theme";
import { PageHeader } from "../../components/layout/PageHeader";
import { SolarCard, CardHeader } from "../../components/ui/SolarCard";

export default function MapPage() {
    const [panel, setPanel] = useState(null);
    useEffect(() => { const d = localStorage.getItem("selectedPanel"); if (d) setPanel(JSON.parse(d)); }, []);

    return (
        <div>
            <PageHeader title="GIS Mapping" subtitle="Real-time panel location overlay" />
            <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr", gap: 16 }}>
                <div style={{
                    height: 500, background: "linear-gradient(135deg,#E0F2FE,#BAE6FD,#7DD3FC)",
                    borderRadius: 16, border: "1px solid #E2E8F0",
                    display: "flex", alignItems: "center", justifyContent: "center",
                    position: "relative", color: "#0369A1", fontWeight: 600, fontSize: 15,
                }}>
                    🗺 Map View
                    {panel && (
                        <div style={{
                            position: "absolute", background: colors.danger, color: "#fff",
                            padding: "6px 14px", borderRadius: 20, fontSize: 13, fontWeight: 600,
                            boxShadow: "0 4px 12px rgba(239,68,68,.35)",
                        }}>
                            📍 {panel.name}
                        </div>
                    )}
                </div>
                <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                    <SolarCard style={{ flex: 1 }}>
                        <CardHeader title="Thermal" />
                        <div style={{ height: 200, background: "linear-gradient(135deg,#FEF3C7,#FCA5A5)", borderRadius: 12, display: "flex", alignItems: "center", justifyContent: "center", color: "#92400E", fontWeight: 500 }}>Thermal Layer</div>
                    </SolarCard>
                    <SolarCard style={{ flex: 1 }}>
                        <CardHeader title="RGB" />
                        <div style={{ height: 200, background: "linear-gradient(135deg,#E0F2FE,#DBEAFE)", borderRadius: 12, display: "flex", alignItems: "center", justifyContent: "center", color: "#1E40AF", fontWeight: 500 }}>RGB Layer</div>
                    </SolarCard>
                </div>
            </div>
        </div>
    );
}
