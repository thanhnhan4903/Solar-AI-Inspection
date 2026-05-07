import React from "react";
import { AlertTriangle, Zap, Download } from "lucide-react";
import { colors } from "../../constants/theme";
import { PageHeader } from "../../components/layout/PageHeader";
import { KpiCard } from "../../components/ui/KpiCard";
import { SolarCard, CardHeader } from "../../components/ui/SolarCard";
import { ActionButton } from "../../components/ui/ActionButton";

export default function ReportPage({ data, batchId }) {
    // 1. Logic tính toán số liệu thực tế từ AI
    const allPanels = data?.flatMap(img => img.panels) || [];
    const faultyPanels = allPanels.filter(p => p.total_panel_loss > 0);
    
    // Đếm từng loại lỗi
    const stats = {
        "Hotspot": allPanels.filter(p => p.defects.some(d => d.type.toLowerCase().includes("hotspot"))).length,
        "Crack": allPanels.filter(p => p.defects.some(d => d.type.toLowerCase().includes("crack"))).length,
        "Soiling": allPanels.filter(p => p.defects.some(d => d.type.toLowerCase().includes("soil"))).length,
    };

    const rows = [
        { label: "Hotspot", value: stats.Hotspot, color: colors.danger },
        { label: "Soiling", value: stats.Soiling, color: colors.warning },
        { label: "Crack", value: stats.Crack, color: colors.purple },
    ];

    const totalFaults = faultyPanels.length;

    const handleDownloadPDF = () => {
        if (!batchId) return alert("Vui lòng tải lên dữ liệu và chạy AI ở màn hình Dashboard trước!");
        window.open(`http://127.0.0.1:8000/api/v1/download-report/${batchId}`, "_blank");
    };

    return (
        <div>
            <PageHeader title="Inspection Report" subtitle={`Fault summary for Batch #${batchId || 'N/A'}`} />
            
            <div style={{ display: "grid", gridTemplateColumns: "repeat(2,1fr)", gap: 16, marginBottom: 20 }}>
                <KpiCard icon={<AlertTriangle size={20} />} label="Total Faulty Panels" value={totalFaults} accent={colors.danger} />
                <KpiCard icon={<Zap size={20} />} label="Avg. Efficiency Loss" value={`${totalFaults > 0 ? (faultyPanels.reduce((a,b) => a + b.total_panel_loss, 0) / totalFaults).toFixed(1) : 0}%`} accent={colors.warning} />
            </div>

            <SolarCard style={{ marginBottom: 20 }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", paddingRight: 20 }}>
                    <CardHeader title="Fault Breakdown" />
                    <ActionButton onClick={handleDownloadPDF} icon={<Download size={15} />}>
                        Download Full PDF
                    </ActionButton>
                </div>
                <div style={{ display: "flex", flexDirection: "column", gap: 16, padding: "0 20px 20px" }}>
                    {rows.map(r => (
                        <div key={r.label}>
                            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
                                <span style={{ fontSize: 14, fontWeight: 500, color: "#0F172A" }}>{r.label}</span>
                                <span style={{ fontSize: 14, fontWeight: 700, color: r.color }}>{r.value} tấm</span>
                            </div>
                            <div style={{ height: 8, background: "#F1F5F9", borderRadius: 10, overflow: "hidden" }}>
                                <div style={{ height: "100%", width: `${totalFaults > 0 ? (r.value / totalFaults) * 100 : 0}%`, background: r.color, borderRadius: 10, transition: "width 1s ease" }} />
                            </div>
                        </div>
                    ))}
                    <div style={{ borderTop: "1px solid #F1F5F9", paddingTop: 14, display: "flex", justifyContent: "space-between" }}>
                        <span style={{ fontWeight: 600, color: "#0F172A" }}>Health Status</span>
                        <span style={{ fontWeight: 700, color: colors.primary }}>
                            {allPanels.length > 0 ? ((allPanels.length - totalFaults) / allPanels.length * 100).toFixed(1) : 0}% Healthy
                        </span>
                    </div>
                </div>
            </SolarCard>
        </div>
    );
}