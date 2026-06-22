import React, { useState, useEffect } from "react";
import { AlertTriangle, Zap, Download, LayoutGrid, ShieldAlert, Award, MapPin, Eye, Percent, CheckCircle, TrendingUp } from "lucide-react";
import { colors } from "../../constants/theme";
import { PageHeader } from "../../components/layout/PageHeader";
import { KpiCard } from "../../components/ui/KpiCard";
import { SolarCard, CardHeader } from "../../components/ui/SolarCard";
import { ActionButton } from "../../components/ui/ActionButton";
import { downloadReportUrl, fetchLatestBatch } from "../../api";
import { normalizePanel, computeInspectionSummary } from "../../utils/inspectionData";

const DEFECT_NAME_MAP = {
    "hotspot_single_cell": "Điểm nóng đơn cell (Single Hotspot)",
    "hotspot_multi_cell": "Điểm nóng đa cụm (Multi Hotspot)",
    "shading": "Bóng che khuất (Shading)",
    "soiling": "Bụi bẩn bám tụ (Soiling)",
    "crack": "Vết nứt vật lý (Physical Crack)",
};

const SEVERITY_MAP = {
    "very_minor": "Rất nhẹ",
    "minor": "Nhẹ",
    "moderate": "Cần chú ý",
    "severe": "Ưu tiên bảo trì",
    "replace": "Cần thay thế"
};

const SEVERITY_COLORS = {
    "very_minor": "#94a3b8", // Slate
    "minor": "#eab308",      // Yellow
    "moderate": "#f97316",   // Orange
    "severe": "#ef4444",     // Red
    "replace": "#a855f7",    // Purple
};

const LOCATION_MAP = {
    "upper-left": "Trên - Trái",
    "upper-right": "Trên - Phải",
    "upper-center": "Trên - Giữa",
    "middle-left": "Giữa - Trái",
    "middle-right": "Giữa - Phải",
    "middle-center": "Chính Giữa",
    "lower-left": "Dưới - Trái",
    "lower-right": "Dưới - Phải",
    "lower-center": "Dưới - Giữa",
    "center": "Giữa",
    "left": "Trái",
    "right": "Phải",
    "upper": "Trên",
    "lower": "Dưới"
};

const REVIEW_STATUS_LABEL = {
    "confirmed_defect": { label: "Đúng có lỗi", color: "#ef4444", bg: "rgba(239,68,68,0.1)" },
    "needs_review":     { label: "Xem xét",     color: "#f59e0b", bg: "rgba(245,158,11,0.1)" },
    "unreviewed":       { label: "Chưa duyệt",  color: "#94a3b8", bg: "rgba(148,163,184,0.1)" },
    "false_positive":   { label: "Không phải lỗi", color: "#64748b", bg: "rgba(100,116,139,0.1)" },
};

function DonutChart({ data, totalLabel }) {
    if (!data || data.length === 0) {
        return (
            <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: 110, color: "#94a3b8", flexDirection: "column", gap: 6 }}>
                <TrendingUp size={22} style={{ opacity: 0.3 }} />
                <span style={{ fontSize: 11 }}>Chưa có dữ liệu</span>
            </div>
        );
    }

    const total = data.reduce((s, d) => s + d.value, 0) || 1;
    const radius = 42;
    const cx = 55, cy = 55;
    const strokeWidth = 14;
    const circumference = 2 * Math.PI * radius;

    let offset = 0;
    const segments = data.map(d => {
        const pct = d.value / total;
        const dash = pct * circumference;
        const gap = circumference - dash;
        const seg = { ...d, dash, gap, offset: offset * circumference, pct };
        offset += pct;
        return seg;
    });

    return (
        <svg width={110} height={110} viewBox="0 0 110 110">
            <circle cx={cx} cy={cy} r={radius} fill="none" stroke="#f1f5f9" strokeWidth={strokeWidth} />
            {segments.map((seg, i) => (
                <circle
                    key={i}
                    cx={cx} cy={cy} r={radius}
                    fill="none"
                    stroke={seg.color}
                    strokeWidth={strokeWidth}
                    strokeDasharray={`${seg.dash} ${seg.gap}`}
                    strokeDashoffset={circumference / 4 - seg.offset}
                    strokeLinecap="butt"
                    style={{ transition: "stroke-dashoffset 0.8s ease" }}
                />
            ))}
            <text x={cx} y={cy - 4} textAnchor="middle" style={{ fontSize: 14, fontWeight: 800, fill: "#1e293b" }}>
                {totalLabel}
            </text>
            <text x={cx} y={cy + 8} textAnchor="middle" style={{ fontSize: 9, fill: "#64748b" }}>
                tổng lỗi
            </text>
        </svg>
    );
}

export default function ReportPage({ data, batchId }) {
    const [isDownloading, setIsDownloading] = useState(false);
    const [projectMetadata, setProjectMetadata] = useState({
        projectName: "",
        location: "",
        scanTime: "",
        operator: "",
        device: "",
        panelPower: 600,
        systemCapacity: "",
        supervisor: "",
        aiModel: ""
    });

    useEffect(() => {
        const handler = () => {
            console.log("Review synced, ReportPage updated via props");
            if (batchId) {
                fetchLatestBatch().then(res => {
                    if (res.data) {
                        setProjectMetadata({
                            projectName: res.data.project_name || "",
                            location: res.data.location || "",
                            scanTime: res.data.scan_time || "",
                            operator: res.data.operator || "",
                            device: res.data.device || "",
                            panelPower: res.data.panel_power || 600,
                            systemCapacity: res.data.system_capacity || "",
                            supervisor: res.data.supervisor || "",
                            aiModel: res.data.ai_model || ""
                        });
                    }
                }).catch(console.error);
            }
        };
        window.addEventListener("review-sync-completed", handler);
        handler(); // Chạy luôn lần đầu tiên
        return () => window.removeEventListener("review-sync-completed", handler);
    }, [batchId]);

    // 1. Lọc tất cả tấm pin và chuẩn hóa
    const allPanels = data?.flatMap(img => (img.panels || []).map(p => {
        const normalizedP = normalizePanel(p);
        return {
            ...normalizedP,
            imageFilename: img.filename,
            rgbImage: img.rgb_image
        };
    })) || [];

    const summary = computeInspectionSummary(data || []);
    const totalPanels = summary.total_panels;
    const totalFaults = summary.faulty_panels;
    const totalPowerLossW = summary.total_power_loss_w;
    const healthRate = summary.normal_panel_ratio_percent.toFixed(1);

    const faultyPanels = allPanels.filter(p => p.status === "faulty");

    const confirmedFaults = allPanels.filter(p => p.review_status === "confirmed_defect").length;
    const falsePositives = allPanels.filter(p => p.review_status === "false_positive").length;

    // 2. Thống kê lỗi - đồng bộ với dashboard
    const classifyDefect = (rawName) => {
        const cls = (rawName || "").toLowerCase();
        if (cls.includes("hotspot_multi") || cls.includes("multi_cell") || cls.includes("multicell") || cls.includes("multi-cell")) return "hotspot multi cell";
        if (cls.includes("hotspot_single") || cls.includes("single_cell") || cls.includes("single-cell")) return "hotspot single cell";
        if (cls.includes("hotspot") || cls.includes("hot")) return "hotspot single cell";
        if (cls.includes("crack") || cls.includes("nut")) return "crack";
        if (cls.includes("shad") || cls.includes("shadow") || cls.includes("soil") || cls.includes("soiling") || cls.includes("dirt")) return "shading";
        if (cls.includes("diode")) return "diode";
        return rawName ? rawName.replace(/_/g, " ") : "khác";
    };

    const FAULT_COLORS = {
        "hotspot single cell": "#ef4444",   // Đỏ tươi — Hotspot đơn
        "hotspot multi cell":  "#ff6b35",   // Cam — Hotspot đa
        "crack":               "#f59e0b",   // Vàng cam — Crack
        "shading":             "#8b5cf6",   // Tím — Shading
        "diode":               "#06b6d4",   // Cyan — Diode
    };

    const FAULT_LABELS = {
        "hotspot single cell": "hotspot single cell",
        "hotspot multi cell":  "hotspot multi_cell",
        "crack":               "Crack",
        "shading":             "Shading",
        "diode":               "Diode",
    };

    const getFaultColor = (group) => FAULT_COLORS[group.toLowerCase()] || "#94a3b8";
    const getFaultLabel = (group) => FAULT_LABELS[group.toLowerCase()] || group;

    const defectCounts = {};
    (data || []).forEach((img) => {
        (img.panels || []).forEach((p) => {
            const reviewStatus = p.review_status || "unreviewed";
            const include = reviewStatus === "false_positive" ? false : (p.include_in_report !== undefined ? p.include_in_report : true);
            if (include) {
                (p.defects || []).forEach((d) => {
                    const group = classifyDefect(d.class_name || d.type || "");
                    defectCounts[group] = (defectCounts[group] || 0) + 1;
                });
            }
        });
    });

    const donutChartData = Object.entries(defectCounts)
        .sort((a, b) => b[1] - a[1])
        .map(([group, value]) => ({
            label: getFaultLabel(group),
            rawGroup: group,
            value,
            color: getFaultColor(group),
        }));

    const totalDefectsCount = donutChartData.reduce((s, d) => s + d.value, 0);

    const estimatedLossMWp = totalPowerLossW / 1000000;
    const estimatedLosskWp = totalPowerLossW / 1000;
    const powerLossValue = estimatedLossMWp >= 1 ? `${estimatedLossMWp.toFixed(2)} MWp` : `${estimatedLosskWp.toFixed(2)} kWp`;

    const handleDownloadPDF = async () => {
        setIsDownloading(true);
        try {
            window.open(downloadReportUrl(batchId || 0), "_blank");
        } catch (error) {
            console.error(error);
        } finally {
            setTimeout(() => setIsDownloading(false), 2000);
        }
    };

    return (
        <div style={{ paddingBottom: 40 }}>
            {/* Header báo cáo */}
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
                <PageHeader title="Báo Cáo Kiểm Tra Chi Tiết" />
                <ActionButton 
                    onClick={handleDownloadPDF} 
                    icon={isDownloading ? null : <Download size={16} />}
                    disabled={isDownloading}
                    style={{
                        background: isDownloading ? "#cbd5e1" : "linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%)",
                        boxShadow: "0 10px 15px -3px rgba(14, 165, 233, 0.3)",
                        padding: "12px 24px",
                        fontSize: 14,
                        fontWeight: 600,
                        color: "#fff",
                        borderRadius: 12,
                        transition: "all 0.3s ease"
                    }}
                >
                    {isDownloading ? "Đang xuất PDF..." : "Tải Báo Cáo PDF"}
                </ActionButton>
            </div>
            
            {/* 1. THÔNG TIN ĐỢT KIỂM TRA (Full-width) */}
            <SolarCard style={{ marginBottom: 12, borderRadius: 16, boxShadow: "0 4px 6px -1px rgba(0, 0, 0, 0.05)", border: "1px solid #f1f5f9" }}>
                <CardHeader title="Thông tin đợt kiểm tra" />
                <div style={{ padding: "0 24px 14px" }}>
                    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "8px 24px" }}>
                        <div>
                            <span style={{ fontSize: 11, color: "#94a3b8", fontWeight: 700, display: "block", textTransform: "uppercase", letterSpacing: "0.5px", marginBottom: 2 }}>Tên dự án</span>
                            <span style={{ fontSize: 13, color: "#334155", fontWeight: 600 }}>{projectMetadata.projectName || "—"}</span>
                        </div>
                        <div>
                            <span style={{ fontSize: 11, color: "#94a3b8", fontWeight: 700, display: "block", textTransform: "uppercase", letterSpacing: "0.5px", marginBottom: 2 }}>Địa điểm</span>
                            <span style={{ fontSize: 13, color: "#334155", fontWeight: 600 }}>{projectMetadata.location || "—"}</span>
                        </div>
                        <div>
                            <span style={{ fontSize: 11, color: "#94a3b8", fontWeight: 700, display: "block", textTransform: "uppercase", letterSpacing: "0.5px", marginBottom: 2 }}>Công suất pin</span>
                            <span style={{ fontSize: 13, color: "#334155", fontWeight: 600 }}>{projectMetadata.panelPower ? `${projectMetadata.panelPower} W` : "—"}</span>
                        </div>
                        <div>
                            <span style={{ fontSize: 11, color: "#94a3b8", fontWeight: 700, display: "block", textTransform: "uppercase", letterSpacing: "0.5px", marginBottom: 2 }}>Ngày kiểm tra</span>
                            <span style={{ fontSize: 13, color: "#334155", fontWeight: 600 }}>{projectMetadata.scanTime || "—"}</span>
                        </div>
                        <div>
                            <span style={{ fontSize: 11, color: "#94a3b8", fontWeight: 700, display: "block", textTransform: "uppercase", letterSpacing: "0.5px", marginBottom: 2 }}>Đơn vị quét</span>
                            <span style={{ fontSize: 13, color: "#334155", fontWeight: 600 }}>{projectMetadata.operator || "—"}</span>
                        </div>
                        <div>
                            <span style={{ fontSize: 11, color: "#94a3b8", fontWeight: 700, display: "block", textTransform: "uppercase", letterSpacing: "0.5px", marginBottom: 2 }}>Công suất hệ thống</span>
                            <span style={{ fontSize: 13, color: "#334155", fontWeight: 600 }}>{projectMetadata.systemCapacity || "—"}</span>
                        </div>
                        <div>
                            <span style={{ fontSize: 11, color: "#94a3b8", fontWeight: 700, display: "block", textTransform: "uppercase", letterSpacing: "0.5px", marginBottom: 2 }}>UAV</span>
                            <span style={{ fontSize: 13, color: "#334155", fontWeight: 600 }}>{projectMetadata.device || "—"}</span>
                        </div>
                        <div>
                            <span style={{ fontSize: 11, color: "#94a3b8", fontWeight: 700, display: "block", textTransform: "uppercase", letterSpacing: "0.5px", marginBottom: 2 }}>Người vận hành</span>
                            <span style={{ fontSize: 13, color: "#334155", fontWeight: 600 }}>{projectMetadata.supervisor || "—"}</span>
                        </div>
                        <div>
                            <span style={{ fontSize: 11, color: "#94a3b8", fontWeight: 700, display: "block", textTransform: "uppercase", letterSpacing: "0.5px", marginBottom: 2 }}>Model AI</span>
                            <span style={{ fontSize: 13, color: "#334155", fontWeight: 600 }}>{projectMetadata.aiModel || "—"}</span>
                        </div>
                    </div>
                </div>
            </SolarCard>

            {/* 2. HAI CỘT: TỔNG QUAN KẾT QUẢ & THỐNG KÊ LỖI THEO LOẠI */}
            <div style={{ display: "grid", gridTemplateColumns: "1.3fr 1fr", gap: 12, marginBottom: 12 }}>
                {/* Cột trái: Tổng quan kết quả */}
                <SolarCard style={{ borderRadius: 16, boxShadow: "0 4px 6px -1px rgba(0, 0, 0, 0.05)", border: "1px solid #f1f5f9", display: "flex", flexDirection: "column" }}>
                    <CardHeader title="Tổng quan kết quả" />
                    <div style={{ padding: "0 24px 16px", flex: 1, display: "flex", alignItems: "center" }}>
                        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 12, width: "100%" }}>
                            {[
                                { label: "Tổng số panel", value: totalPanels.toLocaleString(), color: "#2563eb", bg: "#f0f9ff", border: "#e0f2fe" },
                                { label: "Panel lỗi", value: totalFaults.toLocaleString(), color: "#ef4444", bg: "#fef2f2", border: "#fee2e2" },
                                { label: "Tỉ lệ lỗi", value: `${totalPanels > 0 ? (totalFaults / totalPanels * 100).toFixed(2) : 0}%`, color: "#ea580c", bg: "#fff7ed", border: "#ffedd5" },
                                { label: "Công suất ước tính", value: powerLossValue, color: "#10b981", bg: "#f0fdf4", border: "#dcfce7" },
                                { label: "Số lỗi đã duyệt", value: confirmedFaults.toLocaleString(), color: "#b91c1c", bg: "#fff5f5", border: "#ffe3e3" },
                                { label: "False Positive", value: falsePositives.toLocaleString(), color: "#16a34a", bg: "#f4fbf7", border: "#e6f7ed" }
                            ].map((item, i) => (
                                <div key={i} style={{ background: item.bg, border: `1px solid ${item.border}`, borderRadius: 10, padding: "12px 10px", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center" }}>
                                    <span style={{ fontSize: 10, color: "#64748b", fontWeight: 700, marginBottom: 4, textAlign: "center", textTransform: "uppercase", letterSpacing: "0.4px" }}>{item.label}</span>
                                    <span style={{ fontSize: 18, color: item.color, fontWeight: 800 }}>{item.value}</span>
                                </div>
                            ))}
                        </div>
                    </div>
                </SolarCard>

                {/* Cột phải: Thống kê lỗi theo loại */}
                <SolarCard style={{ borderRadius: 16, boxShadow: "0 4px 6px -1px rgba(0, 0, 0, 0.05)", border: "1px solid #f1f5f9", display: "flex", flexDirection: "column" }}>
                    <CardHeader title="Thống kê lỗi theo loại" />
                    <div style={{ padding: "0 24px 16px", flex: 1, display: "flex", alignItems: "center", gap: 20, justifyContent: "center" }}>
                        {/* Donut Chart */}
                        <div style={{ flexShrink: 0 }}>
                            <DonutChart data={donutChartData} totalLabel={totalDefectsCount.toLocaleString()} />
                        </div>
                        {/* Legend */}
                        <div style={{ display: "flex", flexDirection: "column", gap: 6, width: 200 }}>
                            {(() => {
                                const primaryGroups = ["hotspot single cell", "hotspot multi cell", "crack", "shading", "diode"];
                                const otherCount = Object.entries(defectCounts)
                                    .filter(([group]) => !primaryGroups.includes(group))
                                    .reduce((acc, [, val]) => acc + val, 0);

                                const legendItems = [
                                    { group: "hotspot single cell", label: "hotspot single cell", color: "#ef4444" },
                                    { group: "hotspot multi cell", label: "hotspot multi_cell", color: "#ff6b35" },
                                    { group: "crack", label: "Crack", color: "#f59e0b" },
                                    { group: "shading", label: "Shading", color: "#8b5cf6" },
                                    { group: "diode", label: "Diode", color: "#06b6d4" },
                                ];

                                if (otherCount > 0) {
                                    legendItems.push({ group: "khác", label: "Khác", color: "#94a3b8" });
                                }

                                return legendItems.map((d, i) => {
                                    const val = d.group === "khác" ? otherCount : (defectCounts[d.group] || 0);
                                    const pct = totalDefectsCount > 0 ? ((val / totalDefectsCount) * 100).toFixed(1) : "0.0";
                                    return (
                                        <div key={i} style={{ display: "flex", alignItems: "center", gap: 6 }}>
                                            <div style={{ width: 8, height: 8, borderRadius: "50%", background: d.color, flexShrink: 0 }} />
                                            <span style={{ fontSize: 12, color: "#475569", fontWeight: 600, flex: 1 }}>{d.label}</span>
                                            <span style={{ fontSize: 12, color: "#1e293b", fontWeight: 700 }}>{pct}% ({val})</span>
                                        </div>
                                    );
                                });
                            })()}
                        </div>
                    </div>
                </SolarCard>
            </div>

            {/* Bảng chi tiết các tấm pin lỗi kèm so sánh kẹp song song ảnh Thermal & RGB */}
            <SolarCard style={{ borderRadius: 16, boxShadow: "0 4px 6px -1px rgba(0, 0, 0, 0.05)", border: "1px solid #f1f5f9" }}>
                <CardHeader title={`Danh Sách Tấm Pin Bị Lỗi Chi Tiết (${totalFaults} Tấm)`} />
                <div style={{ padding: "0 24px 24px", display: "flex", flexDirection: "column", gap: 20 }}>
                    {totalFaults === 0 ? (
                        <div style={{ textAlign: "center", padding: "40px 0", color: "#64748b" }}>
                            <CheckCircle size={48} color={colors.primary} style={{ margin: "0 auto 16px", display: "block" }} />
                            <p style={{ margin: 0, fontSize: 16, fontWeight: 600 }}>Hệ thống hoàn hảo! Không phát hiện tấm pin lỗi nào.</p>
                        </div>
                    ) : (
                        faultyPanels.map((p, idx) => {
                            const [x1, y1, x2, y2] = p.bbox || [];
                            const hasBbox = x1 !== undefined && y1 !== undefined;

                            // GPS thực tế từ EXIF hoặc ước tính
                            const lat = p.gps_lat ? p.gps_lat.toFixed(6) : (10.8231 + p.row * 0.00015).toFixed(6);
                            const lng = p.gps_lng ? p.gps_lng.toFixed(6) : (106.6297 + p.col * 0.00025).toFixed(6);

                            return (
                                <div key={idx} style={{ 
                                    border: "1px solid #e2e8f0", 
                                    borderRadius: 12, 
                                    overflow: "hidden",
                                    background: "#fff",
                                    transition: "all 0.2s"
                                }}>
                                    {/* Sub-header của tấm pin */}
                                    <div style={{ 
                                        background: "#f8fafc", 
                                        padding: "12px 20px", 
                                        borderBottom: "1px solid #e2e8f0",
                                        display: "flex",
                                        justifyContent: "space-between",
                                        alignItems: "center"
                                    }}>
                                        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
                                            <span style={{ 
                                                fontWeight: 700, 
                                                fontSize: 15, 
                                                color: colors.danger,
                                                background: "#fef2f2",
                                                padding: "4px 10px",
                                                borderRadius: 6,
                                                border: "1px solid #fee2e2"
                                            }}>
                                                Tấm {p.local_id}
                                            </span>
                                            <span style={{ fontSize: 13, color: "#64748b", fontWeight: 500 }}>
                                                Hàng {p.row} · Cột {p.col}
                                            </span>
                                            <div style={{ display: "flex", alignItems: "center", gap: 4, color: "#0ea5e9", fontSize: 12, fontWeight: 600 }}>
                                                <MapPin size={14} /> GPS: {lat}, {lng}
                                            </div>
                                        </div>
                                        
                                        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                                            <span style={{ 
                                                background: `${SEVERITY_COLORS[p.worst_severity] || "#94a3b8"}15`,
                                                border: `1px solid ${SEVERITY_COLORS[p.worst_severity] || "#94a3b8"}30`,
                                                color: SEVERITY_COLORS[p.worst_severity] || "#94a3b8",
                                                padding: "4px 12px",
                                                borderRadius: 20,
                                                fontSize: 12,
                                                fontWeight: 700,
                                                textTransform: "uppercase"
                                            }}>
                                                {SEVERITY_MAP[p.worst_severity] || "Không rõ"}
                                            </span>
                                            {/* Review status badge */}
                                            {(() => {
                                                const rs = p.review_status || "unreviewed";
                                                const rsInfo = REVIEW_STATUS_LABEL[rs] || REVIEW_STATUS_LABEL["unreviewed"];
                                                return (
                                                    <span style={{
                                                        background: rsInfo.bg,
                                                        border: `1px solid ${rsInfo.color}40`,
                                                        color: rsInfo.color,
                                                        padding: "4px 10px",
                                                        borderRadius: 20,
                                                        fontSize: 11,
                                                        fontWeight: 700,
                                                    }}>
                                                        {rsInfo.label}
                                                    </span>
                                                );
                                            })()}
                                        </div>
                                    </div>

                                    {/* Nội dung chi tiết */}
                                    <div style={{ display: "grid", gridTemplateColumns: "1.1fr 1fr", gap: 24, padding: 20 }}>
                                        {/* Chi tiết thông tin lỗi */}
                                        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                                            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
                                                <div style={{ background: "#f8fafc", borderRadius: 8, padding: "10px 14px", border: "1px solid #f1f5f9" }}>
                                                    <span style={{ fontSize: 11, color: "#94a3b8", fontWeight: 600, display: "block", textTransform: "uppercase" }}>Hao hụt công suất</span>
                                                    <span style={{ fontSize: 16, fontWeight: 700, color: colors.danger }}>{p.total_panel_loss} W</span>
                                                </div>
                                                <div style={{ background: "#f8fafc", borderRadius: 8, padding: "10px 14px", border: "1px solid #f1f5f9" }}>
                                                    <span style={{ fontSize: 11, color: "#94a3b8", fontWeight: 600, display: "block", textTransform: "uppercase" }}>Độ tin cậy YOLO (Panel)</span>
                                                    <span style={{ fontSize: 16, fontWeight: 700, color: "#0ea5e9" }}>{p.confidence != null ? `${(p.confidence * 100).toFixed(1)}%` : "Chưa có dữ liệu"}</span>
                                                </div>
                                            </div>

                                            <div style={{ background: "#f8fafc", borderRadius: 8, padding: 14, border: "1px solid #f1f5f9" }}>
                                                <span style={{ fontSize: 11, color: "#94a3b8", fontWeight: 600, display: "block", textTransform: "uppercase", marginBottom: 6 }}>Danh sách dị thường ({p.defects?.length || 0})</span>
                                                <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                                                    {p.defects?.map((d, di) => (
                                                        <div key={di} style={{ borderBottom: di < p.defects.length - 1 ? "1px solid #e2e8f0" : "none", paddingBottom: 6 }}>
                                                            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                                                                {(() => {
                                                                    const cname = (d.class_name || d.type || "").toLowerCase();
                                                                    let displayName = "shading";
                                                                    if (cname.includes("single")) displayName = "hotspot_single_cell";
                                                                    else if (cname.includes("multi")) displayName = "hotspot_multi_cell";
                                                                    else if (cname.includes("crack")) displayName = "crack";
                                                                    else if (cname.includes("shading") || cname.includes("soil") || cname.includes("soiling") || cname.includes("dirt") || cname.includes("shade") || cname.includes("shadow")) displayName = "shading";
                                                                    return <span style={{ fontSize: 13, fontWeight: 700, color: "#1e293b" }}>{displayName}</span>;
                                                                })()}
                                                                <span style={{ fontSize: 12, fontWeight: 600, color: "#64748b" }}>
                                                                    Độ tin cậy YOLO: {d.confidence != null ? `${(d.confidence * 100).toFixed(1)}%` : "Chưa có dữ liệu"}
                                                                </span>
                                                            </div>
                                                            <div style={{ display: "flex", gap: 12, fontSize: 12, color: "#64748b", marginTop: 2 }}>
                                                                <span>Vị trí trong tấm: <b style={{ color: "#3b82f6" }}>{LOCATION_MAP[d.location_in_panel] || d.location_in_panel}</b></span>
                                                                {d.relative_position?.u !== undefined && (
                                                                    <span>Tọa độ u,v: <b>({d.relative_position.u.toFixed(2)}, {d.relative_position.v.toFixed(2)})</b></span>
                                                                )}
                                                            </div>
                                                        </div>
                                                    ))}
                                                </div>
                                            </div>

                                            <div style={{ fontSize: 12, color: "#64748b", display: "flex", flexDirection: "column", gap: 4 }}>
                                                <div>Ảnh gốc nhiệt: <b>{p.imageFilename}</b></div>
                                                <div>Ảnh quang học đối chiếu: <b>{p.rgbImage || "Không tìm thấy ảnh ghép cặp"}</b></div>
                                                <div>Khuyến nghị: <b style={{ color: "#0ea5e9" }}>{p.recommendation}</b></div>
                                            </div>
                                        </div>

                                        {/* Kẹp song song hai ảnh đã gán lỗi */}
                                        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                                            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
                                                {/* Ảnh nhiệt Crop */}
                                                <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                                                    <span style={{ fontSize: 11, fontWeight: 600, color: "#64748b", textTransform: "uppercase", textAlign: "center" }}>Ảnh nhiệt đã chú thích</span>
                                                    <div style={{ border: "2px solid #fee2e2", borderRadius: 8, overflow: "hidden", aspectRatio: "5/4", background: "#000" }}>
                                                        {hasBbox ? (
                                                            <img 
                                                                src={`http://127.0.0.1:8000/api/v1/panel-image?filename=${p.imageFilename}&x1=${x1}&y1=${y1}&x2=${x2}&y2=${y2}&polygon=${p.polygon ? p.polygon.join(',') : ''}`}
                                                                style={{ width: "100%", height: "100%", objectFit: "cover" }}
                                                                alt={`Thermal ${p.local_id}`}
                                                                onError={(e) => { e.target.src = "https://images.unsplash.com/photo-1508514177221-188b1cf16e9d?auto=format&fit=crop&w=300&q=80"; }}
                                                            />
                                                        ) : (
                                                            <div style={{ width: "100%", height: "100%", display: "flex", alignItems: "center", justifyContent: "center", color: "#64748b", fontSize: 12 }}>Không có khung tọa độ</div>
                                                        )}
                                                    </div>
                                                </div>

                                                {/* Ảnh RGB Crop */}
                                                <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                                                    <span style={{ fontSize: 11, fontWeight: 600, color: "#64748b", textTransform: "uppercase", textAlign: "center" }}>Ảnh RGB / ảnh bối cảnh</span>
                                                    <div style={{ border: "2px solid #dcfce7", borderRadius: 8, overflow: "hidden", aspectRatio: "5/4", background: "#000" }}>
                                                        {hasBbox && p.rgbImage ? (
                                                            <img 
                                                                src={`http://127.0.0.1:8000/api/v1/panel-image?filename=${p.rgbImage}&x1=${x1}&y1=${y1}&x2=${x2}&y2=${y2}`}
                                                                style={{ width: "100%", height: "100%", objectFit: "cover" }}
                                                                alt={`RGB ${p.local_id}`}
                                                                onError={(e) => { e.target.src = "https://images.unsplash.com/photo-1508514177221-188b1cf16e9d?auto=format&fit=crop&w=300&q=80"; }}
                                                            />
                                                        ) : (
                                                            <div style={{ width: "100%", height: "100%", display: "flex", alignItems: "center", justifyContent: "center", color: "#64748b", fontSize: 12 }}>Không có ảnh đối chiếu</div>
                                                        )}
                                                    </div>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            );
                        })
                    )}
                </div>
            </SolarCard>
        </div>
    );
}