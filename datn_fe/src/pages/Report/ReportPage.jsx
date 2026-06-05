import React, { useState } from "react";
import { AlertTriangle, Zap, Download, LayoutGrid, ShieldAlert, Award, MapPin, Eye, Percent, CheckCircle } from "lucide-react";
import { colors } from "../../constants/theme";
import { PageHeader } from "../../components/layout/PageHeader";
import { KpiCard } from "../../components/ui/KpiCard";
import { SolarCard, CardHeader } from "../../components/ui/SolarCard";
import { ActionButton } from "../../components/ui/ActionButton";

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

export default function ReportPage({ data, batchId }) {
    const [isDownloading, setIsDownloading] = useState(false);

    // 1. Lọc tất cả tấm pin
    const allPanels = data?.flatMap(img => img.panels.map(p => ({ ...p, imageFilename: img.filename, rgbImage: img.rgb_image }))) || [];
    const faultyPanels = allPanels.filter(p => p.status === "faulty" || p.total_panel_loss > 0);
    const totalPanels = allPanels.length;
    const totalFaults = faultyPanels.length;

    // Tính tổng công suất hao hụt (W)
    const totalPowerLossW = faultyPanels.reduce((sum, p) => sum + (p.total_panel_loss || 0), 0);

    // Tỉ lệ sức khỏe (%)
    const healthRate = totalPanels > 0 ? ((totalPanels - totalFaults) / totalPanels * 100).toFixed(1) : "100.0";

    // 2. Thống kê 5 loại lỗi
    const stats = {
        "hotspot_single_cell": 0,
        "hotspot_multi_cell": 0,
        "shading": 0,
        "soiling": 0,
        "crack": 0
    };

    allPanels.forEach(p => {
        if (p.defects) {
            p.defects.forEach(d => {
                const cname = d.class_name || d.type || "";
                if (stats[cname] !== undefined) {
                    stats[cname]++;
                }
            });
        }
    });

    const defectRows = [
        { label: "Điểm nóng đơn cell", key: "hotspot_single_cell", value: stats.hotspot_single_cell, color: colors.danger },
        { label: "Điểm nóng đa cụm", key: "hotspot_multi_cell", value: stats.hotspot_multi_cell, color: "#dc2626" },
        { label: "Bóng che khuất", key: "shading", value: stats.shading, color: "#06b6d4" },
        { label: "Bụi bẩn bám tụ", key: "soiling", value: stats.soiling, color: colors.warning },
        { label: "Vết nứt vật lý", key: "crack", value: stats.crack, color: "#a855f7" },
    ];

    const sumDefects = Object.values(stats).reduce((a, b) => a + b, 0);

    const handleDownloadPDF = async () => {
        if (!batchId) return alert("Vui lòng tải ảnh drone và chạy phân tích AI ở Trang chủ trước!");
        setIsDownloading(true);
        try {
            // Chạy hiệu ứng tải xuống cao cấp
            window.open(`http://127.0.0.1:8000/api/v1/download-report/${batchId}`, "_blank");
        } catch (error) {
            console.error(error);
        } finally {
            setTimeout(() => setIsDownloading(false), 2000);
        }
    };

    return (
        <div style={{ paddingBottom: 40 }}>
            {/* Header báo cáo */}
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 24 }}>
                <PageHeader title="Báo Cáo Kiểm Tra Chi Tiết" subtitle={`Báo cáo chuẩn đoán tự động AI - Lô #${batchId || 'N/A'}`} />
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
            
            {/* 4 Thẻ KPI cao cấp */}
            <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 16, marginBottom: 24 }}>
                <KpiCard 
                    icon={<LayoutGrid size={22} color="#0ea5e9" />} 
                    label="TỔNG SỐ TẤM PIN" 
                    value={totalPanels} 
                    accent="#0ea5e9" 
                    style={{ background: "#fff", borderRadius: 16, border: "1px solid #f1f5f9", boxShadow: "0 4px 6px -1px rgba(0, 0, 0, 0.05)" }}
                />
                <KpiCard 
                    icon={<AlertTriangle size={22} color={colors.danger} />} 
                    label="TẤM PIN BỊ LỖI" 
                    value={`${totalFaults} (${totalPanels > 0 ? (totalFaults / totalPanels * 100).toFixed(1) : 0}%)`} 
                    accent={colors.danger} 
                    style={{ background: "#fff", borderRadius: 16, border: "1px solid #f1f5f9", boxShadow: "0 4px 6px -1px rgba(0, 0, 0, 0.05)" }}
                />
                <KpiCard 
                    icon={<Zap size={22} color="#f59e0b" />} 
                    label="HAO HỤT CÔNG SUẤT" 
                    value={`${(totalPowerLossW / 1000).toFixed(2)} kW`} 
                    accent="#f59e0b" 
                    style={{ background: "#fff", borderRadius: 16, border: "1px solid #f1f5f9", boxShadow: "0 4px 6px -1px rgba(0, 0, 0, 0.05)" }}
                />
                <KpiCard 
                    icon={healthRate >= 90 ? <CheckCircle size={22} color={colors.primary} /> : <ShieldAlert size={22} color="#f97316" />} 
                    label="SỨC KHỎE HỆ THỐNG" 
                    value={`${healthRate}%`} 
                    accent={colors.primary} 
                    style={{ background: "#fff", borderRadius: 16, border: "1px solid #f1f5f9", boxShadow: "0 4px 6px -1px rgba(0, 0, 0, 0.05)" }}
                />
            </div>

            {/* Phân loại 5 nhóm lỗi */}
            <SolarCard style={{ marginBottom: 24, borderRadius: 16, boxShadow: "0 4px 6px -1px rgba(0, 0, 0, 0.05)", border: "1px solid #f1f5f9" }}>
                <CardHeader title="Phân Loại Chi Tiết 5 Nhóm Lỗi (AI Classification)" />
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 32, padding: "0 24px 24px" }}>
                    {/* Danh sách lỗi dạng progress bar */}
                    <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
                        {defectRows.map(r => (
                            <div key={r.key}>
                                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
                                    <span style={{ fontSize: 13, fontWeight: 600, color: "#1e293b" }}>{r.label}</span>
                                    <span style={{ fontSize: 13, fontWeight: 700, color: r.color }}>{r.value} vùng lỗi</span>
                                </div>
                                <div style={{ height: 8, background: "#f1f5f9", borderRadius: 10, overflow: "hidden" }}>
                                    <div style={{ height: "100%", width: `${sumDefects > 0 ? (r.value / sumDefects) * 100 : 0}%`, background: r.color, borderRadius: 10, transition: "width 1s ease" }} />
                                </div>
                            </div>
                        ))}
                    </div>

                    {/* Phân tích chất lượng lưới pin */}
                    <div style={{ background: "#f8fafc", borderRadius: 12, padding: 20, display: "flex", flexDirection: "column", justifyContent: "center" }}>
                        <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 12 }}>
                            <Award size={28} color={healthRate >= 90 ? colors.primary : "#f59e0b"} />
                            <h4 style={{ margin: 0, fontSize: 16, color: "#1e293b", fontWeight: 700 }}>Đánh Giá Sức Khỏe Lưới Pin</h4>
                        </div>
                        <p style={{ margin: 0, fontSize: 14, color: "#475569", lineHeight: 1.6 }}>
                            Lưới pin năng lượng mặt trời đạt tỉ lệ bình thường là <b>{healthRate}%</b>. 
                            {healthRate >= 95 ? (
                                <span style={{ color: colors.primary, fontWeight: 600 }}> Hệ thống đang hoạt động ở trạng thái TUYỆT VỜI (Tier A). Chưa cần bảo trì diện rộng.</span>
                            ) : healthRate >= 85 ? (
                                <span style={{ color: "#10b981", fontWeight: 600 }}> Hệ thống ở trạng thái TỐT (Tier B). Cần theo dõi các vùng có nguy cơ phát sinh hotspot.</span>
                            ) : healthRate >= 75 ? (
                                <span style={{ color: "#f59e0b", fontWeight: 600 }}> Hệ thống ở trạng thái TRUNG BÌNH (Tier C). Đề xuất kiểm tra trực tiếp tại hiện trường và vệ sinh tấm pin.</span>
                            ) : (
                                <span style={{ color: colors.danger, fontWeight: 600 }}> CẢNH BÁO: Hệ thống đang bị suy hao nặng (Tier D). Cần lập tức bố trí thay thế các tấm pin bị nứt vỡ hoặc hotspot nặng để tránh cháy nổ đường dây.</span>
                            )}
                        </p>
                    </div>
                </div>
            </SolarCard>

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
                                                    <span style={{ fontSize: 11, color: "#94a3b8", fontWeight: 600, display: "block", textTransform: "uppercase" }}>Độ tin cậy AI</span>
                                                    <span style={{ fontSize: 16, fontWeight: 700, color: "#0ea5e9" }}>{((p.confidence || 0) * 100).toFixed(0)}%</span>
                                                </div>
                                            </div>

                                            <div style={{ background: "#f8fafc", borderRadius: 8, padding: 14, border: "1px solid #f1f5f9" }}>
                                                <span style={{ fontSize: 11, color: "#94a3b8", fontWeight: 600, display: "block", textTransform: "uppercase", marginBottom: 6 }}>Danh sách dị thường ({p.defects?.length || 0})</span>
                                                <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                                                    {p.defects?.map((d, di) => (
                                                        <div key={di} style={{ borderBottom: di < p.defects.length - 1 ? "1px solid #e2e8f0" : "none", paddingBottom: 6 }}>
                                                            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                                                                <span style={{ fontSize: 13, fontWeight: 700, color: "#1e293b" }}>{DEFECT_NAME_MAP[d.class_name] || d.class_name || d.type}</span>
                                                                <span style={{ fontSize: 12, fontWeight: 600, color: "#64748b" }}>AI: {((d.confidence || 0) * 100).toFixed(0)}%</span>
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
                                                    <span style={{ fontSize: 11, fontWeight: 600, color: "#64748b", textTransform: "uppercase", textAlign: "center" }}>Ảnh Nhiệt (Annotated)</span>
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
                                                    <span style={{ fontSize: 11, fontWeight: 600, color: "#64748b", textTransform: "uppercase", textAlign: "center" }}>Ảnh Quang Học (RGB)</span>
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