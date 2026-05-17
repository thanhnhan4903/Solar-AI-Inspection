import React, { useState, useRef } from "react";
import { LayoutGrid, AlertCircle, DollarSign, Upload, Trash2, Loader2, Play, Cpu, CheckCircle, XCircle, AlertTriangle, Eye, X, ChevronRight } from "lucide-react";
import { colors } from "../../constants/theme";
import { PageHeader } from "../../components/layout/PageHeader";
import { KpiCard } from "../../components/ui/KpiCard";
import { SolarCard, CardHeader } from "../../components/ui/SolarCard";
import { ActionButton } from "../../components/ui/ActionButton";
import axios from "axios";

const API = "http://127.0.0.1:8000";

// ─────────────────────────────────────────
// Quality Review Modal
// ─────────────────────────────────────────
function QualityReviewModal({ qualityData, onConfirm, onCancel, isRunningAI }) {
    const { overall_status, quality_report = [] } = qualityData;
    const [expandedIdx, setExpandedIdx] = useState(null);

    const TIER = {
        ok:      { color: "#10b981", bg: "rgba(16,185,129,0.12)", icon: <CheckCircle size={14}/>, label: "Tốt" },
        warning: { color: "#f59e0b", bg: "rgba(245,158,11,0.12)",  icon: <AlertTriangle size={14}/>, label: "Cảnh báo" },
        poor:    { color: "#ef4444", bg: "rgba(239,68,68,0.12)",   icon: <XCircle size={14}/>,      label: "Kém" },
        error:   { color: "#64748b", bg: "rgba(100,116,139,0.12)", icon: <XCircle size={14}/>,      label: "Lỗi" },
    };

    const overallTier = TIER[overall_status] || TIER.warning;

    const counts = quality_report.reduce((acc, r) => {
        acc[r.quality_status] = (acc[r.quality_status] || 0) + 1;
        return acc;
    }, {});

    const headerMsg = {
        ok:      "✅ Tất cả ảnh đạt chất lượng tốt. Bạn có thể tiếp tục phân tích AI!",
        warning: "⚠️ Một số ảnh có dấu hiệu bất thường. Bạn vẫn có thể tiếp tục AI hoặc chụp lại để kết quả tốt hơn.",
        poor:    "🔴 Ảnh có chất lượng kém. Khuyến nghị chụp lại. Bạn vẫn có thể bỏ qua và tiếp tục AI.",
    }[overall_status] || "Xử lý hoàn tất. Kiểm tra chi tiết bên dưới.";

    return (
        <div style={{
            position: "fixed", inset: 0, zIndex: 9999,
            background: "rgba(2,8,23,0.85)", backdropFilter: "blur(8px)",
            display: "flex", alignItems: "center", justifyContent: "center",
            padding: 24,
        }}>
            <div style={{
                background: "linear-gradient(145deg, #0f172a, #1e293b)",
                border: `1px solid ${overallTier.color}40`,
                borderRadius: 20,
                boxShadow: `0 0 60px ${overallTier.color}20, 0 25px 50px rgba(0,0,0,0.6)`,
                width: "min(880px, 96vw)",
                maxHeight: "88vh",
                display: "flex", flexDirection: "column",
                overflow: "hidden",
            }}>
                {/* Header */}
                <div style={{
                    padding: "24px 28px 20px",
                    borderBottom: "1px solid rgba(255,255,255,0.07)",
                    background: `linear-gradient(135deg, ${overallTier.color}15, transparent)`,
                }}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
                        <div>
                            <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 8 }}>
                                <Eye size={20} color={overallTier.color} />
                                <span style={{ color: "#F8FAFC", fontWeight: 700, fontSize: 18 }}>
                                    Kết Quả Tiền Xử Lý Ảnh
                                </span>
                                <span style={{
                                    background: overallTier.bg,
                                    border: `1px solid ${overallTier.color}50`,
                                    color: overallTier.color,
                                    padding: "2px 10px", borderRadius: 20, fontSize: 12, fontWeight: 700,
                                }}>
                                    {overallTier.label.toUpperCase()}
                                </span>
                            </div>
                            <p style={{ color: "#94A3B8", fontSize: 13, margin: 0, maxWidth: 560 }}>
                                {headerMsg}
                            </p>
                        </div>
                        <button onClick={onCancel} disabled={isRunningAI}
                            style={{ background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.1)", color: "#94A3B8", borderRadius: 8, padding: "6px 10px", cursor: "pointer" }}>
                            <X size={16} />
                        </button>
                    </div>

                    {/* Count chips */}
                    <div style={{ display: "flex", gap: 10, marginTop: 14 }}>
                        {Object.entries(counts).map(([status, n]) => {
                            const t = TIER[status] || TIER.warning;
                            return (
                                <div key={status} style={{
                                    display: "flex", alignItems: "center", gap: 6,
                                    background: t.bg, border: `1px solid ${t.color}50`,
                                    borderRadius: 20, padding: "4px 12px", fontSize: 12, color: t.color, fontWeight: 600,
                                }}>
                                    {t.icon} {n} ảnh {t.label.toLowerCase()}
                                </div>
                            );
                        })}
                    </div>
                </div>

                {/* Image grid */}
                <div style={{ flex: 1, overflowY: "auto", padding: "20px 28px", minHeight: 0 }}>
                    {quality_report.map((item, idx) => {
                        const t = TIER[item.quality_status] || TIER.warning;
                        const isExpanded = expandedIdx === idx;
                        const m = item.metrics || {};

                        return (
                            <div key={idx} style={{
                                border: `1px solid ${t.color}35`,
                                borderRadius: 14,
                                background: "rgba(15,23,42,0.6)",
                                overflow: "hidden",
                                transition: "box-shadow 0.2s",
                                marginBottom: 12,
                                flexShrink: 0,
                            }}>
                                {/* Row */}
                                <div style={{ display: "flex", alignItems: "center", gap: 14, padding: "12px 16px", cursor: "pointer" }}
                                    onClick={() => setExpandedIdx(isExpanded ? null : idx)}>

                                    {/* Thumbnail */}
                                    <div style={{
                                        width: 80, height: 56, borderRadius: 8, overflow: "hidden",
                                        background: "#000", flexShrink: 0, border: `2px solid ${t.color}40`,
                                        position: "relative",
                                    }}>
                                        <img
                                            src={`${API}${item.preview_url}`}
                                            alt={item.filename}
                                            style={{ width: "100%", height: "100%", objectFit: "cover" }}
                                            onError={e => { e.target.style.display = "none"; }}
                                        />
                                    </div>

                                    {/* Info */}
                                    <div style={{ flex: 1, minWidth: 0 }}>
                                        <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
                                            <span style={{ color: "#F8FAFC", fontWeight: 600, fontSize: 13, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                                                {item.filename}
                                            </span>
                                            <span style={{
                                                display: "flex", alignItems: "center", gap: 4,
                                                background: t.bg, border: `1px solid ${t.color}50`,
                                                color: t.color, padding: "1px 8px", borderRadius: 12, fontSize: 11, fontWeight: 700, flexShrink: 0,
                                            }}>
                                                {t.icon} {t.label}
                                            </span>
                                        </div>
                                        {item.issues_vi.length > 0 ? (
                                            <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                                                {item.issues_vi.map((vi, i) => (
                                                    <span key={i} style={{ fontSize: 11, color: "#CBD5E1", background: "rgba(255,255,255,0.05)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 6, padding: "2px 8px" }}>
                                                        {vi}
                                                    </span>
                                                ))}
                                            </div>
                                        ) : (
                                            <span style={{ fontSize: 11, color: "#64748B" }}>Không phát hiện vấn đề</span>
                                        )}
                                    </div>

                                    {/* Expand toggle */}
                                    <ChevronRight size={16} color="#475569" style={{ transform: isExpanded ? "rotate(90deg)" : "none", transition: "0.2s", flexShrink: 0 }} />
                                </div>

                                {/* Expanded metrics */}
                                {isExpanded && (
                                    <div style={{ borderTop: "1px solid rgba(255,255,255,0.06)", padding: "14px 16px", display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(160px, 1fr))", gap: 10 }}>
                                        {[
                                            { label: "Blur Score", val: m.blur_score?.toFixed(1), good: m.blur_score >= 60, unit: "" },
                                            { label: "Noise Score", val: m.noise_score?.toFixed(1), good: m.noise_score <= 18, unit: "" },
                                            { label: "Viền đen", val: m.black_border_ratio !== undefined ? (m.black_border_ratio * 100).toFixed(1) : null, good: m.black_border_ratio <= 0.5, unit: "%" },
                                            { label: "Độ sáng TB", val: m.brightness_mean?.toFixed(1), good: true, unit: "" },
                                            { label: "Dải sáng (DR)", val: m.dynamic_range?.toFixed(1), good: m.dynamic_range >= 40, unit: "" },
                                            { label: "Quá sáng", val: m.high_saturation_ratio !== undefined ? (m.high_saturation_ratio * 100).toFixed(1) : null, good: m.high_saturation_ratio <= 0.15, unit: "%" },
                                        ].map(({ label, val, good, unit }) => val !== null && val !== undefined && (
                                            <div key={label} style={{ background: "rgba(255,255,255,0.04)", borderRadius: 8, padding: "8px 12px" }}>
                                                <div style={{ fontSize: 10, color: "#64748B", textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: 2 }}>{label}</div>
                                                <div style={{ fontSize: 15, fontWeight: 700, color: good ? "#10b981" : "#f59e0b", fontFamily: "monospace" }}>{val}{unit}</div>
                                            </div>
                                        ))}
                                    </div>
                                )}
                            </div>
                        );
                    })}
                </div>

                {/* Footer actions */}
                <div style={{
                    padding: "18px 28px",
                    borderTop: "1px solid rgba(255,255,255,0.07)",
                    display: "flex", justifyContent: "flex-end", gap: 12,
                    background: "rgba(15,23,42,0.5)",
                }}>
                    <button
                        onClick={onCancel}
                        disabled={isRunningAI}
                        style={{
                            padding: "10px 22px", borderRadius: 10,
                            background: "transparent", border: "1px solid rgba(255,255,255,0.15)",
                            color: "#94A3B8", fontSize: 14, fontWeight: 600, cursor: "pointer",
                            opacity: isRunningAI ? 0.5 : 1,
                        }}>
                        ✕ Hủy (Chụp Lại)
                    </button>
                    <button
                        onClick={onConfirm}
                        disabled={isRunningAI}
                        style={{
                            padding: "10px 28px", borderRadius: 10,
                            background: isRunningAI
                                ? "rgba(14,165,233,0.4)"
                                : "linear-gradient(135deg, #0EA5E9, #6366F1)",
                            border: "none", color: "#fff", fontSize: 14, fontWeight: 700,
                            cursor: isRunningAI ? "not-allowed" : "pointer",
                            display: "flex", alignItems: "center", gap: 8,
                            boxShadow: isRunningAI ? "none" : "0 4px 20px rgba(14,165,233,0.4)",
                            transition: "all 0.2s",
                        }}>
                        {isRunningAI
                            ? <><Loader2 size={16} className="animate-spin" /> Đang phân tích AI...</>
                            : <><Play size={14} fill="white" /> Chạy AI Phân Tích</>
                        }
                    </button>
                </div>
            </div>
        </div>
    );
}

// ─────────────────────────────────────────
// Home Page
// ─────────────────────────────────────────
export default function Home({ data, onAnalysisComplete, onReset }) {
    const [isUploading, setIsUploading] = useState(false);
    const [isRunningAI, setIsRunningAI] = useState(false);
    const [isResetting, setIsResetting] = useState(false);
    const [isUpdatingModel, setIsUpdatingModel] = useState(false);
    const [statusText, setStatusText] = useState("");
    const [qualityData, setQualityData] = useState(null); // triggers modal

    const fileInputRef = useRef(null);
    const folderInputRef = useRef(null);
    const modelInputRef = useRef(null);

    // ── Stats ──
    const allPanels = data?.flatMap(img => img.panels) || [];
    const totalPanels = allPanels.length;
    const faultyPanels = allPanels.filter(p => p.total_panel_loss > 0);
    const totalFaults = faultyPanels.length;
    const estimatedLoss = faultyPanels.reduce((sum, p) => sum + (p.total_panel_loss * 0.5), 0);

    // ── Bước 1: Upload + Tiền xử lý → mở modal ──
    const handleUploadFiles = async (e) => {
        const files = Array.from(e.target.files);
        if (files.length === 0) return;

        setIsUploading(true);
        try {
            setStatusText("Đang tải dữ liệu lên...");
            const formData = new FormData();
            files.forEach(file => formData.append("files", file));
            await axios.post(`${API}/api/v1/upload-drone-data`, formData);

            setStatusText("Đang tiền xử lý & kiểm tra chất lượng...");
            const res = await axios.get(`${API}/api/v1/process-thermal`);

            // Hiện modal với kết quả quality
            setQualityData(res.data);
        } catch (error) {
            alert("Lỗi: " + (error.response?.data?.detail || error.message));
        }
        setIsUploading(false);
        setStatusText("");
        if (fileInputRef.current) fileInputRef.current.value = "";
        if (folderInputRef.current) folderInputRef.current.value = "";
    };

    // ── Bước 2: Người dùng xác nhận → chạy AI ──
    const handleRunAI = async () => {
        setIsRunningAI(true);
        try {
            const analyzeForm = new FormData();
            const userStr = localStorage.getItem("user");
            if (userStr) {
                const user = JSON.parse(userStr);
                analyzeForm.append("user_id", user.id);
            }
            const res = await axios.post(`${API}/api/v1/analyze-all`, analyzeForm);
            if (onAnalysisComplete) {
                onAnalysisComplete(res.data.data, res.data.batch_id);
            }
            setQualityData(null);
            alert(`✅ Thành công! Đã phân tích xong ${res.data.data.length} ảnh.`);
        } catch (error) {
            alert("Lỗi AI: " + (error.response?.data?.detail || error.message));
        }
        setIsRunningAI(false);
    };

    const handleCancelModal = () => {
        if (!isRunningAI) setQualityData(null);
    };

    // ── Model update ──
    const handleUpdateModel = async (e) => {
        const file = e.target.files[0];
        if (!file) return;
        if (!file.name.endsWith('.pt')) {
            alert("Vui lòng chọn file định dạng .pt!");
            if (modelInputRef.current) modelInputRef.current.value = "";
            return;
        }
        setIsUpdatingModel(true);
        try {
            const formData = new FormData();
            formData.append("file", file);
            const res = await axios.post(`${API}/api/v1/update-ai-model`, formData);
            if (res.data.error) alert(res.data.error);
            else alert(res.data.message || "Đã tải trọng số AI mới thành công!");
        } catch (error) {
            alert("Lỗi khi tải trọng số: " + (error.response?.data?.detail || error.message));
        }
        setIsUpdatingModel(false);
        if (modelInputRef.current) modelInputRef.current.value = "";
    };

    // ── Reset ──
    const handleSystemReset = async () => {
        if (!window.confirm("Hành động này sẽ xóa sạch dữ liệu và Database. Bạn có chắc chắn?")) return;
        setIsResetting(true);
        try {
            await axios.post(`${API}/api/v1/reset-system`);
            if (onReset) onReset();
            alert("Hệ thống đã được đưa về trạng thái mặc định.");
        } catch (error) {
            alert("Không thể reset: " + error.message);
        }
        setIsResetting(false);
    };

    const isAnyLoading = isUploading || isRunningAI || isResetting || isUpdatingModel;

    return (
        <div>
            {/* Quality Review Modal */}
            {qualityData && (
                <QualityReviewModal
                    qualityData={qualityData}
                    onConfirm={handleRunAI}
                    onCancel={handleCancelModal}
                    isRunningAI={isRunningAI}
                />
            )}

            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-end", marginBottom: 24 }}>
                <PageHeader title="Dashboard" subtitle="Solar farm monitoring overview (Real-time AI Data)" />
                <div style={{ display: "flex", gap: 12 }}>
                    {/* Upload Files */}
                    <input
                        type="file"
                        accept=".zip,.rar,.jpg,.jpeg,.png,image/*"
                        multiple
                        ref={fileInputRef}
                        onChange={handleUploadFiles}
                        style={{ display: "none" }}
                    />
                    <ActionButton
                        onClick={() => fileInputRef.current?.click()}
                        disabled={isAnyLoading}
                        icon={isUploading ? <Loader2 className="animate-spin" size={16} /> : <Upload size={16} />}
                        style={{ background: "linear-gradient(135deg, #0EA5E9, #8B5CF6)", color: "white", border: "none" }}
                    >
                        {isUploading ? statusText : "Tải lên (File/Zip/Rar)"}
                    </ActionButton>

                    {/* Upload Folder */}
                    <input
                        type="file"
                        webkitdirectory=""
                        multiple
                        ref={folderInputRef}
                        onChange={handleUploadFiles}
                        style={{ display: "none" }}
                    />
                    <ActionButton
                        onClick={() => folderInputRef.current?.click()}
                        disabled={isAnyLoading}
                        icon={isUploading ? <Loader2 className="animate-spin" size={16} /> : <Upload size={16} />}
                        style={{ background: "linear-gradient(135deg, #10B981, #3B82F6)", color: "white", border: "none" }}
                    >
                        {isUploading ? statusText : "Tải Thư Mục"}
                    </ActionButton>

                    {/* Update Model */}
                    <input type="file" accept=".pt" ref={modelInputRef} onChange={handleUpdateModel} style={{ display: "none" }} />
                    <ActionButton
                        onClick={() => modelInputRef.current?.click()}
                        disabled={isAnyLoading}
                        icon={isUpdatingModel ? <Loader2 className="animate-spin" size={16} /> : <Cpu size={16} />}
                        style={{ background: "transparent", color: colors.primary, border: `1px solid ${colors.primary}50` }}
                    >
                        {isUpdatingModel ? "Đang Update AI..." : "Thay model AI"}
                    </ActionButton>

                    {/* Reset */}
                    <ActionButton
                        onClick={handleSystemReset}
                        disabled={isAnyLoading}
                        icon={isResetting ? <Loader2 className="animate-spin" size={16} /> : <Trash2 size={16} />}
                        style={{ background: "#fee2e2", color: colors.error, border: `1px solid ${colors.error}30` }}
                    >
                        Reset System
                    </ActionButton>
                </div>
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 16, marginBottom: 24 }}>
                <KpiCard icon={<LayoutGrid size={20} />} label="TỔNG SỐ HÌNH ẢNH" value={totalPanels.toLocaleString()} accent={colors.primary} />
                <KpiCard icon={<AlertCircle size={20} />} label="FAULTS DETECTED" value={totalFaults} accent={colors.danger} />
                <KpiCard icon={<DollarSign size={20} />} label="ESTIMATED LOSS ($)" value={`$${estimatedLoss.toFixed(2)}`} accent={colors.warning} />
            </div>

            <SolarCard>
                <CardHeader title="AI Analysis Breakdown" />
                <div style={{ padding: "0 20px 20px" }}>
                    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 10 }}>
                        <span style={{ fontSize: 14 }}>Hệ thống ổn định:</span>
                        <span style={{ fontWeight: "bold", color: colors.success }}>
                            {totalPanels > 0 ? ((totalPanels - totalFaults) / totalPanels * 100).toFixed(1) : 0}%
                        </span>
                    </div>
                    <div style={{ height: 12, background: "#f1f5f9", borderRadius: 10, overflow: "hidden" }}>
                        <div style={{
                            height: "100%",
                            width: `${totalPanels > 0 ? ((totalPanels - totalFaults) / totalPanels * 100) : 0}%`,
                            background: colors.success,
                            transition: "width 0.5s ease-in-out"
                        }} />
                    </div>
                    <p style={{ marginTop: 15, fontSize: 13, color: "#64748b" }}>
                        * Dữ liệu được cập nhật từ lần quét AI gần nhất (Batch ID: {totalPanels > 0 ? "Active" : "None"}).
                    </p>
                </div>
            </SolarCard>
        </div>
    );
}
