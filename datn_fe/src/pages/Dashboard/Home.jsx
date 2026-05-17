import React, { useState, useRef } from "react";
import {
    LayoutGrid,
    AlertCircle,
    DollarSign,
    Upload,
    Trash2,
    Loader2,
    Play,
    Cpu,
    CheckCircle,
    XCircle,
    AlertTriangle,
    Eye,
    X,
    ChevronRight,
    TrendingUp,
} from "lucide-react";
import { colors } from "../../constants/theme";
import { PageHeader } from "../../components/layout/PageHeader";
import { KpiCard } from "../../components/ui/KpiCard";
import { ActionButton } from "../../components/ui/ActionButton";
import axios from "axios";
import solarFarmAerial from "../../assets/solar_farm_aerial.png";

const API = "http://127.0.0.1:8000";

// ─────────────────────────────────────────
// Mini bar chart component
// ─────────────────────────────────────────
function AnomalyBarChart({ data }) {
    if (!data || data.length === 0) {
        return (
            <div
                style={{
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    height: "100%",
                    flexDirection: "column",
                    gap: 12,
                    color: "#94a3b8",
                }}
            >
                <TrendingUp size={40} style={{ opacity: 0.3 }} />
                <p style={{ margin: 0, fontSize: 13 }}>No analysis data available</p>
            </div>
        );
    }

    const max = Math.max(...data.map((d) => d.value), 1);
    const barColors = [
        "#f97316",
        "#ef4444",
        "#06b6d4",
        "#eab308",
        "#a855f7",
        "#10b981",
        "#0ea5e9",
        "#f59e0b",
    ];

    return (
        <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>
            <div
                style={{
                    flex: 1,
                    display: "flex",
                    alignItems: "flex-end",
                    gap: 10,
                    padding: "8px 0",
                }}
            >
                {data.map((item, i) => {
                    const pct = (item.value / max) * 100;

                    return (
                        <div
                            key={i}
                            style={{
                                flex: 1,
                                display: "flex",
                                flexDirection: "column",
                                alignItems: "center",
                                gap: 4,
                            }}
                        >
                            <span style={{ fontSize: 11, fontWeight: 700, color: "#1e293b" }}>
                                {item.value}
                            </span>
                            <div
                                style={{
                                    width: "100%",
                                    height: `${Math.max(pct * 1.4, 4)}px`,
                                    background: `linear-gradient(180deg, ${barColors[i % barColors.length]}, ${barColors[i % barColors.length]}bb)`,
                                    borderRadius: "4px 4px 0 0",
                                    transition: "height 0.5s ease",
                                    minHeight: 4,
                                }}
                            />
                        </div>
                    );
                })}
            </div>

            <div style={{ display: "flex", gap: 10, borderTop: "2px solid #e2e8f0", paddingTop: 6 }}>
                {data.map((item, i) => (
                    <div
                        key={i}
                        style={{
                            flex: 1,
                            textAlign: "center",
                            fontSize: 9,
                            color: "#64748b",
                            wordBreak: "break-word",
                            lineHeight: 1.2,
                        }}
                    >
                        {item.label}
                    </div>
                ))}
            </div>
        </div>
    );
}

// ─────────────────────────────────────────
// Quality Review Modal
// ─────────────────────────────────────────
function QualityReviewModal({ qualityData, onConfirm, onCancel, isRunningAI }) {
    const { overall_status, quality_report = [] } = qualityData;
    const [expandedIdx, setExpandedIdx] = useState(null);

    const TIER = {
        ok: {
            color: "#10b981",
            bg: "rgba(16,185,129,0.12)",
            icon: <CheckCircle size={14} />,
            label: "Tốt",
        },
        warning: {
            color: "#f59e0b",
            bg: "rgba(245,158,11,0.12)",
            icon: <AlertTriangle size={14} />,
            label: "Cảnh báo",
        },
        poor: {
            color: "#ef4444",
            bg: "rgba(239,68,68,0.12)",
            icon: <XCircle size={14} />,
            label: "Kém",
        },
        error: {
            color: "#64748b",
            bg: "rgba(100,116,139,0.12)",
            icon: <XCircle size={14} />,
            label: "Lỗi",
        },
    };

    const overallTier = TIER[overall_status] || TIER.warning;

    const counts = quality_report.reduce((acc, r) => {
        acc[r.quality_status] = (acc[r.quality_status] || 0) + 1;
        return acc;
    }, {});

    const headerMsg =
        {
            ok: "✅ Tất cả ảnh đạt chất lượng tốt. Bạn có thể tiếp tục phân tích AI!",
            warning:
                "⚠️ Một số ảnh có dấu hiệu bất thường. Bạn vẫn có thể tiếp tục AI hoặc chụp lại để kết quả tốt hơn.",
            poor: "🔴 Ảnh có chất lượng kém. Khuyến nghị chụp lại. Bạn vẫn có thể bỏ qua và tiếp tục AI.",
        }[overall_status] || "Xử lý hoàn tất. Kiểm tra chi tiết bên dưới.";

    return (
        <div
            style={{
                position: "fixed",
                inset: 0,
                zIndex: 9999,
                background: "rgba(2,8,23,0.85)",
                backdropFilter: "blur(8px)",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                padding: 24,
            }}
        >
            <div
                style={{
                    background: "linear-gradient(145deg, #0f172a, #1e293b)",
                    border: `1px solid ${overallTier.color}40`,
                    borderRadius: 20,
                    boxShadow: `0 0 60px ${overallTier.color}20, 0 25px 50px rgba(0,0,0,0.6)`,
                    width: "min(880px, 96vw)",
                    maxHeight: "88vh",
                    display: "flex",
                    flexDirection: "column",
                    overflow: "hidden",
                }}
            >
                <div
                    style={{
                        padding: "24px 28px 20px",
                        borderBottom: "1px solid rgba(255,255,255,0.07)",
                        background: `linear-gradient(135deg, ${overallTier.color}15, transparent)`,
                    }}
                >
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
                        <div>
                            <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 8 }}>
                                <Eye size={20} color={overallTier.color} />
                                <span style={{ color: "#F8FAFC", fontWeight: 700, fontSize: 18 }}>
                                    Kết Quả Tiền Xử Lý Ảnh
                                </span>
                                <span
                                    style={{
                                        background: overallTier.bg,
                                        border: `1px solid ${overallTier.color}50`,
                                        color: overallTier.color,
                                        padding: "2px 10px",
                                        borderRadius: 20,
                                        fontSize: 12,
                                        fontWeight: 700,
                                    }}
                                >
                                    {overallTier.label.toUpperCase()}
                                </span>
                            </div>
                            <p style={{ color: "#94A3B8", fontSize: 13, margin: 0, maxWidth: 560 }}>
                                {headerMsg}
                            </p>
                        </div>

                        <button
                            onClick={onCancel}
                            disabled={isRunningAI}
                            style={{
                                background: "rgba(255,255,255,0.06)",
                                border: "1px solid rgba(255,255,255,0.1)",
                                color: "#94A3B8",
                                borderRadius: 8,
                                padding: "6px 10px",
                                cursor: "pointer",
                            }}
                        >
                            <X size={16} />
                        </button>
                    </div>

                    <div style={{ display: "flex", gap: 10, marginTop: 14 }}>
                        {Object.entries(counts).map(([status, n]) => {
                            const t = TIER[status] || TIER.warning;

                            return (
                                <div
                                    key={status}
                                    style={{
                                        display: "flex",
                                        alignItems: "center",
                                        gap: 6,
                                        background: t.bg,
                                        border: `1px solid ${t.color}50`,
                                        borderRadius: 20,
                                        padding: "4px 12px",
                                        fontSize: 12,
                                        color: t.color,
                                        fontWeight: 600,
                                    }}
                                >
                                    {t.icon} {n} ảnh {t.label.toLowerCase()}
                                </div>
                            );
                        })}
                    </div>
                </div>

                <div style={{ flex: 1, overflowY: "auto", padding: "20px 28px", minHeight: 0 }}>
                    {quality_report.map((item, idx) => {
                        const t = TIER[item.quality_status] || TIER.warning;
                        const isExpanded = expandedIdx === idx;
                        const m = item.metrics || {};

                        return (
                            <div
                                key={idx}
                                style={{
                                    border: `1px solid ${t.color}35`,
                                    borderRadius: 14,
                                    background: "rgba(15,23,42,0.6)",
                                    overflow: "hidden",
                                    transition: "box-shadow 0.2s",
                                    marginBottom: 12,
                                    flexShrink: 0,
                                }}
                            >
                                <div
                                    style={{
                                        display: "flex",
                                        alignItems: "center",
                                        gap: 14,
                                        padding: "12px 16px",
                                        cursor: "pointer",
                                    }}
                                    onClick={() => setExpandedIdx(isExpanded ? null : idx)}
                                >
                                    <div
                                        style={{
                                            width: 80,
                                            height: 56,
                                            borderRadius: 8,
                                            overflow: "hidden",
                                            background: "#000",
                                            flexShrink: 0,
                                            border: `2px solid ${t.color}40`,
                                            position: "relative",
                                        }}
                                    >
                                        <img
                                            src={`${API}${item.preview_url}`}
                                            alt={item.filename}
                                            style={{ width: "100%", height: "100%", objectFit: "cover" }}
                                            onError={(e) => {
                                                e.target.style.display = "none";
                                            }}
                                        />
                                    </div>

                                    <div style={{ flex: 1, minWidth: 0 }}>
                                        <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
                                            <span
                                                style={{
                                                    color: "#F8FAFC",
                                                    fontWeight: 600,
                                                    fontSize: 13,
                                                    overflow: "hidden",
                                                    textOverflow: "ellipsis",
                                                    whiteSpace: "nowrap",
                                                }}
                                            >
                                                {item.filename}
                                            </span>
                                            <span
                                                style={{
                                                    display: "flex",
                                                    alignItems: "center",
                                                    gap: 4,
                                                    background: t.bg,
                                                    border: `1px solid ${t.color}50`,
                                                    color: t.color,
                                                    padding: "1px 8px",
                                                    borderRadius: 12,
                                                    fontSize: 11,
                                                    fontWeight: 700,
                                                    flexShrink: 0,
                                                }}
                                            >
                                                {t.icon} {t.label}
                                            </span>
                                        </div>

                                        {item.issues_vi?.length > 0 ? (
                                            <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                                                {item.issues_vi.map((vi, i) => (
                                                    <span
                                                        key={i}
                                                        style={{
                                                            fontSize: 11,
                                                            color: "#CBD5E1",
                                                            background: "rgba(255,255,255,0.05)",
                                                            border: "1px solid rgba(255,255,255,0.1)",
                                                            borderRadius: 6,
                                                            padding: "2px 8px",
                                                        }}
                                                    >
                                                        {vi}
                                                    </span>
                                                ))}
                                            </div>
                                        ) : (
                                            <span style={{ fontSize: 11, color: "#64748B" }}>
                                                Không phát hiện vấn đề
                                            </span>
                                        )}
                                    </div>

                                    <ChevronRight
                                        size={16}
                                        color="#475569"
                                        style={{
                                            transform: isExpanded ? "rotate(90deg)" : "none",
                                            transition: "0.2s",
                                            flexShrink: 0,
                                        }}
                                    />
                                </div>

                                {isExpanded && (
                                    <div
                                        style={{
                                            borderTop: "1px solid rgba(255,255,255,0.06)",
                                            padding: "14px 16px",
                                            display: "grid",
                                            gridTemplateColumns: "repeat(auto-fill, minmax(160px, 1fr))",
                                            gap: 10,
                                        }}
                                    >
                                        {[
                                            {
                                                label: "Blur Score",
                                                val: m.blur_score?.toFixed(1),
                                                good: m.blur_score >= 60,
                                                unit: "",
                                            },
                                            {
                                                label: "Noise Score",
                                                val: m.noise_score?.toFixed(1),
                                                good: m.noise_score <= 18,
                                                unit: "",
                                            },
                                            {
                                                label: "Viền đen",
                                                val:
                                                    m.black_border_ratio !== undefined
                                                        ? (m.black_border_ratio * 100).toFixed(1)
                                                        : null,
                                                good: m.black_border_ratio <= 0.5,
                                                unit: "%",
                                            },
                                            {
                                                label: "Độ sáng TB",
                                                val: m.brightness_mean?.toFixed(1),
                                                good: true,
                                                unit: "",
                                            },
                                            {
                                                label: "Dải sáng (DR)",
                                                val: m.dynamic_range?.toFixed(1),
                                                good: m.dynamic_range >= 40,
                                                unit: "",
                                            },
                                            {
                                                label: "Quá sáng",
                                                val:
                                                    m.high_saturation_ratio !== undefined
                                                        ? (m.high_saturation_ratio * 100).toFixed(1)
                                                        : null,
                                                good: m.high_saturation_ratio <= 0.15,
                                                unit: "%",
                                            },
                                        ].map(
                                            ({ label, val, good, unit }) =>
                                                val !== null &&
                                                val !== undefined && (
                                                    <div
                                                        key={label}
                                                        style={{
                                                            background: "rgba(255,255,255,0.04)",
                                                            borderRadius: 8,
                                                            padding: "8px 12px",
                                                        }}
                                                    >
                                                        <div
                                                            style={{
                                                                fontSize: 10,
                                                                color: "#64748B",
                                                                textTransform: "uppercase",
                                                                letterSpacing: "0.05em",
                                                                marginBottom: 2,
                                                            }}
                                                        >
                                                            {label}
                                                        </div>
                                                        <div
                                                            style={{
                                                                fontSize: 15,
                                                                fontWeight: 700,
                                                                color: good ? "#10b981" : "#f59e0b",
                                                                fontFamily: "monospace",
                                                            }}
                                                        >
                                                            {val}
                                                            {unit}
                                                        </div>
                                                    </div>
                                                )
                                        )}
                                    </div>
                                )}
                            </div>
                        );
                    })}
                </div>

                <div
                    style={{
                        padding: "18px 28px",
                        borderTop: "1px solid rgba(255,255,255,0.07)",
                        display: "flex",
                        justifyContent: "flex-end",
                        gap: 12,
                        background: "rgba(15,23,42,0.5)",
                    }}
                >
                    <button
                        onClick={onCancel}
                        disabled={isRunningAI}
                        style={{
                            padding: "10px 22px",
                            borderRadius: 10,
                            background: "transparent",
                            border: "1px solid rgba(255,255,255,0.15)",
                            color: "#94A3B8",
                            fontSize: 14,
                            fontWeight: 600,
                            cursor: "pointer",
                            opacity: isRunningAI ? 0.5 : 1,
                        }}
                    >
                        ✕ Hủy (Chụp Lại)
                    </button>
                    <button
                        onClick={onConfirm}
                        disabled={isRunningAI}
                        style={{
                            padding: "10px 28px",
                            borderRadius: 10,
                            background: isRunningAI
                                ? "rgba(14,165,233,0.4)"
                                : "linear-gradient(135deg, #0EA5E9, #6366F1)",
                            border: "none",
                            color: "#fff",
                            fontSize: 14,
                            fontWeight: 700,
                            cursor: isRunningAI ? "not-allowed" : "pointer",
                            display: "flex",
                            alignItems: "center",
                            gap: 8,
                            boxShadow: isRunningAI ? "none" : "0 4px 20px rgba(14,165,233,0.4)",
                            transition: "all 0.2s",
                        }}
                    >
                        {isRunningAI ? (
                            <>
                                <Loader2 size={16} className="animate-spin" /> Đang phân tích AI...
                            </>
                        ) : (
                            <>
                                <Play size={14} fill="white" /> Chạy AI Phân Tích
                            </>
                        )}
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
    const [qualityData, setQualityData] = useState(null);

    const fileInputRef = useRef(null);
    const folderInputRef = useRef(null);
    const modelInputRef = useRef(null);

    const allPanels = data?.flatMap((img) => img.panels || []) || [];
    const totalPanels = allPanels.length;
    const faultyPanels = allPanels.filter((p) => p.total_panel_loss > 0 || p.status === "faulty");
    const totalFaults = faultyPanels.length;
    const estimatedLoss = faultyPanels.reduce((sum, p) => sum + (Number(p.total_panel_loss || 0) * 0.5), 0);
    const healthyRate = totalPanels > 0 ? ((totalPanels - totalFaults) / totalPanels) * 100 : 0;

    const defectCounts = {};
    allPanels.forEach((p) => {
        (p.defects || []).forEach((d) => {
            const name = (d.class_name || d.type || "unknown").replace(/_/g, " ");
            defectCounts[name] = (defectCounts[name] || 0) + 1;
        });
    });

    const chartData = Object.entries(defectCounts)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 8)
        .map(([label, value]) => ({ label, value }));

    // Bước 1: Upload + tiền xử lý, sau đó mở modal quality review
    const handleUploadFiles = async (e) => {
        const files = Array.from(e.target.files || []);
        if (files.length === 0) return;

        setIsUploading(true);

        try {
            setStatusText("Đang tải dữ liệu lên...");
            const formData = new FormData();
            files.forEach((file) => formData.append("files", file));

            await axios.post(`${API}/api/v1/upload-drone-data`, formData);

            setStatusText("Đang tiền xử lý & kiểm tra chất lượng...");
            const res = await axios.get(`${API}/api/v1/process-thermal`);

            setQualityData(res.data);
        } catch (error) {
            alert("Lỗi: " + (error.response?.data?.detail || error.message));
        } finally {
            setIsUploading(false);
            setStatusText("");
            if (fileInputRef.current) fileInputRef.current.value = "";
            if (folderInputRef.current) folderInputRef.current.value = "";
        }
    };

    // Bước 2: Người dùng xác nhận trong modal, sau đó chạy AI
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
        } finally {
            setIsRunningAI(false);
        }
    };

    const handleCancelModal = () => {
        if (!isRunningAI) setQualityData(null);
    };

    const handleUpdateModel = async (e) => {
        const file = e.target.files?.[0];
        if (!file) return;

        if (!file.name.endsWith(".pt")) {
            alert("Vui lòng chọn file định dạng .pt!");
            if (modelInputRef.current) modelInputRef.current.value = "";
            return;
        }

        setIsUpdatingModel(true);

        try {
            const formData = new FormData();
            formData.append("file", file);

            const res = await axios.post(`${API}/api/v1/update-ai-model`, formData);

            if (res.data.error) {
                alert(res.data.error);
            } else {
                alert(res.data.message || "Đã tải trọng số AI mới thành công!");
            }
        } catch (error) {
            alert("Lỗi khi tải trọng số: " + (error.response?.data?.detail || error.message));
        } finally {
            setIsUpdatingModel(false);
            if (modelInputRef.current) modelInputRef.current.value = "";
        }
    };

    const handleSystemReset = async () => {
        if (!window.confirm("Hành động này sẽ xóa sạch dữ liệu và Database. Bạn có chắc chắn?")) return;

        setIsResetting(true);

        try {
            await axios.post(`${API}/api/v1/reset-system`);
            if (onReset) onReset();
            alert("Hệ thống đã được đưa về trạng thái mặc định.");
        } catch (error) {
            alert("Không thể reset: " + error.message);
        } finally {
            setIsResetting(false);
        }
    };

    const isAnyLoading = isUploading || isRunningAI || isResetting || isUpdatingModel;

    return (
        <div>
            {qualityData && (
                <QualityReviewModal
                    qualityData={qualityData}
                    onConfirm={handleRunAI}
                    onCancel={handleCancelModal}
                    isRunningAI={isRunningAI}
                />
            )}

            <div
                style={{
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "flex-end",
                    marginBottom: 24,
                    gap: 16,
                    flexWrap: "wrap",
                }}
            >
                <PageHeader title="Dashboard" subtitle="Solar farm monitoring overview (Real-time AI Data)" />

                <div style={{ display: "flex", gap: 12, flexWrap: "wrap", justifyContent: "flex-end" }}>
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
                        style={{
                            background: "linear-gradient(135deg, #0EA5E9, #8B5CF6)",
                            color: "white",
                            border: "none",
                            boxShadow: "0 4px 12px rgba(14,165,233,0.3)",
                        }}
                    >
                        {isUploading ? statusText : "Tải lên (File/Zip/Rar)"}
                    </ActionButton>

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
                        style={{
                            background: "linear-gradient(135deg, #10B981, #3B82F6)",
                            color: "white",
                            border: "none",
                        }}
                    >
                        {isUploading ? statusText : "Tải Thư Mục"}
                    </ActionButton>

                    <input
                        type="file"
                        accept=".pt"
                        ref={modelInputRef}
                        onChange={handleUpdateModel}
                        style={{ display: "none" }}
                    />

                    <ActionButton
                        onClick={() => modelInputRef.current?.click()}
                        disabled={isAnyLoading}
                        icon={isUpdatingModel ? <Loader2 className="animate-spin" size={16} /> : <Cpu size={16} />}
                        style={{
                            background: "linear-gradient(135deg, #0ea5e920, #8b5cf610)",
                            color: colors.primary,
                            border: `1px solid ${colors.primary}50`,
                            backdropFilter: "blur(4px)",
                        }}
                    >
                        {isUpdatingModel ? "Đang Update AI..." : "Thay model AI"}
                    </ActionButton>

                    <ActionButton
                        onClick={handleSystemReset}
                        disabled={isAnyLoading}
                        icon={isResetting ? <Loader2 className="animate-spin" size={16} /> : <Trash2 size={16} />}
                        style={{
                            background: "#fee2e2",
                            color: colors.error || colors.danger,
                            border: `1px solid ${(colors.error || colors.danger)}30`,
                        }}
                    >
                        Reset System
                    </ActionButton>
                </div>
            </div>

            <div
                style={{
                    display: "grid",
                    gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
                    gap: 16,
                    marginBottom: 24,
                }}
            >
                <KpiCard
                    icon={<LayoutGrid size={20} />}
                    label="TỔNG SỐ HÌNH ẢNH"
                    value={totalPanels.toLocaleString()}
                    accent={colors.primary}
                />
                <KpiCard
                    icon={<AlertCircle size={20} />}
                    label="FAULTS DETECTED"
                    value={totalFaults}
                    accent={colors.danger || colors.error}
                />
                <KpiCard
                    icon={<DollarSign size={20} />}
                    label="ESTIMATED LOSS ($)"
                    value={`$${estimatedLoss.toFixed(2)}`}
                    accent={colors.warning}
                />
            </div>

            <div
                style={{
                    background: "rgba(255,255,255,0.85)",
                    backdropFilter: "blur(16px)",
                    borderRadius: 20,
                    border: "1px solid rgba(226,232,240,0.8)",
                    padding: "24px 32px",
                    boxShadow: "0 10px 30px rgba(0,0,0,0.04)",
                    display: "flex",
                    alignItems: "stretch",
                    justifyContent: "space-between",
                    gap: 32,
                    flexWrap: "wrap",
                }}
            >
                <style>{`
                    @keyframes neonPulse {
                        0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
                        70% { box-shadow: 0 0 0 6px rgba(16, 185, 129, 0); }
                        100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
                    }
                `}</style>

                <div style={{ display: "flex", alignItems: "center", gap: 24, minWidth: 280 }}>
                    <div
                        style={{
                            position: "relative",
                            width: 90,
                            height: 90,
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "center",
                            flexShrink: 0,
                        }}
                    >
                        <svg width="90" height="90" style={{ transform: "rotate(-90deg)" }}>
                            <circle cx="45" cy="45" r="36" fill="transparent" stroke="#f1f5f9" strokeWidth="6" />
                            <circle
                                cx="45"
                                cy="45"
                                r="36"
                                fill="transparent"
                                stroke="url(#progressGradient)"
                                strokeWidth="6"
                                strokeDasharray={226.2}
                                strokeDashoffset={226.2 - (healthyRate / 100) * 226.2}
                                strokeLinecap="round"
                                style={{ transition: "stroke-dashoffset 1.2s ease-in-out" }}
                            />
                            <defs>
                                <linearGradient id="progressGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                                    <stop offset="0%" stopColor="#10B981" />
                                    <stop offset="100%" stopColor="#059669" />
                                </linearGradient>
                            </defs>
                        </svg>

                        <div style={{ position: "absolute", textAlign: "center" }}>
                            <div style={{ fontSize: 16, fontWeight: 800, color: "#1e293b" }}>
                                {totalPanels > 0 ? `${healthyRate.toFixed(1)}%` : "—"}
                            </div>
                        </div>
                    </div>

                    <div>
                        <div
                            style={{
                                fontSize: 11,
                                fontWeight: 600,
                                color: "#64748b",
                                textTransform: "uppercase",
                                letterSpacing: "1px",
                                marginBottom: 4,
                            }}
                        >
                            AI Analysis Breakdown
                        </div>
                        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                            <h3
                                style={{
                                    margin: 0,
                                    fontSize: 20,
                                    fontWeight: 850,
                                    color: "#1e293b",
                                    letterSpacing: "-0.5px",
                                }}
                            >
                                System Stability
                            </h3>
                            <div
                                style={{
                                    width: 8,
                                    height: 8,
                                    borderRadius: "50%",
                                    background: totalPanels > 0 && healthyRate > 90 ? "#10b981" : "#94a3b8",
                                    animation: totalPanels > 0 && healthyRate > 90 ? "neonPulse 2s infinite" : "none",
                                    flexShrink: 0,
                                }}
                            />
                        </div>
                    </div>
                </div>

                <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: 10, minWidth: 260 }}>
                    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
                        <div
                            style={{
                                padding: "10px 14px",
                                background: "#f8fafc",
                                borderRadius: 12,
                                border: "1px solid #f1f5f9",
                            }}
                        >
                            <div
                                style={{
                                    fontSize: 9,
                                    fontWeight: 700,
                                    color: "#94a3b8",
                                    textTransform: "uppercase",
                                    letterSpacing: "0.5px",
                                    marginBottom: 4,
                                }}
                            >
                                Batch Status
                            </div>
                            <div
                                style={{
                                    display: "inline-flex",
                                    alignItems: "center",
                                    padding: "2px 8px",
                                    background: totalPanels > 0 ? "#e6f4ea" : "#f1f5f9",
                                    color: totalPanels > 0 ? "#137333" : "#5f6368",
                                    borderRadius: 6,
                                    fontSize: 10,
                                    fontWeight: 700,
                                    textTransform: "uppercase",
                                }}
                            >
                                {totalPanels > 0 ? "Active" : "None"}
                            </div>
                        </div>

                        <div
                            style={{
                                padding: "10px 14px",
                                background: "#f8fafc",
                                borderRadius: 12,
                                border: "1px solid #f1f5f9",
                            }}
                        >
                            <div
                                style={{
                                    fontSize: 9,
                                    fontWeight: 700,
                                    color: "#94a3b8",
                                    textTransform: "uppercase",
                                    letterSpacing: "0.5px",
                                    marginBottom: 4,
                                }}
                            >
                                Health Profile
                            </div>
                            <div
                                style={{
                                    fontSize: 12,
                                    fontWeight: 800,
                                    color: totalPanels > 0 ? (healthyRate > 95 ? "#10b981" : "#d97706") : "#94a3b8",
                                }}
                            >
                                {totalPanels > 0 ? (healthyRate > 95 ? "EXCELLENT" : "STABLE") : "NO SCAN"}
                            </div>
                        </div>
                    </div>

                    <div style={{ height: 12, background: "#f1f5f9", borderRadius: 10, overflow: "hidden" }}>
                        <div
                            style={{
                                height: "100%",
                                width: `${healthyRate}%`,
                                background: colors.success,
                                transition: "width 0.5s ease-in-out",
                            }}
                        />
                    </div>

                    <p style={{ margin: 0, fontSize: 11, color: "#94a3b8", textAlign: "right" }}>
                        * Data updated from the latest AI scan (Batch ID: {totalPanels > 0 ? "Active" : "None"}).
                    </p>
                </div>

                <div
                    style={{
                        minWidth: 260,
                        flex: "0 1 340px",
                        background: "#f8fafc",
                        borderRadius: 14,
                        border: "1px solid #f1f5f9",
                        padding: 14,
                        height: 150,
                    }}
                >
                    <div
                        style={{
                            fontSize: 11,
                            fontWeight: 700,
                            color: "#64748b",
                            textTransform: "uppercase",
                            letterSpacing: "0.6px",
                            marginBottom: 8,
                        }}
                    >
                        Anomaly Distribution
                    </div>
                    <div style={{ height: 110 }}>
                        <AnomalyBarChart data={chartData} />
                    </div>
                </div>
            </div>

            <div
                style={{
                    marginTop: 24,
                    marginBottom: 10,
                    borderRadius: 16,
                    overflow: "hidden",
                    height: 260,
                    border: "1px solid rgba(226,232,240,0.8)",
                    boxShadow: "0 8px 30px rgba(0,0,0,0.03)",
                }}
            >
                <img
                    src={solarFarmAerial}
                    alt="Solar Farm Ambient Background"
                    style={{
                        width: "100%",
                        height: "100%",
                        objectFit: "cover",
                    }}
                />
            </div>
        </div>
    );
}
