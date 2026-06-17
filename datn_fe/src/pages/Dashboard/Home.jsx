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
    Zap,
    RefreshCw,
    Settings,
    Image,
} from "lucide-react";
import { colors } from "../../constants/theme";
import { PageHeader } from "../../components/layout/PageHeader";
import { KpiCard } from "../../components/ui/KpiCard";
import { ActionButton } from "../../components/ui/ActionButton";
import { 
    fetchLatestBatch, 
    analyzeAll, 
    getAnalyzeProgress, 
    uploadDroneData, 
    processThermal, 
    updateBatchMetadata, 
    reanalyze, 
    resetSystem, 
    updateAiModel 
} from "../../api";
import { computeInspectionSummary } from "../../utils/inspectionData";
import solarFarmAerial from "../../assets/solar_farm_aerial.png";

const API = "http://127.0.0.1:8000";

// ─────────────────────────────────────────
// Mini bar chart component
// ─────────────────────────────────────────
function AnomalyBarChart({ data }) {
    if (!data || data.length === 0) {
        return (
            <div style={{
                display: "flex", alignItems: "center", justifyContent: "center",
                height: "100%", flexDirection: "column", gap: 12, color: "#94a3b8",
            }}>
                <TrendingUp size={32} style={{ opacity: 0.3 }} />
                <p style={{ margin: 0, fontSize: 12 }}>Chưa có dữ liệu phân tích</p>
            </div>
        );
    }

    const max = Math.max(...data.map((d) => d.value), 1);
    const barColors = ["#f97316","#ef4444","#06b6d4","#eab308","#a855f7","#10b981","#0ea5e9","#f59e0b"];
    const MAX_BAR_HEIGHT = 64; // px — giới hạn cứng để không tràn

    return (
        <div style={{ display: "flex", flexDirection: "column", height: "100%", overflow: "hidden" }}>
            {/* Bars area */}
            <div style={{
                flex: 1, display: "flex", alignItems: "flex-end",
                gap: 6, minHeight: 0, overflow: "hidden", paddingBottom: 2,
            }}>
                {data.map((item, i) => {
                    const barH = Math.max(Math.round((item.value / max) * MAX_BAR_HEIGHT), 4);
                    const color = barColors[i % barColors.length];
                    return (
                        <div key={i} style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center", gap: 2, minWidth: 0 }}>
                            <span style={{ fontSize: 10, fontWeight: 700, color: "#334155", lineHeight: 1 }}>
                                {item.value}
                            </span>
                            <div style={{
                                width: "100%", height: barH,
                                background: `linear-gradient(180deg, ${color}, ${color}bb)`,
                                borderRadius: "3px 3px 0 0",
                                transition: "height 0.6s ease",
                            }} />
                        </div>
                    );
                })}
            </div>

            {/* Labels */}
            <div style={{ display: "flex", gap: 6, borderTop: "1px solid #e2e8f0", paddingTop: 4, flexShrink: 0 }}>
                {data.map((item, i) => (
                    <div key={i} style={{
                        flex: 1, textAlign: "center", fontSize: 8,
                        color: "#64748b", wordBreak: "break-word", lineHeight: 1.2, minWidth: 0,
                    }}>
                        {item.label}
                    </div>
                ))}
            </div>
        </div>
    );
}

const formatPercent = (value) => {
    const n = Number(value || 0);
    if (!Number.isFinite(n)) return "0.00%";
    return `${n.toFixed(2)}%`;
};

// ─────────────────────────────────────────
// AI Progress Modal
// ─────────────────────────────────────────
function AIProgressModal({ progress, onClose }) {
    const pct = progress.percent;

    // Tính toán thời gian chờ đợi còn lại (ETA) chuyên nghiệp và ổn định
    let etaText = "Đang tính...";
    const { status, percent, etaSeconds } = progress;
    if (
        status === "running" &&
        percent > 5 &&
        percent < 99 &&
        etaSeconds !== null &&
        Number.isFinite(etaSeconds) &&
        etaSeconds > 0
    ) {
        const m = Math.floor(etaSeconds / 60);
        const s = Math.round(etaSeconds % 60);
        etaText = m > 0 ? `${m}m ${s}s` : `${s}s`;
    } else if (status === "completed") {
        etaText = "Hoàn tất!";
    } else if (status === "error") {
        etaText = "Thất bại";
    }

    return (
        <div style={{
            position: "fixed", inset: 0, zIndex: 10000,
            background: "rgba(2,8,23,0.90)", backdropFilter: "blur(10px)",
            display: "flex", alignItems: "center", justifyContent: "center",
        }}>
            <div style={{
                background: "linear-gradient(145deg, #0f172a, #1e293b)",
                border: progress.status === "error" ? "1px solid rgba(239,68,68,0.35)" : "1px solid rgba(14,165,233,0.35)",
                borderRadius: 24,
                boxShadow: progress.status === "error"
                    ? "0 0 80px rgba(239,68,68,0.15), 0 30px 60px rgba(0,0,0,0.6)"
                    : "0 0 80px rgba(14,165,233,0.15), 0 30px 60px rgba(0,0,0,0.6)",
                width: "min(520px, 92vw)",
                padding: "40px 44px",
                display: "flex", flexDirection: "column", alignItems: "center", gap: 28,
            }}>
                {/* Animated icon */}
                <div style={{ position: "relative", width: 72, height: 72 }}>
                    <svg viewBox="0 0 72 72" width="72" height="72">
                        <circle cx="36" cy="36" r="30" fill="none" stroke="rgba(14,165,233,0.15)" strokeWidth="5" />
                        <circle
                            cx="36" cy="36" r="30" fill="none"
                            stroke="url(#aiGrad)" strokeWidth="5"
                            strokeDasharray={188.5}
                            strokeDashoffset={188.5 - (pct / 100) * 188.5}
                            strokeLinecap="round"
                            style={{ transform: "rotate(-90deg)", transformOrigin: "36px 36px", transition: "stroke-dashoffset 0.5s ease" }}
                        />
                        <defs>
                            <linearGradient id="aiGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                                <stop offset="0%" stopColor={progress.status === "error" ? "#ef4444" : "#0EA5E9"} />
                                <stop offset="100%" stopColor={progress.status === "error" ? "#b91c1c" : "#6366F1"} />
                            </linearGradient>
                        </defs>
                    </svg>
                    <div style={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center" }}>
                        <span style={{ fontSize: 15, fontWeight: 800, color: progress.status === "error" ? "#ef4444" : "#0EA5E9" }}>{formatPercent(pct)}</span>
                    </div>
                </div>

                {/* Title */}
                <div style={{ textAlign: "center" }}>
                    <h3 style={{ margin: "0 0 6px 0", color: "#F8FAFC", fontSize: 20, fontWeight: 700, letterSpacing: "-0.3px" }}>
                        {progress.status === "error" ? "Lỗi phân tích AI" : "Đang phân tích AI..."}
                    </h3>
                    <p style={{ margin: 0, color: progress.status === "error" ? "#ef4444" : "#64748B", fontSize: 13 }}>
                        {progress.status === "error" ? (progress.error || progress.stage) : progress.stage}
                    </p>
                </div>

                {/* Progress bar */}
                <div style={{ width: "100%" }}>
                    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8 }}>
                        <span style={{ fontSize: 12, color: "#94A3B8" }}>
                            {progress.totalImages > 0 ? `Ảnh ${progress.processedImages} / ${progress.totalImages}` : "Chuẩn bị..."}
                        </span>
                    </div>
                    <div style={{ height: 8, background: "rgba(255,255,255,0.06)", borderRadius: 10, overflow: "hidden", position: "relative" }}>
                        {/* Shimmer bg */}
                        <div style={{
                            position: "absolute", inset: 0,
                            background: "linear-gradient(90deg, transparent 25%, rgba(255,255,255,0.04) 50%, transparent 75%)",
                            backgroundSize: "200% 100%",
                            animation: "shimmerAI 1.5s infinite linear",
                        }} />
                        {/* Fill */}
                        <div style={{
                            height: "100%", width: `${pct}%`,
                            background: progress.status === "error"
                                ? "linear-gradient(90deg, #ef4444, #b91c1c)"
                                : "linear-gradient(90deg, #0EA5E9, #6366F1)",
                            borderRadius: 10,
                            transition: "width 0.5s ease",
                            position: "relative",
                        }} />
                    </div>
                    {/* Time metrics */}
                    <div style={{ display: "flex", justifyContent: "space-between", marginTop: 12 }}>
                        <span style={{ fontSize: 11, color: "#64748B" }}>
                            Đã chạy: <b style={{ color: "#CBD5E1" }}>{progress.elapsedSeconds}s</b>
                        </span>
                        <span style={{ fontSize: 11, color: "#64748B" }}>
                            Còn lại (ước tính): <b style={{ color: progress.status === "error" ? "#ef4444" : "#f59e0b" }}>{etaText}</b>
                        </span>
                    </div>
                </div>

                {/* Current file */}
                {progress.filename && (
                    <div style={{
                        background: progress.status === "error" ? "rgba(239,68,68,0.08)" : "rgba(14,165,233,0.08)",
                        border: progress.status === "error" ? "1px solid rgba(239,68,68,0.2)" : "1px solid rgba(14,165,233,0.2)",
                        borderRadius: 10, padding: "10px 18px",
                        width: "100%", boxSizing: "border-box",
                    }}>
                        <div style={{ fontSize: 10, color: "#64748B", textTransform: "uppercase", letterSpacing: "0.5px", marginBottom: 4 }}>
                            Tệp đang xử lý
                        </div>
                        <div style={{ fontSize: 13, color: "#CBD5E1", fontFamily: "monospace", wordBreak: "break-all" }}>
                            {progress.filename}
                        </div>
                    </div>
                )}

                {/* Error Close Button */}
                {progress.status === "error" && (
                    <button
                        onClick={onClose}
                        style={{
                            marginTop: 8,
                            padding: "10px 24px",
                            borderRadius: 10,
                            background: "linear-gradient(135deg, #ef4444, #b91c1c)",
                            border: "none",
                            color: "#fff",
                            fontSize: 14,
                            fontWeight: 700,
                            cursor: "pointer",
                            boxShadow: "0 4px 15px rgba(239,68,68,0.4)",
                            transition: "all 0.2s",
                        }}
                    >
                        Đóng
                    </button>
                )}

                <style>{`
                    @keyframes shimmerAI {
                        0% { background-position: -200% 0; }
                        100% { background-position: 200% 0; }
                    }
                `}</style>
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
                                            padding: "16px",
                                            display: "flex",
                                            justifyContent: "center",
                                            alignItems: "center",
                                            background: "rgba(2, 6, 23, 0.4)",
                                        }}
                                    >
                                        <div style={{
                                            width: "100%",
                                            maxWidth: "600px",
                                            borderRadius: 12,
                                            overflow: "hidden",
                                            border: "1px solid rgba(255, 255, 255, 0.1)",
                                            boxShadow: "0 10px 25px -5px rgba(0, 0, 0, 0.3)",
                                            background: "#000",
                                        }}>
                                            <img
                                                src={`${API}${item.preview_url}`}
                                                alt={item.filename}
                                                style={{ width: "100%", height: "auto", display: "block", maxHeight: "400px", objectFit: "contain" }}
                                                onError={(e) => {
                                                    e.target.style.display = "none";
                                                }}
                                            />
                                        </div>
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
// Project Metadata & Panel Power Modal
// ─────────────────────────────────────────
function ProjectMetadataModal({ 
    metadata, 
    onChange, 
    onSave, 
    onClose, 
    isNewProject = false, 
    isLoading = false 
}) {
    return (
        <div style={{
            position: "fixed", inset: 0, zIndex: 10005,
            background: "rgba(2,8,23,0.85)", backdropFilter: "blur(8px)",
            display: "flex", alignItems: "center", justifyContent: "center",
            padding: 20,
        }}>
            <div style={{
                background: "linear-gradient(145deg, #0f172a, #1e293b)",
                border: "1px solid rgba(14,165,233,0.3)",
                borderRadius: 20,
                boxShadow: "0 20px 50px rgba(0,0,0,0.5), 0 0 40px rgba(14,165,233,0.1)",
                width: "min(550px, 94vw)",
                maxHeight: "90vh",
                display: "flex", flexDirection: "column", overflow: "hidden"
            }}>
                {/* Header */}
                <div style={{
                    padding: "20px 24px",
                    borderBottom: "1px solid rgba(255,255,255,0.06)",
                    display: "flex", justifyContent: "space-between", alignItems: "center",
                    background: "rgba(14,165,233,0.03)"
                }}>
                    <div>
                        <h3 style={{ margin: 0, color: "#f8fafc", fontSize: 18, fontWeight: 700 }}>
                            {isNewProject ? "Thông Tin Dự Án Mới" : "Cấu Hình Thông Tin Dự Án"}
                        </h3>
                        <p style={{ margin: "4px 0 0 0", color: "#64748b", fontSize: 12 }}>
                            {isNewProject ? "Nhập metadata dự án trước khi chạy AI phân tích" : "Chỉnh sửa thông tin dự án & công suất tấm pin"}
                        </p>
                    </div>
                    <button onClick={onClose} style={{
                        background: "rgba(255,255,255,0.05)", border: "none", color: "#94a3b8",
                        borderRadius: 8, padding: 6, cursor: "pointer", transition: "0.2s"
                    }}>
                        <X size={16} />
                    </button>
                </div>

                {/* Content - Inputs */}
                <div style={{ flex: 1, overflowY: "auto", padding: 24, display: "flex", flexDirection: "column", gap: 16 }}>
                    
                    {/* Tên dự án */}
                    <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                        <label style={{ fontSize: 12, fontWeight: 600, color: "#94a3b8" }}>Tên dự án</label>
                        <input 
                            type="text" 
                            value={metadata.projectName || ""} 
                            onChange={(e) => onChange("projectName", e.target.value)}
                            placeholder="Ví dụ: Binh Nguyen Solar Farm Phase 1"
                            style={inputStyle}
                        />
                    </div>

                    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
                        {/* Địa điểm */}
                        <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                            <label style={{ fontSize: 12, fontWeight: 600, color: "#94a3b8" }}>Địa điểm</label>
                            <input 
                                type="text" 
                                value={metadata.location || ""} 
                                onChange={(e) => onChange("location", e.target.value)}
                                placeholder="Ví dụ: Ninh Thuan, Viet Nam"
                                style={inputStyle}
                            />
                        </div>

                        {/* Thời gian quét */}
                        <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                            <label style={{ fontSize: 12, fontWeight: 600, color: "#94a3b8" }}>Thời gian quét</label>
                            <input 
                                type="text" 
                                value={metadata.scanTime || ""} 
                                onChange={(e) => onChange("scanTime", e.target.value)}
                                placeholder="Ví dụ: 2026-06-02"
                                style={inputStyle}
                            />
                        </div>
                    </div>

                    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
                        {/* Đơn vị quét */}
                        <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                            <label style={{ fontSize: 12, fontWeight: 600, color: "#94a3b8" }}>Đơn vị quét</label>
                            <input 
                                type="text" 
                                value={metadata.operator || ""} 
                                onChange={(e) => onChange("operator", e.target.value)}
                                placeholder="Ví dụ: EPC Solar JSC"
                                style={inputStyle}
                            />
                        </div>

                        {/* Thiết bị quét */}
                        <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                            <label style={{ fontSize: 12, fontWeight: 600, color: "#94a3b8" }}>Thiết bị quét</label>
                            <input 
                                type="text" 
                                value={metadata.device || ""} 
                                onChange={(e) => onChange("device", e.target.value)}
                                placeholder="Ví dụ: DJI Matrice 300 RTK"
                                style={inputStyle}
                            />
                        </div>
                    </div>

                    <div style={{ display: "grid", gridTemplateColumns: "1.2fr 0.8fr", gap: 16 }}>
                        {/* Phạm vi quét */}
                        <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                            <label style={{ fontSize: 12, fontWeight: 600, color: "#94a3b8" }}>Phạm vi quét</label>
                            <input 
                                type="text" 
                                value={metadata.scope || ""} 
                                onChange={(e) => onChange("scope", e.target.value)}
                                placeholder="Ví dụ: Inverter Block 01 - 04"
                                style={inputStyle}
                            />
                        </div>

                        {/* Công suất tấm pin */}
                        <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                            <label style={{ fontSize: 12, fontWeight: 600, color: "#0ea5e9" }}>Công suất tấm pin (W)</label>
                            <input 
                                type="text" 
                                value={metadata.panelPower === undefined || metadata.panelPower === null ? "" : metadata.panelPower} 
                                onChange={(e) => {
                                    const val = e.target.value;
                                    if (val === "" || /^[0-9]*\.?[0-9]*$/.test(val)) {
                                        onChange("panelPower", val);
                                    }
                                }}
                                placeholder="Mặc định: 600"
                                style={{
                                    ...inputStyle,
                                    border: "1px solid rgba(14,165,233,0.4)",
                                    color: "#38bdf8",
                                    fontWeight: 700
                                }}
                            />
                        </div>
                    </div>

                    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
                        {/* Công suất hệ thống */}
                        <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                            <label style={{ fontSize: 12, fontWeight: 600, color: "#94a3b8" }}>Công suất hệ thống</label>
                            <input 
                                type="text" 
                                value={metadata.systemCapacity || ""} 
                                onChange={(e) => onChange("systemCapacity", e.target.value)}
                                placeholder="Ví dụ: 1.2 MWp"
                                style={inputStyle}
                            />
                        </div>

                        {/* Người phụ trách */}
                        <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                            <label style={{ fontSize: 12, fontWeight: 600, color: "#94a3b8" }}>Người phụ trách</label>
                            <input 
                                type="text" 
                                value={metadata.supervisor || ""} 
                                onChange={(e) => onChange("supervisor", e.target.value)}
                                placeholder="Ví dụ: Nguyễn Văn A"
                                style={inputStyle}
                            />
                        </div>
                    </div>

                    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
                        {/* Loại dữ liệu */}
                        <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                            <label style={{ fontSize: 12, fontWeight: 600, color: "#94a3b8" }}>Loại dữ liệu</label>
                            <input 
                                type="text" 
                                value={metadata.dataType || ""} 
                                onChange={(e) => onChange("dataType", e.target.value)}
                                placeholder="Mặc định: UAV thermal image"
                                style={inputStyle}
                            />
                        </div>

                        {/* Model AI sử dụng */}
                        <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                            <label style={{ fontSize: 12, fontWeight: 600, color: "#94a3b8" }}>Model AI sử dụng</label>
                            <input 
                                type="text" 
                                value={metadata.aiModel || ""} 
                                onChange={(e) => onChange("aiModel", e.target.value)}
                                placeholder="Mặc định: YOLOv8-Solar-M300"
                                style={inputStyle}
                            />
                        </div>
                    </div>

                    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
                        {/* Phiên bản hệ thống */}
                        <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                            <label style={{ fontSize: 12, fontWeight: 600, color: "#94a3b8" }}>Phiên bản hệ thống</label>
                            <input 
                                type="text" 
                                value={metadata.systemVersion || ""} 
                                onChange={(e) => onChange("systemVersion", e.target.value)}
                                placeholder="Mặc định: O&M Suite v2.4"
                                style={inputStyle}
                            />
                        </div>

                        {/* Spacer */}
                        <div style={{ display: "flex", flexDirection: "column", gap: 6 }} />
                    </div>

                    {/* Ghi chú */}
                    <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                        <label style={{ fontSize: 12, fontWeight: 600, color: "#94a3b8" }}>Ghi chú</label>
                        <textarea 
                            value={metadata.notes || ""} 
                            onChange={(e) => onChange("notes", e.target.value)}
                            placeholder="Nhập ghi chú thêm..."
                            style={{
                                ...inputStyle,
                                height: 60,
                                resize: "none"
                            }}
                        />
                    </div>
                </div>

                {/* Footer Buttons */}
                <div style={{
                    padding: "16px 24px",
                    borderTop: "1px solid rgba(255,255,255,0.06)",
                    display: "flex", justifyContent: "flex-end", gap: 12,
                    background: "rgba(15,23,42,0.4)"
                }}>
                    <button 
                        onClick={onClose} 
                        style={{
                            padding: "9px 18px", borderRadius: 8, background: "transparent",
                            border: "1px solid rgba(255,255,255,0.15)", color: "#94a3b8",
                            fontSize: 13, fontWeight: 600, cursor: "pointer"
                        }}
                    >
                        {isNewProject ? "Bỏ qua & Đóng" : "Đóng"}
                    </button>
                    <button 
                        onClick={onSave} 
                        disabled={isLoading}
                        style={{
                            padding: "9px 24px", borderRadius: 8,
                            background: "linear-gradient(135deg, #0ea5e9, #6366f1)",
                            border: "none", color: "#fff", fontSize: 13, fontWeight: 700,
                            cursor: "pointer", display: "flex", alignItems: "center", gap: 6,
                            boxShadow: "0 4px 15px rgba(14,165,233,0.3)"
                        }}
                    >
                        {isLoading ? <Loader2 size={14} className="animate-spin" /> : null}
                        {isNewProject ? "Lưu & Chạy AI Phân Tích" : "Lưu Thay Đổi"}
                    </button>
                </div>
            </div>
        </div>
    );
}

const inputStyle = {
    background: "rgba(2, 6, 23, 0.4)",
    border: "1px solid rgba(255, 255, 255, 0.08)",
    borderRadius: 10,
    color: "#f8fafc",
    padding: "10px 14px",
    fontSize: 13,
    outline: "none",
    transition: "border-color 0.2s",
    boxSizing: "border-box",
    width: "100%"
};

function computeEtaFallback(elapsedSeconds, percent) {
    if (!Number.isFinite(elapsedSeconds) || elapsedSeconds <= 0) return null;
    if (!Number.isFinite(percent) || percent <= 5 || percent >= 99) return null;

    const totalEstimated = elapsedSeconds / (percent / 100);
    const eta = totalEstimated - elapsedSeconds;

    if (!Number.isFinite(eta) || eta <= 0) return null;
    return eta;
}

function normalizeProgress(raw, elapsedSeconds) {
    const status = raw?.status || (raw?.running ? "running" : "idle");

    const total = Number(raw?.total_images || raw?.total || 0);
    const processed = Number(raw?.processed_images || raw?.processed || 0);

    let percent = Number(raw?.percent ?? raw?.progress ?? 0);

    if (!Number.isFinite(percent)) percent = 0;

    if (percent <= 1 && percent > 0) {
        percent = percent * 100;
    }

    percent = Math.max(0, Math.min(100, percent));

    // Nếu backend báo completed thì UI bắt buộc hiển thị 100%
    if (status === "completed") {
        percent = 100;
    }

    // Nếu backend vẫn running thì không được hiển thị 100%
    // để tránh cảm giác “xong rồi nhưng còn chạy”
    if (status === "running" && percent >= 100) {
        percent = 99;
    }

    let eta = raw?.eta_seconds ?? null;
    if (eta === null || !Number.isFinite(eta) || eta <= 0) {
        eta = computeEtaFallback(elapsedSeconds, percent);
    }

    return {
        status,
        percent,
        totalImages: total,
        processedImages: processed,
        stage: raw?.current_stage || raw?.stage || raw?.message || "Đang xử lý...",
        message: raw?.message || "",
        elapsedSeconds: elapsedSeconds,
        etaSeconds: eta,
        error: raw?.error || null,
        filename: raw?.filename || "",
    };
}

// ─────────────────────────────────────────
// Home Page
// ─────────────────────────────────────────
export default function Home({ data, batchId, onAnalysisComplete, onReset }) {
    const [isUploading, setIsUploading] = useState(false);
    const [isAnalyzing, setIsAnalyzing] = useState(false);
    const [analysisProgress, setAnalysisProgress] = useState({
        status: "idle",
        percent: 0,
        stage: "Chưa bắt đầu",
        processedImages: 0,
        totalImages: 0,
        elapsedSeconds: 0,
        etaSeconds: null,
        message: "",
        error: null,
        filename: "",
    });
    const [isResetting, setIsResetting] = useState(false);
    const [isUpdatingModel, setIsUpdatingModel] = useState(false);
    const [isReanalyzing, setIsReanalyzing] = useState(false);
    const [statusText, setStatusText] = useState("");
    const [qualityData, setQualityData] = useState(null);
    const [uploadMenuOpen, setUploadMenuOpen] = useState(false);

    const [projectMetadata, setProjectMetadata] = useState({
        projectName: "",
        location: "",
        scanTime: "",
        operator: "",
        device: "",
        scope: "",
        panelPower: 600,
        systemCapacity: "",
        supervisor: "",
        notes: "",
        dataType: "UAV thermal image",
        aiModel: "YOLOv8-Solar-M300",
        systemVersion: "O&M Suite v2.4"
    });
    const [showMetadataModal, setShowMetadataModal] = useState(false);
    const [isSavingMetadata, setIsSavingMetadata] = useState(false);

    const fileInputRef = useRef(null);
    const folderInputRef = useRef(null);
    const modelInputRef = useRef(null);
    const uploadMenuRef = useRef(null);

    const progressTimerRef = useRef(null);
    const analysisStartTimeRef = useRef(null);

    React.useEffect(() => {
        return () => {
            if (progressTimerRef.current) {
                clearInterval(progressTimerRef.current);
            }
        };
    }, []);

    React.useEffect(() => {
        const handleClickOutside = (event) => {
            if (uploadMenuRef.current && !uploadMenuRef.current.contains(event.target)) {
                setUploadMenuOpen(false);
            }
        };
        document.addEventListener("mousedown", handleClickOutside);
        return () => {
            document.removeEventListener("mousedown", handleClickOutside);
        };
    }, []);

    const summary = computeInspectionSummary(data || []);
    const totalPanels = summary.total_panels;
    const totalFaults = summary.faulty_panels;
    const estimatedLoss = summary.total_power_loss_w;
    const healthyRate = summary.normal_panel_ratio_percent;

    const defectCounts = {};
    (data || []).forEach((img) => {
        (img.panels || []).forEach((p) => {
            const reviewStatus = p.review_status || "unreviewed";
            const include = reviewStatus === "false_positive" ? false : (p.include_in_report !== undefined ? p.include_in_report : true);
            if (include) {
                (p.defects || []).forEach((d) => {
                    const name = (d.class_name || d.type || "unknown").replace(/_/g, " ");
                    defectCounts[name] = (defectCounts[name] || 0) + 1;
                });
            }
        });
    });

    const chartData = Object.entries(defectCounts)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 8)
        .map(([label, value]) => ({ label, value }));

    React.useEffect(() => {
        const loadMetadata = async () => {
            if (batchId) {
                try {
                    const res = await fetchLatestBatch();
                    if (res.data) {
                        setProjectMetadata({
                            projectName: res.data.project_name || "",
                            location: res.data.location || "",
                            scanTime: res.data.scan_time || "",
                            operator: res.data.operator || "",
                            device: res.data.device || "",
                            scope: res.data.scope || "",
                            panelPower: res.data.panel_power || 600,
                            systemCapacity: res.data.system_capacity || "",
                            supervisor: res.data.supervisor || "",
                            notes: res.data.notes || "",
                            dataType: res.data.data_type || "UAV thermal image",
                            aiModel: res.data.ai_model || "YOLOv8-Solar-M300",
                            systemVersion: res.data.system_version || "O&M Suite v2.4"
                        });
                    }
                } catch (e) {
                    console.error("Error loading project metadata:", e);
                }
            } else {
                setProjectMetadata({
                    projectName: "",
                    location: "",
                    scanTime: "",
                    operator: "",
                    device: "",
                    scope: "",
                    panelPower: 600,
                    systemCapacity: "",
                    supervisor: "",
                    notes: "",
                    dataType: "UAV thermal image",
                    aiModel: "YOLOv8-Solar-M300",
                    systemVersion: "O&M Suite v2.4"
                });
            }
        };
        loadMetadata();
    }, [batchId]);

    React.useEffect(() => {
        const handler = () => {
            if (batchId) {
                fetchLatestBatch().then(res => {
                    if (res.data) {
                        setProjectMetadata({
                            projectName: res.data.project_name || "",
                            location: res.data.location || "",
                            scanTime: res.data.scan_time || "",
                            operator: res.data.operator || "",
                            device: res.data.device || "",
                            scope: res.data.scope || "",
                            panelPower: res.data.panel_power || 600,
                            systemCapacity: res.data.system_capacity || "",
                            supervisor: res.data.supervisor || "",
                            notes: res.data.notes || "",
                            dataType: res.data.data_type || "UAV thermal image",
                            aiModel: res.data.ai_model || "YOLOv8-Solar-M300",
                            systemVersion: res.data.system_version || "O&M Suite v2.4"
                        });
                    }
                }).catch(console.error);
            }
        };
        window.addEventListener("review-sync-completed", handler);
        return () => window.removeEventListener("review-sync-completed", handler);
    }, [batchId]);

    const handleMetadataChange = (key, value) => {
        setProjectMetadata(prev => ({
            ...prev,
            [key]: value
        }));
    };

    const handleSaveMetadata = async () => {
        if (!batchId) {
            // Đây là đợt tải dự án mới, lưu tạm vào state rồi đóng để chạy AI
            setShowMetadataModal(false);
            await handleRunAI();
            return;
        }

        setIsSavingMetadata(true);
        try {
            const res = await updateBatchMetadata({
                batch_id: batchId,
                project_name: projectMetadata.projectName,
                location: projectMetadata.location,
                scan_time: projectMetadata.scanTime,
                operator: projectMetadata.operator,
                device: projectMetadata.device,
                scope: projectMetadata.scope,
                panel_power: parseFloat(projectMetadata.panelPower) || 600,
                system_capacity: projectMetadata.systemCapacity,
                supervisor: projectMetadata.supervisor,
                notes: projectMetadata.notes,
                data_type: projectMetadata.dataType,
                ai_model: projectMetadata.aiModel,
                system_version: projectMetadata.systemVersion
            });
            
            if (res.data.error) {
                alert("Lỗi: " + res.data.error);
            } else {
                alert("✅ Cập nhật cấu hình và tính toán lại hao hụt thành công!");
                setShowMetadataModal(false);
                // Cập nhật lại state chính của React
                const latestRes = await fetchLatestBatch();
                if (latestRes.data && onAnalysisComplete) {
                    onAnalysisComplete(latestRes.data.data, latestRes.data.batch_id, latestRes.data.panel_power);
                }
            }
        } catch (e) {
            alert("Lỗi khi cập nhật thông tin: " + (e.response?.data?.detail || e.message));
        } finally {
            setIsSavingMetadata(false);
        }
    };

    // Bước 1: Upload + tiền xử lý, sau đó mở modal quality review
    const handleUploadFiles = async (e) => {
        const files = Array.from(e.target.files || []);
        if (files.length === 0) return;

        setIsUploading(true);

        try {
            setStatusText("Đang tải dữ liệu lên...");
            const formData = new FormData();
            files.forEach((file) => formData.append("files", file));

            await uploadDroneData(formData);

            setStatusText("Đang tiền xử lý & kiểm tra chất lượng...");
            const res = await processThermal();

            setQualityData(res.data);
            // Mở popup nhập metadata dự án trước khi bấm chạy AI!
            setShowMetadataModal(true);
        } catch (error) {
            alert("Lỗi: " + (error.response?.data?.detail || error.message));
        } finally {
            setIsUploading(false);
            setStatusText("");
            if (fileInputRef.current) fileInputRef.current.value = "";
            if (folderInputRef.current) folderInputRef.current.value = "";
        }
    };

    const startPolling = () => {
        if (progressTimerRef.current) {
            clearInterval(progressTimerRef.current);
        }

        progressTimerRef.current = setInterval(async () => {
            try {
                const elapsed = analysisStartTimeRef.current ? Math.round((Date.now() - analysisStartTimeRef.current) / 1000) : 0;
                const res = await getAnalyzeProgress();
                const raw = res.data;
                const progress = normalizeProgress(raw, elapsed);

                setAnalysisProgress(progress);

                if (progress.status === "completed") {
                    if (progressTimerRef.current) {
                        clearInterval(progressTimerRef.current);
                    }
                    setAnalysisProgress(prev => ({
                        ...prev,
                        percent: 100,
                        stage: "Hoàn tất phân tích. Đang cập nhật dữ liệu...",
                    }));

                    try {
                        const latestRes = await fetchLatestBatch();
                        if (latestRes.data && onAnalysisComplete) {
                            onAnalysisComplete(latestRes.data.data, latestRes.data.batch_id, latestRes.data.panel_power);
                        }
                    } catch (err) {
                        console.error("Error fetching latest batch data:", err);
                    }

                    setTimeout(() => {
                        setIsAnalyzing(false);
                        setQualityData(null);
                        setShowMetadataModal(false);
                    }, 1000);
                } else if (progress.status === "error") {
                    if (progressTimerRef.current) {
                        clearInterval(progressTimerRef.current);
                    }
                }
            } catch (err) {
                console.error("Polling error:", err);
            }
        }, 1000);
    };

    // Bước 2: Người dùng xác nhận trong modal, sau đó chạy AI
    const handleRunAI = async () => {
        setIsAnalyzing(true);
        setAnalysisProgress({
            status: "running",
            percent: 0,
            stage: "Khởi động...",
            processedImages: 0,
            totalImages: 0,
            elapsedSeconds: 0,
            etaSeconds: null,
            message: "Đang chuẩn bị phân tích...",
            error: null,
            filename: "",
        });
        analysisStartTimeRef.current = Date.now();
        setQualityData(null);
        setShowMetadataModal(false);

        try {
            const analyzeForm = new FormData();
            const userStr = localStorage.getItem("user");

            if (userStr) {
                const user = JSON.parse(userStr);
                analyzeForm.append("user_id", user.id);
            }

            // Gửi các trường metadata của dự án kèm theo
            analyzeForm.append("project_name", projectMetadata.projectName);
            analyzeForm.append("location", projectMetadata.location);
            analyzeForm.append("scan_time", projectMetadata.scanTime);
            analyzeForm.append("operator", projectMetadata.operator);
            analyzeForm.append("device", projectMetadata.device);
            
            const scopeData = {
                s: projectMetadata.scope || "",
                sc: projectMetadata.systemCapacity || "",
                sv: projectMetadata.supervisor || "",
                nt: projectMetadata.notes || "",
                dt: projectMetadata.dataType || "UAV thermal image",
                am: projectMetadata.aiModel || "YOLOv8-Solar-M300",
                sys: projectMetadata.systemVersion || "O&M Suite v2.4"
            };
            analyzeForm.append("scope", JSON.stringify(scopeData));
            analyzeForm.append("panel_power", projectMetadata.panelPower);

            const res = await analyzeAll(analyzeForm);

            if (res.data.data) {
                if (onAnalysisComplete) {
                    onAnalysisComplete(res.data.data, res.data.batch_id, parseFloat(projectMetadata.panelPower) || 600);
                }
                setQualityData(null);
                setShowMetadataModal(false);
                setIsAnalyzing(false);
            } else if (res.data.message && res.data.message.includes("Bắt đầu phân tích AI")) {
                startPolling();
            } else {
                // Không có ảnh mới (đã xử lý hết hoặc thông báo khác từ server)
                setIsAnalyzing(false);
                alert(res.data.message || "Không có ảnh mới nào cần phân tích!");
            }
        } catch (error) {
            console.error("Lỗi AI:", error);
            setAnalysisProgress(prev => ({
                ...prev,
                status: "error",
                error: error.response?.data?.detail || error.message || "Đã xảy ra lỗi khi khởi chạy AI",
            }));
            if (progressTimerRef.current) {
                clearInterval(progressTimerRef.current);
            }
        }
    };

    const handleCancelModal = () => {
        if (!isAnalyzing) setQualityData(null);
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

            const res = await updateAiModel(formData);

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

    const handleReanalyze = async () => {
        if (!window.confirm("Bạn có muốn chạy lại phân tích AI trên các ảnh đã tải lên không? (Kết quả cũ sẽ bị xóa và cập nhật theo thuật toán mới)")) return;

        setIsReanalyzing(true);

        try {
            await reanalyze();
            if (onReset) onReset();
            // Tự động chạy lại AI
            await handleRunAI();
        } catch (error) {
            alert("Lỗi khi phân tích lại: " + error.message);
        } finally {
            setIsReanalyzing(false);
        }
    };

    const handleSystemReset = async () => {
        if (!window.confirm("Hành động này sẽ xóa sạch dữ liệu và Database. Bạn có chắc chắn?")) return;

        setIsResetting(true);

        try {
            await resetSystem();
            if (onReset) onReset();
            alert("Hệ thống đã được đưa về trạng thái mặc định.");
        } catch (error) {
            alert("Không thể reset: " + error.message);
        } finally {
            setIsResetting(false);
        }
    };

    const isAnyLoading = isUploading || isAnalyzing || isResetting || isUpdatingModel || isReanalyzing;

    return (
        <div>
            {/* AI Progress modal — hiển thị khi đang chạy AI */}
            {isAnalyzing && (
                <AIProgressModal
                    progress={analysisProgress}
                    onClose={() => setIsAnalyzing(false)}
                />
            )}

            {/* Quality Review modal — sau preprocessing */}
            {qualityData && (
                <QualityReviewModal
                    qualityData={qualityData}
                    onConfirm={handleRunAI}
                    onCancel={handleCancelModal}
                    isRunningAI={isAnalyzing}
                />
            )}

            {/* Project Metadata modal — nhập/chỉnh sửa thông tin dự án */}
            {showMetadataModal && (
                <ProjectMetadataModal
                    metadata={projectMetadata}
                    onChange={handleMetadataChange}
                    onSave={handleSaveMetadata}
                    onClose={() => setShowMetadataModal(false)}
                    isNewProject={!batchId}
                    isLoading={isSavingMetadata}
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
                <PageHeader title="Bảng điều khiển" subtitle={projectMetadata.projectName ? `Dự án: ${projectMetadata.projectName} (Dữ liệu AI thời gian thực)` : "Tổng quan giám sát (Dữ liệu AI thời gian thực)"} />

                <div style={{ display: "flex", gap: 12, flexWrap: "wrap", justifyContent: "flex-end" }}>
                    <input
                        type="file"
                        accept=".zip,.rar,.jpg,.jpeg,.png,image/*"
                        multiple
                        ref={fileInputRef}
                        onChange={handleUploadFiles}
                        style={{ display: "none" }}
                    />

                    <input
                        type="file"
                        webkitdirectory="true"
                        directory=""
                        multiple
                        ref={folderInputRef}
                        onChange={handleUploadFiles}
                        style={{ display: "none" }}
                    />

                    <div ref={uploadMenuRef} style={{ position: "relative", display: "inline-block" }}>
                        <ActionButton
                            onClick={() => {
                                if (!isAnyLoading) {
                                    setUploadMenuOpen(prev => !prev);
                                }
                            }}
                            disabled={isAnyLoading}
                            icon={isUploading ? <Loader2 className="animate-spin" size={16} /> : <Upload size={16} />}
                            style={{
                                background: "linear-gradient(135deg, #0EA5E9, #8B5CF6)",
                                color: "white",
                                border: "none",
                                boxShadow: "0 4px 12px rgba(14,165,233,0.3)",
                            }}
                        >
                            {isUploading ? statusText : "Tải dữ liệu UAV ▾"}
                        </ActionButton>

                        {uploadMenuOpen && (
                            <div style={{
                                position: "absolute",
                                top: "100%",
                                right: 0,
                                marginTop: 8,
                                background: "rgba(15, 23, 42, 0.95)",
                                backdropFilter: "blur(12px)",
                                border: "1px solid rgba(14, 165, 233, 0.3)",
                                borderRadius: 12,
                                boxShadow: "0 10px 25px -5px rgba(0, 0, 0, 0.4), 0 8px 16px -8px rgba(0, 0, 0, 0.4)",
                                padding: 6,
                                zIndex: 100,
                                display: "flex",
                                flexDirection: "column",
                                gap: 4,
                                minWidth: 180,
                            }}>
                                <button
                                    onClick={() => {
                                        setUploadMenuOpen(false);
                                        fileInputRef.current?.click();
                                    }}
                                    style={{
                                        background: "transparent",
                                        border: "none",
                                        color: "#f8fafc",
                                        padding: "10px 14px",
                                        borderRadius: 8,
                                        fontSize: 13,
                                        fontWeight: 600,
                                        textAlign: "left",
                                        cursor: "pointer",
                                        transition: "background 0.2s",
                                        display: "flex",
                                        alignItems: "center",
                                        gap: 8,
                                    }}
                                    onMouseEnter={(e) => e.target.style.background = "rgba(14, 165, 233, 0.15)"}
                                    onMouseLeave={(e) => e.target.style.background = "transparent"}
                                >
                                    <span style={{ fontSize: 14 }}>📄</span> Chọn file / Zip / Rar
                                </button>
                                <button
                                    onClick={() => {
                                        setUploadMenuOpen(false);
                                        folderInputRef.current?.click();
                                    }}
                                    style={{
                                        background: "transparent",
                                        border: "none",
                                        color: "#f8fafc",
                                        padding: "10px 14px",
                                        borderRadius: 8,
                                        fontSize: 13,
                                        fontWeight: 600,
                                        textAlign: "left",
                                        cursor: "pointer",
                                        transition: "background 0.2s",
                                        display: "flex",
                                        alignItems: "center",
                                        gap: 8,
                                    }}
                                    onMouseEnter={(e) => e.target.style.background = "rgba(16, 185, 129, 0.15)"}
                                    onMouseLeave={(e) => e.target.style.background = "transparent"}
                                >
                                    <span style={{ fontSize: 14 }}>📁</span> Chọn thư mục ảnh
                                </button>
                            </div>
                        )}
                    </div>

                    <ActionButton
                        onClick={() => setShowMetadataModal(true)}
                        disabled={isAnyLoading}
                        icon={<Settings size={16} />}
                        style={{
                            background: "linear-gradient(135deg, #475569, #1e293b)",
                            color: "white",
                            border: "none",
                            boxShadow: "0 4px 12px rgba(30,41,59,0.3)",
                        }}
                    >
                        Thông tin dự án
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
                        {isUpdatingModel ? "Đang cập nhật AI..." : "Thay model AI"}
                    </ActionButton>

                    <ActionButton
                        onClick={handleReanalyze}
                        disabled={isAnyLoading || totalPanels === 0}
                        icon={isReanalyzing ? <Loader2 className="animate-spin" size={16} /> : <RefreshCw size={16} />}
                        style={{
                            background: "linear-gradient(135deg, #f59e0b, #ef4444)",
                            color: "white",
                            border: "none",
                            boxShadow: totalPanels > 0 ? "0 4px 12px rgba(245,158,11,0.3)" : "none",
                            opacity: (totalPanels === 0) ? 0.5 : 1,
                        }}
                    >
                        {isReanalyzing ? "Đang chạy lại..." : "Phân tích lại"}
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
                        Reset Hệ thống
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
                    icon={<Image size={20} />}
                    label="SỐ HÌNH ĐƯỢC QUÉT"
                    value={data?.length || 0}
                    accent={colors.purple}
                />
                <KpiCard
                    icon={<LayoutGrid size={20} />}
                    label="TỔNG SỐ TẤM PIN"
                    value={totalPanels.toLocaleString()}
                    accent={colors.primary}
                />
                <KpiCard
                    icon={<AlertCircle size={20} />}
                    label="SỐ LƯỢNG LỖI"
                    value={totalFaults}
                    accent={colors.danger || colors.error}
                />
                <KpiCard
                    icon={<Zap size={20} />}
                    label="SẢN LƯỢNG HAO HỤT (W)"
                    value={`${estimatedLoss.toFixed(2)}W`}
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
                            Chi tiết phân tích AI
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
                                Tỷ lệ panel bình thường
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
                                Trạng thái xử lý
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
                                {totalPanels > 0 ? "Đang chạy" : "Không có"}
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
                                Tình trạng hệ thống
                            </div>
                            <div
                                style={{
                                    fontSize: 12,
                                    fontWeight: 800,
                                    color: totalPanels > 0 ? (healthyRate > 95 ? "#10b981" : "#d97706") : "#94a3b8",
                                }}
                            >
                                {totalPanels > 0 ? (healthyRate > 95 ? "XUẤT SẮC" : "ỔN ĐỊNH") : "CHƯA QUÉT"}
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
                        * Dữ liệu được cập nhật từ lần quét AI gần nhất (ID xử lý: {totalPanels > 0 ? "Đang chạy" : "Không có"}).
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
                        Phân bố bất thường
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
