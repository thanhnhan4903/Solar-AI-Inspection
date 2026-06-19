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
    AlertTriangle,
    Eye,
    X,
    ChevronLeft,
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
// Donut Chart component (SVG pure)
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

// Power Loss Bar Chart by fault type
function PowerLossBarChart({ data }) {
    if (!data || data.length === 0) {
        return (
            <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: 110, color: "#94a3b8", flexDirection: "column", gap: 6 }}>
                <Zap size={24} style={{ opacity: 0.3 }} />
                <span style={{ fontSize: 11 }}>Chưa có dữ liệu phân tích</span>
            </div>
        );
    }

    const maxVal = Math.max(...data.map(d => d.value), 0.001);

    const formatVal = (v, unit) => {
        if (unit === "W") return `${Math.round(v)} W`;
        if (unit === "kWp") return `${v >= 100 ? v.toFixed(0) : v.toFixed(1)} kWp`;
        return `${v.toFixed(2)} MWp`;
    };

    return (
        <div style={{ display: "flex", flexDirection: "column", height: "100%", justifyContent: "space-between" }}>
            <div style={{ flex: 1, display: "flex", alignItems: "flex-end", gap: 8, paddingBottom: 4, height: 80 }}>
                {data.map((item, i) => {
                    const barH = Math.max((item.value / maxVal) * 60, 6);
                    return (
                        <div key={i} style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center", gap: 2, minWidth: 0 }}>
                            <span style={{ fontSize: 8, fontWeight: 700, color: "#475569", textAlign: "center", lineHeight: 1.1 }}>
                                {formatVal(item.value, item.unit || "W")}
                            </span>
                            <div style={{
                                width: "100%", height: barH,
                                background: `linear-gradient(180deg, ${item.color}, ${item.color}bb)`,
                                borderRadius: "3px 3px 0 0",
                                transition: "height 0.7s ease",
                                boxShadow: `0 -2px 6px ${item.color}20`,
                                cursor: "default",
                            }}
                                title={`${item.label}: ${formatVal(item.value, item.unit || "W")}`}
                            />
                        </div>
                    );
                })}
            </div>
            <div style={{ display: "flex", gap: 6, borderTop: "1px solid #e2e8f0", paddingTop: 4 }}>
                {data.map((item, i) => (
                    <div key={i} style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center", gap: 2, minWidth: 0 }}>
                        <div style={{ width: 8, height: 8, borderRadius: 2, background: item.color, flexShrink: 0 }} />
                        <span style={{ textAlign: "center", fontSize: 8, color: "#64748b", lineHeight: 1.1, minWidth: 0, wordBreak: "break-word" }}>
                            {item.label}
                        </span>
                    </div>
                ))}
            </div>
        </div>
    );
}

// Image Gallery component
// Image Gallery component
function ImageGallery({ images, apiBase }) {
    const [activeIdx, setActiveIdx] = useState(0);
    const [thumbPage, setThumbPage] = useState(0);
    const [leftHover, setLeftHover] = useState(false);
    const [rightHover, setRightHover] = useState(false);

    const PAGE_SIZE = 8;
    const totalPages = Math.ceil(images.length / PAGE_SIZE);

    React.useEffect(() => {
        const activePage = Math.floor(activeIdx / PAGE_SIZE);
        setThumbPage(activePage);
    }, [activeIdx]);

    if (!images || images.length === 0) {
        return (
            <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", height: "100%", gap: 10, color: "#94a3b8" }}>
                <Image size={36} style={{ opacity: 0.3 }} />
                <span style={{ fontSize: 12 }}>Chưa có ảnh nào được tải lên</span>
            </div>
        );
    }

    const currentImg = images[activeIdx];
    const imgSrc = `${apiBase}/data/precalib/${currentImg.filename}`;

    const visibleImages = images.slice(thumbPage * PAGE_SIZE, (thumbPage + 1) * PAGE_SIZE);
    const paddedImages = [...visibleImages];
    while (paddedImages.length < PAGE_SIZE) {
        paddedImages.push(null);
    }

    const handlePrevThumb = () => {
        setThumbPage(prev => Math.max(0, prev - 1));
    };

    const handleNextThumb = () => {
        setThumbPage(prev => Math.min(totalPages - 1, prev + 1));
    };

    const canScrollLeft = thumbPage > 0;
    const canScrollRight = thumbPage < totalPages - 1;

    return (
        <div style={{ display: "flex", flexDirection: "column", height: "100%", gap: 8 }}>
            {/* Main image */}
            <div style={{ position: "relative", flex: 1, borderRadius: 10, overflow: "hidden", background: "#0f172a", minHeight: 90 }}>
                <img
                    src={imgSrc}
                    alt={currentImg.filename}
                    style={{ width: "100%", height: "100%", objectFit: "cover" }}
                    onError={e => { e.target.src = ""; e.target.style.display = "none"; }}
                />
                {/* Filename overlay */}
                <div style={{
                    position: "absolute", top: 8, left: 10,
                    background: "rgba(15, 23, 42, 0.75)", borderRadius: 6,
                    padding: "4px 8px", fontSize: 11, color: "#fff", fontWeight: 600,
                    maxWidth: "80%", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
                    backdropFilter: "blur(4px)", border: "1px solid rgba(255,255,255,0.15)"
                }} title={currentImg.filename}>
                    {currentImg.filename}
                </div>
                <div style={{
                    position: "absolute", bottom: 8, right: 10,
                    background: "rgba(0,0,0,0.6)", borderRadius: 6,
                    padding: "2px 8px", fontSize: 11, color: "#fff", fontWeight: 600
                }}>
                    {activeIdx + 1} / {images.length}
                </div>
                {/* Nav arrows */}
                {images.length > 1 && (
                    <>
                        <button onClick={() => setActiveIdx(i => Math.max(0, i - 1))} style={{
                            position: "absolute", left: 6, top: "50%", transform: "translateY(-50%)",
                            background: "rgba(0,0,0,0.5)", border: "none", borderRadius: "50%",
                            width: 28, height: 28, display: "flex", alignItems: "center", justifyContent: "center",
                            cursor: "pointer", color: "#fff", fontSize: 14
                        }}>‹</button>
                        <button onClick={() => setActiveIdx(i => Math.min(images.length - 1, i + 1))} style={{
                            position: "absolute", right: 6, top: "50%", transform: "translateY(-50%)",
                            background: "rgba(0,0,0,0.5)", border: "none", borderRadius: "50%",
                            width: 28, height: 28, display: "flex", alignItems: "center", justifyContent: "center",
                            cursor: "pointer", color: "#fff", fontSize: 14
                        }}>›</button>
                    </>
                )}
            </div>

            {/* Thumbnails strip with nav arrows */}
            <div style={{ display: "flex", alignItems: "center", gap: 6, flexShrink: 0 }}>
                {/* Left arrow */}
                <button
                    onClick={handlePrevThumb}
                    disabled={!canScrollLeft}
                    onMouseEnter={() => setLeftHover(true)}
                    onMouseLeave={() => setLeftHover(false)}
                    style={{
                        background: !canScrollLeft
                            ? "rgba(15, 23, 42, 0.4)"
                            : leftHover
                            ? "rgba(15, 23, 42, 0.95)"
                            : "rgba(15, 23, 42, 0.8)",
                        border: "1px solid rgba(255, 255, 255, 0.15)",
                        borderRadius: 6,
                        width: 24,
                        height: 36,
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        color: !canScrollLeft ? "rgba(255, 255, 255, 0.2)" : "#fff",
                        cursor: !canScrollLeft ? "not-allowed" : "pointer",
                        transition: "all 0.2s",
                        padding: 0,
                    }}
                >
                    <ChevronLeft size={16} />
                </button>

                {/* Thumbnails container */}
                <div style={{ display: "flex", gap: 6, flex: 1, justifyContent: "flex-start", overflow: "hidden", height: 36 }}>
                    {paddedImages.map((img, idx) => {
                        const actualIdx = thumbPage * PAGE_SIZE + idx;
                        if (!img) {
                            return (
                                <div
                                    key={`placeholder-${idx}`}
                                    style={{
                                        flex: "1 1 0px", maxWidth: 52, height: 36, borderRadius: 6,
                                        border: "2px solid transparent",
                                        background: "transparent"
                                    }}
                                />
                            );
                        }
                        return (
                            <div
                                key={actualIdx}
                                onClick={() => setActiveIdx(actualIdx)}
                                style={{
                                    flex: "1 1 0px", maxWidth: 52, height: 36, borderRadius: 6,
                                    overflow: "hidden", cursor: "pointer",
                                    border: actualIdx === activeIdx ? "2px solid #0ea5e9" : "2px solid transparent",
                                    background: "#0f172a", transition: "border 0.2s"
                                }}
                            >
                                <img
                                    src={`${apiBase}/data/precalib/${img.filename}`}
                                    alt=""
                                    style={{ width: "100%", height: "100%", objectFit: "cover" }}
                                    onError={e => { e.target.style.display = "none"; }}
                                />
                            </div>
                        );
                    })}
                </div>

                {/* Right arrow */}
                <button
                    onClick={handleNextThumb}
                    disabled={!canScrollRight}
                    onMouseEnter={() => setRightHover(true)}
                    onMouseLeave={() => setRightHover(false)}
                    style={{
                        background: !canScrollRight
                            ? "rgba(15, 23, 42, 0.4)"
                            : rightHover
                            ? "rgba(15, 23, 42, 0.95)"
                            : "rgba(15, 23, 42, 0.8)",
                        border: "1px solid rgba(255, 255, 255, 0.15)",
                        borderRadius: 6,
                        width: 24,
                        height: 36,
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        color: !canScrollRight ? "rgba(255, 255, 255, 0.2)" : "#fff",
                        cursor: !canScrollRight ? "not-allowed" : "pointer",
                        transition: "all 0.2s",
                        padding: 0,
                    }}
                >
                    <ChevronRight size={16} />
                </button>
            </div>
        </div>
    );
}

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
    const MAX_BAR_HEIGHT = 64;

    return (
        <div style={{ display: "flex", flexDirection: "column", height: "100%", overflow: "hidden" }}>
            <div style={{ flex: 1, display: "flex", alignItems: "flex-end", gap: 6, minHeight: 0, overflow: "hidden", paddingBottom: 2 }}>
                {data.map((item, i) => {
                    const barH = Math.max(Math.round((item.value / max) * MAX_BAR_HEIGHT), 4);
                    const color = barColors[i % barColors.length];
                    return (
                        <div key={i} style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center", gap: 2, minWidth: 0 }}>
                            <span style={{ fontSize: 10, fontWeight: 700, color: "#334155", lineHeight: 1 }}>{item.value}</span>
                            <div style={{ width: "100%", height: barH, background: `linear-gradient(180deg, ${color}, ${color}bb)`, borderRadius: "3px 3px 0 0", transition: "height 0.6s ease" }} />
                        </div>
                    );
                })}
            </div>
            <div style={{ display: "flex", gap: 6, borderTop: "1px solid #e2e8f0", paddingTop: 4, flexShrink: 0 }}>
                {data.map((item, i) => (
                    <div key={i} style={{ flex: 1, textAlign: "center", fontSize: 8, color: "#64748b", wordBreak: "break-word", lineHeight: 1.2, minWidth: 0 }}>
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
// InfoPanel — Thông tin đợt kiểm tra có pagination + mũi tên
// ─────────────────────────────────────────
// Phân bổ cố định: trang 1 = 4 mục, trang 2 = 4 mục, trang 3 = 4 mục
const PAGE_SIZES = [4, 4, 4];

function InfoPanel({ rows, onEdit, isAnyLoading }) {
    const [page, setPage] = useState(0);

    // Chia rows theo phân bổ cố định PAGE_SIZES
    const pages = [];
    let offset = 0;
    for (const size of PAGE_SIZES) {
        const chunk = rows.slice(offset, offset + size);
        if (chunk.length > 0) pages.push(chunk);
        offset += size;
        if (offset >= rows.length) break;
    }

    const totalPages = pages.length;
    const currentRows = pages[page] || [];
    const hasPrev = page > 0;
    const hasNext = page < totalPages - 1;

    return (
        <div style={{ background: "#fff", borderRadius: 10, border: "1px solid #e2e8f0", padding: 16, boxShadow: "0 1px 4px rgba(0,0,0,0.03)", display: "flex", flexDirection: "column", gap: 10, height: "100%", boxSizing: "border-box" }}>
            {/* Header + mũi tên điều hướng */}
            <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                <Settings size={15} color="#6366f1" />
                <span style={{ fontSize: 12, fontWeight: 700, color: "#334155", textTransform: "uppercase", letterSpacing: "0.5px", flex: 1 }}>Thông tin đợt kiểm tra</span>
                {/* Mũi tên điều hướng trang */}
                <div style={{ display: "flex", alignItems: "center", gap: 3 }}>
                    {totalPages > 1 && (
                        <span style={{ fontSize: 10, color: "#94a3b8", fontWeight: 600, marginRight: 1 }}>
                            {page + 1}/{totalPages}
                        </span>
                    )}
                    <button
                        onClick={() => setPage(p => Math.max(0, p - 1))}
                        disabled={!hasPrev}
                        title="Trang trước"
                        style={{
                            width: 24, height: 24, borderRadius: 5,
                            border: "1px solid #e2e8f0",
                            background: hasPrev ? "#f8fafc" : "#f1f5f9",
                            color: hasPrev ? "#334155" : "#cbd5e1",
                            fontSize: 12, cursor: hasPrev ? "pointer" : "default",
                            display: "flex", alignItems: "center", justifyContent: "center",
                            transition: "all 0.15s", padding: 0,
                        }}
                    >
                        ←
                    </button>
                    <button
                        onClick={() => setPage(p => Math.min(totalPages - 1, p + 1))}
                        disabled={!hasNext}
                        title="Trang tiếp theo"
                        style={{
                            width: 24, height: 24, borderRadius: 5,
                            border: "1px solid #e2e8f0",
                            background: hasNext ? "#0ea5e9" : "#f1f5f9",
                            color: hasNext ? "#fff" : "#cbd5e1",
                            fontSize: 12, cursor: hasNext ? "pointer" : "default",
                            display: "flex", alignItems: "center", justifyContent: "center",
                            transition: "all 0.15s", padding: 0,
                            boxShadow: hasNext ? "0 1px 4px rgba(14,165,233,0.2)" : "none",
                        }}
                    >
                        →
                    </button>
                </div>
            </div>

            {/* Lưới hiển thị thông tin trang hiện tại */}
            <div style={{ display: "flex", flexDirection: "column", gap: "8px", flex: 1 }}>
                {currentRows.map((row, idx) => (
                    <div key={idx} style={{
                        display: "flex",
                        alignItems: "center",
                        gap: 8,
                        padding: "8px 12px",
                        borderRadius: 6,
                        background: "#f8fafc",
                        border: "1px solid #f1f5f9",
                        width: "100%",
                        boxSizing: "border-box",
                        flex: 1,
                        minHeight: 0
                    }}>
                        <span style={{ fontSize: 14, flexShrink: 0 }}>{row.icon}</span>
                        <div style={{ minWidth: 0, flex: 1 }}>
                            <div style={{ fontSize: 10, color: "#94a3b8", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.4px", marginBottom: 1 }}>{row.label}</div>
                            {/* Hiển thị đầy đủ nội dung — wrap nếu dài, không cắt bỏ */}
                            <div style={{ fontSize: 12, color: "#334155", fontWeight: 600, lineHeight: 1.3, wordBreak: "break-word" }}>
                                {row.value}
                            </div>
                        </div>
                    </div>
                ))}
            </div>

            {/* Dot indicators (nếu > 1 trang) */}
            {totalPages > 1 && (
                <div style={{ display: "flex", justifyContent: "center", gap: 4, marginTop: -2 }}>
                    {pages.map((_, pi) => (
                        <button
                            key={pi}
                            onClick={() => setPage(pi)}
                            style={{
                                width: pi === page ? 12 : 4, height: 4,
                                borderRadius: 2, border: "none",
                                background: pi === page ? "#0ea5e9" : "#cbd5e1",
                                cursor: "pointer", padding: 0,
                                transition: "all 0.2s",
                            }}
                        />
                    ))}
                </div>
            )}

            {/* Nút Chỉnh sửa */}
            <button
                onClick={onEdit}
                disabled={isAnyLoading}
                style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 6, padding: "8px 0", borderRadius: 6, border: "none", background: "linear-gradient(135deg,#0ea5e9,#6366f1)", color: "#fff", fontSize: 12, fontWeight: 700, cursor: isAnyLoading ? "not-allowed" : "pointer", boxShadow: "0 2px 6px rgba(14,165,233,0.2)", transition: "all 0.2s", opacity: isAnyLoading ? 0.7 : 1 }}
            >
                <Settings size={13} fill="white" /> Chỉnh sửa thông tin
            </button>
        </div>
    );
}

// ─────────────────────────────────────────
// Home Page
// ─────────────────────────────────────────
export default function Home({ data, batchId, onAnalysisComplete, onReset, onViewOnMap }) {
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

    // Helper: classify defect class name into canonical group
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

    // ── Màu sắc chuẩn cho từng nhóm lỗi (phân biệt hotspot đơn/đa) ──
    const FAULT_COLORS = {
        "hotspot single cell": "#ef4444",   // Đỏ tươi — Hotspot đơn
        "hotspot multi cell":  "#ff6b35",   // Cam — Hotspot đa
        "crack":               "#f59e0b",   // Vàng cam — Crack
        "shading":             "#8b5cf6",   // Tím — Shading
        "diode":               "#06b6d4",   // Cyan — Diode
    };
    const getFaultColor = (group) => FAULT_COLORS[group.toLowerCase()] || "#94a3b8";

    // Tên hiển thị đẹp hơn cho từng nhóm
    const FAULT_LABELS = {
        "hotspot single cell": "hotspot single cell",
        "hotspot multi cell":  "hotspot multi_cell",
        "crack":               "Crack",
        "shading":             "Shading",
        "diode":               "Diode",
    };
    const getFaultLabel = (group) => FAULT_LABELS[group.toLowerCase()] || group;

    // ── Tính công suất hao hụt theo từng nhóm lỗi ──
    // Backend lưu power_loss_w ở cấp panel — phân bổ theo tỷ lệ số defect từng nhóm trong panel
    const defectPowerLossW = {};
    (data || []).forEach((img) => {
        (img.panels || []).forEach((p) => {
            const reviewStatus = p.review_status || "unreviewed";
            const include = reviewStatus === "false_positive" ? false : (p.include_in_report !== undefined ? p.include_in_report : true);
            if (!include) return;

            const panelLossW = Number(p.power_loss_w ?? p.total_panel_loss ?? 0);
            const defects = p.defects || [];
            if (defects.length === 0) return;

            // Đếm số defect theo nhóm trong panel này
            const panelGroupCounts = {};
            defects.forEach((d) => {
                const group = classifyDefect(d.class_name || d.type || "");
                panelGroupCounts[group] = (panelGroupCounts[group] || 0) + 1;
            });

            const totalDefectsInPanel = Object.values(panelGroupCounts).reduce((s, v) => s + v, 0);
            if (totalDefectsInPanel === 0) return;

            // Phân bổ power_loss_w theo tỷ lệ số defect
            Object.entries(panelGroupCounts).forEach(([group, cnt]) => {
                const share = (cnt / totalDefectsInPanel) * panelLossW;
                defectPowerLossW[group] = (defectPowerLossW[group] || 0) + share;
            });
        });
    });

    // Xác định đơn vị hiển thị phù hợp
    const totalLossW = Object.values(defectPowerLossW).reduce((s, v) => s + v, 0);
    let lossUnit, lossDivisor;
    if (totalLossW >= 1_000_000) {
        lossUnit = "MWp"; lossDivisor = 1_000_000;
    } else if (totalLossW >= 1_000) {
        lossUnit = "kWp"; lossDivisor = 1_000;
    } else {
        lossUnit = "W"; lossDivisor = 1;
    }

    const powerLossChartData = Object.entries(defectPowerLossW)
        .filter(([, v]) => v > 0)
        .sort((a, b) => b[1] - a[1])
        .map(([group, valueW]) => ({
            label: getFaultLabel(group),
            rawGroup: group,
            value: valueW / lossDivisor,
            color: getFaultColor(group),
            unit: lossUnit,
        }));

    // ── Dữ liệu Donut chart (dùng defectCounts đã phân nhóm ở trên) ──
    const donutData = Object.entries(defectCounts)
        .sort((a, b) => b[1] - a[1])
        .map(([group, value]) => ({
            label: getFaultLabel(group),
            rawGroup: group,
            value,
            color: getFaultColor(group),
        }));

    const totalFaultCount = donutData.reduce((s, d) => s + d.value, 0);
    const estimatedLossMWp = estimatedLoss / 1_000_000;

    return (
        <div style={{ fontFamily: "'Inter', 'DM Sans', system-ui, sans-serif" }}>
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

            {/* ── HIDDEN FILE INPUTS ── */}
            <input type="file" accept=".zip,.rar,.jpg,.jpeg,.png,image/*" multiple ref={fileInputRef} onChange={handleUploadFiles} style={{ display: "none" }} />
            <input type="file" webkitdirectory="true" directory="" multiple ref={folderInputRef} onChange={handleUploadFiles} style={{ display: "none" }} />
            <input type="file" accept=".pt" ref={modelInputRef} onChange={handleUpdateModel} style={{ display: "none" }} />

            {/* ── HEADER ── */}
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16, flexWrap: "wrap", gap: 8 }}>
                <div>
                    <h1 style={{ margin: 0, fontSize: 20, fontWeight: 700, color: "#0f172a" }}>Bảng điều khiển</h1>
                    {projectMetadata.projectName && (
                        <p style={{ margin: "2px 0 0", fontSize: 12, color: "#64748b", fontWeight: 500 }}>
                            Dự án: {projectMetadata.projectName}
                        </p>
                    )}
                </div>
                <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                    {/* Thay model AI */}
                    <button onClick={() => modelInputRef.current?.click()} disabled={isAnyLoading}
                        style={{ display: "flex", alignItems: "center", gap: 4, padding: "7px 12px", borderRadius: 6, border: "1px solid rgba(14,165,233,0.4)", background: "rgba(14,165,233,0.06)", color: "#0ea5e9", fontSize: 12, fontWeight: 600, cursor: "pointer", transition: "all 0.2s" }}>
                        {isUpdatingModel ? <Loader2 size={14} className="animate-spin" /> : <Cpu size={14} />}
                        {isUpdatingModel ? "Đang cập nhật..." : "Thay model AI"}
                    </button>

                    {/* Phân tích lại */}
                    <button onClick={handleReanalyze} disabled={isAnyLoading || totalPanels === 0}
                        style={{ display: "flex", alignItems: "center", gap: 4, padding: "7px 12px", borderRadius: 6, border: "none", background: "linear-gradient(135deg,#f59e0b,#ef4444)", color: "#fff", fontSize: 12, fontWeight: 600, cursor: (isAnyLoading || totalPanels === 0) ? "not-allowed" : "pointer", opacity: totalPanels === 0 ? 0.5 : 1, transition: "all 0.2s" }}>
                        {isReanalyzing ? <Loader2 size={14} className="animate-spin" /> : <RefreshCw size={14} />}
                        {isReanalyzing ? "Đang chạy lại..." : "Phân tích lại"}
                    </button>

                    {/* Reset */}
                    <button onClick={handleSystemReset} disabled={isAnyLoading}
                        style={{ display: "flex", alignItems: "center", gap: 4, padding: "7px 12px", borderRadius: 6, border: "1px solid rgba(239,68,68,0.3)", background: "#fff1f1", color: "#ef4444", fontSize: 12, fontWeight: 600, cursor: isAnyLoading ? "not-allowed" : "pointer", transition: "all 0.2s" }}>
                        {isResetting ? <Loader2 size={14} className="animate-spin" /> : <Trash2 size={14} />}
                        {isResetting ? "Đang reset..." : "Reset Hệ thống"}
                    </button>
                </div>
            </div>

            {/* ── KPI CARDS ── */}
            <div className="kpi-grid-container">
                {[
                    { icon: <Image size={18} />, label: "TỔNG ẢNH UAV", value: (data?.length || 0).toLocaleString(), unit: "ảnh", accent: "#8b5cf6" },
                    { icon: <LayoutGrid size={18} />, label: "TỔNG PANEL", value: totalPanels.toLocaleString(), unit: "panel", accent: "#0ea5e9" },
                    { icon: <AlertCircle size={18} />, label: "PANEL LỖI", value: totalFaults.toLocaleString(), unit: "panel lỗi", accent: "#ef4444" },
                    { icon: <Zap size={18} />, label: "CÔNG SUẤT HAO HỤT ƯỚC TÍNH", value: estimatedLossMWp >= 1 ? estimatedLossMWp.toFixed(2) : (estimatedLoss / 1000).toFixed(2), unit: estimatedLossMWp >= 1 ? "MWp" : "kWp", accent: "#f59e0b" },
                ].map((card, i) => (
                    <div key={i} style={{ background: "#fff", borderRadius: 10, padding: "12px 16px", border: "1px solid #e2e8f0", borderTop: `2px solid ${card.accent}`, boxShadow: "0 1px 4px rgba(0,0,0,0.03)", transition: "box-shadow 0.2s" }}
                        onMouseEnter={e => e.currentTarget.style.boxShadow = `0 4px 12px ${card.accent}15`}
                        onMouseLeave={e => e.currentTarget.style.boxShadow = "0 1px 4px rgba(0,0,0,0.03)"}>
                        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
                            <p style={{ margin: 0, fontSize: 11, fontWeight: 700, color: "#94a3b8", textTransform: "uppercase", letterSpacing: "0.5px" }}>{card.label}</p>
                            <div style={{ width: 30, height: 30, borderRadius: 6, background: `${card.accent}12`, display: "flex", alignItems: "center", justifyContent: "center", color: card.accent }}>{card.icon}</div>
                        </div>
                        <div style={{ display: "flex", alignItems: "baseline", gap: 4 }}>
                            <span style={{ fontSize: 24, fontWeight: 800, color: "#0f172a", lineHeight: 1 }}>{card.value}</span>
                            <span style={{ fontSize: 11, color: "#64748b", fontWeight: 500 }}>{card.unit}</span>
                        </div>
                    </div>
                ))}
            </div>

            {/* ── MIDDLE 3-COLUMN SECTION ── */}
            <div className="middle-grid-container">
                {/* Col 1: TẢI DỮ LIỆU UAV */}
                <div style={{ background: "#fff", borderRadius: 10, border: "1px solid #e2e8f0", padding: 16, boxShadow: "0 1px 4px rgba(0,0,0,0.03)", display: "flex", flexDirection: "column", gap: 10, height: "100%", boxSizing: "border-box" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                        <Upload size={15} color="#0ea5e9" />
                        <span style={{ fontSize: 12, fontWeight: 700, color: "#334155", textTransform: "uppercase", letterSpacing: "0.5px" }}>Tải dữ liệu UAV</span>
                    </div>

                    {/* Drop zone */}
                    <div
                        style={{ border: "2px dashed #cbd5e1", borderRadius: 10, padding: "20px 16px", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 8, background: "#f8fafc", transition: "all 0.2s", flex: 1 }}
                        onMouseEnter={e => { e.currentTarget.style.borderColor = "#0ea5e9"; e.currentTarget.style.background = "#f0f9ff"; }}
                        onMouseLeave={e => { e.currentTarget.style.borderColor = "#cbd5e1"; e.currentTarget.style.background = "#f8fafc"; }}
                        onDragOver={e => e.preventDefault()}
                        onDrop={e => { e.preventDefault(); if (!isAnyLoading && e.dataTransfer.files.length > 0) { const dt = e.dataTransfer; handleUploadFiles({ target: { files: dt.files } }); } }}
                    >
                        <div style={{ width: 36, height: 36, borderRadius: "50%", background: "rgba(14,165,233,0.1)", display: "flex", alignItems: "center", justifyContent: "center" }}>
                            <Upload size={18} color="#0ea5e9" />
                        </div>
                        <p style={{ margin: 0, fontSize: 11, color: "#475569", fontWeight: 500, textAlign: "center", lineHeight: 1.3 }}>
                            Kéo thả thư mục hoặc file Zip/Rar vào đây<br /><span style={{ fontSize: 10, color: "#94a3b8" }}>hoặc chọn hình thức tải lên</span>
                        </p>
                        <div style={{ display: "flex", gap: 6, width: "100%", justifyContent: "center", marginTop: 2 }}>
                            <button
                                onClick={e => { e.stopPropagation(); if (!isAnyLoading) fileInputRef.current?.click(); }}
                                disabled={isAnyLoading}
                                style={{ padding: "6px 12px", borderRadius: 5, border: "1px solid #cbd5e1", background: "#fff", color: "#334155", fontSize: 11, fontWeight: 600, cursor: isAnyLoading ? "not-allowed" : "pointer", display: "flex", alignItems: "center", gap: 3 }}
                            >
                                📄 Tải file
                            </button>
                            <button
                                onClick={e => { e.stopPropagation(); if (!isAnyLoading) folderInputRef.current?.click(); }}
                                disabled={isAnyLoading}
                                style={{ padding: "6px 12px", borderRadius: 5, border: "none", background: "#0ea5e9", color: "#fff", fontSize: 11, fontWeight: 600, cursor: isAnyLoading ? "not-allowed" : "pointer", display: "flex", alignItems: "center", gap: 3 }}
                            >
                                📁 Thư mục
                            </button>
                        </div>
                    </div>

                    {/* Upload progress */}
                    {isUploading && (
                        <div>
                            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                                <span style={{ fontSize: 12, color: "#475569", fontWeight: 500 }}>Đang tải lên...</span>
                                <span style={{ fontSize: 12, color: "#0ea5e9", fontWeight: 600 }}>100%</span>
                            </div>
                            <div style={{ height: 6, background: "#e2e8f0", borderRadius: 6, overflow: "hidden" }}>
                                <div style={{ height: "100%", width: "100%", background: "linear-gradient(90deg,#0ea5e9,#6366f1)", borderRadius: 6, animation: "shimmerLoad 1.5s infinite" }} />
                            </div>
                        </div>
                    )}

                    {/* Format info */}
                    <div style={{ fontSize: 12, color: "#94a3b8" }}>
                        <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 3 }}>
                            <CheckCircle size={14} color={data?.length > 0 ? "#10b981" : "#94a3b8"} />
                            <span style={{ color: data?.length > 0 ? "#10b981" : "#64748b", fontWeight: 500 }}>
                                {data?.length > 0 ? `✓ Đã tải ${data.length} ảnh · Tổng: ${data.length} ảnh` : "Hỗ trợ định dạng: JPG, PNG, TIF, RJPG"}
                            </span>
                        </div>
                        {data?.length > 0 && <div style={{ color: "#0ea5e9", fontWeight: 500 }}>Đã tải lên &amp; phân tích AI thành công · 100%</div>}
                    </div>
                </div>

                {/* Col 2: THÔNG TIN ĐỢT KIỂM TRA — với pagination */}
                {(() => {
                    const infoRows = [
                        { icon: "🏭", label: "Tên dự án", value: projectMetadata.projectName || "—", fullWidth: true },
                        { icon: "📅", label: "Ngày kiểm tra", value: projectMetadata.scanTime || "—" },
                        { icon: "📍", label: "Địa điểm", value: projectMetadata.location || "—" },
                        { icon: "🚁", label: "UAV", value: projectMetadata.device || "—" },
                        { icon: "🏢", label: "Đơn vị quét", value: projectMetadata.operator || "—" },
                        { icon: "👤", label: "Người vận hành", value: projectMetadata.supervisor || "—" },
                        { icon: "⚡", label: "Công suất pin", value: projectMetadata.panelPower ? `${projectMetadata.panelPower} W` : "—" },
                        { icon: "🔌", label: "Công suất hệ thống", value: projectMetadata.systemCapacity || "—" },
                        { icon: "📊", label: "Loại dữ liệu", value: projectMetadata.dataType || "—" },
                        { icon: "🤖", label: "Model AI", value: projectMetadata.aiModel || "—" },
                        { icon: "💻", label: "Hệ thống", value: projectMetadata.systemVersion || "—" },
                        { icon: "📝", label: "Ghi chú", value: projectMetadata.notes || "Kiểm tra định kỳ tháng 6", fullWidth: true },
                    ];
                    return <InfoPanel rows={infoRows} onEdit={() => setShowMetadataModal(true)} isAnyLoading={isAnyLoading} />;
                })()}

                {/* Col 3: ẢNH UAV ĐÃ TẢI LÊN */}
                <div className="middle-card-uav" style={{ background: "#fff", borderRadius: 10, border: "1px solid #e2e8f0", padding: 16, boxShadow: "0 1px 4px rgba(0,0,0,0.03)", display: "flex", flexDirection: "column", gap: 8, boxSizing: "border-box" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                        <Image size={15} color="#f59e0b" />
                        <span style={{ fontSize: 12, fontWeight: 700, color: "#334155", textTransform: "uppercase", letterSpacing: "0.5px" }}>Ảnh UAV đã tải lên</span>
                        {data?.length > 0 && <span style={{ marginLeft: "auto", fontSize: 11, color: "#64748b", fontWeight: 500 }}>{data.length} ảnh</span>}
                    </div>
                    <div style={{ flex: 1, minHeight: 0 }}>
                        <ImageGallery images={data || []} apiBase={API} />
                    </div>
                </div>
            </div>

            {/* ── BOTTOM 2-COLUMN SECTION ── */}
            <div className="bottom-grid-container">
                {/* Col 1: THỐNG KÊ LỖI THEO LOẠI */}
                <div style={{ background: "#fff", borderRadius: 10, border: "1px solid #e2e8f0", padding: 16, boxShadow: "0 1px 4px rgba(0,0,0,0.03)" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 10 }}>
                        <AlertCircle size={15} color="#ef4444" />
                        <span style={{ fontSize: 12, fontWeight: 700, color: "#334155", textTransform: "uppercase", letterSpacing: "0.5px" }}>Thống kê lỗi theo loại</span>
                    </div>

                    <div style={{ display: "flex", gap: 16, alignItems: "center" }}>
                        {/* Donut */}
                        <div style={{ flexShrink: 0 }}>
                            <DonutChart data={donutData} totalLabel={totalFaultCount.toLocaleString()} />
                        </div>

                        {/* Legend table */}
                        <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: 4 }}>
                            <div style={{ display: "grid", gridTemplateColumns: "auto 1fr auto auto", gap: "2px 8px", fontSize: 11, color: "#94a3b8", fontWeight: 600, paddingBottom: 4, borderBottom: "1px solid #f1f5f9" }}>
                                <span>Loại lỗi</span><span></span><span>Số lượng</span><span>Tỷ lệ</span>
                            </div>
                            {donutData.length === 0 ? (
                                <div style={{ color: "#94a3b8", fontSize: 12, textAlign: "center", paddingTop: 8 }}>Chưa có dữ liệu</div>
                            ) : donutData.map((d, i) => (
                                <div key={i} style={{ display: "grid", gridTemplateColumns: "10px 1fr auto auto", gap: "0 8px", alignItems: "center", fontSize: 12, padding: "2px 0" }}>
                                    <div style={{ width: 10, height: 10, borderRadius: 2, background: d.color, flexShrink: 0 }} />
                                    <span style={{ color: "#334155", fontWeight: 500, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{d.label}</span>
                                    <span style={{ color: "#0f172a", fontWeight: 700, textAlign: "right" }}>{d.value.toLocaleString()}</span>
                                    <span style={{ color: "#64748b", textAlign: "right", minWidth: 36 }}>{((d.value / Math.max(totalFaultCount, 1)) * 100).toFixed(1)}%</span>
                                </div>
                            ))}
                        </div>
                    </div>

                    <button
                        onClick={onViewOnMap}
                        style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 4, width: "100%", padding: "7px 0", marginTop: 12, borderRadius: 6, border: "1px solid #e2e8f0", background: "transparent", color: "#64748b", fontSize: 12, fontWeight: 600, cursor: "pointer" }}
                    >
                        Xem chi tiết <ChevronRight size={12} />
                    </button>
                </div>

                {/* Col 2: CÔNG SUẤT HAO HỤT THEO LOẠI LỖI */}
                <div style={{ background: "#fff", borderRadius: 10, border: "1px solid #e2e8f0", padding: 16, boxShadow: "0 1px 4px rgba(0,0,0,0.03)", display: "flex", flexDirection: "column" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 10 }}>
                        <Zap size={15} color="#f59e0b" />
                        <span style={{ fontSize: 12, fontWeight: 700, color: "#334155", textTransform: "uppercase", letterSpacing: "0.5px" }}>Công suất hao hụt theo loại lỗi</span>
                        {lossUnit && powerLossChartData.length > 0 && (
                            <span style={{ marginLeft: "auto", fontSize: 11, color: "#94a3b8", fontWeight: 500 }}>Đơn vị: {lossUnit}</span>
                        )}
                    </div>
                    <div style={{ flex: 1, minHeight: 140 }}>
                        <PowerLossBarChart data={powerLossChartData} />
                    </div>
                    {powerLossChartData.length > 0 ? (
                        <div style={{ marginTop: 8, padding: "8px 12px", borderRadius: 6, background: "#fffbeb", border: "1px solid #fde68a", fontSize: 11, color: "#92400e", display: "flex", alignItems: "center", gap: 4 }}>
                            <Zap size={12} color="#f59e0b" />
                            <span>
                                <b style={{ color: powerLossChartData[0]?.color }}>{powerLossChartData[0]?.label}</b>
                                {" "}là lỗi gây thất thoát lớn nhất —{" "}
                                <b>{powerLossChartData[0]?.value?.toFixed(lossUnit === "W" ? 0 : 2)} {lossUnit}</b>
                            </span>
                        </div>
                    ) : (
                        <div style={{ marginTop: 8, padding: "8px 12px", borderRadius: 6, background: "#f8fafc", border: "1px solid #e2e8f0", fontSize: 11, color: "#94a3b8", textAlign: "center" }}>
                            Chưa có dữ liệu công suất hao hụt
                        </div>
                    )}
                </div>

            </div>

            <style>{`
                @keyframes shimmerLoad {
                    0% { transform: translateX(-100%); }
                    100% { transform: translateX(100%); }
                }

                /* KPI Grid */
                .kpi-grid-container {
                    display: grid;
                    grid-template-columns: repeat(4, 1fr);
                    gap: 12px;
                    margin-bottom: 16px;
                }
                @media (max-width: 1200px) {
                    .kpi-grid-container {
                        grid-template-columns: repeat(2, 1fr);
                    }
                }
                @media (max-width: 640px) {
                    .kpi-grid-container {
                        grid-template-columns: 1fr;
                    }
                }

                /* Middle 3-Column Grid */
                .middle-grid-container {
                    display: grid;
                    grid-template-columns: 1fr 1fr 1fr;
                    gap: 16px;
                    margin-bottom: 16px;
                }
                @media (max-width: 1024px) {
                    .middle-grid-container {
                        grid-template-columns: 1fr;
                    }
                }

                /* Middle Card (UAV Image Gallery Card) */
                .middle-card-uav {
                    height: 400px;
                }
                @media (max-height: 900px), (max-width: 1400px) {
                    .middle-card-uav {
                        height: 350px;
                    }
                }
                @media (max-height: 768px), (max-width: 1200px) {
                    .middle-card-uav {
                        height: 320px;
                    }
                }
                @media (max-width: 1024px) {
                    .middle-card-uav {
                        height: 350px;
                    }
                }

                /* Bottom 2-Column Grid */
                .bottom-grid-container {
                    display: grid;
                    grid-template-columns: 1fr 1.6fr;
                    gap: 16px;
                }
                @media (max-width: 1024px) {
                    .bottom-grid-container {
                        grid-template-columns: 1fr;
                    }
                }
            `}</style>
        </div>
    );
}
