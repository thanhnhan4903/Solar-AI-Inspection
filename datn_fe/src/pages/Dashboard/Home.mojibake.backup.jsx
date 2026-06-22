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
    XCircle,
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

const API = "";

// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
// Mini bar chart component
// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
// Donut Chart component (SVG pure)
function DonutChart({ data, totalLabel }) {
    if (!data || data.length === 0) {
        return (
            <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: 110, color: "#94a3b8", flexDirection: "column", gap: 6 }}>
                <TrendingUp size={22} style={{ opacity: 0.3 }} />
                <span style={{ fontSize: 11 }}>ChÆ°a cÃ³ dá»¯ liá»‡u</span>
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
                tá»•ng lá»—i
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
                <span style={{ fontSize: 11 }}>ChÆ°a cÃ³ dá»¯ liá»‡u phÃ¢n tÃ­ch</span>
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
function ImageGallery({ images, apiBase, activeIdx, setActiveIdx }) {
    const [thumbPage, setThumbPage] = useState(0);
    const [leftHover, setLeftHover] = useState(false);
    const [rightHover, setRightHover] = useState(false);

    const PAGE_SIZE = 9;
    const totalPages = Math.ceil(images.length / PAGE_SIZE);

    React.useEffect(() => {
        const activePage = Math.floor(activeIdx / PAGE_SIZE);
        setThumbPage(activePage);
    }, [activeIdx]);

    if (!images || images.length === 0) {
        return (
            <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", height: "100%", gap: 10, color: "#94a3b8" }}>
                <Image size={36} style={{ opacity: 0.3 }} />
                <span style={{ fontSize: 12 }}>ChÆ°a cÃ³ áº£nh nÃ o Ä‘Æ°á»£c táº£i lÃªn</span>
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
                        }}>â€¹</button>
                        <button onClick={() => setActiveIdx(i => Math.min(images.length - 1, i + 1))} style={{
                            position: "absolute", right: 6, top: "50%", transform: "translateY(-50%)",
                            background: "rgba(0,0,0,0.5)", border: "none", borderRadius: "50%",
                            width: 28, height: 28, display: "flex", alignItems: "center", justifyContent: "center",
                            cursor: "pointer", color: "#fff", fontSize: 14
                        }}>â€º</button>
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
                <p style={{ margin: 0, fontSize: 12 }}>ChÆ°a cÃ³ dá»¯ liá»‡u phÃ¢n tÃ­ch</p>
            </div>
        );
    }

    const max = Math.max(...data.map((d) => d.value), 1);
    const barColors = ["#f97316", "#ef4444", "#06b6d4", "#eab308", "#a855f7", "#10b981", "#0ea5e9", "#f59e0b"];
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

// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
// AI Progress Modal
// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
function AIProgressModal({ progress, onClose }) {
    const pct = progress.percent;

    // TÃ­nh toÃ¡n thá»i gian chá» Ä‘á»£i cÃ²n láº¡i (ETA) chuyÃªn nghiá»‡p vÃ  á»•n Ä‘á»‹nh
    let etaText = "Äang tÃ­nh...";
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
        etaText = "HoÃ n táº¥t!";
    } else if (status === "error") {
        etaText = "Tháº¥t báº¡i";
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
                        {progress.status === "error" ? "Lá»—i phÃ¢n tÃ­ch AI" : "Äang phÃ¢n tÃ­ch AI..."}
                    </h3>
                    <p style={{ margin: 0, color: progress.status === "error" ? "#ef4444" : "#64748B", fontSize: 13 }}>
                        {progress.status === "error" ? (progress.error || progress.stage) : progress.stage}
                    </p>
                </div>

                {/* Progress bar */}
                <div style={{ width: "100%" }}>
                    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8 }}>
                        <span style={{ fontSize: 12, color: "#94A3B8" }}>
                            {progress.totalImages > 0 ? `áº¢nh ${progress.processedImages} / ${progress.totalImages}` : "Chuáº©n bá»‹..."}
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
                            ÄÃ£ cháº¡y: <b style={{ color: "#CBD5E1" }}>{progress.elapsedSeconds}s</b>
                        </span>
                        <span style={{ fontSize: 11, color: "#64748B" }}>
                            CÃ²n láº¡i (Æ°á»›c tÃ­nh): <b style={{ color: progress.status === "error" ? "#ef4444" : "#f59e0b" }}>{etaText}</b>
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
                            Tá»‡p Ä‘ang xá»­ lÃ½
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
                        ÄÃ³ng
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

// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
// Quality Review Modal
// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
function QualityReviewModal({ qualityData, onConfirm, onCancel, isRunningAI }) {
    const { overall_status, quality_report = [] } = qualityData;
    const [expandedIdx, setExpandedIdx] = useState(null);

    const TIER = {
        ok: {
            color: "#10b981",
            bg: "rgba(16,185,129,0.12)",
            icon: <CheckCircle size={14} />,
            label: "Tá»‘t",
        },
        warning: {
            color: "#f59e0b",
            bg: "rgba(245,158,11,0.12)",
            icon: <AlertTriangle size={14} />,
            label: "Cáº£nh bÃ¡o",
        },
        poor: {
            color: "#ef4444",
            bg: "rgba(239,68,68,0.12)",
            icon: <XCircle size={14} />,
            label: "KÃ©m",
        },
        error: {
            color: "#64748b",
            bg: "rgba(100,116,139,0.12)",
            icon: <XCircle size={14} />,
            label: "Lá»—i",
        },
    };

    const overallTier = TIER[overall_status] || TIER.warning;

    const counts = quality_report.reduce((acc, r) => {
        acc[r.quality_status] = (acc[r.quality_status] || 0) + 1;
        return acc;
    }, {});

    const headerMsg =
        {
            ok: "âœ… Táº¥t cáº£ áº£nh Ä‘áº¡t cháº¥t lÆ°á»£ng tá»‘t. Báº¡n cÃ³ thá»ƒ tiáº¿p tá»¥c phÃ¢n tÃ­ch AI!",
            warning:
                "âš ï¸ Má»™t sá»‘ áº£nh cÃ³ dáº¥u hiá»‡u báº¥t thÆ°á»ng. Báº¡n váº«n cÃ³ thá»ƒ tiáº¿p tá»¥c AI hoáº·c chá»¥p láº¡i Ä‘á»ƒ káº¿t quáº£ tá»‘t hÆ¡n.",
            poor: "ðŸ”´ áº¢nh cÃ³ cháº¥t lÆ°á»£ng kÃ©m. Khuyáº¿n nghá»‹ chá»¥p láº¡i. Báº¡n váº«n cÃ³ thá»ƒ bá» qua vÃ  tiáº¿p tá»¥c AI.",
        }[overall_status] || "Xá»­ lÃ½ hoÃ n táº¥t. Kiá»ƒm tra chi tiáº¿t bÃªn dÆ°á»›i.";

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
                                    Káº¿t Quáº£ Tiá»n Xá»­ LÃ½ áº¢nh
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
                                    {t.icon} {n} áº£nh {t.label.toLowerCase()}
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
                                                KhÃ´ng phÃ¡t hiá»‡n váº¥n Ä‘á»
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
                        âœ• Há»§y (Chá»¥p Láº¡i)
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
                                <Loader2 size={16} className="animate-spin" /> Äang phÃ¢n tÃ­ch AI...
                            </>
                        ) : (
                            <>
                                <Play size={14} fill="white" /> Cháº¡y AI PhÃ¢n TÃ­ch
                            </>
                        )}
                    </button>
                </div>
            </div>
        </div>
    );
}


// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
// Project Metadata & Panel Power Modal
// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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
                            {isNewProject ? "ThÃ´ng Tin Dá»± Ãn Má»›i" : "Cáº¥u HÃ¬nh ThÃ´ng Tin Dá»± Ãn"}
                        </h3>
                        <p style={{ margin: "4px 0 0 0", color: "#64748b", fontSize: 12 }}>
                            {isNewProject ? "Nháº­p metadata dá»± Ã¡n trÆ°á»›c khi cháº¡y AI phÃ¢n tÃ­ch" : "Chá»‰nh sá»­a thÃ´ng tin dá»± Ã¡n & cÃ´ng suáº¥t táº¥m pin"}
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

                    {/* TÃªn dá»± Ã¡n */}
                    <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                        <label style={{ fontSize: 12, fontWeight: 600, color: "#94a3b8" }}>TÃªn dá»± Ã¡n</label>
                        <input
                            type="text"
                            value={metadata.projectName || ""}
                            onChange={(e) => onChange("projectName", e.target.value)}
                            placeholder="VÃ­ dá»¥: Binh Nguyen Solar Farm Phase 1"
                            style={inputStyle}
                        />
                    </div>

                    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
                        {/* Äá»‹a Ä‘iá»ƒm */}
                        <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                            <label style={{ fontSize: 12, fontWeight: 600, color: "#94a3b8" }}>Äá»‹a Ä‘iá»ƒm</label>
                            <input
                                type="text"
                                value={metadata.location || ""}
                                onChange={(e) => onChange("location", e.target.value)}
                                placeholder="VÃ­ dá»¥: Ninh Thuan, Viet Nam"
                                style={inputStyle}
                            />
                        </div>

                        {/* Thá»i gian quÃ©t */}
                        <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                            <label style={{ fontSize: 12, fontWeight: 600, color: "#94a3b8" }}>Thá»i gian quÃ©t</label>
                            <input
                                type="text"
                                value={metadata.scanTime || ""}
                                onChange={(e) => onChange("scanTime", e.target.value)}
                                placeholder="VÃ­ dá»¥: 2026-06-02"
                                style={inputStyle}
                            />
                        </div>
                    </div>

                    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
                        {/* ÄÆ¡n vá»‹ quÃ©t */}
                        <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                            <label style={{ fontSize: 12, fontWeight: 600, color: "#94a3b8" }}>ÄÆ¡n vá»‹ quÃ©t</label>
                            <input
                                type="text"
                                value={metadata.operator || ""}
                                onChange={(e) => onChange("operator", e.target.value)}
                                placeholder="VÃ­ dá»¥: EPC Solar JSC"
                                style={inputStyle}
                            />
                        </div>

                        {/* Thiáº¿t bá»‹ quÃ©t */}
                        <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                            <label style={{ fontSize: 12, fontWeight: 600, color: "#94a3b8" }}>Thiáº¿t bá»‹ quÃ©t</label>
                            <input
                                type="text"
                                value={metadata.device || ""}
                                onChange={(e) => onChange("device", e.target.value)}
                                placeholder="VÃ­ dá»¥: DJI Matrice 300 RTK"
                                style={inputStyle}
                            />
                        </div>
                    </div>

                    <div style={{ display: "grid", gridTemplateColumns: "1.2fr 0.8fr", gap: 16 }}>
                        {/* Pháº¡m vi quÃ©t */}
                        <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                            <label style={{ fontSize: 12, fontWeight: 600, color: "#94a3b8" }}>Pháº¡m vi quÃ©t</label>
                            <input
                                type="text"
                                value={metadata.scope || ""}
                                onChange={(e) => onChange("scope", e.target.value)}
                                placeholder="VÃ­ dá»¥: Inverter Block 01 - 04"
                                style={inputStyle}
                            />
                        </div>

                        {/* CÃ´ng suáº¥t táº¥m pin */}
                        <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                            <label style={{ fontSize: 12, fontWeight: 600, color: "#0ea5e9" }}>CÃ´ng suáº¥t táº¥m pin (W)</label>
                            <input
                                type="text"
                                value={metadata.panelPower === undefined || metadata.panelPower === null ? "" : metadata.panelPower}
                                onChange={(e) => {
                                    const val = e.target.value;
                                    if (val === "" || /^[0-9]*\.?[0-9]*$/.test(val)) {
                                        onChange("panelPower", val);
                                    }
                                }}
                                placeholder="Máº·c Ä‘á»‹nh: 600"
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
                        {/* CÃ´ng suáº¥t há»‡ thá»‘ng */}
                        <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                            <label style={{ fontSize: 12, fontWeight: 600, color: "#94a3b8" }}>CÃ´ng suáº¥t há»‡ thá»‘ng</label>
                            <input
                                type="text"
                                value={metadata.systemCapacity || ""}
                                onChange={(e) => onChange("systemCapacity", e.target.value)}
                                placeholder="VÃ­ dá»¥: 1.2 MWp"
                                style={inputStyle}
                            />
                        </div>

                        {/* NgÆ°á»i phá»¥ trÃ¡ch */}
                        <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                            <label style={{ fontSize: 12, fontWeight: 600, color: "#94a3b8" }}>NgÆ°á»i phá»¥ trÃ¡ch</label>
                            <input
                                type="text"
                                value={metadata.supervisor || ""}
                                onChange={(e) => onChange("supervisor", e.target.value)}
                                placeholder="VÃ­ dá»¥: Nguyá»…n VÄƒn A"
                                style={inputStyle}
                            />
                        </div>
                    </div>

                    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
                        {/* Loáº¡i dá»¯ liá»‡u */}
                        <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                            <label style={{ fontSize: 12, fontWeight: 600, color: "#94a3b8" }}>Loáº¡i dá»¯ liá»‡u</label>
                            <input
                                type="text"
                                value={metadata.dataType || ""}
                                onChange={(e) => onChange("dataType", e.target.value)}
                                placeholder="Máº·c Ä‘á»‹nh: UAV thermal image"
                                style={inputStyle}
                            />
                        </div>

                        {/* Model AI sá»­ dá»¥ng */}
                        <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                            <label style={{ fontSize: 12, fontWeight: 600, color: "#94a3b8" }}>Model AI sá»­ dá»¥ng</label>
                            <input
                                type="text"
                                value={metadata.aiModel || ""}
                                onChange={(e) => onChange("aiModel", e.target.value)}
                                placeholder="Máº·c Ä‘á»‹nh: YOLOv8-Solar-M300"
                                style={inputStyle}
                            />
                        </div>
                    </div>

                    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
                        {/* PhiÃªn báº£n há»‡ thá»‘ng */}
                        <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                            <label style={{ fontSize: 12, fontWeight: 600, color: "#94a3b8" }}>PhiÃªn báº£n há»‡ thá»‘ng</label>
                            <input
                                type="text"
                                value={metadata.systemVersion || ""}
                                onChange={(e) => onChange("systemVersion", e.target.value)}
                                placeholder="Máº·c Ä‘á»‹nh: O&M Suite v2.4"
                                style={inputStyle}
                            />
                        </div>

                        {/* Spacer */}
                        <div style={{ display: "flex", flexDirection: "column", gap: 6 }} />
                    </div>

                    {/* Ghi chÃº */}
                    <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                        <label style={{ fontSize: 12, fontWeight: 600, color: "#94a3b8" }}>Ghi chÃº</label>
                        <textarea
                            value={metadata.notes || ""}
                            onChange={(e) => onChange("notes", e.target.value)}
                            placeholder="Nháº­p ghi chÃº thÃªm..."
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
                        {isNewProject ? "Bá» qua & ÄÃ³ng" : "ÄÃ³ng"}
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
                        {isNewProject ? "LÆ°u & Cháº¡y AI PhÃ¢n TÃ­ch" : "LÆ°u Thay Äá»•i"}
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

    // Náº¿u backend bÃ¡o completed thÃ¬ UI báº¯t buá»™c hiá»ƒn thá»‹ 100%
    if (status === "completed") {
        percent = 100;
    }

    // Náº¿u backend váº«n running thÃ¬ khÃ´ng Ä‘Æ°á»£c hiá»ƒn thá»‹ 100%
    // Ä‘á»ƒ trÃ¡nh cáº£m giÃ¡c â€œxong rá»“i nhÆ°ng cÃ²n cháº¡yâ€
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
        stage: raw?.current_stage || raw?.stage || raw?.message || "Äang xá»­ lÃ½...",
        message: raw?.message || "",
        elapsedSeconds: elapsedSeconds,
        etaSeconds: eta,
        error: raw?.error || null,
        filename: raw?.filename || "",
    };
}

// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
// InfoPanel â€” ThÃ´ng tin Ä‘á»£t kiá»ƒm tra cÃ³ pagination + mÅ©i tÃªn
// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
// PhÃ¢n bá»• cá»‘ Ä‘á»‹nh: trang 1 = 4 má»¥c, trang 2 = 4 má»¥c, trang 3 = 4 má»¥c
const PAGE_SIZES = [4, 4, 4];

function InfoPanel({ rows, onEdit, isAnyLoading }) {
    const [page, setPage] = useState(0);

    // Chia rows theo phÃ¢n bá»• cá»‘ Ä‘á»‹nh PAGE_SIZES
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
            {/* Header + mÅ©i tÃªn Ä‘iá»u hÆ°á»›ng */}
            <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                <Settings size={15} color="#6366f1" />
                <span style={{ fontSize: 12, fontWeight: 700, color: "#334155", textTransform: "uppercase", letterSpacing: "0.5px", flex: 1 }}>ThÃ´ng tin Ä‘á»£t kiá»ƒm tra</span>
                {/* MÅ©i tÃªn Ä‘iá»u hÆ°á»›ng trang */}
                <div style={{ display: "flex", alignItems: "center", gap: 3 }}>
                    {totalPages > 1 && (
                        <span style={{ fontSize: 10, color: "#94a3b8", fontWeight: 600, marginRight: 1 }}>
                            {page + 1}/{totalPages}
                        </span>
                    )}
                    <button
                        onClick={() => setPage(p => Math.max(0, p - 1))}
                        disabled={!hasPrev}
                        title="Trang trÆ°á»›c"
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
                        â†
                    </button>
                    <button
                        onClick={() => setPage(p => Math.min(totalPages - 1, p + 1))}
                        disabled={!hasNext}
                        title="Trang tiáº¿p theo"
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
                        â†’
                    </button>
                </div>
            </div>

            {/* LÆ°á»›i hiá»ƒn thá»‹ thÃ´ng tin trang hiá»‡n táº¡i */}
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
                            {/* Hiá»ƒn thá»‹ Ä‘áº§y Ä‘á»§ ná»™i dung â€” wrap náº¿u dÃ i, khÃ´ng cáº¯t bá» */}
                            <div style={{ fontSize: 12, color: "#334155", fontWeight: 600, lineHeight: 1.3, wordBreak: "break-word" }}>
                                {row.value}
                            </div>
                        </div>
                    </div>
                ))}
            </div>

            {/* Dot indicators (náº¿u > 1 trang) */}
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

            {/* NÃºt Chá»‰nh sá»­a */}
            <button
                onClick={onEdit}
                disabled={isAnyLoading}
                style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 6, padding: "8px 0", borderRadius: 6, border: "none", background: "linear-gradient(135deg,#0ea5e9,#6366f1)", color: "#fff", fontSize: 12, fontWeight: 700, cursor: isAnyLoading ? "not-allowed" : "pointer", boxShadow: "0 2px 6px rgba(14,165,233,0.2)", transition: "all 0.2s", opacity: isAnyLoading ? 0.7 : 1 }}
            >
                <Settings size={13} fill="white" /> Chá»‰nh sá»­a thÃ´ng tin
            </button>
        </div>
    );
}

// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
// Home Page
// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
export default function Home({ data, batchId, onAnalysisComplete, onReset, onViewOnMap }) {
    const [isUploading, setIsUploading] = useState(false);
    const [isAnalyzing, setIsAnalyzing] = useState(false);
    const [analysisProgress, setAnalysisProgress] = useState({
        status: "idle",
        percent: 0,
        stage: "ChÆ°a báº¯t Ä‘áº§u",
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
    const [activeImageIdx, setActiveImageIdx] = useState(0);
    const [showImageDetailModal, setShowImageDetailModal] = useState(false);

    React.useEffect(() => {
        setActiveImageIdx(0);
    }, [data]);
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
        return rawName ? rawName.replace(/_/g, " ") : "khÃ¡c";
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
            // ÄÃ¢y lÃ  Ä‘á»£t táº£i dá»± Ã¡n má»›i, lÆ°u táº¡m vÃ o state rá»“i Ä‘Ã³ng Ä‘á»ƒ cháº¡y AI
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
                alert("Lá»—i: " + res.data.error);
            } else {
                alert("âœ… Cáº­p nháº­t cáº¥u hÃ¬nh vÃ  tÃ­nh toÃ¡n láº¡i hao há»¥t thÃ nh cÃ´ng!");
                setShowMetadataModal(false);
                // Cáº­p nháº­t láº¡i state chÃ­nh cá»§a React
                const latestRes = await fetchLatestBatch();
                if (latestRes.data && onAnalysisComplete) {
                    onAnalysisComplete(latestRes.data.data, latestRes.data.batch_id, latestRes.data.panel_power);
                }
            }
        } catch (e) {
            alert("Lá»—i khi cáº­p nháº­t thÃ´ng tin: " + (e.response?.data?.detail || e.message));
        } finally {
            setIsSavingMetadata(false);
        }
    };

    // BÆ°á»›c 1: Upload + tiá»n xá»­ lÃ½, sau Ä‘Ã³ má»Ÿ modal quality review
    const handleUploadFiles = async (e) => {
        const files = Array.from(e.target.files || []);
        if (files.length === 0) return;

        setIsUploading(true);

        try {
            setStatusText("Äang táº£i dá»¯ liá»‡u lÃªn...");
            const formData = new FormData();
            files.forEach((file) => formData.append("files", file));

            await uploadDroneData(formData);

            setStatusText("Äang tiá»n xá»­ lÃ½ & kiá»ƒm tra cháº¥t lÆ°á»£ng...");
            const res = await processThermal();

            setQualityData(res.data);
            // Má»Ÿ popup nháº­p metadata dá»± Ã¡n trÆ°á»›c khi báº¥m cháº¡y AI!
            setShowMetadataModal(true);
        } catch (error) {
            alert("Lá»—i: " + (error.response?.data?.detail || error.message));
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
                        stage: "HoÃ n táº¥t phÃ¢n tÃ­ch. Äang cáº­p nháº­t dá»¯ liá»‡u...",
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

    // BÆ°á»›c 2: NgÆ°á»i dÃ¹ng xÃ¡c nháº­n trong modal, sau Ä‘Ã³ cháº¡y AI
    const handleRunAI = async () => {
        setIsAnalyzing(true);
        setAnalysisProgress({
            status: "running",
            percent: 0,
            stage: "Khá»Ÿi Ä‘á»™ng...",
            processedImages: 0,
            totalImages: 0,
            elapsedSeconds: 0,
            etaSeconds: null,
            message: "Äang chuáº©n bá»‹ phÃ¢n tÃ­ch...",
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

            // Gá»­i cÃ¡c trÆ°á»ng metadata cá»§a dá»± Ã¡n kÃ¨m theo
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
            } else if (res.data.message && res.data.message.includes("Báº¯t Ä‘áº§u phÃ¢n tÃ­ch AI")) {
                startPolling();
            } else {
                // KhÃ´ng cÃ³ áº£nh má»›i (Ä‘Ã£ xá»­ lÃ½ háº¿t hoáº·c thÃ´ng bÃ¡o khÃ¡c tá»« server)
                setIsAnalyzing(false);
                alert(res.data.message || "KhÃ´ng cÃ³ áº£nh má»›i nÃ o cáº§n phÃ¢n tÃ­ch!");
            }
        } catch (error) {
            console.error("Lá»—i AI:", error);
            setAnalysisProgress(prev => ({
                ...prev,
                status: "error",
                error: error.response?.data?.detail || error.message || "ÄÃ£ xáº£y ra lá»—i khi khá»Ÿi cháº¡y AI",
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
            alert("Vui lÃ²ng chá»n file Ä‘á»‹nh dáº¡ng .pt!");
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
                alert(res.data.message || "ÄÃ£ táº£i trá»ng sá»‘ AI má»›i thÃ nh cÃ´ng!");
            }
        } catch (error) {
            alert("Lá»—i khi táº£i trá»ng sá»‘: " + (error.response?.data?.detail || error.message));
        } finally {
            setIsUpdatingModel(false);
            if (modelInputRef.current) modelInputRef.current.value = "";
        }
    };

    const handleReanalyze = async () => {
        if (!window.confirm("Báº¡n cÃ³ muá»‘n cháº¡y láº¡i phÃ¢n tÃ­ch AI trÃªn cÃ¡c áº£nh Ä‘Ã£ táº£i lÃªn khÃ´ng? (Káº¿t quáº£ cÅ© sáº½ bá»‹ xÃ³a vÃ  cáº­p nháº­t theo thuáº­t toÃ¡n má»›i)")) return;

        setIsReanalyzing(true);

        try {
            await reanalyze();
            if (onReset) onReset();
            // Tá»± Ä‘á»™ng cháº¡y láº¡i AI
            await handleRunAI();
        } catch (error) {
            alert("Lá»—i khi phÃ¢n tÃ­ch láº¡i: " + error.message);
        } finally {
            setIsReanalyzing(false);
        }
    };

    const handleSystemReset = async () => {
        if (!window.confirm("HÃ nh Ä‘á»™ng nÃ y sáº½ xÃ³a sáº¡ch dá»¯ liá»‡u vÃ  Database. Báº¡n cÃ³ cháº¯c cháº¯n?")) return;

        setIsResetting(true);

        try {
            await resetSystem();
            if (onReset) onReset();
            alert("Há»‡ thá»‘ng Ä‘Ã£ Ä‘Æ°á»£c Ä‘Æ°a vá» tráº¡ng thÃ¡i máº·c Ä‘á»‹nh.");
        } catch (error) {
            alert("KhÃ´ng thá»ƒ reset: " + error.message);
        } finally {
            setIsResetting(false);
        }
    };

    const isAnyLoading = isUploading || isAnalyzing || isResetting || isUpdatingModel || isReanalyzing;

    // â”€â”€ MÃ u sáº¯c chuáº©n cho tá»«ng nhÃ³m lá»—i (phÃ¢n biá»‡t hotspot Ä‘Æ¡n/Ä‘a) â”€â”€
    const FAULT_COLORS = {
        "hotspot single cell": "#ef4444",   // Äá» tÆ°Æ¡i â€” Hotspot Ä‘Æ¡n
        "hotspot multi cell": "#ff6b35",   // Cam â€” Hotspot Ä‘a
        "crack": "#f59e0b",   // VÃ ng cam â€” Crack
        "shading": "#8b5cf6",   // TÃ­m â€” Shading
        "diode": "#06b6d4",   // Cyan â€” Diode
    };
    const getFaultColor = (group) => FAULT_COLORS[group.toLowerCase()] || "#94a3b8";

    // TÃªn hiá»ƒn thá»‹ Ä‘áº¹p hÆ¡n cho tá»«ng nhÃ³m
    const FAULT_LABELS = {
        "hotspot single cell": "hotspot single cell",
        "hotspot multi cell": "hotspot multi_cell",
        "crack": "Crack",
        "shading": "Shading",
        "diode": "Diode",
    };
    const getFaultLabel = (group) => FAULT_LABELS[group.toLowerCase()] || group;

    // â”€â”€ TÃ­nh cÃ´ng suáº¥t hao há»¥t theo tá»«ng nhÃ³m lá»—i â”€â”€
    // Backend lÆ°u power_loss_w á»Ÿ cáº¥p panel â€” phÃ¢n bá»• theo tá»· lá»‡ sá»‘ defect tá»«ng nhÃ³m trong panel
    const defectPowerLossW = {};
    (data || []).forEach((img) => {
        (img.panels || []).forEach((p) => {
            const reviewStatus = p.review_status || "unreviewed";
            const include = reviewStatus === "false_positive" ? false : (p.include_in_report !== undefined ? p.include_in_report : true);
            if (!include) return;

            const panelLossW = Number(p.power_loss_w ?? p.total_panel_loss ?? 0);
            const defects = p.defects || [];
            if (defects.length === 0) return;

            // Äáº¿m sá»‘ defect theo nhÃ³m trong panel nÃ y
            const panelGroupCounts = {};
            defects.forEach((d) => {
                const group = classifyDefect(d.class_name || d.type || "");
                panelGroupCounts[group] = (panelGroupCounts[group] || 0) + 1;
            });

            const totalDefectsInPanel = Object.values(panelGroupCounts).reduce((s, v) => s + v, 0);
            if (totalDefectsInPanel === 0) return;

            // PhÃ¢n bá»• power_loss_w theo tá»· lá»‡ sá»‘ defect
            Object.entries(panelGroupCounts).forEach(([group, cnt]) => {
                const share = (cnt / totalDefectsInPanel) * panelLossW;
                defectPowerLossW[group] = (defectPowerLossW[group] || 0) + share;
            });
        });
    });

    // XÃ¡c Ä‘á»‹nh Ä‘Æ¡n vá»‹ hiá»ƒn thá»‹ phÃ¹ há»£p
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

    // â”€â”€ Dá»¯ liá»‡u Donut chart (dÃ¹ng defectCounts Ä‘Ã£ phÃ¢n nhÃ³m á»Ÿ trÃªn) â”€â”€
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
            {/* AI Progress modal â€” hiá»ƒn thá»‹ khi Ä‘ang cháº¡y AI */}
            {isAnalyzing && (
                <AIProgressModal
                    progress={analysisProgress}
                    onClose={() => setIsAnalyzing(false)}
                />
            )}

            {/* Quality Review modal â€” sau preprocessing */}
            {qualityData && (
                <QualityReviewModal
                    qualityData={qualityData}
                    onConfirm={handleRunAI}
                    onCancel={handleCancelModal}
                    isRunningAI={isAnalyzing}
                />
            )}

            {/* Project Metadata modal â€” nháº­p/chá»‰nh sá»­a thÃ´ng tin dá»± Ã¡n */}
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

            {/* Full Image Detail Modal - hiá»ƒn thá»‹ kÃ­ch thÆ°á»›c gá»‘c tá»± nhiÃªn, trÃ¡nh bá»‹ zoom/crop */}
            {showImageDetailModal && data && data[activeImageIdx] && (
                <div
                    style={{
                        position: "fixed",
                        top: 0,
                        left: 0,
                        right: 0,
                        bottom: 0,
                        background: "rgba(15, 23, 42, 0.9)",
                        zIndex: 9999,
                        display: "flex",
                        flexDirection: "column",
                        alignItems: "center",
                        justifyContent: "center",
                        padding: 24,
                        backdropFilter: "blur(8px)",
                    }}
                    onClick={() => setShowImageDetailModal(false)}
                >
                    <div
                        style={{
                            position: "relative",
                            maxWidth: "90%",
                            maxHeight: "85%",
                            display: "flex",
                            flexDirection: "column",
                            alignItems: "center",
                            justifyContent: "center",
                            background: "#1e293b",
                            borderRadius: 12,
                            padding: 16,
                            border: "1px solid rgba(255,255,255,0.1)",
                            boxShadow: "0 25px 50px -12px rgba(0,0,0,0.5)",
                        }}
                        onClick={e => e.stopPropagation()}
                    >
                        {/* Header */}
                        <div style={{
                            display: "flex",
                            justifyContent: "space-between",
                            alignItems: "center",
                            width: "100%",
                            marginBottom: 12,
                            color: "#fff"
                        }}>
                            <span style={{ fontSize: 13, fontWeight: 600, color: "#cbd5e1" }}>
                                {data[activeImageIdx].filename}
                            </span>
                            <button
                                onClick={() => setShowImageDetailModal(false)}
                                style={{
                                    background: "transparent",
                                    border: "none",
                                    color: "#94a3b8",
                                    cursor: "pointer",
                                    padding: 4,
                                    display: "flex",
                                    alignItems: "center",
                                    justifyContent: "center",
                                    transition: "color 0.15s"
                                }}
                                onMouseEnter={e => e.currentTarget.style.color = "#ef4444"}
                                onMouseLeave={e => e.currentTarget.style.color = "#94a3b8"}
                            >
                                <X size={20} />
                            </button>
                        </div>

                        {/* Image area with nav arrows overlay */}
                        <div style={{
                            position: "relative",
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "center",
                            background: "#0f172a",
                            borderRadius: 8,
                            maxHeight: "75vh",
                        }}>
                            <img
                                src={`${API}/data/precalib/${data[activeImageIdx].filename}`}
                                alt={data[activeImageIdx].filename}
                                style={{
                                    maxWidth: "100%",
                                    maxHeight: "70vh",
                                    objectFit: "contain",
                                    borderRadius: 4,
                                    display: "block"
                                }}
                            />

                            {/* Nav arrows inside modal */}
                            {data.length > 1 && (
                                <>
                                    <button
                                        onClick={(e) => { e.stopPropagation(); setActiveImageIdx(i => Math.max(0, i - 1)); }}
                                        disabled={activeImageIdx === 0}
                                        style={{
                                            position: "absolute",
                                            left: 12,
                                            background: "rgba(15,23,42,0.6)",
                                            border: "none",
                                            borderRadius: "50%",
                                            width: 36,
                                            height: 36,
                                            display: "flex",
                                            alignItems: "center",
                                            justifyContent: "center",
                                            color: activeImageIdx === 0 ? "rgba(255,255,255,0.3)" : "#fff",
                                            cursor: activeImageIdx === 0 ? "not-allowed" : "pointer",
                                            fontSize: 20,
                                            transition: "background 0.2s",
                                        }}
                                        onMouseEnter={e => { if (activeImageIdx !== 0) e.currentTarget.style.background = "rgba(15,23,42,0.9)"; }}
                                        onMouseLeave={e => { if (activeImageIdx !== 0) e.currentTarget.style.background = "rgba(15,23,42,0.6)"; }}
                                    >
                                        â€¹
                                    </button>
                                    <button
                                        onClick={(e) => { e.stopPropagation(); setActiveImageIdx(i => Math.min(data.length - 1, i + 1)); }}
                                        disabled={activeImageIdx === data.length - 1}
                                        style={{
                                            position: "absolute",
                                            right: 12,
                                            background: "rgba(15,23,42,0.6)",
                                            border: "none",
                                            borderRadius: "50%",
                                            width: 36,
                                            height: 36,
                                            display: "flex",
                                            alignItems: "center",
                                            justifyContent: "center",
                                            color: activeImageIdx === data.length - 1 ? "rgba(255,255,255,0.3)" : "#fff",
                                            cursor: activeImageIdx === data.length - 1 ? "not-allowed" : "pointer",
                                            fontSize: 20,
                                            transition: "background 0.2s",
                                        }}
                                        onMouseEnter={e => { if (activeImageIdx !== data.length - 1) e.currentTarget.style.background = "rgba(15,23,42,0.9)"; }}
                                        onMouseLeave={e => { if (activeImageIdx !== data.length - 1) e.currentTarget.style.background = "rgba(15,23,42,0.6)"; }}
                                    >
                                        â€º
                                    </button>
                                </>
                            )}
                        </div>

                        {/* Footer */}
                        <div style={{ marginTop: 12, display: "flex", justifyContent: "space-between", width: "100%", fontSize: 11, color: "#94a3b8" }}>
                            <span>Chá»‰ sá»‘: {activeImageIdx + 1} / {data.length}</span>
                            <span>KÃ­ch thÆ°á»›c gá»‘c (Tá»· lá»‡ thá»±c táº¿)</span>
                        </div>
                    </div>
                </div>
            )}

            {/* â”€â”€ HIDDEN FILE INPUTS â”€â”€ */}
            <input type="file" accept=".zip,.rar,.jpg,.jpeg,.png,image/*" multiple ref={fileInputRef} onChange={handleUploadFiles} style={{ display: "none" }} />
            <input type="file" webkitdirectory="true" directory="" multiple ref={folderInputRef} onChange={handleUploadFiles} style={{ display: "none" }} />
            <input type="file" accept=".pt" ref={modelInputRef} onChange={handleUpdateModel} style={{ display: "none" }} />

            {/* â”€â”€ HEADER â”€â”€ */}
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16, flexWrap: "wrap", gap: 8 }}>
                <div>
                    <h1 style={{ margin: 0, fontSize: 20, fontWeight: 700, color: "#0f172a" }}>Báº£ng Ä‘iá»u khiá»ƒn</h1>
                    {projectMetadata.projectName && (
                        <p style={{ margin: "2px 0 0", fontSize: 12, color: "#64748b", fontWeight: 500 }}>
                            Dá»± Ã¡n: {projectMetadata.projectName}
                        </p>
                    )}
                </div>
                <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                    {/* Thay model AI */}
                    <button onClick={() => modelInputRef.current?.click()} disabled={isAnyLoading}
                        style={{ display: "flex", alignItems: "center", gap: 4, padding: "7px 12px", borderRadius: 6, border: "1px solid rgba(14,165,233,0.4)", background: "rgba(14,165,233,0.06)", color: "#0ea5e9", fontSize: 12, fontWeight: 600, cursor: "pointer", transition: "all 0.2s" }}>
                        {isUpdatingModel ? <Loader2 size={14} className="animate-spin" /> : <Cpu size={14} />}
                        {isUpdatingModel ? "Äang cáº­p nháº­t..." : "Thay model AI"}
                    </button>

                    {/* PhÃ¢n tÃ­ch láº¡i */}
                    <button onClick={handleReanalyze} disabled={isAnyLoading || totalPanels === 0}
                        style={{ display: "flex", alignItems: "center", gap: 4, padding: "7px 12px", borderRadius: 6, border: "none", background: "linear-gradient(135deg,#f59e0b,#ef4444)", color: "#fff", fontSize: 12, fontWeight: 600, cursor: (isAnyLoading || totalPanels === 0) ? "not-allowed" : "pointer", opacity: totalPanels === 0 ? 0.5 : 1, transition: "all 0.2s" }}>
                        {isReanalyzing ? <Loader2 size={14} className="animate-spin" /> : <RefreshCw size={14} />}
                        {isReanalyzing ? "Äang cháº¡y láº¡i..." : "PhÃ¢n tÃ­ch láº¡i"}
                    </button>

                    {/* Reset */}
                    <button onClick={handleSystemReset} disabled={isAnyLoading}
                        style={{ display: "flex", alignItems: "center", gap: 4, padding: "7px 12px", borderRadius: 6, border: "1px solid rgba(239,68,68,0.3)", background: "#fff1f1", color: "#ef4444", fontSize: 12, fontWeight: 600, cursor: isAnyLoading ? "not-allowed" : "pointer", transition: "all 0.2s" }}>
                        {isResetting ? <Loader2 size={14} className="animate-spin" /> : <Trash2 size={14} />}
                        {isResetting ? "Äang reset..." : "Reset Há»‡ thá»‘ng"}
                    </button>
                </div>
            </div>

            {/* â”€â”€ KPI CARDS â”€â”€ */}
            <div className="kpi-grid-container">
                {[
                    { icon: <Image size={18} />, label: "Tá»”NG áº¢NH UAV", value: (data?.length || 0).toLocaleString(), unit: "áº£nh", accent: "#8b5cf6" },
                    { icon: <LayoutGrid size={18} />, label: "Tá»”NG PANEL", value: totalPanels.toLocaleString(), unit: "panel", accent: "#0ea5e9" },
                    { icon: <AlertCircle size={18} />, label: "PANEL Lá»–I", value: totalFaults.toLocaleString(), unit: "panel lá»—i", accent: "#ef4444" },
                    { icon: <Zap size={18} />, label: "CÃ”NG SUáº¤T HAO Há»¤T Æ¯á»šC TÃNH", value: estimatedLossMWp >= 1 ? estimatedLossMWp.toFixed(2) : (estimatedLoss / 1000).toFixed(2), unit: estimatedLossMWp >= 1 ? "MWp" : "kWp", accent: "#f59e0b" },
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

            {/* â”€â”€ MIDDLE 3-COLUMN SECTION â”€â”€ */}
            <div className="middle-grid-container">
                {/* Col 1: Táº¢I Dá»® LIá»†U UAV */}
                <div style={{ background: "#fff", borderRadius: 10, border: "1px solid #e2e8f0", padding: 16, boxShadow: "0 1px 4px rgba(0,0,0,0.03)", display: "flex", flexDirection: "column", gap: 10, height: "100%", boxSizing: "border-box" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                        <Upload size={15} color="#0ea5e9" />
                        <span style={{ fontSize: 12, fontWeight: 700, color: "#334155", textTransform: "uppercase", letterSpacing: "0.5px" }}>Táº£i dá»¯ liá»‡u UAV</span>
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
                            KÃ©o tháº£ thÆ° má»¥c hoáº·c file Zip/Rar vÃ o Ä‘Ã¢y<br /><span style={{ fontSize: 10, color: "#94a3b8" }}>hoáº·c chá»n hÃ¬nh thá»©c táº£i lÃªn</span>
                        </p>
                        <div style={{ display: "flex", gap: 6, width: "100%", justifyContent: "center", marginTop: 2 }}>
                            <button
                                onClick={e => { e.stopPropagation(); if (!isAnyLoading) fileInputRef.current?.click(); }}
                                disabled={isAnyLoading}
                                style={{ padding: "6px 12px", borderRadius: 5, border: "1px solid #cbd5e1", background: "#fff", color: "#334155", fontSize: 11, fontWeight: 600, cursor: isAnyLoading ? "not-allowed" : "pointer", display: "flex", alignItems: "center", gap: 3 }}
                            >
                                ðŸ“„ Táº£i file
                            </button>
                            <button
                                onClick={e => { e.stopPropagation(); if (!isAnyLoading) folderInputRef.current?.click(); }}
                                disabled={isAnyLoading}
                                style={{ padding: "6px 12px", borderRadius: 5, border: "none", background: "#0ea5e9", color: "#fff", fontSize: 11, fontWeight: 600, cursor: isAnyLoading ? "not-allowed" : "pointer", display: "flex", alignItems: "center", gap: 3 }}
                            >
                                ðŸ“ ThÆ° má»¥c
                            </button>
                        </div>
                    </div>

                    {/* Upload progress */}
                    {isUploading && (
                        <div>
                            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                                <span style={{ fontSize: 12, color: "#475569", fontWeight: 500 }}>Äang táº£i lÃªn...</span>
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
                                {data?.length > 0 ? `âœ“ ÄÃ£ táº£i ${data.length} áº£nh Â· Tá»•ng: ${data.length} áº£nh` : "Há»— trá»£ Ä‘á»‹nh dáº¡ng: JPG, PNG, TIF, RJPG"}
                            </span>
                        </div>
                        {data?.length > 0 && <div style={{ color: "#0ea5e9", fontWeight: 500 }}>ÄÃ£ táº£i lÃªn &amp; phÃ¢n tÃ­ch AI thÃ nh cÃ´ng Â· 100%</div>}
                    </div>
                </div>

                {/* Col 2: THÃ”NG TIN Äá»¢T KIá»‚M TRA â€” vá»›i pagination */}
                {(() => {
                    const infoRows = [
                        { icon: "ðŸ­", label: "TÃªn dá»± Ã¡n", value: projectMetadata.projectName || "â€”", fullWidth: true },
                        { icon: "ðŸ“…", label: "NgÃ y kiá»ƒm tra", value: projectMetadata.scanTime || "â€”" },
                        { icon: "ðŸ“", label: "Äá»‹a Ä‘iá»ƒm", value: projectMetadata.location || "â€”" },
                        { icon: "ðŸš", label: "UAV", value: projectMetadata.device || "â€”" },
                        { icon: "ðŸ¢", label: "ÄÆ¡n vá»‹ quÃ©t", value: projectMetadata.operator || "â€”" },
                        { icon: "ðŸ‘¤", label: "NgÆ°á»i váº­n hÃ nh", value: projectMetadata.supervisor || "â€”" },
                        { icon: "âš¡", label: "CÃ´ng suáº¥t pin", value: projectMetadata.panelPower ? `${projectMetadata.panelPower} W` : "â€”" },
                        { icon: "ðŸ”Œ", label: "CÃ´ng suáº¥t há»‡ thá»‘ng", value: projectMetadata.systemCapacity || "â€”" },
                        { icon: "ðŸ“Š", label: "Loáº¡i dá»¯ liá»‡u", value: projectMetadata.dataType || "â€”" },
                        { icon: "ðŸ¤–", label: "Model AI", value: projectMetadata.aiModel || "â€”" },
                        { icon: "ðŸ’»", label: "Há»‡ thá»‘ng", value: projectMetadata.systemVersion || "â€”" },
                        { icon: "ðŸ“", label: "Ghi chÃº", value: projectMetadata.notes || "Kiá»ƒm tra Ä‘á»‹nh ká»³ thÃ¡ng 6", fullWidth: true },
                    ];
                    return <InfoPanel rows={infoRows} onEdit={() => setShowMetadataModal(true)} isAnyLoading={isAnyLoading} />;
                })()}

                {/* Col 3: áº¢NH ÄÃƒ QUA TIá»€N Xá»¬ LÃ */}
                <div className="middle-card-uav" style={{ background: "#fff", borderRadius: 10, border: "1px solid #e2e8f0", padding: 16, boxShadow: "0 1px 4px rgba(0,0,0,0.03)", display: "flex", flexDirection: "column", gap: 8, boxSizing: "border-box" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                        <Image size={15} color="#f59e0b" />
                        <span style={{ fontSize: 12, fontWeight: 700, color: "#334155", textTransform: "uppercase", letterSpacing: "0.5px" }}>áº¢nh Ä‘Ã£ qua tiá»n xá»­ lÃ½</span>
                        {data?.length > 0 && (
                            <button
                                onClick={() => setShowImageDetailModal(true)}
                                style={{
                                    marginLeft: 12,
                                    padding: "2px 8px",
                                    borderRadius: 4,
                                    border: "1px solid #cbd5e1",
                                    background: "#f8fafc",
                                    color: "#334155",
                                    fontSize: 10,
                                    fontWeight: 600,
                                    cursor: "pointer",
                                    transition: "all 0.15s",
                                }}
                                onMouseEnter={e => { e.currentTarget.style.borderColor = "#0ea5e9"; e.currentTarget.style.background = "#f0f9ff"; e.currentTarget.style.color = "#0ea5e9"; }}
                                onMouseLeave={e => { e.currentTarget.style.borderColor = "#cbd5e1"; e.currentTarget.style.background = "#f8fafc"; e.currentTarget.style.color = "#334155"; }}
                            >
                                xem chi tiáº¿t
                            </button>
                        )}
                        {data?.length > 0 && <span style={{ marginLeft: "auto", fontSize: 11, color: "#64748b", fontWeight: 500 }}>{data.length} áº£nh</span>}
                    </div>
                    <div style={{ flex: 1, minHeight: 0 }}>
                        <ImageGallery images={data || []} apiBase={API} activeIdx={activeImageIdx} setActiveIdx={setActiveImageIdx} />
                    </div>
                </div>
            </div>

            {/* â”€â”€ BOTTOM 2-COLUMN SECTION â”€â”€ */}
            <div className="bottom-grid-container">
                {/* Col 1: THá»NG KÃŠ Lá»–I THEO LOáº I */}
                <div style={{ background: "#fff", borderRadius: 10, border: "1px solid #e2e8f0", padding: 16, boxShadow: "0 1px 4px rgba(0,0,0,0.03)" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 10 }}>
                        <AlertCircle size={15} color="#ef4444" />
                        <span style={{ fontSize: 12, fontWeight: 700, color: "#334155", textTransform: "uppercase", letterSpacing: "0.5px" }}>Thá»‘ng kÃª lá»—i theo loáº¡i</span>
                    </div>

                    <div style={{ display: "flex", gap: 16, alignItems: "center" }}>
                        {/* Donut */}
                        <div style={{ flexShrink: 0 }}>
                            <DonutChart data={donutData} totalLabel={totalFaultCount.toLocaleString()} />
                        </div>

                        {/* Legend table */}
                        <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: 4 }}>
                            <div style={{ display: "grid", gridTemplateColumns: "auto 1fr auto auto", gap: "2px 8px", fontSize: 11, color: "#94a3b8", fontWeight: 600, paddingBottom: 4, borderBottom: "1px solid #f1f5f9" }}>
                                <span>Loáº¡i lá»—i</span><span></span><span>Sá»‘ lÆ°á»£ng</span><span>Tá»· lá»‡</span>
                            </div>
                            {donutData.length === 0 ? (
                                <div style={{ color: "#94a3b8", fontSize: 12, textAlign: "center", paddingTop: 8 }}>ChÆ°a cÃ³ dá»¯ liá»‡u</div>
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
                        Xem chi tiáº¿t <ChevronRight size={12} />
                    </button>
                </div>

                {/* Col 2: CÃ”NG SUáº¤T HAO Há»¤T THEO LOáº I Lá»–I */}
                <div style={{ background: "#fff", borderRadius: 10, border: "1px solid #e2e8f0", padding: 16, boxShadow: "0 1px 4px rgba(0,0,0,0.03)", display: "flex", flexDirection: "column" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 10 }}>
                        <Zap size={15} color="#f59e0b" />
                        <span style={{ fontSize: 12, fontWeight: 700, color: "#334155", textTransform: "uppercase", letterSpacing: "0.5px" }}>CÃ´ng suáº¥t hao há»¥t theo loáº¡i lá»—i</span>
                        {lossUnit && powerLossChartData.length > 0 && (
                            <span style={{ marginLeft: "auto", fontSize: 11, color: "#94a3b8", fontWeight: 500 }}>ÄÆ¡n vá»‹: {lossUnit}</span>
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
                                {" "}lÃ  lá»—i gÃ¢y tháº¥t thoÃ¡t lá»›n nháº¥t â€”{" "}
                                <b>{powerLossChartData[0]?.value?.toFixed(lossUnit === "W" ? 0 : 2)} {lossUnit}</b>
                            </span>
                        </div>
                    ) : (
                        <div style={{ marginTop: 8, padding: "8px 12px", borderRadius: 6, background: "#f8fafc", border: "1px solid #e2e8f0", fontSize: 11, color: "#94a3b8", textAlign: "center" }}>
                            ChÆ°a cÃ³ dá»¯ liá»‡u cÃ´ng suáº¥t hao há»¥t
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
                    grid-template-columns: 1fr 1.1fr 2.3fr;
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
                    height: 420px;
                }
                @media (max-height: 900px), (max-width: 1400px) {
                    .middle-card-uav {
                        height: 380px;
                    }
                }
                @media (max-height: 768px), (max-width: 1200px) {
                    .middle-card-uav {
                        height: 340px;
                    }
                }
                @media (max-width: 1024px) {
                    .middle-card-uav {
                        height: 380px;
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
