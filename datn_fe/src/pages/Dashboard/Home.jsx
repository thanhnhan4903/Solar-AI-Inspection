import React, { useState, useRef } from "react";
import { LayoutGrid, AlertCircle, DollarSign, Upload, Trash2, Loader2, Cpu, TrendingUp, Sun, Leaf, Globe, Thermometer, Zap } from "lucide-react";
import { colors } from "../../constants/theme";
import { PageHeader } from "../../components/layout/PageHeader";
import { KpiCard } from "../../components/ui/KpiCard";
import { ActionButton } from "../../components/ui/ActionButton";
import axios from "axios";
import solarFarmAerial from "../../assets/solar_farm_aerial.png";

// Mini bar chart component
function AnomalyBarChart({ data }) {
    if (!data || data.length === 0) {
        return (
            <div style={{
                display: "flex", alignItems: "center", justifyContent: "center",
                height: "100%", flexDirection: "column", gap: 12, color: "#94a3b8"
            }}>
                <TrendingUp size={40} style={{ opacity: 0.3 }} />
                <p style={{ margin: 0, fontSize: 13 }}>No analysis data available</p>
            </div>
        );
    }
    const max = Math.max(...data.map(d => d.value), 1);
    const barColors = ["#f97316", "#ef4444", "#06b6d4", "#eab308", "#a855f7", "#10b981", "#0ea5e9", "#f59e0b"];
    return (
        <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>
            <div style={{ flex: 1, display: "flex", alignItems: "flex-end", gap: 10, padding: "8px 0" }}>
                {data.map((item, i) => {
                    const pct = (item.value / max) * 100;
                    return (
                        <div key={i} style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center", gap: 4 }}>
                            <span style={{ fontSize: 11, fontWeight: 700, color: "#1e293b" }}>{item.value}</span>
                            <div style={{
                                width: "100%", height: `${Math.max(pct * 1.4, 4)}px`,
                                background: `linear-gradient(180deg, ${barColors[i % barColors.length]}, ${barColors[i % barColors.length]}bb)`,
                                borderRadius: "4px 4px 0 0",
                                transition: "height 0.5s ease",
                                minHeight: 4,
                            }} />
                        </div>
                    );
                })}
            </div>
            {/* X labels */}
            <div style={{ display: "flex", gap: 10, borderTop: "2px solid #e2e8f0", paddingTop: 6 }}>
                {data.map((item, i) => (
                    <div key={i} style={{ flex: 1, textAlign: "center", fontSize: 9, color: "#64748b", wordBreak: "break-word", lineHeight: 1.2 }}>
                        {item.label}
                    </div>
                ))}
            </div>
        </div>
    );
}

export default function Home({ data, onAnalysisComplete, onReset }) {
    const [isProcessing, setIsProcessing] = useState(false);
    const [isResetting, setIsResetting] = useState(false);
    const [statusText, setStatusText] = useState("");
    const fileInputRef = useRef(null);
    const folderInputRef = useRef(null);
    const modelInputRef = useRef(null);
    const [isUpdatingModel, setIsUpdatingModel] = useState(false);

    // Tính toán số liệu thực
    const allPanels = data?.flatMap(img => img.panels) || [];
    const totalPanels = allPanels.length;
    const faultyPanels = allPanels.filter(p => p.total_panel_loss > 0 || p.status === "faulty");
    const totalFaults = faultyPanels.length;
    const estimatedLoss = faultyPanels.reduce((sum, p) => sum + (p.total_panel_loss * 0.5), 0);
    const healthyRate = totalPanels > 0 ? ((totalPanels - totalFaults) / totalPanels * 100) : 0;

    // Thống kê loại lỗi cho biểu đồ
    const defectCounts = {};
    allPanels.forEach(p => {
        (p.defects || []).forEach(d => {
            const name = (d.class_name || d.type || "unknown").replace(/_/g, " ");
            defectCounts[name] = (defectCounts[name] || 0) + 1;
        });
    });
    const chartData = Object.entries(defectCounts)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 8)
        .map(([label, value]) => ({ label, value }));

    const handleUploadAndAnalyze = async (e) => {
        const files = Array.from(e.target.files);
        if (files.length === 0) return;
        setIsProcessing(true);
        try {
            setStatusText("Uploading drone images...");
            const formData = new FormData();
            files.forEach(file => formData.append("files", file));
            await axios.post("http://127.0.0.1:8000/api/v1/upload-drone-data", formData);

            setStatusText("Precalibrating thermal/RGB pairs...");
            await axios.get("http://127.0.0.1:8000/api/v1/process-thermal");

            setStatusText("Running AI model diagnostics...");
            const analyzeForm = new FormData();
            const userStr = localStorage.getItem("user");
            if (userStr) {
                const user = JSON.parse(userStr);
                analyzeForm.append("user_id", user.id);
            }
            const res = await axios.post("http://127.0.0.1:8000/api/v1/analyze-all", analyzeForm);
            if (res.data.error) throw new Error(res.data.error);
            if (onAnalysisComplete) onAnalysisComplete(res.data.data, res.data.batch_id);
            setStatusText("");
            alert(`Success! AI model completed scanning ${res.data.data.length} images.`);
        } catch (error) {
            alert("Processing Error: " + (error.response?.data?.detail || error.message));
            setStatusText("");
        }
        setIsProcessing(false);
        if (fileInputRef.current) fileInputRef.current.value = "";
        if (folderInputRef.current) folderInputRef.current.value = "";
    };

    const handleUpdateModel = async (e) => {
        const file = e.target.files[0];
        if (!file) return;
        if (!file.name.endsWith('.pt')) {
            alert("Please select a valid PyTorch model weights file (.pt)!");
            if (modelInputRef.current) modelInputRef.current.value = "";
            return;
        }
        setIsUpdatingModel(true);
        try {
            const formData = new FormData();
            formData.append("file", file);
            const res = await axios.post("http://127.0.0.1:8000/api/v1/update-ai-model", formData);
            if (res.data.error) alert(res.data.error);
            else alert(res.data.message || "New AI model weights loaded successfully!");
        } catch (error) {
            alert("Error loading weights: " + (error.response?.data?.detail || error.message));
        }
        setIsUpdatingModel(false);
        if (modelInputRef.current) modelInputRef.current.value = "";
    };

    const handleSystemReset = async () => {
        if (!window.confirm("This action will completely wipe all data and the Database. Are you sure?")) return;
        setIsResetting(true);
        try {
            await axios.post("http://127.0.0.1:8000/api/v1/reset-system");
            if (onReset) onReset();
            alert("The system has been successfully reset to default.");
        } catch (error) {
            alert("Could not reset: " + error.message);
        }
        setIsResetting(false);
    };

    return (
        <div>
            {/* Header row */}
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-end", marginBottom: 24 }}>
                <PageHeader title="Dashboard" subtitle="Solar farm monitoring overview" />
                <div style={{ display: "flex", gap: 10, flexWrap: "wrap", justifyContent: "flex-end" }}>
                    <input type="file" accept=".zip,.rar,.jpg,.jpeg,.png,image/*" multiple ref={fileInputRef} onChange={handleUploadAndAnalyze} style={{ display: "none" }} />
                    <ActionButton
                        onClick={() => fileInputRef.current?.click()}
                        disabled={isProcessing || isResetting}
                        icon={isProcessing ? <Loader2 className="animate-spin" size={16} /> : <Upload size={16} />}
                        style={{ background: "linear-gradient(135deg, #0EA5E9, #8B5CF6)", color: "white", border: "none", boxShadow: "0 4px 12px rgba(14,165,233,0.3)" }}
                    >
                        {isProcessing ? statusText : "Upload & Analyze"}
                    </ActionButton>

                    <input type="file" webkitdirectory="" multiple ref={folderInputRef} onChange={handleUploadAndAnalyze} style={{ display: "none" }} />

                    <input type="file" accept=".pt" ref={modelInputRef} onChange={handleUpdateModel} style={{ display: "none" }} />
                    <ActionButton
                        onClick={() => modelInputRef.current?.click()}
                        disabled={isProcessing || isResetting || isUpdatingModel}
                        style={{ background: "linear-gradient(135deg, #0ea5e920, #8b5cf610)", color: colors.primary, border: `1px solid ${colors.primary}50`, backdropFilter: "blur(4px)" }}
                        icon={isUpdatingModel ? <Loader2 className="animate-spin" size={16} /> : <Cpu size={16} />}
                    >
                        {isUpdatingModel ? "Updating AI..." : "Replace AI Model"}
                    </ActionButton>

                    <ActionButton
                        onClick={handleSystemReset}
                        disabled={isProcessing || isResetting}
                        style={{ background: "#fee2e2", color: colors.danger, border: `1px solid ${colors.danger}30` }}
                        icon={isResetting ? <Loader2 className="animate-spin" size={16} /> : <Trash2 size={16} />}
                    >
                        Reset System
                    </ActionButton>
                </div>
            </div>

            {/* KPI Cards */}
            <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 16, marginBottom: 24 }}>
                <KpiCard
                    icon={<LayoutGrid size={20} />}
                    label="Total Panels"
                    value={totalPanels.toLocaleString()}
                    accent={colors.primary}
                />
                <KpiCard
                    icon={<AlertCircle size={20} />}
                    label="Faults Detected"
                    value={totalFaults}
                    accent={colors.danger}
                />
                <KpiCard
                    icon={<DollarSign size={20} />}
                    label="Estimated Loss ($)"
                    value={`$${estimatedLoss.toFixed(2)}`}
                    accent={colors.warning}
                />
            </div>



            {/* AI Analysis Breakdown */}
            <div style={{
                background: "rgba(255,255,255,0.85)",
                backdropFilter: "blur(16px)",
                borderRadius: 20,
                border: "1px solid rgba(226,232,240,0.8)",
                padding: "24px 32px",
                boxShadow: "0 10px 30px rgba(0,0,0,0.04)",
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                gap: 40,
                flexWrap: "wrap",
            }}>
                <style>{`
                    @keyframes neonPulse {
                        0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
                        70% { box-shadow: 0 0 0 6px rgba(16, 185, 129, 0); }
                        100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
                    }
                `}</style>

                {/* Left Side: Circular Ring Gauge */}
                <div style={{ display: "flex", alignItems: "center", gap: 24, minWidth: 260 }}>
                    <div style={{ position: "relative", width: 90, height: 90, display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>
                        <svg width="90" height="90" style={{ transform: "rotate(-90deg)" }}>
                            <circle
                                cx="45"
                                cy="45"
                                r="36"
                                fill="transparent"
                                stroke="#f1f5f9"
                                strokeWidth="6"
                            />
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
                        {/* Center text */}
                        <div style={{ position: "absolute", textAlign: "center" }}>
                            <div style={{ fontSize: 16, fontWeight: 800, color: "#1e293b" }}>
                                {totalPanels > 0 ? `${healthyRate.toFixed(1)}%` : "—"}
                            </div>
                        </div>
                    </div>

                    {/* Title & Status */}
                    <div>
                        <div style={{ fontSize: 11, fontWeight: 600, color: "#64748b", textTransform: "uppercase", letterSpacing: "1px", marginBottom: 4 }}>
                            AI Analysis Breakdown
                        </div>
                        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                            <h3 style={{ margin: 0, fontSize: 20, fontWeight: 850, color: "#1e293b", letterSpacing: "-0.5px" }}>
                                System Stability
                            </h3>
                            <div style={{
                                width: 8,
                                height: 8,
                                borderRadius: "50%",
                                background: totalPanels > 0 && healthyRate > 90 ? "#10b981" : "#94a3b8",
                                animation: totalPanels > 0 && healthyRate > 90 ? "neonPulse 2s infinite" : "none",
                                flexShrink: 0
                            }} />
                        </div>
                    </div>
                </div>

                {/* Right Side: Operations Detail Cards */}
                <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: 10, minWidth: 260 }}>
                    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
                        <div style={{ padding: "10px 14px", background: "#f8fafc", borderRadius: 12, border: "1px solid #f1f5f9" }}>
                            <div style={{ fontSize: 9, fontWeight: 700, color: "#94a3b8", textTransform: "uppercase", letterSpacing: "0.5px", marginBottom: 4 }}>
                                Batch Status
                            </div>
                            <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                                <div style={{
                                    padding: "2px 8px",
                                    background: totalPanels > 0 ? "#e6f4ea" : "#f1f5f9",
                                    color: totalPanels > 0 ? "#137333" : "#5f6368",
                                    borderRadius: 6,
                                    fontSize: 10,
                                    fontWeight: 700,
                                    textTransform: "uppercase"
                                }}>
                                    {totalPanels > 0 ? "Active" : "None"}
                                </div>
                            </div>
                        </div>

                        <div style={{ padding: "10px 14px", background: "#f8fafc", borderRadius: 12, border: "1px solid #f1f5f9" }}>
                            <div style={{ fontSize: 9, fontWeight: 700, color: "#94a3b8", textTransform: "uppercase", letterSpacing: "0.5px", marginBottom: 4 }}>
                                Health Profile
                            </div>
                            <div style={{ fontSize: 12, fontWeight: 800, color: totalPanels > 0 ? (healthyRate > 95 ? "#10b981" : "#d97706") : "#94a3b8" }}>
                                {totalPanels > 0 ? (healthyRate > 95 ? "EXCELLENT" : "STABLE") : "NO SCAN"}
                            </div>
                        </div>
                    </div>

                    <p style={{ margin: 0, fontSize: 11, color: "#94a3b8", textAlign: "right" }}>
                        * Data updated from the latest AI scan (Batch ID: {totalPanels > 0 ? "Active" : "None"}).
                    </p>
                </div>
            </div>

            {/* Bottom Section: Ambient Solar Farm Background Banner */}
            <div style={{ 
                marginTop: 24, 
                marginBottom: 10,
                borderRadius: 16,
                overflow: "hidden",
                height: 260,
                border: "1px solid rgba(226,232,240,0.8)",
                boxShadow: "0 8px 30px rgba(0,0,0,0.03)"
            }}>
                <img 
                    src={solarFarmAerial} 
                    alt="Solar Farm Ambient Background" 
                    style={{ 
                        width: "100%", 
                        height: "100%", 
                        objectFit: "cover"
                    }} 
                />
            </div>
        </div>
    );
}