import React, { useState, useRef } from "react";
import { LayoutGrid, AlertCircle, DollarSign, Upload, Trash2, Loader2, Play, Cpu } from "lucide-react";
import { colors } from "../../constants/theme";
import { PageHeader } from "../../components/layout/PageHeader";
import { KpiCard } from "../../components/ui/KpiCard";
import { SolarCard, CardHeader } from "../../components/ui/SolarCard";
import { ActionButton } from "../../components/ui/ActionButton";
import axios from "axios";

export default function Home({ data, onAnalysisComplete, onReset }) {
    const [isProcessing, setIsProcessing] = useState(false);
    const [isResetting, setIsResetting] = useState(false);
    const [statusText, setStatusText] = useState("");
    const fileInputRef = useRef(null);
    const modelInputRef = useRef(null);
    const [isUpdatingModel, setIsUpdatingModel] = useState(false);

    // TÍNH TOÁN SỐ LIỆU THẬT
    const allPanels = data?.flatMap(img => img.panels) || [];
    const totalPanels = allPanels.length;
    const faultyPanels = allPanels.filter(p => p.total_panel_loss > 0);
    const totalFaults = faultyPanels.length;
    
    // Giả định tổn thất tài chính dựa trên % hỏng
    const estimatedLoss = faultyPanels.reduce((sum, p) => sum + (p.total_panel_loss * 0.5), 0);

    const handleUploadAndAnalyze = async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        setIsProcessing(true);
        try {
            setStatusText("Đang tải dữ liệu lên...");
            const formData = new FormData();
            formData.append("file", file);
            await axios.post("http://127.0.0.1:8000/api/v1/upload-drone-data", formData);

            setStatusText("Đang tiền hiệu chỉnh ảnh...");
            await axios.get("http://127.0.0.1:8000/api/v1/process-thermal");

            setStatusText("Đang phân tích AI...");
            const analyzeForm = new FormData();
            const userStr = localStorage.getItem("user");
            if (userStr) {
                const user = JSON.parse(userStr);
                analyzeForm.append("user_id", user.id);
            }
            const res = await axios.post("http://127.0.0.1:8000/api/v1/analyze-all", analyzeForm);
            
            if (onAnalysisComplete) {
                onAnalysisComplete(res.data.data, res.data.batch_id);
            }
            setStatusText("");
            alert(`Thành công! Đã phân tích xong ${res.data.data.length} ảnh.`);
        } catch (error) {
            alert("Lỗi xử lý: " + (error.response?.data?.detail || error.message));
            setStatusText("");
        }
        setIsProcessing(false);
        if (fileInputRef.current) fileInputRef.current.value = "";
    };

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
            const res = await axios.post("http://127.0.0.1:8000/api/v1/update-ai-model", formData);
            if (res.data.error) {
                alert(res.data.error);
            } else {
                alert(res.data.message || "Đã tải trọng số AI mới thành công!");
            }
        } catch (error) {
            alert("Lỗi khi tải trọng số: " + (error.response?.data?.detail || error.message));
        }
        setIsUpdatingModel(false);
        if (modelInputRef.current) modelInputRef.current.value = "";
    };

    const handleSystemReset = async () => {
        if (!window.confirm("Hành động này sẽ xóa sạch dữ liệu và Database. Bạn có chắc chắn?")) return;
        setIsResetting(true);
        try {
            await axios.post("http://127.0.0.1:8000/api/v1/reset-system");
            if (onReset) onReset();
            alert("Hệ thống đã được đưa về trạng thái mặc định.");
        } catch (error) {
            alert("Không thể reset: " + error.message);
        }
        setIsResetting(false);
    };

    return (
        <div>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-end", marginBottom: 24 }}>
                <PageHeader title="Dashboard" subtitle="Solar farm monitoring overview (Real-time AI Data)" />
                <div style={{ display: "flex", gap: 12 }}>
                    <input 
                        type="file" 
                        accept=".zip" 
                        ref={fileInputRef} 
                        onChange={handleUploadAndAnalyze} 
                        style={{ display: "none" }} 
                    />
                    <ActionButton 
                        onClick={() => fileInputRef.current?.click()} 
                        disabled={isProcessing || isResetting}
                        icon={isProcessing ? <Loader2 className="animate-spin" size={16} /> : <Upload size={16} />}
                        style={{ background: "linear-gradient(135deg, #0EA5E9, #8B5CF6)", color: "white", border: "none" }}
                    >
                        {isProcessing ? statusText : "Tải lên & Phân tích"}
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
                        disabled={isProcessing || isResetting || isUpdatingModel}
                        style={{ background: "transparent", color: colors.primary, border: `1px solid ${colors.primary}50` }}
                        icon={isUpdatingModel ? <Loader2 className="animate-spin" size={16} /> : <Cpu size={16} />}
                    >
                        {isUpdatingModel ? "Đang Update AI..." : "Thay model AI"}
                    </ActionButton>

                    <ActionButton 
                        onClick={handleSystemReset} 
                        disabled={isProcessing || isResetting}
                        style={{ background: "#fee2e2", color: colors.error, border: `1px solid ${colors.error}30` }}
                        icon={isResetting ? <Loader2 className="animate-spin" size={16} /> : <Trash2 size={16} />}
                    >
                        Reset System
                    </ActionButton>
                </div>
            </div>
            
            <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 16, marginBottom: 24 }}>
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
                    accent={colors.danger} 
                />
                <KpiCard 
                    icon={<DollarSign size={20} />} 
                    label="ESTIMATED LOSS ($)" 
                    value={`$${estimatedLoss.toFixed(2)}`} 
                    accent={colors.warning} 
                />
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
                    {/* Thanh biểu đồ đơn giản */}
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