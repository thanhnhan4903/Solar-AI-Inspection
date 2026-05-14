import React, { useState, useMemo } from "react";
import { Search, Filter, ArrowUpDown } from "lucide-react";
import { colors } from "../../constants/theme";
import { PageHeader } from "../../components/layout/PageHeader";
import { ActionButton } from "../../components/ui/ActionButton";
import { BadgePill } from "../../components/ui/BadgePill";

// Lấy ảnh gốc precalib từ backend
const IMAGE_BASE_API = "http://127.0.0.1:8000/data/precalib/";

export default function PanelPage({ data, onSelect, onNavigate }) {
    if (!data || data.length === 0) return <div style={{ textAlign: "center", padding: 100 }}><ActionButton onClick={() => onNavigate("home")}>Vui lòng chạy AI tại Dashboard trước</ActionButton></div>;

    const [searchQuery, setSearchQuery] = useState("");
    const [statusFilter, setStatusFilter] = useState("all");
    const [sortBy, setSortBy] = useState("id_asc");

    const allImages = useMemo(() => {
        return data.map((img, index) => {
            const faultyPanels = img.panels.filter(p => p.total_panel_loss > 0);
            return {
                ...img,
                id: `Hình ${index + 1}`,
                status: faultyPanels.length > 0 ? "defective" : "healthy",
                faulty_count: faultyPanels.length
            };
        });
    }, [data]);

    const filteredAndSortedImages = useMemo(() => {
        let result = [...allImages];

        if (searchQuery.trim() !== "") {
            result = result.filter(img => 
                img.id.toLowerCase().includes(searchQuery.toLowerCase()) || 
                img.filename.toLowerCase().includes(searchQuery.toLowerCase())
            );
        }

        if (statusFilter !== "all") {
            result = result.filter(img => img.status === statusFilter);
        }

        result.sort((a, b) => {
            if (sortBy === "id_asc") return a.id.localeCompare(b.id, undefined, { numeric: true });
            if (sortBy === "id_desc") return b.id.localeCompare(a.id, undefined, { numeric: true });
            if (sortBy === "loss_desc") return b.faulty_count - a.faulty_count;
            if (sortBy === "loss_asc") return a.faulty_count - b.faulty_count;
            return 0;
        });

        return result;
    }, [allImages, searchQuery, statusFilter, sortBy]);

    return (
        <div>
            <PageHeader title="Quản lý Hình ảnh" subtitle={`Đang hiển thị ${filteredAndSortedImages.length} / ${allImages.length} hình`} />
            
            {/* Filter & Sort Bar */}
            <div style={{ display: "flex", gap: 16, marginBottom: 24, background: "#fff", padding: 16, borderRadius: 16, border: "1px solid #e2e8f0", alignItems: "center" }}>
                <div style={{ flex: 1, position: "relative" }}>
                    <Search size={18} style={{ position: "absolute", left: 12, top: "50%", transform: "translateY(-50%)", color: "#94a3b8" }} />
                    <input 
                        type="text" 
                        placeholder="Tìm kiếm ID hình ảnh hoặc tên file..." 
                        value={searchQuery}
                        onChange={e => setSearchQuery(e.target.value)}
                        style={{ width: "100%", padding: "10px 10px 10px 38px", borderRadius: 8, border: "1px solid #e2e8f0", outline: "none", fontSize: 14 }}
                    />
                </div>
                
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                    <Filter size={18} color="#64748b" />
                    <select 
                        value={statusFilter} 
                        onChange={e => setStatusFilter(e.target.value)}
                        style={{ padding: "10px 16px 10px 12px", borderRadius: 8, border: "1px solid #e2e8f0", outline: "none", background: "#f8fafc", cursor: "pointer", fontSize: 14 }}
                    >
                        <option value="all">Tất cả trạng thái</option>
                        <option value="healthy">Bình thường</option>
                        <option value="defective">Có lỗi</option>
                    </select>
                </div>

                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                    <ArrowUpDown size={18} color="#64748b" />
                    <select 
                        value={sortBy} 
                        onChange={e => setSortBy(e.target.value)}
                        style={{ padding: "10px 16px 10px 12px", borderRadius: 8, border: "1px solid #e2e8f0", outline: "none", background: "#f8fafc", cursor: "pointer", fontSize: 14 }}
                    >
                        <option value="id_asc">ID (A-Z)</option>
                        <option value="id_desc">ID (Z-A)</option>
                        <option value="loss_desc">Số lỗi giảm dần</option>
                        <option value="loss_asc">Số lỗi tăng dần</option>
                    </select>
                </div>
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 24 }}>
                {filteredAndSortedImages.map((img, index) => (
                    <div key={index} onClick={() => onSelect(img)} style={{ background: "#fff", borderRadius: 16, border: "1px solid #E2E8F0", overflow: "hidden", cursor: "pointer", transition: "transform 0.2s, box-shadow 0.2s" }}
                        onMouseEnter={e => { e.currentTarget.style.transform = "translateY(-4px)"; e.currentTarget.style.boxShadow = "0 12px 30px rgba(0,0,0,0.1)"; }}
                        onMouseLeave={e => { e.currentTarget.style.transform = "none"; e.currentTarget.style.boxShadow = "none"; }}
                    >
                        <div style={{ position: "relative", width: "100%", paddingTop: "75%", background: "#000" }}>
                            {/* Hiển thị toàn cảnh ảnh gốc */}
                            <img 
                                src={`${IMAGE_BASE_API}${img.filename}`} 
                                style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%", objectFit: "cover" }} 
                                alt={img.id} 
                            />
                        </div>
                        <div style={{ padding: "16px 20px" }}>
                            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
                                <span style={{ fontWeight: 700, fontSize: 15, color: "#1E293B" }}>{img.id}</span>
                                <BadgePill type={img.status === "defective" ? "hotspot" : "healthy"} />
                            </div>
                            <div style={{ fontSize: 13, color: "#64748B", display: "flex", justifyContent: "space-between" }}>
                                <span>{img.total_panels} tấm pin</span>
                                {img.faulty_count > 0 && <span style={{ color: colors.error, fontWeight: 600 }}>{img.faulty_count} tấm bị lỗi</span>}
                            </div>
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
}