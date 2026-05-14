import React, { useState, useMemo } from "react";
import { Search, Filter, ArrowUpDown } from "lucide-react";
import { colors } from "../../constants/theme";
import { PageHeader } from "../../components/layout/PageHeader";
import { ActionButton } from "../../components/ui/ActionButton";
import { BadgePill } from "../../components/ui/BadgePill";

// Gọi API lấy ảnh cắt 50% có khung xanh
const PANEL_IMAGE_API = "http://127.0.0.1:8000/api/v1/panel-image";

export default function PanelPage({ data, onSelect, onNavigate }) {
    if (!data || data.length === 0) return <div style={{ textAlign: "center", padding: 100 }}><ActionButton onClick={() => onNavigate("home")}>Vui lòng chạy AI tại Dashboard trước</ActionButton></div>;

    const [searchQuery, setSearchQuery] = useState("");
    const [statusFilter, setStatusFilter] = useState("all");
    const [sortBy, setSortBy] = useState("id_asc");

    const allPanels = useMemo(() => {
        if (!data) return [];
        return data.flatMap(img => img.panels.map(p => ({ 
            ...p, 
            sourceFile: img.filename,
            imgW: img.image_width,
            imgH: img.image_height,
            status: p.total_panel_loss > 0 ? "defective" : "healthy" 
        })));
    }, [data]);

    const filteredAndSortedPanels = useMemo(() => {
        let result = [...allPanels];

        if (searchQuery.trim() !== "") {
            result = result.filter(p => p.local_id.toLowerCase().includes(searchQuery.toLowerCase()));
        }

        if (statusFilter !== "all") {
            result = result.filter(p => p.status === statusFilter);
        }

        result.sort((a, b) => {
            if (sortBy === "id_asc") return a.local_id.localeCompare(b.local_id, undefined, { numeric: true });
            if (sortBy === "id_desc") return b.local_id.localeCompare(a.local_id, undefined, { numeric: true });
            if (sortBy === "loss_desc") return (b.total_panel_loss || 0) - (a.total_panel_loss || 0);
            if (sortBy === "loss_asc") return (a.total_panel_loss || 0) - (b.total_panel_loss || 0);
            return 0;
        });

        return result;
    }, [allPanels, searchQuery, statusFilter, sortBy]);

    return (
        <div>
            <PageHeader title="Panel Management" subtitle={`${filteredAndSortedPanels.length} / ${allPanels.length} panels showing`} />
            
            {/* Filter & Sort Bar */}
            <div style={{ display: "flex", gap: 16, marginBottom: 24, background: "#fff", padding: 16, borderRadius: 16, border: "1px solid #e2e8f0", alignItems: "center" }}>
                <div style={{ flex: 1, position: "relative" }}>
                    <Search size={18} style={{ position: "absolute", left: 12, top: "50%", transform: "translateY(-50%)", color: "#94a3b8" }} />
                    <input 
                        type="text" 
                        placeholder="Tìm kiếm ID tấm pin..." 
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
                        <option value="loss_desc">Mức độ lỗi giảm dần</option>
                        <option value="loss_asc">Mức độ lỗi tăng dần</option>
                    </select>
                </div>
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 16 }}>
                {filteredAndSortedPanels.map((p, index) => (
                    <div key={index} onClick={() => onSelect(p)} style={{ background: "#fff", borderRadius: 16, border: "1px solid #E2E8F0", overflow: "hidden", cursor: "pointer", transition: "transform 0.2s, box-shadow 0.2s" }}
                        onMouseEnter={e => { e.currentTarget.style.transform = "translateY(-2px)"; e.currentTarget.style.boxShadow = "0 8px 25px rgba(0,0,0,0.1)"; }}
                        onMouseLeave={e => { e.currentTarget.style.transform = "none"; e.currentTarget.style.boxShadow = "none"; }}
                    >
                        <div style={{ position: "relative", width: "100%", paddingTop: "75%", background: "#000" }}>
                            {/* Ảnh đã được cắt 50% và vẽ khung xanh từ Backend */}
                            <img 
                                src={`${PANEL_IMAGE_API}?filename=${p.sourceFile}&x1=${p.box[0]}&y1=${p.box[1]}&x2=${p.box[2]}&y2=${p.box[3]}${p.polygon ? `&polygon=${p.polygon.flat().join(',')}` : ''}`} 
                                style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%", objectFit: "cover" }} 
                                alt={p.local_id} 
                            />
                        </div>
                        <div style={{ padding: 12, display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                            <span style={{ fontWeight: 600, fontSize: 13 }}>{p.local_id}</span>
                            <BadgePill type={p.status === "defective" ? "hotspot" : "healthy"} />
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
}