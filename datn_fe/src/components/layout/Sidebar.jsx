import React from "react";
import { BarChart3, Map, Grid3X3, LogOut, Sun, Navigation } from "lucide-react";
import { colors } from "../../constants/theme";

const NAV = [
    { key: "home", icon: BarChart3, label: "Dashboard" },
    { key: "panel", icon: Grid3X3, label: "Panel" },
    { key: "ops", icon: Map, label: "Solar Operations" },
    { key: "report", icon: BarChart3, label: "Report" },
];

export function Sidebar({ onNavigate, onLogout, activePage }) {
    return (
        <aside style={{
            width: 240, background: colors.sidebar, display: "flex", flexDirection: "column",
            justifyContent: "space-between", padding: "24px 16px", flexShrink: 0,
        }}>
            <div>
                {/* Logo */}
                <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 36, paddingLeft: 8 }}>
                    <div style={{
                        width: 34, height: 34, borderRadius: 10,
                        background: "linear-gradient(135deg,#0EA5E9,#8B5CF6)",
                        display: "flex", alignItems: "center", justifyContent: "center",
                    }}>
                        <Sun size={18} color="#fff" />
                    </div>
                    <span style={{ fontSize: 17, fontWeight: 700, color: "#fff", letterSpacing: "-0.3px" }}>
                        Solar<span style={{ color: colors.primary }}>AI</span>
                    </span>
                </div>

                {/* Nav items */}
                <nav style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                    {NAV.map(({ key, icon: Icon, label }) => {
                        const active = activePage === key;
                        return (
                            <button key={key} onClick={() => onNavigate(key)} style={{
                                display: "flex", alignItems: "center", gap: 12,
                                padding: "10px 14px", borderRadius: 10, border: "none", cursor: "pointer",
                                background: active ? "linear-gradient(90deg,#0EA5E920,#8B5CF610)" : "transparent",
                                color: active ? colors.primary : "#94A3B8",
                                fontWeight: active ? 600 : 400,
                                fontSize: 14, transition: "all .15s",
                                borderLeft: active ? `3px solid ${colors.primary}` : "3px solid transparent",
                            }}
                                onMouseEnter={e => { if (!active) e.currentTarget.style.background = colors.sidebarHover; e.currentTarget.style.color = "#CBD5E1"; }}
                                onMouseLeave={e => { if (!active) e.currentTarget.style.background = "transparent"; e.currentTarget.style.color = "#94A3B8"; }}
                            >
                                <Icon size={17} />
                                {label}
                            </button>
                        );
                    })}
                </nav>
            </div>

            <button onClick={onLogout} style={{
                display: "flex", alignItems: "center", gap: 10, padding: "10px 14px",
                borderRadius: 10, border: "1px solid #EF444430", background: "#EF444410",
                color: "#F87171", fontSize: 14, fontWeight: 500, cursor: "pointer", transition: "all .15s",
            }}
                onMouseEnter={e => { e.currentTarget.style.background = "#EF444425"; }}
                onMouseLeave={e => { e.currentTarget.style.background = "#EF444410"; }}
            >
                <LogOut size={16} /> Logout
            </button>
        </aside>
    );
}
