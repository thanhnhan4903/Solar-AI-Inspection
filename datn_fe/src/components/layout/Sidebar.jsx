import React from "react";
import { BarChart3, Map, Grid3X3, LogOut } from "lucide-react";
import { colors } from "../../constants/theme";
import epcLogo from "../../assets/epc_solar_logo.png";

const NAV = [
    { key: "home", icon: BarChart3, label: "Bảng điều khiển" },
    { key: "ops", icon: Map, label: "Bản đồ phân tích" },
    { key: "report", icon: BarChart3, label: "Báo cáo" },
];

export function Sidebar({ onNavigate, onLogout, activePage }) {
    return (
        <aside style={{
            width: 240,
            background: "linear-gradient(180deg, #0F172A 0%, #1a2744 100%)",
            display: "flex",
            flexDirection: "column",
            justifyContent: "space-between",
            padding: "24px 16px",
            flexShrink: 0,
            borderRight: "1px solid rgba(255,255,255,0.06)",
            boxShadow: "4px 0 20px rgba(0,0,0,0.3)",
        }}>
            <div>
                {/* Logo EPC Solar */}
                <div style={{
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    marginBottom: 24,
                    padding: "4px 6px",
                    background: "rgba(255,255,255,0.98)",
                    borderRadius: 6,
                    height: "44px",
                    boxShadow: "0 1px 6px rgba(0,0,0,0.15)",
                    overflow: "hidden"
                }}>
                    <img
                        src={epcLogo}
                        alt="EPC Solar"
                        style={{ height: "100%", width: "auto", objectFit: "contain" }}
                    />
                </div>

                {/* Divider */}
                <div style={{ height: 1, background: "rgba(255,255,255,0.08)", marginBottom: 16 }} />

                {/* Nav items */}
                <nav style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                    {NAV.map(({ key, icon: Icon, label }) => {
                        const active = activePage === key;
                        return (
                            <button
                                key={key}
                                onClick={() => onNavigate(key)}
                                style={{
                                    display: "flex",
                                    alignItems: "center",
                                    gap: 12,
                                    padding: "10px 14px",
                                    borderRadius: 10,
                                    border: "none",
                                    cursor: "pointer",
                                    background: active
                                        ? "linear-gradient(90deg, rgba(14,165,233,0.2), rgba(139,92,246,0.1))"
                                        : "transparent",
                                    color: active ? colors.primary : "#94A3B8",
                                    fontWeight: active ? 600 : 400,
                                    fontSize: 14,
                                    transition: "all .15s",
                                    borderLeft: active ? `3px solid ${colors.primary}` : "3px solid transparent",
                                    textAlign: "left",
                                    width: "100%",
                                }}
                                onMouseEnter={e => {
                                    if (!active) {
                                        e.currentTarget.style.background = "rgba(255,255,255,0.06)";
                                        e.currentTarget.style.color = "#CBD5E1";
                                    }
                                }}
                                onMouseLeave={e => {
                                    if (!active) {
                                        e.currentTarget.style.background = "transparent";
                                        e.currentTarget.style.color = "#94A3B8";
                                    }
                                }}
                            >
                                <Icon size={17} />
                                {label}
                            </button>
                        );
                    })}
                </nav>
            </div>

            {/* Version badge */}
            <div>
                <div style={{
                    fontSize: 11,
                    color: "#475569",
                    textAlign: "center",
                    marginBottom: 12,
                    letterSpacing: "0.5px",
                }}>
                    Kiểm Tra Điện Mặt Trời AI v2.0
                </div>
                <button
                    onClick={onLogout}
                    style={{
                        display: "flex",
                        alignItems: "center",
                        gap: 10,
                        padding: "10px 14px",
                        borderRadius: 10,
                        border: "1px solid rgba(239,68,68,0.3)",
                        background: "rgba(239,68,68,0.08)",
                        color: "#F87171",
                        fontSize: 14,
                        fontWeight: 500,
                        cursor: "pointer",
                        transition: "all .15s",
                        width: "100%",
                    }}
                    onMouseEnter={e => { e.currentTarget.style.background = "rgba(239,68,68,0.18)"; }}
                    onMouseLeave={e => { e.currentTarget.style.background = "rgba(239,68,68,0.08)"; }}
                >
                    <LogOut size={16} /> Đăng xuất
                </button>
            </div>
        </aside>
    );
}
