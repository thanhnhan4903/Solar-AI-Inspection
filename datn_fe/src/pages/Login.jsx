import React, { useState, useEffect } from "react";
import { Sun, Lock, User, Cpu, ShieldCheck, AlertTriangle } from "lucide-react";
import { loginUser } from "../api";

export default function Login({ onLogin }) {
    const [username, setUsername] = useState("admin123");
    const [password, setPassword] = useState("admin123");
    const [error, setError] = useState("");
    const [isLoading, setIsLoading] = useState(false);
    const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });

    useEffect(() => {
        const handleMouseMove = (e) => {
            setMousePosition({
                x: (e.clientX / window.innerWidth) * 100,
                y: (e.clientY / window.innerHeight) * 100,
            });
        };
        window.addEventListener("mousemove", handleMouseMove);
        return () => window.removeEventListener("mousemove", handleMouseMove);
    }, []);

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError("");
        
        if (!username.trim() || !password.trim()) { 
            setError("Vui lòng nhập đầy đủ tài khoản và mật khẩu."); 
            return; 
        }

        setIsLoading(true);
        try {
            const res = await loginUser(username, password);
            if (res.data.error) {
                setError(res.data.error);
            } else {
                // Save user data
                localStorage.setItem("user", JSON.stringify(res.data.user));
                onLogin();
            }
        } catch (err) {
            setError("Lỗi kết nối tới máy chủ.");
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div style={{
            display: "flex", alignItems: "center", justifyContent: "center",
            height: "100vh", width: "100vw",
            backgroundColor: "#030712",
            backgroundImage: `
                radial-gradient(circle at ${mousePosition.x}% ${mousePosition.y}%, rgba(14, 165, 233, 0.15) 0%, transparent 40%),
                linear-gradient(rgba(255, 255, 255, 0.02) 1px, transparent 1px),
                linear-gradient(90deg, rgba(255, 255, 255, 0.02) 1px, transparent 1px)
            `,
            backgroundSize: "100% 100%, 40px 40px, 40px 40px",
            backgroundPosition: "0 0, 0 0, 0 0",
            fontFamily: "'Space Grotesk', 'Inter', sans-serif",
            overflow: "hidden",
            position: "relative"
        }}>
            {/* Glowing Orbs */}
            <div style={{ position: "absolute", width: "60vw", height: "60vw", borderRadius: "50%", background: "radial-gradient(circle, rgba(14, 165, 233, 0.05) 0%, transparent 70%)", top: "-20%", left: "-10%", filter: "blur(60px)", animation: "pulse 8s infinite alternate" }} />
            <div style={{ position: "absolute", width: "50vw", height: "50vw", borderRadius: "50%", background: "radial-gradient(circle, rgba(139, 92, 246, 0.05) 0%, transparent 70%)", bottom: "-20%", right: "-10%", filter: "blur(60px)", animation: "pulse 10s infinite alternate-reverse" }} />

            <div style={{
                position: "relative", 
                width: 440, 
                background: "rgba(15, 23, 42, 0.6)",
                backdropFilter: "blur(20px)", 
                WebkitBackdropFilter: "blur(20px)",
                border: "1px solid rgba(14, 165, 233, 0.2)",
                borderTop: "1px solid rgba(14, 165, 233, 0.5)",
                borderRadius: 24, 
                padding: "48px 40px", 
                boxShadow: "0 32px 80px rgba(0,0,0,0.6), inset 0 0 32px rgba(14, 165, 233, 0.05)",
                zIndex: 10
            }}>
                {/* Tech Accents */}
                <div style={{ position: "absolute", top: -1, left: 40, width: 80, height: 1, background: "linear-gradient(90deg, transparent, #0EA5E9, transparent)" }} />
                <div style={{ position: "absolute", bottom: -1, right: 40, width: 80, height: 1, background: "linear-gradient(90deg, transparent, #8B5CF6, transparent)" }} />

                <div style={{ display: "flex", flexDirection: "column", alignItems: "center", marginBottom: 36 }}>
                    {/* EPC Solar Logo */}
                    <div style={{ position: "relative", marginBottom: 16 }}>
                        <div style={{
                            width: 80, height: 80, borderRadius: 20,
                            background: "linear-gradient(135deg, rgba(14, 165, 233, 0.1), rgba(139, 92, 246, 0.1))",
                            border: "1px solid rgba(14, 165, 233, 0.3)",
                            display: "flex", alignItems: "center", justifyContent: "center",
                            boxShadow: "0 0 20px rgba(14, 165, 233, 0.2), inset 0 0 15px rgba(14, 165, 233, 0.2)",
                            position: "relative",
                            overflow: "hidden"
                        }}>
                            <Cpu size={40} color="#0EA5E9" style={{ position: "absolute", opacity: 0.3 }} />
                            <Sun size={36} color="#38BDF8" style={{ filter: "drop-shadow(0 0 8px rgba(56, 189, 248, 0.8))" }} />
                        </div>
                        {/* Decorative dots */}
                        <div style={{ position: "absolute", top: -5, right: -5, width: 10, height: 10, borderRadius: "50%", background: "#0EA5E9", boxShadow: "0 0 10px #0EA5E9" }} />
                        <div style={{ position: "absolute", bottom: -5, left: -5, width: 10, height: 10, borderRadius: "50%", background: "#8B5CF6", boxShadow: "0 0 10px #8B5CF6" }} />
                    </div>
                    
                    <h1 style={{ color: "#F8FAFC", fontSize: 26, fontWeight: 800, letterSpacing: 1, margin: "0 0 4px 0", textShadow: "0 2px 10px rgba(0,0,0,0.5)" }}>
                        EPC <span style={{ color: "#38BDF8" }}>SOLAR</span>
                    </h1>
                    <div style={{ color: "#94A3B8", fontSize: 13, letterSpacing: 4, fontWeight: 600, textTransform: "uppercase" }}>
                        ĐÀ NẴNG
                    </div>
                    <div style={{ marginTop: 12, padding: "4px 12px", background: "rgba(16, 185, 129, 0.1)", border: "1px solid rgba(16, 185, 129, 0.3)", borderRadius: 100, display: "flex", alignItems: "center", gap: 6 }}>
                        <ShieldCheck size={14} color="#34D399" />
                        <span style={{ color: "#34D399", fontSize: 12, fontWeight: 600 }}>AI Inspection System</span>
                    </div>
                </div>

                <form onSubmit={handleSubmit} style={{ display: "flex", flexDirection: "column", gap: 20 }}>
                    <div style={{ position: "relative" }}>
                        <div style={{ position: "absolute", top: 14, left: 16, color: "#64748B" }}>
                            <User size={18} />
                        </div>
                        <input 
                            placeholder="Tài khoản" 
                            value={username} 
                            onChange={e => setUsername(e.target.value)}
                            style={{
                                width: "100%", padding: "14px 16px 14px 44px", borderRadius: 12,
                                border: "1px solid rgba(255,255,255,0.08)", background: "rgba(0,0,0,0.2)",
                                color: "#F8FAFC", fontSize: 15, outline: "none", transition: "all .2s",
                                boxSizing: "border-box", letterSpacing: 0.5
                            }} 
                            onFocus={e => { e.target.style.borderColor = "#0EA5E9"; e.target.style.background = "rgba(14, 165, 233, 0.05)"; e.target.style.boxShadow = "0 0 0 3px rgba(14, 165, 233, 0.1)"; }}
                            onBlur={e => { e.target.style.borderColor = "rgba(255,255,255,0.08)"; e.target.style.background = "rgba(0,0,0,0.2)"; e.target.style.boxShadow = "none"; }} 
                        />
                    </div>

                    <div style={{ position: "relative" }}>
                        <div style={{ position: "absolute", top: 14, left: 16, color: "#64748B" }}>
                            <Lock size={18} />
                        </div>
                        <input 
                            placeholder="Mật khẩu" 
                            type="password" 
                            value={password} 
                            onChange={e => setPassword(e.target.value)}
                            style={{
                                width: "100%", padding: "14px 16px 14px 44px", borderRadius: 12,
                                border: "1px solid rgba(255,255,255,0.08)", background: "rgba(0,0,0,0.2)",
                                color: "#F8FAFC", fontSize: 15, outline: "none", transition: "all .2s",
                                boxSizing: "border-box", letterSpacing: 2
                            }} 
                            onFocus={e => { e.target.style.borderColor = "#8B5CF6"; e.target.style.background = "rgba(139, 92, 246, 0.05)"; e.target.style.boxShadow = "0 0 0 3px rgba(139, 92, 246, 0.1)"; }}
                            onBlur={e => { e.target.style.borderColor = "rgba(255,255,255,0.08)"; e.target.style.background = "rgba(0,0,0,0.2)"; e.target.style.boxShadow = "none"; }} 
                        />
                    </div>

                    {error && (
                        <div style={{ 
                            padding: "10px 14px", background: "rgba(239, 68, 68, 0.1)", border: "1px solid rgba(239, 68, 68, 0.3)", 
                            borderRadius: 10, color: "#FCA5A5", fontSize: 13, textAlign: "center", display: "flex", alignItems: "center", justifyContent: "center", gap: 6 
                        }}>
                            <AlertTriangle size={14} /> {error}
                        </div>
                    )}

                    <button 
                        type="submit"
                        disabled={isLoading}
                        style={{
                            padding: "16px", borderRadius: 12, border: "none", cursor: isLoading ? "not-allowed" : "pointer", 
                            fontWeight: 700, fontSize: 15, color: "#fff", letterSpacing: 1,
                            background: "linear-gradient(135deg, #0EA5E9, #8B5CF6)",
                            marginTop: 10, transition: "all .2s", position: "relative", overflow: "hidden",
                            boxShadow: "0 10px 20px rgba(14, 165, 233, 0.3)"
                        }}
                        onMouseEnter={e => { if(!isLoading) { e.currentTarget.style.transform = "translateY(-2px)"; e.currentTarget.style.boxShadow = "0 15px 25px rgba(139, 92, 246, 0.4)"; } }}
                        onMouseLeave={e => { if(!isLoading) { e.currentTarget.style.transform = "translateY(0)"; e.currentTarget.style.boxShadow = "0 10px 20px rgba(14, 165, 233, 0.3)"; } }}
                    >
                        {isLoading ? (
                            <span style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 8 }}>
                                <Cpu size={18} style={{ animation: "spin 2s linear infinite" }} /> ĐANG XÁC THỰC...
                            </span>
                        ) : "ĐĂNG NHẬP HỆ THỐNG"}
                        
                        {/* Shimmer effect */}
                        <div style={{
                            position: "absolute", top: 0, left: "-100%", width: "50%", height: "100%",
                            background: "linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent)",
                            animation: isLoading ? "none" : "shimmer 3s infinite"
                        }} />
                    </button>
                    
                    <div style={{ display: "flex", justifyContent: "space-between", marginTop: 8 }}>
                        <span style={{ color: "#64748B", fontSize: 13 }}>Version 2.4.0</span>
                        <span style={{ color: "#0EA5E9", fontSize: 13, cursor: "pointer", transition: "color 0.2s" }} onMouseEnter={e => e.currentTarget.style.color = "#38BDF8"} onMouseLeave={e => e.currentTarget.style.color = "#0EA5E9"}>Quên mật khẩu?</span>
                    </div>
                </form>
            </div>
            
            {/* Inject keyframes for animations */}
            <style>{`
                @keyframes pulse {
                    0% { transform: scale(1); opacity: 0.5; }
                    100% { transform: scale(1.1); opacity: 0.8; }
                }
                @keyframes spin {
                    from { transform: rotate(0deg); }
                    to { transform: rotate(360deg); }
                }
                @keyframes shimmer {
                    0% { left: -100%; }
                    20% { left: 200%; }
                    100% { left: 200%; }
                }
                * {
                    scrollbar-width: thin;
                    scrollbar-color: #334155 transparent;
                }
            `}</style>
        </div>
    );
}
