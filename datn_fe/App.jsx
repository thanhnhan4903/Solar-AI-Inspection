import React, { useState, useEffect } from "react";
import { fetchLatestBatch } from "./src/api";
import { Sidebar } from "./src/components/layout/Sidebar";
import { colors } from "./src/constants/theme";

// Pages
import Login from "./src/pages/Login";
import Home from "./src/pages/Dashboard/Home";
import PanelPage from "./src/pages/Panel/PanelPage";
import PanelDetail from "./src/pages/Panel/PanelDetail";
import UnifiedDashboard from "./src/pages/Unified/UnifiedDashboard";
import ReportPage from "./src/pages/Report/ReportPage";
// import UploadBatchPage from "./src/pages/Upload/UploadBatchPage";

export default function App() {
    const [isAuth, setIsAuth] = useState(localStorage.getItem("isAuth") === "true");
    const [page, setPage] = useState("home");
    const [selectedPanel, setSelectedPanel] = useState(null);
    const [activePage, setActivePage] = useState("home");

    // DỮ LIỆU TỪ BACKEND
    const [aiResults, setAiResults] = useState([]); 
    const [currentBatchId, setCurrentBatchId] = useState(null);
    const [currentPanelPower, setCurrentPanelPower] = useState(600);

    const [mapFocusTarget, setMapFocusTarget] = useState(null);

    useEffect(() => {
        if (isAuth) {
            fetchLatestBatch().then(res => {
                if (res.data && res.data.batch_id) {
                    setAiResults(res.data.data);
                    setCurrentBatchId(res.data.batch_id);
                    setCurrentPanelPower(res.data.panel_power || 600);
                }
            }).catch(console.error);
        }
    }, [isAuth]);

    const handleLogin = () => { localStorage.setItem("isAuth", "true"); setIsAuth(true); };
    const handleLogout = () => { localStorage.removeItem("isAuth"); setIsAuth(false); };
    
    // Hàm điều hướng chung (từ Sidebar)
    const navigate = (p) => { 
        if (p !== "ops") setMapFocusTarget(null);
        setPage(p); 
        setActivePage(p); 
    };

    if (!isAuth) return <Login onLogin={handleLogin} />;

    const isFullPage = page === "ops";

    return (
        <div style={{ display: "flex", height: "100vh", width: "100vw", fontFamily: "'DM Sans', system-ui, sans-serif", overflow: "hidden",
            background: "linear-gradient(135deg, #e8f4fd 0%, #f0f9ff 30%, #fef9ee 60%, #f0fdf4 100%)" }}>
            <Sidebar onNavigate={navigate} onLogout={handleLogout} activePage={activePage} />
            <div style={{ flex: 1, minWidth: 0, padding: isFullPage ? 0 : "28px 32px", overflow: isFullPage ? "hidden" : "auto", position: "relative",
                backgroundImage: `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='800' height='600' viewBox='0 0 800 600'%3E%3Cdefs%3E%3CradialGradient id='sun' cx='75%25' cy='20%25' r='30%25'%3E%3Cstop offset='0%25' stop-color='%23fbbf24' stop-opacity='0.12'/%3E%3Cstop offset='100%25' stop-color='%23f59e0b' stop-opacity='0'/%3E%3C/radialGradient%3E%3CradialGradient id='sun2' cx='10%25' cy='80%25' r='25%25'%3E%3Cstop offset='0%25' stop-color='%230ea5e9' stop-opacity='0.08'/%3E%3Cstop offset='100%25' stop-color='%230ea5e9' stop-opacity='0'/%3E%3C/radialGradient%3E%3C/defs%3E%3Crect width='800' height='600' fill='url(%23sun)'/%3E%3Crect width='800' height='600' fill='url(%23sun2)'/%3E%3Cg opacity='0.04' stroke='%23f59e0b' stroke-width='1' fill='none'%3E%3Crect x='50' y='400' width='60' height='40' rx='3'/%3E%3Crect x='120' y='400' width='60' height='40' rx='3'/%3E%3Crect x='190' y='400' width='60' height='40' rx='3'/%3E%3Crect x='50' y='450' width='60' height='40' rx='3'/%3E%3Crect x='120' y='450' width='60' height='40' rx='3'/%3E%3Crect x='190' y='450' width='60' height='40' rx='3'/%3E%3C/g%3E%3Ccircle cx='600' cy='120' r='80' fill='none' stroke='%23fbbf24' stroke-width='1' opacity='0.08'/%3E%3Ccircle cx='600' cy='120' r='50' fill='none' stroke='%23fbbf24' stroke-width='1' opacity='0.06'/%3E%3Cline x1='600' y1='30' x2='600' y2='10' stroke='%23fbbf24' stroke-width='2' opacity='0.1'/%3E%3Cline x1='600' y1='210' x2='600' y2='230' stroke='%23fbbf24' stroke-width='2' opacity='0.1'/%3E%3Cline x1='510' y1='120' x2='490' y2='120' stroke='%23fbbf24' stroke-width='2' opacity='0.1'/%3E%3Cline x1='690' y1='120' x2='710' y2='120' stroke='%23fbbf24' stroke-width='2' opacity='0.1'/%3E%3Cline x1='537' y1='47' x2='523' y2='33' stroke='%23fbbf24' stroke-width='2' opacity='0.1'/%3E%3Cline x1='663' y1='193' x2='677' y2='207' stroke='%23fbbf24' stroke-width='2' opacity='0.1'/%3E%3Cline x1='663' y1='47' x2='677' y2='33' stroke='%23fbbf24' stroke-width='2' opacity='0.1'/%3E%3Cline x1='537' y1='193' x2='523' y2='207' stroke='%23fbbf24' stroke-width='2' opacity='0.1'/%3E%3C/svg%3E")`,
                backgroundSize: "cover",
                backgroundRepeat: "no-repeat",
                backgroundPosition: "top right",
            }}>
                
                {page === "home" && (
                    <Home 
                        data={aiResults} 
                        batchId={currentBatchId}
                        onAnalysisComplete={(data, batchId, panelPower) => {
                            setAiResults(data);
                            setCurrentBatchId(batchId);
                            if (panelPower !== undefined && panelPower !== null) {
                                setCurrentPanelPower(panelPower);
                            }
                        }}
                        onReset={() => {
                            setAiResults([]);
                            setCurrentBatchId(null);
                            setCurrentPanelPower(600);
                        }}
                    />
                )}

                {page === "panel" && (
                    <PanelPage 
                        data={aiResults} 
                        onSelect={(p) => { setSelectedPanel(p); navigate("detail"); }} 
                        onNavigate={navigate} 
                    />
                )}

                {page === "detail" && (
                    <PanelDetail 
                        panel={selectedPanel} 
                        data={aiResults}
                        panelPower={currentPanelPower}
                        onSelect={(p) => setSelectedPanel(p)}
                        onBack={() => navigate("panel")} 
                        onViewOnMap={(img) => {
                            setMapFocusTarget(img.filename);
                            setPage("ops");
                            setActivePage("ops");
                        }}
                    />
                )}

                {page === "report" && <ReportPage data={aiResults} batchId={currentBatchId} />}

                {page === "ops" && <UnifiedDashboard data={aiResults} panelPower={currentPanelPower} focusTarget={mapFocusTarget} />}
            </div>
        </div>
    );
}