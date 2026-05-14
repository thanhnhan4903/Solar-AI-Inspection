import React, { useState } from "react";
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

    const [mapFocusTarget, setMapFocusTarget] = useState(null);

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
        <div style={{ display: "flex", height: "100vh", width: "100vw", background: colors.bg, fontFamily: "'DM Sans', system-ui, sans-serif", overflow: "hidden" }}>
            <Sidebar onNavigate={navigate} onLogout={handleLogout} activePage={activePage} />
            <div style={{ flex: 1, padding: isFullPage ? 0 : "28px 32px", overflowY: isFullPage ? "hidden" : "auto", position: "relative" }}>
                
                {page === "home" && (
                    <Home 
                        data={aiResults} 
                        onAnalysisComplete={(data, batchId) => {
                            setAiResults(data);
                            setCurrentBatchId(batchId);
                        }}
                        onReset={() => {
                            setAiResults([]);
                            setCurrentBatchId(null);
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

                {page === "ops" && <UnifiedDashboard data={aiResults} focusTarget={mapFocusTarget} />}
            </div>
        </div>
    );
}