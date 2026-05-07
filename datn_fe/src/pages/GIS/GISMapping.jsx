import React, { useState, useMemo, useEffect } from 'react';
import { 
    MapContainer, 
    TileLayer, 
    Marker, 
    Popup, 
    Tooltip, 
    Polygon, 
    useMap,
    CircleMarker
} from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import { 
    Search, 
    Filter, 
    Maximize, 
    Layers, 
    Info, 
    Activity, 
    AlertTriangle, 
    CheckCircle,
    X,
    ChevronRight,
    Map as MapIcon,
    Thermometer
} from 'lucide-react';
import { generateMockPanels, PANEL_STATUS, STATUS_COLORS } from './mockData';
import './MapStyles.css';

// Fix Leaflet marker icon issue in React
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
    iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
    iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
    shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
});

// Component to handle Map interactions like Fullscreen
const MapController = ({ zoom }) => {
    const map = useMap();
    useEffect(() => {
        if (zoom) map.setZoom(zoom);
    }, [zoom, map]);
    return null;
};

export default function GISMapping() {
    const [panels] = useState(() => generateMockPanels(150));
    const [searchQuery, setSearchQuery] = useState('');
    const [statusFilter, setStatusFilter] = useState('All');
    const [viewMode, setViewMode] = useState('polygon'); // 'marker' or 'polygon'
    const [showHeatmap, setShowHeatmap] = useState(false);
    const [selectedPanel, setSelectedPanel] = useState(null);

    // Derived data
    const filteredPanels = useMemo(() => {
        return panels.filter(panel => {
            const matchesSearch = panel.id.toLowerCase().includes(searchQuery.toLowerCase());
            const matchesStatus = statusFilter === 'All' || panel.status === statusFilter;
            return matchesSearch && matchesStatus;
        });
    }, [panels, searchQuery, statusFilter]);

    const stats = useMemo(() => {
        const counts = {
            Total: panels.length,
            [PANEL_STATUS.HEALTHY]: 0,
            [PANEL_STATUS.HOTSPOT]: 0,
            [PANEL_STATUS.SOILING]: 0,
            [PANEL_STATUS.CRACK]: 0,
            [PANEL_STATUS.OFFLINE]: 0,
        };
        panels.forEach(p => counts[p.status]++);
        return counts;
    }, [panels]);

    const handlePanelClick = (panel) => {
        setSelectedPanel(panel);
    };

    return (
        <div className="gis-container">
            {/* Header / Search Overlay */}
            <div className="map-overlay filter-sidebar">
                <div className="flex items-center justify-between mb-6">
                    <h2 className="text-xl font-bold flex items-center gap-2">
                        <MapIcon className="text-blue-400" size={24} />
                        GIS Mapping
                    </h2>
                    <div className="text-xs bg-blue-500/20 text-blue-400 px-2 py-1 rounded">LIVE</div>
                </div>

                {/* Search */}
                <div className="relative mb-4">
                    <Search className="absolute left-3 top-2.5 text-gray-400" size={18} />
                    <input 
                        type="text" 
                        placeholder="Search Panel ID..." 
                        className="w-full bg-white/5 border border-white/10 rounded-lg py-2 pl-10 pr-4 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 transition"
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                    />
                </div>

                {/* Filters */}
                <div className="space-y-4">
                    <div>
                        <label className="text-xs text-gray-400 uppercase font-semibold mb-2 block">Status Filter</label>
                        <div className="grid grid-cols-2 gap-2">
                            <button 
                                onClick={() => setStatusFilter('All')}
                                className={`text-xs py-2 px-3 rounded-lg border transition ${statusFilter === 'All' ? 'bg-blue-600 border-blue-500 text-white' : 'bg-white/5 border-white/10 text-gray-400 hover:bg-white/10'}`}
                            >
                                All Panels
                            </button>
                            {Object.values(PANEL_STATUS).map(status => (
                                <button 
                                    key={status}
                                    onClick={() => setStatusFilter(status)}
                                    className={`text-xs py-2 px-3 rounded-lg border transition flex items-center gap-2 ${statusFilter === status ? 'bg-blue-600 border-blue-500 text-white' : 'bg-white/5 border-white/10 text-gray-400 hover:bg-white/10'}`}
                                >
                                    <span className="w-2 h-2 rounded-full" style={{ background: STATUS_COLORS[status] }}></span>
                                    {status}
                                </button>
                            ))}
                        </div>
                    </div>

                    <div>
                        <label className="text-xs text-gray-400 uppercase font-semibold mb-2 block">Map Layers</label>
                        <div className="space-y-2">
                            <button 
                                onClick={() => setViewMode(viewMode === 'polygon' ? 'marker' : 'polygon')}
                                className="w-full flex items-center justify-between bg-white/5 border border-white/10 p-3 rounded-lg text-sm hover:bg-white/10 transition"
                            >
                                <div className="flex items-center gap-3">
                                    <Layers size={18} className="text-blue-400" />
                                    <span>{viewMode === 'polygon' ? 'Polygon Mode' : 'Marker Mode'}</span>
                                </div>
                                <div className={`w-10 h-5 rounded-full relative transition ${viewMode === 'polygon' ? 'bg-blue-600' : 'bg-gray-600'}`}>
                                    <div className={`absolute top-1 w-3 h-3 bg-white rounded-full transition-all ${viewMode === 'polygon' ? 'right-1' : 'left-1'}`}></div>
                                </div>
                            </button>
                            <button 
                                onClick={() => setShowHeatmap(!showHeatmap)}
                                className="w-full flex items-center justify-between bg-white/5 border border-white/10 p-3 rounded-lg text-sm hover:bg-white/10 transition"
                            >
                                <div className="flex items-center gap-3">
                                    <Thermometer size={18} className="text-orange-400" />
                                    <span>Thermal Heatmap</span>
                                </div>
                                <div className={`w-10 h-5 rounded-full relative transition ${showHeatmap ? 'bg-orange-600' : 'bg-gray-600'}`}>
                                    <div className={`absolute top-1 w-3 h-3 bg-white rounded-full transition-all ${showHeatmap ? 'right-1' : 'left-1'}`}></div>
                                </div>
                            </button>
                        </div>
                    </div>
                </div>

                {/* Mini Stats in Sidebar */}
                <div className="mt-8 pt-6 border-t border-white/10">
                    <h3 className="text-sm font-bold mb-4">Quick Stats</h3>
                    <div className="space-y-3">
                        <div className="flex justify-between items-center">
                            <span className="text-sm text-gray-400">Total Panels</span>
                            <span className="text-sm font-mono">{stats.Total}</span>
                        </div>
                        <div className="flex justify-between items-center">
                            <span className="text-sm text-gray-400">Healthy</span>
                            <span className="text-sm font-mono text-green-400">{stats[PANEL_STATUS.HEALTHY]}</span>
                        </div>
                        <div className="flex justify-between items-center">
                            <span className="text-sm text-gray-400">Issues Detected</span>
                            <span className="text-sm font-mono text-red-400">{stats.Total - stats[PANEL_STATUS.HEALTHY]}</span>
                        </div>
                    </div>
                </div>
            </div>

            {/* Top Stats Overlay */}
            <div className="map-overlay stats-panel">
                <div className="flex items-center gap-3 pr-4 border-r border-white/10">
                    <div className="p-2 bg-green-500/20 rounded-lg">
                        <CheckCircle size={20} className="text-green-500" />
                    </div>
                    <div>
                        <div className="text-[10px] text-gray-400 uppercase font-bold">Health Rate</div>
                        <div className="text-lg font-bold">94.2%</div>
                    </div>
                </div>
                <div className="flex items-center gap-3 pr-4 border-r border-white/10">
                    <div className="p-2 bg-red-500/20 rounded-lg">
                        <AlertTriangle size={20} className="text-red-500" />
                    </div>
                    <div>
                        <div className="text-[10px] text-gray-400 uppercase font-bold">Critical Faults</div>
                        <div className="text-lg font-bold">{stats[PANEL_STATUS.HOTSPOT]}</div>
                    </div>
                </div>
                <div className="flex items-center gap-3">
                    <div className="p-2 bg-blue-500/20 rounded-lg">
                        <Activity size={20} className="text-blue-500" />
                    </div>
                    <div>
                        <div className="text-[10px] text-gray-400 uppercase font-bold">Active Drones</div>
                        <div className="text-lg font-bold">02</div>
                    </div>
                </div>
            </div>

            {/* Legend Overlay */}
            <div className="map-overlay legend-panel">
                <div className="text-[10px] text-gray-400 uppercase font-bold mb-2">Map Legend</div>
                <div className="flex flex-wrap gap-x-4 gap-y-2">
                    {Object.entries(STATUS_COLORS).map(([status, color]) => (
                        <div key={status} className="flex items-center gap-2">
                            <div className="w-3 h-3 rounded-sm" style={{ background: color }}></div>
                            <span className="text-[11px] text-gray-300">{status}</span>
                        </div>
                    ))}
                </div>
            </div>

            {/* Main Map */}
            <MapContainer 
                center={[10.8231, 106.6297]} 
                zoom={19} 
                scrollWheelZoom={true}
                zoomControl={false} // Custom zoom position or style
            >
                <TileLayer
                    url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                    attribution='&copy; OpenStreetMap contributors'
                />
                <MapController />

                {/* Panel Layers */}
                {filteredPanels.map(panel => (
                    viewMode === 'polygon' ? (
                        <Polygon
                            key={panel.id}
                            positions={panel.polygon}
                            pathOptions={{
                                color: STATUS_COLORS[panel.status],
                                fillColor: STATUS_COLORS[panel.status],
                                fillOpacity: showHeatmap ? (panel.tempMax / 100) : 0.6,
                                weight: 1
                            }}
                            eventHandlers={{
                                click: () => handlePanelClick(panel),
                            }}
                        >
                            <Tooltip permanent={false} direction="top">
                                <div className="text-xs font-bold">{panel.id}</div>
                                <div className="text-[10px]">{panel.status} | {panel.tempMax}°C</div>
                            </Tooltip>
                        </Polygon>
                    ) : (
                        <CircleMarker
                            key={panel.id}
                            center={[panel.lat, panel.lng]}
                            radius={6}
                            pathOptions={{
                                color: 'white',
                                fillColor: STATUS_COLORS[panel.status],
                                fillOpacity: 0.9,
                                weight: 1
                            }}
                            eventHandlers={{
                                click: () => handlePanelClick(panel),
                            }}
                        >
                            <Tooltip>
                                <div className="text-xs font-bold">{panel.id}</div>
                            </Tooltip>
                        </CircleMarker>
                    )
                ))}

                {/* Heatmap Layer Effect (Simulated with CircleMarkers) */}
                {showHeatmap && filteredPanels.filter(p => p.status === PANEL_STATUS.HOTSPOT).map(p => (
                    <CircleMarker
                        key={`heat-${p.id}`}
                        center={[p.lat, p.lng]}
                        radius={20}
                        pathOptions={{
                            color: 'transparent',
                            fillColor: '#ef4444',
                            fillOpacity: 0.3,
                        }}
                        className="hotspot-marker"
                    />
                ))}

                {/* Custom Popup for selected panel */}
                {selectedPanel && (
                    <Popup 
                        position={[selectedPanel.lat, selectedPanel.lng]}
                        onClose={() => setSelectedPanel(null)}
                    >
                        <div className="popup-header">
                            <div className="flex items-center gap-2">
                                <div className="w-3 h-3 rounded-full" style={{ background: STATUS_COLORS[selectedPanel.status] }}></div>
                                <span className="font-bold">{selectedPanel.id}</span>
                            </div>
                            <span className="status-badge" style={{ background: `${STATUS_COLORS[selectedPanel.status]}33`, color: STATUS_COLORS[selectedPanel.status] }}>
                                {selectedPanel.status}
                            </span>
                        </div>
                        <div className="popup-body">
                            <div className="grid grid-cols-2 gap-4">
                                <div>
                                    <div className="text-[10px] text-gray-400 uppercase">String / Row</div>
                                    <div className="text-sm font-medium">{selectedPanel.stringId} / Row {selectedPanel.row}</div>
                                </div>
                                <div>
                                    <div className="text-[10px] text-gray-400 uppercase">Max Temp</div>
                                    <div className="text-sm font-medium text-orange-400">{selectedPanel.tempMax}°C</div>
                                </div>
                                <div>
                                    <div className="text-[10px] text-gray-400 uppercase">AI Confidence</div>
                                    <div className="text-sm font-medium text-blue-400">{selectedPanel.confidence}%</div>
                                </div>
                                <div>
                                    <div className="text-[10px] text-gray-400 uppercase">Last Update</div>
                                    <div className="text-xs text-gray-400">{new Date(selectedPanel.lastUpdated).toLocaleTimeString()}</div>
                                </div>
                            </div>
                            {selectedPanel.status !== PANEL_STATUS.HEALTHY && (
                                <div className="mt-4 p-2 bg-red-500/10 border border-red-500/20 rounded text-[11px] text-red-400 flex items-start gap-2">
                                    <AlertTriangle size={14} className="mt-0.5 shrink-0" />
                                    <span>Anomalous thermal signature detected. High probability of cell bypass failure.</span>
                                </div>
                            )}
                        </div>
                        <div className="popup-footer">
                            <button className="w-full bg-blue-600 hover:bg-blue-700 text-white py-2 rounded-lg text-sm font-semibold transition flex items-center justify-center gap-2">
                                View Detail
                                <ChevronRight size={16} />
                            </button>
                        </div>
                    </Popup>
                )}
            </MapContainer>

            {/* Bottom Left Toolbar */}
            <div className="absolute bottom-8 left-8 z-[1000] flex gap-2">
                <button className="p-3 bg-white/10 backdrop-blur-md border border-white/20 rounded-xl text-white hover:bg-white/20 transition shadow-lg">
                    <Maximize size={20} />
                </button>
                <button className="p-3 bg-white/10 backdrop-blur-md border border-white/20 rounded-xl text-white hover:bg-white/20 transition shadow-lg">
                    <Info size={20} />
                </button>
            </div>
        </div>
    );
}
