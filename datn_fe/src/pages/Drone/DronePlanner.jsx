import React, { useEffect, useRef, useState } from 'react';
import mapboxgl from 'mapbox-gl';
import 'mapbox-gl/dist/mapbox-gl.css';
import * as turf from '@turf/turf';
import { 
    Battery, 
    Wifi, 
    Compass, 
    Navigation, 
    Wind, 
    Settings, 
    Play, 
    Square, 
    Trash2, 
    Layers, 
    Maximize2, 
    Target,
    Camera,
    Thermometer
} from 'lucide-react';
import { generateZigzagPath, calculateTotalDistance } from './pathUtils';
import './DroneStyles.css';

// TODO: Replace with your actual Mapbox Token
mapboxgl.accessToken = 'pk.eyJ1IjoiYW50aWdyYXZpdHktYWkiLCJhIjoiY202eDlxNmt4MDNqZzJqcHlsbmx1dzJneCJ9.dummy-token';

export default function DronePlanner() {
    const mapContainerRef = useRef(null);
    const mapRef = useRef(null);
    const [points, setPoints] = useState([]);
    const [path, setPath] = useState([]);
    const [altitude, setAltitude] = useState(50);
    const [speed, setSpeed] = useState(5);
    const [overlap, setOverlap] = useState(80);
    const [is3D, setIs3D] = useState(false);
    const [isFlying, setIsFlying] = useState(false);
    const [dronePos, setDronePos] = useState(null);

    // Initialize Map
    useEffect(() => {
        const map = new mapboxgl.Map({
            container: mapContainerRef.current,
            style: 'mapbox://styles/mapbox/satellite-v9',
            center: [106.6297, 10.8231], // Default center
            zoom: 17,
            pitch: 0,
            bearing: 0,
            antialias: true
        });

        map.addControl(new mapboxgl.NavigationControl(), 'top-left');

        map.on('load', () => {
            mapRef.current = map;

            // Add sources for boundary and path
            map.addSource('boundary', {
                type: 'geojson',
                data: { type: 'FeatureCollection', features: [] }
            });

            map.addLayer({
                id: 'boundary-line',
                type: 'line',
                source: 'boundary',
                paint: { 'line-color': '#00ffff', 'line-width': 3 }
            });

            map.addLayer({
                id: 'boundary-fill',
                type: 'fill',
                source: 'boundary',
                paint: { 'fill-color': '#00ffff', 'fill-opacity': 0.1 }
            });

            map.addSource('path', {
                type: 'geojson',
                data: { type: 'FeatureCollection', features: [] }
            });

            map.addLayer({
                id: 'path-line',
                type: 'line',
                source: 'path',
                paint: { 'line-color': '#00ff00', 'line-width': 2, 'line-dasharray': [2, 1] }
            });

            // Waypoints layer
            map.addSource('waypoints', {
                type: 'geojson',
                data: { type: 'FeatureCollection', features: [] }
            });

            map.addLayer({
                id: 'waypoints-points',
                type: 'circle',
                source: 'waypoints',
                paint: {
                    'circle-radius': 6,
                    'circle-color': '#fff',
                    'circle-stroke-width': 2,
                    'circle-stroke-color': '#00ff00'
                }
            });
        });

        map.on('click', (e) => {
            const coords = [e.lngLat.lng, e.lngLat.lat];
            setPoints(prev => [...prev, coords]);
        });

        return () => map.remove();
    }, []);

    // Update map when points or settings change
    useEffect(() => {
        if (!mapRef.current || points.length < 3) return;

        const polygon = turf.polygon([[...points, points[0]]]);
        mapRef.current.getSource('boundary').setData(polygon);

        // Generate flight path (zigzag)
        const spacing = 100 - overlap; // Simple mapping of overlap to spacing
        const zigzagPath = generateZigzagPath(polygon, spacing);
        
        if (zigzagPath) {
            setPath(zigzagPath);
            mapRef.current.getSource('path').setData(turf.lineString(zigzagPath));
            
            // Set waypoints
            mapRef.current.getSource('waypoints').setData({
                type: 'FeatureCollection',
                features: zigzagPath.map((p, i) => turf.point(p, { index: i + 1 }))
            });
        }
    }, [points, overlap]);

    const handleReset = () => {
        setPoints([]);
        setPath([]);
        if (mapRef.current) {
            mapRef.current.getSource('boundary').setData({ type: 'FeatureCollection', features: [] });
            mapRef.current.getSource('path').setData({ type: 'FeatureCollection', features: [] });
            mapRef.current.getSource('waypoints').setData({ type: 'FeatureCollection', features: [] });
        }
    };

    const toggle3D = () => {
        setIs3D(!is3D);
        mapRef.current.easeTo({
            pitch: is3D ? 0 : 60,
            bearing: is3D ? 0 : -30,
            duration: 1000
        });
    };

    const startMission = () => {
        if (path.length < 2) return;
        setIsFlying(true);
        // Simple simulation would go here
    };

    return (
        <div className="drone-planner-container">
            <div id="map-container" ref={mapContainerRef} />

            {/* Top Status Bar */}
            <div className="drone-overlay top-controls glass-panel px-6 py-2">
                <div className="flex items-center gap-6">
                    <div className="flex items-center gap-2">
                        <Battery className="text-green-400" size={18} />
                        <span className="text-sm font-bold">85%</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <Wifi className="text-blue-400" size={18} />
                        <span className="text-sm font-bold">HD</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <Navigation className="text-blue-400" size={18} />
                        <span className="text-sm font-bold">GPS: 18</span>
                    </div>
                    <div className="h-6 w-px bg-white/20 mx-2" />
                    <button onClick={toggle3D} className={`control-btn ${is3D ? 'text-blue-400' : ''}`}>
                        <Maximize2 size={20} />
                    </button>
                    <button onClick={() => mapRef.current.flyTo({ zoom: 17 })} className="control-btn">
                        <Target size={20} />
                    </button>
                </div>
            </div>

            {/* Right Mission Sidebar */}
            <div className="drone-overlay drone-sidebar glass-panel">
                <div className="flex items-center gap-2 mb-6">
                    <Settings className="text-blue-400" size={24} />
                    <h2 className="text-xl font-bold uppercase tracking-wider">Mission Plan</h2>
                </div>

                <div className="space-y-6">
                    <div>
                        <div className="flex justify-between text-sm mb-2">
                            <span className="text-gray-400">Flight Altitude</span>
                            <span className="text-blue-400 font-bold">{altitude}m</span>
                        </div>
                        <input type="range" min="10" max="120" value={altitude} onChange={(e) => setAltitude(e.target.value)} />
                    </div>

                    <div>
                        <div className="flex justify-between text-sm mb-2">
                            <span className="text-gray-400">Flight Speed</span>
                            <span className="text-blue-400 font-bold">{speed}m/s</span>
                        </div>
                        <input type="range" min="1" max="15" value={speed} onChange={(e) => setSpeed(e.target.value)} />
                    </div>

                    <div>
                        <div className="flex justify-between text-sm mb-2">
                            <span className="text-gray-400">Side Overlap</span>
                            <span className="text-blue-400 font-bold">{overlap}%</span>
                        </div>
                        <input type="range" min="50" max="90" value={overlap} onChange={(e) => setOverlap(e.target.value)} />
                    </div>

                    <div className="grid grid-cols-2 gap-4 pt-4">
                        <button className="flex flex-col items-center gap-2 p-3 bg-white/5 border border-white/10 rounded-xl hover:bg-white/10 transition">
                            <Camera size={20} className="text-blue-400" />
                            <span className="text-[10px] uppercase font-bold">RGB Mode</span>
                        </button>
                        <button className="flex flex-col items-center gap-2 p-3 bg-white/5 border border-white/10 rounded-xl hover:bg-white/10 transition">
                            <Thermometer size={20} className="text-orange-400" />
                            <span className="text-[10px] uppercase font-bold">Thermal</span>
                        </button>
                    </div>

                    <div className="pt-6 border-t border-white/10 space-y-3">
                        <div className="flex justify-between text-xs">
                            <span className="text-gray-400">Total Distance</span>
                            <span className="font-mono">{path.length > 0 ? calculateTotalDistance(path).toFixed(0) : 0} m</span>
                        </div>
                        <div className="flex justify-between text-xs">
                            <span className="text-gray-400">Est. Flight Time</span>
                            <span className="font-mono">{path.length > 0 ? (calculateTotalDistance(path) / speed / 60).toFixed(1) : 0} min</span>
                        </div>
                        <div className="flex justify-between text-xs">
                            <span className="text-gray-400">Photo Count</span>
                            <span className="font-mono">{path.length * 2}</span>
                        </div>
                    </div>

                    <div className="flex gap-3 pt-6">
                        <button onClick={handleReset} className="flex-1 py-3 px-4 border border-red-500/30 text-red-500 rounded-xl hover:bg-red-500/10 transition flex items-center justify-center gap-2 font-bold text-sm">
                            <Trash2 size={18} />
                            RESET
                        </button>
                        <button 
                            onClick={startMission}
                            disabled={path.length === 0}
                            className={`flex-[2] py-3 px-4 rounded-xl transition flex items-center justify-center gap-2 font-bold text-sm ${path.length > 0 ? 'start-btn' : 'bg-gray-700 text-gray-500 cursor-not-allowed'}`}
                        >
                            <Play size={18} fill="currentColor" />
                            START MISSION
                        </button>
                    </div>
                </div>
            </div>

            {/* Bottom HUD */}
            <div className="drone-overlay drone-hud">
                <div className="radar-circle">
                    <div className="radar-sweep" />
                    <div className="absolute inset-0 flex items-center justify-center">
                        <Navigation className="text-blue-400" size={24} style={{ transform: 'rotate(-45deg)' }} />
                    </div>
                </div>
                
                <div className="hud-item glass-panel">
                    <div className="hud-label">Speed</div>
                    <div className="hud-value">{isFlying ? speed : '0.0'}<span className="text-xs ml-1 text-gray-400">m/s</span></div>
                </div>

                <div className="hud-item glass-panel">
                    <div className="hud-label">Height</div>
                    <div className="hud-value">{isFlying ? altitude : '0.0'}<span className="text-xs ml-1 text-gray-400">m</span></div>
                </div>

                <div className="hud-item glass-panel">
                    <div className="hud-label">Coordinates</div>
                    <div className="text-[12px] font-mono">
                        10.8231° N<br />
                        106.6297° E
                    </div>
                </div>
            </div>

            {/* Instructions Overlay */}
            {points.length === 0 && (
                <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                    <div className="bg-black/60 backdrop-blur-md px-8 py-4 rounded-2xl border border-white/20 text-center animate-pulse">
                        <p className="text-white font-bold tracking-widest text-sm mb-1">MISSION INITIALIZATION</p>
                        <p className="text-blue-400 text-xs">CLICK ON MAP TO DRAW SOLAR FARM BOUNDARY</p>
                    </div>
                </div>
            )}
        </div>
    );
}
