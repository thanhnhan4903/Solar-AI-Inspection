/**
 * Generates mock data for solar panels
 * Each panel has GPS coordinates, status, and metadata
 */

export const PANEL_STATUS = {
    HEALTHY: 'Healthy',
    HOTSPOT: 'Hotspot',
    SOILING: 'Soiling',
    CRACK: 'Crack',
    OFFLINE: 'Offline'
};

export const STATUS_COLORS = {
    [PANEL_STATUS.HEALTHY]: '#10b981', // Green
    [PANEL_STATUS.HOTSPOT]: '#ef4444', // Red
    [PANEL_STATUS.SOILING]: '#f59e0b', // Yellow
    [PANEL_STATUS.CRACK]: '#3b82f6',   // Blue
    [PANEL_STATUS.OFFLINE]: '#6b7280'  // Gray
};

export const generateMockPanels = (count = 120) => {
    const panels = [];
    // Central point (example: a solar farm in Vietnam or elsewhere)
    const baseLat = 10.8231;
    const baseLng = 106.6297;
    
    const rows = 12;
    const cols = Math.ceil(count / rows);
    
    const latSpacing = 0.00015;
    const lngSpacing = 0.00025;

    for (let i = 0; i < count; i++) {
        const row = Math.floor(i / cols);
        const col = i % cols;
        
        const statusValues = Object.values(PANEL_STATUS);
        // Random status with weighting towards Healthy
        const rand = Math.random();
        let status = PANEL_STATUS.HEALTHY;
        if (rand > 0.9) status = PANEL_STATUS.HOTSPOT;
        else if (rand > 0.8) status = PANEL_STATUS.SOILING;
        else if (rand > 0.75) status = PANEL_STATUS.CRACK;
        else if (rand > 0.7) status = PANEL_STATUS.OFFLINE;

        panels.push({
            id: `PNL-${1000 + i}`,
            stringId: `STR-${Math.floor(i / 10) + 1}`,
            row: row + 1,
            col: col + 1,
            lat: baseLat + (row * latSpacing),
            lng: baseLng + (col * lngSpacing),
            status,
            tempMax: (25 + Math.random() * 50).toFixed(1),
            confidence: (85 + Math.random() * 14).toFixed(1),
            lastUpdated: new Date().toISOString(),
            // Polygon coordinates for each panel (approx 2x1 meters in lat/lng)
            polygon: [
                [baseLat + (row * latSpacing) - 0.00005, baseLng + (col * lngSpacing) - 0.0001],
                [baseLat + (row * latSpacing) + 0.00005, baseLng + (col * lngSpacing) - 0.0001],
                [baseLat + (row * latSpacing) + 0.00005, baseLng + (col * lngSpacing) + 0.0001],
                [baseLat + (row * latSpacing) - 0.00005, baseLng + (col * lngSpacing) + 0.0001],
            ]
        });
    }
    return panels;
};
