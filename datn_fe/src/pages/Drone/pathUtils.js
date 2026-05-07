import * as turf from '@turf/turf';

/**
 * Generates a zigzag flight path inside a polygon
 * @param {Object} polygon - GeoJSON Polygon feature
 * @param {number} spacing - Distance between lines in meters
 * @param {number} angle - Angle of the lines in degrees
 */
export const generateZigzagPath = (polygon, spacing = 20, angle = 0) => {
    if (!polygon || !polygon.geometry) return null;

    // Convert spacing to degrees (approximate)
    const spacingDeg = spacing / 111320; 

    // Get bounding box
    const bbox = turf.bbox(polygon);
    const minX = bbox[0];
    const minY = bbox[1];
    const maxX = bbox[2];
    const maxY = bbox[3];

    // Expand bbox slightly to cover entire polygon
    const gridLines = [];
    const rotationRad = (angle * Math.PI) / 180;

    // Create parallel lines
    for (let i = minY - spacingDeg; i < maxY + spacingDeg; i += spacingDeg) {
        const line = turf.lineString([
            [minX - spacingDeg, i],
            [maxX + spacingDeg, i]
        ]);
        
        // Rotate line if angle is provided
        const rotatedLine = turf.transformRotate(line, angle, { pivot: turf.center(polygon) });
        
        // Intersect with polygon
        const intersect = turf.lineIntersect(rotatedLine, polygon);
        
        if (intersect.features.length >= 2) {
            // Sort points by distance to start point
            const sortedPoints = intersect.features.sort((a, b) => {
                return turf.distance(rotatedLine.geometry.coordinates[0], a) - 
                       turf.distance(rotatedLine.geometry.coordinates[0], b);
            });
            
            gridLines.push([
                sortedPoints[0].geometry.coordinates,
                sortedPoints[sortedPoints.length - 1].geometry.coordinates
            ]);
        }
    }

    // Join lines in zigzag
    const path = [];
    for (let i = 0; i < gridLines.length; i++) {
        if (i % 2 === 0) {
            path.push(gridLines[i][0]);
            path.push(gridLines[i][1]);
        } else {
            path.push(gridLines[i][1]);
            path.push(gridLines[i][0]);
        }
    }

    return path;
};

/**
 * Calculates total distance of a path in meters
 */
export const calculateTotalDistance = (coordinates) => {
    if (!coordinates || coordinates.length < 2) return 0;
    const line = turf.lineString(coordinates);
    return turf.length(line, { units: 'meters' });
};
