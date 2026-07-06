export function normalizeDefect(defect = {}) {
  const className = defect.class_name || defect.type || "";
  const polygon = defect.polygon || [];
  const displayPolygon = defect.display_polygon || polygon;
  const analysisPolygon = defect.analysis_polygon || polygon;
  const areaRatioPercent = defect.area_ratio_percent !== undefined ? defect.area_ratio_percent : (defect.loss !== undefined ? defect.loss : null);
  
  return {
    ...defect,
    class_name: className,
    type: className,
    confidence: defect.confidence || 0,
    polygon,
    display_polygon: displayPolygon,
    analysis_polygon: analysisPolygon,
    area_ratio_percent: areaRatioPercent,
    loss: areaRatioPercent,
    location_in_panel: defect.location_in_panel || "",
    severity: defect.severity || "healthy",
    recommendation: defect.recommendation || "Kiểm tra",
    thermal_validation_status: defect.thermal_validation_status || "not_run",
    relative_thermal_delta: defect.relative_thermal_delta || null,
    suggested_review_status: defect.suggested_review_status || null,
  };
}

export function normalizePanel(panel = {}) {
  const rawDefects = panel.defects || [];
  const normalizedDefects = Array.isArray(rawDefects) ? rawDefects.map(normalizeDefect) : [];
  
  const reviewStatus = panel.review_status || "unreviewed";
  const reviewLabel = panel.review_label || {
    "unreviewed": "Chưa duyệt",
    "confirmed_defect": "Đã xác nhận",
    "needs_review": "Cần kiểm tra lại",
    "false_positive": "Bỏ qua"
  }[reviewStatus] || "Chưa duyệt";
  
  const includeInReport = reviewStatus === "false_positive" ? false : (panel.include_in_report !== undefined ? panel.include_in_report : true);
  
  const status = reviewStatus === "false_positive" ? "healthy" : (normalizedDefects.length > 0 ? "faulty" : "healthy");
  
  const polygon = panel.polygon || [];
  const outerPolygon = panel.outer_polygon || polygon;
  const innerPolygon = panel.inner_polygon || [];
  const calcPolygon = panel.calc_polygon || innerPolygon;

  return {
    ...panel,
    local_id: panel.local_id || "",
    row: panel.row !== undefined ? panel.row : 0,
    col: panel.col !== undefined ? panel.col : 0,
    center: panel.center || [0, 0],
    polygon,
    outer_polygon: outerPolygon,
    inner_polygon: innerPolygon,
    calc_polygon: calcPolygon,
    defects: normalizedDefects,
    status,
    worst_severity: panel.worst_severity || "healthy",
    recommendation: panel.recommendation || "Không cần xử lý",
    total_panel_loss: panel.total_panel_loss || 0.0,
    total_defect_area_ratio_percent: panel.total_defect_area_ratio_percent || 0.0,
    confidence: panel.confidence || 0.0,
    main_defect_class: panel.main_defect_class || null,
    review_status: reviewStatus,
    review_label: reviewLabel,
    review_note: panel.review_note || "",
    include_in_report: includeInReport,
    maintenance_priority: panel.maintenance_priority || "medium",
    reviewer_name: panel.reviewer_name || "",
    reviewed_at: panel.reviewed_at || "",
  };
}

export function normalizeBatchPayload(payload = {}) {
  const rawData = payload.data || [];
  const normalizedData = rawData.map(img => {
    const rawPanels = img.panels || [];
    const normalizedPanels = Array.isArray(rawPanels) ? rawPanels.map(normalizePanel) : [];
    return {
      ...img,
      total_panels: normalizedPanels.length,
      panels: normalizedPanels
    };
  });
  
  let summary = payload.summary;
  if (!summary) {
    summary = computeInspectionSummary(normalizedData);
  }
  
  return {
    ...payload,
    summary,
    data: normalizedData
  };
}

export function computeInspectionSummary(images = []) {
  let totalImages = images.length;
  let totalPanels = 0;
  let normalPanels = 0;
  let faultyPanels = 0;
  let totalDefects = 0;
  let includedDefects = 0;
  let excludedDefects = 0;
  
  let unreviewed = 0;
  let confirmedDefect = 0;
  let needsReview = 0;
  let falsePositive = 0;
  
  let totalPowerLossW = 0;

  images.forEach(img => {
    const panels = img.panels || [];
    totalPanels += panels.length;
    panels.forEach(p => {
      if (p.status === "healthy") {
        normalPanels++;
      } else {
        faultyPanels++;
      }
      
      const nDefects = p.defects.length;
      totalDefects += nDefects;
      
      if (p.include_in_report) {
        includedDefects += nDefects;
        totalPowerLossW += p.total_panel_loss || 0;
      } else {
        excludedDefects += nDefects;
      }
      
      if (nDefects > 0) {
        const revStatus = p.review_status;
        if (revStatus === "confirmed_defect") {
          confirmedDefect++;
        } else if (revStatus === "needs_review") {
          needsReview++;
        } else if (revStatus === "false_positive") {
          falsePositive++;
        } else {
          unreviewed++;
        }
      }
    });
  });
  
  const normalPanelRatioPercent = totalPanels > 0 ? (normalPanels / totalPanels * 100) : 100.0;
  
  return {
    totalImages,
    totalPanels,
    normalPanels,
    faultyPanels,
    totalDefects,
    includedDefects,
    excludedDefects,
    unreviewed,
    confirmedDefect,
    needsReview,
    falsePositive,
    totalPowerLossW: Number(totalPowerLossW.toFixed(2)),
    normalPanelRatioPercent: Number(normalPanelRatioPercent.toFixed(2)),

    total_images: totalImages,
    total_panels: totalPanels,
    normal_panels: normalPanels,
    faulty_panels: faultyPanels,
    total_defects: totalDefects,
    included_defects: includedDefects,
    excluded_defects: excludedDefects,
    confirmed_defect: confirmedDefect,
    needs_review: needsReview,
    false_positive: falsePositive,
    total_power_loss_w: Number(totalPowerLossW.toFixed(2)),
    normal_panel_ratio_percent: Number(normalPanelRatioPercent.toFixed(2))
  };
}
