# Integration Plan - Edge Grid Segmented Proposal Refinement

This document outlines the design and integration gate logic for incorporating the `edge_grid_segmented4` algorithm from [edge_grid_proposal_DJI_0029.py](file:///c:/Solar_Inspection_Project/datn_be/scratch/edge_grid_proposal_DJI_0029.py) into the main pipeline in [panel_processor.py](file:///c:/Solar_Inspection_Project/datn_be/app/services/panel_processor.py).

---

## 1. Helper Activation Conditions (When to Call)

The helper must not run globally on all panels. It is activated selectively under the following constraints:
- **String/Column Targeting**: Only trigger the edge grid proposal for a group of panels in the same column/string that currently have poor YOLO bounding boxes or distorted string-lattice predictions (e.g. they are failed/rejected candidates or have extreme projection drift).
- **Minimum Group Size**: Require at least **4 target panels** in the same column/string that are contiguous or near-contiguous (e.g., sharing consecutive row indices or having minor gaps).
- **Localized ROI**: Construct a bounding box around the target panel group, add a margin (default `roi_margin=45`), and run the edge detector and line probes strictly inside this localized sub-region. Do not process the entire image.

---

## 2. Proposal Acceptance Criteria (When to Accept)

Each proposed panel polygon within the grid must satisfy all of the following validation gates to replace the existing polygon:
- **Geometry**: The proposal must be a clean 4-point convex quadrilateral.
- **Area Ratio**: The area of the proposed polygon relative to the current YOLO bbox/polygon must be within a reasonable scale:
  $$\text{area\_ratio} \in [0.75, 1.35]$$
- **Center Shift**: The spatial drift between the centroid of the proposed polygon and the original YOLO centroid must be constrained:
  $$\text{center\_shift} \le 25\text{ px}$$
- **Overlap Control**: The proposed polygon must not overlap significantly with any neighboring panels:
  $$\text{overlap\_ratio} \le 0.05$$
- **Lateral Intrusions**: The left and right vertical boundaries must not cut deeply into the bounding areas of neighboring columns (checking boundary distance margins).
- **Horizontal Regularity**: The detected horizontal gap consensus lines must be sufficient in number and exhibit approximately uniform spacing (no massive skew or mismatch in panel heights).

---

## 3. Proposal Rejection Criteria (When to Reject)

A proposal for a panel group or individual panel must be rejected if any of the following anomalies are detected:
- **Missing Horizontal Gaps**: Fewer than the required number of horizontal gap lines are detected to bound the top and bottom of all panels in the group.
- **Vertical Boundary Fallback**: The segment fitting fails or falls back to the baseline parameters excessively (e.g., more than 50% of the segments lose edge support), indicating poor edge gradients.
- **Extreme Scale/Position Drift**: The proposal is either too small, too large ($\text{area\_ratio}$ out of bounds), or is shifted too far from the anchor YOLO box ($\text{center\_shift} > 25\text{ px}$).
- **High Overlap**: The proposal overlaps heavily with an already accepted adjacent panel.
- **Visual Intrusion Risk**: The proposed coordinates visually clip adjacent structures or are misaligned with the visual panel boundaries.

---

## 4. Rejection Handling

When a proposal fails the validation gate:
- **Action**: Retain the current panel polygon shape (the original YOLO bounding box or the initial string-lattice prediction).
- **Tracing**: Write a rejection trace status to the debug log:
  ```python
  "final_polygon_reason": "edge_grid_segmented4_rejected_<reason>"
  ```
  *(e.g., `edge_grid_segmented4_rejected_large_center_shift`, `edge_grid_segmented4_rejected_no_horizontal_gaps`)*

---

## 5. Acceptance Handling

When a proposal successfully passes all validation gates:
- **Action**: Overwrite the panel's polygon coordinates with the globalized proposed coordinates.
- **Metadata Fields**: Update the panel's refinement tracking properties:
  - `final_polygon_source` = `"edge_grid_segmented4"`
  - `final_polygon_stage` = `"edge_grid_refinement"`
  - `final_polygon_reason` = `"segmented_boundary_projection"`

---

## 6. Trace & Debug Output Schema

The output trace file (e.g. `final_polygon_trace.jsonl`) should be appended with the following diagnostic properties to aid in quantitative audits:

| Field Name | Type | Description |
| :--- | :--- | :--- |
| `edge_grid_attempted` | `bool` | Indicates if this panel was targeted for the edge grid optimization. |
| `edge_grid_accept` | `bool` | `True` if the proposed polygon replaced the old shape. |
| `edge_grid_reject_reason` | `str` | `None` if accepted; otherwise the specific rejection reason. |
| `edge_grid_area_ratio` | `float` | Proposed area divided by the original panel area. |
| `edge_grid_center_shift` | `float` | Euclidean distance between centroids (in pixels). |
| `edge_grid_overlap` | `float` | Maximum intersection-over-area with any neighboring panel. |
| `edge_grid_vertical_mode` | `str` | Mode used (always `"segmented"`). |
| `edge_grid_n_segments` | `int` | Number of segments configured (always `4`). |
