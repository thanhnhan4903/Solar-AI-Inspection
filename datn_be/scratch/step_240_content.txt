Created At: 2026-06-05T13:41:52Z
Completed At: 2026-06-05T13:41:53Z
File Path: `file:///c:/Solar_Inspection_Project/datn_be/scratch/edge_grid_full_image_DJI_0029.py`
Total Lines: 931
Total Bytes: 40708
Showing lines 570 to 800
The following code has been modified to include a line number before every line, in the format: <line_number>: <original_line>. Please note that any changes targeting the original code should remove the line number, colon, and leading space.
570:         
571:         # 2. Build profile projection along normal for each raw gap and find dark-valley center
572:         candidate_y_snaps = []
573:         for y, x_start, x_end in raw_horiz_gaps:
574:             xs = np.linspace(x_start, x_end, 30)
575:             profile_sums = np.zeros(2 * 8 + 1)
576:             profile_counts = np.zeros(2 * 8 + 1)
577:             for dn in range(-8, 9):
578:                 for x in xs:
579:                     x_sample = x - dn * math.sin(best_theta)
580:                     y_sample = m_perp * (x - x_mid) + y + dn * math.cos(best_theta)
581:                     x_idx = int(round(x_sample))
582:                     y_idx = int(round(y_sample))
583:                     if 0 <= x_idx < gray.shape[1] and 0 <= y_idx < gray.shape[0]:
584:                         profile_sums[dn + 8] += gray[y_idx, x_idx]
585:                         profile_counts[dn + 8] += 1
586:             profile = profile_sums / np.maximum(profile_counts, 1)
587:             smoothed = np.convolve(profile, np.ones(3)/3.0, mode="same")
588:             
589:             # Find best valley
590:             best_valley_idx = -1
591:             best_depth = -1.0
592:             for v in range(1, len(smoothed) - 1):
593:                 if smoothed[v] < smoothed[v-1] and smoothed[v] < smoothed[v+1]:
594:                     left_peak = np.max(smoothed[:v])
595:                     right_peak = np.max(smoothed[v+1:])
596:                     depth = min(left_peak, right_peak) - smoothed[v]
597:                     if depth > best_depth:
59
<truncated 8311 bytes>
not is_gate_passed:
766:                     rejected_count += 1
767:                     print(f"DEBUG {block_name} line {c['grid_idx']}: valley_score={dark_valley_score:.2f}, support={support_ratio:.2f}, offset={mean_abs_offset:.2f}, pitch_dev={pitch_deviation:.2f}")
768:                     
769:             snapped_horiz_gaps.append({
770:                 "y_snap": y_snap,
771:                 "is_good": is_gate_passed,
772:                 "support_ratio": support_ratio,
773:                 "mean_offset": mean_abs_offset,
774:                 "pitch_deviation": pitch_deviation,
775:                 "support_pts": support_pts,
776:                 "x_start": x_start,
777:                 "x_end": x_end,
778:                 "dark_valley_score": dark_valley_score
779:             })
780:             
781:         # --- DRAW VISUALS ---
782:         for g in snapped_horiz_gaps:
783:             y_snap_anchor = g["y_snap"]
784:             x_start = g["x_start"]
785:             x_end = g["x_end"]
786:             is_good = g["is_good"]
787:             
788:             y_start_draw = int(round(get_oriented_line_y_at_x(m_perp, x_mid, y_snap_anchor, x_start)))
789:             y_end_draw = int(round(get_oriented_line_y_at_x(m_perp, x_mid, y_snap_anchor, x_end)))
790:             
791:             # Draw on support image
792:             color = (0, 255, 0) if is_good else (0, 0, 255)
793:             cv2.line(img_support, (x_start, y_start_draw), (x_end, y_end_draw), color, 2)
794:             for pt in g["support_pts"]:
795:                 cv2.circle(img_support, (pt[0], int(round(pt[1]))), 1, (255, 0, 255), -1)
796:                 
797:             # Draw on snap image
798:             cv2.line(img_snap, (x_start, y_start_draw), (x_end, y_end_draw), (0, 255, 255), 2)
799:             
800:         # Draw vertical boundaries
The above content does NOT show the entire file contents. If you need to view any lines of the file which were not shown to complete your task, call this tool again to view those lines.
