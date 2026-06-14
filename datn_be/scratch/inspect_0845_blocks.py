import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
import edge_grid_full_image_test as eg

panels = eg.load_panels_from_logs()
blocks = eg.group_into_blocks(panels)
print("Total blocks:", len(blocks))
for idx, b in enumerate(blocks):
    block_name = f"block_{idx + 1}"
    n_cols = len(b["columns"])
    row_count = max(len(col) for col in b["columns"]) if b["columns"] else 0
    local_xs = [p["bbox"][0] for p in b["panels"]] + [p["bbox"][2] for p in b["panels"]]
    min_x, max_x = min(local_xs), max(local_xs)
    print(f"Block name: {block_name}, n_cols: {n_cols}, row_count: {row_count}, x_range: {min_x:.1f} to {max_x:.1f}, panels count: {len(b['panels'])}")
    for col_idx, col in enumerate(b["columns"]):
        xs = [p["bbox"][0] for p in col] + [p["bbox"][2] for p in col]
        ys = [p["bbox"][1] for p in col] + [p["bbox"][3] for p in col]
        print(f"  Col {col_idx}: panels: {len(col)}, cx_avg: {sum(xs)/(2*len(xs)):.1f}, x_range: {min(xs):.1f} to {max(xs):.1f}, y_range: {min(ys):.1f} to {max(ys):.1f}")
