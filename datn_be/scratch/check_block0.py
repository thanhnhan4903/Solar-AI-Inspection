import json
import numpy as np

TRACE_PATH = "data/results/debug_logs/DJI_0029_R_final_polygon_trace.jsonl"
panels = []
with open(TRACE_PATH, "r", encoding="utf-8") as f:
    for line in f:
        panels.append(json.loads(line))

for p in panels:
    bbox = p["bbox"]
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    p["w_bbox"] = w
    p["h_bbox"] = h
    p["area_bbox"] = w * h
    p["aspect_bbox"] = w / h if h > 0 else 1.0

all_ws = [p["w_bbox"] for p in panels]
all_hs = [p["h_bbox"] for p in panels]
all_areas = [p["area_bbox"] for p in panels]
all_aspects = [p["aspect_bbox"] for p in panels]

global_median_width = float(np.median(all_ws))
global_median_height = float(np.median(all_hs))
global_median_area = float(np.median(all_areas))
global_median_aspect = float(np.median(all_aspects))

input_panels = []
for p in panels:
    src = p["final_polygon_source"]
    is_clean = False
    if src in {"string_lattice_middle", "middle_locked_endpoint"}:
        is_clean = True
    elif src == "yolo_original":
        w_in_range = (0.7 * global_median_aspect <= p["aspect_bbox"] <= 1.3 * global_median_aspect)
        area_in_range = (0.65 * global_median_area <= p["area_bbox"] <= 1.35 * global_median_area)
        clamped = p.get("final_polygon_clamped", False)
        near_edge = p.get("near_edge", False)
        if w_in_range and area_in_range and not clamped and not near_edge:
            is_clean = True
    if is_clean:
        input_panels.append(p)

input_centers = np.array([p["center"] for p in input_panels], dtype=np.float32)
n_input = len(input_panels)

adj = {i: [] for i in range(n_input)}
for i in range(n_input):
    for j in range(i + 1, n_input):
        dist = np.linalg.norm(input_centers[i] - input_centers[j])
        if dist <= 120.0:
            adj[i].append(j)
            adj[j].append(i)

visited = set()
clusters = []
for i in range(n_input):
    if i not in visited:
        cluster = []
        queue = [i]
        visited.add(i)
        while queue:
            curr = queue.pop(0)
            cluster.append(curr)
            for neighbor in adj[curr]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        clusters.append(cluster)

print(f"Total clusters: {len(clusters)}")
for idx, c in enumerate(clusters):
    raw_idxs = [input_panels[node]["raw_idx"] for node in c]
    print(f"Cluster {idx} has size {len(c)}: raw_idxs={raw_idxs[:10]}...")
    xs = [input_panels[node]["center"][0] for node in c]
    print(f"  - X range: {min(xs):.1f} to {max(xs):.1f}")
