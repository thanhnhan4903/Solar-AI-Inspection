import json
import numpy as np

TRACE_PATH = "data/results/debug_logs/DJI_0029_R_final_polygon_trace.jsonl"
panels = []
with open(TRACE_PATH, "r", encoding="utf-8") as f:
    for line in f:
        panels.append(json.loads(line))

input_panels = [p for p in panels if p["final_polygon_source"] in {"string_lattice_middle", "middle_locked_endpoint"}]
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

clusters.sort(key=len, reverse=True)

targets = [25, 40, 46, 47, 52, 62, 65, 67, 68, 77, 78, 86]
for t in targets:
    p = next(x for x in panels if x["raw_idx"] == t)
    cx, cy = p["center"]
    dists = []
    for c_idx, cl in enumerate(clusters):
        min_d = min(np.linalg.norm(np.array([cx, cy]) - input_centers[node]) for node in cl)
        dists.append((c_idx, min_d))
    dists.sort(key=lambda x: x[1])
    print(f"raw_idx={t}: nearest cluster={dists[0][0]} with distance={dists[0][1]:.2f} px")
