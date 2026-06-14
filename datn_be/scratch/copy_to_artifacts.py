import shutil
from pathlib import Path

src_dir = Path("data/results/debug")
dest_dir = Path("C:/Users/ThanhNhan/.gemini/antigravity-ide/brain/cbb9b909-1dfd-49be-b36a-582e8aefc671")

stems = ["DJI_0029_R", "DJI_0843_R", "DJI_0845_R", "DJI_0087_R"]
suffixes = ["blocks", "support_points", "line_snap", "panels_from_snap"]

for stem in stems:
    for suffix in suffixes:
        src = src_dir / f"debug_{stem}_{suffix}.JPG"
        if src.exists():
            dest = dest_dir / f"debug_{stem}_{suffix}.JPG"
            shutil.copy2(src, dest)
            print(f"Copied {src.name} to artifacts")
        else:
            print(f"Warning: {src} not found")
