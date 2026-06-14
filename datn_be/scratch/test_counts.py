import re
import subprocess
import sys
from pathlib import Path

content = Path("scratch/edge_grid_full_image_test.py").read_text(encoding="utf-8")
stem_pattern = re.compile(r'IMAGE_STEM\s*=\s*"([^"]+)"')

for stem in ["DJI_0029_R", "DJI_0843_R", "DJI_0845_R", "DJI_0087_R"]:
    modified = stem_pattern.sub(f'IMAGE_STEM = "{stem}"', content)
    tmp_file = Path("scratch/_tmp_run.py")
    tmp_file.write_text(modified, encoding="utf-8")
    
    res = subprocess.run([sys.executable, str(tmp_file)], capture_output=True, text=True)
    if tmp_file.exists():
        tmp_file.unlink()
        
    counts = re.findall(r"TOTAL DRAWN PANELS FROM SNAP: (\d+)", res.stdout)
    if counts:
        print(f"{stem}: {counts[0]}")
    else:
        print(f"{stem}: FAILED")
        if res.stderr:
            print(res.stderr)
