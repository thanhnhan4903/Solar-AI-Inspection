import subprocess
import sys
import re
import shutil
from pathlib import Path

# Paths
base_file = Path("scratch/edge_grid_full_image_test.py")
backup_dir = Path("data/results/baseline_backup")
backup_dir.mkdir(parents=True, exist_ok=True)

stems = ["DJI_0029_R", "DJI_0843_R", "DJI_0845_R", "DJI_0087_R"]

content = base_file.read_text(encoding="utf-8")

# Find how IMAGE_STEM is defined
stem_pattern = re.compile(r'IMAGE_STEM\s*=\s*"([^"]+)"')
match = stem_pattern.search(content)
if not match:
    print("Could not find IMAGE_STEM pattern in test file.")
    sys.exit(1)
original_stem = match.group(1)
print(f"Original IMAGE_STEM in file: {original_stem}")

for stem in stems:
    print(f"\n==========================================")
    print(f"RUNNING STEM: {stem}")
    print(f"==========================================")
    modified_content = stem_pattern.sub(f'IMAGE_STEM = "{stem}"', content)
    
    tmp_file = Path("scratch/_tmp_run.py")
    tmp_file.write_text(modified_content, encoding="utf-8")
    
    res = subprocess.run([sys.executable, str(tmp_file)], capture_output=True, text=True)
    
    print(res.stdout)
    if res.stderr:
        print("STDERR:")
        print(res.stderr)
        
    if tmp_file.exists():
        tmp_file.unlink()
        
    # Copy generated debug files to backup
    out_dir = Path("data/results/debug")
    for suffix in ["blocks", "support_points", "line_snap", "panels_from_snap"]:
        src = out_dir / f"debug_{stem}_{suffix}.JPG"
        if src.exists():
            dest = backup_dir / f"debug_{stem}_{suffix}.JPG"
            shutil.copy2(src, dest)
            print(f"Backed up {src.name} to {dest}")
        else:
            print(f"Warning: {src} not found")
stem_pattern = re.compile(r'IMAGE_STEM\s*=\s*"([^"]+)"')
match = stem_pattern.search(content)
if not match:
    print("Could not find IMAGE_STEM pattern in test file.")
    sys.exit(1)
original_stem = match.group(1)
print(f"Original IMAGE_STEM in file: {original_stem}")

for stem in stems:
    print(f"\n==========================================")
    print(f"RUNNING STEM: {stem}")
    print(f"==========================================")
    modified_content = stem_pattern.sub(f'IMAGE_STEM = "{stem}"', content)
    
    tmp_file = Path("scratch/_tmp_run.py")
    tmp_file.write_text(modified_content, encoding="utf-8")
    
    res = subprocess.run([sys.executable, str(tmp_file)], capture_output=True, text=True)
    
    print(res.stdout)
    if res.stderr:
        print("STDERR:")
        print(res.stderr)
        
    if tmp_file.exists():
        tmp_file.unlink()
        
    # Copy generated debug files to backup
    out_dir = Path("data/results/debug")
    for suffix in ["blocks", "support_points", "line_snap", "panels_from_snap"]:
        src = out_dir / f"debug_{stem}_{suffix}.JPG"
        if src.exists():
            dest = backup_dir / f"debug_{stem}_{suffix}.JPG"
            shutil.copy2(src, dest)
            print(f"Backed up {src.name} to {dest}")
        else:
            print(f"Warning: {src} not found")
