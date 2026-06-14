import subprocess, sys, re

base_code = open('scratch/edge_grid_full_image_test.py').read()

for stem in ['DJI_0845_R', 'DJI_0029_R', 'DJI_0087_R']:
    code = base_code.replace('IMAGE_STEM = "DJI_0843_R"', f'IMAGE_STEM = "{stem}"')
    with open('scratch/_tmp_test.py', 'w') as f:
        f.write(code)
    r = subprocess.run([sys.executable, 'scratch/_tmp_test.py'], capture_output=True, text=True, timeout=120)
    lines = (r.stdout + r.stderr).splitlines()
    total = [l for l in lines if 'TOTAL DRAWN' in l or 'Error' in l or 'rror' in l or 'Traceback' in l]
    fallbacks = [l for l in lines if 'PANEL_FALLBACK' in l]
    print(f"{stem}: panels={' | '.join(total[-2:])} | fallbacks={len(fallbacks)}")
