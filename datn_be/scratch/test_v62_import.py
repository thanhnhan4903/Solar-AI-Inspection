import sys
sys.path.insert(0, '.')
from app.services.pv_panel_snapper import snap_panels_with_v61, _load_v62_module
print("Import pv_panel_snapper: OK")

mod = _load_v62_module()
if mod is not None:
    print("v62 module load: OK")
    has_be = hasattr(mod, "process_image_for_backend")
    has_panels = hasattr(mod, "_BACKEND_RESULT_PANELS")
    has_main = hasattr(mod, "main")
    print(f"  Has process_image_for_backend: {has_be}")
    print(f"  Has _BACKEND_RESULT_PANELS: {has_panels}")
    print(f"  Has main: {has_main}")
    if has_be and has_panels:
        print("ALL OK - v62 module ready for use.")
    else:
        print("MISSING FUNCTIONS - check v62 file")
else:
    print("v62 module load: FAIL (see log above)")
