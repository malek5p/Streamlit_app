"""Regenerate docs/reference_case_phantom.json (regression reference, NOT a validation benchmark).
Run from the project root:  python scripts/make_reference_case.py"""
import json, platform, sys
from pathlib import Path
import numpy as np, scipy
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import thermoseeg_3d as t3

TARGET, AZ, EL, RES = [-22.0, 4.0, 10.0], 180.0, 10.0, 2.0
SOLVER = dict(power_w=3.0, exposure_s=60.0, cooldown_s=30.0, active_length_mm=10.0, source_radius_mm=4.0)


def build() -> dict:
    vol = t3.make_phantom(RES)
    d = t3.direction_from_angles(AZ, EL)
    entry = t3.find_entry_on_surface(vol, np.array(TARGET), d)
    out = {"purpose": "regression reference for the synthetic phantom; not a validation benchmark",
           "tool_version": "ThermoSEEG_3D_v3",
           "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
           "input": {"phantom_resolution_mm": RES, "volume_shape": list(vol.shape), "volume_sha256": vol.sha256,
                     "target_mm": TARGET, "azimuth_deg": AZ, "elevation_deg": EL,
                     "entry_mm": [round(float(v), 3) for v in entry]},
           "solver_parameters": SOLVER, "results": {}}
    for preset in t3.PRESETS:
        p = t3.ThermalParams(preset=preset, **SOLVER)
        m = t3.run_litt_comparison(vol, np.array(TARGET), d, p)["metrics"]
        out["results"][preset] = {k: m[k] for k in (
            "max_temp_with_csf_c", "max_temp_without_csf_c", "max_csf_dT_reduction_c", "model_valid")}
        out["results"][preset]["ablation_volume_with_csf_mm3"] = m["with_csf"]["ablation_volume_mm3"]
        out["results"][preset]["ablation_volume_without_csf_mm3"] = m["without_csf"]["ablation_volume_mm3"]
    return out


if __name__ == "__main__":
    dest = ROOT / "docs" / "reference_case_phantom.json"
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(build(), indent=2))
    print("wrote", dest)