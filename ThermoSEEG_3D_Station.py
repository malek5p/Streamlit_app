"""ThermoSEEG 3D Planning Station (research prototype).

Run:  python -m streamlit run ThermoSEEG_3D_Station.py

3D volumes (synthetic phantom or NIfTI upload) -> brain / CSF / vessel layers in world mm (RAS+) ->
live 3D trajectory control -> transient Pennes bioheat model with live perfusion / CSF / vessel-sink
controls.  NOT clinically validated; see the disclaimer in the app.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import streamlit as st

import thermoseeg_3d as t3
import thermoseeg_3d_view as view
from thermoseeg_config import RunConfigError, parse_exported_run_config_3d, widget_values_3d
from thermoseeg_theme_dark import (STATUS_TONE, html_table, inject_theme, kpi_tiles, panel, pill_html,
                                   render_topbar, section_header, sidebar_brand)

TOOL_VERSION = "ThermoSEEG_3D_v3"

st.set_page_config(page_title="ThermoSEEG 3D Planning Station", layout="wide")
inject_theme()

DISCLAIMER = (
    "Research and education prototype. Tissue constants are literature-order values, not calibrated to any patient; "
    "CSF is a high-effective-conductivity region and vessels a heat-sink term (no flow solver); segmentations are "
    "user-supplied or heuristic and unvalidated; above 100 °C the model is flagged invalid and CEM43 is suppressed. "
    "NOT FOR CLINICAL DECISIONS OR SURGICAL GUIDANCE."
)
ALL_LAYERS = ["Patient brain", "Patient CSF", "Vessels", "Risk regions", "Trajectory", "SEEG contacts", "Applicator capsule",
              "Region sphere", "Temperature field", "LITT ablation", "CSF ΔT reduction"]


# ----------------------------------------------------------------------------- cached helpers
DEMO_DIR = Path(__file__).resolve().parent / "data" / "demo"
DEMO_MNI = DEMO_DIR / "mni152_brain_T1_2mm.nii.gz"
DEMO_FORREST = DEMO_DIR / "sub-01_ses-forrestgump_T1w.nii.gz"


def demo_file_ok(path: Path) -> bool:
    """True only for a readable NIfTI-1/2 file (a corrupt or placeholder file must not offer a broken case)."""
    try:
        if not path.exists():
            return False
        import gzip
        op = gzip.open if path.suffix == ".gz" else open
        with op(path, "rb") as fh:
            head = fh.read(348)
        return len(head) >= 348 and int.from_bytes(head[:4], "little") in (348, 540) or \
            int.from_bytes(head[:4], "big") in (348, 540)
    except Exception:
        return False


FORREST_STRIP_MM = 12.0   # tested on this volume: 6 mm leaves scalp/neck in the heuristic brain mask, 12 mm does not


@st.cache_resource(show_spinner="Loading the bundled MNI152 template…")
def get_mni(work_mm: float) -> t3.Volume3D:
    v = t3.load_nifti_volume({"anatomy": DEMO_MNI.read_bytes()}, work_mm, anatomy_is_brain_only=True)
    v.source = "MNI152 template (bundled, skull-stripped)"
    return v


@st.cache_resource(show_spinner="Loading StudyForrest T1w and building heuristic brain/CSF masks…")
def get_forrest(work_mm: float) -> t3.Volume3D:
    v = t3.load_nifti_volume({"anatomy": DEMO_FORREST.read_bytes()}, work_mm, strip_mm=FORREST_STRIP_MM)
    v.source = "StudyForrest sub-01 T1w (heuristic brain/CSF masks)"
    return v


@st.cache_resource(show_spinner="Building synthetic phantom…")
def get_phantom(res_mm: float) -> t3.Volume3D:
    return t3.make_phantom(res_mm)


@st.cache_data(show_spinner="Extracting brain / CSF / vessel surfaces…", max_entries=6)
def get_static_meshes(vol_id: str, _vol: t3.Volume3D):
    return {
        "Patient brain": t3.surface_mesh(_vol.labels > 0, 0.5, _vol.spacing, _vol.origin, smooth_sigma=1.0),
        "Patient CSF": t3.surface_mesh(_vol.labels == t3.LABEL_CSF, 0.5, _vol.spacing, _vol.origin, smooth_sigma=0.8),
        "Vessels": t3.surface_mesh(_vol.labels == t3.LABEL_VESSEL, 0.5, _vol.spacing, _vol.origin, smooth_sigma=0.6),
    }


@st.cache_data(show_spinner=False, max_entries=6)
def get_risk(vol_id: str, risk_key: tuple, _vol: t3.Volume3D):
    use_pct, pct, manual, min_mm3 = risk_key
    return t3.heuristic_risk_mask(_vol.anatomy, _vol.labels > 0, _vol.spacing, use_pct, pct, manual, min_mm3)


@st.cache_data(show_spinner=False, max_entries=6)
def get_risk_mesh(vol_id: str, risk_key: tuple, _vol: t3.Volume3D, _risk_mask):
    return t3.surface_mesh(_risk_mask, 0.5, _vol.spacing, _vol.origin, smooth_sigma=0.6)


@st.cache_data(show_spinner=False, max_entries=6)
def get_obstacle_distance(vol_id: str, risk_key: tuple, _vol: t3.Volume3D, _risk_mask):
    return t3.obstacle_distance_map_mm(_vol, _risk_mask)


@st.cache_data(show_spinner="Solving the transient bioheat equation (with and without CSF)…", max_entries=8)
def run_litt(vol_id: str, target_key: tuple, dir_key: tuple, params_key: tuple,
             _vol: t3.Volume3D, _params: t3.ThermalParams):
    return t3.run_litt_comparison(_vol, np.array(target_key), np.array(dir_key), _params)


_PLOTLY_HAS_WIDTH = "width" in inspect.signature(st.plotly_chart).parameters


def show_plotly(fig, key: str) -> None:
    # Newer Streamlit names the sizing argument `width`; older versions would silently treat it as a Plotly
    # config kwarg (deprecation warning), so choose the argument from the real signature.
    if _PLOTLY_HAS_WIDTH:
        st.plotly_chart(fig, key=key, width="stretch")
    else:
        st.plotly_chart(fig, key=key, use_container_width=True)


def fmt_pct(v) -> str:
    return "n/a" if v is None else f"{v:.1f} %"



def _apply_settings_upload() -> None:
    """on_change callback of the (advanced) settings uploader: validated import of a ThermoSEEG 3D JSON export."""
    f = st.session_state.get("t3d_settings_upload")
    if f is None:
        return
    try:
        parsed_run = parse_exported_run_config_3d(f.getvalue())
        imported = widget_values_3d(parsed_run)
        target_mm = imported.pop("_target_mm", None)
        for k, v in imported.items():
            st.session_state[k] = v
        st.session_state["_t3d_pending_target"] = (
            None if target_mm is None else {"target": target_mm, "sha": parsed_run.get("volume_sha256")})
        st.session_state["_t3d_settings_msg"] = ("ok", "Saved settings loaded. The saved target is restored only if the "
                                                      "loaded volume matches the saved SHA-256.")
    except RunConfigError as exc:
        st.session_state["_t3d_settings_msg"] = ("err", f"Cannot load settings: {exc}")


# ----------------------------------------------------------------------------- header
st.markdown('<div class="ws-topbar"><div class="ws-logo"><div class="ws-logo-mark">🧠</div>Thermo<b>SEEG</b></div>'
            '<div class="ws-nav"></div><span class="ws-chip hot">Web 3D station · RAS+ mm</span>'
            '<span class="ws-chip">Research prototype</span></div>', unsafe_allow_html=True)
st.warning(DISCLAIMER)
with st.expander("Scope, assumptions and what this app does NOT claim"):
    st.markdown(
        "**What it is.** A computational research prototype for exploring SEEG-guided LITT planning geometry and "
        "transient bioheat simulations, including a modelled CSF-dependent cooling term. This browser app is a "
        "re-implementation for web demonstration; the team's PySide6/PyVista desktop workstation is the primary 3D "
        "interface and the two are not feature-identical.")
    st.markdown(
        "**Does NOT claim:** clinically validated brain/CSF segmentation; outcome prediction or safe-ablation "
        "guarantees; experimentally calibrated CSF heat-transfer coefficients; validation against clinical LITT "
        "data; automatic identification of real lesions; a verified registration or electrode-margin workflow; "
        "readiness for clinical use.")
    st.markdown(
        "**Masks.** Brain, CSF and vessel masks are user-supplied or heuristic and must be inspected against the MRI. "
        "**Model.** Transient Pennes bioheat equation; the CSF sink (W/m³/K) is an empirical, uncalibrated parameter; "
        "the 'team desktop parity' preset reuses the desktop defaults, with the perfusion parameter interpreted as 1/s "
        "(unconfirmed). Above 100 °C the model is flagged invalid and CEM43 is suppressed. Ablation volumes shown are "
        "either the geometric applicator capsule or the CEM43 ≥ 240 min region of this uncalibrated model.")

# ----------------------------------------------------------------------------- sidebar: settings import
with st.sidebar:
    sidebar_brand()
    # ------------------------------------------------------------------------- 1 · Scan / volume
    st.subheader("1 · Scan")
    real_cases = ((["StudyForrest T1w (real scan, bundled)"] if demo_file_ok(DEMO_FORREST) else [])
                  + (["MNI152 template (bundled)"] if demo_file_ok(DEMO_MNI) else []))
    sources = real_cases + ["Upload your scan (NIfTI · DICOM · ZIP · JPG/PNG)", "Synthetic phantom (testing only)"]
    for _p in (DEMO_MNI, DEMO_FORREST):
        if _p.exists() and not demo_file_ok(_p):
            st.caption(f"⚠️ {_p.name} is not a valid NIfTI file and is hidden; replace it with the original download.")
    source = st.radio("Scan source", sources, key="t3d_source")
    vol = None
    if source.startswith("Synthetic"):
        res_mm = st.select_slider("Phantom resolution (mm)", options=[2.0, 2.5, 3.0], value=2.0)
        vol = get_phantom(float(res_mm))
    elif source.startswith("MNI152"):
        st.caption("Local demo file (skull-stripped template, not a patient). Source and licence must be confirmed "
                   "before any redistribution. CSF here is a heuristic intensity class.")
        res_mm = st.select_slider("Template resolution (mm)", options=[2.0, 2.5, 3.0], value=2.5, key="t3d_mni_res")
        try:
            vol = get_mni(float(res_mm))
        except t3.ThermoSEEG3DError as exc:
            st.error(str(exc))
    elif source.startswith("StudyForrest"):
        st.caption("Real T1w head scan of a healthy volunteer (public StudyForrest data), not a patient. Brain and CSF "
                   f"masks are HEURISTIC (scalp strip fixed at {FORREST_STRIP_MM:.0f} mm for this volume); inspect them "
                   "in the Imaging tab.")
        res_mm = st.select_slider("Volume resolution (mm)", options=[2.0, 2.5, 3.0], value=2.5, key="t3d_forrest_res")
        try:
            vol = get_forrest(float(res_mm))
        except t3.ThermoSEEG3DError as exc:
            st.error(str(exc))
    else:
        st.caption("Upload a real scan: NIfTI (.nii / .nii.gz), a DICOM series, a ZIP of either, or a stack of JPG/PNG "
                   "slices. On a public host uploads are processed on that server: use de-identified or public data only.")
        scan_files = st.file_uploader("Scan (required)", type=["nii", "gz", "zip", "dcm", "jpg", "jpeg", "png"],
                                      accept_multiple_files=True, key="t3d_up_anat",
                                      help="Select one file (NIfTI or ZIP) or several files (all DICOM slices, or all JPG/PNG slices).")
        with st.expander("JPG/PNG slices: spacing (only used for image stacks)"):
            img_px = st.number_input("Pixel spacing (mm)", 0.05, 10.0, 1.0, 0.05, key="t3d_img_px")
            img_sl = st.number_input("Slice spacing (mm)", 0.05, 20.0, 1.0, 0.05, key="t3d_img_slice")
            st.caption("Images carry no geometry, so spacing is taken from these values. Assumed: axial slices in "
                       "file-name order (inferior to superior), radiological display (image left = patient right).")
        with st.expander("Optional masks (NIfTI on the same grid as the scan)"):
            up_brain = st.file_uploader("Brain mask (recommended)", type=["nii", "gz"], key="t3d_up_brain")
            up_csf = st.file_uploader("CSF mask (optional)", type=["nii", "gz"], key="t3d_up_csf")
            up_vessel = st.file_uploader("Vessel mask or angiography (optional)", type=["nii", "gz"], key="t3d_up_vessel")
        work_mm = st.select_slider("Working resolution (mm)", options=[2.0, 2.5, 3.0, 4.0], value=2.5)
        stripped = st.checkbox("Scan is already skull-stripped", value=False,
                               help="Use non-zero voxels as the brain mask when no brain mask is uploaded.")
        csf_dark = st.checkbox("Heuristic CSF: CSF darker than tissue (T1-like)", value=True,
                               help="Used only when no CSF mask is uploaded.")
        vessel_thr = st.slider("Heuristic vessel threshold (% of max vesselness)", 10, 80, 30)
        if st.button("Load scan", type="primary"):
            if not scan_files:
                st.error("Please upload the scan first.")
            else:
                try:
                    with st.spinner("Reading, converting and resampling the scan…"):
                        blob, scan_notes, kind = t3.normalise_scan_upload(
                            [(f.name, f.getvalue()) for f in scan_files], float(img_px), float(img_sl))
                        payload = {"anatomy": blob}
                        for _key, _f in (("brain", up_brain), ("csf", up_csf), ("vessel", up_vessel)):
                            if _f is not None:
                                payload[_key] = _f.getvalue()
                        loaded = t3.load_nifti_volume(
                            payload, float(work_mm), bool(csf_dark), vessel_thr / 100.0,
                            float(st.session_state.get("t3d_strip_mm", 6)), anatomy_is_brain_only=bool(stripped))
                        loaded.notes = scan_notes + loaded.notes
                        loaded.source = f"Uploaded scan ({kind})"
                        st.session_state["t3d_upload_vol"] = loaded
                        st.session_state["t3d_candidates"] = []
                except t3.ThermoSEEG3DError as exc:
                    st.session_state.pop("t3d_upload_vol", None)
                    st.error(str(exc))
        vol = st.session_state.get("t3d_upload_vol")
        if vol is not None:
            st.success(f"Loaded: {vol.source} · {vol.shape[0]}×{vol.shape[1]}×{vol.shape[2]} voxels")

    if st.button("↺ Reset view & settings", key="t3d_reset_btn"):
        for _k in [k for k in st.session_state
                   if k.startswith("t3d_") and k not in ("t3d_upload_vol", "t3d_source", "t3d_reset_btn")
                   and not k.startswith("t3d_up_")]:
            del st.session_state[_k]
        st.rerun()

    # ------------------------------------------------------------------------- 1 · Procedure type
    st.subheader("2 · Procedure Type")
    procedure = st.radio("Procedure Type", ["LITT Thermal Ablation", "SEEG Electrode Planning"], key="t3d_proc",
                         help="Select the modality to configure parameters.")
    is_litt = procedure.startswith("LITT")

    # ------------------------------------------------------------------------- 2 · Parameters
    st.divider()
    st.subheader("3 · Parameters")
    params = t3.ThermalParams()
    n_contacts, contact_spacing = 8, 3.5
    use_tt, target_peak = False, 62
    if is_litt:
        params.power_w = st.slider("Laser Power (W)", 2.0, 13.0, 3.0, 0.5, key="t3d_laser_power",
                                   disabled=bool(st.session_state.get("t3d_use_tt", False)),
                                   help="Absorbed power of the uncalibrated 3D model. The 2D range (2-13 W) is kept, but the 2D "
                                        "default of 10 W drives this 3D model above 100 °C (flagged invalid), so the default is 3 W.")
        use_tt = st.checkbox("Set power from target peak temperature", key="t3d_use_tt",
                             help="Solves for the absorbed power that gives this MODELLED peak temperature "
                                  "(the model is linear in power). Not a measured tissue temperature.")
        target_peak = st.slider("Target peak temperature (°C)", 45, 95, 62, 1, key="t3d_tt", disabled=not use_tt)
        params.exposure_s = st.slider("Duration (s)", 5, 120, 60, 5, key="t3d_exposure")
        params.cooldown_s = st.slider("Cooldown after laser off (s)", 0, 180, 30, 5, key="t3d_cooldown")
        params.active_length_mm = st.slider("Active length (mm)", 3.0, 30.0, 10.0, 1.0, key="t3d_active_len")
        params.source_radius_mm = st.slider("Source radius (mm)", 1.0, 10.0, 4.0, 0.5, key="t3d_src_radius")
        st.caption("Transient 3D bioheat model (uncalibrated): duration sets the exposure time of the simulated field; "
                   "not a treatment prediction.")
        st.markdown("**Tissue model (live)**")
        params.preset = st.selectbox("Model preset", list(t3.PRESETS), key="t3d_preset",
                                     help="'Team desktop parity' uses the defaults found in the desktop solver "
                                          "(CSF sink 15000). Both are uncalibrated.")
        P = t3.PRESETS[params.preset]
        params.perfusion_scale = st.slider("Perfusion × (brain tissue)", 0.0, 3.0, 1.0, 0.1,
                                           help="Scales the preset's baseline brain perfusion.")
        params.conductivity_scale = st.slider("Tissue conductivity × (brain)", 0.5, 2.0, 1.0, 0.05, key="t3d_cond_scale",
                                              help="Scales the preset's brain thermal conductivity (live).")
        params.vessel_sink_scale = st.slider("Vessel heat-sink strength ×", 0.0, 50.0, 10.0, 1.0)
        params.csf_k_multiplier = st.slider("CSF effective conductivity ×", 1.0, 20.0, float(P["csf_k_multiplier"]), 0.5,
                                            key=f"t3d_csfk_{params.preset}",
                                            help="Stand-in for convective heat removal by CSF; an assumption, not measured.")
        params.csf_sink_w_m3k = st.slider("CSF cooling sink (W/m³/K)", 0.0, 50000.0, float(P["csf_sink_w_m3k"]), 500.0,
                                          key=f"t3d_csfsink_{params.preset}",
                                          help="Empirical volumetric cooling: CSF mask × sink × (T − 37 °C). Not calibrated.")
    else:
        n_contacts = st.slider("Num Contacts", 2, 18, 8, key="t3d_n_contacts")
        contact_spacing = st.slider("Spacing (mm)", 1.0, 10.0, 3.5, 0.5, key="t3d_contact_spacing",
                                    help="3D uses millimetres (the 2D planner used pixels).")

    # ------------------------------------------------------------------------- 3 · Image processing (risk regions)
    st.divider()
    st.subheader("4 · Image Processing (Risk Regions)")
    use_pct = st.checkbox("Auto Threshold (percentile inside brain)", value=True, key="t3d_auto_threshold",
                          help="Threshold = chosen percentile of voxel intensity inside the 3D brain mask.")
    risk_pct = st.slider("High-Signal Percentile (heuristic)", 80, 99, t3.RISK_PERCENTILE_DEFAULT, disabled=not use_pct,
                         key="t3d_risk_percentile",
                         help="Top X% brightest voxels inside the brain become risk regions. If the risk volume is too "
                              "large, RAISE this value (98-99).")
    manual_thr = st.slider("Manual Threshold (0-255)", 0, 255, 140, disabled=use_pct, key="t3d_manual_threshold")
    strip_mm = st.slider("Scalp/Skull Strip Radius (mm)", 0, 12, 6, key="t3d_strip_mm",
                         help="Opening radius that removes thin scalp/skull bridges from the HEURISTIC brain mask "
                              "(applied when volumes are loaded; unused when a brain mask is supplied). 0 disables stripping.")
    min_risk_mm3 = st.slider("Min Risk Region Volume (mm³)", 0, 500, 30, key="t3d_min_risk_mm3",
                             help="Connected risk regions smaller than this are discarded (2D: area in px²).")
    safety_buffer = st.slider("Safety Buffer (mm)", 0.0, 15.0, t3.DEFAULT_SAFETY_BUFFER_MM, 0.5, key="t3d_safety_buffer",
                              help="A trajectory closer than this to a vessel/risk region is flagged DANGER. "
                                   "Geometric setting, not a clinical margin. Measurements are always in mm in 3D.")
    st.caption("Risk regions are intensity-based heuristics (not vessel or functional-anatomy segmentation).")

if vol is None:
    st.info("Upload a scan in the sidebar (NIfTI, DICOM series, ZIP or JPG/PNG slices) and press 'Load scan' to begin. "
            "The synthetic phantom is available for testing only.")
    st.stop()

for _n in vol.notes:
    if _n.startswith("Brain mask: HEURISTIC"):
        st.warning("⚠️ " + _n)
vol_id = vol.sha256[:12]
lo_b, hi_b = vol.bounds_mm

# restore a saved target (only when the loaded volume is the one the settings were exported from)
_pend = st.session_state.pop("_t3d_pending_target", None)
if _pend is not None:
    if _pend.get("sha") == vol.sha256:
        for _ax, _k in enumerate("xyz"):
            _val = float(np.clip(round(_pend["target"][_ax]), lo_b[_ax], hi_b[_ax]))
            st.session_state[f"t3d_t{_k}_{vol_id}"] = _val
            st.session_state[f"t3d_n{_k}_{vol_id}"] = _val
    else:
        st.warning("Saved target not restored: the loaded volume's SHA-256 differs from the one in the settings file.")

# ----------------------------------------------------------------------------- sidebar: trajectory
with st.sidebar:
    st.divider()
    st.subheader("5 · Trajectory (world mm)")
    dt0 = np.clip(vol.default_target if vol.default_target is not None else (lo_b + hi_b) / 2, lo_b, hi_b)
    coord_mode = st.radio("Target input", ["Sliders", "Exact x/y/z (mm)"], horizontal=True, key="t3d_coord_mode")
    if coord_mode == "Sliders":
        tx = st.slider("Target x · Right+", float(lo_b[0]), float(hi_b[0]), float(round(dt0[0])), 1.0, key=f"t3d_tx_{vol_id}")
        ty = st.slider("Target y · Anterior+", float(lo_b[1]), float(hi_b[1]), float(round(dt0[1])), 1.0, key=f"t3d_ty_{vol_id}")
        tz = st.slider("Target z · Superior+", float(lo_b[2]), float(hi_b[2]), float(round(dt0[2])), 1.0, key=f"t3d_tz_{vol_id}")
    else:
        tx = st.number_input("Target x (mm) · Right+", float(lo_b[0]), float(hi_b[0]), float(round(dt0[0])), 0.5, key=f"t3d_nx_{vol_id}")
        ty = st.number_input("Target y (mm) · Anterior+", float(lo_b[1]), float(hi_b[1]), float(round(dt0[1])), 0.5, key=f"t3d_ny_{vol_id}")
        tz = st.number_input("Target z (mm) · Superior+", float(lo_b[2]), float(hi_b[2]), float(round(dt0[2])), 0.5, key=f"t3d_nz_{vol_id}")
    target = np.array([tx, ty, tz], dtype=float)
    entry_mode = st.radio("Entry", ["Angles → brain-surface entry", "Entry x/y/z (manual)"], key="t3d_entry_mode")
    if entry_mode.startswith("Angles"):
        az = st.slider("Azimuth (°) · 0 = Right, 90 = Anterior", 0, 359, 180, 1, key=f"t3d_az_{vol_id}")
        el = st.slider("Elevation (°)", -80, 80, 10, 1, key=f"t3d_el_{vol_id}")
        entry_manual = None
    else:
        ex = st.number_input("Entry x (mm)", float(lo_b[0]), float(hi_b[0]), float(round(dt0[0] - 40)), 1.0, key=f"t3d_ex_{vol_id}")
        ey = st.number_input("Entry y (mm)", float(lo_b[1]), float(hi_b[1]), float(round(dt0[1])), 1.0, key=f"t3d_ey_{vol_id}")
        ez = st.number_input("Entry z (mm)", float(lo_b[2]), float(hi_b[2]), float(round(dt0[2])), 1.0, key=f"t3d_ez_{vol_id}")
        entry_manual = np.array([ex, ey, ez], dtype=float)

    seeg2 = None
    with st.expander("Companion SEEG electrode" if is_litt else "Second SEEG electrode"):
        seeg2_on = st.checkbox("Add electrode", key="t3d_seeg2_on")
        if seeg2_on:
            d2x = st.slider("Target offset x (mm)", -30.0, 30.0, 12.0, 1.0)
            d2y = st.slider("Target offset y (mm)", -30.0, 30.0, 0.0, 1.0)
            d2z = st.slider("Target offset z (mm)", -30.0, 30.0, 0.0, 1.0)
            az2 = st.slider("Electrode azimuth (°)", 0, 359, 90, 1)
            el2 = st.slider("Electrode elevation (°)", -80, 80, 0, 1)
            nc2 = st.slider("Electrode contacts", 2, 18, 8, key="t3d_nc2")
            sp2 = st.slider("Electrode spacing (mm)", 1.0, 10.0, 3.5, 0.5, key="t3d_sp2")
            seeg2 = (np.array([d2x, d2y, d2z]), az2, el2, nc2, sp2)
    region_cfg = None
    with st.expander("Structure sphere (user-defined)"):
        st.caption("A sphere you place yourself. It is NOT anatomy.")
        region_on = st.checkbox("Add sphere", key="t3d_region_on")
        if region_on:
            role = st.radio("Role", ["Target to cover", "Structure to protect"], key="t3d_region_role")
            rdx = st.slider("Offset x from target (mm)", -30.0, 30.0, 0.0, 1.0, key="t3d_rdx")
            rdy = st.slider("Offset y from target (mm)", -30.0, 30.0, 0.0, 1.0, key="t3d_rdy")
            rdz = st.slider("Offset z from target (mm)", -30.0, 30.0, 0.0, 1.0, key="t3d_rdz")
            rrad = st.slider("Radius (mm)", 1.0, 25.0, 6.0, 0.5, key="t3d_rrad")
            region_cfg = {"center": target + np.array([rdx, rdy, rdz]), "radius": float(rrad), "role": role,
                          "label": "Target region" if role.startswith("Target") else "Protected structure"}

    st.divider()
    with st.expander("Advanced · load saved settings (JSON)"):
        st.file_uploader("ThermoSEEG 3D settings export (.json)", type=["json"], key="t3d_settings_upload",
                         on_change=_apply_settings_upload)
        _msg = st.session_state.get("_t3d_settings_msg")
        if _msg:
            (st.success if _msg[0] == "ok" else st.error)(_msg[1])

# ----------------------------------------------------------------------------- geometry
if vol.label_at(target) == t3.LABEL_BG:
    st.error("The target is outside the intracranial mask. Move the target (x/y/z sliders).")
    st.stop()
if entry_manual is None:
    entry = t3.find_entry_on_surface(vol, target, t3.direction_from_angles(az, el))
    if entry is None:
        st.error("Could not find a brain-surface entry along this direction. Change azimuth/elevation.")
        st.stop()
else:
    entry = entry_manual
if np.linalg.norm(entry - target) < 1.0:
    st.error("Entry and target must be at least 1 mm apart.")
    st.stop()
out_dir = t3.unit(entry - target)
risk_key = (bool(use_pct), int(risk_pct), int(manual_thr), int(min_risk_mm3))
try:
    risk = get_risk(vol_id, risk_key, vol)
except t3.ThermoSEEG3DError as exc:
    st.error(str(exc))
    st.stop()
if risk["plateau_warning"]:
    st.warning(f"⚠️ Risk regions cover {risk['fraction_of_brain'] * 100:.0f} % of the brain: the threshold sits on a tissue "
               "plateau. Raise the High-Signal Percentile (or the manual threshold).")
vdist = get_obstacle_distance(vol_id, risk_key, vol, risk["mask"])
analysis = t3.analyze_trajectory(vol, entry, target, vdist, buffer_mm=float(safety_buffer))
active_far = target + out_dir * params.active_length_mm if is_litt else None

seeg2_info = None
if seeg2 is not None:
    t2 = target + seeg2[0]
    e2 = t3.find_entry_on_surface(vol, t2, t3.direction_from_angles(seeg2[1], seeg2[2])) if vol.label_at(t2) != t3.LABEL_BG else None
    if e2 is None or np.linalg.norm(e2 - t2) < 1.0:
        st.warning("Second electrode: target outside the mask or no valid entry along that direction; electrode not shown.")
    else:
        seeg2_info = {"entry": e2, "target": t2, "analysis": t3.analyze_trajectory(vol, e2, t2, vdist, buffer_mm=float(safety_buffer)),
                      "contacts": t3.seeg_contacts(e2, t2, seeg2[3], seeg2[4]),
                      "clearance_mm": t3.segment_segment_distance(entry, target, e2, t2)}

capsule = None
if is_litt:
    capsule = t3.ablation_capsule_metrics(vol, target, out_dir, params.active_length_mm, params.source_radius_mm)

# ----------------------------------------------------------------------------- thermal run
result = None
power_note = None
if is_litt and use_tt:
    try:
        _p1 = replace(params, power_w=1.0)
        _base = run_litt(vol_id, tuple(np.round(target, 3)), tuple(np.round(out_dir, 5)), _p1.key(), vol, _p1)
        _rise1 = float(_base["metrics"]["max_temp_with_csf_c"]) - t3.BODY_TEMP_C
        if _rise1 <= 0.05:
            st.error("The source does not heat the tissue at this target; move the target.")
            st.stop()
        params.power_w = round((float(target_peak) - t3.BODY_TEMP_C) / _rise1, 2)
        power_note = f"Required absorbed power for a modelled peak of {target_peak} °C: {params.power_w:.2f} W"
        if not 2.0 <= params.power_w <= 13.0:
            power_note += " (outside the 2-13 W slider range)"
    except t3.ThermoSEEG3DError as exc:
        st.error(str(exc))
        st.stop()
if is_litt:
    try:
        result = run_litt(vol_id, tuple(np.round(target, 3)), tuple(np.round(out_dir, 5)), params.key(), vol, params)
    except t3.ThermoSEEG3DError as exc:
        st.error(str(exc))
        st.stop()
contacts = None if is_litt else t3.seeg_contacts(entry, target, n_contacts, contact_spacing)

region_stats = None
region_dist = None
if region_cfg is not None:
    seg_a, seg_b = (target, active_far) if is_litt else (entry, target)
    r_extra = params.source_radius_mm if is_litt else 0.0
    region_dist = max(0.0, t3.point_segment_distance(region_cfg["center"], seg_a, seg_b) - region_cfg["radius"] - r_extra)
    if result is not None:
        region_stats = t3.region_thermal_stats(vol, result, region_cfg["center"], region_cfg["radius"])

tab_plan, tab_img, tab_therm, tab_report = st.tabs(["Planning", "Imaging", "Thermal Simulation", "Report"])

with tab_plan:
    # ----------------------------------------------------------------------------- layout
    section_header("1", "3D planning view", f"{vol.source} · {vol.shape[0]}×{vol.shape[1]}×{vol.shape[2]} voxels · "
                   f"{vol.spacing[0]:.2f} mm")
    kpi_tiles([
        ("Trajectory length", f"{analysis['length_mm']:.1f} mm", f"az {analysis['azimuth_deg']:.0f}° · el {analysis['elevation_deg']:.0f}°", "neutral"),
        ("Min obstacle clearance",
         "no vessels" if analysis["min_vessel_clearance_mm"] is None else f"{analysis['min_vessel_clearance_mm']:.1f} mm",
         f"vessel/risk · buffer {safety_buffer:.1f} mm", "neutral"),
        ("CSF crossing (path)", f"{analysis['csf_crossing_mm']:.1f} mm", "along the trajectory", "neutral"),
        ("CSF exposure (capsule)", "—" if capsule is None else f"{capsule['csf_exposure_fraction'] * 100:.1f} %",
         "applicator volume in CSF", "neutral"),
        ("Heuristic status", analysis["status"].replace("_", " ").title(), "software status only",
         STATUS_TONE.get(analysis["status"], "neutral")),
    ])
    for reason in analysis["reasons"]:
        st.caption(f"• {reason}")

    col_plot, col_info = st.columns([3, 1])
    with col_plot:
        view_name = st.radio("Camera", list(view.CAMERAS), horizontal=True, key="t3d_camera")
        plot_slot = st.container()
        ctl_slot = st.container()

    default_layers = (["Patient brain", "Patient CSF", "Vessels", "Risk regions", "Trajectory", "Applicator capsule", "Region sphere",
                       "Temperature field", "LITT ablation", "CSF ΔT reduction"]
                      if is_litt else ["Patient brain", "Patient CSF", "Vessels", "Risk regions", "Trajectory", "SEEG contacts", "Region sphere"])
    with ctl_slot:
        layers = st.multiselect("Layers (also toggle via the legend)", ALL_LAYERS + [],
                                default=default_layers + (["SEEG contacts"] if seeg2_info and is_litt else []),
                                key=f"t3d_layers_{procedure[:4]}_{int(seeg2_info is not None)}")
        snap_time = None
        iso_levels = []
        if result is not None:
            times = [round(float(t), 1) for t in result["with_csf"]["times"]]
            nearest = int(np.argmin([abs(t - params.exposure_s) for t in times]))
            snap_time = st.select_slider("Time (s) · scrub the transient solution", options=times, value=times[nearest])
            iso_levels = st.multiselect("Temperature iso-surfaces (°C)", [43, 50, 60, 70, 80, 90], default=[43, 50, 60])

    static_meshes = get_static_meshes(vol_id, vol)
    if "Risk regions" in layers and risk["mask"].any():
        static_meshes = {**static_meshes, "Risk regions": get_risk_mesh(vol_id, risk_key, vol, risk["mask"])}
    temp_shells, ablation_mesh, dT_mesh, capsule_mesh, temp_max = [], None, None, None, 80.0
    fields = {}
    if result is not None:
        wc = result["with_csf"]
        snap = wc["snapshots"][times.index(snap_time)]
        box = result["box"]
        crop_origin = vol.origin + np.array([b[0] for b in box]) * vol.spacing
        temp_max = max(float(wc["peak"].max()), 50.0)
        for lvl in sorted(iso_levels):
            if lvl < float(snap.max()):
                m = t3.surface_mesh(snap, float(lvl), vol.spacing, crop_origin)
                if m is not None:
                    temp_shells.append((float(lvl), m))
        if wc["dose"] is not None and float(wc["dose"].max()) > t3.CEM43_ABLATION_MIN:
            ablation_mesh = t3.surface_mesh(wc["dose"], t3.CEM43_ABLATION_MIN, vol.spacing, crop_origin)
        dTp = result["dT_reduction_peak"]
        if float(dTp.max()) > 0.5:
            dT_mesh = t3.surface_mesh(dTp, max(0.5, 0.3 * float(dTp.max())), vol.spacing, crop_origin)
        if "Applicator capsule" in layers:
            capsule_mesh = t3.capsule_surface(target, active_far, params.source_radius_mm)
        shape = vol.shape
        fields = {
            "Temperature at selected time (°C)": (t3.embed(np.where(snap > 37.3, snap, np.nan), box, shape), 37.0, temp_max, "Inferno"),
            "Peak temperature (°C)": (t3.embed(np.where(wc["peak"] > 37.3, wc["peak"], np.nan), box, shape), 37.0, temp_max, "Inferno"),
            "CSF ΔT reduction (°C)": (t3.embed(np.where(dTp > 0.05, dTp, np.nan), box, shape), 0.0, max(1.0, float(dTp.max())), "Viridis"),
        }
        if wc["dose"] is not None:
            d = wc["dose"]
            fields["CEM43 dose (log10 min)"] = (t3.embed(np.where(d > 0.01, np.log10(np.maximum(d, 1e-3)), np.nan), box, shape), -2.0, 6.0, "Hot")

    title = f"ThermoSEEG · {procedure} · target ({target[0]:.1f}, {target[1]:.1f}, {target[2]:.1f}) mm"
    fig3d = view.build_scene(
        vol, static_meshes, target, entry, procedure, layers,
        active_far_point=active_far, capsule_mesh=capsule_mesh,
        contacts=contacts["contacts"] if contacts else None,
        seeg_line=(seeg2_info["entry"], seeg2_info["target"]) if seeg2_info else None,
        seeg_contact_list=seeg2_info["contacts"]["contacts"] if seeg2_info else None,
        region=region_cfg, temp_shells=temp_shells, temp_max_c=temp_max, ablation_mesh=ablation_mesh, dT_mesh=dT_mesh,
        view=view_name, title=title)
    with plot_slot:
        show_plotly(fig3d, "t3d_scene")

    # ----------------------------------------------------------------------------- left info panels
    with col_info:
        sx = vol.stats_ml()
        st.markdown(pill_html("Heuristic status: " + analysis["status"].replace("_", " ").title(),
                              STATUS_TONE.get(analysis["status"], "neutral")), unsafe_allow_html=True)
        panel("Risk assessment (geometric)", [
            ("Min obstacle clearance", "no vessels/risk" if analysis["min_vessel_clearance_mm"] is None
             else f"{analysis['min_vessel_clearance_mm']:.1f} mm"),
            ("CSF crossing (path)", f"{analysis['csf_crossing_mm']:.1f} mm"),
            ("Path outside brain mask", f"{analysis['outside_mask_mm']:.1f} mm"),
            ("CSF exposure (capsule)", "—" if capsule is None else f"{capsule['csf_exposure_fraction'] * 100:.1f} %")])
        st.caption("Motor / language / hippocampal risk needs a validated atlas segmentation, which this prototype does not have.")
        _plan = [("Procedure", procedure.split()[0]), ("SEEG electrodes", (1 if not is_litt else 0) + (1 if seeg2_info else 0)),
                 ("LITT targets", 1 if is_litt else 0)]
        if result is not None:
            _plan += [("Modelled peak (with CSF)", f"{result['metrics']['max_temp_with_csf_c']:.1f} °C"),
                      ("Laser power", f"{params.power_w:.2f} W"),
                      ("Ablation zone (CEM43 ≥ 240)", "n/a" if result["metrics"]["with_csf"]["ablation_volume_mm3"] is None
                       else f"{result['metrics']['with_csf']['ablation_volume_mm3'] / 1000:.2f} mL")]
        panel("Current plan", _plan)
        if power_note:
            st.caption(power_note)
        panel("Case", [("Source", vol.source), ("SHA-256", vol.sha256[:12] + "…"),
                       ("Brain", f"{sx['brain tissue']:.0f} mL"), ("CSF", f"{sx['CSF']:.0f} mL"),
                       ("Vessels", f"{sx['vessel']:.2f} mL" if sx["vessel"] > 0 or vol.source == "synthetic phantom"
                        else "not supplied (no map)"),
                       ("Risk regions", f"{risk['volume_ml']:.2f} mL"),
                       ("Risk threshold", f"{risk['threshold_display_0_255']:.0f} / 255")])
        panel("Geometry (world mm, RAS+)", [
            ("Target", f"{target[0]:.1f}, {target[1]:.1f}, {target[2]:.1f}"),
            ("Entry", f"{entry[0]:.1f}, {entry[1]:.1f}, {entry[2]:.1f}"),
            ("Length", f"{analysis['length_mm']:.1f} mm"),
            ("Azimuth / elevation", f"{analysis['azimuth_deg']:.0f}° / {analysis['elevation_deg']:.0f}°")])
        if result is not None:
            m = result["metrics"]
            rel = m["relative_dose_reduction_pct"]
            panel("LITT applicator", [("Active length", f"{params.active_length_mm:.1f} mm"),
                                      ("Radius", f"{params.source_radius_mm:.1f} mm"),
                                      ("Capsule volume", f"{capsule['capsule_volume_mm3'] / 1000:.2f} mL"),
                                      ("Capsule in CSF", fmt_pct(capsule["csf_exposure_fraction"] * 100))])
            panel("Modelled temperature", [("Max + CSF", f"{m['max_temp_with_csf_c']:.2f} °C"),
                                           ("Max − CSF", f"{m['max_temp_without_csf_c']:.2f} °C"),
                                           ("Max CSF ΔT", f"{m['max_csf_dT_reduction_c']:.2f} °C"),
                                           ("Mean CSF heating ratio", "n/a" if m["mean_csf_heating_ratio"] is None else m["mean_csf_heating_ratio"])])
            av = m["with_csf"]["ablation_volume_mm3"]
            avr = m["ablation_volume_reduction_pct"]
            panel("Thermal dose · CEM43 (min)", [
                ("Max + CSF", t3.fmt_dose(m["with_csf"]["max_cem43_min"])),
                ("Max − CSF", t3.fmt_dose(m["without_csf"]["max_cem43_min"])),
                ("Relative dose reduction", "n/a" if rel is None else f"{rel:.1f} %"),
                (f"Ablation zone (≥{t3.CEM43_ABLATION_MIN:.0f})", "n/a" if av is None else f"{av / 1000:.2f} mL"),
                ("CSF changes ablation volume", "n/a" if avr is None else f"{-avr:+.1f} %")])
            st.caption(f"Solver: {m['steps']} explicit steps, dt = {m['dt_s']} s")
        if contacts is not None:
            panel("SEEG electrode", [("Contacts placed", len(contacts["contacts"])), ("Span", f"{contacts['required_span_mm']} mm"),
                                     ("Fits trajectory", "yes" if contacts["fits"] else "no")])
        if seeg2_info is not None:
            panel("Second SEEG electrode", [
                ("Target", ", ".join(f"{v:.1f}" for v in seeg2_info["target"])),
                ("Length", f"{seeg2_info['analysis']['length_mm']:.1f} mm"),
                ("Min clearance to main path", f"{seeg2_info['clearance_mm']:.1f} mm"),
                ("Status", seeg2_info["analysis"]["status"].replace("_", " ").title())])
        if region_cfg is not None:
            rows = [("Role", region_cfg["role"]), ("Radius", f"{region_cfg['radius']:.1f} mm"),
                    ("Volume", f"{4 / 3 * np.pi * region_cfg['radius'] ** 3 / 1000:.2f} mL"),
                    ("Distance path/applicator → sphere", f"{region_dist:.1f} mm")]
            if region_stats and region_stats.get("inside_thermal_box"):
                rows.append(("Max temperature inside", f"{region_stats['max_temp_c']:.1f} °C"))
                rows.append(("Max CEM43 inside", t3.fmt_dose(region_stats["max_cem43_min"])))
                if region_cfg["role"].startswith("Target"):
                    cov = region_stats["coverage_pct"]
                    rows.append((f"Coverage (CEM43 ≥ {t3.CEM43_ABLATION_MIN:.0f})", "n/a" if cov is None else f"{cov:.1f} %"))
            panel("Structure sphere", rows)
        st.caption("Research prototype · not clinically validated")

    if result is not None:
        for w in result["warnings"]:
            st.error(f"🔥 {w}")


    st.markdown("**Views through the target**")
    _tf = fields.get("Temperature at selected time (°C)") if fields else None
    if _tf is not None:
        _arr, _vmin, _vmax, _cs = _tf
        _cfig = view.build_slices(vol, _arr, target, entry, "Temperature (°C)", _vmin, _vmax, _cs)
    else:
        _cfig = view.build_slices(vol, None, target, entry, "", 0, 1)
    show_plotly(_cfig, "t3d_slices_compact")

with tab_img:
    # ----------------------------------------------------------------------------- slices + curve
    section_header("2", "Orthogonal slices through the target", "anatomy + CSF (cyan) + vessels (red) + selected field")
    if fields:
        field_name = st.selectbox("Overlay field", list(fields.keys()) + ["None (anatomy + labels only)"])
        if field_name in fields:
            arr, vmin, vmax, cs = fields[field_name]
            slice_fig = view.build_slices(vol, arr, target, entry, field_name, vmin, vmax, cs)
        else:
            slice_fig = view.build_slices(vol, None, target, entry, "", 0, 1)
    else:
        slice_fig = view.build_slices(vol, None, target, entry, "", 0, 1)
    show_plotly(slice_fig, "t3d_slices")

    with st.expander("Volume notes"):
        for n in vol.notes:
            st.write("• " + n)


with tab_therm:
    if result is not None:
        section_header("3", "Transient response", "peak temperature and cumulative CEM43 versus time")
        show_plotly(view.build_time_curve(result, params.exposure_s), "t3d_curve")

    if result is None:
        st.info("Thermal simulation is available in LITT mode: choose 'LITT Thermal Ablation' in the sidebar.")
    else:
        st.caption(power_note or f"Absorbed power {params.power_w:.2f} W for {params.exposure_s:.0f} s (+{params.cooldown_s:.0f} s cooldown).")
        for w in result["warnings"]:
            st.error(f"🔥 {w}")
    # ----------------------------------------------------------------------------- tissue properties
    with st.expander("Tissue properties used by the model", expanded=True):
        P = t3.PRESETS[params.preset]
        h_brain = P["brain"]["perfusion_1_per_s"] * P["blood_rho"] * P["blood_c"]
        rows = []
        for lab, key in ((t3.LABEL_BRAIN, "brain"), (t3.LABEL_CSF, "csf"), (t3.LABEL_VESSEL, "vessel")):
            t = P[key]
            k = t["k"] * (params.csf_k_multiplier if lab == t3.LABEL_CSF else params.conductivity_scale if lab == t3.LABEL_BRAIN else 1.0)
            perf = {t3.LABEL_BRAIN: params.perfusion_scale * h_brain, t3.LABEL_CSF: params.csf_sink_w_m3k,
                    t3.LABEL_VESSEL: params.vessel_sink_scale * h_brain}[lab]
            rows.append([t3.LABEL_NAMES[lab], f"{k:.3f}", f"{t['rho']:.0f}", f"{t['c']:.0f}", f"{perf:.0f}"])
        html_table(["Tissue", "k (W/m/K)", "ρ (kg/m³)", "c (J/kg/K)", "h perfusion/sink (W/m³/K)"], rows)
        st.caption(f"Preset: {params.preset}. Constants are unvalidated and not all have literature citations. "
                   "Pennes: ρc ∂T/∂t = ∇·(k∇T) − h (T − 37 °C) + Q. Active segment: distal tip at the target, "
                   "extending back toward the entry (same convention as the applicator capsule drawn here).")

with tab_report:
    # ----------------------------------------------------------------------------- candidates
    if "t3d_candidates" not in st.session_state:
        st.session_state["t3d_candidates"] = []
    cc1, cc2, _ = st.columns([1, 1, 4])
    with cc1:
        if st.button("Capture candidate"):
            rec = {"#": len(st.session_state["t3d_candidates"]) + 1,
                   "target": ", ".join(f"{v:.0f}" for v in target), "az/el": f"{analysis['azimuth_deg']:.0f}/{analysis['elevation_deg']:.0f}",
                   "length mm": f"{analysis['length_mm']:.1f}",
                   "vessel mm": "—" if analysis["min_vessel_clearance_mm"] is None else f"{analysis['min_vessel_clearance_mm']:.1f}",
                   "CSF path mm": f"{analysis['csf_crossing_mm']:.1f}"}
            if result is not None:
                rec["CSF caps %"] = f"{capsule['csf_exposure_fraction'] * 100:.1f}"
                rec["max °C"] = f"{result['metrics']['max_temp_with_csf_c']:.1f}"
                rec["ΔT CSF °C"] = f"{result['metrics']['max_csf_dT_reduction_c']:.1f}"
            st.session_state["t3d_candidates"].append(rec)
    with cc2:
        if st.button("Clear candidates"):
            st.session_state["t3d_candidates"] = []
    if st.session_state["t3d_candidates"]:
        cols = list(st.session_state["t3d_candidates"][-1].keys())
        html_table(cols, [[r.get(c, "") for c in cols] for r in st.session_state["t3d_candidates"][-10:]])

    # ----------------------------------------------------------------------------- export
    section_header("4", "Export")


    def _jsonable(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o).__name__)


    sx_vessel_supplied = bool((vol.labels == t3.LABEL_VESSEL).any())
    record = {
        "metadata": {"tool_version": TOOL_VERSION, "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                     "disclaimer": DISCLAIMER, "determinism": "not established"},
        "volume": {"vessel_map_supplied": bool(sx_vessel_supplied), "source": vol.source, "sha256": vol.sha256, "shape": list(vol.shape), "spacing_mm": vol.spacing.tolist(),
                   "origin_mm": vol.origin.tolist(), "axes": "RAS+ (x Right, y Anterior, z Superior)",
                   "volumes_ml": vol.stats_ml(), "notes": vol.notes},
        "procedure": procedure,
        "reproducibility": {"configuration": {
            "procedure": procedure,
            "laser_power_w": params.power_w, "duration_s": int(params.exposure_s), "cooldown_s": int(params.cooldown_s),
            "active_length_mm": params.active_length_mm, "source_radius_mm": params.source_radius_mm,
            "contact_count": int(n_contacts), "contact_spacing_mm": float(contact_spacing),
            "use_percentile": bool(use_pct), "risk_percentile": int(risk_pct), "manual_threshold": int(manual_thr),
            "strip_mm": int(strip_mm), "min_risk_volume_mm3": int(min_risk_mm3), "safety_buffer_mm": float(safety_buffer),
            "target_mm": target.tolist()}},
        "risk_regions": {"threshold_0_255": risk["threshold_display_0_255"], "volume_ml": risk["volume_ml"],
                         "fraction_of_brain": risk["fraction_of_brain"], "note": "heuristic intensity mask, not vessel segmentation"},
        "points_mm": {"target": target.tolist(), "entry": entry.tolist()},
        "trajectory_analysis": analysis,
        "applicator_capsule": capsule,
        "thermal_parameters": None if result is None else params.__dict__,
        "thermal_metrics": None if result is None else result["metrics"],
        "thermal_warnings": None if result is None else result["warnings"],
        "seeg": contacts,
        "second_seeg": None if seeg2_info is None else {
            "entry": seeg2_info["entry"].tolist(), "target": seeg2_info["target"].tolist(),
            "analysis": seeg2_info["analysis"], "contacts": seeg2_info["contacts"],
            "clearance_to_main_mm": seeg2_info["clearance_mm"]},
        "structure_sphere": None if region_cfg is None else {
            "role": region_cfg["role"], "center_mm": region_cfg["center"].tolist(), "radius_mm": region_cfg["radius"],
            "distance_mm": region_dist, "thermal_stats": region_stats, "note": "user-defined sphere, not anatomy"},
        "candidates": st.session_state.get("t3d_candidates", []),
    }
    c1, c2 = st.columns(2)
    with c1:
        st.download_button("📥 Download run record (JSON)", json.dumps(record, indent=2, default=_jsonable),
                           file_name=f"ThermoSEEG_3D_{vol_id}.json", mime="application/json")
    with c2:
        if st.toggle("Prepare interactive 3D (standalone HTML)"):
            st.download_button("🌐 Download interactive 3D (HTML)", fig3d.to_html(include_plotlyjs="cdn", full_html=True),
                               file_name=f"ThermoSEEG_3D_{vol_id}.html", mime="text/html")