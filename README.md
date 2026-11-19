# ThermoSEEG RC — research prototype

ThermoSEEG is a **2D research and educational prototype** for image-mask geometry, threshold sensitivity, finite candidate search, geometric SEEG contact placement, and a simplified synthetic thermal demonstration. It is not clinically validated and must not be used for clinical decisions or surgical guidance.

## Live demo

**Open the 3D Planning Station in your browser: `https://thermoseeg.streamlit.app/`** (replace after deploying; see "Deploying a public demo" below).

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://thermoseeg.streamlit.app/)

The live demo opens on the **3D Planning Station** with a **synthetic phantom**, so no patient data is needed. Live controls: target x/y/z (sliders or exact mm), entry angles, laser power, duration, perfusion, tissue conductivity, vessel heat sink, CSF cooling, and a time scrubber over the transient solution. The hosted app is the browser re-implementation, not the PySide6/PyVista desktop workstation.

## Two apps in one

- **3D Planning Station** (`ThermoSEEG_3D_Station.py`, the main page; dark workstation theme): 3D volumes in RAS+ millimetre coordinates, brain / CSF / vessel layers, live 3D trajectory control (target x/y/z, azimuth/elevation or manual entry, camera presets), SEEG contacts and an optional second electrode with path-to-path clearance, a user-defined structure sphere (target to cover or structure to protect), CSF exposure of the applicator capsule, and a transient Pennes bioheat model with live perfusion, vessel heat-sink and CSF controls, time scrubbing, cumulative CEM43 curves and a CSF-aware vs conventional comparison. Candidate trajectories can be captured into a table and exported with the run record.
- **2D Slice Planner** (`ThermoSEEG_RC_UI.py`, second page): the original single-slice research prototype described below.

## Runtime and installation

Use 64-bit **Python 3.13** (3.11-3.13 are supported; do not use 3.14, the pinned libraries have no wheels for it yet).

PowerShell:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run ThermoSEEG_3D_Station.py
```

Git Bash (Windows) / macOS / Linux:

```bash
py -3.13 -m venv .venv          # macOS/Linux: python3.13 -m venv .venv
source .venv/Scripts/activate   # macOS/Linux: source .venv/bin/activate
python --version                # must print 3.13.x
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run ThermoSEEG_3D_Station.py
```

Run only the 2D planner with `python -m streamlit run ThermoSEEG_RC_UI.py`.

`requirements.txt` is the supported install file. `requirements-lock.txt` pins the original 2D-only dependency set and does **not** include the 3D libraries (plotly, scikit-image, nibabel); regenerate it with pip-tools before using `--require-hashes`.

## Deploying a public demo (Streamlit Community Cloud)

1. Push the code to a GitHub repository you own (fork it if you are not the owner). Keep `requirements.txt` in the repository root.
2. Go to share.streamlit.io, sign in with GitHub, choose **Create app**, and select the repository and branch.
3. Set **Main file path** to `ThermoSEEG_3D_Station.py` (not `ThermoSEEG_RC_UI.py`, which is the 2D page). Under **Advanced settings** choose Python 3.13.
4. Deploy, then paste the resulting `https://....streamlit.app` link into the "Live demo" section above.

Do not commit patient data, and do not invite judges to upload identifiable patient images to a public host: uploads are processed on that server. Use the built-in phantom or de-identified/public datasets.

## 3D Planning Station: inputs and scope

- **Scan upload (3D station):** NIfTI (`.nii`, `.nii.gz`), a **DICOM series** (MR or CT; loose `.dcm` files or a ZIP), a **ZIP** of any of these, or a **stack of JPG/PNG slices** (at least 8; pixel and slice spacing are entered by the user because images carry no geometry, and orientation is assumed axial/radiological). Everything is converted to one NIfTI volume and then passes through the same loader, Gaussian heat source, bioheat solver and CEM43 pipeline. DICOM patient identifiers are not read, stored or exported. A single JPG/PNG cannot form a 3D volume (use the 2D Slice Planner). Uploads are processed on the hosting server: use de-identified or public data only.
- Volumes: NIfTI-1 (`.nii`, `.nii.gz`; NIfTI-2 needs nibabel): anatomy (required), brain mask (strongly recommended), CSF mask, vessel mask or angiography (optional). All volumes must share the anatomy voxel grid (same shape and affine); no registration is performed. Every volume is resampled onto an axis-aligned RAS+ grid (2-4 mm) using the **full affine**, so flips, anisotropic voxels and oblique acquisitions keep exact world millimetre coordinates (unit-tested, including a 12 degree oblique case). The same world coordinates are used for target, entry, trajectories and exports.
- Built-in cases (offered only when the file exists under `data/demo/`): `mni152_brain_T1_2mm.nii.gz` (MNI152) and `sub-01_ses-forrestgump_T1w.nii.gz` (StudyForrest T1w; heuristic brain mask with a 12 mm scalp strip, tested on this volume). Neither is distributed with this repository until licences are confirmed; `data/` is git-ignored. Judges can always use the phantom or upload their own NIfTI.
- Optional local demo volume: if a skull-stripped template is placed at `data/demo/mni152_brain_T1_2mm.nii.gz` the app offers it as "MNI152 template". It is **not distributed with this repository** (source and licence unconfirmed) and `data/` is git-ignored. The default public demo is the generated synthetic phantom.
- **Real T1 scans still containing skull, scalp and neck need an external brain mask.** The built-in heuristic only works on images that are already skull-stripped (tested here: on a real defaced T1w it includes scalp and neck). Generate a mask once with a dedicated tool and upload it as the brain mask, for example (check each tool's documentation for the exact options):
  - FreeSurfer SynthStrip: `mri_synthstrip -i T1w.nii.gz -o T1w_brain.nii.gz -m brain_mask.nii.gz`
  - HD-BET: `hd-bet -i T1w.nii.gz -o T1w_brain.nii.gz`
  - FSL BET: `bet T1w.nii.gz T1w_brain -m -f 0.4`
  A segmentation tool such as SynthSeg additionally provides ventricle/CSF labels that can be exported as a CSF mask.
- If no brain/CSF/vessel mask is supplied, **heuristic** fallbacks are used (Otsu brain, intensity CSF, Hessian vesselness). They are not validated segmentations and are labelled as such in the app and exports.
- Thermal model: transient Pennes bioheat equation, explicit finite differences on a cropped box around the probe. CSF is a high-effective-conductivity, non-perfused region (the "CSF effective conductivity x" slider is an assumption); vessels are a heat-sink term, not a flow solver. Tissue constants are literature-order values, uncalibrated. Peak temperature above 100 C marks the model invalid and CEM43 is suppressed; doses above 1e6 equivalent minutes are shown as saturated.
- Trajectory checks (vessel clearance, CSF crossing, intracranial containment) use experimental millimetre thresholds, not clinical safety limits.
- Not clinically validated. Research and education only.

## Data sources and licences

| Data | What it is | Source / version | Licence / citation | May be redistributed? |
|---|---|---|---|---|
| Synthetic phantom | Generated in code (`make_phantom`) | this repository | project licence | yes |
| `sub-01_ses-forrestgump_T1w.nii.gz` | T1-weighted MRI (StudyForrest) | studyforrest.org; **confirm exact file/version and record it here** | ODC PDDL (per https://studyforrest.org/access.html): cite the project and report derivative availability | **yes, once source/version is documented** |
| `mni152_brain_T1_2mm.nii.gz` | skull-stripped MNI152-type template | **not yet verified** (FSL, ICBM 152 or another distribution) | **not yet verified** (FSL is generally non-commercial) | **no, until confirmed** |

Fill this table in, with citations, before any public release. No patient data may be committed or uploaded to a public host.

## Model assumptions and limitations (3D station)

- Transient Pennes bioheat equation, explicit finite differences, on a cropped box around the probe; paired runs with and without the CSF model use an identical heat source.
- Heat source: Gaussian radial profile around the active segment (sigma = radius / 2), absorbed in non-CSF tissue; wavelength is not modelled.
- **Active segment:** distal tip at the target, extending back toward the entry. The desktop code reportedly builds the thermal source from the entry end while `AblationGeometry` uses the distal end; this must be reconciled in the desktop solver before claiming geometry parity.
- CSF models: (a) effective conductivity multiplier (assumption) and (b) an empirical volumetric cooling sink, CSF mask x sink x (T - 37 C), as in the desktop solver. The sink (default 15000, interpreted as W/m^3/K) is **uncalibrated**.
- Presets: "Literature-order (default)" and "Team desktop parity" (desktop defaults: tissue 1050 kg/m^3, 3600 J/kg/K, 0.51 W/m/K; blood 1060, 3600; perfusion parameter 0.01 interpreted as 1/s and **unconfirmed**; CSF 1007, 4180, 0.60). Not all constants have literature citations yet.
- Above 100 C the model is flagged invalid and CEM43 is suppressed; doses above 1e6 equivalent minutes are shown as saturated. Ablation volumes are either the geometric applicator capsule or the CEM43 >= 240 min region of this uncalibrated model, not a validated lesion boundary.
- Brain, CSF and vessel masks are user-supplied or heuristic. No segmentation is validated, no registration is performed, and no named anatomical labels (hippocampus, amygdala, ...) are provided.
- No comparison with experimental or clinical measurements exists.

## Suitable description for the submission

> ThermoSEEG is a computational research prototype for exploring SEEG-guided LITT planning geometry and transient bioheat simulations, including a modeled CSF-dependent cooling term. It provides MRI loading, preliminary heuristic anatomy masks, editable trajectory/target geometry, and comparative temperature and CEM43 visualizations.

Do **not** claim: clinically validated segmentation; outcome prediction or safe-ablation guarantees; experimentally calibrated CSF coefficients; validation against clinical LITT data; automatic lesion identification; a verified registration or electrode-margin workflow; clinical readiness. The browser app is a re-implementation for web demonstration; the PySide6/PyVista desktop workstation is the primary 3D interface and the two are not feature-identical.

## Reference case and test plan

- `docs/reference_case_phantom.json` records the exact synthetic input, geometry, solver parameters and outputs (both presets). Regenerate with `python scripts/make_reference_case.py`; `tests/test_reference_case.py` checks the engine still reproduces it. This is a **regression** reference, not a validation benchmark.
- Before submission run three cases and record screenshots, failures and limitations: **A** synthetic phantom (geometry, coordinates, qualitative thermal behaviour); **B** a skull-stripped template (NIfTI loading, world-coordinate correspondence, visualisation); **C** a T1w volume **with an externally generated brain mask** (only after provenance and licence are confirmed). For each, check coordinates/orientation, mask quality (`python scripts/check_masks.py ...` plus visual inspection), target and trajectory placement, active-segment position, hotspot location, with/without-CSF difference, live recomputation, temperature and CEM43 updates, and readability of the dark theme in the demo browser.

## Supported inputs

- JPG, JPEG, PNG raster slices.
- Single-frame, single-channel, monochrome DICOM MR or CT slices. MR is the intended input; CT is decoded only, and requires explicit acknowledgement because the ROI heuristic assumes MRI-like contrast and is not validated for CT.
- DICOM Modality LUT or Rescale Slope/Intercept is applied before VOI LUT/window when present. The resulting display values are then scaled per slice to 8-bit for the existing segmentation pipeline; they are not quantitative tissue intensities; original HU or MR signal values are not preserved.
- Uploads are limited to 50 MB, 4096 pixels per image side, and 16 million pixels before full pixel decoding. Color, multi-frame, non-MR/CT, empty, constant-intensity, malformed, or unsupported DICOM data is rejected with an error message. CT analysis requires acknowledgement that MRI-oriented ROI heuristics are unvalidated for CT.
- Pixel Spacing is read as `[row, column]` and reported as horizontal/vertical spacing. Orientation and position are validated and exported as metadata but are not used to construct 3D coordinates.

## Analysis outline

1. Decode image data and create an 8-bit display representation.
2. Build a heuristic foreground mask using Otsu thresholding, morphology, and largest connected component. This is not a brain segmentation; morphological closing can bridge small gaps. No CSF or tissue classification is performed.
3. Extract high-signal regions by the configured intensity threshold. This is not vessel or functional-anatomy segmentation.
4. Evaluate a 2D raster trajectory, pixel-based risk clearance, original ROI containment, and a separately reported tolerated ROI.
5. Vary only the risk threshold in three sensitivity cases while holding the ROI fixed. This stability measure does not measure accuracy, mask quality, or data reliability.
6. In LITT mode, render an uncalibrated static Gaussian demonstration. Duration scales synthetic CEM43 only. If the numerical guard exceeds 100 °C, CEM43 and dose area are suppressed; the displayed temperature field is clipped at the documented display ceiling.

## Units and interpretation

- Path length and minimum Euclidean clearance in millimeters use horizontal and vertical Pixel Spacing when available or an explicitly entered isotropic manual spacing.
- The operative clearance thresholds, risk dilation, candidate score, contact placement and thermal kernels remain pixel-based experimental settings. Therefore metric distances do not imply calibrated physical safety thresholds.
- SEEG contact fit is geometric only. The displayed contact span in millimeters is measured between rounded contact coordinates.
- CEM43 inputs derive from an unvalidated synthetic temperature field. The 240-minute reference threshold is not universal and has no patient-specific interpretation here.
- “Heuristic checks passed” is a software status only. It does not mean safe, valid anatomy, or clinical suitability.

## Exports and privacy

The app offers JSON, annotated PNG, and PDF research reports. JSON records the image SHA-256, image dimensions, processing settings, software versions, timestamps, spacing source, sensitivity scenarios, candidate-search audit, and known scope limitations. Patient identifiers are not added to exports. Uploaded bytes are not intentionally saved to disk, but image arrays can remain in process memory and the Streamlit data cache for up to five minutes (and two processed images or four distance maps) until eviction or process restart; operate only under the team's data-governance rules.

The JSON is a run record and its `reproducibility.configuration` section can be loaded from the sidebar. Import restores supported slider/checkbox values and trajectory points only when the uploaded image SHA-256 matches; mismatched image points are not restored. Reproduction still requires the same image bytes, compatible pinned library environment, and software version.

## Automated tests

```powershell
python -m unittest discover -s tests -v
```

`tests/test_thermoseeg_3d.py` covers the 3D engine (energy conservation without perfusion, analytic steady state with perfusion, monotonic response to power and perfusion, CEM43 reference values, dose suppression above 100 C, trajectory and mesh geometry). The Streamlit/Plotly interface and NIfTI loading need environment-backed smoke tests.

The suite covers core geometry, mask extraction, target validation, contact placement, thermal dose masking and numerical-guard suppression, exported-config edge cases, plus generated synthetic DICOM ingestion. It is not a clinical validation study. Full Streamlit interaction, PDF visual rendering, large-image performance, and broad real-world DICOM vendor compatibility still require environment-backed testing.

## Scientific and engineering limitations

- One 2D slice only; no 3D spatial reconstruction or inter-slice anatomy.
- ROI and risk masks are heuristic intensity masks, not anatomy.
- No vessel, CSF, eloquent cortex, tissue property, perfusion, or patient-specific thermal model.
- No transient bioheat simulation, empirical calibration, external reference comparison, or clinical validation.
- Sensitivity has only three threshold cases and a fixed ROI.
- Auto-search checks a finite sampled set with local pixel refinement. Its score is an experimental ranking rule, not a global optimum. Exported counters distinguish proposed candidates from full path evaluations and unique evaluated coordinates.
- DICOM spatial tags are recorded but only Pixel Spacing contributes to the 2D metric calculations. Presentation LUT and specialized multi-frame/colored objects are not supported.
- Python/package versions are captured per export. Determinism is not claimed: JSON records `determinism.claim = not_established`; neither repeat-run nor cross-environment reproducibility has been established. SHA-256 is used for full image identities, session keys, and signatures; output filenames show its first 16 hex characters. Local and UTC timestamps derive from one instant. Clearance thresholds are experimental pixel values, not clinical safety limits.

Experimental calibration, anatomical ground truth, independent validation data, and expert review are responsibilities of the research team and are not claimed by this software.