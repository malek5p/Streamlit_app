"""
ThermoSEEG_RC_UI.py
===================
Trajectory Planning Prototype with a Simplified Static Gaussian Thermal Approximation
(Release Candidate v5.1, UI edition)

This research prototype provides 2D heuristic image-mask analysis, geometric checks,
threshold sensitivity, candidate search, and an uncalibrated synthetic thermal demonstration.

Architecture
------------
1. Heuristic foreground ROI (Otsu + opening + largest component + closing + outer-contour hole fill)
   is only a heuristic geometric mask, not brain-tissue segmentation. Original and tolerated masks are reported separately; the closing step may bridge small gaps.
2. High-signal risk regions = top X% brightest pixels INSIDE the ROI (intensity heuristic,
   NOT an anatomical segmentation of vessels, CSF or eloquent cortex).
3. Dual-Mask System: raw risk mask (thermal attenuation) vs. dilated mask (collision).
4. Single threshold source (_derive_threshold) shared by base run and sensitivity analysis; sensitivity holds ROI fixed.
5. Sensitivity engine: perturbs the threshold with a FIXED ROI; verdict stability first,
   clearance variability second.
6. Candidate search: samples the ROI edge and applies the geometric checks used by the validator
   (inside ROI, outside risk mask, minimum clearance, path stays inside ROI).
7. Thermal model: simplified STATIC Gaussian approximation parameterised by laser power.
   Duration scales the CEM43 dose only. It is NOT a prediction of actual tissue temperature.
   "Thermal attenuation" is applied around intensity-based risk regions (no CSF/vessel model).
8. Live navigation simulation isolated in an st.fragment.
9. Click selection uses a nonce so stale clicks are never re-consumed.
"""

import hashlib
from io import BytesIO
import json
import logging
import platform
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version as package_version
from typing import Tuple, List, Dict, Any, Optional

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from scipy.ndimage import distance_transform_edt, gaussian_filter
from thermoseeg_dicom import (decode_dicom_slice, inspect_dicom_modality,
                              MAX_UPLOAD_BYTES, MAX_IMAGE_SIDE, MAX_IMAGE_PIXELS)
from thermoseeg_config import RunConfigError, parse_exported_run_config, configuration_widget_values

from thermoseeg_theme import (
    inject_theme, render_topbar, section_header, eyebrow, kpi_tiles, pill_html,
    sidebar_brand, STATUS_TONE, CONFIDENCE_TONE,
)

try:
    from streamlit_image_coordinates import streamlit_image_coordinates
    CLICK_SELECTION_AVAILABLE = True
except ImportError:
    CLICK_SELECTION_AVAILABLE = False

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, LongTable, TableStyle
    PDF_EXPORT_AVAILABLE = True
except ImportError:
    PDF_EXPORT_AVAILABLE = False


# ============================================================================
# SECTION 0: CONSTANTS & CONFIGURATION
# ============================================================================

# --- Physics & Biology Constants ---
BASELINE_BODY_TEMP_C = 37.0
CEM43_THRESHOLD_TEMP_C = 43.0
CEM43_R_HIGH = 0.5
CEM43_R_LOW = 0.25
CEM43_ABLATION_THRESHOLD_MIN = 240.0   # Experimental reference threshold; not universal
LITT_GAUSSIAN_SIGMA_BASE_PX = 8.0
LITT_POWER_TO_DELTA_T_FACTOR = 4.8
RISK_ATTENUATION_FACTOR = 0.30         # attenuation near intensity-based risk regions
RISK_ATTENUATION_SIGMA_PX = 4.0
THERMAL_NUMERICAL_GUARD_C = 100.0            # Legacy numerical guard only; not a validated model boundary
TEMP_DISPLAY_MAX_C = 80.0
CEM43_DISPLAY_MAX_MIN = 480.0          # Heatmap saturation so huge doses stay readable
# Slider is restricted to the legacy numerical guard; this is not a physical operating limit.
MAX_LASER_POWER_W = 13.0

# --- Heuristic Geometry Constants (experimental pixel values; not clinical safety limits) ---
HEURISTIC_MIN_CLEARANCE_PX = 10.0
HEURISTIC_WARNING_CLEARANCE_PX = 25.0
HEURISTIC_CLEARANCE_SCORE_CAP_PX = 60.0
DEFAULT_DILATION_RADIUS_PX = 5
ROI_ENTRY_EDGE_TOL_PX = 6.0     # entry deeper than this inside the ROI triggers a warning
ROI_PATH_TOLERANCE_PX = 2       # ROI dilation used for path containment (edge pixels / rasterisation)

# --- Segmentation (heuristic) ---
RISK_PERCENTILE_DEFAULT = 97    # heuristic starting point, NOT a scientific constant
ROI_CLOSE_KERNEL_PX = 9
ROI_STRIP_RADIUS_DEFAULT_PX = 7   # opening radius that removes thin scalp/skull rings
RISK_MASK_MAX_FRACTION = 0.15     # risk mask above this share of the ROI = threshold sits on a tissue plateau

# --- UI Defaults ---
DEFAULT_LASER_POWER_W = 10.0
DEFAULT_ABLATION_TIME_SEC = 30.0
DEFAULT_SEEG_CONTACTS_COUNT = 8
DEFAULT_SEEG_SPACING_PX = 10.0

# --- Sensitivity Analysis Parameters ---
SENSITIVITY_THRESHOLD_SHIFT = 5

# --- Bounded image input ---
MAX_IMAGE_DIM_PX = 4096
MAX_IMAGE_PIXELS = 16_000_000

SCIENTIFIC_LIMITATIONS = (
    "2D single-slice analysis; no 3D reconstruction.",
    "ROI is a heuristic foreground mask, not validated brain-tissue segmentation.",
    "Risk regions are intensity-based and do not identify vessels or functional anatomy.",
    "Thermal model is an uncalibrated static Gaussian approximation, not a bioheat model.",
    "No experimental calibration, external reference comparison, or clinical validation.",
    "CT is decoded but the ROI and risk heuristics are not validated for CT.",
)

SEARCH_COUNT_KEYS = (
    "initial_candidates_generated", "initial_candidates_unique", "initial_path_evaluations",
    "initial_candidates_accepted", "refinement_path_evaluations", "refinement_unique_new_coordinates",
    "refinement_candidates_accepted", "total_path_evaluations", "unique_coordinates_path_evaluated",
)

TOOL_VERSION = "ThermoSEEG_RC_v5.1-research"
logger = logging.getLogger("thermoseeg")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


# ============================================================================
# SECTION 1: TYPE DEFINITIONS & HELPERS
# ============================================================================

PointTuple = Tuple[int, int]
MaskArray = np.ndarray
FloatArray = np.ndarray


def _streamlit_version() -> Tuple[int, int]:
    parts = st.__version__.split(".")[:2]
    out = []
    for p in parts:
        digits = "".join(ch for ch in p if ch.isdigit())
        out.append(int(digits) if digits else 0)
    while len(out) < 2:
        out.append(0)
    return out[0], out[1]


# Newer Streamlit deprecates use_container_width in favour of width="stretch".
STRETCH: Dict[str, Any] = (
    {"width": "stretch"} if _streamlit_version() >= (1, 50) else {"use_container_width": True}
)


def clamp(value: float, min_val: float, max_val: float) -> float:
    return max(min_val, min(max_val, value))


CEM43_SATURATION_MIN = 1.0e6   # beyond this the synthetic dose number carries no interpretable meaning


def format_cem43(value: float) -> str:
    """Display helper: very large synthetic doses are shown as saturated instead of 1e12-1e17."""
    return f">{CEM43_SATURATION_MIN:.0e} (saturated)" if value > CEM43_SATURATION_MIN else f"{value:.2f}"


def _roi_heuristic_modality_description(dicom_metadata: Optional[Dict[str, Any]]) -> str:
    """Describe intended contrast without assuming every raster has DICOM metadata."""
    metadata = dicom_metadata if isinstance(dicom_metadata, dict) else {}
    if metadata.get("modality") == "CT":
        return "MRI-like contrast; CT use is unvalidated"
    if not metadata:
        return "MRI-like contrast; raster modality unverified"
    return "MRI-like contrast"


def _library_versions() -> Dict[str, Optional[str]]:
    names = {"streamlit": "streamlit", "numpy": "numpy", "opencv": "opencv-python-headless",
             "pandas": "pandas", "scipy": "scipy", "pillow": "Pillow", "pydicom": "pydicom",
             "reportlab": "reportlab", "streamlit_image_coordinates": "streamlit-image-coordinates"}
    out: Dict[str, Optional[str]] = {}
    for label, dist in names.items():
        try:
            out[label] = package_version(dist)
        except PackageNotFoundError:
            fallback = "opencv-python" if label == "opencv" else None
            if fallback:
                try:
                    out[label] = package_version(fallback)
                except PackageNotFoundError:
                    out[label] = None
            else:
                out[label] = None
    return out


def _build_pdf_report(report_data: Dict[str, Any]) -> Optional[bytes]:
    if not PDF_EXPORT_AVAILABLE:
        return None
    from html import escape
    stream = BytesIO()
    doc = SimpleDocTemplate(stream, pagesize=letter, title="ThermoSEEG Research Report")
    styles = getSampleStyleSheet()
    body = styles["BodyText"]
    body.fontSize = 8
    body.leading = 10
    story = [Paragraph("ThermoSEEG | 2D Research Prototype Report", styles["Title"]),
             Paragraph("Not clinically validated. Not for clinical decisions or surgical guidance.", styles["BodyText"]), Spacer(1, 12)]
    rows = [[Paragraph("<b>Field</b>", body), Paragraph("<b>Value</b>", body)]]
    def add(section: str, mapping: Dict[str, Any]) -> None:
        rows.append([Paragraph(f"<b>{escape(section)}</b>", styles["Heading2"]), ""])
        for key, value in mapping.items():
            if isinstance(value, (dict, list)):
                value = json.dumps(value, ensure_ascii=False, default=str)
            rows.append([Paragraph(escape(str(key)), body), Paragraph(escape(str(value)), body)])
    add("Metadata", report_data.get("metadata", {}))
    add("Reproducibility Settings", report_data.get("reproducibility", {}))
    add("Image Processing", report_data.get("image_processing", {}))
    add("Selected Points", report_data.get("points", {}))
    add("Trajectory Analysis", report_data.get("trajectory_analysis", {}))
    add("Model Limitations", report_data.get("model_specific_summary", {}))
    add("Scientific Limitations", {f"{i + 1}": item for i, item in enumerate(SCIENTIFIC_LIMITATIONS)})
    table = LongTable(rows, colWidths=[155, 300], hAlign="LEFT", repeatRows=1, splitByRow=1, splitInRow=1)
    table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey), ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(table)
    story.append(Paragraph("Scientific limits: heuristic image masks; no clinical validation, anatomical segmentation, 3D reconstruction, patient-specific thermal calibration, or safety determination.", body))
    def footer(canvas, document):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.drawString(36, 22, "Research prototype only — not for clinical decisions or surgical guidance.")
        canvas.drawRightString(letter[0] - 36, 22, f"Page {document.page}")
        canvas.restoreState()
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return stream.getvalue()


def get_status_color(status: str) -> Tuple[int, int, int]:
    colors = {
        "HEURISTIC_CHECKS_PASSED": (0, 150, 100),
        "WARNING": (0, 215, 255),
        "DANGER": (0, 0, 255),
        "ERROR": (255, 0, 0),
    }
    return colors.get(status, (255, 255, 255))


def generate_config_signature(
    file_hash: str,
    use_otsu: bool,
    manual_thresh: int,
    risk_percentile: float,
    roi_strip_px: int,
    min_area: int,
    dilation_rad: int,
    target_pt: Optional[PointTuple],
    entry_pt: Optional[PointTuple],
    proc_mode: str,
    laser_power: float,
    abl_time: float,
    num_contacts: int,
    spacing_px: float,
) -> str:
    sig_str = (
        f"{file_hash}|{use_otsu}|{manual_thresh}|{risk_percentile}|{roi_strip_px}|{min_area}|{dilation_rad}|"
        f"{target_pt}|{entry_pt}|{proc_mode}|{laser_power}|{abl_time}|"
        f"{num_contacts}|{spacing_px}"
    )
    return hashlib.sha256(sig_str.encode()).hexdigest()


# ============================================================================
# SECTION 2: GEOMETRY & TRAJECTORY HELPERS
# ============================================================================

def bresenham_line(p1: PointTuple, p2: PointTuple) -> List[PointTuple]:
    x1, y1 = p1
    x2, y2 = p2
    points = []
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy

    x, y = x1, y1
    while True:
        points.append((x, y))
        if x == x2 and y == y2:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
    return points


def calculate_geometry(entry: PointTuple, target: PointTuple, spacing_xy: Optional[Tuple[float, float]] = None) -> Dict[str, float]:
    ex, ey = entry
    tx, ty = target
    length = float(np.hypot(tx - ex, ty - ey))
    angle_rad = np.arctan2(ty - ey, tx - ex)
    angle_deg = float(np.degrees(angle_rad))
    result = {"length": length, "angle": angle_deg}
    if spacing_xy:
        sx, sy = spacing_xy
        result["length_mm"] = float(np.hypot((tx - ex) * sx, (ty - ey) * sy))
        result["angle_physical_deg"] = float(np.degrees(np.arctan2((ty-ey)*sy, (tx-ex)*sx)))
    return result


def _in_bounds_xy(line_points: List[PointTuple], w: int, h: int) -> Tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(line_points, dtype=int)
    if pts.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    xs, ys = pts[:, 0], pts[:, 1]
    ok = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    return xs[ok], ys[ok]


def check_collision(mask: MaskArray, line_points: List[PointTuple]) -> bool:
    h, w = mask.shape
    xs, ys = _in_bounds_xy(line_points, w, h)
    if xs.size == 0:
        return False
    return bool(np.any(mask[ys, xs] > 0))


def min_clearance_along_line(dist_map: FloatArray, line_points: List[PointTuple]) -> float:
    h, w = dist_map.shape
    xs, ys = _in_bounds_xy(line_points, w, h)
    if xs.size == 0:
        return 0.0
    return float(dist_map[ys, xs].min())


def check_roi_containment(roi_allowed: MaskArray, line_points: List[PointTuple]) -> int:
    """Number of trajectory pixels that fall OUTSIDE the allowed ROI (out-of-image counts too)."""
    h, w = roi_allowed.shape
    pts = np.asarray(line_points, dtype=int)
    if pts.size == 0:
        return 0
    xs, ys = pts[:, 0], pts[:, 1]
    inside_img = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    outside = int(np.count_nonzero(~inside_img))
    outside += int(np.count_nonzero(roi_allowed[ys[inside_img], xs[inside_img]] == 0))
    return outside


def build_cumulative_depths(points: List[PointTuple]) -> List[float]:
    depths = [0.0]
    for i in range(1, len(points)):
        x1, y1 = points[i - 1]
        x2, y2 = points[i]
        depths.append(depths[-1] + float(np.hypot(x2 - x1, y2 - y1)))
    return depths


def get_telemetry(
    points: List[PointTuple],
    dist_map: FloatArray,
    idx: int,
    cum_depths: List[float],
) -> Dict[str, Any]:
    idx = int(clamp(int(idx), 0, len(points) - 1))
    x, y = points[idx]
    total_len = cum_depths[-1] if cum_depths[-1] > 0 else 1.0
    curr_depth = cum_depths[idx]
    prog = (curr_depth / total_len) * 100

    h, w = dist_map.shape
    d_to_struct = float(dist_map[y, x]) if (0 <= x < w and 0 <= y < h) else 0.0

    return {
        "position": (int(x), int(y)),
        "current_depth": curr_depth,
        "total_length": total_len,
        "progress_pct": round(prog, 1),
        "distance_to_structure": d_to_struct,
        "proximity_alert": d_to_struct < HEURISTIC_MIN_CLEARANCE_PX,
    }


# ============================================================================
# SECTION 3: SEGMENTATION (Brain ROI + High-Signal Risk Regions)
# ============================================================================

def _brain_roi(
    gray_blurred: np.ndarray, strip_radius_px: int = ROI_STRIP_RADIUS_DEFAULT_PX
) -> np.ndarray:
    """
    Heuristic foreground ROI extraction:
      Otsu -> opening -> largest component -> closing -> outer-contour hole fill; closing may bridge small gaps.
    Opening runs BEFORE closing on purpose: closing first would bridge the dark skull gap
    and fuse scalp fat to the brain. Defines the ALLOWED region (not an obstacle).
    NOT a validated brain segmentation; the largest component is not guaranteed to be
    brain tissue, and the approach assumes an MRI-like contrast (CT bone is not handled).
    """
    _, roi = cv2.threshold(gray_blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if strip_radius_px > 0:
        k = int(strip_radius_px) * 2 + 1
        roi = cv2.morphologyEx(roi, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))

    kernel = np.ones((ROI_CLOSE_KERNEL_PX, ROI_CLOSE_KERNEL_PX), np.uint8)
    roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel)

    n, lab, stats, _ = cv2.connectedComponentsWithStats(roi, connectivity=8)
    if n > 1:
        big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        roi = np.where(lab == big, 255, 0).astype(np.uint8)

    # Fill internal holes (dark ventricles/sulci) so the ROI is one solid allowed region.
    # Filling is appropriate HERE (ROI), unlike the obstacle mask.
    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(roi)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return filled


def _derive_threshold(
    gray_blurred: np.ndarray,
    roi: np.ndarray,
    use_otsu: bool,
    manual_threshold: int,
    risk_percentile: float,
) -> float:
    """Single source of truth for the risk threshold (base run AND sensitivity)."""
    if not use_otsu:
        return float(manual_threshold)
    vals = gray_blurred[roi > 0]
    if vals.size == 0:
        return 255.0
    return float(np.percentile(vals, risk_percentile))


def _compute_masks_core(
    gray_blurred: np.ndarray,
    use_otsu: bool,
    manual_threshold: int,
    min_area: int,
    dilation_radius: int,
    risk_percentile: float = RISK_PERCENTILE_DEFAULT,
    roi: Optional[np.ndarray] = None,
    threshold_override: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (raw_mask, dilated_mask).
    raw_mask: high-signal risk regions inside the brain ROI (thermal physics).
    dilated_mask: buffered risk regions (trajectory heuristic / collision).
    `threshold_override` lets the sensitivity analysis shift the threshold while
    keeping the exact same ROI.
    """
    if roi is None:
        roi = _brain_roi(gray_blurred)

    if threshold_override is not None:
        t = float(threshold_override)
    else:
        t = _derive_threshold(gray_blurred, roi, use_otsu, manual_threshold, risk_percentile)

    binary = np.where((gray_blurred >= t) & (roi > 0), 255, 0).astype(np.uint8)

    # Area filter via connected components (no contour filling -> no solid-ring problem).
    n, lab, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned = np.zeros_like(binary)
    if n > 1:
        keep = np.where(stats[1:, cv2.CC_STAT_AREA] >= min_area)[0] + 1
        if keep.size:
            cleaned[np.isin(lab, keep)] = 255

    if dilation_radius > 0:
        k_size = dilation_radius * 2 + 1
        kernel = np.ones((k_size, k_size), np.uint8)
        dilated = cv2.dilate(cleaned, kernel, iterations=1)
    else:
        dilated = cleaned.copy()

    return cleaned, dilated


@st.cache_data(show_spinner=False, ttl=300, max_entries=2)
def load_and_preprocess_image(
    file_bytes: bytes,
    threshold_value: int,
    use_otsu: bool,
    risk_percentile: float,
    min_area: int,
    dilation_radius: int,
    roi_strip_px: int,
) -> Dict[str, Any]:
    started = time.perf_counter()
    try:
        if not file_bytes:
            raise ValueError("The uploaded file is empty.")
        if len(file_bytes) > MAX_UPLOAD_BYTES:
            raise ValueError("Upload exceeds the 50 MB file-size limit.")
        dicom_data = decode_dicom_slice(file_bytes, max_pixels=MAX_IMAGE_PIXELS)
        pixel_spacing_mm = dicom_data["pixel_spacing_mm_xy"] if dicom_data else None
        dicom_metadata = dicom_data["metadata"] if dicom_data else None
        if dicom_data:
            original = cv2.cvtColor(dicom_data["gray"], cv2.COLOR_GRAY2BGR)
        else:
            try:
                with Image.open(BytesIO(file_bytes)) as image_header:
                    image_width, image_height = image_header.size
            except Exception as exc:
                raise ValueError("Invalid or unsupported raster image format.") from exc
            if image_width < 2 or image_height < 2 or image_width > MAX_IMAGE_DIM_PX or image_height > MAX_IMAGE_DIM_PX or image_width * image_height > MAX_IMAGE_PIXELS:
                raise ValueError("Image dimensions exceed the supported limit (4096 px per side and 16 million pixels).")
            file_array = np.frombuffer(file_bytes, dtype=np.uint8)
            original = cv2.imdecode(file_array, cv2.IMREAD_COLOR)
        if original is None:
            raise ValueError("Invalid image format.")
        if original.ndim != 3 or original.shape[0] < 2 or original.shape[1] < 2:
            raise ValueError("Image dimensions are invalid or too small for analysis.")
        if original.shape[0] > MAX_IMAGE_DIM_PX or original.shape[1] > MAX_IMAGE_DIM_PX or original.shape[0] * original.shape[1] > MAX_IMAGE_PIXELS:
            raise ValueError("Decoded image dimensions exceed the supported limit (4096 px per side and 16 million pixels).")

        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        if int(gray.min()) == int(gray.max()):
            raise ValueError("The image has constant intensity and cannot produce a meaningful threshold mask.")
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        roi = _brain_roi(blurred, roi_strip_px)
        if not np.any(roi):
            raise ValueError("No foreground region was detected. Check image contrast and ROI strip radius.")
        raw_mask, dilated_mask = _compute_masks_core(
            blurred, use_otsu, threshold_value, min_area, dilation_radius,
            risk_percentile=risk_percentile, roi=roi,
        )

        k_roi = ROI_PATH_TOLERANCE_PX * 2 + 1
        roi_allowed = cv2.dilate(roi, np.ones((k_roi, k_roi), np.uint8))
        roi_depth = distance_transform_edt(roi > 0)   # px inside the ROI (0 outside)

        return {
            "original": original,
            "gray": gray,
            "blurred": blurred,
            "roi": roi,
            "roi_allowed": roi_allowed,
            "pixel_spacing_mm_xy": pixel_spacing_mm,
            "dicom_metadata": dicom_metadata,
            "roi_depth": roi_depth,
            "raw_structure_mask": raw_mask,   # thermal physics
            "structure_mask": dilated_mask,   # safety / collision
            "processing_seconds": round(time.perf_counter() - started, 4),
        }
    except Exception as e:
        logger.exception("Image processing failed (%s bytes)", len(file_bytes))
        st.error(f"Image could not be analyzed: {e}")
        return {}


def _distance_map(structure_mask: MaskArray) -> FloatArray:
    if structure_mask.max() == 0:
        h, w = structure_mask.shape
        return np.full((h, w), float(np.hypot(h, w)))
    return distance_transform_edt(~(structure_mask > 0))


@st.cache_data(show_spinner=False, ttl=300, max_entries=4)
def compute_distance_map_cached(structure_mask: MaskArray, sampling=(1.0, 1.0)) -> FloatArray:
    """Euclidean distance in pixel units, optionally accounting for anisotropic spacing."""
    if structure_mask.max() == 0:
        h, w = structure_mask.shape
        return np.full((h, w), float(np.hypot(h * sampling[0], w * sampling[1])))
    return distance_transform_edt(~(structure_mask > 0), sampling=sampling)


# ============================================================================
# SECTION 4: SENSITIVITY ANALYSIS ENGINE (Collision Aware)
# ============================================================================

def perform_sensitivity_analysis(
    blurred_gray: np.ndarray,
    roi: np.ndarray,
    base_manual_threshold: int,
    use_otsu: bool,
    risk_percentile: float,
    min_area: int,
    dilation_radius: int,
    entry: PointTuple,
    target: PointTuple,
    leaves_roi: bool = False,
) -> Dict[str, Any]:
    """
    Tests stability of the trajectory under threshold perturbations (T-shift, T, T+shift)
    with the ROI held fixed.

    The stability index reflects whether the *heuristic verdict* changes under this limited test:
      - collision appears/disappears across scenarios  -> LOW
      - heuristic status changes                           -> MODERATE
      - otherwise graded by clearance coefficient of variation.
    A consistently colliding trajectory is stable (and already flagged DANGER),
    so it is not penalised here.
    """
    base_thresh = _derive_threshold(
        blurred_gray, roi, use_otsu, base_manual_threshold, risk_percentile
    )

    thresholds_to_test = [
        base_thresh - SENSITIVITY_THRESHOLD_SHIFT,
        base_thresh,
        base_thresh + SENSITIVITY_THRESHOLD_SHIFT,
    ]

    line_pts = bresenham_line(entry, target)
    clearances: List[float] = []
    statuses: List[str] = []
    collision_flags: List[bool] = []
    risk_fractions: List[float] = []
    roi_px = max(1, int(np.count_nonzero(roi)))

    for t_val in thresholds_to_test:
        t_clamped = float(max(0.0, min(255.0, t_val)))

        temp_raw, temp_dilated = _compute_masks_core(
            blurred_gray, False, 0, min_area, dilation_radius,
            roi=roi, threshold_override=t_clamped,
        )
        risk_fractions.append(float(np.count_nonzero(temp_raw)) / roi_px)
        temp_dist = _distance_map(temp_dilated)

        intersects = check_collision(temp_dilated, line_pts)
        min_d = min_clearance_along_line(temp_dist, line_pts)

        clearances.append(min_d)
        collision_flags.append(intersects)

        # Collision and DANGER are the same status category in the main evaluator.
        if intersects or leaves_roi or min_d < HEURISTIC_MIN_CLEARANCE_PX:
            statuses.append("DANGER")
        elif min_d < HEURISTIC_WARNING_CLEARANCE_PX:
            statuses.append("WARNING")
        else:
            statuses.append("HEURISTIC_CHECKS_PASSED")

    mean_clearance = float(np.mean(clearances))
    std_clearance = float(np.std(clearances))

    if mean_clearance < 1e-6:
        cv_score = 0.0 if std_clearance < 1e-6 else 1.0
    else:
        cv_score = std_clearance / mean_clearance

    collision_flip = len(set(collision_flags)) > 1
    # The mask itself is unreliable if a +-shift makes it swallow a large part of the ROI
    # (threshold sits on the tissue intensity plateau), regardless of this one trajectory.
    max_risk_fraction = max(risk_fractions) if risk_fractions else 0.0
    mask_unstable = max_risk_fraction > RISK_MASK_MAX_FRACTION
    status_flip = len(set(statuses)) > 1

    if collision_flip or mask_unstable:
        conf_label = "LOW"
    elif status_flip:
        conf_label = "MODERATE"
    elif cv_score < 0.1:
        conf_label = "HIGH"
    elif cv_score < 0.3:
        conf_label = "MODERATE"
    else:
        conf_label = "LOW"

    return {
        "sensitivity_stability_index": conf_label,
        "variability_score": round(float(cv_score), 3),
        "mean_stable_clearance": round(mean_clearance, 2),
        "status_flipped": status_flip,
        "collision_flipped": collision_flip,
        "mask_unstable": mask_unstable,
        "max_risk_fraction": round(max_risk_fraction, 3),
        "scenario_statuses": statuses,
        "scenarios": [
            {"threshold": round(float(max(0.0, min(255.0, t))), 3),
             "clearance_px": round(float(clearances[i]), 3),
             "collision": bool(collision_flags[i]),
             "risk_fraction": round(float(risk_fractions[i]), 5),
             "heuristic_status": statuses[i]}
            for i, t in enumerate(thresholds_to_test)
        ],
    }


# ============================================================================
# SECTION 5: VALIDATION & SAFETY ENGINE
# ============================================================================

def validate_target(
    target: PointTuple, structure_mask: MaskArray, roi: MaskArray, w: int, h: int
) -> Tuple[bool, str]:
    if not (0 <= target[0] < w and 0 <= target[1] < h):
        return False, "Target out of bounds."
    if roi[target[1], target[0]] == 0:
        return False, "Target outside the heuristic brain ROI."
    if structure_mask[target[1], target[0]] > 0:
        return False, "Target inside high-signal risk region."
    return True, "Valid Target."


def validate_entry(
    entry: PointTuple,
    structure_mask: MaskArray,
    dist_map: FloatArray,
    roi_allowed: MaskArray,
    roi_depth: FloatArray,
    w: int,
    h: int,
) -> Tuple[bool, str, bool]:
    """Returns (is_valid, message, is_deep_inside_roi_warning)."""
    if not (0 <= entry[0] < w and 0 <= entry[1] < h):
        return False, "Entry out of bounds.", False

    if roi_allowed[entry[1], entry[0]] == 0:
        return False, "Entry outside the heuristic brain ROI.", False

    if structure_mask[entry[1], entry[0]] > 0:
        return False, "Entry inside high-signal risk region.", False

    d = float(dist_map[entry[1], entry[0]])
    if d < HEURISTIC_MIN_CLEARANCE_PX:
        return False, f"Entry too close to risk region ({d:.1f}px < {HEURISTIC_MIN_CLEARANCE_PX:.0f}px).", False

    depth = float(roi_depth[entry[1], entry[0]])
    if depth > ROI_ENTRY_EDGE_TOL_PX:
        return True, f"Warning: Entry is {depth:.0f}px inside the brain ROI (not on its edge).", True

    return True, "", False


def evaluate_safety_metrics(
    min_dist_px: float,
    intersects: bool,
    leaves_roi: bool,
    blurred_gray: np.ndarray,
    roi: np.ndarray,
    base_manual_threshold: int,
    use_otsu: bool,
    risk_percentile: float,
    min_area: int,
    dilation_radius: int,
    entry: PointTuple,
    target: PointTuple,
) -> Dict[str, Any]:
    """
    Status is DANGER on collision, ROI violation, or clearance < DANGER threshold.
    clearance_score_pct is a DISTANCE-based score (min clearance / HEURISTIC_CLEARANCE_SCORE_CAP_PX),
    NOT a probability of safety.
    """
    if intersects or leaves_roi or min_dist_px < HEURISTIC_MIN_CLEARANCE_PX:
        status = "DANGER"
    elif min_dist_px < HEURISTIC_WARNING_CLEARANCE_PX:
        status = "WARNING"
    else:
        status = "HEURISTIC_CHECKS_PASSED"

    clearance_score = clamp((min_dist_px / HEURISTIC_CLEARANCE_SCORE_CAP_PX) * 100, 0, 100)
    if intersects or min_dist_px < HEURISTIC_MIN_CLEARANCE_PX:
        clearance_score = 0.0

    sens_res = perform_sensitivity_analysis(
        blurred_gray, roi, base_manual_threshold, use_otsu, risk_percentile,
        min_area, dilation_radius, entry, target, leaves_roi=leaves_roi,
    )

    return {
        "status": status,
        "clearance_score_pct": round(float(clearance_score), 1),
        "min_distance_px": round(float(min_dist_px), 2),
        "intersects": bool(intersects),
        "leaves_roi": bool(leaves_roi),
        "sensitivity_stability_index": sens_res["sensitivity_stability_index"],
        "variability_score": sens_res["variability_score"],
        "status_flipped": sens_res["status_flipped"],
        "collision_flipped": sens_res["collision_flipped"],
        "mask_unstable": sens_res["mask_unstable"],
        "max_risk_fraction": sens_res["max_risk_fraction"],
        "sensitivity_scenarios": sens_res["scenarios"],
        "color_bgr": get_status_color(status),
    }


# ============================================================================
# SECTION 6: LITT THERMAL MODEL & SEEG PLANNING
# ============================================================================

def simulate_litt_thermal_field(
    target: PointTuple,
    raw_structure_mask: MaskArray,
    tissue_mask: MaskArray,
    power_watts: float,
    duration_sec: float,
) -> Dict[str, Any]:
    """
    Simplified STATIC Gaussian thermal approximation parameterised by laser power.
    Duration affects the CEM43 thermal dose only, NOT the spatial temperature field.
    This is not a transient Pennes bio-heat simulation and does not predict actual
    tissue temperature. Attenuation is applied around intensity-based risk regions only
    (there is no CSF or vessel model).
    """
    h, w = raw_structure_mask.shape
    y_coords, x_coords = np.ogrid[:h, :w]
    dist_from_target = np.hypot(x_coords - target[0], y_coords - target[1])

    sigma = LITT_GAUSSIAN_SIGMA_BASE_PX + (power_watts * 0.4)
    peak_delta_t = power_watts * LITT_POWER_TO_DELTA_T_FACTOR

    raw_delta_t = peak_delta_t * np.exp(-(dist_from_target ** 2) / (2 * sigma ** 2))

    # Attenuation around the raw intensity-based risk regions (not safety buffers).
    risk_influence = gaussian_filter(
        (raw_structure_mask > 0).astype(float), sigma=RISK_ATTENUATION_SIGMA_PX
    )
    max_inf = risk_influence.max()
    risk_influence_norm = risk_influence / max_inf if max_inf > 0 else np.zeros_like(risk_influence)

    reduction_factor = 1.0 - (RISK_ATTENUATION_FACTOR * risk_influence_norm)
    adjusted_delta_t = raw_delta_t * reduction_factor

    temp_map = BASELINE_BODY_TEMP_C + adjusted_delta_t
    tissue = tissue_mask > 0
    max_temp = float(np.max(temp_map[tissue])) if np.any(tissue) else float(np.max(temp_map))
    model_valid = max_temp <= THERMAL_NUMERICAL_GUARD_C
    model_warning = ""
    if not model_valid:
        model_warning = (
            f"Synthetic model value {max_temp:.0f} °C exceeds the legacy numerical guard of {THERMAL_NUMERICAL_GUARD_C:.0f} °C. "
            "Derived CEM43 and dose-area values are suppressed because this numerical guard was exceeded."
        )

    temp_phys = temp_map
    temp_display = np.clip(temp_phys, BASELINE_BODY_TEMP_C, TEMP_DISPLAY_MAX_C)

    risk_attenuation_effect = float(np.max(raw_delta_t - adjusted_delta_t))
    if not model_valid:
        # Keep a display-only temperature field, but suppress dose-derived values entirely.
        return {
            "temp_map": temp_display,
            "cem43_map": np.zeros_like(temp_display, dtype=float),
            "max_temp_c": round(max_temp, 2),
            "display_temp_max_c": TEMP_DISPLAY_MAX_C,
            "max_cem43_min": None,
            "mean_cem43_min": None,
            "risk_attenuation_c": round(risk_attenuation_effect, 2),
            "ablation_area_px": None,
            "structure_max_cem43_min": None,
            "numerical_guard_passed": False,
            "dose_valid": False,
            "model_warning": model_warning,
        }

    cem43_map = np.zeros_like(temp_phys, dtype=float)
    minutes = duration_sec / 60.0

    mask_high = temp_phys >= CEM43_THRESHOLD_TEMP_C
    cem43_map[mask_high] = minutes * (CEM43_R_HIGH ** (CEM43_THRESHOLD_TEMP_C - temp_phys[mask_high]))

    mask_mid = (temp_phys >= BASELINE_BODY_TEMP_C) & (temp_phys < CEM43_THRESHOLD_TEMP_C)
    cem43_map[mask_mid] = minutes * (CEM43_R_LOW ** (CEM43_THRESHOLD_TEMP_C - temp_phys[mask_mid]))

    cem43_map[~tissue] = 0.0
    max_cem43 = float(np.max(cem43_map[tissue])) if np.any(tissue) else 0.0
    mean_cem43 = float(np.mean(cem43_map[tissue])) if np.any(tissue) else 0.0
    ablation_area_px = int(np.sum((cem43_map >= CEM43_ABLATION_THRESHOLD_MIN) & tissue))
    structure_pixels = raw_structure_mask > 0
    structure_max_cem43 = float(np.max(cem43_map[structure_pixels])) if np.any(structure_pixels) else 0.0

    return {
        "temp_map": temp_display,
        "cem43_map": cem43_map,
        "max_temp_c": round(max_temp, 2),
        "display_temp_max_c": TEMP_DISPLAY_MAX_C,
        "max_cem43_min": round(max_cem43, 2),
        "mean_cem43_min": round(mean_cem43, 2),
        "risk_attenuation_c": round(risk_attenuation_effect, 2),
        "ablation_area_px": ablation_area_px,
        "structure_max_cem43_min": round(structure_max_cem43, 2),
        "numerical_guard_passed": model_valid,
        "dose_valid": True,
        "model_warning": model_warning,
    }


def compute_seeg_contacts(
    entry: PointTuple,
    target: PointTuple,
    traj_length_px: float,
    num_contacts_req: int,
    spacing_px: float,
) -> Dict[str, Any]:
    ex, ey = entry
    tx, ty = target

    if traj_length_px <= 0:
        return {
            "contacts": [],
            "fits_within_trajectory": False,
            "required_span_px": 0.0,
            "num_contacts_placed": 0,
        }

    length = traj_length_px
    ux, uy = (tx - ex) / length, (ty - ey) / length

    required_span = (num_contacts_req - 1) * spacing_px
    fits = required_span <= length

    contacts = []
    for i in range(num_contacts_req):
        depth = i * spacing_px
        if depth > length:
            break
        cx = tx - ux * depth
        cy = ty - uy * depth
        contacts.append({
            "id": i + 1,
            "pos": (int(round(cx)), int(round(cy))),
            "depth_px": round(depth, 1),
        })

    return {
        "contacts": contacts,
        "fits_within_trajectory": fits,
        "required_span_px": round(required_span, 1),
        "num_contacts_placed": len(contacts),
    }


# ============================================================================
# SECTION 7: AUTO ENTRY OPTIMIZER
# ============================================================================

def _evaluate_entry(
    pt: PointTuple,
    target: PointTuple,
    structure_mask: MaskArray,
    dist_map: FloatArray,
    roi_allowed: MaskArray,
    diagonal: float,
) -> Dict[str, Any]:
    line_pts = bresenham_line(pt, target)
    intersects = check_collision(structure_mask, line_pts)
    leaves_roi = check_roi_containment(roi_allowed, line_pts) > 0
    min_d = min_clearance_along_line(dist_map, line_pts)
    geo = calculate_geometry(pt, target)
    norm_len = geo["length"] / diagonal if diagonal > 0 else 0.0

    safety_reward = clamp(min_d / HEURISTIC_CLEARANCE_SCORE_CAP_PX, 0, 1) * 100
    score = safety_reward - norm_len * 15.0
    return {
        "entry": pt,
        "score": score,
        "min_dist": min_d,
        "intersects": intersects,
        "leaves_roi": leaves_roi,
        "geo": geo,
        "line_pts": line_pts,
    }


def _entry_is_admissible(
    pt: PointTuple,
    target: PointTuple,
    structure_mask: MaskArray,
    dist_map: FloatArray,
    roi_allowed: MaskArray,
) -> bool:
    """Same rules the manual validator enforces."""
    h, w = structure_mask.shape
    px, py = pt
    if not (0 <= px < w and 0 <= py < h):
        return False
    if pt == target:
        return False
    if roi_allowed[py, px] == 0:
        return False
    if structure_mask[py, px] > 0:
        return False
    if dist_map[py, px] < HEURISTIC_MIN_CLEARANCE_PX:
        return False
    return True


def _roi_edge_candidates(roi: MaskArray, n_candidates: int) -> List[PointTuple]:
    """Evenly spaced points along the outer contour of the largest ROI component."""
    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return []
    cnt = max(contours, key=cv2.contourArea)
    pts = cnt[:, 0, :]
    if len(pts) == 0:
        return []
    idx = np.unique(np.linspace(0, len(pts) - 1, min(n_candidates, len(pts))).astype(int))
    return [(int(pts[i][0]), int(pts[i][1])) for i in idx]


def find_best_entry_candidate(
    structure_mask: MaskArray,
    target: PointTuple,
    dist_map: FloatArray,
    roi: MaskArray,
    roi_allowed: MaskArray,
    roi_depth: FloatArray,
    num_candidates_total: int = 72,
) -> Optional[Dict[str, Any]]:
    """
    Searches the heuristic brain-ROI edge for the best entry point: inside the ROI,
    outside the risk mask, and the whole path stays inside the ROI with min clearance
    >= HEURISTIC_MIN_CLEARANCE_PX (so the result is never rated DANGER by the evaluator). Returns None if none exists.
    """
    h, w = structure_mask.shape
    diagonal = float(np.hypot(w, h))

    results = []
    candidate_audit: List[Dict[str, Any]] = []
    seen = set()
    initial_eval_coords = set()
    edge_candidates = _roi_edge_candidates(roi, num_candidates_total)
    for pt in edge_candidates:
        if pt in seen:
            continue
        seen.add(pt)
        if pt == target:
            candidate_audit.append({"entry_px": list(pt), "accepted": False, "reason": "same_as_target"})
            continue
        if not (0 <= pt[0] < w and 0 <= pt[1] < h):
            candidate_audit.append({"entry_px": list(pt), "accepted": False, "reason": "out_of_bounds"})
            continue
        if roi[pt[1], pt[0]] == 0:
            candidate_audit.append({"entry_px": list(pt), "accepted": False, "reason": "outside_original_roi"})
            continue
        if structure_mask[pt[1], pt[0]] > 0:
            candidate_audit.append({"entry_px": list(pt), "accepted": False, "reason": "inside_dilated_risk_mask"})
            continue
        if dist_map[pt[1], pt[0]] < HEURISTIC_MIN_CLEARANCE_PX:
            candidate_audit.append({"entry_px": list(pt), "accepted": False, "reason": "below_minimum_clearance"})
            continue
        r = _evaluate_entry(pt, target, structure_mask, dist_map, roi, diagonal)
        initial_eval_coords.add(pt)
        if r["intersects"] or r["leaves_roi"] or r["min_dist"] < HEURISTIC_MIN_CLEARANCE_PX:
            reason = "path_collision" if r["intersects"] else "path_leaves_original_roi" if r["leaves_roi"] else "path_clearance_below_limit"
            candidate_audit.append({"entry_px": list(pt), "accepted": False, "reason": reason,
                                    "path_length_px": round(r["geo"]["length"], 3), "clearance_px": round(r["min_dist"], 3)})
            continue
        results.append(r)
        candidate_audit.append({"entry_px": list(pt), "accepted": True, "reason": "passed_geometric_heuristics",
                                "score": round(r["score"], 4), "path_length_px": round(r["geo"]["length"], 3),
                                "clearance_px": round(r["min_dist"], 3)})

    def _counts(refine_coords: set, refine_evals: int, refine_accepted: int) -> Dict[str, int]:
        return {
            "initial_candidates_generated": len(edge_candidates),
            "initial_candidates_unique": len(seen),
            "initial_path_evaluations": len(initial_eval_coords),
            "initial_candidates_accepted": len(results),
            "refinement_path_evaluations": refine_evals,
            "refinement_unique_new_coordinates": len(refine_coords - initial_eval_coords),
            "refinement_candidates_accepted": refine_accepted,
            "total_path_evaluations": len(initial_eval_coords) + refine_evals,
            "unique_coordinates_path_evaluated": len(initial_eval_coords | refine_coords),
        }

    if not results:
        return {"best": None, **_counts(set(), 0, 0),
                "local_refinement_radius_px": 3, "candidate_audit": candidate_audit}

    results.sort(key=lambda r: (-r["score"], r["geo"]["length"], r["entry"]))
    best = results[0]

    # Local refinement (stays near the ROI edge and keeps all admissibility rules)
    bx, by = best["entry"]
    radius = 3
    refinement_evaluated = 0
    refinement_accepted = 0
    refinement_coords = set()
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            nx, ny = bx + dx, by + dy
            cand = (nx, ny)
            if not (0 <= nx < w and 0 <= ny < h):
                continue
            if roi_depth[ny, nx] > ROI_ENTRY_EDGE_TOL_PX:
                candidate_audit.append({"entry_px": [nx, ny], "search_stage": "local_refinement", "accepted": False, "reason": "outside_edge_refinement_band"})
                continue
            if not _entry_is_admissible(cand, target, structure_mask, dist_map, roi):
                reason = "same_as_target" if cand == target else "outside_original_roi" if roi[ny, nx] == 0 else "inside_dilated_risk_mask" if structure_mask[ny, nx] > 0 else "below_minimum_clearance"
                candidate_audit.append({"entry_px": [nx, ny], "search_stage": "local_refinement", "accepted": False, "reason": reason})
                continue
            r = _evaluate_entry(cand, target, structure_mask, dist_map, roi, diagonal)
            refinement_evaluated += 1
            refinement_coords.add(cand)
            valid = not r["intersects"] and not r["leaves_roi"] and r["min_dist"] >= HEURISTIC_MIN_CLEARANCE_PX
            refinement_accepted += int(valid)
            candidate_audit.append({"entry_px": [nx, ny], "search_stage": "local_refinement", "accepted": valid,
                                    "reason": "passed_geometric_heuristics" if valid else "path_collision_or_roi_or_clearance_check_failed",
                                    "score": round(r["score"], 4), "path_length_px": round(r["geo"]["length"], 3),
                                    "clearance_px": round(r["min_dist"], 3)})
            if (not r["intersects"] and not r["leaves_roi"]
                    and r["min_dist"] >= HEURISTIC_MIN_CLEARANCE_PX and r["score"] > best["score"]):
                best = r

    return {"best": best, **_counts(refinement_coords, refinement_evaluated, refinement_accepted),
            "local_refinement_radius_px": radius, "candidate_audit": candidate_audit}


# ============================================================================
# SECTION 8: VISUALIZATION UTILITIES
# ============================================================================

def draw_trajectory_overlay(
    img: MaskArray,
    entry: PointTuple,
    target: PointTuple,
    color: Tuple[int, int, int],
    seeg_contacts: Optional[List[Dict]] = None,
    probe_pos: Optional[PointTuple] = None,
) -> MaskArray:
    out = img.copy()
    cv2.line(out, entry, target, color, 2, cv2.LINE_AA)
    cv2.circle(out, entry, 6, (255, 0, 0), -1)
    cv2.circle(out, target, 6, (0, 255, 0), -1)

    if seeg_contacts:
        for c in seeg_contacts:
            pos = c["pos"]
            cv2.circle(out, pos, 3, (0, 165, 255), -1)
            cv2.putText(out, str(c["id"]), (pos[0] + 4, pos[1] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 165, 255), 1, cv2.LINE_AA)

    if probe_pos:
        cv2.circle(out, probe_pos, 5, (255, 255, 255), -1)
        cv2.circle(out, probe_pos, 7, (0, 0, 0), 1)

    return out


def draw_points_preview(
    img: MaskArray, target: Optional[PointTuple], entry: Optional[PointTuple]
) -> MaskArray:
    out = img.copy()
    if target is not None:
        cv2.circle(out, target, 6, (0, 255, 0), -1)
        cv2.circle(out, target, 9, (0, 0, 0), 1)
    if entry is not None:
        cv2.circle(out, entry, 6, (255, 0, 0), -1)
        cv2.circle(out, entry, 9, (0, 0, 0), 1)
    if target is not None and entry is not None:
        cv2.line(out, entry, target, (200, 200, 200), 1, cv2.LINE_AA)
    return out


def render_heatmap_with_colorbar(
    base_img: MaskArray,
    data_map: FloatArray,
    vmin: float,
    vmax: float,
    colormap: int,
    title: str,
) -> MaskArray:
    h, w = data_map.shape
    if vmax <= vmin:
        vmax = vmin + 1e-6
    clipped = np.clip(data_map, vmin, vmax)
    normalized = ((clipped - vmin) / (vmax - vmin) * 255).astype(np.uint8)

    heat_colored = cv2.applyColorMap(normalized, colormap)

    alpha = 0.5
    blended = cv2.addWeighted(base_img, 1 - alpha, heat_colored, alpha, 0)

    cb_height = h
    cb_width = 40
    cb_img = np.zeros((cb_height, cb_width, 3), dtype=np.uint8)
    for i in range(cb_height):
        val = 255 - int((i / cb_height) * 255)
        cb_img[i, :] = cv2.applyColorMap(np.array([[val]], dtype=np.uint8), colormap)[0, 0]

    cv2.putText(cb_img, f"{vmax:.0f}", (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(cb_img, f"{vmin:.0f}", (3, cb_height - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    final_img = np.hstack([blended, cb_img])

    title_bar = np.zeros((25, final_img.shape[1], 3), dtype=np.uint8)
    cv2.putText(title_bar, title, (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return np.vstack([title_bar, final_img])


def show_bgr(img_bgr: MaskArray, caption: Optional[str] = None) -> None:
    st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption=caption, **STRETCH)


def build_segmentation_overlay(
    original: MaskArray, roi: MaskArray, raw_mask: MaskArray, dilated_mask: MaskArray
) -> MaskArray:
    """Red = risk regions, Orange = safety buffer, Green outline = brain ROI."""
    vis = original.copy()
    vis[raw_mask > 0] = [0, 0, 255]
    buffer_ring = (dilated_mask > 0) & (raw_mask == 0)
    vis[buffer_ring] = [0, 165, 255]
    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, contours, -1, (0, 255, 0), 1)
    return vis


# ============================================================================
# SECTION 9: PLAN BUILDER (pure function, no UI side effects)
# ============================================================================

def build_plan(
    entry: PointTuple,
    target: PointTuple,
    struct_mask: MaskArray,
    raw_mask: MaskArray,
    dist_map: FloatArray,
    roi: MaskArray,
    roi_allowed: MaskArray,
    roi_depth: FloatArray,
    blurred_gray: np.ndarray,
    w: int,
    h: int,
    procedure_mode: str,
    use_otsu: bool,
    manual_thresh: int,
    risk_percentile: float,
    min_area: int,
    dilation_rad: int,
    laser_power: float,
    ablation_time: float,
    num_contacts: int,
    contact_spacing: float,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Returns (results, error_message)."""
    if entry == target:
        return None, "Entry and Target points must be different."

    ok_t, msg_t = validate_target(target, struct_mask, roi, w, h)
    if not ok_t:
        return None, f"Invalid Target: {msg_t}"

    ok_e, msg_e, warn_edge = validate_entry(entry, struct_mask, dist_map, roi_allowed, roi_depth, w, h)
    if not ok_e:
        return None, f"Invalid Entry: {msg_e}"

    line_pts = bresenham_line(entry, target)
    intersects = check_collision(struct_mask, line_pts)
    roi_outside_px = check_roi_containment(roi_allowed, line_pts)
    original_roi_outside_px = check_roi_containment(roi, line_pts)
    leaves_roi = original_roi_outside_px > 0
    min_d = min_clearance_along_line(dist_map, line_pts)
    geo = calculate_geometry(entry, target)

    safety = evaluate_safety_metrics(
        min_d, intersects, leaves_roi, blurred_gray, roi, manual_thresh, use_otsu, risk_percentile,
        min_area, dilation_rad, entry, target,
    )

    results: Dict[str, Any] = {
        "target": target,
        "entry": entry,
        "geometry": geo,
        "safety": safety,
        "line_points": line_pts,
        "entry_warning": msg_e if warn_edge else "",
        "roi_outside_px": roi_outside_px,
        "original_roi_outside_px": original_roi_outside_px,
        "thermal": None,
        "seeg": None,
    }

    if procedure_mode == "LITT Thermal Ablation":
        results["thermal"] = simulate_litt_thermal_field(target, raw_mask, roi, laser_power, ablation_time)
    else:
        results["seeg"] = compute_seeg_contacts(entry, target, geo["length"], num_contacts, contact_spacing)

    return results, None


# ============================================================================
# SECTION 10: LIVE SIMULATION (isolated fragment)
# ============================================================================

HAS_FRAGMENT = hasattr(st, "fragment")
fragment_decorator = st.fragment if HAS_FRAGMENT else (lambda f: f)


def _rerun_sim() -> None:
    """Rerun only the simulation fragment when supported."""
    if HAS_FRAGMENT:
        try:
            st.rerun(scope="fragment")
            return
        except TypeError:
            pass
    st.rerun()


@fragment_decorator
def live_simulation(res: Dict[str, Any], original_img: MaskArray, dist_map: FloatArray) -> None:
    line_pts = res["line_points"]
    n_pts = len(line_pts)

    ctrl_c1, ctrl_c2, ctrl_c3, ctrl_c4 = st.columns([1, 1, 1, 3])
    with ctrl_c1:
        if st.button("▶️ Play", key="sim_play_btn"):
            if st.session_state.sim_step >= n_pts - 1:
                st.session_state.sim_step = 0
            st.session_state.sim_playing = True
    with ctrl_c2:
        if st.button("⏸️ Pause", key="sim_pause_btn"):
            st.session_state.sim_playing = False
    with ctrl_c3:
        if st.button("⏹️ Reset", key="sim_reset_btn"):
            st.session_state.sim_playing = False
            st.session_state.sim_step = 0
    with ctrl_c4:
        speed = st.slider("Speed", 1, 10, 5, key="sim_speed")

    curr_idx = int(clamp(st.session_state.sim_step, 0, n_pts - 1))
    probe_pos = line_pts[curr_idx]

    sim_frame = draw_trajectory_overlay(
        original_img,
        res["entry"],
        res["target"],
        res["safety"]["color_bgr"],
        seeg_contacts=res["seeg"]["contacts"] if res["seeg"] else None,
        probe_pos=probe_pos,
    )

    cum_depths = build_cumulative_depths(line_pts)
    telem = get_telemetry(line_pts, dist_map, curr_idx, cum_depths)

    sim_col, tele_col = st.columns([2, 1])
    with sim_col:
        show_bgr(sim_frame, caption=f"Probe Position: {telem['progress_pct']}%")
    with tele_col:
        st.metric("Depth (px)", f"{telem['current_depth']:.1f} / {telem['total_length']:.1f}")
        st.progress(float(clamp(telem["progress_pct"] / 100, 0.0, 1.0)))
        st.metric("Dist. to Risk Region (px)", f"{telem['distance_to_structure']:.1f}")
        if telem["proximity_alert"]:
            st.error("🚨 Configured pixel-proximity threshold crossed (heuristic only).")
        else:
            st.success("✅ Above configured pixel-clearance alert threshold (heuristic only)")

    # Advance AFTER drawing so every frame is actually shown.
    if st.session_state.sim_playing:
        time.sleep(0.15 / speed)
        steps_per_tick = max(1, n_pts // 200)
        if curr_idx < n_pts - 1:
            st.session_state.sim_step = min(curr_idx + steps_per_tick, n_pts - 1)
        else:
            st.session_state.sim_playing = False
        _rerun_sim()


# ============================================================================
# SECTION 11: MAIN STREAMLIT APP  (UI layer restyled; logic unchanged)
# ============================================================================

st.set_page_config(page_title="ThermoSEEG RC", page_icon="🧠", layout="wide")
inject_theme()

topbar_slot = st.empty()   # filled once we know the workflow stage

disclaimer_text = (
    "**⚠️ MEDICAL DISCLAIMER:** This software is an algorithmic simulation prototype intended for "
    "research and educational demonstration only. It utilizes simplified heuristic models (STATIC Gaussian "
    "temperature approximation parameterized by laser power; duration affects the CEM43 dose estimate only, "
    "not the spatial temperature field) and does not account for patient-specific anatomy, "
    "anisotropic tissue properties, or real-time physiological feedback. "
    "Risk regions are intensity-based heuristics, not anatomical segmentation of vessels or eloquent cortex. "
    "This is 2D only and does not establish brain tissue or CSF segmentation. The ROI is a heuristic foreground mask, and thermal "
    "attenuation is applied around intensity-based risk regions only (no CSF or vessel model). "
    "Trajectory checks are geometric heuristics in one image plane; scalp/skull crossing and 3D anatomy are not modeled. "
    "**DO NOT USE FOR CLINICAL DECISION MAKING OR SURGICAL GUIDANCE.**"
)

with st.sidebar:
    sidebar_brand()
    settings_file = st.file_uploader("Load settings from ThermoSEEG JSON", type=["json"], key="run_settings_upload")
    if settings_file is not None:
        try:
            parsed_run = parse_exported_run_config(settings_file.getvalue())
            import_signature = hashlib.sha256(settings_file.getvalue()).hexdigest()
            if st.session_state.get("_run_settings_signature") != import_signature:
                imported_values = configuration_widget_values(parsed_run)
                widget_key_map = {
                    "procedure_type": "procedure_type", "laser_power": "laser_power",
                    "ablation_duration": "ablation_duration", "seeg_contact_count": "seeg_contact_count",
                    "seeg_contact_spacing": "seeg_contact_spacing", "auto_threshold": "auto_threshold",
                    "risk_percentile": "risk_percentile", "manual_threshold": "manual_threshold",
                    "roi_strip_radius": "roi_strip_radius", "minimum_risk_area": "minimum_risk_area",
                    "risk_dilation": "risk_dilation", "show_mm": "show_mm",
                    "manual_pixel_spacing": "manual_pixel_spacing",
                }
                for source_key, widget_key in widget_key_map.items():
                    st.session_state[widget_key] = imported_values[source_key]
                st.session_state["_run_settings_signature"] = import_signature
                st.session_state["_imported_run_config"] = parsed_run
                st.session_state["_imported_widget_values"] = imported_values
                st.session_state["_config_points_pending"] = {
                    "target": imported_values["target_point"], "entry": imported_values["entry_point"],
                    "image_sha256": parsed_run.get("image_sha256"),
                    "signature": import_signature,
                }
        except RunConfigError as exc:
            st.error(f"Cannot load settings: {exc}")
    if st.session_state.get("_imported_widget_values"):
        st.caption("Saved settings loaded. Confirm that the image SHA-256 matches before reusing the saved points.")

    procedure_mode = st.radio(
        "Procedure Type",
        ["SEEG Electrode Planning", "LITT Thermal Ablation"],
        key="procedure_type",
        help="Select the surgical modality to configure parameters.",
    )

    st.divider()
    st.subheader("Parameters")

    if procedure_mode == "LITT Thermal Ablation":
        laser_power = st.slider(
            "Laser Power (W)", 2.0, MAX_LASER_POWER_W,
            min(DEFAULT_LASER_POWER_W, MAX_LASER_POWER_W), 0.5,
            key="laser_power",
            help="Upper setting follows the legacy synthetic model numerical range; it is not a calibrated operating limit.",
        )
        ablation_time = st.slider("Duration (s)", 5.0, 120.0, DEFAULT_ABLATION_TIME_SEC, 5.0, key="ablation_duration")
        st.caption("Uncalibrated static Gaussian demonstration: duration scales synthetic CEM43 only. Not a tissue temperature or treatment prediction.")
        num_contacts = 0
        contact_spacing = 0.0
    else:
        num_contacts = st.slider("Num Contacts", 2, 18, DEFAULT_SEEG_CONTACTS_COUNT, key="seeg_contact_count")
        contact_spacing = st.slider("Spacing (px)", 2.0, 30.0, DEFAULT_SEEG_SPACING_PX, 1.0, key="seeg_contact_spacing")
        laser_power = 0.0
        ablation_time = 0.0

    st.divider()
    st.subheader("Image Processing (Risk Regions)")
    use_otsu = st.checkbox(
        "Auto Threshold (percentile inside brain ROI)", value=True,
        key="auto_threshold",
        help="Threshold = chosen percentile of pixel intensity inside the brain ROI.",
    )
    risk_percentile_ui = st.slider(
        "High-Signal Percentile (heuristic)", 80, 99, RISK_PERCENTILE_DEFAULT,
        disabled=not use_otsu,
        key="risk_percentile",
        help="Top X% brightest pixels inside the brain ROI become risk regions. "
             "If the red area is too large, RAISE this value (98-99).",
    )
    manual_thresh_ui = st.slider("Manual Threshold", 0, 255, 140, disabled=use_otsu, key="manual_threshold")
    # Normalise so the cache/signature don't change when a disabled slider is irrelevant.
    manual_thresh = 0 if use_otsu else int(manual_thresh_ui)
    risk_percentile = float(risk_percentile_ui) if use_otsu else float(RISK_PERCENTILE_DEFAULT)
    roi_strip_px = st.slider(
        "Scalp/Skull Strip Radius (px)", 0, 20, ROI_STRIP_RADIUS_DEFAULT_PX,
        key="roi_strip_radius",
        help="Opening radius that removes thin scalp/skull rings from the brain ROI. "
             "Raise it until the green ROI outline hugs the brain; 0 disables stripping.",
    )
    min_area = st.slider("Min Risk Region Area (px²)", 0, 200, 30, key="minimum_risk_area")
    dilation_rad = st.slider(
        "Safety Buffer (px)", 0, 15, DEFAULT_DILATION_RADIUS_PX,
        key="risk_dilation",
        help="Expands risk regions for trajectory safety.",
    )

    enable_mm = st.checkbox("Show Measurements in mm", value=False, key="show_mm")


st.warning(disclaimer_text)

with st.expander("About, Methodology & Limitations"):
    about_tab, method_tab, limit_tab = st.tabs(["About", "Methodology", "Limitations"])
    with about_tab:
        st.markdown(f"**{TOOL_VERSION}** is a research and educational prototype for 2D image-mask geometry, threshold sensitivity, candidate search, geometric SEEG contact placement, and a simplified thermal demonstration. It is not a clinical decision support or surgical guidance system.")
    with method_tab:
        st.markdown("""
        1. Raster images are decoded as 8-bit color/grayscale. Supported DICOM MR/CT single slices apply Modality LUT or Rescale Slope/Intercept, then VOI LUT/window when present, then a per-slice display scaling to 8-bit. Segmentation uses this display mapping.
        2. A heuristic Otsu foreground mask is opened, reduced to its largest component, then closed. It is not anatomical segmentation; small gaps can close.
        3. Bright pixels within that mask form heuristic high-signal regions. A dilated copy is used for geometric collision checks.
        4. Trajectories use 2D raster geometry. Candidate search samples a finite edge set and a small local neighborhood; the documented score is experimental.
        5. The LITT demonstration uses an uncalibrated static Gaussian field. Duration changes synthetic CEM43 only. If the 100 °C numerical guard is exceeded, dose-derived values are suppressed.
        """)
    with limit_tab:
        st.markdown("""
        - No 3D anatomy, validated brain/CSF segmentation, vessel map, tissue properties, perfusion, transient heat transfer, or clinical contact validation.
        - Sensitivity varies only the risk threshold in three cases while holding ROI fixed. Its stability label is not model accuracy, mask quality, or data reliability.
        - Metric path length and minimum distance use supplied horizontal/vertical pixel spacing. Heuristic thresholds, risk dilation, contact layout, optimizer score and thermal kernels remain pixel-based.
        - DICOM is limited to monochrome single-frame MR/CT. Orientation and slice metadata are recorded but not used to construct spatial 3D coordinates. Color, multi-frame and other modalities are rejected.
        - Inputs are not intentionally written to disk and identifiers are excluded from exports. Process memory and Streamlit cache may retain data temporarily; follow local data-governance requirements.
        """)

with st.container(border=True):
    section_header("1", "Imaging", "Upload an MRI/CT slice to begin")
    col_upload, col_preview = st.columns([1, 2])
    with col_upload:
        uploaded_file = st.file_uploader("Upload MRI/CT Slice", type=["jpg", "jpeg", "png", "dcm"])
        st.caption("MRI is the intended input. CT is decoded only; ROI heuristics are not validated for CT. Maximum upload: 50 MB; image limits: 4096 px per side and 16 million pixels. DICOM display scaling is not quantitative, and 3D orientation is not reconstructed.")

if not uploaded_file:
    with topbar_slot:
        render_topbar("Imaging", [(procedure_mode, True)])
    st.info("⬆️ Please upload a medical scan image to begin planning.")
    st.stop()

if uploaded_file.size > MAX_UPLOAD_BYTES:
    st.error("Upload exceeds the 50 MB file-size limit.")
    st.stop()
file_bytes = uploaded_file.getvalue()
if len(file_bytes) > MAX_UPLOAD_BYTES:
    st.error("Upload exceeds the 50 MB file-size limit.")
    st.stop()
dicom_modality_hint = inspect_dicom_modality(file_bytes)
if dicom_modality_hint == "CT":
    ct_ack_key = f"ct_heuristic_ack_{hashlib.sha256(file_bytes).hexdigest()[:12]}"
    ct_acknowledged = st.checkbox("I understand that the ROI heuristic is designed around MRI-like contrast and is not validated for CT.", key=ct_ack_key)
    if not ct_acknowledged:
        st.info("Acknowledge the CT limitation before running image analysis.")
        st.stop()
file_sha256 = hashlib.sha256(file_bytes).hexdigest()
file_hash = file_sha256  # full image identity for signatures, session state, and widget keys
file_id_short = file_sha256[:16]  # display and output file names only

with st.spinner("Processing Image..."):
    prep_data = load_and_preprocess_image(
        file_bytes, manual_thresh, use_otsu, risk_percentile, min_area, dilation_rad, roi_strip_px
    )

if not prep_data or "original" not in prep_data:
    st.error("Failed to process image.")
    st.stop()

original_img = prep_data["original"]
roi_mask = prep_data["roi"]                  # heuristic allowed region (NOT an obstacle)
roi_allowed = prep_data["roi_allowed"]       # ROI + rasterization tolerance; reported separately
roi_depth = prep_data["roi_depth"]           # px inside the ROI
raw_mask = prep_data["raw_structure_mask"]   # thermal physics
struct_mask = prep_data["structure_mask"]    # safety / collision
blurred_gray = prep_data["blurred"]
h, w = original_img.shape[:2]
dicom_spacing_xy = prep_data.get("pixel_spacing_mm_xy")
if dicom_spacing_xy:
    pixel_spacing_mm_xy = dicom_spacing_xy
    pixel_spacing_source = "DICOM PixelSpacing"
    st.sidebar.caption(f"DICOM spacing: horizontal {pixel_spacing_mm_xy[0]:g} mm/px, vertical {pixel_spacing_mm_xy[1]:g} mm/px")
elif prep_data.get("dicom_metadata"):
    st.warning("DICOM has no valid PixelSpacing; physical length and clearance will not be reported.")
elif enable_mm:
    manual_spacing = st.sidebar.number_input("Pixel Spacing (mm/px)", 0.01, 10.0, 1.0, 0.01, key="manual_pixel_spacing")
    pixel_spacing_mm_xy = (manual_spacing, manual_spacing)
    pixel_spacing_source = "manual isotropic assumption"
else:
    pixel_spacing_mm_xy = None
    pixel_spacing_source = "unknown"

dist_map = compute_distance_map_cached(struct_mask)
physical_dist_map = (
    compute_distance_map_cached(struct_mask, sampling=(pixel_spacing_mm_xy[1], pixel_spacing_mm_xy[0]))
    if pixel_spacing_mm_xy else None
)

with col_preview:
    tab1, tab2 = st.tabs(["Original Scan", "High-Signal Risk Regions"])
    with tab1:
        show_bgr(original_img, caption=f"Slice Dimensions: {w}x{h}px")
    with tab2:
        show_bgr(
            build_segmentation_overlay(original_img, roi_mask, raw_mask, struct_mask),
            caption="Red = high-signal risk regions | Orange = safety buffer | Green outline = heuristic brain-region ROI",
        )
        st.caption("Mask color legend: red = intensity-derived high-signal region; orange = pixel dilation buffer; green = heuristic foreground ROI. None is an anatomical label.")
        risk_pct_area = 100.0 * float(np.count_nonzero(raw_mask)) / max(1, int(np.count_nonzero(roi_mask)))
        st.caption(f"Risk regions cover {risk_pct_area:.1f}% of the brain ROI.")
        st.caption("If the green outline includes scalp/skull, raise 'Scalp/Skull Strip Radius'.")
        if risk_pct_area > RISK_MASK_MAX_FRACTION * 100:
            st.warning("Risk mask is large. Raise the High-Signal Percentile (98-99) or the manual threshold.")

# --- State Initialization ---
default_keys = {
    "target_pt": None, "entry_pt": None, "results": None,
    "sim_playing": False, "sim_step": 0,
    "selection_stage": "target", "last_hash": "",
    "click_nonce": 0, "result_sig": None, "auto_info": None,
    "prev_auto": None, "auto_search_audit": None,
}
for k, v in default_keys.items():
    if k not in st.session_state:
        st.session_state[k] = v

if st.session_state.last_hash != file_hash:
    st.session_state.target_pt = None
    st.session_state.entry_pt = None
    st.session_state.selection_stage = "target"
    st.session_state.sim_playing = False
    st.session_state.sim_step = 0
    st.session_state.last_hash = file_hash
    st.session_state.results = None
    st.session_state.auto_info = None
    st.session_state.auto_search_audit = None
    st.session_state.click_nonce += 1

pending_points = st.session_state.get("_config_points_pending")
if pending_points:
    if pending_points.get("image_sha256") == file_sha256:
        if st.session_state.get("_config_points_applied_signature") != pending_points.get("signature"):
            st.session_state.target_pt = tuple(pending_points["target"])
            st.session_state.entry_pt = tuple(pending_points["entry"])
            st.session_state.selection_stage = "entry"
            st.session_state["_config_points_applied_signature"] = pending_points.get("signature")
    else:
        st.warning("Saved target and entry were not restored because the uploaded image SHA-256 does not match the settings file.")


def reset_points() -> None:
    st.session_state.target_pt = None
    st.session_state.entry_pt = None
    st.session_state.selection_stage = "target"
    st.session_state.results = None
    st.session_state.auto_info = None
    st.session_state.sim_playing = False
    st.session_state.sim_step = 0
    st.session_state.click_nonce += 1


# --- Interaction UI ---
with st.container(border=True):
    section_header("2", "Define Trajectory Points", "Target → Entry")

    entry_strategy = st.radio("Entry Strategy", ["Manual Selection", "🎯 Smart Candidate Search"], horizontal=True)
    auto_mode = entry_strategy.startswith("🎯")

    # Switching strategy clears a stale result carried over from the other input mode.
    if st.session_state.prev_auto is not None and st.session_state.prev_auto != auto_mode:
        st.session_state.entry_pt = None
        st.session_state.auto_info = None
        st.session_state.auto_search_audit = None
        st.session_state.results = None
        if not auto_mode:
            st.session_state.selection_stage = "target" if st.session_state.target_pt is None else "entry"
            st.session_state.click_nonce += 1
    st.session_state.prev_auto = auto_mode

    input_method = st.radio(
        "Input Method", ["Click on Image", "Manual Coordinates"], horizontal=True, disabled=auto_mode
    )

    if auto_mode:
        st.info("Set a valid Target, then search a limited set of ROI-edge candidates. The result is the best evaluated geometric candidate under experimental scoring, not a clinical recommendation.")

        c_tx, c_ty = st.columns(2)
        default_tx = st.session_state.target_pt[0] if st.session_state.target_pt else int(w * 0.6)
        default_ty = st.session_state.target_pt[1] if st.session_state.target_pt else int(h * 0.6)
        tx = c_tx.number_input("Target X", 0, w - 1, int(default_tx), key=f"atx_{file_hash}")
        ty = c_ty.number_input("Target Y", 0, h - 1, int(default_ty), key=f"aty_{file_hash}")
        new_target = (int(tx), int(ty))
        if new_target != st.session_state.target_pt:
            st.session_state.target_pt = new_target
            # Previous auto entry was computed for a different target.
            st.session_state.entry_pt = None
            st.session_state.auto_info = None
            st.session_state.auto_search_audit = None

        if st.button("Run Auto-Search", type="primary"):
            ok_target, target_message = validate_target(st.session_state.target_pt, struct_mask, roi_mask, w, h)
            if not ok_target:
                st.error(f"Auto-Search stopped: {target_message}")
                st.stop()
            res_auto = find_best_entry_candidate(
                struct_mask, st.session_state.target_pt, dist_map, roi_mask, roi_allowed, roi_depth
            )
            st.session_state.auto_search_audit = res_auto
            if res_auto and res_auto["best"] is not None:
                best = res_auto["best"]
                st.session_state.entry_pt = best["entry"]
                st.session_state.auto_info = {
                    "min_dist": round(float(best["min_dist"]), 1),
                    "length": round(float(best["geo"]["length"]), 1),
                    "score": round(float(best["score"]), 4),
                    **{key: res_auto[key] for key in SEARCH_COUNT_KEYS},
                    "local_refinement_radius_px": res_auto["local_refinement_radius_px"],
                    "candidate_audit": res_auto["candidate_audit"],
                }
                st.rerun()
            else:
                st.session_state.entry_pt = None
                st.session_state.auto_info = None
                st.error("No candidate passed the configured geometric checks. Review the rejection log below; this does not establish anatomy or safety.")
                st.json({key: res_auto[key] for key in SEARCH_COUNT_KEYS})
                st.dataframe(pd.DataFrame(res_auto.get("candidate_audit", [])), hide_index=True, **STRETCH)

        if st.session_state.auto_info and st.session_state.entry_pt:
            info = st.session_state.auto_info
            st.success(
                f"Best evaluated entry candidate: `{st.session_state.entry_pt}` | min clearance {info['min_dist']} px | "
                f"length {info['length']} px | {info['unique_coordinates_path_evaluated']} unique coordinates path-evaluated "
                f"({info['total_path_evaluations']} path evaluations); {info['initial_candidates_accepted']} passed initial search, "
                f"{info['refinement_candidates_accepted']} passed local refinement"
            )
            with st.expander("Candidate search audit"):
                st.caption("Finite sampled search with local refinement; path evaluations count only candidates sent to full trajectory checks. A coordinate may be checked in both stages; score is experimental and ranking is not a clinical recommendation.")
                st.json({key: info[key] for key in SEARCH_COUNT_KEYS})
                st.dataframe(pd.DataFrame(info["candidate_audit"]), hide_index=True, **STRETCH)

        show_bgr(
            draw_points_preview(original_img, st.session_state.target_pt, st.session_state.entry_pt),
            caption="Green = Target, Blue = Entry",
        )

    else:
        if input_method == "Click on Image":
            if not CLICK_SELECTION_AVAILABLE:
                st.error("`streamlit-image-coordinates` library not installed (pip install streamlit-image-coordinates).")
            else:
                stage = st.session_state.selection_stage

                if stage == "target":
                    st.info("🎯 Click to set **Target**.")
                elif stage == "entry":
                    st.info("🔴 Click to set **Entry** (on the edge of the green ROI outline).")
                else:
                    st.success("Both points set. Press Reset to choose new points.")

                if stage in ("target", "entry"):
                    preview = draw_points_preview(
                        original_img, st.session_state.target_pt, st.session_state.entry_pt
                    )
                    pil_img = Image.fromarray(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB))
                    click_key = f"click_coord_{file_hash}_{st.session_state.click_nonce}"
                    coords = streamlit_image_coordinates(pil_img, key=click_key, width=w)

                    if coords:
                        disp_w = coords.get("width") or w
                        disp_h = coords.get("height") or h
                        cx = int(round(coords["x"] * w / disp_w))
                        cy = int(round(coords["y"] * h / disp_h))
                        if 0 <= cx < w and 0 <= cy < h:
                            if stage == "target":
                                st.session_state.target_pt = (cx, cy)
                                st.session_state.selection_stage = "entry"
                            else:
                                st.session_state.entry_pt = (cx, cy)
                                st.session_state.selection_stage = "done"
                            # Remount the component so this click is never consumed twice.
                            st.session_state.click_nonce += 1
                            st.rerun()
                else:
                    show_bgr(
                        draw_points_preview(original_img, st.session_state.target_pt, st.session_state.entry_pt),
                        caption="Green = Target, Blue = Entry",
                    )

                st.write(f"**Target:** `{st.session_state.target_pt}` | **Entry:** `{st.session_state.entry_pt}`")

                if st.button("🔄 Reset Points"):
                    reset_points()
                    st.rerun()
        else:
            col_t, col_e = st.columns(2)
            with col_t:
                tx = st.number_input("Target X", 0, w - 1, int(w * 0.6), key=f"mtx_{file_hash}")
                ty = st.number_input("Target Y", 0, h - 1, int(h * 0.6), key=f"mty_{file_hash}")
                st.session_state.target_pt = (int(tx), int(ty))
            with col_e:
                ex = st.number_input("Entry X", 0, w - 1, int(w * 0.2), key=f"mex_{file_hash}")
                ey = st.number_input("Entry Y", 0, h - 1, int(h * 0.2), key=f"mey_{file_hash}")
                st.session_state.entry_pt = (int(ex), int(ey))

            show_bgr(
                draw_points_preview(original_img, st.session_state.target_pt, st.session_state.entry_pt),
                caption="Green = Target, Blue = Entry",
            )

# Signature is evaluated AFTER inputs update, so stale results are never shown.
current_sig = generate_config_signature(
    file_hash=file_hash,
    use_otsu=use_otsu,
    manual_thresh=manual_thresh,
    risk_percentile=risk_percentile,
    roi_strip_px=roi_strip_px,
    min_area=min_area,
    dilation_rad=dilation_rad,
    target_pt=st.session_state.target_pt,
    entry_pt=st.session_state.entry_pt,
    proc_mode=procedure_mode,
    laser_power=laser_power,
    abl_time=ablation_time,
    num_contacts=num_contacts,
    spacing_px=contact_spacing,
)
if st.session_state.result_sig != current_sig:
    st.session_state.results = None
    st.session_state.result_sig = current_sig
    st.session_state.sim_playing = False
    st.session_state.sim_step = 0

# Top bar reflects the workflow stage (Imaging done -> Planning -> Simulation/Report).
_active_tab = "Planning"
if st.session_state.results:
    _active_tab = "Thermal Simulation" if procedure_mode == "LITT Thermal Ablation" else "Report"
with topbar_slot:
    render_topbar(_active_tab, [(procedure_mode, True), (f"Slice {file_id_short}", False)])

with st.container(border=True):
    section_header("3", "Calculate Plan", "Validate points and run the model")

    can_compute = st.session_state.target_pt is not None and st.session_state.entry_pt is not None

    if can_compute:
        btn_label = (
            "Compute LITT Thermal Model" if procedure_mode == "LITT Thermal Ablation"
            else "Compute SEEG Trajectory"
        )

        if st.button(btn_label, type="primary", **STRETCH):
            plan, err = build_plan(
                entry=st.session_state.entry_pt,
                target=st.session_state.target_pt,
                struct_mask=struct_mask,
                raw_mask=raw_mask,
                dist_map=dist_map,
                roi=roi_mask,
                roi_allowed=roi_allowed,
                roi_depth=roi_depth,
                blurred_gray=blurred_gray,
                w=w,
                h=h,
                procedure_mode=procedure_mode,
                use_otsu=use_otsu,
                manual_thresh=manual_thresh,
                risk_percentile=risk_percentile,
                min_area=min_area,
                dilation_rad=dilation_rad,
                laser_power=laser_power,
                ablation_time=ablation_time,
                num_contacts=num_contacts,
                contact_spacing=contact_spacing,
            )
            if err:
                st.session_state.results = None
                st.error(f"❌ {err}")
            else:
                st.session_state.results = plan
                st.session_state.sim_step = 0
                st.session_state.sim_playing = False
                st.rerun()
    else:
        st.warning("Please define both Target and Entry points above.")

# --- Results Dashboard ---
if st.session_state.results:
    res = st.session_state.results
    measured_length_mm = calculate_geometry(res["entry"], res["target"], pixel_spacing_mm_xy)["length_mm"] if pixel_spacing_mm_xy else None
    measured_clearance_mm = min_clearance_along_line(physical_dist_map, res["line_points"]) if physical_dist_map is not None else None
    measured_contact_span_mm = None
    if pixel_spacing_mm_xy and res.get("seeg") and len(res["seeg"]["contacts"]) >= 2:
        measured_contact_span_mm = calculate_geometry(
            res["seeg"]["contacts"][0]["pos"], res["seeg"]["contacts"][-1]["pos"], pixel_spacing_mm_xy
        )["length_mm"]

    with st.container(border=True):
        section_header("4", "Analysis & Visualization", "Safety audit and thermal / clearance maps")

        if res.get("entry_warning"):
            st.warning(res["entry_warning"])

        if res["safety"]["leaves_roi"]:
            st.error(
                f"🚨 Trajectory crosses the original ROI boundary ({res['original_roi_outside_px']} px outside; {res['roi_outside_px']} px outside tolerated ROI). "
                "Status is forced to DANGER."
            )

        if res["safety"]["mask_unstable"]:
            st.warning(
                f"⚠️ Unstable risk mask: under a ±{SENSITIVITY_THRESHOLD_SHIFT} threshold shift it grows to "
                f"{res['safety']['max_risk_fraction'] * 100:.0f}% of the brain ROI. The threshold sits on the "
                "tissue intensity plateau. Raise the High-Signal Percentile (98-99) or use a manual threshold."
            )
        if res["safety"]["collision_flipped"]:
            st.warning(
                "⚠️ Segmentation sensitivity: a ±threshold change flips this trajectory between "
                "colliding and non-colliding. Treat the result as unreliable."
            )
        elif res["safety"]["status_flipped"]:
            st.info("ℹ️ Heuristic status changes under threshold shifts.")
        st.caption("Sensitivity stability describes only the three threshold scenarios shown below; it does not estimate mask quality, model accuracy, or input-data reliability.")

        # --- KPI summary tiles (same values as the table below) ---
        _st = res["safety"]["status"]
        _cf = res["safety"]["sensitivity_stability_index"]
        kpi_tiles([
            ("Heuristic Checks", _st, "geometric checks only; not clinical safety", STATUS_TONE.get(_st, "neutral")),
            ("Min Clearance", f"{res['safety']['min_distance_px']:.1f} px",
             f"score {res['safety']['clearance_score_pct']:.0f}% (distance-based)", "neutral"),
            ("Sensitivity Stability", _cf, f"3 threshold cases; ROI fixed; variability {res['safety']['variability_score']:.3f}",
             CONFIDENCE_TONE.get(_cf, "neutral")),
            ("Trajectory Length", f"{res['geometry']['length']:.1f} px",
             f"angle {res['geometry']['angle']:.1f}°", "neutral"),
        ])

        rows = [
            ("Trajectory Length (px)", f"{res['geometry']['length']:.2f}"),
            ("Entry Angle (deg; image coordinates)", f"{res['geometry']['angle']:.2f}"),
            ("Min Clearance (px)", f"{res['safety']['min_distance_px']:.2f}"),
            ("Experimental Clearance Score (not a safety probability)", f"{res['safety']['clearance_score_pct']:.1f}"),
            ("Inside Original ROI? (vs tolerated ROI reported in JSON)",
             "YES ✅" if not res["safety"]["leaves_roi"] else f"NO 🚨 ({res['original_roi_outside_px']} px outside original ROI)"),
            ("Heuristic Check Status (not clinical safety)", res["safety"]["status"]),
            ("Sensitivity Stability Index", res["safety"]["sensitivity_stability_index"]),
            ("Variability Score", f"{res['safety']['variability_score']:.3f}"),
            ("Collision Detected?", "YES 🚨" if res["safety"]["intersects"] else "NO ✅"),
        ]

        if res["seeg"]:
            rows.append(("SEEG Contacts Placed", f"{res['seeg']['num_contacts_placed']}"))
            rows.append(("Geometric Contact Fit?", "YES ✅" if res["seeg"]["fits_within_trajectory"] else "NO ⚠️"))
            rows.append(("Required Contact Span (px)", f"{res['seeg']['required_span_px']:.1f}"))

        if res["thermal"]:
            th = res["thermal"]
            rows.append(("Synthetic model temperature, unclipped maximum (°C)", f"{th['max_temp_c']:.2f}"))
            rows.append((f"Temperature heatmap display ceiling (°C; clipped)", f"{th['display_temp_max_c']:.1f}"))
            if th["dose_valid"]:
                rows.append(("Max synthetic CEM43 within heuristic ROI (min)", format_cem43(th['max_cem43_min'])))
                rows.append(("Mean synthetic CEM43 over heuristic ROI (min)", format_cem43(th['mean_cem43_min'])))
                rows.append((f"Modelled dose area within heuristic ROI, CEM43≥{CEM43_ABLATION_THRESHOLD_MIN:.0f} (px²)", f"{th['ablation_area_px']}"))
                rows.append(("Max synthetic CEM43 in image-derived risk mask (min)", format_cem43(th['structure_max_cem43_min'])))
            else:
                rows.append(("CEM43 / dose area", "Unavailable: numerical guard exceeded"))
            rows.append(("Risk-Region Thermal Attenuation (°C)", f"{th['risk_attenuation_c']:.2f}"))

        if pixel_spacing_mm_xy:
            rows.append(("Length (mm; spacing-aware)", f"{measured_length_mm:.2f}"))
            rows.append(("Angle (degrees; spacing-aware image-plane axes)", f"{calculate_geometry(res['entry'], res['target'], pixel_spacing_mm_xy)['angle_physical_deg']:.2f}"))
            rows.append(("Minimum clearance (mm; anisotropic spacing)", f"{measured_clearance_mm:.2f}"))
            if res["seeg"]:
                rows.append(("Geometric contact span (mm; from rounded points)", f"{measured_contact_span_mm:.2f}" if measured_contact_span_mm is not None else "Not available"))

        df_metrics = pd.DataFrame(rows, columns=["Metric", "Value"])
        st.dataframe(df_metrics, hide_index=True, **STRETCH)

        if res["thermal"] and not res["thermal"]["numerical_guard_passed"]:
            st.error(f"🔥 {res['thermal']['model_warning']}")
        if res["thermal"] and res["thermal"]["dose_valid"] and res["thermal"]["structure_max_cem43_min"] >= CEM43_ABLATION_THRESHOLD_MIN:
            st.warning(
                "The unvalidated modelled dose inside image-derived high-signal regions exceeds the configured reference threshold "
                f"(CEM43 ≥ {CEM43_ABLATION_THRESHOLD_MIN:.0f} min); this is not a clinical threshold."
            )
        if res["seeg"] and not res["seeg"]["fits_within_trajectory"]:
            st.warning("Requested contact span is longer than the trajectory; some contacts were not placed.")

        vis_col1, vis_col2 = st.columns(2)

        with vis_col1:
            eyebrow("Planned Trajectory")
            annot_img = draw_trajectory_overlay(
                original_img,
                res["entry"],
                res["target"],
                res["safety"]["color_bgr"],
                seeg_contacts=res["seeg"]["contacts"] if res["seeg"] else None,
            )
            show_bgr(annot_img)

        with vis_col2:
            if procedure_mode == "LITT Thermal Ablation" and res["thermal"]:
                eyebrow("Simplified Static Gaussian Thermal Approximation")
                temp_vis = render_heatmap_with_colorbar(
                    original_img,
                    res["thermal"]["temp_map"],
                    BASELINE_BODY_TEMP_C,
                    TEMP_DISPLAY_MAX_C,
                    cv2.COLORMAP_JET,
                    f"Synthetic temperature display (clipped at {TEMP_DISPLAY_MAX_C:g} C)",
                )
                show_bgr(temp_vis)

                if res["thermal"]["dose_valid"]:
                    cem_vmax = clamp(res["thermal"]["max_cem43_min"], 1.0, CEM43_DISPLAY_MAX_MIN)
                    cem_vis = render_heatmap_with_colorbar(
                        original_img, res["thermal"]["cem43_map"], 0.0, cem_vmax,
                        cv2.COLORMAP_HOT, "Synthetic CEM43 estimate (min; unvalidated)",
                    )
                    show_bgr(cem_vis)
                else:
                    st.warning("CEM43 and dose-area outputs are suppressed because the numerical guard was exceeded.")
            else:
                eyebrow("Distance Map (Clearance)")
                dist_norm = np.clip(dist_map, 0, HEURISTIC_CLEARANCE_SCORE_CAP_PX)
                dist_vis = render_heatmap_with_colorbar(
                    original_img,
                    dist_norm,
                    0.0,
                    HEURISTIC_CLEARANCE_SCORE_CAP_PX,
                    cv2.COLORMAP_VIRIDIS,
                    "Distance to Risk Region (px)",
                )
                show_bgr(dist_vis)

    # --- Live Simulation Section (fragment: reruns do not re-execute the whole app) ---
    with st.container(border=True):
        section_header("5", "Live Navigation Simulation", "Probe telemetry along the trajectory")
        live_simulation(res, original_img, dist_map)

    # --- Export Section ---
    with st.container(border=True):
        section_header("6", "Export & Reporting", "JSON summary and annotated image")

        exp_c1, exp_c2 = st.columns(2)

        with exp_c1:
            export_time = datetime.now().astimezone()
            export_dict = {
                "metadata": {
                    "tool_version": TOOL_VERSION,
                    "timestamp_local": export_time.isoformat(timespec="seconds"),
                    "timestamp_utc": export_time.astimezone(timezone.utc).isoformat(timespec="seconds"),
                    "python_version": platform.python_version(),
                    "library_versions": _library_versions(),
                    "image_sha256": file_sha256,
                    "processing_seconds": prep_data.get("processing_seconds"),
                    "image_dimensions_px": {"width": int(w), "height": int(h)},
                    "image_type": uploaded_file.name.rsplit(".", 1)[-1].lower(),
                    "pixel_intensity_handling": "DICOM modality LUT/rescale and VOI LUT/window applied when present; display scaled per slice to 0-255" if prep_data.get("dicom_metadata") else "8-bit image intensity",
                    "dicom_metadata": prep_data.get("dicom_metadata"),
                    "pixel_spacing_source": pixel_spacing_source,
                    "roi_heuristic_designed_for_modality": _roi_heuristic_modality_description(prep_data.get("dicom_metadata")),
                    "scope": "2D research prototype; not clinically validated",
                    "scientific_limitations": list(SCIENTIFIC_LIMITATIONS),
                    "disclaimer": disclaimer_text,
                    "pixel_spacing_mm": list(pixel_spacing_mm_xy) if pixel_spacing_mm_xy else None,
                    "patient_identifiers_exported": False,
                    "retention_note": "No intentional disk write; Streamlit cache TTL is five minutes with bounded entries, and process/session memory may retain data until eviction or restart.",
                },
                "procedure": procedure_mode,
                "reproducibility": {
                    "settings_schema_version": 1,
                    "analysis_complete": True,
                    "image_sha256": file_sha256,
                    "determinism": {
                        "claim": "not_established",
                        "expected_scope": "same image bytes + same settings + same pinned environment",
                        "verified_repeat_runs": False,
                        "cross_environment_reproducibility_verified": False,
                    },
                    "scientific_limitations": list(SCIENTIFIC_LIMITATIONS),
                    "ct_heuristic_acknowledged": bool((prep_data.get("dicom_metadata") or {}).get("modality") == "CT"),
                    "timestamped_environment": True,
                    "manual_raster_spacing_requires_user_provenance": pixel_spacing_source == "manual isotropic assumption",
                    "uncomputed_or_unavailable": [
                        "3D trajectory/anatomical reconstruction", "anatomical tissue segmentation",
                        "clinical validation", "patient-specific thermal dose validation",
                    ] + ([] if res.get("thermal") is None or res["thermal"].get("dose_valid") else ["CEM43 and dose area suppressed: numerical guard exceeded"]),
                    "configuration": {
                        "procedure": procedure_mode, "use_otsu": use_otsu,
                        "manual_threshold": manual_thresh if not use_otsu else None,
                        "risk_percentile": risk_percentile if use_otsu else None,
                        "roi_strip_radius_px": roi_strip_px, "minimum_risk_area_px2": min_area,
                        "risk_buffer_px": dilation_rad, "target_px": list(res["target"]),
                        "entry_px": list(res["entry"]), "pixel_spacing_mm_xy": list(pixel_spacing_mm_xy) if pixel_spacing_mm_xy else None,
                        "laser_power_w": laser_power if res["thermal"] else None,
                        "duration_s": ablation_time if res["thermal"] else None,
                        "contact_count": num_contacts if res["seeg"] else None,
                        "contact_spacing_px": contact_spacing if res["seeg"] else None,
                    },
                },
                "image_processing": {
                    "roi": "heuristic foreground region (Otsu + opening + largest component + closing + outer-contour hole fill); not tissue segmentation",
                    "roi_strip_radius_px": roi_strip_px,
                    "risk_regions": "intensity-based high-signal regions inside ROI (heuristic, not anatomical)",
                    "auto_threshold": use_otsu,
                    "risk_percentile": risk_percentile if use_otsu else None,
                    "manual_threshold": None if use_otsu else manual_thresh,
                    "min_area_px2": min_area,
                    "buffer_px": dilation_rad,
                    "threshold_final": float(_derive_threshold(blurred_gray, roi_mask, use_otsu, manual_thresh, risk_percentile)),
                    "sensitivity_threshold_shift": SENSITIVITY_THRESHOLD_SHIFT,
                    "sensitivity_roi_held_fixed": True,
                    "roi_closing_kernel_px": ROI_CLOSE_KERNEL_PX,
                    "sensitivity_scenarios": res["safety"]["sensitivity_scenarios"],
                },
                "points": {
                    "target_px": list(res["target"]),
                    "entry_px": list(res["entry"]),
                },
                "optimizer_search": st.session_state.get("auto_info") or st.session_state.get("auto_search_audit"),
                "optimizer_search_counter_note": "Path evaluations count candidates actually sent to full trajectory checks. A coordinate may be path-evaluated in both stages; unique_coordinates_path_evaluated counts their union.",
                "algorithm_settings": {"candidate_points": 72, "local_refinement_radius_px": 3, "optimizer_score_clearance_cap_px": HEURISTIC_CLEARANCE_SCORE_CAP_PX, "optimizer_path_length_penalty": 15.0},
                "trajectory_analysis": {
                    "length_px": res["geometry"]["length"],
                    "length_mm": measured_length_mm,
                    "angle_deg_in_image_coordinates": res["geometry"]["angle"],
                    "angle_deg_spacing_aware": calculate_geometry(res["entry"], res["target"], pixel_spacing_mm_xy)["angle_physical_deg"] if pixel_spacing_mm_xy else None,
                    "min_clearance_px": res["safety"]["min_distance_px"],
                    "min_clearance_mm": measured_clearance_mm,
                    "geometric_contact_span_mm": measured_contact_span_mm,
                    "metric_measurement_note": "Physical EDT clearance and length use horizontal/vertical spacing; heuristic thresholds, contact placement and thermal kernels remain pixel-based.",
                    "clearance_score_pct": res["safety"]["clearance_score_pct"],
                    "clearance_score_note": f"Experimental distance score scaled to {HEURISTIC_CLEARANCE_SCORE_CAP_PX:g}px; not probability or clinical safety",
                    "inside_original_roi": res["original_roi_outside_px"] == 0,
                    "inside_tolerated_roi": res["roi_outside_px"] == 0,
                    "original_roi_outside_path_pixels": res["original_roi_outside_px"],
                    "tolerated_roi_outside_path_pixels": res["roi_outside_px"],
                    "roi_path_tolerance_px": ROI_PATH_TOLERANCE_PX,
                    "auto_optimizer_scoring": "clearance_reward=min(clearance_px/60,1)*100 minus 15*(path_length/image_diagonal); experimental heuristic",
                    "status": res["safety"]["status"],
                    "status_interpretation": "heuristic image-mask and geometric checks only; not a safety determination",
                    "collision": res["safety"]["intersects"],
                    "sensitivity_stability_index": res["safety"]["sensitivity_stability_index"],
                    "variability_score": res["safety"]["variability_score"],
                    "risk_mask_unstable_under_sensitivity": res["safety"]["mask_unstable"],
                    "max_risk_fraction_under_sensitivity": res["safety"]["max_risk_fraction"],
                    "status_flipped_under_sensitivity": res["safety"]["status_flipped"],
                    "collision_flipped_under_sensitivity": res["safety"]["collision_flipped"],
                    "sensitivity_scenarios": res["safety"]["sensitivity_scenarios"],
                    "entry_warning": res.get("entry_warning") or None,
                },
                "model_specific_summary": {
                    "litt_thermal": {
                        "model": "simplified static Gaussian approximation; duration affects CEM43 only; synthetic field, not a physical temperature prediction",
                        "scope_limitations": ["2D single slice", "no 3D orientation reconstruction", "display-scaled DICOM values feed segmentation", "no patient-specific properties or perfusion", "no transient heat transfer", "not clinically validated"],
                        "parameters": {"baseline_temp_c": BASELINE_BODY_TEMP_C, "sigma_base_px": LITT_GAUSSIAN_SIGMA_BASE_PX, "power_to_delta_t_factor": LITT_POWER_TO_DELTA_T_FACTOR, "risk_attenuation_factor": RISK_ATTENUATION_FACTOR, "risk_attenuation_sigma_px": RISK_ATTENUATION_SIGMA_PX, "cem43_r_high": CEM43_R_HIGH, "cem43_r_low": CEM43_R_LOW, "dose_threshold_min": CEM43_ABLATION_THRESHOLD_MIN},
                        "parameter_provenance": "Legacy prototype constants retained for demonstration; not derived from a documented empirical calibration or patient-specific measurements.",
                        "max_temp_c": res["thermal"]["max_temp_c"],
                        "temperature_display_clipped_at_c": res["thermal"]["display_temp_max_c"],
                        "numerical_guard_passed": res["thermal"]["numerical_guard_passed"],
                        "dose_values_available": res["thermal"]["dose_valid"],
                        "scientifically_validated": False,
                        "model_warning": res["thermal"]["model_warning"] or None,
                        "max_synthetic_cem43_over_heuristic_roi_min": res["thermal"]["max_cem43_min"],
                        "modelled_dose_area_px_within_heuristic_roi": res["thermal"]["ablation_area_px"],
                        "area_is_anatomically_segmented_tissue": False,
                        "mean_synthetic_cem43_over_heuristic_roi_min": res["thermal"]["mean_cem43_min"],
                        "risk_region_max_cem43_min": res["thermal"]["structure_max_cem43_min"],
                        "risk_attenuation_c": res["thermal"]["risk_attenuation_c"],
                        "assumptions": ["static Gaussian field", "duration affects dose only", "heuristic attenuation near image intensity regions", "unvalidated coefficients", "CEM43 is derived from synthetic temperature field"],
                        "dose_threshold_is_universal": False,
                        "laser_power_w": laser_power,
                        "duration_s": ablation_time,
                    } if res["thermal"] else None,
                    "seeg_contacts": {
                        "count": res["seeg"]["num_contacts_placed"],
                        "geometric_fit_only": res["seeg"]["fits_within_trajectory"],
                        "anatomical_or_clinical_contact_validation": False,
                        "required_span_px": res["seeg"]["required_span_px"],
                        "contacts": [
                            {"id": c["id"], "pos_px": list(c["pos"]), "depth_px": c["depth_px"]}
                            for c in res["seeg"]["contacts"]
                        ],
                    } if res["seeg"] else None,
                },
            }
            st.download_button(
                label="📥 Download Plan Summary (JSON)",
                data=json.dumps(export_dict, indent=2),
                file_name=f"ThermoSEEG_Plan_{file_id_short}.json",
                mime="application/json",
            )
            if PDF_EXPORT_AVAILABLE:
                pdf_bytes = _build_pdf_report(export_dict)
                st.download_button("📄 Download Research Report (PDF)", data=pdf_bytes,
                                   file_name=f"ThermoSEEG_Report_{file_id_short}.pdf", mime="application/pdf")
            else:
                st.caption("PDF export requires reportlab from requirements.txt.")

        with exp_c2:
            report_img = annot_img.copy()
            header_h = 120
            header = np.zeros((header_h, report_img.shape[1], 3), dtype=np.uint8)
            threshold_report = float(_derive_threshold(blurred_gray, roi_mask, use_otsu, manual_thresh, risk_percentile))
            spacing_report = (f"spacing x/y={pixel_spacing_mm_xy[0]:g}/{pixel_spacing_mm_xy[1]:g} mm/px ({pixel_spacing_source})" if pixel_spacing_mm_xy else "spacing unknown")
            risk_settings = (f"Risk percentile={risk_percentile:g}" if use_otsu else f"Manual threshold={manual_thresh}")
            procedure_settings = (f" | power={laser_power:g}W, duration={ablation_time:g}s" if res["thermal"] else f" | contacts={num_contacts}, spacing={contact_spacing:g}px")
            report_lines = [
                f"ThermoSEEG 2D research report | {procedure_mode} | {w}x{h}px",
                f"Threshold={threshold_report:.1f} | ROI strip={roi_strip_px}px | tolerance={ROI_PATH_TOLERANCE_PX}px | {spacing_report}",
                f"{risk_settings} | min region={min_area}px2 | risk buffer={dilation_rad}px{procedure_settings}",
                "Heuristic masks and simplified models; not clinically validated or for surgical guidance.",
            ]
            for line_idx, line_text in enumerate(report_lines):
                cv2.putText(header, line_text[:180], (10, 22 + line_idx * 27),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1)
            final_report = np.vstack([header, report_img])

            success, encoded = cv2.imencode(".png", final_report)
            if success:
                st.download_button(
                    label="🖼️ Download Annotated Image (PNG)",
                    data=encoded.tobytes(),
                    file_name=f"ThermoSEEG_Image_{file_id_short}.png",
                    mime="image/png",
                )