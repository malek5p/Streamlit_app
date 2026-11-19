"""DICOM single-slice ingestion for the ThermoSEEG 2D research prototype."""
from __future__ import annotations

import io
import math
from typing import Any, Dict, Optional, Tuple

import numpy as np
import cv2

try:
    import pydicom
    from pydicom.pixels import apply_modality_lut, apply_voi_lut
except ImportError:  # Optional for installations that only use raster images.
    pydicom = None
    apply_modality_lut = apply_voi_lut = None


MAX_UPLOAD_BYTES = 50 * 1024 * 1024
MAX_IMAGE_SIDE = 4096
MAX_IMAGE_PIXELS = 16_000_000
MODALITY_PROCESSING_NOTES = {
    "MR": "Decoded; ROI heuristic assumes MRI-like contrast but is not validated as brain segmentation.",
    "CT": "Decoded only. ROI heuristic does not handle CT bone/soft-tissue contrast and is NOT validated for CT.",
}


def inspect_dicom_modality(data: bytes) -> Optional[str]:
    """Read the modality tag without decoding the pixel payload."""
    if pydicom is None or len(data) < 132 or data[128:132] != b"DICM":
        return None
    try:
        ds = pydicom.dcmread(io.BytesIO(data), stop_before_pixels=True)
    except Exception as exc:
        raise ValueError("DICOM file header is invalid or unsupported.") from exc
    return str(getattr(ds, "Modality", "")).upper() or None


def decode_dicom_slice(data: bytes, max_pixels: int = MAX_IMAGE_PIXELS) -> Optional[Dict[str, Any]]:
    """Decode a supported monochrome 2D MR/CT slice into an 8-bit display array.

    Applies Modality LUT/Rescale and VOI LUT/window where present. The returned
    display array is not quantitative data; segmentation uses this display mapping.
    Does not return or export patient identifiers.
    """
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError("Upload exceeds the 50 MB file-size limit.")
    if isinstance(max_pixels, bool) or not isinstance(max_pixels, int) or max_pixels < 1:
        raise ValueError("DICOM pixel limit must be a positive integer.")
    if pydicom is None:
        if data[128:132] == b"DICM":
            raise ValueError("DICOM input requires pydicom; install the pinned project requirements.")
        return None
    has_preamble = data[128:132] == b"DICM"
    try:
        ds = pydicom.dcmread(io.BytesIO(data), force=not has_preamble)
    except Exception:
        if has_preamble:
            raise ValueError("DICOM file header is invalid or unsupported.")
        return None
    if not hasattr(ds, "PixelData"):
        return None
    modality = str(getattr(ds, "Modality", "")).upper()
    if modality not in {"MR", "CT"}:
        raise ValueError(f"Unsupported DICOM modality '{modality or 'unspecified'}'; expected MR or CT.")
    if int(getattr(ds, "SamplesPerPixel", 1)) != 1:
        raise ValueError("Color DICOM is unsupported; upload a single-channel MR/CT slice.")
    if int(getattr(ds, "NumberOfFrames", 1)) != 1:
        raise ValueError("Multi-frame DICOM is unsupported; upload one 2D slice.")
    try:
        rows, columns = int(ds.Rows), int(ds.Columns)
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("DICOM Rows/Columns are missing or invalid.") from exc
    if rows < 2 or columns < 2:
        raise ValueError("DICOM image dimensions are too small for analysis.")
    if rows > MAX_IMAGE_SIDE or columns > MAX_IMAGE_SIDE or rows * columns > max_pixels:
        raise ValueError(f"DICOM image dimensions exceed the supported limit (4096 px per side and {max_pixels:,} pixels).")
    photometric = str(getattr(ds, "PhotometricInterpretation", "MONOCHROME2")).upper()
    if photometric not in {"MONOCHROME1", "MONOCHROME2"}:
        raise ValueError(f"Unsupported DICOM photometric interpretation: {photometric}.")
    if hasattr(ds, "RescaleSlope") or hasattr(ds, "RescaleIntercept"):
        try:
            slope = float(getattr(ds, "RescaleSlope", 1.0))
            intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("DICOM RescaleSlope/RescaleIntercept must be numeric.") from exc
        if not math.isfinite(slope) or not math.isfinite(intercept) or slope == 0:
            raise ValueError("DICOM RescaleSlope must be finite and non-zero; RescaleIntercept must be finite.")
    try:
        pixels = np.asarray(ds.pixel_array)
    except Exception as exc:
        syntax = str(getattr(getattr(ds, "file_meta", None), "TransferSyntaxUID", "unknown"))
        raise ValueError(f"DICOM pixel data could not be decoded (Transfer Syntax {syntax}); an unsupported syntax or missing codec may be responsible.") from exc
    if pixels.ndim != 2 or pixels.size == 0 or not np.isfinite(pixels).all():
        raise ValueError("DICOM pixel data must be a non-empty finite 2D array.")
    if pixels.shape != (rows, columns):
        raise ValueError("Decoded DICOM pixel dimensions do not match the Rows/Columns header.")
    try:
        modality_pixels = np.asarray(apply_modality_lut(pixels, ds), dtype=np.float64)
    except Exception as exc:
        raise ValueError(f"DICOM Modality LUT/Rescale could not be applied: {type(exc).__name__}.") from exc
    has_voi = bool(getattr(ds, "VOILUTSequence", None)) or (
        getattr(ds, "WindowCenter", None) is not None and getattr(ds, "WindowWidth", None) is not None
    )
    try:
        display_pixels = np.asarray(apply_voi_lut(modality_pixels, ds), dtype=np.float64) if has_voi else modality_pixels
    except Exception as exc:
        raise ValueError(f"DICOM VOI LUT/window could not be applied: {type(exc).__name__}.") from exc
    if not np.isfinite(display_pixels).all():
        raise ValueError("DICOM modality/VOI transform produced non-finite pixels.")
    low, high = float(display_pixels.min()), float(display_pixels.max())
    if high <= low:
        raise ValueError("DICOM image has no intensity range and cannot be segmented.")
    gray = np.clip((display_pixels - low) * (255.0 / (high - low)), 0, 255).astype(np.uint8)
    if photometric == "MONOCHROME1":
        gray = 255 - gray
    spacing_xy = None
    if getattr(ds, "PixelSpacing", None) is not None:
        try:
            if len(ds.PixelSpacing) != 2:
                raise ValueError
            row_mm, col_mm = map(float, ds.PixelSpacing)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("DICOM PixelSpacing must contain exactly two numeric values [row, column].") from exc
        if not (np.isfinite(row_mm) and np.isfinite(col_mm) and row_mm > 0 and col_mm > 0):
            raise ValueError("DICOM PixelSpacing must contain two positive finite values.")
        spacing_xy = (col_mm, row_mm)
    orientation = None
    if getattr(ds, "ImageOrientationPatient", None) is not None:
        orientation = np.asarray(ds.ImageOrientationPatient, dtype=np.float64)
        if orientation.shape != (6,) or not np.isfinite(orientation).all():
            raise ValueError("DICOM ImageOrientationPatient must contain six finite direction cosines.")
        row_dir, col_dir = orientation[:3], orientation[3:]
        if not (np.isclose(np.linalg.norm(row_dir), 1.0, atol=1e-3)
                and np.isclose(np.linalg.norm(col_dir), 1.0, atol=1e-3)
                and np.isclose(np.dot(row_dir, col_dir), 0.0, atol=1e-3)):
            raise ValueError("DICOM ImageOrientationPatient direction cosines are not orthonormal.")
    position = None
    if getattr(ds, "ImagePositionPatient", None) is not None:
        position = np.asarray(ds.ImagePositionPatient, dtype=np.float64)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError("DICOM ImagePositionPatient must contain three finite values.")
    thickness = None
    if getattr(ds, "SliceThickness", None) is not None:
        thickness = float(ds.SliceThickness)
        if not np.isfinite(thickness) or thickness <= 0:
            raise ValueError("DICOM SliceThickness, when provided, must be positive and finite.")
    metadata = {
        "modality": modality,
        "photometric_interpretation": photometric,
        "modality_transform_applied": bool(getattr(ds, "ModalityLUTSequence", None)) or hasattr(ds, "RescaleSlope"),
        "voi_transform_applied": has_voi,
        "image_orientation_patient": orientation.tolist() if orientation is not None else None,
        "image_position_patient": position.tolist() if position is not None else None,
        "slice_thickness_mm": thickness,
        "spatial_orientation_used_in_analysis": False,
        "display_conversion_note": "Modality/VOI values are per-slice scaled to 8-bit for heuristic segmentation; this is not quantitative intensity analysis.",
        "quantitative_intensity_preserved": False,
        "intensity_scaling": "per-slice min-max display scaling to 0-255 (not original HU/signal values)",
        "roi_heuristic_designed_for_modality": modality == "MR",
        "modality_processing_note": MODALITY_PROCESSING_NOTES[modality],
    }
    return {"gray": gray, "pixel_spacing_mm_xy": spacing_xy, "metadata": metadata}
