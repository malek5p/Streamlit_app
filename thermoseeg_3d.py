"""3D engine for the ThermoSEEG research prototype (no UI, no plotting).

Scope (read before trusting any number)
---------------------------------------
* Volumes live on a regular millimetre grid in RAS+ world coordinates
  (x: Right+, y: Anterior+, z: Superior+).  Oblique acquisitions are NOT resampled
  to a true oblique frame; a note is recorded when the affine is not axis-aligned.
* Brain / CSF / vessel labels come from user-supplied masks when available.  The
  heuristic fallbacks (Otsu brain, intensity CSF, Hessian vesselness) are
  convenience tools, NOT validated segmentations.
* Thermal model: transient Pennes bioheat equation on the grid
      rho*c dT/dt = div(k grad T) - h (T - Ta) + Q(x, t)
  with literature-order tissue constants that are NOT calibrated to any patient.
  CSF is modelled as a high-effective-conductivity, non-perfused region; vessels as
  a heat-sink term.  No flow solver, no temperature-dependent properties, no
  vaporisation physics.  Above 100 C the model is flagged invalid and CEM43 is
  suppressed (never reported as a number).
* Nothing here is clinically validated.  Research / education only.
"""
from __future__ import annotations

import gzip
import hashlib
import io
import math
import re
import struct
import time
import zipfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage as ndi

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LABEL_BG, LABEL_BRAIN, LABEL_CSF, LABEL_VESSEL = 0, 1, 2, 3
LABEL_NAMES = {LABEL_BG: "background", LABEL_BRAIN: "brain tissue", LABEL_CSF: "CSF", LABEL_VESSEL: "vessel"}

BODY_TEMP_C = 37.0
ARTERIAL_TEMP_C = 37.0
MAX_VALID_TEMP_C = 100.0
CEM43_REF_C = 43.0
CEM43_ABLATION_MIN = 240.0
CEM43_SATURATION_MIN = 1.0e6   # above this the dose number is meaningless; shown as ">1e6"

MAX_UPLOAD_BYTES = 200 * 1024 * 1024
MAX_DECOMPRESSED_BYTES = 800 * 1024 * 1024
MAX_INPUT_VOXELS = 100_000_000
MAX_WORKING_VOXELS = 4_000_000

# Tissue constants (SI). Literature-order values, unvalidated; sliders scale them.
TISSUE = {
    LABEL_BRAIN: {"k": 0.51, "rho": 1040.0, "c": 3630.0, "perfusion_1_per_s": 0.0087},
    LABEL_CSF: {"k": 0.57, "rho": 1007.0, "c": 4096.0, "perfusion_1_per_s": 0.0},
    LABEL_VESSEL: {"k": 0.52, "rho": 1050.0, "c": 3617.0, "perfusion_1_per_s": 0.0},
}
BLOOD_RHO = 1050.0
BLOOD_C = 3617.0

DEFAULT_PRESET = "Literature-order (default)"
PARITY_PRESET = "Team desktop parity"
# Constant sets (SI). The parity preset reproduces the defaults found in the team's desktop solver
# (thermal_field.py): tissue 1050/3600/0.51, blood 1060/3600, perfusion parameter 0.01 (interpreted here as
# 1/s, UNCONFIRMED), CSF 1007/4180/0.60 and an empirical CSF sink of 15000 (interpreted as W/m^3/K, UNCALIBRATED).
PRESETS = {
    DEFAULT_PRESET: {
        "brain": {"k": 0.51, "rho": 1040.0, "c": 3630.0, "perfusion_1_per_s": 0.0087},
        "csf": {"k": 0.57, "rho": 1007.0, "c": 4096.0, "perfusion_1_per_s": 0.0},
        "vessel": {"k": 0.52, "rho": 1050.0, "c": 3617.0, "perfusion_1_per_s": 0.0},
        "blood_rho": 1050.0, "blood_c": 3617.0, "csf_k_multiplier": 4.0, "csf_sink_w_m3k": 0.0},
    PARITY_PRESET: {
        "brain": {"k": 0.51, "rho": 1050.0, "c": 3600.0, "perfusion_1_per_s": 0.01},
        "csf": {"k": 0.60, "rho": 1007.0, "c": 4180.0, "perfusion_1_per_s": 0.0},
        "vessel": {"k": 0.51, "rho": 1060.0, "c": 3600.0, "perfusion_1_per_s": 0.0},
        "blood_rho": 1060.0, "blood_c": 3600.0, "csf_k_multiplier": 1.0, "csf_sink_w_m3k": 15000.0},
}

# Experimental geometric thresholds (NOT clinical safety limits)
DANGER_CLEARANCE_MM = 3.0
WARNING_CLEARANCE_MM = 7.5
CSF_CROSSING_WARN_MM = 3.0
RISK_PERCENTILE_DEFAULT = 97          # heuristic starting point (same as the 2D prototype), not a scientific constant
RISK_MASK_MAX_FRACTION = 0.15         # risk mask above this share of the brain = threshold sits on a tissue plateau
DEFAULT_SAFETY_BUFFER_MM = 5.0        # 2D default of 5 px, read as ~5 mm; geometric setting, not a clinical margin


class ThermoSEEG3DError(ValueError):
    """User-facing error (bad file, mismatched grids, ...)."""


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------
@dataclass(eq=False)
class Volume3D:
    anatomy: np.ndarray            # float32 in [0, 1]
    labels: np.ndarray             # uint8 (see LABEL_*)
    spacing: np.ndarray            # (3,) mm per voxel along x, y, z
    origin: np.ndarray             # (3,) world mm of voxel (0, 0, 0) centre
    source: str = "unknown"
    notes: List[str] = field(default_factory=list)
    default_target: Optional[np.ndarray] = None
    sha256: str = ""

    @property
    def shape(self) -> Tuple[int, int, int]:
        return tuple(int(v) for v in self.labels.shape)

    @property
    def bounds_mm(self) -> Tuple[np.ndarray, np.ndarray]:
        lo = self.origin - 0.5 * self.spacing
        hi = self.origin + (np.array(self.shape) - 0.5) * self.spacing
        return lo, hi

    def world_to_index(self, p_mm) -> np.ndarray:
        return (np.asarray(p_mm, dtype=float) - self.origin) / self.spacing

    def index_to_world(self, idx) -> np.ndarray:
        return self.origin + np.asarray(idx, dtype=float) * self.spacing

    def axis_coords(self, axis: int) -> np.ndarray:
        return self.origin[axis] + np.arange(self.shape[axis]) * self.spacing[axis]

    def voxel_volume_mm3(self) -> float:
        return float(np.prod(self.spacing))

    def stats_ml(self) -> Dict[str, float]:
        vv = self.voxel_volume_mm3() / 1000.0
        return {LABEL_NAMES[l]: round(float(np.count_nonzero(self.labels == l)) * vv, 2)
                for l in (LABEL_BRAIN, LABEL_CSF, LABEL_VESSEL)}

    def label_at(self, p_mm) -> int:
        idx = np.rint(self.world_to_index(p_mm)).astype(int)
        if np.any(idx < 0) or np.any(idx >= np.array(self.shape)):
            return LABEL_BG
        return int(self.labels[tuple(idx)])


@dataclass
class ThermalParams:
    power_w: float = 3.0
    exposure_s: float = 60.0
    cooldown_s: float = 30.0
    absorbed_fraction: float = 1.0
    active_length_mm: float = 10.0
    source_radius_mm: float = 4.0
    perfusion_scale: float = 1.0
    vessel_sink_scale: float = 10.0
    csf_k_multiplier: float = 4.0
    margin_mm: float = 25.0
    n_snapshots: int = 12
    preset: str = DEFAULT_PRESET
    csf_sink_w_m3k: float = 0.0
    conductivity_scale: float = 1.0

    def key(self) -> Tuple:
        return (self.preset,) + tuple(round(float(v), 6) for v in (
            self.power_w, self.exposure_s, self.cooldown_s, self.absorbed_fraction,
            self.active_length_mm, self.source_radius_mm, self.perfusion_scale,
            self.vessel_sink_scale, self.csf_k_multiplier, self.margin_mm, self.n_snapshots,
            self.csf_sink_w_m3k, self.conductivity_scale))


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        raise ThermoSEEG3DError("Zero-length direction vector.")
    return v / n


def _dist_to_segment(X, Y, Z, a, b) -> np.ndarray:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ab = b - a
    denom = float(ab @ ab)
    px, py, pz = X - a[0], Y - a[1], Z - a[2]
    if denom < 1e-12:
        return np.sqrt(px ** 2 + py ** 2 + pz ** 2)
    t = np.clip((px * ab[0] + py * ab[1] + pz * ab[2]) / denom, 0.0, 1.0)
    return np.sqrt((px - t * ab[0]) ** 2 + (py - t * ab[1]) ** 2 + (pz - t * ab[2]) ** 2)


def direction_from_angles(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    """Unit vector pointing from the target OUTWARD toward the entry.
    azimuth 0 deg = +x (Right), 90 deg = +y (Anterior); elevation +90 deg = +z (Superior)."""
    az, el = math.radians(azimuth_deg), math.radians(elevation_deg)
    return np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])


def angles_from_direction(d) -> Tuple[float, float]:
    d = unit(d)
    return float(math.degrees(math.atan2(d[1], d[0])) % 360.0), float(math.degrees(math.asin(np.clip(d[2], -1, 1))))


def sample_line(entry, target, step_mm: float = 0.5) -> np.ndarray:
    entry, target = np.asarray(entry, float), np.asarray(target, float)
    length = float(np.linalg.norm(target - entry))
    n = max(2, int(math.ceil(length / step_mm)) + 1)
    t = np.linspace(0.0, 1.0, n)[:, None]
    return entry[None, :] * (1 - t) + target[None, :] * t


def find_entry_on_surface(vol: Volume3D, target, outward_dir, step_mm: float = 0.5,
                          max_len_mm: float = 300.0) -> Optional[np.ndarray]:
    """March from the target along `outward_dir` until the intracranial mask is left.
    Returns the last intracranial sample (the surface entry) or None if the target is outside."""
    d = unit(outward_dir)
    target = np.asarray(target, float)
    s = np.arange(0.0, max_len_mm, step_mm)
    pts = target[None, :] + s[:, None] * d[None, :]
    idx = np.rint((pts - vol.origin) / vol.spacing).astype(int)
    inb = np.all((idx >= 0) & (idx < np.array(vol.shape)), axis=1)
    inside = np.zeros(len(s), dtype=bool)
    ii = idx[inb]
    inside[inb] = vol.labels[ii[:, 0], ii[:, 1], ii[:, 2]] > 0
    if not inside[0]:
        return None
    outside_idx = np.flatnonzero(~inside)
    if outside_idx.size == 0:
        return None
    return pts[outside_idx[0] - 1]


def vessel_distance_map_mm(vol: Volume3D) -> Optional[np.ndarray]:
    v = vol.labels == LABEL_VESSEL
    if not v.any():
        return None
    return ndi.distance_transform_edt(~v, sampling=vol.spacing).astype(np.float32)


def analyze_trajectory(vol: Volume3D, entry, target, vessel_dist: Optional[np.ndarray],
                       step_mm: float = 0.5, buffer_mm: float = 0.0) -> Dict[str, Any]:
    entry, target = np.asarray(entry, float), np.asarray(target, float)
    pts = sample_line(entry, target, step_mm)
    idx = np.rint((pts - vol.origin) / vol.spacing).astype(int)
    inb = np.all((idx >= 0) & (idx < np.array(vol.shape)), axis=1)
    lab = np.zeros(len(pts), dtype=np.uint8)
    ii = idx[inb]
    lab[inb] = vol.labels[ii[:, 0], ii[:, 1], ii[:, 2]]
    seg = float(np.linalg.norm(target - entry)) / max(1, len(pts) - 1)
    length = float(np.linalg.norm(target - entry))
    min_clear = None
    if vessel_dist is not None:
        vals = np.full(len(pts), np.inf, dtype=float)
        vals[inb] = vessel_dist[ii[:, 0], ii[:, 1], ii[:, 2]]
        min_clear = float(vals.min())
    vessel_hit = bool(np.any(lab == LABEL_VESSEL))
    csf_len = float(np.count_nonzero(lab == LABEL_CSF)) * seg
    outside_len = float(np.count_nonzero(lab == LABEL_BG)) * seg
    d = unit(target - entry) if length > 0 else np.zeros(3)
    az, el = angles_from_direction(-d) if length > 0 else (0.0, 0.0)
    reasons = []
    status = "HEURISTIC_CHECKS_PASSED"
    if vessel_hit:
        status = "DANGER"; reasons.append("trajectory passes through a vessel label")
    elif min_clear is not None and min_clear < DANGER_CLEARANCE_MM:
        status = "DANGER"; reasons.append(f"vessel clearance {min_clear:.1f} mm < {DANGER_CLEARANCE_MM:.0f} mm")
    elif min_clear is not None and buffer_mm > 0 and min_clear <= buffer_mm:
        status = "DANGER"; reasons.append(f"path enters the {buffer_mm:.1f} mm safety buffer around obstacles ({min_clear:.1f} mm)")
    if outside_len > 1.0:
        status = "DANGER"; reasons.append(f"{outside_len:.1f} mm of the path is outside the intracranial mask")
    if status != "DANGER":
        if min_clear is not None and min_clear < WARNING_CLEARANCE_MM:
            status = "WARNING"; reasons.append(f"vessel clearance {min_clear:.1f} mm < {WARNING_CLEARANCE_MM:.1f} mm")
        if csf_len > CSF_CROSSING_WARN_MM:
            status = "WARNING"; reasons.append(f"{csf_len:.1f} mm of the path crosses CSF")
    return {
        "length_mm": round(length, 2),
        "azimuth_deg": round(az, 1), "elevation_deg": round(el, 1),
        "min_vessel_clearance_mm": None if min_clear is None else round(min_clear, 2),
        "vessel_intersection": vessel_hit,
        "csf_crossing_mm": round(csf_len, 2),
        "outside_mask_mm": round(outside_len, 2),
        "safety_buffer_mm": float(buffer_mm),
        "status": status, "reasons": reasons,
        "thresholds_note": "Experimental geometric thresholds, not clinical safety limits.",
    }


def seeg_contacts(entry, target, n_contacts: int, spacing_mm: float) -> Dict[str, Any]:
    entry, target = np.asarray(entry, float), np.asarray(target, float)
    length = float(np.linalg.norm(target - entry))
    if length <= 0:
        return {"contacts": [], "fits": False, "required_span_mm": 0.0}
    u = (target - entry) / length
    span = (n_contacts - 1) * spacing_mm
    contacts = []
    for i in range(n_contacts):
        depth = i * spacing_mm
        if depth > length:
            break
        contacts.append({"id": i + 1, "depth_mm": round(depth, 2),
                         "pos_mm": [round(float(v), 2) for v in (target - u * depth)]})
    return {"contacts": contacts, "fits": span <= length, "required_span_mm": round(span, 2)}


def point_segment_distance(p, a, b) -> float:
    p, a, b = (np.asarray(v, float) for v in (p, a, b))
    ab = b - a
    den = float(ab @ ab)
    if den <= 1e-12:
        return float(np.linalg.norm(p - a))
    t = float(np.clip((p - a) @ ab / den, 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def segment_segment_distance(a0, a1, b0, b1) -> float:
    """Minimum distance between two 3D segments (Ericson, Real-Time Collision Detection)."""
    p1, q1, p2, q2 = (np.asarray(v, float) for v in (a0, a1, b0, b1))
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = float(d1 @ d1), float(d2 @ d2), float(d2 @ r)
    eps = 1e-12
    if a <= eps and e <= eps:
        return float(np.linalg.norm(r))
    if a <= eps:
        s, t = 0.0, float(np.clip(f / e, 0, 1))
    else:
        c = float(d1 @ r)
        if e <= eps:
            t, s = 0.0, float(np.clip(-c / a, 0, 1))
        else:
            b = float(d1 @ d2)
            denom = a * e - b * b
            s = float(np.clip((b * f - c * e) / denom, 0, 1)) if denom > eps else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t, s = 0.0, float(np.clip(-c / a, 0, 1))
            elif t > 1.0:
                t, s = 1.0, float(np.clip((b - c) / a, 0, 1))
    return float(np.linalg.norm((p1 + d1 * s) - (p2 + d2 * t)))


def sphere_surface(center, radius: float, n: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """UV sphere as (verts, faces) in mm for plotting a planning/structure sphere."""
    u = np.linspace(0, 2 * np.pi, n, endpoint=False)
    v = np.linspace(0.0, np.pi, n // 2 + 1)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    verts = np.stack([np.cos(uu) * np.sin(vv), np.sin(uu) * np.sin(vv), np.cos(vv)], axis=-1).reshape(-1, 3)
    verts = np.asarray(center, float)[None, :] + radius * verts
    nv = len(v)
    faces = []
    for i in range(n):
        for j in range(nv - 1):
            a, b = i * nv + j, ((i + 1) % n) * nv + j
            faces += [[a, b, b + 1], [a, b + 1, a + 1]]
    return verts.astype(np.float32), np.array(faces, dtype=np.int32)


def _local_world_grid(vol: Volume3D, lo_mm, hi_mm):
    lo = np.maximum(np.floor(vol.world_to_index(lo_mm)).astype(int), 0)
    hi = np.minimum(np.ceil(vol.world_to_index(hi_mm)).astype(int) + 1, np.array(vol.shape))
    box = tuple((int(lo[i]), int(hi[i])) for i in range(3))
    ax = [vol.origin[i] + np.arange(box[i][0], box[i][1]) * vol.spacing[i] for i in range(3)]
    X, Y, Z = np.meshgrid(*ax, indexing="ij")
    return box, X, Y, Z


def ablation_capsule_metrics(vol: Volume3D, tip, direction_out, active_length_mm: float,
                             radius_mm: float) -> Dict[str, Any]:
    """Geometric applicator capsule (cylinder + hemispherical caps): volume and CSF exposure fraction."""
    tip = np.asarray(tip, float)
    far = tip + unit(direction_out) * active_length_mm
    pad = radius_mm + 2 * float(np.max(vol.spacing))
    box, X, Y, Z = _local_world_grid(vol, np.minimum(tip, far) - pad, np.maximum(tip, far) + pad)
    d = _dist_to_segment(X, Y, Z, tip, far)
    m = d <= radius_mm
    lab = vol.labels[tuple(slice(a, b) for a, b in box)]
    vv = vol.voxel_volume_mm3()
    n, n_csf = int(m.sum()), int((m & (lab == LABEL_CSF)).sum())
    return {"capsule_volume_mm3": n * vv, "capsule_csf_volume_mm3": n_csf * vv,
            "csf_exposure_fraction": (n_csf / n) if n else 0.0,
            "analytic_volume_mm3": math.pi * radius_mm ** 2 * active_length_mm + 4.0 / 3.0 * math.pi * radius_mm ** 3}


def capsule_surface(tip, far, radius_mm: float, step_mm: float = 0.75) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Smooth surface of the applicator capsule (cylinder + hemispherical caps) for plotting."""
    from skimage import measure
    tip, far = np.asarray(tip, float), np.asarray(far, float)
    pad = radius_mm + 2 * step_mm
    lo, hi = np.minimum(tip, far) - pad, np.maximum(tip, far) + pad
    ax = [np.arange(lo[i], hi[i] + step_mm, step_mm) for i in range(3)]
    X, Y, Z = np.meshgrid(*ax, indexing="ij")
    field = (radius_mm - _dist_to_segment(X, Y, Z, tip, far)).astype(np.float32)
    try:
        verts, faces, _, _ = measure.marching_cubes(field, level=0.0, spacing=(step_mm,) * 3)
    except (ValueError, RuntimeError):
        return None
    return (verts + lo[None, :]).astype(np.float32), faces.astype(np.int32)


def region_thermal_stats(vol: Volume3D, result: Dict[str, Any], center, radius_mm: float) -> Dict[str, Any]:
    """Statistics inside a user-defined sphere, using the WITH-CSF thermal run."""
    wc = result["with_csf"]
    box = wc["box"]
    ax = [vol.origin[i] + np.arange(box[i][0], box[i][1]) * vol.spacing[i] for i in range(3)]
    X, Y, Z = np.meshgrid(*ax, indexing="ij")
    c = np.asarray(center, float)
    m = (X - c[0]) ** 2 + (Y - c[1]) ** 2 + (Z - c[2]) ** 2 <= radius_mm ** 2
    n = int(m.sum())
    out: Dict[str, Any] = {"voxels": n, "volume_mm3": n * vol.voxel_volume_mm3(), "inside_thermal_box": n > 0}
    if n == 0:
        return out
    out["max_temp_c"] = float(wc["peak"][m].max())
    if wc["dose"] is None:
        out["max_cem43_min"] = None
        out["coverage_pct"] = None
    else:
        out["max_cem43_min"] = float(wc["dose"][m].max())
        out["coverage_pct"] = 100.0 * float(np.count_nonzero(wc["dose"][m] >= CEM43_ABLATION_MIN)) / n
    return out


# ---------------------------------------------------------------------------
# Meshes
# ---------------------------------------------------------------------------
def surface_mesh(field_or_mask: np.ndarray, level: float, spacing, origin,
                 smooth_sigma: float = 0.0) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Marching-cubes iso-surface in world mm. Returns (verts (N,3), faces (M,3)) or None."""
    from skimage import measure
    f = np.asarray(field_or_mask, dtype=np.float32)
    if smooth_sigma > 0:
        f = ndi.gaussian_filter(f, smooth_sigma)
    if not (f.min() < level < f.max()):
        return None
    pad_val = float(f.min()) if f.min() < level else 0.0
    fp = np.pad(f, 1, mode="constant", constant_values=pad_val)
    try:
        verts, faces, _, _ = measure.marching_cubes(fp, level=level, spacing=tuple(float(s) for s in spacing))
    except (ValueError, RuntimeError):
        return None
    verts = verts - np.asarray(spacing, float)[None, :] + np.asarray(origin, float)[None, :]
    return verts.astype(np.float32), faces.astype(np.int32)


# ---------------------------------------------------------------------------
# Synthetic phantom (so the app is testable with no patient data)
# ---------------------------------------------------------------------------
def _hash_volume(anat, labels, spacing, origin, tag: str) -> str:
    h = hashlib.sha256()
    h.update(tag.encode())
    h.update(np.ascontiguousarray(labels).tobytes())
    h.update(np.ascontiguousarray((anat * 255).astype(np.uint8)).tobytes())
    h.update(np.asarray(spacing, float).tobytes())
    h.update(np.asarray(origin, float).tobytes())
    return h.hexdigest()


def make_phantom(spacing_mm: float = 2.0, seed: int = 0) -> Volume3D:
    s = float(spacing_mm)
    half = np.array([76.0, 94.0, 70.0])
    shape = np.maximum(8, np.round(2 * half / s)).astype(int)
    spacing = np.full(3, 2 * half[0] / shape[0])
    spacing = 2 * half / shape
    origin = -half + spacing / 2
    ax = [origin[i] + np.arange(shape[i]) * spacing[i] for i in range(3)]
    X, Y, Z = np.meshgrid(*ax, indexing="ij")
    a, b, c = 66.0, 84.0, 58.0
    r2 = (X / a) ** 2 + (Y / b) ** 2 + (Z / c) ** 2
    env = r2 <= 1.0
    csf = env & (r2 >= 0.95 ** 2)
    for cx in (-11.0, 11.0):
        csf |= ((X - cx) / 6.0) ** 2 + ((Y + 8.0) / 27.0) ** 2 + ((Z - 10.0) / 9.0) ** 2 <= 1.0
    csf |= (X / 1.5) ** 2 + ((Y + 5.0) / 8.0) ** 2 + ((Z - 2.0) / 8.0) ** 2 <= 1.0
    for (cx, cy, cz) in ((-58, 30, 15), (58, -20, 20), (-50, -40, 30)):
        csf |= ((X - cx) / 1.6) ** 2 + ((Y - cy) / 14.0) ** 2 + ((Z - cz) / 12.0) ** 2 <= 1.0
    csf &= env
    vessel = np.zeros(shape, dtype=bool)
    for (p0, p1, rad) in (((-8, -4, -2), (-48, 6, 6), 1.6), ((12, -30, 10), (34, 18, 38), 1.3),
                          ((0, 40, -10), (0, 40, 30), 1.0), ((-30, -40, 20), (-44, -10, 40), 1.2)):
        vessel |= _dist_to_segment(X, Y, Z, p0, p1) <= rad
    vessel &= env
    csf &= ~vessel
    labels = np.zeros(shape, dtype=np.uint8)
    labels[env] = LABEL_BRAIN
    labels[csf] = LABEL_CSF
    labels[vessel] = LABEL_VESSEL
    rng = np.random.default_rng(seed)
    anat = np.full(shape, 0.05, dtype=np.float32)
    anat[labels == LABEL_BRAIN] = 0.62
    anat[labels == LABEL_CSF] = 0.15
    anat[labels == LABEL_VESSEL] = 0.92
    anat = ndi.gaussian_filter(anat, 0.8) + rng.normal(0, 0.015, shape).astype(np.float32)
    anat = np.clip(anat, 0, 1).astype(np.float32)
    vol = Volume3D(anat, labels, spacing, origin, source="synthetic phantom",
                   notes=["Synthetic phantom: ellipsoid 'brain', CSF shell/ventricles/sulci, four tubular 'vessels'. "
                          "Not anatomy; for demonstration and testing only."],
                   default_target=np.array([-22.0, 4.0, 10.0]))
    vol.sha256 = _hash_volume(anat, labels, spacing, origin, "phantom")
    return vol


# ---------------------------------------------------------------------------
# NIfTI loading (built-in NIfTI-1 reader, exact affine resampling) + heuristics
# ---------------------------------------------------------------------------
_NIFTI_DTYPES = {2: np.uint8, 4: np.int16, 8: np.int32, 16: np.float32, 64: np.float64, 256: np.int8,
                 512: np.uint16, 768: np.uint32, 1024: np.int64, 1280: np.uint64}


def _decompress(raw: bytes, label: str) -> bytes:
    if len(raw) > MAX_UPLOAD_BYTES:
        raise ThermoSEEG3DError(f"{label}: file exceeds the {MAX_UPLOAD_BYTES // 2**20} MB limit.")
    if raw[:2] == b"\x1f\x8b":
        try:
            with gzip.GzipFile(fileobj=io.BytesIO(raw)) as gz:
                raw = gz.read(MAX_DECOMPRESSED_BYTES + 1)
        except Exception as exc:
            raise ThermoSEEG3DError(f"{label}: corrupt gzip file ({type(exc).__name__}).") from exc
        if len(raw) > MAX_DECOMPRESSED_BYTES:
            raise ThermoSEEG3DError(f"{label}: decompressed size exceeds the limit.")
    return raw


def _quaternion_affine(b, c, d, qx, qy, qz, pixdim) -> np.ndarray:
    a2 = 1.0 - (b * b + c * c + d * d)
    a = math.sqrt(a2) if a2 > 1e-7 else 0.0
    if a2 <= 1e-7:
        n = math.sqrt(b * b + c * c + d * d) or 1.0
        b, c, d = b / n, c / n, d / n
    R = np.array([[a * a + b * b - c * c - d * d, 2 * b * c - 2 * a * d, 2 * b * d + 2 * a * c],
                  [2 * b * c + 2 * a * d, a * a + c * c - b * b - d * d, 2 * c * d - 2 * a * b],
                  [2 * b * d - 2 * a * c, 2 * c * d + 2 * a * b, a * a + d * d - c * c - b * b]])
    qfac = -1.0 if pixdim[0] < 0 else 1.0
    R[:, 2] *= qfac
    A = np.eye(4)
    A[:3, :3] = R * np.array(pixdim[1:4])[None, :]
    A[:3, 3] = (qx, qy, qz)
    return A


def _parse_nifti1(raw: bytes, label: str) -> Optional[Tuple[np.ndarray, np.ndarray, List[str]]]:
    """Returns (data float32, affine 4x4, notes) or None if the header is not NIfTI-1 (e.g. NIfTI-2)."""
    if len(raw) < 352:
        raise ThermoSEEG3DError(f"{label}: too small to be a NIfTI file.")
    endian = None
    for e in ("<", ">"):
        if struct.unpack(e + "i", raw[:4])[0] == 348:
            endian = e
    if endian is None:
        return None
    e = endian
    dim = struct.unpack(e + "8h", raw[40:56])
    ndim = dim[0]
    if not 3 <= ndim <= 4 or (ndim == 4 and dim[4] != 1):
        raise ThermoSEEG3DError(f"{label}: expected a single 3D volume, header dim = {dim[:ndim + 1]}.")
    shape = tuple(int(v) for v in dim[1:4])
    if min(shape) < 2 or int(np.prod(shape, dtype=np.int64)) > MAX_INPUT_VOXELS:
        raise ThermoSEEG3DError(f"{label}: invalid or oversized volume shape {shape}.")
    datatype = struct.unpack(e + "h", raw[70:72])[0]
    if datatype not in _NIFTI_DTYPES:
        raise ThermoSEEG3DError(f"{label}: unsupported NIfTI datatype code {datatype}.")
    pixdim = struct.unpack(e + "8f", raw[76:108])
    vox_offset = int(struct.unpack(e + "f", raw[108:112])[0])
    slope, inter = struct.unpack(e + "2f", raw[112:120])
    qform_code, sform_code = struct.unpack(e + "2h", raw[252:256])
    q = struct.unpack(e + "6f", raw[256:280])
    dt = np.dtype(_NIFTI_DTYPES[datatype]).newbyteorder(e)
    n = int(np.prod(shape, dtype=np.int64))
    if vox_offset < 348 or len(raw) < vox_offset + n * dt.itemsize:
        raise ThermoSEEG3DError(f"{label}: file is truncated.")
    data = np.frombuffer(raw, dtype=dt, count=n, offset=vox_offset).reshape(shape, order="F")
    data = data.astype(np.float32)
    if slope not in (0.0, 1.0) or inter != 0.0:
        data = data * (slope if slope != 0.0 else 1.0) + inter
    notes: List[str] = []
    if sform_code > 0:
        rows = [struct.unpack(e + "4f", raw[o:o + 16]) for o in (280, 296, 312)]
        affine = np.vstack([np.array(rows, dtype=float), [0, 0, 0, 1.0]])
    elif qform_code > 0:
        affine = _quaternion_affine(*q, pixdim)
    else:
        affine = np.diag([pixdim[1], pixdim[2], pixdim[3], 1.0])
        notes.append(f"{label}: no sform/qform in the header; spacing taken from pixdim and the origin set to 0 "
                     "(world coordinates are NOT scanner coordinates).")
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    return data, affine, notes


def _read_nifti(raw: bytes, label: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    raw = _decompress(raw, label)
    parsed = _parse_nifti1(raw, label)
    if parsed is not None:
        data, affine, notes = parsed
    else:  # NIfTI-2 and other exotic headers: delegate to nibabel when available
        try:
            import nibabel as nib
            img = nib.Nifti2Image.from_bytes(raw)
            data = np.nan_to_num(np.asarray(img.dataobj, dtype=np.float32))
            affine = np.asarray(img.affine, dtype=float)
            notes = []
        except Exception as exc:
            raise ThermoSEEG3DError(f"{label}: not a readable NIfTI file ({type(exc).__name__}).") from exc
        if data.ndim == 4 and data.shape[3] == 1:
            data = data[..., 0]
        if data.ndim != 3:
            raise ThermoSEEG3DError(f"{label}: expected a 3D volume, got shape {data.shape}.")
    lin = affine[:3, :3]
    sp = np.sqrt((lin ** 2).sum(axis=0))
    if np.any(sp <= 0) or not np.all(np.isfinite(affine)):
        raise ThermoSEEG3DError(f"{label}: invalid affine/voxel spacing in header.")
    if np.abs(np.abs(lin / sp[None, :]) - np.round(np.abs(lin / sp[None, :]))).max() > 0.03:
        notes.append(f"{label}: oblique acquisition; resampled onto an axis-aligned RAS+ grid using the full affine "
                     "(world mm coordinates preserved).")
    return data, affine, notes


# ---------------------------------------------------------------------------
# Scan upload normalisation: NIfTI / DICOM series / ZIP / JPG-PNG stacks  ->  one NIfTI-1 blob
# ---------------------------------------------------------------------------
IMAGE_EXTS = (".jpg", ".jpeg", ".png")
MIN_STACK_SLICES = 8          # a single 2D image cannot define a 3D volume
MAX_ZIP_MEMBERS = 5000


def nifti1_bytes(data: np.ndarray, affine: np.ndarray, gz: bool = True) -> bytes:
    """Minimal NIfTI-1 writer (float32, sform = affine). Used to feed converted scans into the normal loader."""
    data = np.asarray(data, dtype="<f4")
    affine = np.asarray(affine, dtype=float)
    hdr = bytearray(352)
    struct.pack_into("<i", hdr, 0, 348)
    struct.pack_into("<8h", hdr, 40, 3, *data.shape, 1, 1, 1, 1)
    struct.pack_into("<h", hdr, 70, 16)
    struct.pack_into("<h", hdr, 72, 32)
    sp = np.sqrt((affine[:3, :3] ** 2).sum(0))
    struct.pack_into("<8f", hdr, 76, 1.0, *sp, 0, 0, 0, 0)
    struct.pack_into("<f", hdr, 108, 352.0)
    struct.pack_into("<2f", hdr, 112, 1.0, 0.0)
    struct.pack_into("<2h", hdr, 252, 0, 1)
    for row, off in zip(range(3), (280, 296, 312)):
        struct.pack_into("<4f", hdr, off, *affine[row])
    hdr[344:348] = b"n+1\0"
    raw = bytes(hdr) + data.tobytes(order="F")
    return gzip.compress(raw, compresslevel=1) if gz else raw


def _natural_key(name: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def _is_dicom_blob(blob: bytes) -> bool:
    return len(blob) > 132 and blob[128:132] == b"DICM"


def _is_nifti_name(name: str) -> bool:
    return name.lower().endswith((".nii", ".nii.gz"))


def _zip_members(blob: bytes) -> List[Tuple[str, bytes]]:
    try:
        zf = zipfile.ZipFile(io.BytesIO(blob))
    except zipfile.BadZipFile as exc:
        raise ThermoSEEG3DError("The ZIP file is corrupt or not a ZIP archive.") from exc
    infos = [i for i in zf.infolist() if not i.is_dir()
             and not i.filename.startswith("__MACOSX") and not i.filename.split("/")[-1].startswith(".")]
    if not infos:
        raise ThermoSEEG3DError("The ZIP file contains no files.")
    if len(infos) > MAX_ZIP_MEMBERS:
        raise ThermoSEEG3DError(f"The ZIP file has more than {MAX_ZIP_MEMBERS} files.")
    if sum(i.file_size for i in infos) > MAX_DECOMPRESSED_BYTES:
        raise ThermoSEEG3DError("The ZIP file expands beyond the size limit.")
    return [(i.filename, zf.read(i)) for i in infos]


def _volume_from_dicom_slices(slices: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Pure geometry: slices = dicts with pixels (rows, cols), ipp (3,), iop (6,), ps (row_mm, col_mm), all LPS.
    Returns (data[col, row, slice], RAS affine, notes)."""
    if len(slices) < 4:
        raise ThermoSEEG3DError(f"Only {len(slices)} DICOM image(s) found; a 3D volume needs a series (>= 4 slices).")
    iop = np.asarray(slices[0]["iop"], float)
    r, c = iop[:3], iop[3:]
    n = np.cross(r, c)
    for s in slices:
        if not np.allclose(s["iop"], iop, atol=1e-3):
            raise ThermoSEEG3DError("DICOM slices have different orientations; upload one series only.")
        if s["pixels"].shape != slices[0]["pixels"].shape:
            raise ThermoSEEG3DError("DICOM slices have different matrix sizes; upload one series only.")
    pos = np.array([float(np.dot(s["ipp"], n)) for s in slices])
    order = np.argsort(pos)
    pos, sl = pos[order], [slices[i] for i in order]
    d = np.diff(pos)
    if np.any(d <= 1e-3):
        raise ThermoSEEG3DError("Duplicate DICOM slice positions (several series or repeated images mixed together).")
    spacing = float((pos[-1] - pos[0]) / (len(pos) - 1))
    notes = [f"DICOM series: {len(sl)} slices, slice spacing {spacing:.2f} mm."]
    if float(d.max() - d.min()) > 0.05 * spacing:
        notes.append("Slice spacing is not uniform; the volume was built assuming uniform spacing (check for missing slices).")
    ipp0 = np.asarray(sl[0]["ipp"], float)
    step = (np.asarray(sl[-1]["ipp"], float) - ipp0) / (len(sl) - 1)
    if float(np.linalg.norm(step - n * spacing)) > 0.5:
        notes.append("Gantry tilt / shear detected and NOT corrected.")
    row_mm, col_mm = sl[0]["ps"]
    A = np.eye(4)
    A[:3, 0] = r * col_mm
    A[:3, 1] = c * row_mm
    A[:3, 2] = n * spacing
    A[:3, 3] = ipp0
    A[:2, :] *= -1.0                      # DICOM LPS -> RAS+
    data = np.stack([s["pixels"].T for s in sl], axis=2).astype(np.float32)
    return data, A, notes


def _read_dicom_series(members: List[Tuple[str, bytes]]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    try:
        import pydicom
    except ImportError as exc:
        raise ThermoSEEG3DError("DICOM input requires pydicom (listed in requirements.txt).") from exc
    by_uid: Dict[str, List[Any]] = {}
    for _, blob in members:
        try:
            ds = pydicom.dcmread(io.BytesIO(blob), force=True)
        except Exception:
            continue
        if not hasattr(ds, "PixelData"):
            continue
        by_uid.setdefault(str(getattr(ds, "SeriesInstanceUID", "")), []).append(ds)
    if not by_uid:
        raise ThermoSEEG3DError("No readable DICOM images were found.")
    uid = max(by_uid, key=lambda k: len(by_uid[k]))
    chosen = by_uid[uid]
    notes = []
    if len(by_uid) > 1:
        notes.append(f"{len(by_uid)} DICOM series found; the largest ({len(chosen)} slices) was used.")
    modality = str(getattr(chosen[0], "Modality", "")).upper()
    if modality not in ("MR", "CT"):
        raise ThermoSEEG3DError(f"Unsupported DICOM modality '{modality or 'unspecified'}'; expected MR or CT.")
    if modality == "CT":
        notes.append("CT series: the heuristic brain/CSF masks assume MRI-like contrast and are NOT valid for CT; supply masks.")
    slices = []
    for ds in chosen:
        try:
            if int(getattr(ds, "SamplesPerPixel", 1)) != 1 or int(getattr(ds, "NumberOfFrames", 1)) != 1:
                raise ThermoSEEG3DError("Color or multi-frame DICOM is unsupported; upload a single-channel slice series.")
            iop = [float(v) for v in ds.ImageOrientationPatient]
            ipp = [float(v) for v in ds.ImagePositionPatient]
            ps = [float(v) for v in ds.PixelSpacing]
        except ThermoSEEG3DError:
            raise
        except Exception as exc:
            raise ThermoSEEG3DError("DICOM slices lack orientation, position or pixel-spacing tags.") from exc
        try:
            px = np.asarray(ds.pixel_array, dtype=np.float32)
        except Exception as exc:
            raise ThermoSEEG3DError("DICOM pixel data could not be decoded (unsupported compression or missing codec).") from exc
        if px.ndim != 2:
            raise ThermoSEEG3DError("DICOM slices must be 2D single-frame images.")
        slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
        inter = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
        slices.append({"pixels": px * slope + inter, "ipp": ipp, "iop": iop, "ps": ps})
    data, aff, n2 = _volume_from_dicom_slices(slices)
    notes += n2 + [f"Modality {modality}. Patient identifiers in the DICOM headers are not read, stored or exported."]
    return data, aff, notes


def _volume_from_image_stack(items: List[Tuple[str, bytes]], pixel_mm: float,
                             slice_mm: float) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    if len(items) < MIN_STACK_SLICES:
        raise ThermoSEEG3DError(
            f"{len(items)} JPG/PNG image(s) cannot form a 3D volume: a stack of at least {MIN_STACK_SLICES} slices "
            "(one image per slice, in a ZIP or selected together) is needed. For a single slice use the 2D Slice Planner page.")
    try:
        from PIL import Image
    except ImportError as exc:
        raise ThermoSEEG3DError("JPG/PNG input requires Pillow (listed in requirements.txt).") from exc
    if not (0.05 <= float(pixel_mm) <= 10.0 and 0.05 <= float(slice_mm) <= 20.0):
        raise ThermoSEEG3DError("Pixel spacing must be 0.05-10 mm and slice spacing 0.05-20 mm.")
    arrays = []
    for name, blob in sorted(items, key=lambda t: _natural_key(t[0])):
        try:
            im = Image.open(io.BytesIO(blob))
            im.load()
            arrays.append(np.asarray(im.convert("L"), dtype=np.float32))
        except Exception as exc:
            raise ThermoSEEG3DError(f"Could not read image '{name}'.") from exc
    h, w = arrays[0].shape
    if any(a.shape != (h, w) for a in arrays):
        raise ThermoSEEG3DError("All slice images must have the same width and height.")
    if h * w * len(arrays) > MAX_INPUT_VOXELS // 2:
        raise ThermoSEEG3DError("The image stack is too large; downscale the images or use fewer slices.")
    data = np.stack([a.T for a in arrays], axis=2)                    # (cols, rows, slices)
    px, sz = float(pixel_mm), float(slice_mm)
    A = np.eye(4)
    A[0, 0], A[1, 1], A[2, 2] = -px, -px, sz                          # radiological display: image left = patient right
    A[:3, 3] = [(w - 1) * px / 2, (h - 1) * px / 2, -(len(arrays) - 1) * sz / 2]
    notes = [f"JPG/PNG stack: {len(arrays)} slices of {w}x{h} px, pixel {px:g} mm, slice spacing {sz:g} mm (as entered). "
             "ASSUMED: axial slices in file-name order from inferior to superior, radiological left-right display, "
             "anterior at the top. World coordinates are NOT scanner coordinates; intensities are 8-bit display values."]
    return data, A, notes


def normalise_scan_upload(files: List[Tuple[str, bytes]], pixel_mm: float = 1.0,
                          slice_mm: float = 1.0) -> Tuple[bytes, List[str], str]:
    """files = [(filename, bytes)]. Accepts a NIfTI, a ZIP (NIfTI / DICOM series / JPG-PNG slices), several DICOM
    or JPG/PNG files. Returns (NIfTI-1 bytes, notes, kind)."""
    if not files:
        raise ThermoSEEG3DError("No scan file was provided.")
    members: List[Tuple[str, bytes]] = []
    for name, blob in files:
        if name.lower().endswith(".zip") or blob[:4] == b"PK\x03\x04":
            members += _zip_members(blob)
        else:
            members.append((name, blob))
    niis = [m for m in members if _is_nifti_name(m[0])]
    if niis:
        name, blob = max(niis, key=lambda m: len(m[1]))
        note = [f"NIfTI input: {name.split('/')[-1]}."]
        if len(niis) > 1:
            note.append(f"{len(niis)} NIfTI files found; the largest was used as the anatomy.")
        return blob, note, "nifti"
    dicoms = [m for m in members if m[0].lower().endswith(".dcm") or _is_dicom_blob(m[1])]
    if dicoms:
        data, aff, notes = _read_dicom_series(dicoms)
        return nifti1_bytes(data, aff), notes, "dicom"
    imgs = [m for m in members if m[0].lower().endswith(IMAGE_EXTS)]
    if imgs:
        data, aff, notes = _volume_from_image_stack(imgs, pixel_mm, slice_mm)
        return nifti1_bytes(data, aff), notes, "images"
    raise ThermoSEEG3DError("Unsupported file type. Upload NIfTI (.nii/.nii.gz), a DICOM series, a ZIP of either, "
                            "or a stack of JPG/PNG slices.")


def grid_for(shape, affine, target_mm: float) -> Tuple[Tuple[int, int, int], np.ndarray, np.ndarray]:
    """Axis-aligned RAS+ grid (shape, spacing, origin) covering the volume at ~target_mm isotropic voxels."""
    n = np.array(shape, float)
    corners = np.array([[i, j, k, 1.0] for i in (-0.5, n[0] - 0.5) for j in (-0.5, n[1] - 0.5) for k in (-0.5, n[2] - 0.5)])
    w = (np.asarray(affine) @ corners.T).T[:, :3]
    lo, hi = w.min(axis=0), w.max(axis=0)
    shape_out = np.maximum(4, np.ceil((hi - lo) / target_mm)).astype(int)
    if int(np.prod(shape_out)) > MAX_WORKING_VOXELS:
        raise ThermoSEEG3DError("Working grid would exceed 4 million voxels; choose a coarser working resolution.")
    spacing = np.full(3, float(target_mm))
    return tuple(int(v) for v in shape_out), spacing, lo + spacing / 2


def resample_to_ras_grid(arr: np.ndarray, affine, shape_out, spacing_out, origin_out, order: int = 1) -> np.ndarray:
    """Resample onto the RAS+ grid, applying the full voxel->world affine (handles flips, permutations, obliquity)."""
    affine = np.asarray(affine, float)
    lin, t = affine[:3, :3], affine[:3, 3]
    in_sp = np.sqrt((lin ** 2).sum(axis=0))
    sigma = np.maximum(0.0, 0.5 * (float(np.mean(spacing_out)) / in_sp - 1.0))   # anti-alias before decimation
    a = arr.astype(np.float32, copy=False)
    if np.any(sigma > 0.05):
        a = ndi.gaussian_filter(a, sigma, mode="nearest")
    inv = np.linalg.inv(lin)
    out = ndi.affine_transform(a, inv @ np.diag(spacing_out), offset=inv @ (np.asarray(origin_out) - t),
                               output_shape=tuple(shape_out), order=order, mode="constant", cval=0.0, prefilter=False)
    return out.astype(np.float32)


def _ball(radius_vox: float) -> np.ndarray:
    r = max(1, int(round(radius_vox)))
    g = np.arange(-r, r + 1)
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    return (X ** 2 + Y ** 2 + Z ** 2) <= r * r


def _otsu(values: np.ndarray, bins: int = 256) -> float:
    hist, edges = np.histogram(values, bins=bins)
    hist = hist.astype(float)
    centers = 0.5 * (edges[:-1] + edges[1:])
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * centers) / np.maximum(w0, 1e-12)
    m1 = ((hist * centers).sum() - np.cumsum(hist * centers)) / np.maximum(w1, 1e-12)
    between = w0 * w1 * (m0 - m1) ** 2
    return float(centers[int(np.argmax(between))])


def _largest_component(m: np.ndarray) -> np.ndarray:
    lab, n = ndi.label(m)
    if n == 0:
        raise ThermoSEEG3DError("Mask is empty after processing; please supply a brain mask.")
    sizes = ndi.sum(m, lab, index=np.arange(1, n + 1))
    return lab == (1 + int(np.argmax(sizes)))


def heuristic_brain_mask(anat: np.ndarray, spacing, strip_mm: float = 6.0) -> np.ndarray:
    """Otsu -> opening (strips thin scalp bridges) -> largest component -> hole fill. NOT skull stripping."""
    sm = ndi.gaussian_filter(anat, 1.0)
    m = sm > _otsu(sm[sm > 0] if np.any(sm > 0) else sm.ravel())
    if strip_mm > 0:
        m = ndi.binary_opening(m, structure=_ball(strip_mm / float(np.mean(spacing))))
    return ndi.binary_fill_holes(_largest_component(m))


def brain_mask_from_stripped(anat: np.ndarray) -> np.ndarray:
    """For anatomy that is already skull-stripped (background exactly ~0): non-zero voxels, largest component, filled."""
    m = anat > 0.02 * float(anat.max() or 1.0)
    return ndi.binary_fill_holes(_largest_component(m))


def heuristic_csf_mask(anat: np.ndarray, brain: np.ndarray, csf_dark: bool = True, pct: float = 8.0) -> np.ndarray:
    """Darkest (T1-like) or brightest (T2-like) intensity class inside the brain via 3-class multi-Otsu."""
    vals = anat[brain]
    try:
        from skimage.filters import threshold_multiotsu
        t1, t2 = threshold_multiotsu(vals, classes=3)
        m = brain & ((anat <= t1) if csf_dark else (anat >= t2))
    except Exception:
        thr = np.percentile(vals, pct if csf_dark else 100 - pct)
        m = brain & ((anat <= thr) if csf_dark else (anat >= thr))
    m = ndi.binary_opening(m, structure=np.ones((2, 2, 2), bool))
    return m


def vesselness(anat: np.ndarray, spacing, sigmas_mm=(1.0, 1.5, 2.0)) -> np.ndarray:
    """Simple multi-scale Hessian 'tube' filter for BRIGHT vessels (Sato-like). Heuristic."""
    mean_sp = float(np.mean(spacing))
    best = np.zeros(anat.shape, dtype=np.float32)
    for s_mm in sigmas_mm:
        s = s_mm / mean_sp
        H = np.empty(anat.shape + (3, 3), dtype=np.float32)
        for i in range(3):
            for j in range(i, 3):
                order = [0, 0, 0]
                order[i] += 1
                order[j] += 1
                h = ndi.gaussian_filter(anat, s, order=tuple(order)) * (s ** 2)
                H[..., i, j] = h
                H[..., j, i] = h
        ev = np.linalg.eigvalsh(H)  # ascending
        v = np.where((ev[..., 0] < 0) & (ev[..., 1] < 0), -ev[..., 1], 0.0).astype(np.float32)
        best = np.maximum(best, v)
    return best


def heuristic_risk_mask(anat: np.ndarray, brain: np.ndarray, spacing, use_percentile: bool = True,
                        percentile: float = 97.0, manual_threshold: int = 140,
                        min_volume_mm3: float = 30.0) -> Dict[str, Any]:
    """3D port of the 2D 'risk regions' heuristic: bright (high-signal) voxels inside the brain.

    * use_percentile=True: threshold = `percentile` of the anatomy intensity inside the brain mask.
    * use_percentile=False: threshold = `manual_threshold` on the 0-255 display scale (anat * 255).
    * Connected components smaller than `min_volume_mm3` are removed.
    HEURISTIC intensity mask, NOT a vessel segmentation."""
    vals = anat[brain]
    if vals.size == 0:
        raise ThermoSEEG3DError("Brain mask is empty; cannot compute risk regions.")
    thr = float(np.percentile(vals, float(percentile))) if use_percentile else float(manual_threshold) / 255.0
    m = brain & (anat >= thr)
    vv = float(np.prod(spacing))
    min_vox = int(math.ceil(float(min_volume_mm3) / vv)) if min_volume_mm3 > 0 else 0
    if min_vox > 1 and m.any():
        lab, n = ndi.label(m)
        sizes = ndi.sum(m, lab, index=np.arange(1, n + 1))
        keep = np.concatenate([[False], np.asarray(sizes) >= min_vox])
        m = keep[lab]
    frac = float(m.sum()) / max(1, int(brain.sum()))
    return {"mask": m, "threshold_display_0_255": round(thr * 255.0, 1), "fraction_of_brain": frac,
            "volume_ml": float(m.sum()) * vv / 1000.0,
            "plateau_warning": frac > RISK_MASK_MAX_FRACTION}


def obstacle_distance_map_mm(vol: "Volume3D", risk: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """Distance (mm) to the nearest obstacle = vessel label OR heuristic risk voxel."""
    obst = vol.labels == LABEL_VESSEL
    if risk is not None:
        obst = obst | risk
    if not obst.any():
        return None
    return ndi.distance_transform_edt(~obst, sampling=vol.spacing).astype(np.float32)


def load_anatomy_only(raw: bytes, working_mm: float = 1.5) -> Dict[str, Any]:
    """Read one NIfTI anatomy and resample it onto an axis-aligned RAS+ grid (exact world mm).
    Used by the 2D slice planner to browse a 3D volume slice by slice. Returns normalised anatomy in [0, 1]."""
    data, aff, notes = _read_nifti(raw, "anatomy")
    shape_out, spacing, origin = grid_for(data.shape, aff, working_mm)
    a = resample_to_ras_grid(data, aff, shape_out, spacing, origin, order=1)
    pos = a[a > 0]
    lo = float(np.percentile(pos, 1)) if pos.size else 0.0
    hi = float(np.percentile(a, 99.5))
    anat = np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)
    idx = np.argwhere(anat > 0.2)
    centre = origin + (idx.mean(axis=0) if idx.size else np.array(anat.shape) / 2.0) * spacing
    return {"anatomy": anat, "spacing": spacing, "origin": origin, "notes": notes, "centre_mm": centre,
            "sha256": hashlib.sha256(raw).hexdigest()}


PLANES = ("Axial", "Coronal", "Sagittal")


@dataclass
class SliceMap:
    """Maps pixels of a displayed 2D slice to 3D world mm (RAS+).
    Axial: columns = x (Right+), rows top->bottom = y descending (Anterior up).
    Coronal: columns = x (Right+), rows = z descending (Superior up).
    Sagittal: columns = y (Anterior+ to the right), rows = z descending (Superior up)."""
    plane: str
    index: int
    shape_hw: Tuple[int, int]
    spacing: np.ndarray
    origin: np.ndarray

    def pixel_to_world(self, col: float, row: float) -> np.ndarray:
        h = self.shape_hw[0]
        o, s, i = self.origin, self.spacing, self.index
        if self.plane == "Axial":
            return np.array([o[0] + col * s[0], o[1] + (h - 1 - row) * s[1], o[2] + i * s[2]])
        if self.plane == "Coronal":
            return np.array([o[0] + col * s[0], o[1] + i * s[1], o[2] + (h - 1 - row) * s[2]])
        return np.array([o[0] + i * s[0], o[1] + col * s[1], o[2] + (h - 1 - row) * s[2]])

    def world_to_pixel(self, p) -> Tuple[float, float]:
        h = self.shape_hw[0]
        p = (np.asarray(p, float) - self.origin) / self.spacing
        if self.plane == "Axial":
            return float(p[0]), float(h - 1 - p[1])
        if self.plane == "Coronal":
            return float(p[0]), float(h - 1 - p[2])
        return float(p[1]), float(h - 1 - p[2])

    @property
    def in_plane_spacing_mm(self) -> Tuple[float, float]:
        s = self.spacing
        return {"Axial": (s[0], s[1]), "Coronal": (s[0], s[2]), "Sagittal": (s[1], s[2])}[self.plane]


def slice_index_range(anat: np.ndarray, plane: str) -> int:
    return anat.shape[{"Axial": 2, "Coronal": 1, "Sagittal": 0}[plane]]


def slice_to_image8(anat: np.ndarray, spacing, origin, plane: str, index: int) -> Tuple[np.ndarray, SliceMap]:
    """8-bit grayscale slice (rows top->bottom) and its pixel->world map."""
    if plane not in PLANES:
        raise ThermoSEEG3DError(f"Unknown plane '{plane}'.")
    index = int(np.clip(index, 0, slice_index_range(anat, plane) - 1))
    if plane == "Axial":
        sl = anat[:, :, index].T[::-1, :]
    elif plane == "Coronal":
        sl = anat[:, index, :].T[::-1, :]
    else:
        sl = anat[index, :, :].T[::-1, :]
    img = np.clip(np.rint(sl * 255.0), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(img), SliceMap(plane, index, img.shape, np.asarray(spacing, float), np.asarray(origin, float))


def world_to_slice_index(plane: str, p_mm, spacing, origin) -> int:
    ax = {"Axial": 2, "Coronal": 1, "Sagittal": 0}[plane]
    return int(round((float(p_mm[ax]) - float(origin[ax])) / float(spacing[ax])))


def build_labels(brain: np.ndarray, csf: Optional[np.ndarray], vessel: Optional[np.ndarray]) -> np.ndarray:
    labels = np.zeros(brain.shape, dtype=np.uint8)
    labels[brain] = LABEL_BRAIN
    if csf is not None:
        labels[csf & brain] = LABEL_CSF
    if vessel is not None:
        labels[vessel & brain] = LABEL_VESSEL
    return labels


def load_nifti_volume(files: Dict[str, bytes], working_spacing_mm: float = 2.5,
                      csf_dark: bool = True, vessel_threshold_frac: float = 0.3,
                      strip_mm: float = 6.0, anatomy_is_brain_only: bool = False) -> Volume3D:
    """files: 'anatomy' (required) and optional 'brain', 'csf', 'vessel' raw NIfTI bytes.
    All volumes must share one voxel grid (identical shape and affine); no registration is attempted.
    Everything is resampled onto an axis-aligned RAS+ grid, so world mm coordinates stay exact."""
    if "anatomy" not in files:
        raise ThermoSEEG3DError("An anatomy volume is required.")
    notes: List[str] = []
    raw: Dict[str, np.ndarray] = {}
    shape0 = aff0 = None
    for key, blob in files.items():
        data, aff, n = _read_nifti(blob, key)
        notes += n
        if shape0 is None:
            shape0, aff0 = data.shape, aff
        elif data.shape != shape0 or not np.allclose(aff, aff0, rtol=1e-3, atol=1e-2):
            raise ThermoSEEG3DError(f"'{key}' does not share the anatomy voxel grid (shape/affine differ). "
                                    "Register/resample it to the anatomy first; no registration is attempted.")
        raw[key] = data
    shape_out, spacing, origin = grid_for(shape0, aff0, working_spacing_mm)
    anat_r = resample_to_ras_grid(raw["anatomy"], aff0, shape_out, spacing, origin, order=1)
    lo, hi = np.percentile(anat_r[anat_r > 0], 1) if np.any(anat_r > 0) else 0.0, np.percentile(anat_r, 99.5)
    anat = np.clip((anat_r - lo) / max(float(hi - lo), 1e-6), 0, 1).astype(np.float32)

    def mask_of(key):
        return resample_to_ras_grid((raw[key] != 0).astype(np.float32), aff0, shape_out, spacing, origin, 1) > 0.5

    if "brain" in raw:
        brain = mask_of("brain")
        notes.append("Brain mask: user-supplied.")
    elif anatomy_is_brain_only:
        brain = brain_mask_from_stripped(anat_r)
        notes.append("Brain mask: non-zero voxels of the skull-stripped anatomy (as declared by the user).")
    else:
        brain = heuristic_brain_mask(anat, spacing, strip_mm)
        notes.append("Brain mask: HEURISTIC (Otsu/opening/largest component). It only works for images that are "
                     "already skull-stripped; on a T1 that still contains skull, scalp and neck it will include "
                     "non-brain tissue. Supply a brain mask (e.g. from SynthStrip, HD-BET or FSL BET).")
    if not brain.any():
        raise ThermoSEEG3DError("The brain mask is empty after resampling.")
    if "csf" in raw:
        csf = mask_of("csf")
        notes.append("CSF mask: user-supplied.")
        if not (csf & brain).any():
            notes.append("Warning: the CSF mask does not overlap the brain mask.")
    else:
        csf = heuristic_csf_mask(anat, brain, csf_dark)
        notes.append("CSF mask: HEURISTIC intensity class (3-class multi-Otsu) inside the brain; verify before use.")
    vessel = None
    if "vessel" in raw:
        vimg = resample_to_ras_grid(raw["vessel"], aff0, shape_out, spacing, origin, 1)
        if np.unique(np.round(vimg[::3, ::3, ::3], 3)).size <= 3:
            vessel = vimg > 0.5 * float(vimg.max() or 1.0)
            notes.append("Vessel map: user-supplied binary mask.")
        else:
            v = vesselness(vimg / max(float(vimg.max()), 1e-6), spacing)
            v = np.where(brain, v, 0)
            vessel = v > vessel_threshold_frac * float(v.max() or 1.0)
            notes.append(f"Vessel map: HEURISTIC Hessian vesselness (threshold {vessel_threshold_frac:.0%} of max).")
    if vessel is None:
        notes.append("Vessel map: NOT SUPPLIED. A vessel volume of 0 mL means no vessel mask/angiography was provided, "
                     "not that the anatomy has no vessels; vessel clearance and the vessel heat-sink are inactive.")
    labels = build_labels(brain, csf, vessel)
    vol = Volume3D(anat, labels, spacing, origin, source="NIfTI upload", notes=notes)
    par = np.argwhere(labels == LABEL_BRAIN)
    if par.size:
        c = par.mean(axis=0)
        vol.default_target = vol.index_to_world(c)
        if vol.label_at(vol.default_target) == LABEL_BG:
            near = par[np.argmin(((par - c) ** 2).sum(axis=1))]
            vol.default_target = vol.index_to_world(near)
    vol.sha256 = _hash_volume(anat, labels, spacing, origin, "nifti")
    return vol


# ---------------------------------------------------------------------------
# Thermal model (transient Pennes bioheat, explicit finite differences)
# ---------------------------------------------------------------------------
def cem43_rate_per_min(T_c: np.ndarray) -> np.ndarray:
    """Sapareto-Dewey equivalent minutes per minute at temperature T (clipped to the valid range)."""
    T = np.minimum(np.asarray(T_c, dtype=np.float64), MAX_VALID_TEMP_C)
    R = np.where(T >= CEM43_REF_C, 0.5, 0.25)
    return np.where(T >= BODY_TEMP_C, R ** (CEM43_REF_C - T), 0.0)


def crop_box(vol: Volume3D, tip, direction_out, p: ThermalParams) -> Tuple[Tuple[int, int], ...]:
    tip = np.asarray(tip, float)
    far = tip + unit(direction_out) * p.active_length_mm
    lo_mm = np.minimum(tip, far) - p.margin_mm - 2 * p.source_radius_mm
    hi_mm = np.maximum(tip, far) + p.margin_mm + 2 * p.source_radius_mm
    lo = np.floor(vol.world_to_index(lo_mm)).astype(int)
    hi = np.ceil(vol.world_to_index(hi_mm)).astype(int) + 1
    lo = np.maximum(lo, 0)
    hi = np.minimum(hi, np.array(vol.shape))
    if np.any(hi - lo < 4):
        raise ThermoSEEG3DError("Target is too close to the volume boundary for a thermal run.")
    return tuple((int(lo[i]), int(hi[i])) for i in range(3))


def _box_slices(box) -> Tuple[slice, slice, slice]:
    return tuple(slice(a, b) for a, b in box)


def _property_fields(labels_c: np.ndarray, p: ThermalParams, csf_model: bool):
    P = PRESETS.get(p.preset, PRESETS[DEFAULT_PRESET])
    lab = labels_c.copy()
    lab[lab == LABEL_BG] = LABEL_BRAIN          # skull/scalp/air treated as brain-like (crop is intracranial)
    if not csf_model:
        lab[lab == LABEL_CSF] = LABEL_BRAIN     # conventional model: CSF is ordinary tissue (no sink, no extra conduction)
    k = np.empty(lab.shape, dtype=np.float64)
    C = np.empty(lab.shape, dtype=np.float64)
    h = np.zeros(lab.shape, dtype=np.float64)
    h_brain = P["brain"]["perfusion_1_per_s"] * P["blood_rho"] * P["blood_c"]
    for l, key in ((LABEL_BRAIN, "brain"), (LABEL_CSF, "csf"), (LABEL_VESSEL, "vessel")):
        m = lab == l
        k[m] = P[key]["k"]
        C[m] = P[key]["rho"] * P[key]["c"]
    k[lab == LABEL_CSF] *= p.csf_k_multiplier
    k[lab == LABEL_BRAIN] *= p.conductivity_scale
    h[lab == LABEL_BRAIN] = p.perfusion_scale * h_brain
    h[lab == LABEL_VESSEL] = p.vessel_sink_scale * h_brain
    h[lab == LABEL_CSF] = p.csf_sink_w_m3k      # empirical CSF cooling: sink * (T - Tbaseline), as in the desktop model
    return k, C, h


def source_field(vol: Volume3D, box, tip, direction_out, p: ThermalParams) -> np.ndarray:
    """Heat source Q (W/m^3): Gaussian radial profile around the active line segment, absorbed
    in non-CSF tissue only, normalised so that sum(Q dV) = absorbed power."""
    sl = _box_slices(box)
    ax = [vol.origin[i] + np.arange(box[i][0], box[i][1]) * vol.spacing[i] for i in range(3)]
    X, Y, Z = np.meshgrid(*ax, indexing="ij")
    tip = np.asarray(tip, float)
    far = tip + unit(direction_out) * p.active_length_mm
    d = _dist_to_segment(X, Y, Z, tip, far)
    sigma = max(0.5, p.source_radius_mm / 2.0)
    g = np.exp(-(d ** 2) / (2 * sigma ** 2))
    g[vol.labels[sl] == LABEL_CSF] = 0.0
    dV = float(np.prod(vol.spacing)) * 1e-9
    total = float(g.sum()) * dV
    if total <= 0:
        raise ThermoSEEG3DError("The heat source does not overlap non-CSF tissue; move the target.")
    return p.power_w * p.absorbed_fraction * g / total


def run_thermal(vol: Volume3D, box, Q: np.ndarray, p: ThermalParams, csf_model: bool,
                tip_mm=None) -> Dict[str, Any]:
    t0 = time.perf_counter()
    sl = _box_slices(box)
    k, C, h = _property_fields(vol.labels[sl], p, csf_model)
    dx = vol.spacing * 1e-3
    kf = []
    for ax in range(3):
        k1 = np.take(k, np.arange(k.shape[ax] - 1), axis=ax)
        k2 = np.take(k, np.arange(1, k.shape[ax]), axis=ax)
        kf.append(2 * k1 * k2 / (k1 + k2) / dx[ax])
    total = float(p.exposure_s + p.cooldown_s)
    if total <= 0:
        raise ThermoSEEG3DError("Exposure + cooldown must be positive.")
    dt_stab = 0.9 * float(C.min()) / (2 * float(k.max()) * float(np.sum(1.0 / dx ** 2)) + float(h.max()))
    n_steps = max(1, int(math.ceil(total / dt_stab)))
    dt = total / n_steps
    T = np.full(k.shape, BODY_TEMP_C, dtype=np.float64)
    peak = T.copy()
    dose = np.zeros_like(T)
    n_snap = max(2, int(p.n_snapshots))
    snap_steps = sorted(set(int(round(v)) for v in np.linspace(0, n_steps, n_snap)))
    snap_times, snaps = [], []
    tip_idx = tuple(int(np.clip(round(float(i)), 0, n - 1)) for i, n in zip(
        vol.world_to_index(tip_mm) - np.array([b[0] for b in box]), k.shape)) if tip_mm is not None else None
    hist = {"t_s": [], "max_temp_c": [], "max_cem43_min": [], "tip_temp_c": [], "tip_cem43_min": []}
    for step in range(n_steps + 1):
        if step in snap_steps:
            snap_times.append(step * dt)
            snaps.append(T.astype(np.float32))
        if step == n_steps:
            break
        div = 0.0
        for ax in range(3):
            div = div + np.diff(kf[ax] * np.diff(T, axis=ax), axis=ax, prepend=0.0, append=0.0) / dx[ax]
        on = (step + 0.5) * dt < p.exposure_s
        rhs = div - h * (T - ARTERIAL_TEMP_C)
        if on:
            rhs = rhs + Q
        T = T + dt * rhs / C
        np.maximum(peak, T, out=peak)
        dose += dt / 60.0 * cem43_rate_per_min(T)
        hist["t_s"].append((step + 1) * dt)
        hist["max_temp_c"].append(float(T.max()))
        hist["max_cem43_min"].append(float(dose.max()))
        if tip_idx is not None:
            hist["tip_temp_c"].append(float(T[tip_idx]))
            hist["tip_cem43_min"].append(float(dose[tip_idx]))
    max_t = float(peak.max())
    valid = bool(np.isfinite(max_t) and max_t <= MAX_VALID_TEMP_C)
    return {
        "times": snap_times, "snapshots": np.stack(snaps), "peak": peak.astype(np.float32),
        "dose": dose.astype(np.float32) if valid else None,
        "max_temp_c": max_t, "valid": valid, "dt_s": dt, "steps": n_steps,
        "box": box, "csf_model": csf_model, "runtime_s": time.perf_counter() - t0,
        "history": {kk: np.array(vv) for kk, vv in hist.items()} if valid else
                   {"t_s": np.array(hist["t_s"]), "max_temp_c": np.array(hist["max_temp_c"]),
                    "max_cem43_min": None, "tip_temp_c": np.array(hist["tip_temp_c"]), "tip_cem43_min": None},
    }


def run_litt_comparison(vol: Volume3D, target, direction_out, p: ThermalParams) -> Dict[str, Any]:
    """Run the thermal model with and without the CSF model using an identical heat source."""
    if vol.label_at(target) == LABEL_BG:
        raise ThermoSEEG3DError("Target is outside the intracranial mask.")
    box = crop_box(vol, target, direction_out, p)
    Q = source_field(vol, box, target, direction_out, p)
    with_csf = run_thermal(vol, box, Q, p, csf_model=True, tip_mm=target)
    no_csf = run_thermal(vol, box, Q, p, csf_model=False, tip_mm=target)
    vv = vol.voxel_volume_mm3()
    dT = no_csf["peak"] - with_csf["peak"]
    heated = (no_csf["peak"] - BODY_TEMP_C) > 1.0
    ratio = None
    if heated.any():
        ratio = float(np.mean((with_csf["peak"][heated] - BODY_TEMP_C) / (no_csf["peak"][heated] - BODY_TEMP_C)))
    both_valid = with_csf["valid"] and no_csf["valid"]

    def dose_stats(res):
        if res["dose"] is None:
            return {"max_cem43_min": None, "ablation_volume_mm3": None}
        d = res["dose"]
        return {"max_cem43_min": float(d.max()),
                "ablation_volume_mm3": float(np.count_nonzero(d >= CEM43_ABLATION_MIN)) * vv}

    ds_c, ds_n = dose_stats(with_csf), dose_stats(no_csf)
    rel = None
    if (both_valid and ds_n["max_cem43_min"] and 1e-9 < ds_n["max_cem43_min"] <= CEM43_SATURATION_MIN
            and ds_c["max_cem43_min"] <= CEM43_SATURATION_MIN):
        rel = 100.0 * (ds_n["max_cem43_min"] - ds_c["max_cem43_min"]) / ds_n["max_cem43_min"]
    abl_red = None
    if both_valid and ds_n["ablation_volume_mm3"]:
        abl_red = 100.0 * (ds_n["ablation_volume_mm3"] - ds_c["ablation_volume_mm3"]) / ds_n["ablation_volume_mm3"]
    warnings = []
    if not with_csf["valid"] or not no_csf["valid"]:
        warnings.append(f"Peak temperature exceeds {MAX_VALID_TEMP_C:.0f} C: the model is not physically meaningful; "
                        "CEM43 and ablation volume are suppressed. Reduce power/exposure.")
    return {
        "with_csf": with_csf, "no_csf": no_csf, "dT_reduction_peak": dT.astype(np.float32),
        "box": box, "source_peak_w_m3": float(Q.max()),
        "metrics": {
            "max_temp_with_csf_c": round(with_csf["max_temp_c"], 2),
            "max_temp_without_csf_c": round(no_csf["max_temp_c"], 2),
            "max_csf_dT_reduction_c": round(float(dT.max()), 2),
            "mean_csf_heating_ratio": None if ratio is None else round(ratio, 3),
            "with_csf": ds_c, "without_csf": ds_n,
            "relative_dose_reduction_pct": None if rel is None else round(rel, 2),
            "ablation_volume_reduction_pct": None if abl_red is None else round(abl_red, 2),
            "model_valid": both_valid,
            "dt_s": round(with_csf["dt_s"], 3), "steps": with_csf["steps"],
        },
        "warnings": warnings,
    }


def embed(crop_arr: np.ndarray, box, shape, fill=np.nan) -> np.ndarray:
    out = np.full(shape, fill, dtype=np.float32)
    out[_box_slices(box)] = crop_arr
    return out


def fmt_dose(v: Optional[float]) -> str:
    if v is None:
        return "n/a (model invalid)"
    return f">{CEM43_SATURATION_MIN:.0e} (saturated)" if v > CEM43_SATURATION_MIN else f"{v:.3g}"