"""Plotly figure builders for the ThermoSEEG 3D Planning Station (presentation only; dark theme)."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
from plotly.subplots import make_subplots

import thermoseeg_3d as t3

BG = "#0B0F14"
PANEL = "#121820"
LINE = "#243040"
INK = "#E6EDF3"
MUTED = "#8B98A9"
CYAN = "#43C6D5"
MAGENTA = "#D85C9E"
AMBER = "#F0A34A"

RISK_COLOR = "#F59E0B"
BRAIN_COLOR = "#CBD5E1"
CSF_COLOR = CYAN
VESSEL_COLOR = "#EF4444"
LITT_COLOR = MAGENTA
SEEG_COLOR = AMBER
CONTACT_COLOR = "#F87171"
ABLATION_COLOR = "#FF4D6D"
DT_COLOR = "#34D399"
REGION_COLOR = AMBER
TEMP_COLORSCALE = "Inferno"

CAMERAS = {
    "3D": dict(eye=dict(x=-1.6, y=-1.3, z=0.7), up=dict(x=0, y=0, z=1)),
    "Axial": dict(eye=dict(x=0.0, y=0.001, z=2.6), up=dict(x=0, y=1, z=0)),
    "Coronal": dict(eye=dict(x=0.0, y=-2.6, z=0.001), up=dict(x=0, y=0, z=1)),
    "Sagittal": dict(eye=dict(x=-2.6, y=0.0, z=0.001), up=dict(x=0, y=0, z=1)),
}


def _mesh(mesh, name: str, color: str, opacity: float, group: Optional[str] = None,
          show_legend: bool = True) -> go.Mesh3d:
    verts, faces = mesh
    return go.Mesh3d(
        x=verts[:, 0], y=verts[:, 1], z=verts[:, 2], i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        name=name, color=color, opacity=opacity, showlegend=show_legend, legendgroup=group or name,
        flatshading=True, lighting=dict(ambient=0.55, diffuse=0.8, specular=0.15, roughness=0.6),
        hovertemplate="x %{x:.1f}, y %{y:.1f}, z %{z:.1f} mm<extra>" + name + "</extra>")


def temperature_colour(level_c: float, tmax_c: float) -> str:
    frac = float(np.clip((level_c - t3.BODY_TEMP_C) / max(tmax_c - t3.BODY_TEMP_C, 1e-6), 0, 1))
    return sample_colorscale(TEMP_COLORSCALE, [min(0.92, frac)])[0]


def _line(fig, p0, p1, color, width, name, show=True):
    fig.add_trace(go.Scatter3d(
        x=[p0[0], p1[0]], y=[p0[1], p1[1]], z=[p0[2], p1[2]], mode="lines", line=dict(color=color, width=width),
        name=name, showlegend=show,
        hovertemplate="x %{x:.1f}, y %{y:.1f}, z %{z:.1f} mm<extra>" + name + "</extra>"))


def _point(fig, p, color, name, size=6):
    fig.add_trace(go.Scatter3d(
        x=[p[0]], y=[p[1]], z=[p[2]], mode="markers", marker=dict(size=size, color=color), name=name,
        showlegend=False, hovertemplate=name + "<br>x %{x:.1f}, y %{y:.1f}, z %{z:.1f} mm<extra></extra>"))


def build_scene(
    vol: t3.Volume3D,
    static_meshes: Dict[str, Optional[tuple]],
    target: np.ndarray,
    entry: np.ndarray,
    procedure: str,
    layers: Sequence[str],
    active_far_point: Optional[np.ndarray] = None,
    capsule_mesh: Optional[tuple] = None,
    contacts: Optional[List[dict]] = None,
    seeg_line: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    seeg_contact_list: Optional[List[dict]] = None,
    region: Optional[dict] = None,
    temp_shells: Optional[List[Tuple[float, tuple]]] = None,
    temp_max_c: float = 80.0,
    ablation_mesh: Optional[tuple] = None,
    dT_mesh: Optional[tuple] = None,
    view: str = "3D",
    title: str = "",
) -> go.Figure:
    fig = go.Figure()
    lo, hi = vol.bounds_mm
    if "Patient brain" in layers and static_meshes.get("Patient brain"):
        fig.add_trace(_mesh(static_meshes["Patient brain"], "Patient brain", BRAIN_COLOR, 0.10))
    if "Patient CSF" in layers and static_meshes.get("Patient CSF"):
        fig.add_trace(_mesh(static_meshes["Patient CSF"], "Patient CSF", CSF_COLOR, 0.30))
    if "Vessels" in layers and static_meshes.get("Vessels"):
        fig.add_trace(_mesh(static_meshes["Vessels"], "Vessels", VESSEL_COLOR, 0.9))
    if "Risk regions" in layers and static_meshes.get("Risk regions"):
        fig.add_trace(_mesh(static_meshes["Risk regions"], "Risk regions (heuristic)", RISK_COLOR, 0.45))
    if "Temperature field" in layers and temp_shells:
        for n, (level, mesh) in enumerate(temp_shells):
            fig.add_trace(_mesh(mesh, f"Temperature field (≥{level:.0f} °C)" if n == 0 else f"≥{level:.0f} °C",
                                temperature_colour(level, temp_max_c), 0.28, group="Temperature field",
                                show_legend=(n == 0)))
    if "LITT ablation" in layers and ablation_mesh is not None:
        fig.add_trace(_mesh(ablation_mesh, f"LITT ablation (CEM43 ≥ {t3.CEM43_ABLATION_MIN:.0f})", ABLATION_COLOR, 0.55,
                            group="LITT ablation"))
    if "CSF ΔT reduction" in layers and dT_mesh is not None:
        fig.add_trace(_mesh(dT_mesh, "CSF ΔT reduction", DT_COLOR, 0.30))
    if "Applicator capsule" in layers and capsule_mesh is not None:
        fig.add_trace(_mesh(capsule_mesh, "Applicator capsule (geometric)", LITT_COLOR, 0.30))
    if "Region sphere" in layers and region is not None:
        rv, rf = t3.sphere_surface(region["center"], region["radius"])
        fig.add_trace(_mesh((rv, rf), region["label"], REGION_COLOR, 0.22))

    if "Trajectory" in layers:
        name = "SEEG trajectory" if procedure.startswith("SEEG") else "LITT trajectory"
        color = SEEG_COLOR if procedure.startswith("SEEG") else LITT_COLOR
        _line(fig, entry, target, color, 7, name)
        _point(fig, target, "#34D399", "Target")
        _point(fig, entry, "#E6EDF3", "Entry")
        if active_far_point is not None:
            _line(fig, target, active_far_point, "#F97316", 14, "LITT active segment")
        if seeg_line is not None:
            _line(fig, seeg_line[0], seeg_line[1], SEEG_COLOR, 6, "SEEG electrode")
    if "SEEG contacts" in layers:
        first = True
        for lst in (contacts, seeg_contact_list):
            if lst:
                pos = np.array([c["pos_mm"] for c in lst])
                fig.add_trace(go.Scatter3d(
                    x=pos[:, 0], y=pos[:, 1], z=pos[:, 2], mode="markers+text",
                    marker=dict(size=5, color=CONTACT_COLOR, line=dict(color="#7F1D1D", width=1)),
                    text=[str(c["id"]) for c in lst], textposition="top center", textfont=dict(color=INK),
                    name="SEEG contacts", showlegend=first,
                    hovertemplate="Contact %{text}<br>x %{x:.1f}, y %{y:.1f}, z %{z:.1f} mm<extra></extra>"))
                first = False

    if "Temperature field" in layers and temp_shells:
        fig.add_trace(go.Scatter3d(
            x=[float(target[0])], y=[float(target[1])], z=[float(target[2])], mode="markers",
            marker=dict(size=1, opacity=0.0, color=[t3.BODY_TEMP_C], colorscale=TEMP_COLORSCALE,
                        cmin=t3.BODY_TEMP_C, cmax=float(temp_max_c), showscale=True,
                        colorbar=dict(title=dict(text="Temperature (°C)", font=dict(color=INK)), orientation="h",
                                      len=0.45, x=0.5, y=0.02, thickness=12, tickfont=dict(color=INK))),
            showlegend=False, hoverinfo="skip"))

    axis = dict(backgroundcolor=BG, gridcolor=LINE, zerolinecolor=LINE, showbackground=True,
                color=MUTED, tickfont=dict(color=MUTED))
    fig.update_layout(
        title=dict(text=title, x=0.01, font=dict(size=14, color=INK)),
        scene=dict(
            xaxis=dict(title="x · Right + (mm)", range=[float(lo[0]), float(hi[0])], **axis),
            yaxis=dict(title="y · Anterior + (mm)", range=[float(lo[1]), float(hi[1])], **axis),
            zaxis=dict(title="z · Superior + (mm)", range=[float(lo[2]), float(hi[2])], **axis),
            aspectmode="data", camera=CAMERAS.get(view, CAMERAS["3D"]), bgcolor=BG),
        uirevision=f"view-{view}", height=700, margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(x=0.99, xanchor="right", y=0.99, bgcolor="rgba(18,24,32,0.92)", bordercolor=LINE, borderwidth=1,
                    font=dict(color=INK, size=12)),
        paper_bgcolor=PANEL, font=dict(color=INK))
    return fig


def build_slices(
    vol: t3.Volume3D,
    field_full: Optional[np.ndarray],
    target: np.ndarray,
    entry: np.ndarray,
    field_label: str,
    vmin: float,
    vmax: float,
    colorscale: str = TEMP_COLORSCALE,
    show_labels: bool = True,
) -> go.Figure:
    """Axial / coronal / sagittal slices through the target: anatomy + CSF/vessel labels + field overlay."""
    ti = np.clip(np.rint(vol.world_to_index(target)).astype(int), 0, np.array(vol.shape) - 1)
    cx, cy, cz = (vol.axis_coords(a) for a in range(3))
    views = [
        ("Axial (z = %.1f mm)" % target[2], lambda a: a[:, :, ti[2]].T, cx, cy, (0, 1), "x (mm)", "y (mm)"),
        ("Coronal (y = %.1f mm)" % target[1], lambda a: a[:, ti[1], :].T, cx, cz, (0, 2), "x (mm)", "z (mm)"),
        ("Sagittal (x = %.1f mm)" % target[0], lambda a: a[ti[0], :, :].T, cy, cz, (1, 2), "y (mm)", "z (mm)"),
    ]
    fig = make_subplots(rows=1, cols=3, subplot_titles=[v[0] for v in views], horizontal_spacing=0.05)
    csf = (vol.labels == t3.LABEL_CSF).astype(np.float32)
    ves = (vol.labels == t3.LABEL_VESSEL).astype(np.float32)
    for n, (_, take, xs, ys, (ax_a, ax_b), xl, yl) in enumerate(views, start=1):
        fig.add_trace(go.Heatmap(z=take(vol.anatomy), x=xs, y=ys, colorscale="Greys", showscale=False,
                                 hoverinfo="skip"), row=1, col=n)
        if show_labels:
            for arr, rgba, nm in ((csf, "rgba(67,198,213,0.45)", "CSF"), (ves, "rgba(239,68,68,0.85)", "Vessel")):
                sl = take(arr)
                if sl.max() > 0:
                    fig.add_trace(go.Heatmap(z=np.where(sl > 0, 1.0, np.nan), x=xs, y=ys, showscale=False,
                                             colorscale=[[0, rgba], [1, rgba]], hoverinfo="skip", name=nm),
                                  row=1, col=n)
        if field_full is not None:
            sl = take(field_full)
            if np.isfinite(sl).any():
                fig.add_trace(go.Heatmap(z=sl, x=xs, y=ys, coloraxis="coloraxis", opacity=0.8,
                                         hovertemplate="%{x:.1f}, %{y:.1f} mm: %{z:.3g}<extra>" + field_label + "</extra>"),
                              row=1, col=n)
        fig.add_trace(go.Scatter(x=[entry[ax_a], target[ax_a]], y=[entry[ax_b], target[ax_b]], mode="lines",
                                 line=dict(color=MAGENTA, width=2), showlegend=False, hoverinfo="skip"), row=1, col=n)
        fig.add_trace(go.Scatter(x=[target[ax_a]], y=[target[ax_b]], mode="markers",
                                 marker=dict(size=8, color="#34D399", line=dict(color="white", width=1)),
                                 showlegend=False, hoverinfo="skip"), row=1, col=n)
        anchor = "x" if n == 1 else f"x{n}"
        fig.update_yaxes(scaleanchor=anchor, scaleratio=1, title_text=yl, gridcolor=LINE, color=MUTED, row=1, col=n)
        fig.update_xaxes(title_text=xl, gridcolor=LINE, color=MUTED, row=1, col=n)
    fig.update_layout(
        height=380, margin=dict(l=10, r=10, t=40, b=10), uirevision="keep", paper_bgcolor=PANEL, plot_bgcolor=BG,
        font=dict(color=INK),
        coloraxis=dict(colorscale=colorscale, cmin=float(vmin), cmax=float(vmax),
                       colorbar=dict(title=dict(text=field_label, font=dict(color=INK)), thickness=12, len=0.9,
                                     tickfont=dict(color=INK))))
    return fig


def build_time_curve(result: dict, exposure_s: float) -> go.Figure:
    """Peak temperature and cumulative CEM43 versus time, with and without the CSF model."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    hw, hn = result["with_csf"]["history"], result["no_csf"]["history"]
    fig.add_trace(go.Scatter(x=hw["t_s"], y=hw["max_temp_c"], name="Max temp + CSF (°C)", line=dict(color=CYAN, width=2)),
                  secondary_y=False)
    fig.add_trace(go.Scatter(x=hn["t_s"], y=hn["max_temp_c"], name="Max temp − CSF (°C)",
                             line=dict(color=CYAN, width=2, dash="dot")), secondary_y=False)
    if hw["max_cem43_min"] is not None and hn["max_cem43_min"] is not None:
        fig.add_trace(go.Scatter(x=hw["t_s"], y=np.maximum(hw["max_cem43_min"], 1e-3), name="Max CEM43 + CSF (min)",
                                 line=dict(color=MAGENTA, width=2)), secondary_y=True)
        fig.add_trace(go.Scatter(x=hn["t_s"], y=np.maximum(hn["max_cem43_min"], 1e-3), name="Max CEM43 − CSF (min)",
                                 line=dict(color=MAGENTA, width=2, dash="dot")), secondary_y=True)
        t_end = float(hw["t_s"][-1])
        fig.add_trace(go.Scatter(x=[0, t_end], y=[t3.CEM43_ABLATION_MIN] * 2, name="CEM43 = 240 min",
                                 line=dict(color=AMBER, width=1, dash="dash")), secondary_y=True)
        fig.update_yaxes(type="log", title_text="Cumulative CEM43 (min, log)", secondary_y=True, gridcolor=LINE,
                         color=MUTED)
    fig.add_vline(x=exposure_s, line=dict(color=MUTED, width=1, dash="dot"))
    fig.update_yaxes(title_text="Peak temperature (°C)", secondary_y=False, gridcolor=LINE, color=MUTED)
    fig.update_xaxes(title_text="Time (s) · dotted vertical line = laser off", gridcolor=LINE, color=MUTED)
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor=PANEL, plot_bgcolor=BG,
                      font=dict(color=INK), legend=dict(orientation="h", y=1.12, font=dict(color=INK)),
                      uirevision="keep")
    return fig