"""Dark clinical-workstation theme for the ThermoSEEG 3D station (presentation only).

Palette from the team's desktop prototype: background #0B0F14, panels #121820,
cyan #43C6D5, magenta #D85C9E, amber #F0A34A.  Independent of thermoseeg_theme.py
(the light theme used by the 2D page), so the two pages do not interfere.
Visual check on a real browser is still required; Streamlit's internal markup can change.
"""
from __future__ import annotations

import html
from typing import Iterable, List, Sequence, Tuple

import streamlit as st

BG = "#0B0F14"
PANEL = "#121820"
PANEL_2 = "#1A2330"
LINE = "#243040"
INK = "#E6EDF3"
MUTED = "#8B98A9"
CYAN = "#43C6D5"
MAGENTA = "#D85C9E"
AMBER = "#F0A34A"
GREEN = "#34D399"
RED = "#F87171"

TONES = {
    "pass": (GREEN, "#10281F"),
    "warn": (AMBER, "#2D2110"),
    "danger": (RED, "#321616"),
    "info": (CYAN, "#0F2A2F"),
    "neutral": (MUTED, PANEL_2),
}
STATUS_TONE = {"HEURISTIC_CHECKS_PASSED": "pass", "WARNING": "warn", "DANGER": "danger"}

_CSS = f"""
<style>
:root {{ color-scheme: dark; }}
.stApp {{ background:{BG}; color:{INK}; }}
[data-testid="stHeader"] {{ background:transparent; }}
#MainMenu, footer {{ visibility:hidden; }}
.block-container {{ padding-top:.6rem; padding-bottom:3rem; max-width:1700px; }}
[data-testid="stSidebar"] {{ background:{PANEL}; border-right:1px solid {LINE}; }}
.stApp h1,.stApp h2,.stApp h3,.stApp h4,.stApp p,.stApp li,.stApp label,
[data-testid="stMarkdownContainer"],[data-testid="stWidgetLabel"] p,[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,[data-testid="stSidebar"] span,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3 {{ color:{INK}; }}
[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3 {{ font-size:11px !important; letter-spacing:.09em; text-transform:uppercase; color:{MUTED} !important; }}
[data-testid="stCaptionContainer"],[data-testid="stCaptionContainer"] * {{ color:{MUTED} !important; }}
hr {{ border-color:{LINE}; }}

/* inputs */
[data-baseweb="input"],[data-baseweb="base-input"],[data-baseweb="select"] > div,.stNumberInput input,textarea {{
  background:{BG} !important; color:{INK} !important; border-color:{LINE} !important; }}
[data-baseweb="input"] input {{ color:{INK} !important; -webkit-text-fill-color:{INK}; }}
[data-baseweb="popover"] ul,[role="listbox"] {{ background:{PANEL} !important; color:{INK} !important; }}
[data-baseweb="tag"] {{ background:{PANEL_2} !important; color:{INK} !important; }}
[data-testid="stSlider"] [role="slider"] {{ background:{CYAN} !important; }}
[data-testid="stSliderThumbValue"] {{ color:{CYAN} !important; }}
[data-testid="stTickBarMin"],[data-testid="stTickBarMax"] {{ color:{MUTED} !important; }}
[data-testid="stFileUploaderDropzone"] {{ background:{BG} !important; border:1.5px dashed {LINE}; }}
[data-testid="stFileUploaderDropzone"] * {{ color:{MUTED} !important; }}
[data-testid="stFileUploaderDropzone"] button {{ background:{CYAN} !important; color:{BG} !important; border:none !important; }}
[data-testid="stFileUploaderDropzone"] button * {{ color:{BG} !important; }}

/* buttons */
.stButton > button,.stDownloadButton > button {{
  background:{PANEL_2}; color:{INK}; border:1px solid {LINE}; border-radius:10px; font-weight:600; }}
.stButton > button:hover,.stDownloadButton > button:hover {{ border-color:{CYAN}; color:{CYAN}; }}
.stButton > button[kind="primary"] {{ background:{CYAN}; color:{BG}; border-color:{CYAN}; }}
.stButton > button[kind="primary"] * {{ color:{BG} !important; }}

/* tabs */
.stTabs [data-baseweb="tab-list"] {{ gap:6px; border-bottom:1px solid {LINE}; }}
.stTabs [data-baseweb="tab"] {{ color:{MUTED} !important; font-weight:600; }}
.stTabs [aria-selected="true"] {{ color:{CYAN} !important; }}
.stTabs [data-baseweb="tab-highlight"] {{ background:{CYAN}; }}

/* containers */
[data-testid="stExpander"] {{ background:{PANEL}; border:1px solid {LINE}; border-radius:12px; }}
[data-testid="stExpander"] summary,[data-testid="stExpander"] summary * {{ color:{INK} !important; }}
[data-testid="stAlert"] {{ background:{PANEL_2} !important; border:1px solid {LINE}; border-radius:12px; }}
[data-testid="stAlert"],[data-testid="stAlert"] * {{ color:{INK} !important; }}
[data-testid="stPlotlyChart"] {{ border:1px solid {LINE}; border-radius:12px; overflow:hidden; background:{PANEL}; }}

/* workstation parts */
.ws-topbar {{ display:flex; align-items:center; gap:24px; background:{PANEL}; border:1px solid {LINE};
  border-radius:14px; padding:10px 18px; margin:4px 0 12px 0; }}
.ws-logo {{ display:flex; align-items:center; gap:10px; font-weight:700; font-size:18px; color:{INK}; }}
.ws-logo-mark {{ width:30px; height:30px; border-radius:9px; background:{CYAN}; color:{BG};
  display:flex; align-items:center; justify-content:center; }}
.ws-logo b {{ color:{CYAN}; }}
.ws-nav {{ display:flex; gap:4px; flex:1; }}
.ws-nav span {{ padding:6px 13px; border-radius:9px; font-size:13px; font-weight:500; color:{MUTED}; }}
.ws-nav span.active {{ background:{CYAN}; color:{BG}; font-weight:700; }}
.ws-chip {{ font-size:11.5px; font-weight:600; padding:5px 11px; border-radius:999px; background:{PANEL_2}; color:{MUTED}; }}
.ws-chip.hot {{ background:#0F2A2F; color:{CYAN}; }}
.ws-sec {{ display:flex; align-items:center; gap:12px; margin:4px 0 8px 0; }}
.ws-sec-num {{ width:26px; height:26px; border-radius:8px; background:#0F2A2F; color:{CYAN};
  font-weight:700; font-size:13px; display:flex; align-items:center; justify-content:center; }}
.ws-sec-title {{ font-size:16px; font-weight:700; color:{INK}; }}
.ws-sec-sub {{ font-size:12.5px; color:{MUTED}; margin-left:auto; }}
.ws-kpis {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:10px; margin:4px 0 10px 0; }}
.ws-kpi {{ background:{PANEL}; border:1px solid {LINE}; border-radius:12px; padding:12px 14px; }}
.ws-kpi .lbl {{ font-size:10.5px; font-weight:600; letter-spacing:.07em; text-transform:uppercase; color:{MUTED}; }}
.ws-kpi .val {{ font-family:Consolas,'Courier New',monospace; font-size:22px; font-weight:600; margin-top:5px; }}
.ws-kpi .sub {{ font-size:11.5px; color:{MUTED}; margin-top:2px; }}
.ws-pill {{ display:inline-block; padding:4px 11px; border-radius:999px; font-size:12px; font-weight:700; letter-spacing:.02em; }}
.ws-card {{ background:{PANEL}; border:1px solid {LINE}; border-radius:12px; padding:12px 14px; margin-bottom:10px; }}
.ws-card h4 {{ margin:0 0 8px 0; font-size:10.5px; font-weight:700; letter-spacing:.09em; text-transform:uppercase; color:{MUTED} !important; }}
.ws-row {{ display:flex; justify-content:space-between; gap:10px; font-size:13px; padding:2px 0; color:{INK}; }}
.ws-row span:first-child {{ color:{MUTED}; }}
.ws-row span:last-child {{ font-family:Consolas,'Courier New',monospace; text-align:right; }}
.ws-table {{ width:100%; border-collapse:collapse; font-size:12.5px; }}
.ws-table th {{ text-align:left; color:{MUTED}; font-weight:600; letter-spacing:.05em; text-transform:uppercase; font-size:10.5px;
  border-bottom:1px solid {LINE}; padding:6px 8px; }}
.ws-table td {{ padding:6px 8px; border-bottom:1px solid {LINE}; color:{INK}; font-family:Consolas,'Courier New',monospace; }}
@media (max-width:768px) {{ .ws-topbar {{ flex-wrap:wrap; gap:10px; }} .ws-nav {{ order:3; flex:0 0 100%; overflow-x:auto; }} }}
</style>
"""

NAV = ["Imaging", "Planning", "Thermal Simulation", "Report"]


def inject_theme() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def render_topbar(active: str, chips: Sequence[Tuple[str, bool]] = ()) -> None:
    idx = NAV.index(active) if active in NAV else 0
    items = "".join(f'<span class="{"active" if i == idx else ""}">{html.escape(n)}</span>' for i, n in enumerate(NAV))
    chip_html = "".join(f'<span class="ws-chip{" hot" if hot else ""}">{html.escape(t)}</span>' for t, hot in chips)
    st.markdown(
        f'<div class="ws-topbar"><div class="ws-logo"><div class="ws-logo-mark">🧠</div>Thermo<b>SEEG</b></div>'
        f'<div class="ws-nav">{items}</div>{chip_html}</div>', unsafe_allow_html=True)


def section_header(num: str, title: str, subtitle: str = "") -> None:
    sub = f'<div class="ws-sec-sub">{html.escape(subtitle)}</div>' if subtitle else ""
    st.markdown(f'<div class="ws-sec"><div class="ws-sec-num">{html.escape(num)}</div>'
                f'<div class="ws-sec-title">{html.escape(title)}</div>{sub}</div>', unsafe_allow_html=True)


def pill_html(text: str, tone: str) -> str:
    fg, bg = TONES.get(tone, TONES["neutral"])
    return f'<span class="ws-pill" style="color:{fg};background:{bg}">{html.escape(text)}</span>'


def kpi_tiles(items: Iterable[Tuple[str, str, str, str]]) -> None:
    """items: (label, value, sub, tone); tone colours the value."""
    cells = []
    for label, value, sub, tone in items:
        fg = INK if tone == "neutral" else TONES.get(tone, TONES["neutral"])[0]
        cells.append(f'<div class="ws-kpi"><div class="lbl">{html.escape(label)}</div>'
                     f'<div class="val" style="color:{fg}">{html.escape(value)}</div>'
                     f'<div class="sub">{html.escape(sub)}</div></div>')
    st.markdown(f'<div class="ws-kpis">{"".join(cells)}</div>', unsafe_allow_html=True)


def panel(title: str, rows: Iterable[Tuple[str, str]]) -> None:
    body = "".join(f'<div class="ws-row"><span>{html.escape(k)}</span><span>{html.escape(str(v))}</span></div>' for k, v in rows)
    st.markdown(f'<div class="ws-card"><h4>{html.escape(title)}</h4>{body}</div>', unsafe_allow_html=True)


def html_table(columns: List[str], rows: Iterable[Sequence]) -> None:
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in columns)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in r) + "</tr>" for r in rows)
    st.markdown(f'<div class="ws-card"><table class="ws-table"><tr>{head}</tr>{body}</table></div>', unsafe_allow_html=True)


def sidebar_brand() -> None:
    st.markdown(f'<div class="ws-logo" style="margin:4px 0 6px 0"><div class="ws-logo-mark">⚙️</div>Configuration</div>',
                unsafe_allow_html=True)