"""
thermoseeg_theme.py
===================
Presentation layer ONLY for ThermoSEEG_RC. Contains no planning, segmentation or
thermal logic: just design tokens, CSS injection and small HTML helpers so the
Streamlit app matches the clinical-dashboard look (light canvas, teal brand,
white rounded cards, pill badges, top navigation bar).
"""

import html
from typing import List, Tuple

import streamlit as st

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------
BRAND = "#0D9488"        # teal (logo / accents)
BRAND_SOFT = "#E6F6F4"
PRIMARY = "#2563EB"      # action blue (primary buttons / active nav)
PRIMARY_DARK = "#1D4ED8"
INK = "#0F172A"
MUTED = "#64748B"
LINE = "#E2E8F0"
CANVAS = "#F4F7FA"
CARD = "#FFFFFF"

TONES = {
    "safe":    ("#047857", "#D1FAE5"),
    "warn":    ("#B45309", "#FEF3C7"),
    "danger":  ("#B91C1C", "#FEE2E2"),
    "info":    ("#1D4ED8", "#DBEAFE"),
    "neutral": ("#334155", "#F1F5F9"),
}

STATUS_TONE = {"HEURISTIC_CHECKS_PASSED": "info", "WARNING": "warn", "DANGER": "danger"}
CONFIDENCE_TONE = {"HIGH": "info", "MODERATE": "warn", "LOW": "danger"}

NAV_TABS = ["Imaging", "Planning", "Thermal Simulation", "Report"]

_CSS = f"""
<style>
html, body, [class*="css"], .stApp {{
    font-family: Arial, -apple-system, 'Segoe UI', sans-serif;
    color: {INK};
}}
.stApp {{ background: {CANVAS}; }}
#MainMenu, footer {{ visibility: hidden; }}
[data-testid="stHeader"] {{ background: transparent; height: 0; }}
.block-container {{ padding-top: 0.6rem; padding-bottom: 3rem; max-width: 1500px; }}

/* ---------- top navigation bar ---------- */
.tsg-topbar {{
    display: flex; align-items: center; gap: 28px;
    background: {CARD}; border: 1px solid {LINE}; border-radius: 14px;
    padding: 10px 20px; margin: 4px 0 14px 0;
    box-shadow: 0 1px 2px rgba(15,23,42,.04);
}}
.tsg-logo {{ display:flex; align-items:center; gap:10px; font-weight:700; font-size:18px; }}
.tsg-logo-mark {{
    width:30px; height:30px; border-radius:9px; background:{BRAND}; color:#fff;
    display:flex; align-items:center; justify-content:center; font-size:16px;
}}
.tsg-logo b {{ color:{BRAND}; font-weight:700; }}
.tsg-nav {{ display:flex; gap:4px; flex:1; }}
.tsg-nav span {{
    padding:7px 14px; border-radius:9px; font-size:13px; font-weight:500; color:{MUTED};
}}
.tsg-nav span.active {{ background:{PRIMARY}; color:#fff; }}
.tsg-nav span.done {{ color:{BRAND}; }}
.tsg-chip {{
    font-size:11.5px; font-weight:600; padding:5px 11px; border-radius:999px;
    background:{BRAND_SOFT}; color:{BRAND};
}}
.tsg-chip.neutral {{ background:#F1F5F9; color:{MUTED}; }}

/* ---------- section headers ---------- */
.tsg-sec {{ display:flex; align-items:center; gap:12px; margin: 2px 0 10px 0; }}
.tsg-sec-num {{
    width:26px; height:26px; border-radius:8px; background:{BRAND_SOFT}; color:{BRAND};
    font-weight:700; font-size:13px; display:flex; align-items:center; justify-content:center;
}}
.tsg-sec-title {{ font-size:16px; font-weight:700; letter-spacing:-.01em; }}
.tsg-sec-sub {{ font-size:12.5px; color:{MUTED}; margin-left:auto; }}

.tsg-eyebrow {{
    font-size:10.5px; font-weight:600; letter-spacing:.08em; text-transform:uppercase; color:{MUTED};
}}

/* ---------- cards (st.container(border=True)) ---------- */
[data-testid="stVerticalBlockBorderWrapper"] {{
    background: {CARD};
    border: 1px solid {LINE} !important;
    border-radius: 14px !important;
    box-shadow: 0 1px 2px rgba(15,23,42,.04);
}}
[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stVerticalBlockBorderWrapper"] {{
    box-shadow: none; background: #FAFCFD;
}}

/* ---------- KPI tiles ---------- */
.tsg-kpis {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap:12px; margin: 4px 0 14px 0; }}
.tsg-kpi {{
    background:{CARD}; border:1px solid {LINE}; border-radius:12px; padding:14px 16px;
}}
.tsg-kpi .lbl {{ font-size:10.5px; font-weight:600; letter-spacing:.07em; text-transform:uppercase; color:{MUTED}; }}
.tsg-kpi .val {{ font-family:Consolas, 'Courier New', monospace; font-size:24px; font-weight:600; margin-top:6px; }}
.tsg-kpi .sub {{ font-size:11.5px; color:{MUTED}; margin-top:2px; }}

.tsg-pill {{
    display:inline-block; padding:4px 11px; border-radius:999px; font-size:12px; font-weight:700;
    letter-spacing:.02em;
}}

/* ---------- sidebar ---------- */
[data-testid="stSidebar"] {{ background:{CARD}; border-right:1px solid {LINE}; }}
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{
    font-size:11px !important; font-weight:700 !important; letter-spacing:.09em;
    text-transform:uppercase; color:{MUTED} !important;
}}

/* ---------- buttons ---------- */
.stButton > button, .stDownloadButton > button {{
    border-radius:10px; font-weight:600; font-size:13.5px; padding:.5rem 1rem;
    border:1px solid {LINE}; background:{CARD}; color:{INK}; transition: all .15s ease;
}}
.stButton > button:hover, .stDownloadButton > button:hover {{
    border-color:{PRIMARY}; color:{PRIMARY};
}}
.stButton > button[kind="primary"] {{
    background:{PRIMARY}; border-color:{PRIMARY}; color:#fff;
}}
.stButton > button[kind="primary"]:hover {{ background:{PRIMARY_DARK}; border-color:{PRIMARY_DARK}; color:#fff; }}

/* ---------- inputs ---------- */
[data-baseweb="input"], [data-baseweb="select"] > div, .stNumberInput input {{ border-radius:9px !important; }}
[data-testid="stFileUploader"] section {{
    border:1.5px dashed #B6C2D1; border-radius:12px; background:#FAFCFD;
}}
.stSlider [role="slider"] {{ background:{PRIMARY} !important; }}

/* ---------- tabs ---------- */
.stTabs [data-baseweb="tab-list"] {{ gap:6px; border-bottom:1px solid {LINE}; }}
.stTabs [data-baseweb="tab"] {{
    border-radius:9px 9px 0 0; padding:8px 14px; font-weight:600; font-size:13px; color:{MUTED};
}}
.stTabs [aria-selected="true"] {{ color:{PRIMARY} !important; }}
.stTabs [data-baseweb="tab-highlight"] {{ background:{PRIMARY}; }}

/* ---------- metrics / tables / images / alerts ---------- */
[data-testid="stMetric"] {{
    background:#F8FAFC; border:1px solid {LINE}; border-radius:12px; padding:12px 14px;
}}
[data-testid="stMetricLabel"] p {{
    font-size:10.5px !important; font-weight:600; letter-spacing:.07em; text-transform:uppercase; color:{MUTED};
}}
[data-testid="stMetricValue"] {{ font-family:Consolas, 'Courier New', monospace; font-weight:600; }}
[data-testid="stDataFrame"] {{ border:1px solid {LINE}; border-radius:12px; overflow:hidden; }}
[data-testid="stImage"] img {{ border-radius:10px; border:1px solid {LINE}; }}
[data-testid="stAlert"] {{ border-radius:12px; border:1px solid {LINE}; }}
:where(button, input, textarea, [tabindex]):focus-visible {{ outline:3px solid {PRIMARY}; outline-offset:2px; }}
@media (max-width: 768px) {{
    .block-container {{ padding-left:1rem; padding-right:1rem; }}
    .tsg-topbar {{ flex-wrap:wrap; gap:10px; padding:10px 12px; }}
    .tsg-nav {{ order:3; flex:0 0 100%; width:100%; min-width:0; overflow-x:auto; }}
    .tsg-nav span {{ white-space:nowrap; padding:7px 10px; }}
    .tsg-sec {{ flex-wrap:wrap; }}
    .tsg-sec-sub {{ width:100%; margin-left:38px; }}
    .tsg-kpis {{ grid-template-columns:repeat(auto-fit,minmax(145px,1fr)); }}
}}
@media (max-width: 400px) {{
    .block-container {{ padding-left:.65rem; padding-right:.65rem; }}
    .tsg-logo {{ font-size:16px; }}
    .tsg-kpis {{ grid-template-columns:1fr 1fr; gap:7px; }}
    .tsg-kpi {{ padding:10px; }}
    .tsg-kpi .val {{ font-size:19px; }}
    .stButton > button, .stDownloadButton > button {{ width:100%; white-space:normal; }}
}}
[data-testid="stCaptionContainer"] {{ color:{MUTED}; }}
hr {{ border-color:{LINE}; margin: 1.2rem 0; }}
.stProgress > div > div > div > div {{ background:{BRAND}; }}
/* ---------- force readable light palette (even if the OS / Streamlit is in dark mode) ---------- */
:root {{ color-scheme: light; }}
.stApp, [data-testid="stSidebar"], [data-testid="stSidebarContent"] {{ color:{INK}; }}
[data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label,
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li,
[data-testid="stRadio"] label, [data-testid="stRadio"] label p,
[data-testid="stCheckbox"] label, [data-testid="stCheckbox"] label p,
[data-testid="stSlider"] label, [data-testid="stSlider"] p,
[data-testid="stTickBarMin"], [data-testid="stTickBarMax"],
[data-testid="stSliderThumbValue"], [data-testid="stNumberInput"] label,
[data-testid="stFileUploader"] label, [data-testid="stSidebar"] label,
[data-testid="stSidebar"] p, [data-testid="stSidebar"] span {{ color:{INK} !important; }}
[data-testid="stSliderThumbValue"] {{ color:{PRIMARY} !important; }}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p,
[data-testid="stImageCaption"] {{ color:{MUTED} !important; }}
.tsg-nav span.active {{ color:#fff !important; }}
.tsg-nav span.done {{ color:{BRAND} !important; }}

/* alerts: pale background + dark readable text */
[data-testid="stAlert"], [data-testid="stAlert"] * {{ color:#1E293B !important; }}

/* inputs */
[data-baseweb="input"], [data-baseweb="input"] input, .stNumberInput input,
[data-baseweb="select"] > div {{ background:#fff !important; color:{INK} !important; }}

/* file uploader */
[data-testid="stFileUploaderDropzone"] {{ background:#FAFCFD !important; }}
[data-testid="stFileUploaderDropzone"], [data-testid="stFileUploaderDropzone"] span,
[data-testid="stFileUploaderDropzone"] small, [data-testid="stFileUploaderDropzone"] p {{ color:{MUTED} !important; }}
[data-testid="stFileUploaderDropzone"] button {{
    background:{PRIMARY} !important; color:#fff !important; border:none !important;
}}
[data-testid="stFileUploaderDropzone"] button * {{ color:#fff !important; }}

/* metrics, tabs, dataframe */
[data-testid="stMetricValue"], [data-testid="stMetricValue"] * {{ color:{INK} !important; }}
[data-testid="stMetricLabel"] p {{ color:{MUTED} !important; }}
.stTabs [data-baseweb="tab"] {{ color:{MUTED} !important; }}
.stTabs [aria-selected="true"] {{ color:{PRIMARY} !important; }}
.stButton > button:not([kind="primary"]) {{ background:#fff !important; color:{INK} !important; }}
.stButton > button[kind="primary"], .stButton > button[kind="primary"] * {{ color:#fff !important; }}
.stDownloadButton > button {{ background:#fff !important; color:{INK} !important; }}
.tsg-sec-title {{ color:{INK} !important; }}
.tsg-sec-sub, .tsg-eyebrow, .tsg-kpi .lbl, .tsg-kpi .sub {{ color:{MUTED} !important; }}
</style>
"""



# ---------------------------------------------------------------------------
# Helpers (HTML only)
# ---------------------------------------------------------------------------
def inject_theme() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def render_topbar(active: str, chips: List[Tuple[str, bool]] = ()) -> None:
    """active: one of NAV_TABS. Tabs before it are shown as completed."""
    idx = NAV_TABS.index(active) if active in NAV_TABS else 0
    items = []
    for i, name in enumerate(NAV_TABS):
        cls = "active" if i == idx else ("done" if i < idx else "")
        items.append(f'<span class="{cls}">{html.escape(name)}</span>')
    chip_html = "".join(
        f'<span class="tsg-chip{"" if hot else " neutral"}">{html.escape(t)}</span>' for t, hot in chips
    )
    st.markdown(
        f"""<div class="tsg-topbar">
              <div class="tsg-logo"><div class="tsg-logo-mark">🧠</div>Thermo<b>SEEG</b></div>
              <div class="tsg-nav">{''.join(items)}</div>
              {chip_html}
            </div>""",
        unsafe_allow_html=True,
    )


def section_header(num: str, title: str, subtitle: str = "") -> None:
    sub = f'<div class="tsg-sec-sub">{html.escape(subtitle)}</div>' if subtitle else ""
    st.markdown(
        f"""<div class="tsg-sec"><div class="tsg-sec-num">{html.escape(num)}</div>
            <div class="tsg-sec-title">{html.escape(title)}</div>{sub}</div>""",
        unsafe_allow_html=True,
    )


def eyebrow(text: str) -> None:
    st.markdown(f'<div class="tsg-eyebrow">{html.escape(text)}</div>', unsafe_allow_html=True)


def pill_html(text: str, tone: str) -> str:
    fg, bg = TONES.get(tone, TONES["neutral"])
    return f'<span class="tsg-pill" style="color:{fg};background:{bg}">{html.escape(text)}</span>'


def kpi_tiles(items: List[Tuple[str, str, str, str]]) -> None:
    """items: (label, value, sub, tone). tone colours the value text."""
    cells = []
    for label, value, sub, tone in items:
        fg = TONES.get(tone, TONES["neutral"])[0] if tone != "neutral" else INK
        cells.append(
            f'<div class="tsg-kpi"><div class="lbl">{html.escape(label)}</div>'
            f'<div class="val" style="color:{fg}">{html.escape(value)}</div>'
            f'<div class="sub">{html.escape(sub)}</div></div>'
        )
    st.markdown(f'<div class="tsg-kpis">{"".join(cells)}</div>', unsafe_allow_html=True)


def sidebar_brand() -> None:
    st.markdown(
        '<div class="tsg-logo" style="margin:4px 0 6px 0"><div class="tsg-logo-mark">⚙️</div>Configuration</div>',
        unsafe_allow_html=True,
    )
