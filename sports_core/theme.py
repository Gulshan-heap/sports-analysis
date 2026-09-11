"""
Shared theme + UI primitives
============================
One dark design system for every sport. The only thing that changes between
sports is the accent colour, which is injected as a CSS custom property so the
same stylesheet re-skins itself.
"""

import streamlit as st

_CSS = """
<style>
/* -- fonts & root -- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

:root {
    --accent:      __ACCENT__;
    --accent-dark: __ACCENT_DARK__;
    --accent-soft: __ACCENT_SOFT__;
    --bg:          #0b0f19;
    --panel:       #0d1117;
    --line:        #21262d;
    --text:        #e6edf3;
    --muted:       #8b949e;
    --team1:       #e94560;
    --team2:       #4a90e2;
}

/* -- hide default streamlit chrome -- */
#MainMenu, footer, header { visibility: hidden; }

/* -- page background -- */
.stApp { background: var(--bg); }

/* -- sidebar -- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #0f1923 100%);
    border-right: 1px solid #1e2d3d;
}
section[data-testid="stSidebar"] * { color: #c9d1d9 !important; }
section[data-testid="stSidebar"] .stTextInput input,
section[data-testid="stSidebar"] .stNumberInput input,
section[data-testid="stSidebar"] .stSlider,
section[data-testid="stSidebar"] .stSelectbox select {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    color: var(--text) !important;
}
section[data-testid="stSidebar"] hr { border-color: var(--line); }

/* -- sport switcher -- */
.sport-switch-label {
    color: #484f58 !important;
    font-size: 0.65rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-bottom: 0.15rem;
}
.brand { text-align: center; padding: 1.1rem 0 0.4rem; }
.brand-mark { font-size: 2.2rem; line-height: 1; }
.brand-name {
    color: var(--text) !important;
    font-weight: 800;
    font-size: 1.05rem;
    margin-top: 0.3rem;
    letter-spacing: -0.2px;
}
.brand-sub { color: var(--muted) !important; font-size: 0.7rem; margin-top: 0.1rem; }

/* -- info chip (device / model status) -- */
.chip {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 0.6rem 0.9rem;
    margin-bottom: 0.8rem;
    text-align: center;
}
.chip.ok  { border-color: #1e4620; }
.chip.bad { border-color: #f85149; }
.chip-title { font-size: 0.8rem; font-weight: 700; }
.chip-sub   { color: #484f58 !important; font-size: 0.68rem; margin-top: 0.15rem; }

/* -- hero banner -- */
.hero {
    background: linear-gradient(135deg, #0d1117 0%, #0f2027 40%, #1a1a2e 100%);
    border: 1px solid var(--line);
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    display: flex;
    align-items: center;
    gap: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: "";
    position: absolute;
    top: -60px; right: -60px;
    width: 300px; height: 300px;
    background: radial-gradient(circle, var(--accent-soft) 0%, transparent 70%);
    pointer-events: none;
}
.hero-icon { font-size: 4rem; line-height: 1; }
.hero-text h1 {
    color: #ffffff;
    font-size: 2.4rem;
    font-weight: 800;
    margin: 0 0 0.3rem;
    letter-spacing: -0.5px;
}
.hero-text h1 span { color: var(--accent); }
.hero-text p { color: var(--muted); font-size: 1rem; margin: 0; }
.hero-badges { display: flex; gap: 0.5rem; margin-top: 0.8rem; flex-wrap: wrap; }
.badge {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 20px;
    padding: 0.2rem 0.7rem;
    font-size: 0.72rem;
    color: var(--muted);
    font-weight: 600;
}

/* -- upload zone -- */
.upload-zone {
    background: var(--panel);
    border: 2px dashed #30363d;
    border-radius: 12px;
    padding: 2.5rem;
    text-align: center;
    transition: border-color 0.2s;
}
.upload-zone:hover { border-color: var(--accent); }

/* -- run button -- */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, var(--accent), var(--accent-dark)) !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 0.75rem 2rem !important;
    letter-spacing: 0.3px !important;
    box-shadow: 0 4px 15px var(--accent-soft) !important;
    transition: all 0.2s !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px var(--accent-soft) !important;
}

/* -- section heading -- */
.sec-head {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin: 2rem 0 1rem;
}
.sec-head-line {
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, var(--line), transparent);
}
.sec-head span {
    color: var(--text);
    font-size: 1.1rem;
    font-weight: 700;
    white-space: nowrap;
}
.sec-head-icon { font-size: 1.2rem; }

/* -- KPI cards -- */
.kpi-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 1rem; margin-bottom: 1rem; }
.kpi {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    position: relative;
    overflow: hidden;
}
.kpi::after {
    content: "";
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 0 0 12px 12px;
}
.kpi.red::after    { background: var(--team1); }
.kpi.blue::after   { background: var(--team2); }
.kpi.gray::after   { background: #30363d; }
.kpi.green::after  { background: #3fb950; }
.kpi.accent::after { background: var(--accent); }
.kpi-label { color: var(--muted); font-size: 0.72rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 0.4rem; }
.kpi-value { color: var(--text); font-size: 2rem; font-weight: 800; line-height: 1; }
.kpi-sub   { color: var(--muted); font-size: 0.75rem; margin-top: 0.25rem; }

/* -- team comparison strip -- */
.team-strip {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 12px;
    overflow: hidden;
    margin-bottom: 1rem;
}
.team-strip-header {
    display: flex;
    justify-content: space-between;
    padding: 0.9rem 1.2rem 0.4rem;
}
.team-strip-header .t1 { color: var(--team1); font-weight: 700; font-size: 0.9rem; }
.team-strip-header .t2 { color: var(--team2); font-weight: 700; font-size: 0.9rem; }
.team-strip-header .label { color: var(--muted); font-size: 0.8rem; }
.control-bar { height: 10px; display: flex; margin: 0 1.2rem 1rem; border-radius: 5px; overflow: hidden; }
.cb-t1 { background: linear-gradient(90deg, #e94560, #c0392b); }
.cb-t2 { background: linear-gradient(90deg, #3a7bd5, #4a90e2); }

/* -- stat row -- */
.stat-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.6rem 1.2rem;
    border-top: 1px solid #161b22;
}
.stat-row:nth-child(even) { background: #080c12; }
.stat-val { font-size: 1.1rem; font-weight: 700; }
.stat-val.red  { color: var(--team1); }
.stat-val.blue { color: var(--team2); }
.stat-name { color: var(--muted); font-size: 0.8rem; font-weight: 600; text-align: center; }

/* -- player table -- */
.stDataFrame { border: 1px solid var(--line) !important; border-radius: 10px !important; }

/* -- frame explorer -- */
.frame-meta {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 10px;
    padding: 1rem 1.4rem;
    display: flex;
    gap: 2rem;
    margin-top: 0.8rem;
    flex-wrap: wrap;
}
.fm-item { display: flex; flex-direction: column; }
.fm-label { color: var(--muted); font-size: 0.7rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.8px; }
.fm-value { color: var(--text); font-size: 1rem; font-weight: 700; }

/* -- feature card -- */
.feat-card {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 12px;
    padding: 1.3rem;
    height: 100%;
    transition: border-color 0.2s, transform 0.2s;
}
.feat-card:hover { border-color: var(--accent); transform: translateY(-2px); }
.feat-icon { font-size: 1.8rem; margin-bottom: 0.5rem; }
.feat-title { color: var(--text); font-size: 0.95rem; font-weight: 700; margin-bottom: 0.3rem; }
.feat-desc  { color: var(--muted); font-size: 0.78rem; line-height: 1.5; }

/* -- steps panel -- */
.steps {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 12px;
    padding: 1.5rem 2rem;
    color: var(--muted);
    line-height: 2;
}
.steps b { color: var(--text); font-weight: 700; }
.steps .hl { color: var(--accent); font-weight: 700; }

/* -- progress bar -- */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, var(--accent), var(--accent-dark)) !important;
}

/* -- tabs -- */
.stTabs [role="tab"] { color: var(--muted) !important; font-weight: 600; }
.stTabs [role="tab"][aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom-color: var(--accent) !important;
}

/* -- metrics (football pipeline stats) -- */
div[data-testid="stMetric"] {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 12px;
    padding: 0.9rem 1.1rem;
}
div[data-testid="stMetricLabel"] p {
    color: var(--muted) !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.6px;
}
div[data-testid="stMetricValue"] {
    color: var(--text) !important;
    font-size: 1.6rem !important;
    font-weight: 800 !important;
}

/* -- alert overrides -- */
.stAlert { border-radius: 10px !important; }

/* -- video -- */
video { border-radius: 10px; }

/* -- small screens -- */
@media (max-width: 900px) {
    .hero { flex-direction: column; text-align: center; padding: 1.8rem; }
    .kpi-grid { grid-template-columns: repeat(2, 1fr); }
}
</style>
"""


def _soften(hex_colour, alpha=0.28):
    """`#rrggbb` -> `rgba(r,g,b,alpha)`, used for glows and shadows."""
    h = hex_colour.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def inject(sport):
    """Apply the shared stylesheet, skinned with this sport's accent colour."""
    css = (_CSS
           .replace("__ACCENT__", sport.accent)
           .replace("__ACCENT_DARK__", sport.accent_dark)
           .replace("__ACCENT_SOFT__", _soften(sport.accent)))
    st.markdown(css, unsafe_allow_html=True)


# -- primitives both sport pages draw with -----------------------------------

def sec(icon, label):
    """A section heading with a trailing rule."""
    st.markdown(f"""
    <div class="sec-head">
        <span class="sec-head-icon">{icon}</span>
        <span>{label}</span>
        <div class="sec-head-line"></div>
    </div>""", unsafe_allow_html=True)


def kpi(label, value, sub="", colour="gray"):
    """Return the HTML for one KPI card (compose several inside `.kpi-grid`)."""
    sub_html = f"<div class='kpi-sub'>{sub}</div>" if sub else ""
    return f"""
    <div class="kpi {colour}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {sub_html}
    </div>"""


def kpi_grid(*cards):
    """Render a row of `kpi()` cards."""
    st.markdown(f"<div class='kpi-grid'>{''.join(cards)}</div>",
                unsafe_allow_html=True)


def hero(sport):
    """The page banner for a sport."""
    badges = "".join(f"<span class='badge'>{b}</span>" for b in sport.badges)
    st.markdown(f"""
    <div class="hero">
        <div class="hero-icon">{sport.icon}</div>
        <div class="hero-text">
            <h1>{sport.label} <span>Analysis</span></h1>
            <p>{sport.tagline}</p>
            <div class="hero-badges">{badges}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def chip(title, sub="", colour="#8b949e", state=""):
    """A small bordered status card for the sidebar."""
    sub_html = f"<div class='chip-sub'>{sub}</div>" if sub else ""
    st.markdown(f"""
    <div class="chip {state}">
        <div class="chip-title" style="color:{colour};">{title}</div>
        {sub_html}
    </div>""", unsafe_allow_html=True)


def rule():
    st.markdown("<hr style='border-color:#21262d;margin:0.5rem 0 0.9rem;'>",
                unsafe_allow_html=True)


def spacer(height="1rem"):
    st.markdown(f"<div style='height:{height}'></div>", unsafe_allow_html=True)


def feature_cards(features, per_row=4):
    """`features` is a sequence of (icon, title, description)."""
    cols = st.columns(per_row)
    for i, (icon, title, desc) in enumerate(features):
        with cols[i % per_row]:
            st.markdown(f"""
            <div class="feat-card">
                <div class="feat-icon">{icon}</div>
                <div class="feat-title">{title}</div>
                <div class="feat-desc">{desc}</div>
            </div>""", unsafe_allow_html=True)
            spacer("0.6rem")
