# ui_themes.py
from __future__ import annotations

from dataclasses import dataclass
import matplotlib as mpl
import streamlit as st

@dataclass(frozen=True)
class Theme:
    name: str
    # core palette
    primary: str
    success: str
    danger: str
    warning: str
    neutral: str

    # surfaces (NEW: split backgrounds)
    app_bg: str         # whole page background (main)
    sidebar_bg: str     # sidebar background
    panel_bg: str       # cards/expanders/boxes background

    plot_bg: str        # plot area background (inside axes)
    paper_bg: str       # figure canvas background (outside plot area)

    grid: str
    text: str
    spine: str

    palette: list[str]

    # matplotlib typography
    font_family: str = "DejaVu Sans"
    base_fontsize: int = 11
    title_size: int = 13
    label_size: int = 11
    line_width: float = 2.0

    # NEW semantic colors
    market_signal: str = "#FF6B00"
    bull_color: str = "#00C853"
    bear_color: str = "#D50000"
    signal_gold: str = "#1f77b4"

DEFAULT_POLISHED = Theme(
    name="Polished (Light)",
    primary="#4F46E5",
    success="#16A34A",
    danger="#DC2626",
    warning="#D97706",
    neutral="#6B7280",
    app_bg="#FBF7F1",
    sidebar_bg="#F3EEE7",
    panel_bg="#FFFFFF",
    plot_bg="#FFFFFF",
    paper_bg="#FBF7F1",
    grid="#E7E0D8",
    text="#1F2937",
    spine="#E6DDD3",

    palette=[
        "#4E79A7",
        "#F28E2B",
        "#E15759",
        "#76B7B2",
        "#59A14F",
        "#EDC948",
        "#B07AA1",
    ],
)

BLOOMBERG_DARK = Theme(
    name="Bloomberg (Dark)",
    primary="#FF6B00",     # dark orange (titles / sliders) - FF6B00
    success="#34D399",
    danger="#FB7185",
    warning="#FBBF24",
    app_bg="#070B16",
    sidebar_bg = "#0A0F1A",
    panel_bg = "#0F172A",
    plot_bg = "#0B1328",
    paper_bg = "#070B16",
    grid="#2A3348",
    text="#F5A623",        # bright amber → normal text
    neutral="#C58A2B",     # muted amber → secondary text
    spine="#3A425A",

    palette=[
        "#FF6B00",  # orange - F5A623
        "#00C1D4",  # cyan
        "#FF6B6B",  # red
        "#FFD166",  # yellow
        "#7B6CF6",  # purple
        "#06D6A0",  # teal
        "#EF476F",  # magenta
        "#118AB2",  # blue
    ],
)


TRADINGVIEW = Theme(
    name="TradingView (Light)",
    primary="#2962FF",
    success="#089981",
    danger="#F23645",
    warning="#F59E0B",
    neutral="#64748B",
    app_bg="#FFFFFF",
    sidebar_bg="#F4F6F9",
    panel_bg="#FFFFFF",
    plot_bg="#FFFFFF",
    paper_bg="#FFFFFF",
    grid="#E2E8F0",
    text="#0F172A",
    spine="#D8E1EA",

    palette=[
        "#2962FF",
        "#26A69A",
        "#EF5350",
        "#FFA726",
        "#AB47BC",
        "#42A5F5",
        "#66BB6A",
    ],
)

THEMES = {
    DEFAULT_POLISHED.name: DEFAULT_POLISHED,
    BLOOMBERG_DARK.name: BLOOMBERG_DARK,
    TRADINGVIEW.name: TRADINGVIEW,
}

def style_signal_table(df, theme, *, columns=None):
    """
    Style signal-like text columns:
      bullish/buy/long -> theme.success
      bearish/sell/short -> theme.danger
      neutral/hold -> theme.neutral
    """
    def color_signal(val):
        if val is None:
            return ""
        v = str(val).strip().lower()
        if any(x in v for x in ("bull", "buy", "long")):
            return f"color: {theme.success}; font-weight:700"
        if any(x in v for x in ("bear", "sell", "short")):
            return f"color: {theme.danger}; font-weight:700"
        if any(x in v for x in ("neutral", "hold")):
            return f"color: {theme.neutral}; font-weight:700"
        return ""

    if not hasattr(df, "style"):
        return df

    if columns is None:
        cols = [c for c in df.columns if "signal" in str(c).lower() or "state" in str(c).lower()]
    else:
        cols = list(columns)

    if not cols:
        return df.style

    return df.style.applymap(color_signal, subset=cols)

def theme_picker(default_name: str = DEFAULT_POLISHED.name, *, key: str = "theme_picker_global") -> Theme:
    if "ui_theme_name" not in st.session_state:
        st.session_state["ui_theme_name"] = default_name

    keys = list(THEMES.keys())
    current = st.session_state.get("ui_theme_name", default_name)
    if current not in keys:
        current = default_name

    name = st.selectbox(
        "Theme",
        options=keys,
        index=keys.index(current),
        key=key,
        help="Changes styling across the app (charts + UI).",
    )
    st.session_state["ui_theme_name"] = name
    return THEMES[name]

def apply_theme(theme: Theme) -> None:
    """Apply Matplotlib rcParams + small Streamlit CSS tweaks."""
    # --- Matplotlib global look ---
    mpl.rcParams.update({
        "figure.facecolor": theme.paper_bg,
        "axes.facecolor": theme.plot_bg,
        "savefig.facecolor": theme.paper_bg,

        "axes.edgecolor": theme.spine,
        "axes.facecolor": theme.plot_bg,
        "xtick.color": theme.text,
        "ytick.color": theme.text,

        "axes.titleweight": "bold",
        "axes.titlesize": theme.title_size,
        "axes.labelsize": theme.label_size,
        "font.size": theme.base_fontsize,
        "font.family": theme.font_family,

        "grid.color": theme.grid,
        "grid.linestyle": "-",
        "grid.alpha": 0.6,
        "lines.linewidth": theme.line_width,

        "legend.frameon": False,
        "legend.labelcolor": theme.primary if theme.name == "Bloomberg (Dark)" else theme.text,
    })

    # --- Streamlit CSS (light touch, not a full app theme override) ---
    st.markdown(
        f"""
        <style>
        
          /* Main app background */
          .stApp {{
            background: {theme.app_bg};
            color: {theme.text};
          }}

          /* Sidebar background */
          [data-testid="stSidebar"] {{
            background: {theme.sidebar_bg};
            border-right: 1px solid {theme.spine};
          }}

          /* Sidebar headings & section titles (robust) */
          [data-testid="stSidebar"] h1,
          [data-testid="stSidebar"] h2,
          [data-testid="stSidebar"] h3,
          [data-testid="stSidebar"] h4,
          [data-testid="stSidebar"] h5,
          [data-testid="stSidebar"] .stMarkdown strong {{
            color: {theme.primary} !important;
          }}

          /* Main container spacing */
          .block-container {{
            padding-top: 1.4rem;
            padding-bottom: 1rem;
          }}

          /* Expanders / cards / panels */
          [data-testid="stExpander"] {{
            border: 1px solid {theme.spine};
            background: {theme.panel_bg};
            border-radius: 12px;
          }}

          .theme-card {{
            background: {theme.panel_bg};
            border: 1px solid {theme.spine};
            border-radius: 12px;
            padding: 12px 16px;
          }}

          .theme-muted {{ color: {theme.neutral}; }}

          /* Inputs */
          .stTextInput input, .stNumberInput input {{
            background: {theme.panel_bg} !important;
            color: {theme.text} !important;
            border: 1px solid {theme.spine} !important;
          }}

          /* BaseWeb selectbox container (sidebar) */
          [data-testid="stSidebar"] [data-baseweb="select"] > div {{
            background: {theme.panel_bg} !important;
            color: {theme.text} !important;
            border: 1px solid {theme.spine} !important;
          }}

          /* Help icons */
          [data-testid="stSidebar"] svg {{
            fill: {theme.text} !important;
            color: {theme.text} !important;
            opacity: 0.9;
          }}

          /* Checkbox / radio text (BaseWeb + Streamlit wrappers) */
          [data-testid="stCheckbox"] label,
          [data-testid="stCheckbox"] span,
          [data-testid="stCheckbox"] p,
          [data-testid="stRadio"] label,
          [data-testid="stRadio"] span,
          [data-testid="stRadio"] p {{
            color: {theme.text} !important;
          }}

          /* Stronger: BaseWeb typography used inside widget labels */
          [data-baseweb="typography"] {{
            color: {theme.text} !important;
          }}
          
          /* Main page titles */
          h1, h2, h3 {{
              color: {theme.primary} !important;
          }}
           
          /* Section headers */
          h4, h5 {{
              color: {theme.primary} !important;
          }}
            
          /* Smaller captions / subtitles */
          .stCaption, small {{
              color: {theme.neutral} !important;
          }}
          
          /* Sidebar section titles (Settings, Weights, etc.) */
          [data-testid="stSidebar"] .stMarkdown h1,
          [data-testid="stSidebar"] .stMarkdown h2,
          [data-testid="stSidebar"] .stMarkdown h3,
          [data-testid="stSidebar"] .stMarkdown p strong {{
              color: {theme.primary} !important;
          }}
          
          /* Tabs: inactive = dark orange, active = dark orange (same) */
          [data-testid="stTabs"] button {{
            color: {theme.primary} !important;
            font-weight: 600 !important;
          }}
          
          [data-testid="stTabs"] button[aria-selected="true"] {{
            color: {theme.primary} !important;
            font-weight: 700 !important;
          }}
          
          /* Tab titles (Gold Acceleration / Intraday RSI Screener / Monte Carlo) */
          [data-testid="stTabs"] button {{
              color: {theme.primary} !important;
              font-weight: 600;
          }}
          
          /* Active tab underline */
          [data-testid="stTabs"] [data-baseweb="tab-highlight"] {{
              background-color: {theme.primary} !important;
          }}
          
          /* Global text defaults (main area): set base color WITHOUT forcing every element */
          html, body, [data-testid="stAppViewContainer"] {{
            color: {theme.text};
          }}
          
          /* Body text (keep bright) */
          .stMarkdown p, .stMarkdown li {{
          color: {theme.text} !important;
            opacity: 0.92;
          }}
          
          /* Captions/subtitles (muted) */
          .stCaption, [data-testid="stCaptionContainer"], small {{
            color: {theme.neutral} !important;
            opacity: 0.95;
          }}
          
          /* Tab underline (active bar) */
          [data-testid="stTabs"] [data-baseweb="tab-highlight"] {{
            background-color: {theme.primary} !important;
          }}
          /* Dataframe container (may be partial depending on Streamlit version) */
          [data-testid="stDataFrame"] {{
          background: {theme.panel_bg} !important;
          color: {theme.text} !important;
          border: 1px solid {theme.spine} !important;
          border-radius: 12px !important;
          overflow: hidden;
          }}

          [data-testid="stDataFrame"] thead tr th {{
          background: {theme.panel_bg} !important;
          color: {theme.text} !important;
          }}

          [data-testid="stDataFrame"] tbody tr td {{
          background: {theme.plot_bg} !important;
          color: {theme.text} !important;
          }}
          
          /* =========================
          Metric / KPI styling
          ========================= */
          
          /* KPI / metric numbers */
          [data-testid="stMetricValue"] {{
            font-size: 2.4rem;
            font-weight: 700;
            color: {theme.primary};
          }}
            
          /* KPI labels */
          [data-testid="stMetricLabel"] {{
            font-size: 1.85rem;
            color: {theme.text};
          }}

        </style>
        """,
        unsafe_allow_html=True,
    )