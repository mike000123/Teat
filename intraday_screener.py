# intraday_screener.py
from __future__ import annotations
from plotly_charts import line_overlay, normalized_price_overlay, candlesticks
import numpy as np
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from datetime import datetime
from streamlit_autorefresh import st_autorefresh
from ui_themes import THEMES, DEFAULT_POLISHED, apply_theme, style_signal_table
import streamlit as st

def style_signal_table(df, theme):
    """
    Apply color styling to signal columns.
    Bullish → green
    Bearish → red
    Neutral → gray
    """

    def color_signal(val):
        if val is None:
            return ""

        v = str(val).lower()

        if "bull" in v or "buy" in v or "long" in v:
            return f"color: {theme.success}; font-weight:700"
        elif "bear" in v or "sell" in v or "short" in v:
            return f"color: {theme.danger}; font-weight:700"
        elif "neutral" in v or "hold" in v:
            return f"color: {theme.neutral}; font-weight:700"

        return ""

    # apply to every column that looks like a signal column
    signal_cols = [c for c in df.columns if "signal" in c.lower()]

    if not signal_cols:
        return df.style

    return df.style.applymap(color_signal, subset=signal_cols)

def plot_candles(ax, ohlc: pd.DataFrame, title: str = ""):
    """
    Minimal candlestick drawing using matplotlib primitives.
    No external libs required.
    """
    if ohlc is None or ohlc.empty:
        ax.set_title(title)
        ax.text(0.5, 0.5, "No intraday OHLC data", ha="center", va="center")
        ax.axis("off")
        return

    x = mdates.date2num(ohlc.index.to_pydatetime())
    width = (x[-1] - x[0]) / max(len(x), 50) * 0.8  # adaptive

    for xi, (o, h, l, c) in zip(x, ohlc[["Open", "High", "Low", "Close"]].to_numpy()):
        up = (c >= o)
        # read current theme (fallbacks if not applied yet)
        theme_name = st.session_state.get("ui_theme_name", "Polished (Light)")
        # If you prefer: pass theme explicitly into plot_candles instead of reading session_state
        col_up = "#16A34A"
        col_down = "#DC2626"
        if "Bloomberg" in theme_name:
            col_up, col_down = "#34D399", "#F87171"
        elif "TradingView" in theme_name:
            col_up, col_down = "#22C55E", "#EF4444"

        col = col_up if up else col_down

        # wick
        ax.plot([xi, xi], [l, h], linewidth=1, color=col)

        # body
        y0 = min(o, c)
        height = max(abs(c - o), 1e-9)
        rect = plt.Rectangle(
            (xi - width / 2, y0),
            width,
            height,
            facecolor=col,
            edgecolor=col,
            linewidth=1,
            alpha=0.9
        )
        ax.add_patch(rect)

    ax.set_title(title)
    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=ohlc.index.tz))
    ax.grid(True, alpha=0.3)

@st.cache_data(ttl=15, show_spinner=False)
def yf_intraday_ohlc(ticker: str, interval: str, period: str, refresh_token: int = 0) -> pd.DataFrame:
    """
    Download intraday OHLC data using yfinance.
    Returns a clean DataFrame with columns: Open, High, Low, Close.
    """

    try:
        import yfinance as yf
    except Exception:
        return pd.DataFrame()

    try:
        df = yf.download(
            tickers=ticker,
            interval=interval,
            period=period,
            progress=False,
            auto_adjust=False,
            prepost=False,
            threads=False,
        )
    except Exception:
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    # Handle MultiIndex columns sometimes returned by yfinance
    if isinstance(df.columns, pd.MultiIndex):
        try:
            df = df.loc[:, ["Open", "High", "Low", "Close"]]
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        except Exception:
            return pd.DataFrame()

    needed = ["Open", "High", "Low", "Close"]
    if not all(c in df.columns for c in needed):
        return pd.DataFrame()

    df = df[needed].copy()

    # Ensure numeric OHLC
    for c in needed:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna()
    if df.empty:
        return pd.DataFrame()

    # Ensure sorted unique index
    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()

    return df

import math

def _yf_period_from_days(days: int) -> str:
    """
    Convert 'days' to a yfinance-compatible period string.
    Examples: 5 -> "5d", 60 -> "60d", 90 -> "3mo", 400 -> "2y"
    """
    days = int(max(1, days))
    if days <= 60:
        return f"{days}d"
    if days <= 365:
        mo = int(math.ceil(days / 30.0))
        return f"{mo}mo"
    yrs = int(math.ceil(days / 365.0))
    return f"{yrs}y"


def _choose_intraday_interval(days: int) -> str:
    """
    Choose a readable intraday interval based on requested lookback days.
    (Avoids 1m over long spans which often returns empty.)
    """
    days = int(max(1, days))
    if days <= 2:
        return "1m"
    if days <= 10:
        return "5m"
    if days <= 30:
        return "15m"
    return "60m"

@st.cache_data(ttl=65, show_spinner=False)
def yf_multi_close_fixed_period(
    tickers: list[str],
    interval: str,
    period: str,
    refresh_token: int = 0,  # <-- add this
) -> pd.DataFrame:
    """
    Multi-ticker close via yfinance with explicit (interval, period).
    Returns DataFrame index=datetime, columns=tickers (close).
    """
    try:
        import yfinance as yf
    except Exception:
        return pd.DataFrame()

    tickers = [t.strip().upper() for t in tickers if t and isinstance(t, str)]
    tickers = list(dict.fromkeys(tickers))
    if not tickers:
        return pd.DataFrame()

    try:
        df = yf.download(
            tickers=tickers,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=True,
            prepost=False,
            group_by="column",
            threads=True,
        )
        if df is None or df.empty:
            return pd.DataFrame()

        # Extract Close
        if isinstance(df.columns, pd.MultiIndex):
            if "Close" in df.columns.get_level_values(0):
                close = df["Close"].copy()
            elif "Adj Close" in df.columns.get_level_values(0):
                close = df["Adj Close"].copy()
            else:
                return pd.DataFrame()
        else:
            # single ticker edge-case
            if "Close" in df.columns:
                close = df[["Close"]].copy()
                close.columns = [tickers[0]]
            else:
                return pd.DataFrame()

        close = close.dropna(how="all")
        if close.empty:
            return pd.DataFrame()

        # tz-naive index
        try:
            if getattr(close.index, "tz", None) is not None:
                close.index = close.index.tz_convert(None)
        except Exception:
            pass

        return close.sort_index()

    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=86400, show_spinner=False)
def load_nasdaq100_tickers() -> list[str]:
    """
    Best-effort NASDAQ-100 constituents via Wikipedia.
    Falls back to a small list if blocked/unavailable.
    """
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/Nasdaq-100")
        for t in tables:
            cols = [str(c).lower() for c in t.columns]
            if any("ticker" in c for c in cols):
                ticker_col = t.columns[[("ticker" in str(c).lower()) for c in t.columns]][0]
                tickers = (
                    t[ticker_col]
                    .astype(str)
                    .str.replace(r"\.", "-", regex=True)
                    .str.strip()
                    .tolist()
                )
                tickers = [x for x in tickers if x and x.lower() != "nan"]
                return list(dict.fromkeys([x.upper() for x in tickers]))
    except Exception:
        pass

    return ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "AVGO", "COST", "AMD"]

@st.cache_data(ttl=86400, show_spinner=False)
def yf_pe_snapshot(tickers: list[str], refresh_token: int = 0) -> pd.DataFrame:
    """
    Best-effort valuation snapshot via yfinance (cached daily).
    Returns DataFrame with columns: Ticker, trailingPE, forwardPE.
    """
    try:
        import yfinance as yf
    except Exception:
        return pd.DataFrame(columns=["Ticker", "P/E (TTM)", "P/E (Fwd)"])

    out = []
    for t in tickers:
        try:
            info = yf.Ticker(t).info or {}
            pe_ttm = info.get("trailingPE", np.nan)
            pe_fwd = info.get("forwardPE", np.nan)
            out.append({"Ticker": t, "P/E (TTM)": pe_ttm, "P/E (Fwd)": pe_fwd})
        except Exception:
            out.append({"Ticker": t, "P/E (TTM)": np.nan, "P/E (Fwd)": np.nan})

    return pd.DataFrame(out)

def render_intraday_rsi_screener_tab(
    *,
    rsi_func,
    zscore_func,
    tickers: list[str],
    refresh_token: int = 0,
):
    # -----------------------------
    # Defaults (session_state)
    # -----------------------------
    theme = THEMES.get(st.session_state.get("ui_theme_name", DEFAULT_POLISHED.name), DEFAULT_POLISHED)
    apply_theme(theme)

    if "use_weighted_scoring" not in st.session_state:
        st.session_state["use_weighted_scoring"] = True
    if "use_dynamic_weights" not in st.session_state:
        st.session_state["use_dynamic_weights"] = True
    if "vol_window" not in st.session_state:
        st.session_state["vol_window"] = 20

    # -----------------------------
    # Theme selector
    # -----------------------------

    st.markdown("## Intraday RSI Screener — Multi-timeframe")

    # You control the universe elsewhere (sidebar). Keep tab clean.
    tickers = [t.strip().upper() for t in (tickers or []) if t and isinstance(t, str)]
    MAX_TICKERS = 20
    tickers = list(dict.fromkeys(tickers))[:MAX_TICKERS]

    if not tickers:
        st.info("No tickers provided from the sidebar selection.")
        return

    # Fixed timeframes (no dropdowns here to avoid confusion)
    DAILY_INTERVAL, DAILY_PERIOD = "1d", "2y"
    I5_INTERVAL, I5_PERIOD = "5m", "5d"

    with st.spinner("Fetching data & computing RSI (Daily / 5m) + P/E…"):
        # Include QQQ for relative strength benchmarking
        tickers_rs = tickers + ["QQQ"] if "QQQ" not in tickers else tickers
        close_d_all = yf_multi_close_fixed_period(tickers_rs, interval="1d", period=DAILY_PERIOD,
                                                  refresh_token=refresh_token)

        # Split: universe vs benchmark
        close_d = close_d_all.drop(columns=["QQQ"], errors="ignore")
        qqq_d = close_d_all["QQQ"].dropna() if (not close_d_all.empty and "QQQ" in close_d_all.columns) else pd.Series(
            dtype=float)

        close_5 = yf_multi_close_fixed_period(tickers, interval="5m", period="5d", refresh_token=refresh_token)
        pe_df = yf_pe_snapshot(tickers, refresh_token=refresh_token)

    st.markdown("### RSI thresholds")

    tcol1, tcol2, tcol3 = st.columns([1.1, 1.1, 1.6])
    with tcol1:
        thr_overbought = st.number_input("Overbought (RSI ≥)", min_value=50, max_value=95, value=70, step=1)
    with tcol2:
        thr_oversold = st.number_input("Oversold (RSI ≤)", min_value=5, max_value=50, value=30, step=1)
    with tcol3:
        show_extremes = st.checkbox("Also show extreme bands (85/15)", value=True)

    st.markdown("### Scoring rules")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        daily_uptrend_thr = st.number_input("Daily RSI uptrend (bullish momentum) ≥", min_value=40, max_value=70, value=55, step=1)
    with c2:
        daily_downtrend_thr = st.number_input("Daily RSI downtrend (bearish momentum) ≤", min_value=30, max_value=60, value=45, step=1)
    with c3:
        pe_hot_thr = st.number_input("P/E hot (Doesn't Worth the Money) ≥", min_value=10, max_value=200, value=80, step=5)
    with c4:
        pe_ok_thr = st.number_input("P/E OK ≤", min_value=5, max_value=150, value=60, step=5)

    st.markdown("## Intraday RSI Screener — Multi-timeframe")

    # ----------------------------
    # Build the TOP-10 multi-timeframe table
    # ----------------------------
    def realized_vol_annualized(series: pd.Series, window: int = 20) -> float:
        """
        Annualized realized vol (%) from daily closes using log returns.
        """
        if series is None or series.dropna().empty:
            return np.nan
        s = series.dropna().astype(float)
        if len(s) <= window + 2:
            return np.nan
        rets = np.log(s).diff().dropna()
        rv = rets.rolling(window).std(ddof=0).iloc[-1]
        if pd.isna(rv):
            return np.nan
        return float(rv * np.sqrt(252.0) * 100.0)

    def vol_regime(series: pd.Series, window: int = 20) -> str:
        """
        HIGH if current vol is meaningfully above its own recent baseline; else LOW.
        Baseline = median vol over last ~1y (252 days) computed from rolling vols.
        """
        if series is None or series.dropna().empty:
            return "N/A"
        s = series.dropna().astype(float)
        if len(s) < 252:
            # not enough history to build a robust baseline
            cur = realized_vol_annualized(s, window=window)
            return "HIGH" if (pd.notna(cur) and cur >= 35.0) else "LOW"  # fallback heuristic

        rets = np.log(s).diff().dropna()
        roll = rets.rolling(window).std(ddof=0) * np.sqrt(252.0) * 100.0
        roll = roll.dropna()
        if roll.empty:
            return "N/A"
        cur = float(roll.iloc[-1])
        base = float(roll.tail(252).median())

        # HIGH if > +25% above baseline
        return "HIGH" if (pd.notna(cur) and pd.notna(base) and base > 0 and cur >= 1.25 * base) else "LOW"

    def last_rsi(series: pd.Series) -> float:
        if series is None or series.empty or len(series) < 20:
            return np.nan
        r = rsi_func(series.astype(float), period=14).dropna()
        return float(r.iloc[-1]) if not r.empty else np.nan

    def last_sma(series: pd.Series, window: int) -> float:
        if series is None or series.dropna().empty or len(series.dropna()) < window:
            return np.nan
        return float(series.dropna().rolling(window).mean().iloc[-1])

    def trailing_return(series: pd.Series, periods: int) -> float:
        """Return over last N bars: (last / prevN) - 1."""
        if series is None or series.dropna().empty:
            return np.nan
        s = series.dropna()
        if len(s) <= periods:
            return np.nan
        try:
            return float(s.iloc[-1] / s.iloc[-(periods + 1)] - 1.0)
        except Exception:
            return np.nan

    def ma_slope(series: pd.Series, ma_window: int = 50, slope_lookback: int = 20) -> float:
        """
        MA slope = MA(t) - MA(t-lookback).
        Returns % slope relative to MA(t-lookback) for comparability.
        """
        if series is None or series.dropna().empty:
            return np.nan
        s = series.dropna()
        if len(s) < (ma_window + slope_lookback + 5):
            return np.nan
        ma = s.rolling(ma_window).mean().dropna()
        if len(ma) <= slope_lookback:
            return np.nan
        a = float(ma.iloc[-1])
        b = float(ma.iloc[-(slope_lookback + 1)])
        if b == 0:
            return np.nan
        return (a / b - 1.0) * 100.0  # % change over slope_lookback

    # --- Volatility window (use session_state, safe default) ---
    vw = int(st.session_state.get("vol_window", 20))

    rows = []
    for t in tickers:
        s_d = close_d[t].dropna() if (not close_d.empty and t in close_d.columns) else pd.Series(dtype=float)
        s_5 = close_5[t].dropna() if (not close_5.empty and t in close_5.columns) else pd.Series(dtype=float)

        # Price priority: 5m > daily
        last_px = np.nan
        last_time = None
        if not s_5.empty:
            last_px = float(s_5.iloc[-1]);
            last_time = s_5.index[-1]
        elif not s_d.empty:
            last_px = float(s_d.iloc[-1]);
            last_time = s_d.index[-1]

        # Trend context from DAILY closes (stable regime filter)
        ma50 = last_sma(s_d, 50)
        ma200 = last_sma(s_d, 200)

        # --- New: Relative Strength vs QQQ (3M) ---
        ret_3m = trailing_return(s_d, 63)  # ~3 months
        qqq_3m = trailing_return(qqq_d, 63)
        rs_vs_qqq_3m = (ret_3m - qqq_3m) * 100.0 if (pd.notna(ret_3m) and pd.notna(qqq_3m)) else np.nan  # pct points

        # --- New: MA50 slope (20 trading days), percent ---
        ma50_slope_20d = ma_slope(s_d, ma_window=50, slope_lookback=20)

        ref_px = float(s_d.iloc[-1]) if not s_d.empty else last_px
        above_ma200 = (pd.notna(ma200) and pd.notna(ref_px) and (ref_px > ma200))
        dist_ma200_pct = (float(ref_px) / float(ma200) - 1.0) * 100.0 if (
                    pd.notna(ma200) and pd.notna(ref_px)) else np.nan

        rows.append({
            "Ticker": t,
            "Last Price": last_px,
            "Daily RSI (Swing/Trend)": last_rsi(s_d),
            "5m RSI (Tactical)": last_rsi(s_5),
            "MA50 (Daily)": ma50,
            "MA200 (Daily)": ma200,
            "Above MA200?": (bool(above_ma200) if (pd.notna(ma200) and pd.notna(ref_px)) else None),
            "Dist to MA200 (%)": dist_ma200_pct,
            "Last Time": last_time,
            "RS vs QQQ (3M, pp)": rs_vs_qqq_3m,
            "MA50 slope (20d, %)": ma50_slope_20d,
            "Realized Vol (ann %, daily)": realized_vol_annualized(s_d, window=vw),
            "Vol Regime": vol_regime(s_d, window=vw),
        })

    screen = pd.DataFrame(rows)

    # ----------------------------
    # Multi-select (checkboxes) + overlay view
    # ----------------------------
    st.markdown("### Chart selection")

    MAX_SELECTED = 8

    left, right = st.columns([1, 3], gap="large")

    with left:
        overlay = st.checkbox("Overlay view (compare tickers on one chart)", value=False)
        metric = st.selectbox("Metric", ["RSI (Daily)", "RSI (5m)", "Price (normalized)"], index=0)

        st.caption("Select tickers (multiple).")
        if "rsi_screener_selected" not in st.session_state:
            # default: first 3 tickers
            st.session_state["rsi_screener_selected"] = screen["Ticker"].tolist()[:3]

        selected = []
        for t in screen["Ticker"].tolist():
            checked = st.checkbox(t, value=(t in st.session_state["rsi_screener_selected"]), key=f"cb_{t}")
            if checked:
                selected.append(t)

        if len(selected) > MAX_SELECTED:
            st.warning(f"Select up to {MAX_SELECTED} tickers.")
            selected = selected[:MAX_SELECTED]

        st.session_state["rsi_screener_selected"] = selected

    with right:
        if not selected:
            st.info("Select one or more tickers on the left.")
        else:
            # Build overlay dataframe
            series_map = {}
            for t in selected:
                s_d = close_d[t].dropna() if (not close_d.empty and t in close_d.columns) else pd.Series(dtype=float)
                s_5 = close_5[t].dropna() if (not close_5.empty and t in close_5.columns) else pd.Series(dtype=float)

                if metric == "RSI (Daily)":
                    r = rsi_func(s_d.astype(float), period=14).dropna()
                    series_map[t] = r.rename(t)
                elif metric == "RSI (5m)":
                    r = rsi_func(s_5.astype(float), period=14).dropna()
                    series_map[t] = r.rename(t)
                else:
                    # Normalize price so overlay makes sense
                    px = s_d.astype(float).dropna()
                    if not px.empty:
                        px = px / float(px.iloc[0]) * 100.0
                    series_map[t] = px.rename(t)

            overlay_df = pd.concat(series_map.values(), axis=1).dropna(how="all")

            is_rsi = metric in ("RSI (Daily)", "RSI (5m)")

            if overlay:
                if metric == "Price (normalized)":
                    fig = normalized_price_overlay(
                        overlay_df,
                        title=f"{metric} — Overlay",
                        theme=theme,
                        height=420,
                    )
                else:
                    fig = line_overlay(
                        overlay_df,
                        title=f"{metric} — Overlay",
                        theme=theme,
                        y_title=metric,
                        rsi_bands=is_rsi,
                        show_extremes=show_extremes,
                        height=420,
                    )
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})
            else:
                for t in selected:
                    st.markdown(f"**{t}**")
                    sub = overlay_df[[t]].dropna()
                    if metric == "Price (normalized)":
                        fig = normalized_price_overlay(
                            sub,
                            title=f"{t} — {metric}",
                            theme=theme,
                            height=360,
                        )
                    else:
                        fig = line_overlay(
                            sub,
                            title=f"{t} — {metric}",
                            theme=theme,
                            y_title=metric,
                            rsi_bands=is_rsi,
                            show_extremes=show_extremes,
                            height=360,
                        )
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})

    # ----------------------------
    # OPTIONAL: Live candlesticks for ONE ticker (driven by sidebar intraday lookback)
    # ----------------------------
    st.markdown("### Live candlesticks (optional)")
    show_candles = st.checkbox("Show live candlesticks (single ticker)", value=False)

    if show_candles:
        pick = selected[0] if selected else screen["Ticker"].iloc[0]

        # Use the GLOBAL sidebar setting (falls back safely if not set)
        lookback_days = int(st.session_state.get("intraday_lookback_days", 5))

        # UX toggle: show full requested window (default) vs just last trading day (cleaner)
        show_last_day_only = st.checkbox("Show only last trading day (cleaner)", value=False)

        # Auto-refresh the candlestick area only
        st_autorefresh(interval=60 * 1000, key="candles_autorefresh")

        # Use GLOBAL sidebar settings
        lookback_days = int(st.session_state.get("intraday_lookback_days", 5))
        user_interval = st.session_state.get("intraday_interval", "5m")  # ✅ comes from Patch A

        # Convert lookback days to a yfinance period
        period = _yf_period_from_days(lookback_days)

        # Start with the user's choice, then fall back to known-good combos (v1-like)
        tries = [
            (user_interval, period),
            ("1m", "1d"),
            ("5m", "5d"),
            ("15m", _yf_period_from_days(min(lookback_days, 60))),
            ("60m", _yf_period_from_days(min(lookback_days, 730))),
        ]

        ohlc = pd.DataFrame()
        for itv, per in tries:
            ohlc = yf_intraday_ohlc(pick, interval=itv, period=per, refresh_token=refresh_token)
            if ohlc is not None and not ohlc.empty:
                interval, period = itv, per  # keep what worked
                break

        ohlc = pd.DataFrame()
        for itv, per in tries:
            ohlc = yf_intraday_ohlc(pick, interval=itv, period=per, refresh_token=refresh_token)
            if ohlc is not None and not ohlc.empty:
                interval, period = itv, per  # keep what worked
                break

        st.caption(f"Fetch used: interval={interval}, period={period}, rows={len(ohlc)}")

        # Build display title times (ET + Athens) based on latest candle in returned data
        if not ohlc.empty:
            ts = ohlc.index.max()
            if getattr(ts, "tzinfo", None) is None:
                ts = ts.tz_localize("UTC")
            ts_et = ts.tz_convert("America/New_York")
            ts_ath = ts.tz_convert("Europe/Athens")
            us_last = ts_et.strftime("%H:%M")
            gr_last = ts_ath.strftime("%H:%M")
        else:
            us_last = "—"
            gr_last = "—"

        title = f"{pick} Candles ({interval}, {period}) | US ET: {us_last} | Athens: {gr_last}"

        # Convert to ET for plotting (Plotly expects datetime index)
        ohlc_plot = ohlc.copy()
        if not ohlc_plot.empty:
            if getattr(ohlc_plot.index, "tz", None) is None:
                ohlc_plot.index = ohlc_plot.index.tz_localize("UTC")
            ohlc_plot.index = ohlc_plot.index.tz_convert("America/New_York")

            # If user prefers: keep only the last trading day (prevents “flat empty” feeling off-hours)
            if show_last_day_only:
                last_day = ohlc_plot.index.max().date()
                day_df = ohlc_plot[ohlc_plot.index.date == last_day]
                if not day_df.empty:
                    ohlc_plot = day_df

        st.caption(f"OHLC rows={len(ohlc_plot)} | cols={list(ohlc_plot.columns)} | dtypes={ohlc_plot.dtypes.to_dict()}")

        fig = candlesticks(
            ohlc_plot,
            title=title,
            theme=theme,
            height=360,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"displayModeBar": True, "scrollZoom": True},
        )

        # App refresh time (local runtime)
        app_now = datetime.now().strftime("%H:%M:%S")

        # Latest candle time (display in US ET)
        if not ohlc.empty:
            ts = ohlc.index.max()
            if getattr(ts, "tzinfo", None) is None:
                ts = ts.tz_localize("UTC")
            ts_et = ts.tz_convert("America/New_York")
            last_bar_et = ts_et.strftime("%H:%M:%S")
        else:
            last_bar_et = "—"

        st.markdown(
            f"<div style='display:flex; justify-content:space-between; font-size:0.85rem; color:gray;'>"
            f"<span>App refreshed (local): {app_now}</span>"
            f"<span>Latest candle (US ET): {last_bar_et}</span>"
            f"</div>",
            unsafe_allow_html=True
        )

        st.markdown(
            f"<div style='text-align: right; font-size: 0.85rem; color: gray;'>"
            f"Last updated (US ET): {last_bar_et}"
            f"</div>",
            unsafe_allow_html=True
        )


    # Merge valuation snapshot (P/E)
    if isinstance(pe_df, pd.DataFrame) and not pe_df.empty:
        screen = screen.merge(pe_df, on="Ticker", how="left")
    else:
        screen["P/E (TTM)"] = np.nan
        screen["P/E (Fwd)"] = np.nan

    if screen.empty:
        st.info("No RSI values computed (insufficient data).")
        return

    # RSI state per timeframe using your chosen thresholds
    def rsi_state(x):
        if pd.isna(x):
            return ""
        if show_extremes:
            if x >= 85:
                return "EXT_OVERBOUGHT"
            if x <= 15:
                return "EXT_OVERSOLD"
        if x >= thr_overbought:
            return "OVERBOUGHT"
        if x <= thr_oversold:
            return "OVERSOLD"
        return "NEUTRAL"

    screen["Daily State"] = screen["Daily RSI (Swing/Trend)"].apply(rsi_state)
    screen["5m State"] = screen["5m RSI (Tactical)"].apply(rsi_state)

    # Sort by Daily RSI by default (you can change to 15m or 5m if you prefer)
    screen = screen.sort_values("Daily RSI (Swing/Trend)", ascending=False)

    def _safe_pe(row) -> float:
        """Prefer TTM P/E; fallback to forward P/E. Ignore non-meaningful (<=0)."""
        pe_ttm = row.get("P/E (TTM)", np.nan)
        pe_fwd = row.get("P/E (Fwd)", np.nan)

        pe = pe_ttm
        if pd.isna(pe) and pd.notna(pe_fwd):
            pe = pe_fwd

        try:
            if pd.notna(pe) and float(pe) <= 0:
                return np.nan
        except Exception:
            return np.nan

        return float(pe) if pd.notna(pe) else np.nan

    def _score_row(row) -> tuple[int, str]:

        """
        Returns (score_int, setup_label).

        - If weighted scoring is OFF: behaves like your original point system.
        - If weighted scoring is ON: builds Trend/Timing/Valuation components, combines via weights,
          then rounds into an integer Score so your existing verdict mapping stays unchanged.
        """

        # --- read scoring engine settings from session_state ---
        use_weighted_scoring = bool(st.session_state.get("use_weighted_scoring", True))
        use_dynamic_weights = bool(st.session_state.get("use_dynamic_weights", True))

        d = row.get("Daily RSI (Swing/Trend)", np.nan)
        i5 = row.get("5m RSI (Tactical)", np.nan)
        rs3m = row.get("RS vs QQQ (3M, pp)", np.nan)
        ma50s = row.get("MA50 slope (20d, %)", np.nan)
        above200 = row.get("Above MA200?", None)
        dist200 = row.get("Dist to MA200 (%)", np.nan)
        pe = _safe_pe(row)
        vreg = row.get("Vol Regime", "N/A")

        if pd.isna(d):
            return (0, "NO DATA")

        i5_missing = pd.isna(i5)

        # -----------------------
        # Component scores
        # -----------------------
        trend = 0.0
        timing = 0.0
        value = 0.0

        # --- Trend filter (MA200) ---
        if above200 is True:
            trend += 1
        elif above200 is False:
            trend -= 1

        # --- Relative Strength vs QQQ (3M) ---
        if pd.notna(rs3m):
            trend += 1 if rs3m > 0 else -1

        # --- MA50 slope (20d) ---
        if pd.notna(ma50s):
            trend += 1 if ma50s > 0 else -1

        # --- Near MA200 pullback bonus ---
        if above200 is True and pd.notna(dist200) and 0 <= dist200 <= 6:
            trend += 1

        # --- Daily RSI momentum context ---
        if d >= daily_uptrend_thr:
            trend += 1
        if d >= daily_uptrend_thr + 5:
            trend += 1
        if d <= daily_downtrend_thr:
            trend -= 1
        if d <= daily_downtrend_thr - 5:
            trend -= 1

        # --- Timing pulse (5m RSI) ---
        if not i5_missing:
            if i5 <= thr_oversold:
                timing += 2
            elif i5 <= (thr_oversold + 10):
                timing += 1

            if i5 >= thr_overbought:
                timing -= 2
            elif i5 >= (thr_overbought - 10):
                timing -= 1

            if show_extremes:
                if i5 <= 15:
                    timing += 1
                if i5 >= 85:
                    timing -= 1

        # --- Valuation guardrails (soft) ---
        if pd.notna(pe):
            if pe >= pe_hot_thr:
                value -= 2
            elif pe <= pe_ok_thr:
                value += 1

        # -----------------------
        # Setup label (unchanged logic)
        # -----------------------
        setup = "MIXED"
        if (above200 is True) and (d >= daily_uptrend_thr) and (not i5_missing) and (i5 <= thr_oversold):
            setup = "PULLBACK_UPTREND"
        elif (above200 is False) and (d <= daily_downtrend_thr) and (not i5_missing) and (i5 >= thr_overbought):
            setup = "BOUNCE_DOWNTREND"
        elif (d >= daily_uptrend_thr + 5) and (not i5_missing) and (40 <= i5 <= 60):
            setup = "TREND_CONTINUATION"

        # -----------------------
        # Combine into final score
        # -----------------------
        if not use_weighted_scoring:
            # Old behavior (equal-ish points): just sum components
            score_float = trend + timing + value
        else:
            # Base weights
            # Trend is the slow filter, Timing is entry/exit pulse, Value is sanity check.
            w_trend, w_timing, w_value = 0.50, 0.35, 0.15

            # Dynamic regime shift
            if use_dynamic_weights and vreg in ("HIGH", "LOW"):
                if vreg == "HIGH":
                    w_trend, w_timing, w_value = 0.45, 0.45, 0.10
                else:  # LOW
                    w_trend, w_timing, w_value = 0.55, 0.25, 0.20

            score_float = (w_trend * trend) + (w_timing * timing) + (w_value * value)

            # Scale to feel similar to the old score magnitude (optional but helpful):
            # Old sums often ranged ~[-6..+6]; weighted tends to compress.
            score_float *= 1.6

        # IMPORTANT: keep verdict mapping intact -> integer score
        score_int = int(np.round(score_float))
        score_int = int(np.clip(score_int, -6, 6))  # safety clamp

        return (score_int, setup)

    def _verdict_from(score: int, setup: str) -> str:
        """Human label derived from numeric score + setup type."""
        if setup == "NO DATA":
            return "NO DATA"

        # Strong signals
        if score >= 3:
            if setup == "PULLBACK_UPTREND":
                return "BUY (pullback in uptrend)"
            if setup == "TREND_CONTINUATION":
                return "BUY (trend strong)"
            return "BUY (screened)"

        if score <= -3:
            if setup == "BOUNCE_DOWNTREND":
                return "SELL (bounce in downtrend)"
            return "SELL (screened)"

        # Medium signals
        if 1 <= score <= 2:
            return "WATCH (buy bias)"
        if -2 <= score <= -1:
            return "WATCH (sell bias)"

        return "NEUTRAL (no edge)"

    # Apply scoring
    tmp = screen.apply(_score_row, axis=1, result_type="expand")
    screen["Score"] = tmp[0].astype(int)
    screen["Setup"] = tmp[1].astype(str)
    screen["Verdict"] = [_verdict_from(int(s), str(u)) for s, u in zip(screen["Score"], screen["Setup"])]

    preferred_cols = [
        "Ticker", "Score", "Verdict", "Setup",
        "Vol Regime", "Realized Vol (ann %, daily)",
        "Last Price",
        "Above MA200?", "Dist to MA200 (%)",
        "Daily RSI (Swing/Trend)", "Daily State",
        "RS vs QQQ (3M, pp)",
        "MA50 slope (20d, %)",
        "5m RSI (Tactical)", "5m State",
        "MA50 (Daily)", "MA200 (Daily)",
        "P/E (TTM)", "P/E (Fwd)",
        "Last Time",
    ]
    cols = [c for c in preferred_cols if c in screen.columns]
    screen = screen[cols]

    # Sort by Score descending (best BUY at top, best SELL at bottom)
    screen = screen.sort_values(["Score", "Daily RSI (Swing/Trend)"], ascending=[False, False])

    # Quick overview
    svals = screen["Score"].dropna()
    cA, cB, cC, cD = st.columns(4)
    cA.metric("Avg Score", f"{svals.mean():.2f}" if len(svals) else "—")
    cB.metric("Top Score", f"{svals.max():.0f}" if len(svals) else "—")
    cC.metric("Bottom Score", f"{svals.min():.0f}" if len(svals) else "—")
    cD.metric("BUY / SELL", f"{(screen['Score'] >= 3).sum()} / {(screen['Score'] <= -3).sum()}")

    st.markdown("### Scoring engine")

    sA, sB, sC = st.columns([1.2, 1.4, 1.8])

    with sA:
        st.checkbox(
            "Use weighted scoring (instead of equal-weight points)",
            key="use_weighted_scoring",
            help="Keeps the same verdict mapping, but converts signals into Trend/Timing/Valuation components with weights."
        )

    with sB:
        st.checkbox(
            "Dynamic weights by volatility regime",
            key="use_dynamic_weights",
            disabled=(not st.session_state["use_weighted_scoring"]),
            help="If enabled, tickers in high-vol regime get more weight on Timing (5m RSI) and less on Valuation."
        )

    with sC:
        st.number_input(
            "Volatility window (days)",
            min_value=10, max_value=60, step=5,
            key="vol_window",
            disabled=(not st.session_state["use_weighted_scoring"]),
            help="Used to classify HIGH vs LOW volatility using daily realized volatility."
        )

    only_actionable = st.checkbox("Show only actionable (Score ≥ 3: Buy or ≤ -3: Sell)", value=False)
    if only_actionable:
        screen = screen[(screen["Score"] >= 3) | (screen["Score"] <= -3)]

    st.caption("This table uses: Daily (2y), 5m (5d). Thresholds apply per timeframe.")
    styled = style_signal_table(screen, theme)
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # ----------------------------
    # Basic overview stats (per timeframe)
    # ----------------------------
    c1, c2 = st.columns(2)

    dvals = screen["Daily RSI (Swing/Trend)"].dropna()
    v5 = screen["5m RSI (Tactical)"].dropna()

    with c1:
        st.markdown("### Daily RSI overview")
        if len(dvals):
            st.write(f"Mean: {dvals.mean():.1f} | Median: {dvals.median():.1f}")
            st.write(
                f"% ≥{thr_overbought}: {(dvals.ge(thr_overbought).mean() * 100):.0f}% | % ≤{thr_oversold}: {(dvals.le(thr_oversold).mean() * 100):.0f}%")
            if show_extremes:
                st.write(f"% ≥85: {(dvals.ge(85).mean() * 100):.0f}% | % ≤15: {(dvals.le(15).mean() * 100):.0f}%")
        else:
            st.write("No data.")

    with c2:
        st.markdown("### 5m RSI overview")
        if len(v5):
            st.write(f"Mean: {v5.mean():.1f} | Median: {v5.median():.1f}")
            st.write(
                f"% ≥{thr_overbought}: {(v5.ge(thr_overbought).mean() * 100):.0f}% | % ≤{thr_oversold}: {(v5.le(thr_oversold).mean() * 100):.0f}%")
            if show_extremes:
                st.write(f"% ≥85: {(v5.ge(85).mean() * 100):.0f}% | % ≤15: {(v5.le(15).mean() * 100):.0f}%")
        else:
            st.write("No data.")
