# intraday_screener.py
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import streamlit as st

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
        # wick
        ax.plot([xi, xi], [l, h], linewidth=1)
        # body
        y0 = min(o, c)
        height = max(abs(c - o), 1e-9)
        rect = plt.Rectangle((xi - width / 2, y0), width, height)
        ax.add_patch(rect)

    ax.set_title(title)
    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.grid(True, alpha=0.3)

@st.cache_data(ttl=65, show_spinner=False)
def yf_intraday_ohlc(
    ticker: str,
    interval: str = "1m",
    period: str = "1d",
    refresh_token: int = 0,
) -> pd.DataFrame:
    """
    Best-effort OHLC for candlesticks.
    Cached very briefly; refresh_token can bust cache on schedule.
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
        )
        if df is None or df.empty:
            return pd.DataFrame()

        # Standardize columns
        needed = ["Open", "High", "Low", "Close"]
        if not all(c in df.columns for c in needed):
            return pd.DataFrame()

        df = df[needed].dropna()
        if df.empty:
            return pd.DataFrame()

        # tz-naive
        try:
            if getattr(df.index, "tz", None) is not None:
                df.index = df.index.tz_convert(None)
        except Exception:
            pass

        return df.sort_index()

    except Exception:
        return pd.DataFrame()

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
    st.markdown("## Intraday RSI Screener (Top 10) — Multi-timeframe")

    # You control the universe elsewhere (sidebar). Keep tab clean.
    tickers = [t.strip().upper() for t in (tickers or []) if t and isinstance(t, str)]
    tickers = list(dict.fromkeys(tickers))[:10]

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


    # ----------------------------
    # Build the TOP-10 multi-timeframe table
    # ----------------------------
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
            "Above MA200?": bool(above_ma200) if (pd.notna(ma200) and pd.notna(ref_px)) else "N/A (need 200d)",
            "Dist to MA200 (%)": dist_ma200_pct,
            "Last Time": last_time,
            "RS vs QQQ (3M, pp)": rs_vs_qqq_3m,
            "MA50 slope (20d, %)": ma50_slope_20d,
        })

    screen = pd.DataFrame(rows)

    st.markdown("### Live candlesticks (optional)")

    show_candles = st.checkbox("Show live candlesticks for selected ticker", value=False)

    pick = st.selectbox("Ticker to display", screen["Ticker"].tolist(), index=0)

    # Use 1m bars for live feel; fallback to 5m if empty
    if show_candles and pick:
        ohlc_1m = yf_intraday_ohlc(pick, interval="1m", period="1d", refresh_token=refresh_token)
        ohlc = ohlc_1m

        if ohlc.empty:
            ohlc_5m = yf_intraday_ohlc(pick, interval="5m", period="5d", refresh_token=refresh_token)
            ohlc = ohlc_5m

        # Window: last 2 hours for 1m, last 2 days for 5m
        if not ohlc.empty:
            end = ohlc.index.max()
            if (ohlc.index.freqstr == "T") or (len(ohlc) > 300):
                start = end - pd.Timedelta(hours=2)
            else:
                start = end - pd.Timedelta(days=2)
            ohlc = ohlc.loc[ohlc.index >= start]

        fig, ax = plt.subplots(figsize=(12.0, 3.6))
        plot_candles(ax, ohlc, title=f"{pick} Candles (auto-refresh driven)")
        fig.autofmt_xdate(rotation=0)
        st.pyplot(fig, use_container_width=True)


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
        Returns (score, setup_label).
        score roughly in [-5, +5]. Positive = buy-leaning, negative = sell-leaning.
        """
        d = row.get("Daily RSI (Swing/Trend)", np.nan)
        i5 = row.get("5m RSI (Tactical)", np.nan)
        rs3m = row.get("RS vs QQQ (3M, pp)", np.nan)
        ma50s = row.get("MA50 slope (20d, %)", np.nan)
        above200 = row.get("Above MA200?", None)
        dist200 = row.get("Dist to MA200 (%)", np.nan)

        pe = _safe_pe(row)

        if pd.isna(d) or pd.isna(i5):
            return (0, "NO DATA")

        score = 0

        # --- Trend filter (MA200) ---
        # If MA200 is available, use it as regime context.
        if above200 is True:
            score += 1
        elif above200 is False:
            score -= 1

        # --- New: Relative Strength vs QQQ (3M) ---
        # Positive means outperforming QQQ (leader). Negative = laggard.
        if pd.notna(rs3m):
            score += 1 if rs3m > 0 else -1

        # --- New: MA50 slope (20d) ---
        # Positive slope means trend is rising; negative slope = deterioration.
        if pd.notna(ma50s):
            score += 1 if ma50s > 0 else -1

        # Optional: reward “near MA200” pullbacks (better entries than buying far above)
        # Only if above MA200 and within +0%..+6% distance.
        if above200 is True and pd.notna(dist200) and 0 <= dist200 <= 6:
            score += 1

        setup = "NEUTRAL"

        # --- Trend context (Daily RSI) ---
        if d >= daily_uptrend_thr:
            score += 1
        if d >= daily_uptrend_thr + 5:
            score += 1  # stronger uptrend
        if d <= daily_downtrend_thr:
            score -= 1
        if d <= daily_downtrend_thr - 5:
            score -= 1  # stronger downtrend

        # --- Timing pulse (5m RSI) ---
        # Oversold helps BUY timing; overbought helps SELL timing
        if i5 <= thr_oversold:
            score += 2
        elif i5 <= (thr_oversold + 10):
            score += 1

        if i5 >= thr_overbought:
            score -= 2
        elif i5 >= (thr_overbought - 10):
            score -= 1

        # Extreme bands (optional)
        if show_extremes:
            if i5 <= 15:
                score += 1
            if i5 >= 85:
                score -= 1

        # --- Setup label (pattern recognition) ---
        # Tightened with MA200 trend context:
        # - Only call it "pullback in uptrend" if ABOVE MA200
        # - Only call it "bounce in downtrend" if BELOW MA200
        if (above200 is True) and (d >= daily_uptrend_thr) and (i5 <= thr_oversold):
            setup = "PULLBACK_UPTREND"
        elif (above200 is False) and (d <= daily_downtrend_thr) and (i5 >= thr_overbought):
            setup = "BOUNCE_DOWNTREND"
        elif (d >= daily_uptrend_thr + 5) and (40 <= i5 <= 60):
            setup = "TREND_CONTINUATION"
        else:
            setup = "MIXED"

        # --- Valuation guardrails (soft) ---
        # Don’t kill the signal if PE is missing, but penalize obvious mania.
        if pd.notna(pe):
            if pe >= pe_hot_thr:
                score -= 2
            elif pe <= pe_ok_thr:
                score += 1

        return (int(score), setup)

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

    only_actionable = st.checkbox("Show only actionable (Score ≥ 3: Buy or ≤ -3: Sell)", value=False)
    if only_actionable:
        screen = screen[(screen["Score"] >= 3) | (screen["Score"] <= -3)]

    st.caption("This table uses: Daily (2y), 5m (5d). Thresholds apply per timeframe.")
    st.dataframe(screen, use_container_width=True, hide_index=True)

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
