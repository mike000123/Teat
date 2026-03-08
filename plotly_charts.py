# plotly_charts.py
from __future__ import annotations
import pandas as pd
import numpy as np
import plotly.graph_objects as go


def plotly_table(df, theme, title=None, height=200):
    """
    Render a themed Plotly table (Bloomberg/dark compatible).
    """
    header_fill = getattr(theme, "panel_bg", theme.plot_bg)
    cell_fill = theme.plot_bg

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=list(df.columns),
                    fill_color=header_fill,
                    font=dict(color=theme.text, size=12),
                    align="left",
                    line_color=theme.spine,
                    height=32,
                ),
                cells=dict(
                    values=[df[c].tolist() for c in df.columns],
                    fill_color=cell_fill,
                    font=dict(color=theme.text, size=12),
                    align="left",
                    line_color=theme.spine,
                    height=28,
                ),
            )
        ]
    )

    fig.update_layout(
        title=dict(text=title, x=0.01, xanchor="left", font=dict(color=theme.text)) if title else None,
        paper_bgcolor=theme.paper_bg,
        plot_bgcolor=theme.plot_bg,
        margin=dict(l=10, r=10, t=36 if title else 10, b=10),
        height=height,  # ✅ now supported
    )
    return fig

def _base_layout(theme, *, title: str, height: int = 420) -> dict:

    TRACE_PALETTE_BLOOMBERG = [
        "#F5A623",
        "#00C1D4",
        "#FF6B6B",
        "#FFD166",
        "#7B6CF6",
        "#06D6A0",
        "#EF476F",
        "#118AB2",
    ]

    return dict(
        title=dict(text=title, x=0.01, xanchor="left", font=dict(color=theme.text)),
        height=height,
        margin=dict(l=18, r=18, t=48, b=18),

        paper_bgcolor=theme.paper_bg,
        plot_bgcolor=theme.plot_bg,

        colorway=theme.palette,

        font=dict(
            color=theme.text,
            family=getattr(theme, "font_family", "DejaVu Sans"),
            size=getattr(theme, "base_fontsize", 11),
        ),

        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showline=True,
            linecolor=theme.spine,
            ticks="outside",
        ),

        yaxis=dict(
            gridcolor=theme.grid,
            zeroline=False,
            showline=True,
            linecolor=theme.spine,
            ticks="outside",
        ),

        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(0,0,0,0)",
        ),

        hovermode="x unified",
    )

def line_overlay(
    df: pd.DataFrame,
    *,
    title: str,
    theme,
    y_title: str | None = None,
    rsi_bands: bool = False,
    show_extremes: bool = True,
    height: int = 420,
) -> go.Figure:
    """
    Multi-series line chart with optional RSI bands (30/70, optional 15/85).
    """
    df = df.copy()
    if df.empty:
        fig = go.Figure()
        fig.update_layout(_base_layout(theme, title=title, height=height))
        fig.add_annotation(text="No data", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
        return fig

    fig = go.Figure()

    for col in df.columns:
        s = df[col].dropna()
        if s.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=s.index,
                y=s.values,
                mode="lines",
                name=str(col),
                line=dict(width=2),
            )
        )

    layout = _base_layout(theme, title=title, height=height)
    if y_title:
        layout["yaxis"]["title"] = dict(text=y_title)

    # RSI bands
    if rsi_bands:
        # core bands
        bands = [(70, theme.danger, "dash"), (30, theme.success, "dash")]
        if show_extremes:
            bands += [(85, theme.danger, "dot"), (15, theme.success, "dot")]

        for y, col, dash in bands:
            fig.add_hline(y=y, line_width=1, line_dash=dash, line_color=col, opacity=0.65)

        layout["yaxis"]["range"] = [0, 100]

    fig.update_layout(**layout)

    # Subtle grid style improvement: y-grid only is already handled; x grid off
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True)

    return fig

def normalized_price_overlay(
    df: pd.DataFrame,
    *,
    title: str,
    theme,
    height: int = 420,
) -> go.Figure:
    """
    Normalizes each series to start at 100.
    """
    if df.empty:
        return line_overlay(df, title=title, theme=theme, height=height)

    norm = df.copy()
    for c in norm.columns:
        s = norm[c].dropna()
        if s.empty:
            continue
        base = float(s.iloc[0])
        if base != 0:
            norm[c] = (norm[c] / base) * 100.0

    return line_overlay(norm, title=title, theme=theme, y_title="Index (start=100)", height=height)

def candlesticks(
    ohlc: pd.DataFrame,
    *,
    title: str,
    theme,
    height: int = 420,
) -> go.Figure:
    """
    Plotly candlestick chart. Expects columns Open/High/Low/Close and datetime index.
    """
    if ohlc is None or ohlc.empty:
        fig = go.Figure()
        fig.update_layout(_base_layout(theme, title=title, height=height))
        fig.add_annotation(text="No intraday OHLC data", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
        return fig

    df = ohlc.copy()
    df = df.dropna()
    if df.empty:
        return candlesticks(df, title=title, theme=theme, height=height)

    fig = go.Figure()

    inc = getattr(theme, "success", "#22C55E")
    dec = getattr(theme, "danger", "#EF4444")

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],

            # ✅ Thin wicks + clearer body colors
            increasing=dict(
                line=dict(color=inc, width=1),
                fillcolor=inc,
            ),
            decreasing=dict(
                line=dict(color=dec, width=1),
                fillcolor=dec,
            ),
        )
    )

    # ✅ Make bodies slightly transparent so overlaps look nicer (optional)
    fig.update_traces(opacity=0.95)

    fig.update_layout(**_base_layout(theme, title=title, height=height))

    # Subtle y-grid on candles (TV-like)
    fig.update_yaxes(showgrid=True, gridcolor=theme.grid)

    # Slightly more padding so candles don't touch edges
    fig.update_xaxes(rangeslider_visible=False)

    # A “terminal-like” look: no x-grid, light y-grid
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True)

    return fig