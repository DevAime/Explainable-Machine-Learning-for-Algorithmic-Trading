"""
ui_components.py

Reusable rendering pieces for the trial screen: the price chart (no future
data visible -- the chart only ever shows the scenario's precomputed
price_window, which already ends at the signal timestamp) and the SHAP
contribution bar chart (plain-language feature names, muted colors).
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import COLORS, FEATURE_DISPLAY_NAMES


def render_price_chart(price_window: list[dict], title: str = "Price (5-min bars)") -> None:
    df = pd.DataFrame(price_window)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    fig = go.Figure(
        data=[
            go.Candlestick(
                x=df["timestamp"],
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
                increasing_line_color=COLORS["up_muted"],
                decreasing_line_color=COLORS["down_muted"],
                increasing_fillcolor=COLORS["up_muted"],
                decreasing_fillcolor=COLORS["down_muted"],
                name="",
            )
        ]
    )
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Price ($)",
        plot_bgcolor=COLORS["panel_bg"],
        paper_bgcolor=COLORS["panel_bg"],
        font=dict(color=COLORS["text_primary"]),
        xaxis_rangeslider_visible=False,
        margin=dict(l=40, r=20, t=40, b=40),
        height=380,
    )
    fig.update_xaxes(gridcolor=COLORS["border"])
    fig.update_yaxes(gridcolor=COLORS["border"])
    st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})


def render_signal_panel(signal: str, confidence: float) -> None:
    label_display = {"up": "Upward", "down": "Downward", "flat": "Flat / Sideways"}
    color = {"up": COLORS["up_muted"], "down": COLORS["down_muted"], "flat": COLORS["flat_muted"]}
    st.markdown(
        f"""
        <div style="
            background-color:{COLORS['panel_bg']};
            border:1px solid {COLORS['border']};
            border-radius:8px;
            padding:16px 20px;
            margin-bottom:12px;
        ">
            <div style="font-size:0.85rem;color:{COLORS['text_muted']};
                        text-transform:uppercase;letter-spacing:0.04em;margin-bottom:6px;">
                Model Signal
            </div>
            <div style="font-size:1.4rem;font-weight:600;color:{color.get(signal, COLORS['text_primary'])};">
                {label_display.get(signal, signal)}
            </div>
            <div style="font-size:0.9rem;color:{COLORS['text_muted']};margin-top:4px;">
                Confidence: {confidence * 100:.1f}%
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_shap_chart(top_features: list[dict]) -> None:
    """top_features: list of {feature, raw_column, shap_value, direction}
    Bar chart uses feature (plain-language) names, never raw column names."""
    df = pd.DataFrame(top_features)
    df = df.sort_values("shap_value")

    colors = [COLORS["shap_positive"] if v > 0 else COLORS["shap_negative"] for v in df["shap_value"]]

    fig = go.Figure(
        go.Bar(
            x=df["shap_value"],
            y=df["feature"],
            orientation="h",
            marker_color=colors,
        )
    )
    fig.update_layout(
        title="What influenced this signal",
        xaxis_title="Contribution (SHAP value)",
        plot_bgcolor=COLORS["panel_bg"],
        paper_bgcolor=COLORS["panel_bg"],
        font=dict(color=COLORS["text_primary"]),
        margin=dict(l=10, r=20, t=40, b=40),
        height=320,
    )
    fig.update_xaxes(gridcolor=COLORS["border"], zerolinecolor=COLORS["border"])
    fig.update_yaxes(gridcolor=COLORS["panel_bg"])
    st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})


def render_progress(trial_index: int, total_trials: int) -> None:
    st.markdown(
        f"""
        <div style="color:{COLORS['text_muted']};font-size:0.9rem;margin-bottom:8px;">
            Trial {trial_index} of {total_trials}
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(trial_index / total_trials)


def inject_base_styles() -> None:
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: {COLORS['background']};
        }}
        div.block-container {{
            max-width: 980px;
            padding-top: 2rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
