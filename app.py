# app.py
# DSA3900 Research Prototype — Streamlit Web App
# Explainable Machine Learning for Algorithmic Trading
# Author: Aime Muganga (670232)
# Run: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from datetime import datetime

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="XAI Trading Study",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# LOAD DATA AND ARTIFACTS
# ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("test_results.csv", parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df

@st.cache_resource
def load_artifacts():
    with open("best_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("top5_features.pkl", "rb") as f:
        top5 = pickle.load(f)
    return model, top5

df         = load_data()
model, top5 = load_artifacts()

SHAP_COLS  = [c for c in df.columns if c.startswith("SHAP_")]
FEAT_NAMES = [c.replace("SHAP_", "") for c in SHAP_COLS]

RESPONSES_FILE = "responses.csv"

# ─────────────────────────────────────────────────────────────
# SIDEBAR — NAVIGATION & PARTICIPANT ID
# ─────────────────────────────────────────────────────────────
st.sidebar.title("📊 XAI Trading Study")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate to:",
    ["🏠 Home", "📉 Trading Scenario", "🔍 With Explanation", "📝 Decision Form", "📊 Results Dashboard"]
)

st.sidebar.markdown("---")
participant_id = st.sidebar.text_input("Participant ID", placeholder="e.g. P001")
if not participant_id:
    st.sidebar.warning("Please enter your Participant ID before submitting decisions.")

# ─────────────────────────────────────────────────────────────
# SESSION STATE — track current scenario index
# ─────────────────────────────────────────────────────────────
if "scenario_idx" not in st.session_state:
    st.session_state.scenario_idx = int(np.random.randint(0, len(df)))

if "last_decision" not in st.session_state:
    st.session_state.last_decision = None

# ─────────────────────────────────────────────────────────────
# HELPER: get scenario row and 30-day price window
# ─────────────────────────────────────────────────────────────
def get_scenario(idx):
    row = df.iloc[idx]
    # Get 30 prior rows of Close prices for the chart
    start = max(0, idx - 30)
    window = df.iloc[start:idx + 1][["Date", "Close"]]
    return row, window

def recommendation_label(pred):
    if pred == 1:
        return "BUY", "#1a9850"
    else:
        return "SELL", "#d73027"

def price_chart(window, title="Last 30 Days — SPY Close Price"):
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(window["Date"], window["Close"], color="steelblue", linewidth=1.8)
    ax.fill_between(window["Date"], window["Close"], alpha=0.08, color="steelblue")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Date")
    ax.set_ylabel("Close Price (USD)")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return fig

def shap_bar_chart(row, top_n=5):
    shap_vals  = [row[f"SHAP_{f}"] for f in FEAT_NAMES]
    shap_series = pd.Series(shap_vals, index=FEAT_NAMES)
    top = shap_series.abs().nlargest(top_n).index
    top_vals = shap_series[top].sort_values()

    colors = ["#d73027" if v < 0 else "#1a9850" for v in top_vals]
    fig, ax = plt.subplots(figsize=(8, 3))
    top_vals.plot(kind="barh", ax=ax, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Top Feature Contributions (SHAP)", fontsize=11)
    ax.set_xlabel("SHAP value  (positive = pushes toward BUY)")
    fig.tight_layout()
    return fig

def auto_summary(row, top_n=3):
    """Generate a plain-English explanation sentence from SHAP values."""
    shap_vals   = {f: row[f"SHAP_{f}"] for f in FEAT_NAMES}
    sorted_feats = sorted(shap_vals.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
    label       = "BUY" if row["Prediction"] == 1 else "SELL"

    parts = []
    for feat, val in sorted_feats:
        direction = "high" if val > 0 else "low"
        parts.append(f"{feat} is {direction} ({val:+.4f})")

    summary = f"The model recommends **{label}** mainly because: {', and '.join(parts)}."
    return summary

# ─────────────────────────────────────────────────────────────
# PAGE 1: HOME
# ─────────────────────────────────────────────────────────────
if page == "🏠 Home":
    st.title("📈 Explainable AI for Algorithmic Trading")
    st.subheader("A Human Decision-Making Study | USIU-Africa DSA3900")
    st.markdown("---")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        ### Welcome, Participant

        This study investigates whether **AI explanations** help people make better trading decisions.
        You will be shown historical market scenarios for the **S&P 500 ETF (SPY)** and asked to
        decide whether to **Buy**, **Hold**, or **Sell**.

        Some scenarios will show only the model's recommendation. Others will also show
        **an explanation** of why the model made that recommendation — highlighting which
        market indicators influenced the decision most.

        ---

        ### Instructions

        1. Enter your **Participant ID** in the sidebar (provided by the researcher).
        2. Navigate to **Trading Scenario** to view a scenario without an explanation.
        3. Navigate to **With Explanation** to view the same scenario with an AI explanation.
        4. Go to **Decision Form** to record your decision, confidence level, and any comments.
        5. Click **Next Scenario** at any time to load a new random scenario.

        ---

        ### Important Notes

        - All scenarios use **real historical data** (SPY, 2010–2024).
        - No real money is involved. This is entirely simulated.
        - There are no right or wrong answers — we are studying your decision process.
        - Your responses are confidential and used only for academic research.
        """)

    with col2:
        st.markdown("### Study Details")
        st.info("""
        **Dataset:** SPY Daily OHLCV  
        **Period:** 2010–2024  
        **Model:** Best of LR / RF / GB  
        **XAI Method:** SHAP  
        **Task:** Binary (BUY / SELL)
        """)
        st.markdown("### Contact")
        st.write("Researcher: Aime Muganga")
        st.write("ID: 670232")
        st.write("Course: DSA3900")

# ─────────────────────────────────────────────────────────────
# PAGE 2: TRADING SCENARIO (no explanation)
# ─────────────────────────────────────────────────────────────
elif page == "📉 Trading Scenario":
    st.title("📉 Trading Scenario")
    st.caption("No AI explanation shown in this view.")
    st.markdown("---")

    if st.button("🔀 Next Scenario"):
        st.session_state.scenario_idx = int(np.random.randint(0, len(df)))

    row, window = get_scenario(st.session_state.scenario_idx)
    label, color = recommendation_label(row["Prediction"])

    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader(f"Scenario Date: {pd.Timestamp(row['Date']).date()}")
        st.pyplot(price_chart(window))

    with col2:
        st.subheader("Model Recommendation")
        st.markdown(
            f"<div style='background-color:{color};padding:30px;border-radius:12px;"
            f"text-align:center;'>"
            f"<h1 style='color:white;font-size:48px;margin:0'>{label}</h1>"
            f"<p style='color:white;margin:0'>Confidence: {row['Proba_BUY']:.1%}</p>"
            f"</div>",
            unsafe_allow_html=True
        )
        st.markdown("---")
        st.metric("Close Price", f"${row['Close']:.2f}")
        st.metric("RSI (14)", f"{row.get('RSI_14', 'N/A'):.1f}" if 'RSI_14' in row else "N/A")
        st.metric("Return (1d)", f"{row.get('Return_1d', 0)*100:.2f}%")

    st.markdown("---")
    st.info("👉 Go to **Decision Form** in the sidebar to record your decision.")

# ─────────────────────────────────────────────────────────────
# PAGE 3: WITH EXPLANATION
# ─────────────────────────────────────────────────────────────
elif page == "🔍 With Explanation":
    st.title("🔍 Trading Scenario — With AI Explanation")
    st.caption("SHAP explanations shown in this view.")
    st.markdown("---")

    if st.button("🔀 Next Scenario"):
        st.session_state.scenario_idx = int(np.random.randint(0, len(df)))

    row, window = get_scenario(st.session_state.scenario_idx)
    label, color = recommendation_label(row["Prediction"])

    # Price chart + recommendation
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader(f"Scenario Date: {pd.Timestamp(row['Date']).date()}")
        st.pyplot(price_chart(window))
    with col2:
        st.subheader("Model Recommendation")
        st.markdown(
            f"<div style='background-color:{color};padding:30px;border-radius:12px;"
            f"text-align:center;'>"
            f"<h1 style='color:white;font-size:48px;margin:0'>{label}</h1>"
            f"<p style='color:white;margin:0'>Confidence: {row['Proba_BUY']:.1%}</p>"
            f"</div>",
            unsafe_allow_html=True
        )
        st.markdown("---")
        st.metric("Close Price", f"${row['Close']:.2f}")
        st.metric("RSI (14)", f"{row.get('RSI_14', 0):.1f}")
        st.metric("Return (1d)", f"{row.get('Return_1d', 0)*100:.2f}%")

    # Explanation section
    st.markdown("---")
    st.subheader("🧠 AI Explanation (SHAP)")

    exp_col1, exp_col2 = st.columns([2, 1])
    with exp_col1:
        st.pyplot(shap_bar_chart(row, top_n=5))
    with exp_col2:
        st.markdown("**Plain-English Summary**")
        st.markdown(auto_summary(row, top_n=3))
        st.markdown("---")
        st.markdown("**Top 5 Important Features (Global)**")
        for i, feat in enumerate(top5, 1):
            st.write(f"{i}. {feat}")

    st.markdown("---")
    st.info("👉 Go to **Decision Form** in the sidebar to record your decision.")

# ─────────────────────────────────────────────────────────────
# PAGE 4: DECISION FORM
# ─────────────────────────────────────────────────────────────
elif page == "📝 Decision Form":
    st.title("📝 Record Your Decision")
    st.markdown("---")

    row, _ = get_scenario(st.session_state.scenario_idx)
    label, color = recommendation_label(row["Prediction"])

    st.markdown(f"**Current Scenario:** {pd.Timestamp(row['Date']).date()} — "
                f"Model says: <span style='color:{color};font-weight:bold'>{label}</span>",
                unsafe_allow_html=True)
    st.markdown("---")

    # Condition toggle
    condition = st.radio(
        "Which condition are you responding to?",
        ["No Explanation (Control)", "With Explanation (Treatment)"],
        horizontal=True
    )

    st.markdown("#### Your Decision")
    decision = st.radio("What would you do?", ["Buy", "Hold", "Sell"], horizontal=True)

    st.markdown("#### Confidence Level")
    confidence = st.slider("How confident are you in this decision? (1 = not at all, 5 = very confident)", 1, 5, 3)

    st.markdown("#### Optional Comments")
    comments = st.text_area("Any reasoning or observations?", placeholder="e.g. RSI looked oversold, trend seemed upward...")

    st.markdown("---")

    if st.button("✅ Submit Decision", type="primary"):
        if not participant_id:
            st.error("Please enter your Participant ID in the sidebar before submitting.")
        else:
            record = {
                "Timestamp":      datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Participant_ID": participant_id,
                "Scenario_Date":  pd.Timestamp(row["Date"]).date(),
                "Scenario_Index": st.session_state.scenario_idx,
                "Condition":      condition,
                "Model_Rec":      label,
                "Decision":       decision,
                "Confidence":     confidence,
                "Comments":       comments,
                "Actual":         int(row["Actual"]),
                "Correct":        int(
                    (decision == "Buy"  and row["Actual"] == 1) or
                    (decision == "Sell" and row["Actual"] == 0)
                )
            }

            record_df = pd.DataFrame([record])
            if os.path.exists(RESPONSES_FILE):
                record_df.to_csv(RESPONSES_FILE, mode="a", header=False, index=False)
            else:
                record_df.to_csv(RESPONSES_FILE, index=False)

            st.success(f"✅ Decision recorded! Thank you, {participant_id}.")
            st.session_state.last_decision = record

    if st.session_state.last_decision:
        with st.expander("Last submitted response"):
            st.json(st.session_state.last_decision)

# ─────────────────────────────────────────────────────────────
# PAGE 5: RESULTS DASHBOARD (researcher view)
# ─────────────────────────────────────────────────────────────
elif page == "📊 Results Dashboard":
    st.title("📊 Results Dashboard")
    st.caption("Researcher view — aggregated participant responses.")
    st.markdown("---")

    if not os.path.exists(RESPONSES_FILE):
        st.warning("No responses collected yet. `responses.csv` not found.")
    else:
        resp = pd.read_csv(RESPONSES_FILE)

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Responses",    len(resp))
        col2.metric("Unique Participants", resp["Participant_ID"].nunique())
        col3.metric("Avg Confidence",      f"{resp['Confidence'].mean():.2f} / 5")
        col4.metric("Overall Accuracy",    f"{resp['Correct'].mean()*100:.1f}%")

        st.markdown("---")

        chart_col1, chart_col2 = st.columns(2)

        # Decision distribution bar chart
        with chart_col1:
            st.subheader("Decision Distribution")
            counts = resp["Decision"].value_counts()
            fig, ax = plt.subplots(figsize=(5, 3))
            counts.plot(kind="bar", ax=ax, color=["#1a9850", "#f0a500", "#d73027"])
            ax.set_title("Frequency of Each Decision")
            ax.set_xlabel("Decision")
            ax.set_ylabel("Count")
            ax.tick_params(axis="x", rotation=0)
            fig.tight_layout()
            st.pyplot(fig)

        # Average confidence per decision type
        with chart_col2:
            st.subheader("Avg Confidence by Decision")
            avg_conf = resp.groupby("Decision")["Confidence"].mean()
            fig2, ax2 = plt.subplots(figsize=(5, 3))
            avg_conf.plot(kind="bar", ax=ax2, color="steelblue")
            ax2.set_title("Average Confidence per Decision Type")
            ax2.set_xlabel("Decision")
            ax2.set_ylabel("Avg Confidence (1–5)")
            ax2.set_ylim(0, 5)
            ax2.tick_params(axis="x", rotation=0)
            fig2.tight_layout()
            st.pyplot(fig2)

        st.markdown("---")

        # Condition comparison
        st.subheader("Accuracy by Condition")
        cond_acc = resp.groupby("Condition")["Correct"].mean().reset_index()
        cond_acc.columns = ["Condition", "Accuracy"]
        cond_acc["Accuracy"] = (cond_acc["Accuracy"] * 100).round(1)
        st.dataframe(cond_acc, use_container_width=True)

        # Confidence by condition
        st.subheader("Confidence by Condition")
        cond_conf = resp.groupby("Condition")["Confidence"].mean().reset_index()
        cond_conf.columns = ["Condition", "Avg Confidence"]
        cond_conf["Avg Confidence"] = cond_conf["Avg Confidence"].round(2)
        st.dataframe(cond_conf, use_container_width=True)

        st.markdown("---")
        st.subheader("Raw Response Data")
        st.dataframe(resp, use_container_width=True)

        # Download button
        csv = resp.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download responses.csv", csv, "responses.csv", "text/csv")