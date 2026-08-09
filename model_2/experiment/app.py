"""
app.py

SPY signal decision experiment -- within-subject, 2 conditions
(explained vs unexplained), counterbalanced across two scenario sets.

Run: streamlit run app.py
"""

from __future__ import annotations

import random
import time

import streamlit as st

from config import (
    COLORS, GROUPS, DECISION_OPTIONS, CONFIDENCE_SCALE,
    CONSENT_FORM_URL
)
from scenarios import build_trial_sequence, load_scenarios, ScenarioLoadError
from ui_components import (
    render_price_chart, render_signal_panel, render_shap_chart,
    render_progress, inject_base_styles,
)
import logging_utils as log

st.set_page_config(page_title="Market Signal Task", layout="centered")
inject_base_styles()


# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------
def init_state():
    defaults = {
        "phase": "consent",       # consent -> id_entry -> task -> thankyou
        "participant_id": None,
        "group": None,
        "trial_sequence": None,
        "trial_index": 0,         # 0-indexed into trial_sequence
        "trial_start_time": None,
        "session_logged": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()


# ---------------------------------------------------------------------------
# Screen: Consent / intro
# ---------------------------------------------------------------------------
def render_consent_screen():
    st.title("Market Signal Task")
    st.write(
        "Thank you for taking part in this study. You'll be shown a series of short "
        "market scenarios. For each one, you'll see recent price movement and a "
        "computer-generated signal, then make your own decision about what you'd do."
    )
    st.write(
        "There is no right or wrong answer -- we're interested in how people make "
        "decisions in this kind of task. The task takes about 10-15 minutes."
    )
    st.write(
        f"Before starting, please review and complete the consent form: "
        f"[Consent form]({CONSENT_FORM_URL})"
    )
    agree = st.checkbox("I have read the consent form and agree to participate.")
    if st.button("Continue", disabled=not agree, type="primary"):
        st.session_state["phase"] = "id_entry"
        st.rerun()


# ---------------------------------------------------------------------------
# Screen: Participant ID entry + duplicate guard + group assignment
# ---------------------------------------------------------------------------
def render_id_entry_screen():
    st.title("Before you begin")
    participant_id = st.text_input("Please enter your participant ID (provided to you separately):")

    if st.button("Start task", type="primary", disabled=not participant_id.strip()):
        pid = participant_id.strip()
        try:
            already_done = log.has_participated(pid)
        except Exception as e:
            st.error(
                "We couldn't verify your participant ID against our records right now "
                "(logging service unavailable). Please try again in a moment."
            )
            st.caption(f"Details: {e}")
            return

        if already_done:
            st.error(
                "This participant ID has already completed this task. "
                "Each participant may only complete the task once."
            )
            return

        group = random.choice(GROUPS)
        try:
            sequence = build_trial_sequence(group)
        except ScenarioLoadError as e:
            st.error(f"Could not load task scenarios: {e}")
            return

        try:
            log.log_session_start(pid, group)
        except Exception as e:
            st.error("Could not start your session (logging service unavailable). Please try again.")
            st.caption(f"Details: {e}")
            return

        st.session_state["participant_id"] = pid
        st.session_state["group"] = group
        st.session_state["trial_sequence"] = sequence
        st.session_state["trial_index"] = 0
        st.session_state["trial_start_time"] = time.time()
        st.session_state["phase"] = "task"
        st.rerun()


# ---------------------------------------------------------------------------
# Screen: Trial
# ---------------------------------------------------------------------------
def render_trial_screen():
    sequence = st.session_state["trial_sequence"]
    idx = st.session_state["trial_index"]
    total = len(sequence)
    trial = sequence[idx]

    if st.session_state["trial_start_time"] is None:
        st.session_state["trial_start_time"] = time.time()

    render_progress(idx + 1, total)

    col_chart, col_decision = st.columns([1.2, 1])

    with col_chart:
        render_price_chart(trial["price_window"])
        render_signal_panel(trial["signal"], trial["confidence"])
        if trial["condition"] == "explained":
            render_shap_chart(trial["top_features"])

    with col_decision:
        st.markdown(
            f"""<div style="font-size:0.85rem;color:{COLORS['text_muted']};
                text-transform:uppercase;letter-spacing:0.04em;margin-bottom:10px;">
                Your decision
            </div>""",
            unsafe_allow_html=True,
        )
        decision = st.radio(
            "What would you do?",
            options=DECISION_OPTIONS,
            index=None,
            key=f"decision_{idx}",
            label_visibility="collapsed",
        )
        st.write("")
        confidence_rating = st.slider(
            "How confident are you in this decision?",
            min_value=1, max_value=5, value=3,
            key=f"confidence_{idx}",
        )
        st.write("")
        submit_disabled = decision is None
        if st.button("Submit and continue", type="primary", disabled=submit_disabled, key=f"submit_{idx}"):
            response_time = time.time() - st.session_state["trial_start_time"]
            try:
                log.log_trial(
                    participant_id=st.session_state["participant_id"],
                    group=st.session_state["group"],
                    scenario_id=trial["scenario_id"],
                    trial_index=idx,
                    condition=trial["condition"],
                    signal_shown=trial["signal"],
                    confidence_shown=trial["confidence"],
                    decision=decision,
                    confidence_rating=confidence_rating,
                    response_time_seconds=round(response_time, 3),
                )
            except Exception as e:
                st.error(
                    "Your response couldn't be saved (logging service unavailable). "
                    "Please try submitting again."
                )
                st.caption(f"Details: {e}")
                return

            # Advance only forward -- no back button, no way to revisit a trial.
            if idx + 1 >= total:
                st.session_state["phase"] = "thankyou"
            else:
                st.session_state["trial_index"] = idx + 1
                st.session_state["trial_start_time"] = time.time()
            st.rerun()


# ---------------------------------------------------------------------------
# Screen: Thank you
# ---------------------------------------------------------------------------
def render_thankyou_screen():
    if not st.session_state["session_logged"]:
        try:
            log.log_session_complete(st.session_state["participant_id"])
        except Exception:
            pass  # best-effort; trial-level logs already capture the data
        st.session_state["session_logged"] = True

    st.title("Thank you")
    st.write("You've completed all trials. We appreciate your time.")


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
phase = st.session_state["phase"]
if phase == "consent":
    render_consent_screen()
elif phase == "id_entry":
    render_id_entry_screen()
elif phase == "task":
    render_trial_screen()
elif phase == "thankyou":
    render_thankyou_screen()
