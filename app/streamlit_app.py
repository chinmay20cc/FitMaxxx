from __future__ import annotations

import io
import json
import os
import sys
import zipfile
from datetime import datetime
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
SRC_DIR = PROJECT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(APP_DIR))

import ui  # noqa: E402
import coach  # noqa: E402
import fitmaxxx_store  # noqa: E402
import wearable_simulator  # noqa: E402
from pipeline import (  # noqa: E402
    FS,
    detect_acc_units,
    prepare_signals,
    windows_from_signals,
    smooth_probabilities,
    window_centers,
    session_metrics,
)
from raw_features import extract  # noqa: E402
from robust_features import extract_robust  # noqa: E402


# ---------------------------------------------
# Configuration and secrets
# ---------------------------------------------
st.set_page_config(
    page_title="FitMaxxx | Your fitness companion",
    page_icon="♡",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(ui.CSS, unsafe_allow_html=True)


def get_secret(name: str, default: str = "") -> str:
    value = os.environ.get(name, "")
    if value:
        return value
    try:
        return str(st.secrets.get(name, default))
    except Exception:
        return default


PUBLIC_DEMO = get_secret("FITMAXXX_PUBLIC_DEMO", "0").strip().lower() in {"1", "true", "yes"}
DB_PATH = PROJECT_DIR / "data" / "fitmaxxx.sqlite3"
REPORTS = PROJECT_DIR / "reports"
SAMPLE = APP_DIR / "sample_data"

MODEL_CONFIG = {
    "Everyday placement": {
        "path": PROJECT_DIR / "models" / "robust_model.joblib",
        "extract": extract_robust,
        "metrics": REPORTS / "robust_model_metrics.json",
        "detail": "42 features · orientation-robust model",
    },
    "Waist placement": {
        "path": PROJECT_DIR / "models" / "raw_model.joblib",
        "extract": extract,
        "metrics": REPORTS / "raw_model_metrics.json",
        "detail": "44 features · raw sensor baseline",
    },
}

NAV_ITEMS = ["Overview", "Wearable studio", "Activity lab", "AI coach", "Progress", "Model lab"]


@st.cache_resource
def load_model(model_path: str):
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"{path.name} is missing from models/.")
    return joblib.load(path)


@st.cache_data
def load_json(path: str):
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def parse_sensor_zip(uploaded_file):
    with zipfile.ZipFile(io.BytesIO(uploaded_file.getvalue())) as archive:
        files = {
            Path(name).name.lower(): name
            for name in archive.namelist()
            if name.lower().endswith(".csv") and "__macosx" not in name.lower()
        }
        required = ("totalacceleration.csv", "gyroscope.csv")
        missing = [name for name in required if name not in files]
        if missing:
            raise ValueError("The ZIP must contain TotalAcceleration.csv and Gyroscope.csv.")

        def read(name):
            return pd.read_csv(archive.open(files[name]))

        return read(required[0]), read(required[1])


def load_recording(source: str):
    if source == "Built-in session":
        acc_path = SAMPLE / "demo_acc.csv"
        gyro_path = SAMPLE / "demo_gyro.csv"
        if not acc_path.exists() or not gyro_path.exists():
            raise FileNotFoundError("The sample recording is missing from app/sample_data/.")
        return pd.read_csv(acc_path), pd.read_csv(gyro_path), "Sample session"

    if source == "Sensor Logger ZIP":
        uploaded = st.file_uploader("Sensor Logger ZIP", type=["zip"], key="activity_zip")
        if uploaded is None:
            raise ValueError("Choose a Sensor Logger ZIP first.")
        acc, gyro = parse_sensor_zip(uploaded)
        return acc, gyro, uploaded.name

    left, right = st.columns(2)
    with left:
        acc_file = st.file_uploader("TotalAcceleration.csv", type=["csv"], key="activity_acc")
    with right:
        gyro_file = st.file_uploader("Gyroscope.csv", type=["csv"], key="activity_gyro")
    if acc_file is None or gyro_file is None:
        raise ValueError("Choose both sensor CSV files first.")
    return pd.read_csv(acc_file), pd.read_csv(gyro_file), "Uploaded phone session"


def run_activity_analysis(acc: pd.DataFrame, gyro: pd.DataFrame, model_label: str, smoothing: int, weight: float):
    config = MODEL_CONFIG[model_label]
    bundle = load_model(str(config["path"]))
    unit_label, divisor = detect_acc_units(acc)
    signals, start = prepare_signals(acc, gyro, divisor)
    windows = windows_from_signals(signals)
    features = config["extract"](windows)
    probabilities = bundle["model"].predict_proba(features)
    smoothed, class_ids, confidence = smooth_probabilities(probabilities, smoothing)
    labels = bundle["label_names"]
    activities = [labels[int(idx)] for idx in class_ids]
    centers = window_centers(len(activities), start)
    metrics = session_metrics(activities, weight)
    summary = pd.Series(activities).value_counts().rename_axis("activity").reset_index(name="windows")
    top_activity = summary.iloc[0]["activity"] if not summary.empty else "—"
    return {
        "signals": signals,
        "start": start,
        "unit_label": unit_label,
        "windows": windows,
        "features": features,
        "probabilities": smoothed,
        "activities": activities,
        "centers": centers,
        "confidence": confidence,
        "metrics": metrics,
        "summary": summary,
        "top_activity": top_activity,
        "model_label": model_label,
        "duration_seconds": len(signals) / FS,
    }


def metric(label: str, value: str, help_text: str = ""):
    st.metric(label, value, help=help_text or None)


def make_profile():
    if PUBLIC_DEMO:
        return st.session_state.get("fitmaxxx_profile", fitmaxxx_store.DEFAULT_PROFILE.copy())
    return fitmaxxx_store.load_profile(DB_PATH)


def render_overview(profile: dict, snapshot: dict):
    st.markdown(ui.hero_html(profile.get("display_name", "there")), unsafe_allow_html=True)

    steps_pct = 100 * snapshot["steps"] / max(snapshot["steps_goal"], 1)
    active_pct = 100 * snapshot["active_minutes"] / max(snapshot["active_goal_min"], 1)
    sleep_pct = 100 * snapshot["sleep_hours"] / 8.0
    score = int(snapshot.get("activity_score", round((min(steps_pct, 100) + min(active_pct, 100) + min(sleep_pct, 100)) / 3)))

    st.markdown(ui.section_head("Your day at a glance", "A balanced view of movement and recovery"), unsafe_allow_html=True)
    ring_col, device_col = st.columns([1.08, 1], gap="large")

    with ring_col:
        ring_html = ui.progress_rings_html(steps_pct, active_pct, sleep_pct, score)
        legend = f"""
        <div class='fm-ring-legend'>
          <div class='fm-ring-row'><span class='fm-ring-label'><i class='fm-dot' style='background:#e65a90'></i>Steps</span><span class='fm-ring-value'>{snapshot['steps']:,} / {snapshot['steps_goal']:,}</span></div>
          <div class='fm-ring-row'><span class='fm-ring-label'><i class='fm-dot' style='background:#f39cba'></i>Active time</span><span class='fm-ring-value'>{snapshot['active_minutes']} / {snapshot['active_goal_min']} min</span></div>
          <div class='fm-ring-row'><span class='fm-ring-label'><i class='fm-dot' style='background:#c94d83'></i>Sleep</span><span class='fm-ring-value'>{snapshot['sleep_hours']:.1f} / 8 h</span></div>
          <div style='color:#9b8995;font-size:.72rem;padding-top:.55rem'>Daily goals at a glance</div>
        </div>
        """
        st.markdown(
            f"<div class='fm-card'><div class='fm-card-title'>Daily progress</div><p class='fm-card-sub'>Three habits that add up</p><div class='fm-ring-layout'>{ring_html}{legend}</div></div>",
            unsafe_allow_html=True,
        )

    with device_col:
        device_name = str(snapshot.get("device", st.session_state.fitmaxxx_device))
        st.markdown(
            f"<div class='fm-card'><div class='fm-card-title'>Your wearable</div><p class='fm-card-sub'>Today's simulated device snapshot</p>"
            f"<div class='fm-watch-row'>{ui.watch_art_html()}<div style='flex:1;min-width:0'><span class='fm-device-pill'>⌚ &nbsp;{ui.escape(device_name)}</span><p class='fm-card-sub' style='margin-top:.5rem'>Personal activity feed</p><span class='fm-status'>● &nbsp;Active</span></div></div>"
            f"<div class='fm-device-stats'>"
            f"<div class='fm-device-stat'><div class='fm-device-icon'>👟</div><div class='fm-device-value'>{snapshot['steps']:,}</div><div class='fm-device-label'>Steps</div></div>"
            f"<div class='fm-device-stat'><div class='fm-device-icon'>♡</div><div class='fm-device-value'>{snapshot['current_hr']}</div><div class='fm-device-label'>BPM</div></div>"
            f"<div class='fm-device-stat'><div class='fm-device-icon'>♨</div><div class='fm-device-value'>{snapshot['calories_kcal']}</div><div class='fm-device-label'>kcal</div></div>"
            f"<div class='fm-device-stat'><div class='fm-device-icon'>☾</div><div class='fm-device-value'>{snapshot['sleep_hours']:.1f}</div><div class='fm-device-label'>Sleep h</div></div>"
            f"</div></div>",
            unsafe_allow_html=True,
        )

    st.write("")
    weekly_col, hr_col = st.columns(2, gap="large")
    with weekly_col:
        st.markdown(ui.section_head("Weekly movement", "Steps per day · simulated wearable data"), unsafe_allow_html=True)
        weekly = pd.DataFrame({"Day": snapshot["weekly_days"], "Steps": snapshot["weekly_steps"]})
        chart = alt.Chart(weekly).mark_bar(cornerRadiusTopLeft=7, cornerRadiusTopRight=7, color="#ee8eae").encode(
            x=alt.X("Day:N", sort=snapshot["weekly_days"], axis=alt.Axis(title=None, labelColor="#786b77", domain=False, tickColor="transparent")),
            y=alt.Y("Steps:Q", axis=alt.Axis(title=None, labelColor="#8a7b87", gridColor="#f2e4ea", domain=False)),
            tooltip=["Day", alt.Tooltip("Steps:Q", format=",")],
        ).properties(height=245)
        with st.container(border=True):
            st.altair_chart(chart, width="stretch")

    with hr_col:
        st.markdown(ui.section_head("Heart-rate trend", "Illustrative daily trace"), unsafe_allow_html=True)
        heart = pd.DataFrame({"Time": snapshot["hr_labels"], "BPM": snapshot["hr_trace"]})
        chart = alt.Chart(heart).mark_area(line={"color": "#df5b8c", "strokeWidth": 2.5}, color="#f4b5ca", opacity=.25).encode(
            x=alt.X("Time:N", axis=alt.Axis(title=None, labelColor="#786b77", domain=False, tickColor="transparent")),
            y=alt.Y("BPM:Q", scale=alt.Scale(zero=False), axis=alt.Axis(title="bpm", labelColor="#8a7b87", gridColor="#f2e4ea", domain=False)),
            tooltip=["Time", "BPM"],
        ).properties(height=245)
        with st.container(border=True):
            st.altair_chart(chart, width="stretch")

    st.write("")
    st.markdown(ui.focus_html(profile.get("goal", "Build consistency"), profile.get("steps_goal", 8000)), unsafe_allow_html=True)


def render_wearable_studio(profile: dict):
    st.markdown("<div class='fm-page-head'><h1>Wearable studio</h1><p>Explore a simulated device feed.</p></div>", unsafe_allow_html=True)
    st.markdown("<span class='fm-badge'>SIMULATION MODE</span>", unsafe_allow_html=True)
    device_names = list(wearable_simulator.DEVICE_PROFILES)
    current = st.session_state.fitmaxxx_device
    if current not in device_names:
        current = device_names[0]
        st.session_state.fitmaxxx_device = current
    selected = st.selectbox("Device profile", device_names, index=device_names.index(current))
    if selected != st.session_state.fitmaxxx_device:
        st.session_state.fitmaxxx_device = selected
        st.session_state.fitmaxxx_snapshot = wearable_simulator.generate_snapshot(selected, profile, seed=2026 + st.session_state.fitmaxxx_refresh_count)
        st.rerun()
    spec = wearable_simulator.DEVICE_PROFILES[selected]
    st.markdown(ui.device_card_html(selected, spec.get("family", "Device profile")), unsafe_allow_html=True)
    if st.button("Refresh simulated reading", type="primary"):
        st.session_state.fitmaxxx_refresh_count += 1
        st.session_state.fitmaxxx_snapshot = wearable_simulator.generate_snapshot(selected, profile, seed=2026 + st.session_state.fitmaxxx_refresh_count)
        st.rerun()
    snap = st.session_state.fitmaxxx_snapshot
    cols = st.columns(4)
    values = [("Steps", f"{snap['steps']:,}"), ("Heart rate", f"{snap['current_hr']} bpm"), ("Sleep", f"{snap['sleep_hours']:.1f} h"), ("Estimated energy", f"{snap['calories_kcal']} kcal")]
    for col, (label, value) in zip(cols, values):
        with col:
            metric(label, value)
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown(ui.section_head("Heart-rate trace", "Simulated readings"), unsafe_allow_html=True)
        st.line_chart(pd.DataFrame({"BPM": snap["hr_trace"]}, index=snap["hr_labels"]), color="#df5b8c")
    with right:
        st.markdown(ui.section_head("Daily movement", "Activity summary"), unsafe_allow_html=True)
        metric("Active minutes", f"{snap['active_minutes']} min", f"Target {snap['active_goal_min']} min")
        metric("Floors", str(snap["floors"]))
        metric("Movement score", f"{snap['activity_score']}/100")


def render_activity_lab(profile: dict):
    st.markdown("<div class='fm-page-head'><h1>Activity lab</h1><p>Recognise movement from a phone sensor recording.</p></div>", unsafe_allow_html=True)
    source = st.radio("Recording source", ["Built-in session", "Sensor Logger ZIP", "Two CSV files"], horizontal=True)
    model_label = st.selectbox("Recognition model", list(MODEL_CONFIG), help="Choose the model that best matches the expected phone placement.")
    smoothing = st.select_slider("Prediction smoothing", options=[1, 3, 5, 7, 9], value=3)
    if st.button("Analyse recording", type="primary"):
        try:
            acc_df, gyro_df, source_label = load_recording(source)
            result = run_activity_analysis(acc_df, gyro_df, model_label, smoothing, float(profile["weight_kg"]))
            result["source_label"] = source_label
            st.session_state.fitmaxxx_activity_result = result
            st.success("Analysis complete")
        except Exception as exc:
            st.error(f"Could not analyse the recording: {exc}")
    result = st.session_state.get("fitmaxxx_activity_result")
    if result:
        m = result["metrics"]
        cols = st.columns(4)
        items = [("Recording", f"{m['duration_min']:.1f} min"), ("Movement", f"{m['active_min']:.1f} min"), ("Estimated energy", f"{m['kcal']:.1f} kcal"), ("Model confidence", f"{100*np.mean(result['confidence']):.0f}%")]
        for col, (label, value) in zip(cols, items):
            with col:
                metric(label, value)
        timeline = pd.DataFrame({
            "Time (s)": np.round(result["centers"] - result["start"], 2),
            "Activity": [result["activities"][i].replace("_", " ").title() for i in range(len(result["activities"]))],
            "Confidence": np.round(result["confidence"], 3),
        })
        st.markdown(ui.section_head("Predicted activity timeline", result["source_label"]), unsafe_allow_html=True)
        st.dataframe(timeline, hide_index=True, width="stretch", height=260)
        st.download_button("Download activity timeline", timeline.to_csv(index=False).encode("utf-8"), "fitmaxxx_activity_timeline.csv", "text/csv")
        st.markdown(ui.section_head("Time by activity", "Prediction windows"), unsafe_allow_html=True)
        summary = result["summary"].copy()
        summary["Activity"] = summary["activity"].str.replace("_", " ").str.title()
        chart = alt.Chart(summary).mark_bar(cornerRadiusEnd=7, color="#e982a7").encode(
            x=alt.X("windows:Q", title="Prediction windows"),
            y=alt.Y("Activity:N", sort="-x", title=None),
            tooltip=["Activity", "windows"],
        ).properties(height=220)
        st.altair_chart(chart, width="stretch")
        st.caption("Activity and energy values are estimates based on the selected model and supplied recording.")
    else:
        st.markdown("<div class='fm-soft-note'>Choose a sample recording or upload your own phone data to see the activity timeline.</div>", unsafe_allow_html=True)


def render_coach(profile: dict, snapshot: dict):
    st.markdown("<div class='fm-page-head'><h1>Your AI coach</h1><p>Personalised ideas for building a routine that lasts.</p></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='fm-card'><div class='fm-eyebrow'>YOUR CURRENT FOCUS</div><h2 style='margin:.25rem 0'>{ui.escape(profile['goal'])}</h2><span style='color:#8d7d89'>{ui.escape(profile['fitness_level'])} · {int(profile['steps_goal']):,} steps/day · {int(profile['active_goal_min'])} active min/day</span></div>", unsafe_allow_html=True)
    activity_payload = None
    activity_result = st.session_state.get("fitmaxxx_activity_result")
    if activity_result:
        activity_payload = {**activity_result["metrics"], "confidence": float(np.mean(activity_result["confidence"])), "top_activity": activity_result["top_activity"]}
    context = coach.build_context(profile, snapshot, activity_payload)
    api_key = get_secret("ANTHROPIC_API_KEY")
    model_name = get_secret("ANTHROPIC_MODEL", "claude-haiku-5-5")
    if st.button("Create my coaching brief", type="primary"):
        with st.spinner("Preparing your coaching brief…"):
            note, _source = coach.generate_note(context, api_key, model_name)
        st.session_state.fitmaxxx_coach_note = note
    note = st.session_state.get("fitmaxxx_coach_note")
    if note:
        st.markdown(f"<div class='fm-soft-note'>{note}</div>", unsafe_allow_html=True)
    st.markdown("### Ask your coach")
    with st.form("fitmaxxx_coach_question", clear_on_submit=True):
        question = st.text_input("Your question", placeholder="How can I build a more consistent walking routine?")
        ask = st.form_submit_button("Ask FitMaxxx")
    if ask and question.strip():
        with st.spinner("Thinking…"):
            response = coach.answer_question(question, context, api_key, model_name)
        st.markdown(f"<div class='fm-card'><strong>{ui.escape(question)}</strong><p>{ui.escape(response)}</p></div>", unsafe_allow_html=True)


def render_progress(snapshot: dict, profile: dict):
    st.markdown("<div class='fm-page-head'><h1>Your progress</h1><p>Keep a record of the small wins.</p></div>", unsafe_allow_html=True)
    if st.button("Save today's snapshot", type="primary"):
        if PUBLIC_DEMO:
            st.session_state.setdefault("fitmaxxx_sessions", []).append({
                "saved_at": pd.Timestamp.now(), "source": "Wearable simulation",
                "device": snapshot.get("device"), "steps": snapshot.get("steps"),
                "active_minutes": snapshot.get("active_minutes"), "sleep_hours": snapshot.get("sleep_hours"),
                "calories_kcal": snapshot.get("calories_kcal"), "activity_score": snapshot.get("activity_score"),
                "current_hr": snapshot.get("current_hr"),
            })
        else:
            fitmaxxx_store.save_session(DB_PATH, snapshot, profile, source="Wearable simulation")
        st.success("Saved to your progress history.")
    if PUBLIC_DEMO:
        sessions = pd.DataFrame(st.session_state.get("fitmaxxx_sessions", []))
        if not sessions.empty:
            sessions["saved_at"] = pd.to_datetime(sessions["saved_at"])
    else:
        sessions = fitmaxxx_store.load_sessions(DB_PATH)
    if sessions.empty:
        st.markdown("<div class='fm-soft-note'>Your saved sessions will appear here. Save today's snapshot to start your history.</div>", unsafe_allow_html=True)
        return
    cols = st.columns(3)
    for col, (label, value) in zip(cols, [("Saved sessions", str(len(sessions))), ("Average activity score", f"{sessions['activity_score'].mean():.0f}/100"), ("Total active time", f"{sessions['active_minutes'].sum():.0f} min")]):
        with col:
            metric(label, value)
    hist = sessions.copy()
    chart = alt.Chart(hist).mark_line(point=True, color="#df5b8c", strokeWidth=2.5).encode(
        x=alt.X("saved_at:T", title=None),
        y=alt.Y("activity_score:Q", scale=alt.Scale(domain=[0, 100]), title="Activity score"),
        tooltip=["saved_at:T", "device:N", "steps:Q", "active_minutes:Q", "activity_score:Q"],
    ).properties(height=260)
    st.altair_chart(chart, width="stretch")
    st.dataframe(hist.sort_values("saved_at", ascending=False), hide_index=True, width="stretch")
    with st.expander("Manage history"):
        confirm = st.checkbox("I understand this removes all saved sessions")
        if st.button("Clear history", disabled=not confirm):
            if PUBLIC_DEMO:
                st.session_state["fitmaxxx_sessions"] = []
            else:
                fitmaxxx_store.clear_sessions(DB_PATH)
            st.rerun()


def render_model_lab():
    st.markdown("<div class='fm-page-head'><h1>Model lab</h1><p>Benchmark results from the UCI HAR test split.</p></div>", unsafe_allow_html=True)
    benchmark_path = REPORTS / "model_comparison.csv"
    if benchmark_path.exists():
        benchmark = pd.read_csv(benchmark_path)
        if "Accuracy" in benchmark.columns:
            benchmark["Accuracy (%)"] = benchmark["Accuracy"] * 100
            chart = alt.Chart(benchmark).mark_bar(cornerRadiusEnd=7, color="#e982a7").encode(
                x=alt.X("Accuracy (%):Q", scale=alt.Scale(domain=[0, 100]), title="Test accuracy (%)"),
                y=alt.Y("Model:N", sort="-x", title=None),
                tooltip=["Model", alt.Tooltip("Accuracy (%):Q", format=".2f")],
            ).properties(height=250)
            st.altair_chart(chart, width="stretch")
        st.dataframe(benchmark, hide_index=True, width="stretch")
    metric_cols = st.columns(2)
    for col, (label, config) in zip(metric_cols, MODEL_CONFIG.items()):
        with col:
            data = load_json(str(config["metrics"]))
            st.markdown(f"<div class='fm-card'><div class='fm-card-title'>{ui.escape(label)}</div><p class='fm-card-sub'>{ui.escape(config['detail'])}</p>", unsafe_allow_html=True)
            if data:
                metric("Test accuracy", f"{100 * data.get('accuracy', 0):.2f}%")
                if "macro_f1" in data:
                    metric("Macro F1", f"{data['macro_f1']:.3f}")
                st.caption(f"{data.get('test_windows', '—')} test windows")
            else:
                st.caption("Evaluation metrics are not available yet.")
            st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------
# Profile and navigation state
# ---------------------------------------------
profile = make_profile()

if "fitmaxxx_device" not in st.session_state:
    device_names = list(wearable_simulator.DEVICE_PROFILES)
    preferred = "Apple Watch (simulated)"
    st.session_state.fitmaxxx_device = preferred if preferred in device_names else device_names[0]
if "fitmaxxx_refresh_count" not in st.session_state:
    st.session_state.fitmaxxx_refresh_count = 0
if "fitmaxxx_snapshot" not in st.session_state:
    st.session_state.fitmaxxx_snapshot = wearable_simulator.generate_snapshot(st.session_state.fitmaxxx_device, profile, seed=2026)
if "fitmaxxx_activity_result" not in st.session_state:
    st.session_state.fitmaxxx_activity_result = None


# ---------------------------------------------
# Sidebar
# ---------------------------------------------
with st.sidebar:
    st.markdown(ui.sidebar_brand_html(), unsafe_allow_html=True)
    st.markdown("<div class='fm-eyebrow' style='margin:0 0 .45rem'>WORKSPACE</div>", unsafe_allow_html=True)
    page = st.radio("Navigation", NAV_ITEMS, label_visibility="collapsed", key="fitmaxxx_navigation", format_func=lambda x: {"Overview": "⌂  Overview", "Wearable studio": "◉  Wearable studio", "Activity lab": "⌁  Activity lab", "AI coach": "✧  AI coach", "Progress": "▥  Progress", "Model lab": "◈  Model lab"}[x])
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<div class='fm-eyebrow'>YOUR PROFILE</div>", unsafe_allow_html=True)
    st.markdown(ui.profile_html(profile.get("display_name", "Alex"), "Fitness enthusiast"), unsafe_allow_html=True)
    with st.form("fitmaxxx_profile_form"):
        display_name = st.text_input("Name", value=str(profile.get("display_name", "Alex")))
        age = st.number_input("Age", min_value=18, max_value=90, value=int(profile.get("age", 25)))
        weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=float(profile.get("weight_kg", 70.0)), step=0.5)
        goals = ["Build consistency", "Improve stamina", "Weight management", "Strength and balance"]
        current_goal = profile.get("goal", goals[0])
        goal = st.selectbox("Main goal", goals, index=goals.index(current_goal) if current_goal in goals else 0)
        levels = ["Beginner", "Intermediate", "Advanced"]
        current_level = profile.get("fitness_level", "Intermediate")
        level = st.selectbox("Fitness level", levels, index=levels.index(current_level) if current_level in levels else 1)
        steps_goal = st.number_input("Daily step target", min_value=1000, max_value=30000, value=int(profile.get("steps_goal", 8000)), step=500)
        active_goal = st.number_input("Active-minute target", min_value=10, max_value=180, value=int(profile.get("active_goal_min", 45)), step=5)
        save = st.form_submit_button("Save profile", use_container_width=True)
    if save:
        profile = {"display_name": display_name.strip() or "Alex", "age": int(age), "weight_kg": float(weight), "goal": goal, "fitness_level": level, "steps_goal": int(steps_goal), "active_goal_min": int(active_goal)}
        if PUBLIC_DEMO:
            st.session_state.fitmaxxx_profile = profile
        else:
            fitmaxxx_store.save_profile(DB_PATH, profile)
        st.session_state.fitmaxxx_snapshot = wearable_simulator.generate_snapshot(st.session_state.fitmaxxx_device, profile, seed=2026 + st.session_state.fitmaxxx_refresh_count)
        st.success("Profile saved")
        st.rerun()
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<div class='fm-sidebar-small'>SIMULATED WEARABLE DATA<br>Device readings are generated for this demo.</div>", unsafe_allow_html=True)

snapshot = st.session_state.fitmaxxx_snapshot


# ---------------------------------------------
# Page routing
# ---------------------------------------------
if page == "Overview":
    render_overview(profile, snapshot)
elif page == "Wearable studio":
    render_wearable_studio(profile)
elif page == "Activity lab":
    render_activity_lab(profile)
elif page == "AI coach":
    render_coach(profile, snapshot)
elif page == "Progress":
    render_progress(snapshot, profile)
elif page == "Model lab":
    render_model_lab()

st.markdown("<div style='text-align:center;padding:1.4rem 0 .3rem;color:#b09eaa;font-size:.72rem'>FITMAXXX &nbsp;·&nbsp; Move with intention.</div>", unsafe_allow_html=True)
