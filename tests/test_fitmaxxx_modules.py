import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import fitmaxxx_store
import wearable_simulator
import coach


def test_simulator_is_deterministic_with_seed():
    a = wearable_simulator.generate_snapshot("Apple Watch (simulated)", seed=42)
    b = wearable_simulator.generate_snapshot("Apple Watch (simulated)", seed=42)
    assert a["steps"] == b["steps"]
    assert a["current_hr"] == b["current_hr"]
    assert a["source"] == "SIMULATED"


def test_simulator_rejects_unknown_device():
    with pytest.raises(ValueError):
        wearable_simulator.generate_snapshot("Connected watch")


def test_simulator_respects_profile_goal():
    profile = {"goal": "Improve stamina", "steps_goal": 10000, "weight_kg": 70}
    data = wearable_simulator.generate_snapshot("Garmin Forerunner (simulated)", profile, seed=5)
    assert data["steps_goal"] == 10000
    assert 0 <= data["activity_score"] <= 100


def test_profile_roundtrip(tmp_path):
    db = tmp_path / "fitmaxxx.db"
    profile = {"display_name": "CC", "goal": "Improve stamina", "weight_kg": 78.5}
    fitmaxxx_store.save_profile(db, profile)
    loaded = fitmaxxx_store.load_profile(db)
    assert loaded["display_name"] == "CC"
    assert loaded["goal"] == "Improve stamina"
    assert loaded["weight_kg"] == 78.5


def test_session_history_roundtrip(tmp_path):
    db = tmp_path / "fitmaxxx.db"
    snapshot = wearable_simulator.generate_snapshot("Fitbit Charge (simulated)", seed=7)
    profile = fitmaxxx_store.DEFAULT_PROFILE.copy()
    fitmaxxx_store.save_session(db, snapshot, profile)
    history = fitmaxxx_store.load_sessions(db)
    assert len(history) == 1
    assert history.iloc[0]["source"] == "Wearable simulation"
    assert history.iloc[0]["device"] == "Fitbit Charge (simulated)"


def test_coach_context_uses_aggregate_values_only():
    profile = {"display_name": "CC", "goal": "Build consistency", "fitness_level": "Beginner"}
    wearable = wearable_simulator.generate_snapshot("Galaxy Watch (simulated)", seed=3)
    context = coach.build_context(profile, wearable)
    assert context["profile"]["goal"] == "Build consistency"
    assert "steps" in context["wearable_summary"]
    assert "raw_sensor_data" not in context


def test_local_coach_note_is_personalized():
    profile = {"display_name": "CC", "goal": "Improve stamina", "fitness_level": "Beginner", "steps_goal": 9000, "active_goal_min": 40}
    wearable = wearable_simulator.generate_snapshot("Garmin Forerunner (simulated)", profile, seed=11)
    context = coach.build_context(profile, wearable)
    note = coach.local_coach_note(context)
    assert "CC" in note
    assert "stamina" in note.lower()


def test_smart_ring_profile_available():
    data = wearable_simulator.generate_snapshot("Oura Ring (simulated)", seed=9)
    assert data["device_family"] == "Smart ring"
    assert data["source"] == "SIMULATED"


def test_local_coach_answers_steps_question():
    profile = {"display_name": "CC", "goal": "Build consistency", "fitness_level": "Beginner", "steps_goal": 9000, "active_goal_min": 40}
    wearable = wearable_simulator.generate_snapshot("Fitbit Charge (simulated)", profile, seed=15)
    context = coach.build_context(profile, wearable)
    answer = coach.answer_question("How can I get more steps?", context)
    assert "steps" in answer.lower() or "step" in answer.lower()
    assert "walk" in answer.lower()
