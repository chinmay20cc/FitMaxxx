import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import coach  # noqa: E402
import history  # noqa: E402
from insights import detect_impacts, longest_still, simulate_fall, still_stretches  # noqa: E402
from robust_features import extract_robust  # noqa: E402


def random_rotation(seed=3):
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    return q if np.linalg.det(q) > 0 else -q


def test_robust_features_ignore_phone_orientation():
    rng = np.random.default_rng(0)
    w = rng.normal(size=(5, 128, 6))
    w[:, :, 2] += 1.0  # gravity-like offset
    r = random_rotation()
    rotated = w.copy()
    rotated[:, :, :3] = w[:, :, :3] @ r.T
    rotated[:, :, 3:] = w[:, :, 3:] @ r.T
    assert np.allclose(extract_robust(w), extract_robust(rotated), atol=1e-6)


def test_still_stretches_and_longest():
    names = ["SITTING"] * 40 + ["WALKING"] * 5 + ["LAYING"] * 100
    runs = still_stretches(names, min_minutes=0.5)
    assert len(runs) == 2
    assert longest_still(names)["minutes"] == runs[1]["minutes"]
    assert longest_still(["WALKING"] * 3) is None


def quiet_signal(seconds=120):
    n = seconds * 50
    sig = np.zeros((n, 6))
    sig[:, 2] = 1.0
    sig += np.random.default_rng(1).normal(0, 0.01, sig.shape)
    return sig


def test_simulated_fall_is_detected_as_still():
    events = detect_impacts(simulate_fall(quiet_signal(), 60))
    assert len(events) == 1 and events[0]["then_still"] is True


def test_spike_while_moving_is_not_flagged_still():
    sig = quiet_signal(60)
    sig[:, :3] += np.random.default_rng(2).normal(0, 0.4, (len(sig), 3))
    sig[1500, :3] = [0, 0, 4.0]
    events = detect_impacts(sig)
    assert events and not any(e["then_still"] for e in events)


def sample_stats(conf=90, placement="waist"):
    return {
        "source": "demo", "placement": placement, "duration_minutes": 5.0, "walking_minutes": 1.0,
        "on_feet_minutes": 2.0, "sitting_or_laying_minutes": 3.0, "estimated_kcal": 12.0, "movement_score": 55,
        "minutes_per_activity": {"WALKING": 1.0, "SITTING": 3.0, "STANDING": 1.0},
        "model_confidence_percent": conf, "longest_still_stretch_minutes": 3.0,
        "still_stretch_alerts": 1, "impact_events": 0,
    }


def test_rule_note_has_three_points_and_caution():
    note = coach.rule_based_note(sample_stats(conf=60, placement="any"))
    assert note.count("\n- ") + 1 == 3
    assert "Treat this with care" in note


def test_llm_note_returns_none_when_client_fails(monkeypatch):
    fake = types.ModuleType("anthropic")

    class Boom:
        def __init__(self, **kw):
            raise RuntimeError("no network")

    fake.Anthropic = Boom
    monkeypatch.setitem(sys.modules, "anthropic", fake)
    assert coach.llm_note(sample_stats(), "key") is None


def test_llm_note_reads_text_blocks(monkeypatch):
    fake = types.ModuleType("anthropic")
    sent = {}

    class Client:
        def __init__(self, **kw):
            self.messages = self

        def create(self, **kw):
            sent.update(kw)
            block = types.SimpleNamespace(type="text", text="- one\n- two\n- three")
            return types.SimpleNamespace(content=[block])

    fake.Anthropic = Client
    monkeypatch.setitem(sys.modules, "anthropic", fake)
    assert coach.llm_note(sample_stats(), "key").startswith("- one")
    assert "walking_minutes" in sent["messages"][0]["content"]


def test_history_roundtrip_and_streak(tmp_path):
    db = tmp_path / "h.db"
    row = {"label": "t", "duration_min": 5, "active_min": 2, "upright_min": 3, "sedentary_min": 2,
           "kcal": 10, "score": 80, "confidence": 0.8, "top_activity": "Walking", "placement": "any"}
    for day in ("2026-10-28", "2026-10-29", "2026-10-30"):
        history.save_session(db, row, when=f"{day}T09:00:00")
    history.save_session(db, dict(row, score=40), when="2026-10-25T09:00:00")
    df = history.load_sessions(db)
    assert len(df) == 4
    assert history.streak_days(df, target=70) == 3
    history.clear_sessions(db)
    assert history.load_sessions(db).empty


def test_rule_note_flags_off_waist_own_recording():
    stats = dict(sample_stats(conf=90, placement="any"), source="own recording")
    assert "not worn at the waist" in coach.rule_based_note(stats)
    assert "Treat this" not in coach.rule_based_note(dict(sample_stats(conf=90, placement="any")))
