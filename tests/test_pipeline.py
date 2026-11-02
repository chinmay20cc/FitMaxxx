import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pipeline import (  # noqa: E402
    ACTIVE, csv_to_windows, detect_acc_units, pedometer_total, session_metrics,
    smooth_probabilities, summarize, window_centers,
)


def make_df(seconds=10, scale=1.0, offset=(0, 0, 0)):
    t = np.arange(0, seconds, 0.02)
    n = len(t)
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "seconds_elapsed": t,
        "x": rng.normal(offset[0], scale, n),
        "y": rng.normal(offset[1], scale, n),
        "z": rng.normal(offset[2], scale, n),
    })


def test_window_shape():
    w = csv_to_windows(make_df(), make_df())
    assert w.shape[1:] == (128, 6)
    assert len(w) >= 1


def test_short_recording_rejected():
    with pytest.raises(ValueError):
        csv_to_windows(make_df(1), make_df(1))


def test_missing_columns_rejected():
    with pytest.raises(ValueError):
        csv_to_windows(make_df().drop(columns=["z"]), make_df())


def test_summary_minutes():
    s = summarize(["WALKING"] * 47 + ["SITTING"] * 3)
    assert abs(s["minutes"].sum() - 50 * 1.28 / 60) < 0.05


def test_summary_rejects_bad_weight():
    with pytest.raises(ValueError):
        summarize(["WALKING"], weight_kg=500)


def test_units_metres_per_second_squared():
    acc = make_df(scale=0.3, offset=(0, 0, 9.8))
    assert detect_acc_units(acc)[0] == "m/s²"


def test_units_g():
    acc = make_df(scale=0.05, offset=(0, 0, 1.0))
    assert detect_acc_units(acc)[0] == "g"


def test_gravity_free_file_rejected():
    with pytest.raises(ValueError):
        detect_acc_units(make_df(scale=0.5))


def test_smoothing_removes_single_flicker():
    probs = np.array([[1, 0]] * 4 + [[0, 1]] + [[1, 0]] * 4, dtype=float)
    _, idx, conf = smooth_probabilities(probs, 5)
    assert set(idx) == {0}
    assert conf.max() <= 1.0


def test_smoothing_k1_is_identity():
    probs = np.random.default_rng(1).dirichlet(np.ones(3), size=6)
    _, idx, _ = smooth_probabilities(probs, 1)
    assert (idx == probs.argmax(axis=1)).all()


def test_session_metrics_all_sitting_scores_zero_burn():
    m = session_metrics(["SITTING"] * 20)
    assert m["rings"]["move"] == 0
    assert m["score"] < 40


def test_session_metrics_all_walking_fills_move_ring():
    m = session_metrics(["WALKING"] * 20)
    assert m["rings"]["move"] >= 1
    assert m["active_min"] == pytest.approx(m["duration_min"])


def test_pedometer_total_uses_start_offset():
    ped = pd.DataFrame({"seconds_elapsed": [-100, 10, 20, 30], "steps": [0, 5, 12, 20]})
    assert pedometer_total(ped, start=0, end=30) == 20


def test_window_centers_spacing():
    c = window_centers(4, start=1.0)
    assert np.allclose(np.diff(c), 1.28)


def test_active_set():
    assert "SITTING" not in ACTIVE and "WALKING" in ACTIVE
