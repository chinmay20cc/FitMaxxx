"""Sedentary nudges and impact detection."""
import numpy as np
from scipy.signal import find_peaks

from pipeline import FS, STEP_SEC

SEDENTARY = {"SITTING", "LAYING"}


def still_stretches(activity_names, min_minutes=0.0):
    """Unbroken sitting/laying runs. Returns list of dicts sorted by start time."""
    runs, start = [], None
    names = list(activity_names)
    for w, name in enumerate(names + [None]):
        sedentary = name in SEDENTARY
        if sedentary and start is None:
            start = w
        if not sedentary and start is not None:
            minutes = (w - start) * STEP_SEC / 60
            if minutes >= min_minutes:
                runs.append({
                    "start_s": round(start * STEP_SEC, 1),
                    "end_s": round(w * STEP_SEC, 1),
                    "minutes": round(minutes, 2),
                })
            start = None
    return runs


def longest_still(activity_names):
    runs = still_stretches(activity_names)
    return max(runs, key=lambda r: r["minutes"]) if runs else None


def detect_impacts(signals, peak_g=2.5, still_seconds=3.0, still_std=0.08, min_gap_seconds=5.0):
    """Find sharp acceleration spikes and check whether the phone is still afterwards.

    signals: (T, 6) array, acceleration in g (gravity included).
    A spike followed by stillness looks like a drop or a fall; a spike while you
    keep moving is just a bump. The two cannot be told apart reliably from one phone.
    """
    mag = np.linalg.norm(signals[:, :3], axis=1)
    peaks, _ = find_peaks(mag, height=peak_g, distance=int(min_gap_seconds * FS))
    events = []
    for p in peaks:
        after = mag[p + int(0.5 * FS): p + int((0.5 + still_seconds) * FS)]
        enough = len(after) >= int(0.8 * still_seconds * FS)
        then_still = bool(enough and after.std() < still_std and abs(after.mean() - 1.0) < 0.2)
        events.append({"time_s": round(float(p) / FS, 1), "peak_g": round(float(mag[p]), 2), "then_still": then_still})
    return events


def simulate_fall(signals, at_sec):
    """Return a copy of signals with a synthetic fall: brief free-fall, impact, then stillness.

    For demonstrating the detector only. Never presented as real data.
    """
    out = signals.copy()
    i = int(at_sec * FS)
    ref = out[max(0, i - 2 * FS): i, :3].mean(axis=0)
    g_hat = ref / (np.linalg.norm(ref) + 1e-9)
    rng = np.random.default_rng(7)

    fall_end = i + int(0.3 * FS)
    out[i:fall_end, :3] = g_hat * 0.3
    out[fall_end:fall_end + 3, :3] = g_hat * 4.0
    rest = slice(fall_end + 3, min(len(out), fall_end + 3 + int(6.5 * FS)))
    n = len(out[rest])
    out[rest, :3] = g_hat * 1.0 + rng.normal(0, 0.01, (n, 3))
    out[i:rest.stop, 3:] = rng.normal(0, 0.005, (len(out[i:rest.stop]), 3))
    return out
