"""Build a small, non-personal demo recording from the UCI HAR *test* set.

Writes app/sample_data/demo_acc.csv, demo_gyro.csv and demo_truth.csv.
Windows are taken every second one (the UCI windows overlap by 50%), so the
signal is continuous. The truth file lets the app show a real accuracy score.
UCI HAR Dataset: Anguita et al. 2013, CC BY 4.0.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from raw_features import extract, load_raw, load_y

PROJECT_DIR = Path(__file__).resolve().parents[1]
OUT = PROJECT_DIR / "app" / "sample_data"
OUT.mkdir(parents=True, exist_ok=True)

NAMES = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "SITTING", "STANDING", "LAYING"]
PER_ACTIVITY = 8  # non-overlapping 2.56 s windows per activity
G = 9.81

windows = load_raw("test")  # (n, 128, 6): total_acc xyz (g), body_gyro xyz (rad/s)
y = load_y("test")
subject = np.loadtxt(PROJECT_DIR / "data" / "UCI HAR Dataset" / "test" / "subject_test.txt", dtype=int)


def first_run(subj, label):
    idx = np.where((subject == subj) & (y == label))[0]
    if len(idx) == 0:
        return None
    run = [idx[0]]
    for i in idx[1:]:
        if i == run[-1] + 1:
            run.append(i)
        else:
            break
    return run


# Pick a REPRESENTATIVE subject: the one whose accuracy is closest to the overall
# test accuracy (not the best one), so the demo does not flatter the model.
bundle = joblib.load(PROJECT_DIR / "models" / "raw_model.joblib")
correct = bundle["model"].predict(extract(windows)) == y
overall = correct.mean()
order = sorted(np.unique(subject), key=lambda s: abs(correct[subject == s].mean() - overall))

chosen = None
for subj in order:
    runs = [first_run(subj, a) for a in range(6)]
    if all(r is not None and len(r) >= 2 * PER_ACTIVITY - 1 for r in runs):
        chosen = (subj, runs)
        break
if chosen is None:
    raise SystemExit("No single test subject has long enough runs for every activity.")

subj, runs = chosen
chunks, truth, t0 = [], [], 0.0
for activity, run in enumerate(runs):
    picked = run[: 2 * PER_ACTIVITY : 2]
    block = np.concatenate([windows[i] for i in picked], axis=0)
    chunks.append(block)
    duration = len(block) / 50
    truth.append({"start_sec": round(t0, 2), "end_sec": round(t0 + duration, 2), "label": NAMES[activity]})
    t0 += duration

signal = np.concatenate(chunks, axis=0)
t = np.arange(len(signal)) / 50 + 0.02

pd.DataFrame({"seconds_elapsed": t, "x": signal[:, 0] * G, "y": signal[:, 1] * G, "z": signal[:, 2] * G}).to_csv(OUT / "demo_acc.csv", index=False)
pd.DataFrame({"seconds_elapsed": t, "x": signal[:, 3], "y": signal[:, 4], "z": signal[:, 5]}).to_csv(OUT / "demo_gyro.csv", index=False)
pd.DataFrame(truth).to_csv(OUT / "demo_truth.csv", index=False)
print(f"Subject {subj} (their accuracy {correct[subject == subj].mean():.2f} vs overall {overall:.2f}): {len(signal)/50:.0f} s demo saved in {OUT}")
