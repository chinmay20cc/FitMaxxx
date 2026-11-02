import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

# ==========================================
# 1. CONFIGURATION
# ==========================================

FS = 50
WIN = 128
STEP = 64
STEP_SEC = STEP / FS  # 1.28 s of new data per window

# Approximate MET values for calorie estimation
MET = {
    "WALKING": 3.5,
    "WALKING_UPSTAIRS": 8.0,
    "WALKING_DOWNSTAIRS": 3.5,
    "SITTING": 1.3,
    "STANDING": 1.8,
    "LAYING": 1.0,
}

ACTIVE = {"WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS"}
UPRIGHT = ACTIVE | {"STANDING"}


# ==========================================
# 2. VALIDATE SENSOR DATA
# ==========================================

def validate_sensor_data(df, name):
    required = ["seconds_elapsed", "x", "y", "z"]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")

    df = df[required].copy()
    for column in required:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df.dropna(inplace=True)
    df.sort_values("seconds_elapsed", inplace=True)
    df.drop_duplicates(subset="seconds_elapsed", inplace=True)

    if len(df) < 2:
        raise ValueError(f"Insufficient valid readings in {name}")
    return df


def detect_acc_units(acc):
    """Return (label, divisor_to_g) by looking at the size of the acceleration vector.

    Total acceleration includes gravity, so even its quieter moments (10th percentile)
    sit near 9.8 in m/s^2 or near 1.0 in g. A gravity-removed (linear) file drops
    close to 0 when you are still, so it is rejected: the model was trained on
    gravity-included data.
    """
    acc = validate_sensor_data(acc, "Accelerometer")
    magnitude = np.linalg.norm(acc[["x", "y", "z"]].to_numpy(), axis=1)
    quiet = float(np.percentile(magnitude, 10))
    if quiet > 6:
        return "m/s²", 9.81
    if 0.6 < quiet < 1.4:
        return "g", 1.0
    raise ValueError(
        "This acceleration file has no gravity component (its quiet moments read "
        f"{quiet:.2f}). Use TotalAcceleration.csv, not Accelerometer.csv."
    )


# ==========================================
# 3. SIGNALS AND WINDOWS
# ==========================================

def prepare_signals(acc, gyro, g=9.81):
    """Resample both sensors to 50 Hz. Returns (signals (T, 6) with acc in g, start_sec)."""
    acc = validate_sensor_data(acc, "Accelerometer")
    gyro = validate_sensor_data(gyro, "Gyroscope")

    start = max(acc["seconds_elapsed"].min(), gyro["seconds_elapsed"].min())
    end = min(acc["seconds_elapsed"].max(), gyro["seconds_elapsed"].max())
    if end <= start:
        raise ValueError("Sensor timestamps do not overlap.")

    t = np.arange(start, end, 1 / FS)

    def resample(df):
        return np.column_stack(
            [np.interp(t, df["seconds_elapsed"], df[axis]) for axis in ["x", "y", "z"]]
        )

    return np.hstack([resample(acc) / g, resample(gyro)]), float(start)


def windows_from_signals(signals):
    if len(signals) < WIN:
        raise ValueError("Recording is shorter than one 2.56-second window.")
    return np.stack([signals[i:i + WIN] for i in range(0, len(signals) - WIN + 1, STEP)])


def csv_to_windows(acc, gyro, g=9.81):
    """Convert accelerometer and gyroscope tables into (n, 128, 6) windows."""
    signals, _ = prepare_signals(acc, gyro, g)
    return windows_from_signals(signals)


def window_centers(n_windows, start=0.0):
    """Centre time (seconds) of the new 1.28 s slice each window represents."""
    return start + (np.arange(n_windows) * STEP + WIN / 2) / FS


# ==========================================
# 4. SMOOTHING
# ==========================================

def smooth_probabilities(probs, k=5):
    """Moving average over k neighbouring windows, then pick the top class."""
    probs = np.asarray(probs, dtype=float)
    if k > 1:
        probs = uniform_filter1d(probs, size=k, axis=0, mode="nearest")
    return probs, probs.argmax(axis=1), probs.max(axis=1)


# ==========================================
# 5. SUMMARIES, SCORE AND CALORIES
# ==========================================

def summarize(activity_names, weight_kg=60):
    """Minutes and estimated kcal per activity. kcal = MET x kg x hours."""
    if not 30 <= weight_kg <= 150:
        raise ValueError("Weight must be between 30 and 150 kg.")
    if len(activity_names) == 0:
        raise ValueError("No activity predictions provided.")

    summary = pd.Series(activity_names).value_counts().rename("windows").to_frame()
    summary["minutes"] = summary["windows"] * STEP_SEC / 60
    summary["kcal"] = [
        MET.get(activity, 1.0) * weight_kg * (minutes / 60)
        for activity, minutes in zip(summary.index, summary["minutes"])
    ]
    return summary.round(2)


def session_metrics(activity_names, weight_kg=60, goal_active=0.30, goal_upright=0.60, goal_met=2.0):
    """Session totals plus three ring values (0 to 1+) and a 0-100 proxy score.

    Move   = share of time walking (any kind) vs goal share
    Upright = share of time on your feet vs goal share
    Burn   = average MET above resting (1.0) vs goal average MET above resting
    The score is a transparent heuristic for this project, not a validated metric.
    """
    names = list(activity_names)
    n = len(names)
    if n == 0:
        raise ValueError("No activity predictions provided.")

    duration_min = n * STEP_SEC / 60
    active_share = sum(a in ACTIVE for a in names) / n
    upright_share = sum(a in UPRIGHT for a in names) / n
    avg_met = float(np.mean([MET.get(a, 1.0) for a in names]))

    rings = {
        "move": active_share / goal_active,
        "upright": upright_share / goal_upright,
        "burn": max(avg_met - 1.0, 0.0) / (goal_met - 1.0),
    }
    score = round(100 * float(np.mean([min(v, 1.0) for v in rings.values()])))

    return {
        "duration_min": duration_min,
        "active_min": duration_min * active_share,
        "upright_min": duration_min * upright_share,
        "sedentary_min": duration_min * (1 - upright_share),
        "avg_met": avg_met,
        "kcal": avg_met * weight_kg * duration_min / 60,
        "rings": rings,
        "score": score,
    }


# ==========================================
# 6. CROSS-CHECKS WITH THE PHONE'S OWN SENSORS
# ==========================================

def estimate_steps(signals, active_windows):
    """Count acceleration peaks, only inside windows the model called walking.

    signals: (T, 6) array from prepare_signals (acceleration in g).
    active_windows: boolean array, one value per window.
    """
    magnitude = np.linalg.norm(signals[:, :3], axis=1)
    detrended = magnitude - uniform_filter1d(magnitude, size=FS, mode="nearest")
    smooth = uniform_filter1d(detrended, size=5, mode="nearest")

    peaks, _ = find_peaks(smooth, prominence=0.12, distance=int(0.3 * FS))

    mask = np.zeros(len(signals), dtype=bool)
    for w, is_active in enumerate(active_windows):
        if is_active:
            mask[w * STEP + STEP // 2: w * STEP + STEP // 2 + STEP] = True
    return int(mask[peaks].sum())


def pedometer_total(ped_df, start=0.0, end=None):
    """Steps the phone counted during the recording (cumulative counter)."""
    df = ped_df[["seconds_elapsed", "steps"]].apply(pd.to_numeric, errors="coerce").dropna()
    df = df.sort_values("seconds_elapsed")
    before = df[df["seconds_elapsed"] < start]
    base = before["steps"].iloc[-1] if len(before) else 0
    inside = df[df["seconds_elapsed"] >= start]
    if end is not None:
        inside = inside[inside["seconds_elapsed"] <= end]
    if inside.empty:
        return 0
    return int(inside["steps"].iloc[-1] - base)


def android_agreement(centers, our_names, activity_df):
    """Compare our moving/still calls with Android's own activity events.

    Only windows where Android said walking/running or stationary are scored;
    'unknown' and 'tilting' are skipped. Returns (agreement_percent or None, scored_windows).
    """
    df = activity_df[["seconds_elapsed", "activity"]].copy()
    df["seconds_elapsed"] = pd.to_numeric(df["seconds_elapsed"], errors="coerce")
    df = df.dropna().sort_values("seconds_elapsed")
    if df.empty:
        return None, 0

    idx = np.searchsorted(df["seconds_elapsed"].to_numpy(), centers, side="right") - 1
    hits = total = 0
    for i, name in zip(idx, our_names):
        if i < 0:
            continue
        label = str(df["activity"].iloc[i]).lower()
        if label in ("walking", "running"):
            truth_moving = True
        elif label == "stationary":
            truth_moving = False
        else:
            continue
        total += 1
        hits += truth_moving == (name in ACTIVE)
    return (100 * hits / total if total else None), total
