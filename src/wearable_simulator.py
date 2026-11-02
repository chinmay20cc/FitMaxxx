"""Synthetic wearable readings for FitMaxxx demos; no device connection is made."""
from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np

DEVICE_PROFILES = {
    "Apple Watch (simulated)": {
        "family": "Smartwatch", "resting_hr": 62, "hr_spread": 8,
        "sleep_bias": 7.4, "steps_bias": 7600, "active_bias": 48,
        "calorie_factor": 0.041, "accent": "coral", "asset": "device_watch.svg",
    },
    "Galaxy Watch (simulated)": {
        "family": "Smartwatch", "resting_hr": 65, "hr_spread": 9,
        "sleep_bias": 7.1, "steps_bias": 8200, "active_bias": 52,
        "calorie_factor": 0.040, "accent": "blue", "asset": "device_watch.svg",
    },
    "Garmin Forerunner (simulated)": {
        "family": "Sports watch", "resting_hr": 56, "hr_spread": 7,
        "sleep_bias": 7.6, "steps_bias": 9400, "active_bias": 64,
        "calorie_factor": 0.043, "accent": "lime", "asset": "device_sport.svg",
    },
    "Fitbit Charge (simulated)": {
        "family": "Fitness band", "resting_hr": 64, "hr_spread": 8,
        "sleep_bias": 7.7, "steps_bias": 7100, "active_bias": 43,
        "calorie_factor": 0.039, "accent": "mint", "asset": "device_band.svg",
    },
    "Oura Ring (simulated)": {
        "family": "Smart ring", "resting_hr": 60, "hr_spread": 5,
        "sleep_bias": 7.9, "steps_bias": 6800, "active_bias": 38,
        "calorie_factor": 0.038, "accent": "stone", "asset": "device_ring.svg",
    },
}


def generate_snapshot(
    device: str,
    profile: dict[str, Any] | None = None,
    *,
    seed: int | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Generate plausible but explicitly synthetic daily wearable metrics."""
    if device not in DEVICE_PROFILES:
        raise ValueError(f"Unknown demo device: {device}")
    profile = profile or {}
    spec = DEVICE_PROFILES[device]
    rng = np.random.default_rng(seed)
    goal = str(profile.get("goal", "Build consistency"))
    steps_goal = int(profile.get("steps_goal", 8000))
    weight = float(profile.get("weight_kg", 70))

    goal_factor = {
        "Build consistency": 0.92,
        "Improve stamina": 1.12,
        "Weight management": 1.00,
        "Strength and balance": 0.88,
    }.get(goal, 1.0)
    steps = int(np.clip(rng.normal(spec["steps_bias"] * goal_factor, 1450), 1200, 18000))
    active = int(np.clip(rng.normal(spec["active_bias"] * goal_factor, 12), 8, 150))
    sleep = float(np.clip(rng.normal(spec["sleep_bias"], 0.65), 4.2, 9.5))
    resting_hr = int(np.clip(rng.normal(spec["resting_hr"], 3), 48, 82))
    current_hr = int(np.clip(rng.normal(resting_hr + rng.uniform(7, 24), spec["hr_spread"]), 52, 158))
    floors = int(np.clip(rng.normal(max(1, steps / 1900), 2), 0, 32))
    calories = int(max(80, steps * spec["calorie_factor"] + active * weight * 0.035))
    now = now or datetime.now()

    # An illustrative 12-point heart-rate trace for the dashboard.
    base = np.linspace(resting_hr + 5, current_hr, 12)
    hr_trace = np.clip(base + rng.normal(0, spec["hr_spread"] * 0.52, 12), 48, 175).round().astype(int)
    time_labels = [(now.replace(minute=0, second=0, microsecond=0)).strftime("%H:%M")]
    start_hour = (now.hour - 11) % 24
    time_labels = [f"{(start_hour + i) % 24:02d}:00" for i in range(12)]

    step_progress = min(steps / max(steps_goal, 1), 1.35)
    active_goal = int(profile.get("active_goal_min", 45))
    active_progress = min(active / max(active_goal, 1), 1.35)
    sleep_progress = min(sleep / 8.0, 1.2)
    activity_score = int(round(100 * np.mean([
        min(step_progress, 1), min(active_progress, 1), min(sleep_progress, 1)
    ])))

    return {
        "device": device,
        "device_family": spec["family"],
        "source": "SIMULATED",
        "generated_at": now.isoformat(timespec="seconds"),
        "steps": steps,
        "steps_goal": steps_goal,
        "active_minutes": active,
        "active_goal_min": active_goal,
        "sleep_hours": round(sleep, 1),
        "current_hr": current_hr,
        "resting_hr": resting_hr,
        "floors": floors,
        "calories_kcal": calories,
        "weight_kg": round(weight, 1),
        "activity_score": activity_score,
        "hr_labels": time_labels,
        "hr_trace": hr_trace.tolist(),
        "weekly_steps": [
            int(np.clip(rng.normal(steps * f, 850), 900, 16000))
            for f in (0.72, 0.88, 0.95, 0.81, 1.04, 0.91, 1.0)
        ],
        "weekly_days": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
    }
