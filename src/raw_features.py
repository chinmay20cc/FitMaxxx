from pathlib import Path
import numpy as np


# Project directory
PROJECT_DIR = Path(__file__).resolve().parents[1]

# UCI HAR dataset location
BASE = PROJECT_DIR / "data" / "UCI HAR Dataset"

# Six raw sensor channels
SIGNALS = [
    "total_acc_x",
    "total_acc_y",
    "total_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z"
]


def load_raw(split):

    if split not in {"train", "test"}:
        raise ValueError("split must be train or test")

    folder = BASE / split / "Inertial Signals"

    channels = []

    for signal in SIGNALS:

        file_path = folder / f"{signal}_{split}.txt"

        data = np.loadtxt(file_path)

        channels.append(data)

    # Shape: (samples, 128 readings, 6 channels)
    return np.stack(channels, axis=2)


def load_y(split):

    if split not in {"train", "test"}:
        raise ValueError("split must be train or test")

    return np.loadtxt(
        BASE / split / f"y_{split}.txt",
        dtype=int
    ) - 1


def extract(w):

    """
    Extract 44 statistical features.

    Input:
        (number of windows, 128 readings, 6 channels)

    Output:
        (number of windows, 44 features)
    """

    w = np.asarray(w)

    if w.ndim != 3 or w.shape[2] != 6:
        raise ValueError("Expected data with 6 sensor channels")

    # Seven statistical features per channel
    features = [
        w.mean(axis=1),
        w.std(axis=1),
        w.min(axis=1),
        w.max(axis=1),
        np.median(w, axis=1),
        (w ** 2).mean(axis=1),
        w.max(axis=1) - w.min(axis=1)
    ]

    # Acceleration magnitude
    magnitude = np.linalg.norm(
        w[:, :, :3],
        axis=2
    )

    # Two additional features
    features += [
        magnitude.mean(axis=1, keepdims=True),
        magnitude.std(axis=1, keepdims=True)
    ]

    return np.hstack(features)