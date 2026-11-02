"""Orientation-robust features.

The UCI phones were worn on the waist. A phone in a pocket or hand points a
different way, so raw x/y/z values mean something else. Here every window is
re-expressed relative to its own gravity direction (vertical vs horizontal),
which does not depend on how the phone is turned.
"""
import numpy as np

# 6-class names (UCI order) and the 4-class grouping used by the robust model
ROBUST_NAMES = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "STILL"]
TO_ROBUST = {0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 3}  # sitting, standing, laying -> STILL


def to_invariant(w):
    """w: (n, 128, 6) -> (n, 128, 6) channels that do not depend on phone orientation."""
    acc, gyro = w[:, :, :3], w[:, :, 3:]
    gravity = acc.mean(axis=1, keepdims=True)
    g_hat = gravity / (np.linalg.norm(gravity, axis=2, keepdims=True) + 1e-9)

    a_vert = (acc * g_hat).sum(axis=2)
    a_horiz = np.linalg.norm(acc - a_vert[..., None] * g_hat, axis=2)
    w_vert = (gyro * g_hat).sum(axis=2)
    w_horiz = np.linalg.norm(gyro - w_vert[..., None] * g_hat, axis=2)

    return np.stack(
        [np.linalg.norm(acc, axis=2), a_vert, a_horiz, np.linalg.norm(gyro, axis=2), w_vert, w_horiz],
        axis=2,
    )


def extract_robust(w):
    """(n, 128, 6) windows -> (n, 42) orientation-independent statistical features."""
    x = to_invariant(w)
    feats = [x.mean(1), x.std(1), x.min(1), x.max(1), np.median(x, 1), (x ** 2).mean(1), x.max(1) - x.min(1)]
    return np.hstack(feats)
