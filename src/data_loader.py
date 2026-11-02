from pathlib import Path
import pandas as pd


# Locate the project directory automatically
PROJECT_DIR = Path(__file__).resolve().parents[1]

# Dataset location
DATA_DIR = PROJECT_DIR / "data" / "UCI HAR Dataset"


def _feature_names():
    """
    Load the 561 feature names.
    Make duplicate names unique so each DataFrame column is distinct.
    """
    raw = pd.read_csv(
        DATA_DIR / "features.txt",
        sep=r"\s+",
        header=None
    )[1].astype(str).tolist()

    counts = {}
    unique_names = []

    for name in raw:
        count = counts.get(name, 0)

        if count == 0:
            unique_names.append(name)
        else:
            unique_names.append(f"{name}_{count}")

        counts[name] = count + 1

    return unique_names


def load_split(split):
    """
    Load the training or testing dataset.

    Returns:
        X: Sensor features
        y: Activity labels
        subject: Volunteer IDs
    """
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")

    folder = DATA_DIR / split

    # Load sensor features
    X = pd.read_csv(
        folder / f"X_{split}.txt",
        sep=r"\s+",
        header=None,
        names=_feature_names()
    )

    # Load activity labels
    y = pd.read_csv(
        folder / f"y_{split}.txt",
        sep=r"\s+",
        header=None
    )[0]

    # Load volunteer IDs
    subject = pd.read_csv(
        folder / f"subject_{split}.txt",
        sep=r"\s+",
        header=None
    )[0]

    return X, y, subject


def load_labels():
    """
    Load the mapping between activity IDs and activity names.
    """
    labels = pd.read_csv(
        DATA_DIR / "activity_labels.txt",
        sep=r"\s+",
        header=None,
        index_col=0
    )[1]

    return labels.to_dict()