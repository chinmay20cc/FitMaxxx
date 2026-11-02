import numpy as np
import pandas as pd
import joblib

from pathlib import Path
from raw_features import extract


# ==========================================
# 1. PROJECT PATHS
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_DIR / "data" / "my_recordings" / "extracted"

MODEL_PATH = PROJECT_DIR / "models" / "raw_model.joblib"

ACC_PATH = DATA_DIR / "TotalAcceleration.csv"
GYRO_PATH = DATA_DIR / "Gyroscope.csv"


# ==========================================
# 2. CONFIGURATION
# ==========================================

FS = 50
WIN = 128
STEP = 64

G = 9.81


# ==========================================
# 3. LOAD SENSOR FILES
# ==========================================

print("Loading smartphone sensor recordings...")

acc = pd.read_csv(ACC_PATH)
gyro = pd.read_csv(GYRO_PATH)


required_columns = ["seconds_elapsed", "x", "y", "z"]

for name, df in [
    ("TotalAcceleration", acc),
    ("Gyroscope", gyro)
]:

    missing = set(required_columns) - set(df.columns)

    if missing:
        raise ValueError(
            f"{name}.csv is missing columns: {missing}"
        )

    df = df[required_columns].copy()

    df = df.apply(pd.to_numeric, errors="coerce")

    df.dropna(inplace=True)

    df.sort_values("seconds_elapsed", inplace=True)

    df.drop_duplicates(
        subset="seconds_elapsed",
        inplace=True
    )

    if len(df) < 2:
        raise ValueError(f"Insufficient data in {name}.csv")

    if name == "TotalAcceleration":
        acc = df
    else:
        gyro = df


print("Accelerometer readings:", len(acc))
print("Gyroscope readings:", len(gyro))


# ==========================================
# 4. RESAMPLE TO 50 HZ
# ==========================================

print("\nResampling sensor data to 50 Hz...")

start = max(
    acc["seconds_elapsed"].min(),
    gyro["seconds_elapsed"].min()
)

end = min(
    acc["seconds_elapsed"].max(),
    gyro["seconds_elapsed"].max()
)

if end <= start:
    raise ValueError("Sensor recording timestamps do not overlap.")

t = np.arange(start, end, 1 / FS)


def resample(df):

    return np.column_stack([
        np.interp(
            t,
            df["seconds_elapsed"],
            df[axis]
        )
        for axis in ["x", "y", "z"]
    ])


# Convert acceleration from m/s² to g.
acc_data = resample(acc) / G

gyro_data = resample(gyro)

# Combine six sensor channels.
signals = np.hstack([
    acc_data,
    gyro_data
])

print("Combined sensor shape:", signals.shape)


# ==========================================
# 5. CREATE SLIDING WINDOWS
# ==========================================

print("\nCreating 2.56-second windows...")

if len(signals) < WIN:
    raise ValueError(
        "Recording is too short for one 2.56-second window."
    )

windows = np.stack([
    signals[i:i + WIN]
    for i in range(0, len(signals) - WIN + 1, STEP)
])

print("Window shape:", windows.shape)


# ==========================================
# 6. EXTRACT 44 FEATURES
# ==========================================

print("\nExtracting statistical features...")

X_phone = extract(windows)

print("Extracted features:", X_phone.shape)


# ==========================================
# 7. LOAD TRAINED RANDOM FOREST
# ==========================================

print("\nLoading trained Random Forest model...")

bundle = joblib.load(MODEL_PATH)

model = bundle["model"]
labels = bundle["label_names"]


# ==========================================
# 8. PREDICT ACTIVITIES
# ==========================================

print("\nPredicting smartphone activities...")

predictions = model.predict(X_phone)

activity_names = [
    labels[int(prediction)]
    for prediction in predictions
]


# ==========================================
# 9. CREATE PREDICTION TIMELINE
# ==========================================

recording_start = start

results = pd.DataFrame({

    "Window_Start_Sec": [
        round(i * STEP / FS, 2)
        for i in range(len(predictions))
    ],

    "Window_End_Sec": [
        round((i * STEP + WIN) / FS, 2)
        for i in range(len(predictions))
    ],

    "Predicted_Activity": activity_names

})


# ==========================================
# 10. DISPLAY RESULTS
# ==========================================

print("\n" + "=" * 60)

print("SMARTPHONE ACTIVITY PREDICTIONS")

print("=" * 60)

print(results.to_string(index=False))


# ==========================================
# 11. ACTIVITY SUMMARY
# ==========================================

print("\n--- ACTIVITY SUMMARY ---")

print(
    results["Predicted_Activity"].value_counts()
)


# ==========================================
# 12. SAVE RESULTS
# ==========================================

output_path = (
    PROJECT_DIR / "reports" / "phone_predictions.csv"
)

results.to_csv(output_path, index=False)

print("\nPredictions saved successfully!")

print("Location:", output_path)

print("\nSMARTPHONE ACTIVITY PREDICTION COMPLETED!")