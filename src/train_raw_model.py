import joblib
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from data_loader import load_labels
from raw_features import load_raw, load_y, extract


# ==========================================
# 1. PROJECT PATHS
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

MODEL_DIR = PROJECT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "raw_model.joblib"


# ==========================================
# 2. LOAD ACTIVITY LABELS
# ==========================================

labels = load_labels()

activity_names = [
    labels[i] for i in sorted(labels)
]


# ==========================================
# 3. LOAD AND EXTRACT RAW FEATURES
# ==========================================

print("Loading raw sensor training data...")

X_train = extract(load_raw("train"))
y_train = load_y("train")

print("Loading raw sensor testing data...")

X_test = extract(load_raw("test"))
y_test = load_y("test")


print("\n--- DATASET INFORMATION ---")

print("Training features:", X_train.shape)
print("Training labels:", y_train.shape)

print("Testing features:", X_test.shape)
print("Testing labels:", y_test.shape)


# ==========================================
# 4. CREATE RANDOM FOREST MODEL
# ==========================================

print("\nInitializing Random Forest...")

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)


# ==========================================
# 5. TRAIN THE MODEL
# ==========================================

print("\nTraining Random Forest...")

model.fit(X_train, y_train)

print("Training completed successfully!")


# ==========================================
# 6. MAKE PREDICTIONS
# ==========================================

print("\nPredicting test activities...")

predictions = model.predict(X_test)


# ==========================================
# 7. EVALUATE THE MODEL
# ==========================================

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\n--- RAW FEATURE MODEL RESULTS ---")

print("Accuracy:", round(accuracy, 4))
print("Accuracy percentage:", round(accuracy * 100, 2), "%")


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        target_names=activity_names,
        zero_division=0
    )
)


print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        predictions
    )
)


# ==========================================
# 8. SAVE THE TRAINED MODEL
# ==========================================

model_bundle = {
    "model": model,
    "label_names": activity_names
}

joblib.dump(
    model_bundle,
    MODEL_PATH
)

print("\nModel saved successfully!")
print("Location:", MODEL_PATH)

print("\nRAW SENSOR MODEL TRAINING COMPLETED!")