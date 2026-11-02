import joblib

from pathlib import Path
from data_loader import load_split


# -----------------------------------------
# 1. Locate the project
# -----------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_DIR / "models" / "best_model.joblib"


# -----------------------------------------
# 2. Load the saved model
# -----------------------------------------

print("Loading saved model...")

bundle = joblib.load(MODEL_PATH)

model = bundle["model"]
label_names = bundle["label_names"]

print("Model loaded successfully!")
print("Model type:", type(model).__name__)


# -----------------------------------------
# 3. Load the test dataset
# -----------------------------------------

print("\nLoading test data...")

X_test, y_test, subjects_test = load_split("test")

print("Test samples:", len(X_test))


# -----------------------------------------
# 4. Predict selected samples
# -----------------------------------------

sample_indices = [0, 500, 1500, 2500]

print("\n--- ACTIVITY PREDICTIONS ---\n")

correct = 0

for index in sample_indices:

    # Get one test sample.
    sample = X_test.iloc[[index]]

    # Predict its activity.
    prediction = model.predict(sample)[0]

    # Convert the predicted class into its activity name.
    predicted_activity = label_names[int(prediction)]

    # Actual labels in the original dataset are numbered 1 to 6.
    actual_activity = label_names[int(y_test.iloc[index]) - 1]

    # Compare the prediction with the actual activity.
    is_correct = predicted_activity == actual_activity

    if is_correct:
        correct += 1

    print(f"Sample: {index}")
    print(f"Predicted activity: {predicted_activity}")
    print(f"Actual activity:    {actual_activity}")
    print(f"Correct prediction:  {is_correct}")
    print("-" * 40)


# -----------------------------------------
# 5. Display the sample results
# -----------------------------------------

print("\nCorrect predictions:", correct, "out of", len(sample_indices))

print("\nPrediction testing completed!")





