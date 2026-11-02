from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from xgboost import XGBClassifier

from data_loader import load_split, load_labels


# ==========================================
# 1. PROJECT PATHS
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

FIGURES_DIR = PROJECT_DIR / "reports" / "figures"
MODELS_DIR = PROJECT_DIR / "models"
REPORTS_DIR = PROJECT_DIR / "reports"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. LOAD DATASET
# ==========================================

print("=" * 60)
print("HUMAN ACTIVITY RECOGNITION")
print("Loading dataset...")
print("=" * 60)

labels = load_labels()

# Keep activity names in numerical label order.
activity_names = [
    labels[i] for i in sorted(labels)
]

X_train, y_train, subjects_train = load_split("train")
X_test, y_test, subjects_test = load_split("test")

feature_names = list(X_train.columns)

# XGBoost expects classes numbered from 0 to 5.
# Original dataset labels are numbered from 1 to 6.
Xtr = X_train.to_numpy()
Xte = X_test.to_numpy()

ytr = y_train.to_numpy() - 1
yte = y_test.to_numpy() - 1

print("\nTraining samples:", Xtr.shape[0])
print("Testing samples:", Xte.shape[0])
print("Number of features:", Xtr.shape[1])
print("Number of activities:", len(activity_names))

print("\nActivities:")
for name in activity_names:
    print("-", name)


# ==========================================
# 3. DEFINE FIVE ML MODELS
# ==========================================

models = {
    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    ),

    "SVM": make_pipeline(
        StandardScaler(),
        SVC(
            kernel="rbf",
            C=10,
            gamma="scale"
        )
    ),

    "KNN": make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(
            n_neighbors=5
        )
    ),

    "XGBoost": XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
        eval_metric="mlogloss"
    )
}


# ==========================================
# 4. TRAIN AND EVALUATE EACH MODEL
# ==========================================

results = []
fitted_models = {}

class_ids = np.arange(len(activity_names))

for name, model in models.items():

    print("\n" + "=" * 60)
    print("Training:", name)
    print("=" * 60)

    start_time = time.time()

    # Train the model.
    model.fit(Xtr, ytr)

    # Predict activities in the unseen test set.
    predictions = model.predict(Xte)

    # Calculate evaluation metrics.
    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            yte,
            predictions,
            average="weighted",
            zero_division=0
        )
    )

    accuracy = accuracy_score(yte, predictions)

    elapsed = round(time.time() - start_time, 2)

    # Store results for comparison.
    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Time (seconds)": elapsed
    })

    fitted_models[name] = model

    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1-score :", round(f1, 4))
    print("Time     :", elapsed, "seconds")

    # Generate the confusion matrix.
    cm = confusion_matrix(
        yte,
        predictions,
        labels=class_ids
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=activity_names
    )

    display.plot(
        xticks_rotation=45,
        cmap="Blues",
        values_format="d"
    )

    plt.title(f"Confusion Matrix - {name}")
    plt.tight_layout()

    filename = name.replace(" ", "_")

    plt.savefig(
        FIGURES_DIR / f"cm_{filename}.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print("Confusion matrix saved.")


# ==========================================
# 5. COMPARE ALL FIVE MODELS
# ==========================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="F1",
    ascending=False
).reset_index(drop=True)

print("\n" + "=" * 80)
print("FINAL MODEL COMPARISON")
print("=" * 80)

print(
    results_df.to_string(
        index=False,
        formatters={
            "Accuracy": "{:.4f}".format,
            "Precision": "{:.4f}".format,
            "Recall": "{:.4f}".format,
            "F1": "{:.4f}".format
        }
    )
)

# Save comparison table.
results_df.to_csv(
    REPORTS_DIR / "model_comparison.csv",
    index=False
)

print("\nModel comparison saved to reports/model_comparison.csv")


# ==========================================
# 6. GENERATE MODEL COMPARISON GRAPH
# ==========================================

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1"
]

ax = results_df.set_index("Model")[metrics].plot(
    kind="bar",
    figsize=(11, 6)
)

ax.set_title("Performance Comparison of Five ML Models")
ax.set_xlabel("Machine Learning Model")
ax.set_ylabel("Score")
ax.set_ylim(0, 1.05)

plt.xticks(rotation=20, ha="right")
plt.legend(loc="lower right")
plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "model_comparison.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()

print("Model comparison graph saved.")


# ==========================================
# 7. SHOW THE BEST MODEL
# ==========================================

best_name = results_df.iloc[0]["Model"]
best_model = fitted_models[best_name]

print("\n" + "=" * 60)
print("BEST MODEL:", best_name)
print("=" * 60)

best_predictions = best_model.predict(Xte)

print(
    classification_report(
        yte,
        best_predictions,
        labels=class_ids,
        target_names=activity_names,
        zero_division=0
    )
)


# ==========================================
# 8. RANDOM FOREST FEATURE IMPORTANCE
# ==========================================

print("Generating feature importance graph...")

rf_model = fitted_models["Random Forest"]

importance = pd.Series(
    rf_model.feature_importances_,
    index=feature_names
)

top_features = importance.nlargest(20).sort_values()

plt.figure(figsize=(10, 8))

top_features.plot(kind="barh")

plt.title("Top 20 Important Features - Random Forest")
plt.xlabel("Feature Importance")
plt.ylabel("Feature")

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "feature_importance.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()

print("Feature importance graph saved.")


# ==========================================
# 9. SAVE THE BEST MODEL USING JOBLIB
# ==========================================

model_bundle = {
    "model": best_model,
    "label_names": activity_names
}

model_path = MODELS_DIR / "best_model.joblib"

joblib.dump(
    model_bundle,
    model_path
)

print("\nBest model saved to:", model_path)

print("\n" + "=" * 60)
print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 60)