"""Evaluate the 44-feature raw-signal model on the UCI test set and save
reports/raw_model_metrics.json (read by the Model Lab tab in the app)."""
import json
from pathlib import Path

import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from raw_features import extract, load_raw, load_y

PROJECT_DIR = Path(__file__).resolve().parents[1]

bundle = joblib.load(PROJECT_DIR / "models" / "raw_model.joblib")
model, names = bundle["model"], bundle["label_names"]

X_test, y_test = extract(load_raw("test")), load_y("test")
pred = model.predict(X_test)

report = classification_report(y_test, pred, target_names=names, output_dict=True, zero_division=0)
metrics = {
    "model": f"Random Forest ({model.n_estimators} trees)",
    "features": int(X_test.shape[1]),
    "test_windows": int(len(y_test)),
    "accuracy": float(accuracy_score(y_test, pred)),
    "macro_f1": float(report["macro avg"]["f1-score"]),
    "labels": names,
    "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
    "per_class": {n: {k: float(report[n][k]) for k in ("precision", "recall", "f1-score", "support")} for n in names},
}

out = PROJECT_DIR / "reports" / "raw_model_metrics.json"
out.write_text(json.dumps(metrics, indent=2))
print(f"Accuracy: {metrics['accuracy']:.4f} | macro F1: {metrics['macro_f1']:.4f}")
print("Saved", out)
