"""Train the orientation-robust model and save models/robust_model.joblib
plus reports/robust_model_metrics.json (same layout as raw_model_metrics.json)."""
import json
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from data_loader import load_labels
from raw_features import load_raw, load_y
from robust_features import extract_robust

PROJECT_DIR = Path(__file__).resolve().parents[1]

labels = load_labels()
names = [labels[i] for i in sorted(labels)]

X_train, y_train = extract_robust(load_raw("train")), load_y("train")
X_test, y_test = extract_robust(load_raw("test")), load_y("test")

model = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
pred = model.predict(X_test)

report = classification_report(y_test, pred, target_names=names, output_dict=True, zero_division=0)
metrics = {
    "model": f"Random Forest ({model.n_estimators} trees, orientation-robust)",
    "features": int(X_test.shape[1]),
    "test_windows": int(len(y_test)),
    "accuracy": float(accuracy_score(y_test, pred)),
    "macro_f1": float(report["macro avg"]["f1-score"]),
    "labels": names,
    "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
    "per_class": {n: {k: float(report[n][k]) for k in ("precision", "recall", "f1-score", "support")} for n in names},
}

joblib.dump({"model": model, "label_names": names}, PROJECT_DIR / "models" / "robust_model.joblib")
(PROJECT_DIR / "reports" / "robust_model_metrics.json").write_text(json.dumps(metrics, indent=2))
print(f"Robust model accuracy: {metrics['accuracy']:.4f} | macro F1: {metrics['macro_f1']:.4f}")
