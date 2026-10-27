import os
import shutil

import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from mlflow.tracking import MlflowClient
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

# ---------------- Config ----------------
with open("params.yaml") as f:
    cfg = yaml.safe_load(f)

DATA_PATH = cfg["data"]["path"]
TEST_SIZE = cfg["data"]["test_size"]
SPLIT_SEED = cfg["data"]["split_seed"]
USE_IMPUTATION = cfg["data"]["use_imputation"]
ZERO_INVALID_COLS = cfg["data"]["zero_invalid_columns"]
SELECTION_METRIC = cfg["selection"]["metric"]
MODEL_DIR = cfg["export"]["model_dir"]
EXPERIMENT_NAME = cfg["mlflow"]["experiment_name"]
MODEL_PARAMS = cfg["models"]

# Model names allowed in params.yaml -> the class that implements them
MODEL_CLASSES = {
    "random_forest": RandomForestClassifier,
    "gradient_boosting": GradientBoostingClassifier,
}

VALID_METRICS = ["accuracy", "precision", "recall", "f1"]
if SELECTION_METRIC not in VALID_METRICS:
    raise ValueError(f"selection.metric must be one of {VALID_METRICS}, got '{SELECTION_METRIC}'")
unknown = [name for name in MODEL_PARAMS if name not in MODEL_CLASSES]
if unknown:
    raise ValueError(f"Unknown model(s) in params.yaml: {unknown}. Allowed: {list(MODEL_CLASSES)}")

# ---------------- MLflow setup ----------------
mlflow.set_experiment(EXPERIMENT_NAME)
mlflow.sklearn.autolog()

# ---------------- Data ----------------
column_names = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
]
data = pd.read_csv(DATA_PATH, header=0, names=column_names)

X = data.drop("Outcome", axis=1)
y = data["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=SPLIT_SEED
)
X_train, X_test = X_train.copy(), X_test.copy()

# Optional imputation: medians come from the TRAIN split only (no test leakage)
if USE_IMPUTATION:
    for col in ZERO_INVALID_COLS:
        median = X_train.loc[X_train[col] != 0, col].median()
        X_train[col] = X_train[col].replace(0, median)
        X_test[col] = X_test[col].replace(0, median)

# ---------------- Train and evaluate every candidate ----------------
results = {}

for run_name, params in MODEL_PARAMS.items():
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_param("use_imputation", USE_IMPUTATION)

        model = MODEL_CLASSES[run_name](**params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
        }
        for name, value in metrics.items():
            mlflow.log_metric(f"test_{name}", value)

        results[run_name] = {
            "model": model,
            "run_id": run.info.run_id,
            "metrics": metrics,
        }

# ---------------- Compare and pick the winner ----------------
print("\n=== Model comparison ===")
for run_name, res in results.items():
    m = res["metrics"]
    print(
        f"{run_name:<18} acc={m['accuracy']:.3f}  prec={m['precision']:.3f}  "
        f"rec={m['recall']:.3f}  f1={m['f1']:.3f}"
    )

best_name = max(results, key=lambda n: results[n]["metrics"][SELECTION_METRIC])
best = results[best_name]
print(f"\nBest model by {SELECTION_METRIC}: {best_name} "
      f"({best['metrics'][SELECTION_METRIC]:.3f})")

# Tag runs in MLflow so the winner is visible in the UI
client = MlflowClient()
for run_name, res in results.items():
    client.set_tag(res["run_id"], "selected_for_serving", str(run_name == best_name))

# ---------------- Export the winner for serving ----------------
if os.path.exists(MODEL_DIR):
    shutil.rmtree(MODEL_DIR)
mlflow.sklearn.save_model(best["model"], MODEL_DIR, input_example=X_test.iloc[:1])
print(f"Saved '{best_name}' to ./{MODEL_DIR}/")