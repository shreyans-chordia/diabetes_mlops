import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# ---------------- Config ----------------
USE_IMPUTATION = False          # True = replace impossible zeros with column medians
SELECTION_METRIC = "f1"         # metric used to pick the winner (logged as test_<name>)
MODEL_DIR = "model"             # folder the API loads from
ZERO_INVALID_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

# Candidate models: run_name -> (model class, hyperparameters)
CANDIDATES = {
    "random_forest": (
        RandomForestClassifier,
        {"n_estimators": 100, "max_depth": 5, "random_state": 42},
    ),
    "gradient_boosting": (
        GradientBoostingClassifier,
        {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3, "random_state": 42},
    ),
}

# ---------------- MLflow setup ----------------
mlflow.set_experiment("diabetes-prediction")
mlflow.sklearn.autolog()

# ---------------- Data ----------------
column_names = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
]
data = pd.read_csv("data/diabetes.csv", header=0, names=column_names)

X = data.drop("Outcome", axis=1)
y = data["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
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

for run_name, (model_cls, params) in CANDIDATES.items():
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_param("use_imputation", USE_IMPUTATION)

        model = model_cls(**params)
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