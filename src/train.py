import json
import os
import shutil

import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from mlflow.tracking import MlflowClient
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

COLUMN_NAMES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome",
]
METRIC_NAMES = ["accuracy", "precision", "recall", "f1"]

def make_logistic_regression(**params):
    """Logistic regression needs scaled features; scaling is fitted inside each CV fold."""
    return Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(**params))])


# Model names allowed in params.yaml -> a callable that builds the estimator
MODEL_CLASSES = {
    "random_forest": RandomForestClassifier,
    "gradient_boosting": GradientBoostingClassifier,
    "logistic_regression": make_logistic_regression,
}


def load_config(path="params.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def validate_config(cfg):
    metric = cfg["selection"]["metric"]
    if metric not in METRIC_NAMES:
        raise ValueError(f"selection.metric must be one of {METRIC_NAMES}, got '{metric}'")
    unknown = [name for name in cfg["models"] if name not in MODEL_CLASSES]
    if unknown:
        raise ValueError(f"Unknown model(s) in params.yaml: {unknown}. Allowed: {list(MODEL_CLASSES)}")


def load_split(cfg):
    data = pd.read_csv(cfg["data"]["path"], header=0, names=COLUMN_NAMES)
    X = data.drop("Outcome", axis=1)
    y = data["Outcome"]
    return train_test_split(
        X, y, test_size=cfg["data"]["test_size"], random_state=cfg["data"]["split_seed"]
    )


def impute_zeros(X_train, X_test, columns):
    """Replace impossible zeros with train-set medians (no test leakage)."""
    X_train, X_test = X_train.copy(), X_test.copy()
    for col in columns:
        median = X_train.loc[X_train[col] != 0, col].median()
        X_train[col] = X_train[col].replace(0, median)
        X_test[col] = X_test[col].replace(0, median)
    return X_train, X_test


def test_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }


def cross_validate_model(model, X_train, y_train, n_splits, seed):
    """Stratified k-fold CV on the training split. Returns mean/std per metric."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    # Autolog would otherwise log every fold's fit; keep the experiment readable.
    mlflow.sklearn.autolog(disable=True)
    try:
        scores = cross_validate(model, X_train, y_train, cv=cv, scoring=METRIC_NAMES)
    finally:
        mlflow.sklearn.autolog()
    return {
        name: {
            "mean": float(scores[f"test_{name}"].mean()),
            "std": float(scores[f"test_{name}"].std()),
        }
        for name in METRIC_NAMES
    }


def pick_winner(results, metric):
    """Name of the model with the best cross-validated mean for `metric`."""
    return max(results, key=lambda name: results[name]["cv"][metric]["mean"])


def main():
    cfg = load_config()
    validate_config(cfg)

    metric = cfg["selection"]["metric"]
    use_imputation = cfg["data"]["use_imputation"]
    model_dir = cfg["export"]["model_dir"]
    n_splits = cfg["cv"]["n_splits"]
    cv_seed = cfg["cv"]["seed"]

    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])
    mlflow.sklearn.autolog()

    X_train, X_test, y_train, y_test = load_split(cfg)
    if use_imputation:
        X_train, X_test = impute_zeros(
            X_train, X_test, cfg["data"]["zero_invalid_columns"]
        )

    # ---------------- Train and evaluate every candidate ----------------
    results = {}
    for run_name, params in cfg["models"].items():
        model_cls = MODEL_CLASSES[run_name]
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_param("use_imputation", use_imputation)
            mlflow.log_param("cv_n_splits", n_splits)

            # 1) Cross-validation on the training split (used to pick the winner)
            cv_scores = cross_validate_model(
                model_cls(**params), X_train, y_train, n_splits, cv_seed
            )
            for name, s in cv_scores.items():
                mlflow.log_metric(f"cv_{name}_mean", s["mean"])
                mlflow.log_metric(f"cv_{name}_std", s["std"])

            # 2) Final fit on the full training split, scored once on the test split
            model = model_cls(**params)
            model.fit(X_train, y_train)
            t_scores = test_metrics(y_test, model.predict(X_test))
            for name, value in t_scores.items():
                mlflow.log_metric(f"test_{name}", value)

            results[run_name] = {
                "model": model,
                "run_id": run.info.run_id,
                "cv": cv_scores,
                "test": t_scores,
            }

    # ---------------- Compare and pick the winner ----------------
    print(f"\n=== Model comparison ({n_splits}-fold CV on train split; test split for reference) ===")
    for name, res in results.items():
        cv, t = res["cv"], res["test"]
        print(
            f"{name:<18} cv_f1={cv['f1']['mean']:.3f} (+/-{cv['f1']['std']:.3f})  "
            f"cv_recall={cv['recall']['mean']:.3f}  "
            f"test_f1={t['f1']:.3f}  test_recall={t['recall']:.3f}"
        )

    best_name = pick_winner(results, metric)
    best = results[best_name]
    print(f"\nBest model by CV {metric}: {best_name} ({best['cv'][metric]['mean']:.3f})")

    client = MlflowClient()
    for name, res in results.items():
        client.set_tag(res["run_id"], "selected_for_serving", str(name == best_name))

    # ---------------- Export the winner for serving ----------------
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
    mlflow.sklearn.save_model(best["model"], model_dir, input_example=X_test.iloc[:1])
    print(f"Saved '{best_name}' to ./{model_dir}/")

    # ---------------- Write metrics for DVC ----------------
    def summarize(res):
        return {
            "cv": {k: round(v["mean"], 4) for k, v in res["cv"].items()},
            "cv_std": {k: round(v["std"], 4) for k, v in res["cv"].items()},
            "test": {k: round(v, 4) for k, v in res["test"].items()},
        }

    summary = {
        "best_model": best_name,
        "selection_metric": metric,
        "selection_basis": "cv_mean",
        "use_imputation": use_imputation,
        "best": summarize(best),
        "models": {name: summarize(res) for name, res in results.items()},
    }
    with open("metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Wrote metrics.json")


if __name__ == "__main__":
    main()