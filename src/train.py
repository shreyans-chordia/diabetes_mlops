import json
import math
import os
import shutil

import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from mlflow.tracking import MlflowClient
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
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
    tuning = cfg.get("tuning", {})
    if tuning.get("enabled"):
        spaces = tuning.get("search_spaces", {})
        missing = [name for name in cfg["models"] if name not in spaces]
        if missing:
            raise ValueError(f"tuning is enabled but there is no search space for: {missing}")


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


def search_space_for(name, space):
    """Map a params.yaml search space onto the estimator's parameter names."""
    prefix = "clf__" if name == "logistic_regression" else ""
    return {prefix + key: values for key, values in space.items()}


def make_search(estimator, space, tuning_cfg, metric):
    """Randomized hyperparameter search that optimizes the selection metric."""
    n_combos = math.prod(len(values) for values in space.values())
    return RandomizedSearchCV(
        estimator,
        space,
        n_iter=min(tuning_cfg["n_iter"], n_combos),
        scoring=metric,
        cv=StratifiedKFold(
            n_splits=tuning_cfg["inner_splits"], shuffle=True, random_state=tuning_cfg["seed"]
        ),
        random_state=tuning_cfg["seed"],
        n_jobs=-1,
        refit=True,
    )


def cross_validate_model(model, X_train, y_train, n_splits, seed):
    """Stratified k-fold CV on the training split. Returns mean/std per metric.

    If `model` is a hyperparameter search, the search runs inside every fold,
    so this is nested CV and the score is not inflated by tuning.
    """
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
    tuning_cfg = cfg.get("tuning", {})
    tuning_on = bool(tuning_cfg.get("enabled", False))

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
        build = MODEL_CLASSES[run_name]
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_param("use_imputation", use_imputation)
            mlflow.log_param("cv_n_splits", n_splits)
            mlflow.log_param("tuned", tuning_on)

            if tuning_on:
                space = search_space_for(run_name, tuning_cfg["search_spaces"][run_name])
                candidate = make_search(build(**params), space, tuning_cfg, metric)
            else:
                candidate = build(**params)

            # 1) Cross-validation on the training split (used to pick the winner)
            cv_scores = cross_validate_model(candidate, X_train, y_train, n_splits, cv_seed)
            for name, s in cv_scores.items():
                mlflow.log_metric(f"cv_{name}_mean", s["mean"])
                mlflow.log_metric(f"cv_{name}_std", s["std"])

            # 2) Final fit (with tuning if enabled), scored once on the test split
            candidate.fit(X_train, y_train)
            if tuning_on:
                model = candidate.best_estimator_
                chosen = {k: v for k, v in candidate.best_params_.items()}
            else:
                model = candidate
                chosen = dict(params)

            t_scores = test_metrics(y_test, model.predict(X_test))
            for name, value in t_scores.items():
                mlflow.log_metric(f"test_{name}", value)

            results[run_name] = {
                "model": model,
                "run_id": run.info.run_id,
                "cv": cv_scores,
                "test": t_scores,
                "params": chosen,
            }

    # ---------------- Compare and pick the winner ----------------
    kind = "nested CV" if tuning_on else "CV"
    print(f"\n=== Model comparison ({n_splits}-fold {kind} on train split; test split for reference) ===")
    for name, res in results.items():
        cv, t = res["cv"], res["test"]
        print(
            f"{name:<20} cv_f1={cv['f1']['mean']:.3f} (+/-{cv['f1']['std']:.3f})  "
            f"cv_recall={cv['recall']['mean']:.3f}  "
            f"test_f1={t['f1']:.3f}  test_recall={t['recall']:.3f}"
        )

    best_name = pick_winner(results, metric)
    best = results[best_name]
    print(f"\nBest model by {kind} {metric}: {best_name} ({best['cv'][metric]['mean']:.3f})")
    print(f"Chosen parameters: {best['params']}")

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
            "params": res["params"],
        }

    summary = {
        "best_model": best_name,
        "selection_metric": metric,
        "selection_basis": "nested_cv_mean" if tuning_on else "cv_mean",
        "tuned": tuning_on,
        "use_imputation": use_imputation,
        "best": summarize(best),
        "models": {name: summarize(res) for name, res in results.items()},
    }
    with open("metrics.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print("Wrote metrics.json")


if __name__ == "__main__":
    main()