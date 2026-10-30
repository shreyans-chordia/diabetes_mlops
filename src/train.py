import json
import math
import os
import shutil
from contextlib import contextmanager

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from mlflow.tracking import MlflowClient
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

COLUMN_NAMES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome",
]
METRIC_NAMES = ["accuracy", "precision", "recall", "f1", "roc_auc"]
THRESHOLDS = np.round(np.arange(0.10, 0.901, 0.05), 2)


def make_logistic_regression(**params):
    """Logistic regression needs scaled features; scaling is fitted inside each CV fold."""
    return Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(**params))])


# Model names allowed in params.yaml -> a callable that builds the estimator
MODEL_CLASSES = {
    "random_forest": RandomForestClassifier,
    "gradient_boosting": GradientBoostingClassifier,
    "logistic_regression": make_logistic_regression,
}


@contextmanager
def autolog_paused():
    """Autolog would otherwise log every internal fit as its own MLflow run."""
    mlflow.sklearn.autolog(disable=True)
    try:
        yield
    finally:
        mlflow.sklearn.autolog()


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
    target = cfg["threshold"]["target_recall"]
    if not 0 < target <= 1:
        raise ValueError(f"threshold.target_recall must be in (0, 1], got {target}")


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


def test_metrics(y_true, y_pred, y_score):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_score),
    }


def confusion_counts(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def threshold_table(y_true, proba, thresholds):
    """Precision / recall / F1 when predicting 'diabetic' at proba >= threshold."""
    rows = []
    for t in thresholds:
        pred = (np.asarray(proba) >= t).astype(int)
        rows.append({
            "threshold": round(float(t), 2),
            "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_true, pred, zero_division=0)), 4),
            "f1": round(float(f1_score(y_true, pred, zero_division=0)), 4),
        })
    return rows


def choose_threshold(rows, target_recall):
    """Highest-precision row that still reaches target_recall (fallback: best recall)."""
    reached = [r for r in rows if r["recall"] >= target_recall]
    if reached:
        return max(reached, key=lambda r: (r["precision"], r["threshold"]))
    return max(rows, key=lambda r: (r["recall"], r["threshold"]))


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
    with autolog_paused():
        scores = cross_validate(model, X_train, y_train, cv=cv, scoring=METRIC_NAMES)
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
    target_recall = cfg["threshold"]["target_recall"]
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
                chosen_params = dict(candidate.best_params_)
            else:
                model = candidate
                chosen_params = dict(params)

            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            t_scores = test_metrics(y_test, y_pred, y_proba)
            for name, value in t_scores.items():
                mlflow.log_metric(f"test_{name}", value)

            cm = confusion_counts(y_test, y_pred)
            mlflow.log_dict(cm, "confusion_matrix.json")

            results[run_name] = {
                "model": model,
                "run_id": run.info.run_id,
                "cv": cv_scores,
                "test": t_scores,
                "params": chosen_params,
                "confusion_matrix": cm,
            }

    # ---------------- Compare and pick the winner ----------------
    kind = "nested CV" if tuning_on else "CV"
    print(f"\n=== Model comparison ({n_splits}-fold {kind} on train split; test split for reference) ===")
    for name, res in results.items():
        cv, t = res["cv"], res["test"]
        print(
            f"{name:<20} cv_f1={cv['f1']['mean']:.3f} (+/-{cv['f1']['std']:.3f})  "
            f"cv_recall={cv['recall']['mean']:.3f}  cv_auc={cv['roc_auc']['mean']:.3f}  |  "
            f"test_f1={t['f1']:.3f}  test_recall={t['recall']:.3f}  test_auc={t['roc_auc']:.3f}"
        )

    best_name = pick_winner(results, metric)
    best = results[best_name]
    print(f"\nBest model by {kind} {metric}: {best_name} ({best['cv'][metric]['mean']:.3f})")
    print(f"Chosen parameters: {best['params']}")

    client = MlflowClient()
    for name, res in results.items():
        client.set_tag(res["run_id"], "selected_for_serving", str(name == best_name))

    # ---------------- Threshold analysis for the winner ----------------
    # Out-of-fold probabilities on the TRAIN split choose the threshold;
    # the test split is used once, to report the result.
    with autolog_paused():
        oof_proba = cross_val_predict(
            clone(best["model"]),
            X_train,
            y_train,
            cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cv_seed),
            method="predict_proba",
        )[:, 1]
    rows = threshold_table(y_train, oof_proba, THRESHOLDS)
    chosen = choose_threshold(rows, target_recall)
    reached = chosen["recall"] >= target_recall

    test_proba = best["model"].predict_proba(X_test)[:, 1]
    test_at = threshold_table(y_test, test_proba, [chosen["threshold"]])[0]
    test_cm_at = confusion_counts(y_test, (test_proba >= chosen["threshold"]).astype(int))

    print(f"\n=== Threshold analysis for {best_name} (out-of-fold, train split) ===")
    print("threshold  precision  recall     f1")
    for r in rows:
        marker = "  <-- chosen" if r["threshold"] == chosen["threshold"] else ""
        print(f"{r['threshold']:>9.2f}  {r['precision']:>9.3f}  {r['recall']:>6.3f}  {r['f1']:>5.3f}{marker}")
    note = "" if reached else " (target NOT reached; showing the best recall available)"
    print(f"\nTarget recall {target_recall}: chosen threshold {chosen['threshold']:.2f}{note}")
    print(
        f"Test split at that threshold: precision={test_at['precision']:.3f}  "
        f"recall={test_at['recall']:.3f}  f1={test_at['f1']:.3f}  confusion={test_cm_at}"
    )

    with mlflow.start_run(run_id=best["run_id"]):
        mlflow.log_dict({"rows": rows}, "threshold_analysis.json")
        mlflow.log_metric("chosen_threshold", chosen["threshold"])
        mlflow.log_metric("test_recall_at_threshold", test_at["recall"])
        mlflow.log_metric("test_precision_at_threshold", test_at["precision"])

    # ---------------- Export the winner for serving ----------------
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
    mlflow.sklearn.save_model(best["model"], model_dir, input_example=X_test.iloc[:1])
    print(f"\nSaved '{best_name}' to ./{model_dir}/")

    # ---------------- Write metrics for DVC ----------------
    def summarize(res):
        return {
            "cv": {k: round(v["mean"], 4) for k, v in res["cv"].items()},
            "cv_std": {k: round(v["std"], 4) for k, v in res["cv"].items()},
            "test": {k: round(v, 4) for k, v in res["test"].items()},
            "confusion_matrix": res["confusion_matrix"],
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
        "threshold_analysis": {
            "target_recall": target_recall,
            "target_reached": bool(reached),
            "chosen_threshold": chosen["threshold"],
            "oof_at_threshold": chosen,
            "test_at_threshold": test_at,
            "test_confusion_at_threshold": test_cm_at,
        },
    }
    with open("metrics.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print("Wrote metrics.json")


if __name__ == "__main__":
    main()