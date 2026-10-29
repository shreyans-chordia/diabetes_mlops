import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.train import (
    MODEL_CLASSES,
    impute_zeros,
    make_search,
    pick_winner,
    search_space_for,
    validate_config,
)


def test_impute_zeros_uses_train_median_and_leaves_inputs_alone():
    X_train = pd.DataFrame({"Glucose": [0, 100, 120], "Age": [30, 40, 50]})
    X_test = pd.DataFrame({"Glucose": [0, 90], "Age": [20, 60]})

    tr, te = impute_zeros(X_train, X_test, ["Glucose"])

    assert tr["Glucose"].tolist() == [110.0, 100.0, 120.0]  # median of 100 and 120
    assert te["Glucose"].tolist() == [110.0, 90.0]          # test uses TRAIN median
    assert tr["Age"].tolist() == [30, 40, 50]               # other columns untouched
    assert X_train.loc[0, "Glucose"] == 0                   # originals not modified


def test_pick_winner_uses_cv_mean():
    results = {
        "a": {"cv": {"f1": {"mean": 0.60}}},
        "b": {"cv": {"f1": {"mean": 0.65}}},
    }
    assert pick_winner(results, "f1") == "b"


def test_validate_config_rejects_unknown_metric():
    cfg = {"selection": {"metric": "bogus"}, "models": {"random_forest": {}}}
    with pytest.raises(ValueError):
        validate_config(cfg)



def test_logistic_regression_is_a_scaled_pipeline():
    model = MODEL_CLASSES["logistic_regression"](C=1.0, max_iter=100)
    assert model.steps[0][0] == "scaler"


def test_every_model_can_fit_and_predict_proba():
    # The API calls predict_proba, so every candidate must support it
    X = pd.DataFrame({"a": [0, 1, 2, 3, 4, 5, 6, 7], "b": [1, 0, 1, 0, 1, 0, 1, 0]})
    y = [0, 0, 0, 0, 1, 1, 1, 1]
    for name, build in MODEL_CLASSES.items():
        model = build()
        model.fit(X, y)
        assert model.predict_proba(X).shape == (8, 2), name



def test_search_space_for_prefixes_pipeline_params():
    space = {"C": [0.1, 1.0]}
    assert search_space_for("logistic_regression", space) == {"clf__C": [0.1, 1.0]}
    assert search_space_for("random_forest", space) == {"C": [0.1, 1.0]}


def test_make_search_caps_iterations_to_space_size():
    space = {"C": [0.1, 1.0, 10.0]}
    tuning_cfg = {"n_iter": 20, "inner_splits": 3, "seed": 42}
    search = make_search(LogisticRegression(), space, tuning_cfg, "f1")
    assert search.n_iter == 3


def test_validate_config_requires_search_space_when_tuning():
    cfg = {
        "selection": {"metric": "f1"},
        "models": {"random_forest": {}},
        "tuning": {"enabled": True, "search_spaces": {}},
    }
    with pytest.raises(ValueError):
        validate_config(cfg)