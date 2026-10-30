import yaml

ALLOWED_METRICS = ["accuracy", "precision", "recall", "f1","roc_auc"]
ALLOWED_MODELS = ["random_forest", "gradient_boosting","logistic_regression"]


def load_params():
    with open("params.yaml") as f:
        return yaml.safe_load(f)


def test_params_has_required_sections():
    cfg = load_params()
    for section in ["data", "selection", "export", "mlflow", "models", "cv", "tuning", "threshold"]:
        assert section in cfg, f"missing section: {section}"


def test_selection_metric_is_valid():
    assert load_params()["selection"]["metric"] in ALLOWED_METRICS


def test_test_size_is_a_fraction():
    assert 0 < load_params()["data"]["test_size"] < 1


def test_models_are_known_and_nonempty():
    models = load_params()["models"]
    assert len(models) >= 1
    assert set(models) <= set(ALLOWED_MODELS)


def test_imputation_columns_exist_in_dataset_schema():
    schema = {"Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
              "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"}
    assert set(load_params()["data"]["zero_invalid_columns"]) <= schema