import os

import pandas as pd
import pytest

DATA_PATH = "data/diabetes.csv"

pytestmark = pytest.mark.skipif(
    not os.path.exists(DATA_PATH),
    reason="dataset not found - run `python src/download_data.py` or `dvc pull`",
)


def test_dataset_shape():
    df = pd.read_csv(DATA_PATH)
    assert df.shape == (768, 9)


def test_dataset_columns_and_target():
    df = pd.read_csv(DATA_PATH)
    assert list(df.columns) == [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome",
    ]
    assert set(df["Outcome"].unique()) == {0, 1}