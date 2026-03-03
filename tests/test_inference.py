# test_inference.py
import pytest
import joblib
import numpy as np
import pandas as pd


MODEL_PATH = "model_pipeline.joblib"

# Exemple d'entrée valide basé sur feature_schema.json
SAMPLE_INPUT = pd.DataFrame([{
    "Gender": "Male",
    "Age": 21,
    "Major": "Computer Science",
    "Attendance_Pct": 85.0,
    "Study_Hours_Per_Day": 4.0,
    "Previous_GPA": 3.2,
    "Sleep_Hours": 7.0,
    "Social_Hours_Week": 8,
}])


@pytest.fixture(scope="module")
def pipeline():
    return joblib.load(MODEL_PATH)


def test_predict_returns_array(pipeline):
    result = pipeline.predict(SAMPLE_INPUT)
    assert isinstance(result, np.ndarray), "predict() doit retourner un np.ndarray"


def test_predict_single_sample_is_scalar(pipeline):
    result = pipeline.predict(SAMPLE_INPUT)
    assert result.shape == (1,), f"Attendu shape (1,), obtenu {result.shape}"


def test_predict_output_is_float(pipeline):
    result = pipeline.predict(SAMPLE_INPUT)
    assert np.issubdtype(result.dtype, np.floating), "La prédiction doit être un float (régression)"


def test_predict_output_in_valid_range(pipeline):
    """Le CGPA doit être entre 0 et 4."""
    result = pipeline.predict(SAMPLE_INPUT)
    assert 0.0 <= result[0] <= 4.0, f"Prédiction hors plage [0, 4] : {result[0]}"


def test_predict_batch(pipeline):
    """Vérifie que predict fonctionne sur plusieurs lignes."""
    batch = pd.concat([SAMPLE_INPUT] * 5, ignore_index=True)
    result = pipeline.predict(batch)
    assert result.shape == (5,), f"Attendu shape (5,), obtenu {result.shape}"


def test_no_predict_proba(pipeline):
    """RandomForestRegressor ne doit pas avoir predict_proba."""
    assert not hasattr(pipeline, "predict_proba"), (
        "predict_proba inattendu sur un modèle de régression"
    )
