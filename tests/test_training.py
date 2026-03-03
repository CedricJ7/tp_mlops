# test_training.py
import pytest
import subprocess
import os
import joblib
from sklearn.pipeline import Pipeline


MODEL_PATH = "model_pipeline.joblib"


@pytest.fixture(scope="module", autouse=True)
def run_training():
    """Lance train.py et vérifie qu'il se termine sans erreur."""
    result = subprocess.run(
        ["python", "train.py"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0, (
        f"train.py a échoué (code {result.returncode}):\n{result.stderr}"
    )


def test_model_file_exists():
    assert os.path.exists(MODEL_PATH), f"{MODEL_PATH} non trouvé après entraînement"


def test_model_file_nonempty():
    assert os.path.getsize(MODEL_PATH) > 0, f"{MODEL_PATH} est vide"


def test_model_is_pipeline():
    pipeline = joblib.load(MODEL_PATH)
    assert isinstance(pipeline, Pipeline), "Le modèle chargé n'est pas un Pipeline sklearn"


def test_pipeline_has_preprocessor_and_model():
    pipeline = joblib.load(MODEL_PATH)
    assert "preprocessor" in pipeline.named_steps, "Étape 'preprocessor' manquante"
    assert "model" in pipeline.named_steps, "Étape 'model' manquante"


def test_pipeline_is_fitted():
    """Vérifie que le pipeline a bien été entraîné (attributs fitted présents)."""
    from sklearn.utils.validation import check_is_fitted
    pipeline = joblib.load(MODEL_PATH)
    check_is_fitted(pipeline["model"])
