# api.py
# Lancer avec : uvicorn api:app --reload
# Docs interactives : http://localhost:8000/docs

import json
from contextlib import asynccontextmanager
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Chargement unique du modèle et du schéma au démarrage
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pipeline = joblib.load("model_pipeline.joblib")
    with open("feature_schema.json") as f:
        app.state.schema = json.load(f)
    yield


app = FastAPI(
    title="Student CGPA Predictor",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Schéma d'entrée — Pydantic génère automatiquement l'erreur 422
# ---------------------------------------------------------------------------

class StudentFeatures(BaseModel):
    Gender: Literal["Male", "Female"]
    Age: int = Field(..., ge=18, le=24)
    Major: Literal["Economics", "Business", "Mathematics", "Psychology", "Engineering", "Computer Science"]
    Attendance_Pct: float = Field(..., ge=0.0, le=100.0)
    Study_Hours_Per_Day: float = Field(..., ge=0.0, le=24.0)
    Previous_GPA: float = Field(..., ge=0.0, le=4.0)
    Sleep_Hours: float = Field(..., ge=0.0, le=24.0)
    Social_Hours_Week: int = Field(..., ge=0, le=168)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    """Vérifie que l'API est vivante et que le modèle est chargé."""
    return {
        "status": "ok",
        "model_loaded": app.state.pipeline is not None,
    }


@app.get("/metadata")
def metadata():
    """Retourne la version du modèle, le type de tâche et les variables attendues."""
    schema = app.state.schema
    return {
        "model_version": app.version,
        "task": "regression",
        "target": schema["target"],
        "features": schema["features"],
    }


@app.post("/predict")
def predict(data: StudentFeatures):
    """Reçoit les données d'un étudiant et retourne le CGPA prédit."""
    df = pd.DataFrame([data.model_dump()])
    prediction = app.state.pipeline.predict(df)[0]
    return {
        "predicted_Final_CGPA": round(float(prediction), 4)
    }
