# optimize.py
# Cherche les meilleurs hyperparamètres du pipeline en utilisant le jeu de validation.
# Une fois satisfait, relancer train.py avec les meilleurs params puis eval_performance.py.

import pandas as pd
import numpy as np
import joblib
import json
import time
from datetime import datetime
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

start_time = time.time()

TARGET = "Final_CGPA"

# Load data
train_data = pd.read_csv("data/data_train.csv")
train_data = train_data.drop(columns=[train_data.columns[0]])
valid_data = pd.read_csv("data/data_valid.csv")
valid_data = valid_data.drop(columns=[valid_data.columns[0]])

# Colonnes
num_col = train_data.select_dtypes(include=np.number).columns.tolist()
num_col.remove(TARGET)
cat_col = train_data.select_dtypes(exclude=np.number).columns.tolist()

X_train = train_data.drop(columns=[TARGET])
y_train = train_data[TARGET]
X_valid = valid_data.drop(columns=[TARGET])
y_valid = valid_data[TARGET]

# Pipeline de base (même structure que train.py)
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), num_col),
    ('cat', OneHotEncoder(drop='first'), cat_col)
])

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(random_state=42))
])

# Grille d'hyperparamètres à tester
param_grid = {
    'model__n_estimators': [100, 200, 300],
    'model__max_depth': [None, 10, 20],
    'model__min_samples_split': [2, 5, 10],
}

print(f"Grille de recherche : {sum(len(v) for v in param_grid.values())} valeurs, "
      f"{len(list(__import__('itertools').product(*param_grid.values())))} combinaisons")
print("Lancement du GridSearchCV (scoring=MAE, cv=5)...")

search = GridSearchCV(
    pipeline,
    param_grid,
    scoring='neg_mean_absolute_error',
    cv=5,
    n_jobs=-1,
    verbose=1,
    refit=True  # réentraîne sur tout X_train avec les meilleurs params
)

search.fit(X_train, y_train)

# Évaluation sur validation
best_pipeline = search.best_estimator_
y_pred_valid = best_pipeline.predict(X_valid)
mae_valid = mean_absolute_error(y_valid, y_pred_valid)

print("\n=== Résultats ===" )
print(f"Meilleurs paramètres : {search.best_params_}")
print(f"MAE CV (train)       : {-search.best_score_:.4f}")
print(f"MAE (validation)     : {mae_valid:.4f}")

# Sauvegarde du meilleur pipeline
joblib.dump(best_pipeline, "model_pipeline.joblib")
print("\nMeilleur pipeline sauvegardé dans model_pipeline.joblib")

elapsed = time.time() - start_time

# Sauvegarde des métriques
metrics = {
    "timestamp": datetime.now().isoformat(),
    "duration_seconds": round(elapsed, 2),
    "best_params": search.best_params_,
    "metrics": {
        "mae_cv_train": round(-search.best_score_, 4),
        "mae_valid": round(mae_valid, 4),
    }
}

metrics_path = "metrics.json"
try:
    with open(metrics_path, "r") as f:
        history = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    history = []

history.append(metrics)

with open(metrics_path, "w") as f:
    json.dump(history, f, indent=2)

print(f"Métriques sauvegardées dans {metrics_path}")
print(f"Terminé en {elapsed:.2f} secondes")
