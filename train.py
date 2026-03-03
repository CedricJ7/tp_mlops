# train.py

#Ne séparez pas le prétraitement (preprocessing) de votre modèle. Utilisez un pipeline scikit-
# learn (ColumnTransformer, etc.).
# Pourquoi ? Cela rend l’inférence beaucoup plus simple. Le pipeline complet est sérialisé en une
# seule fois, évitant les fuites de données (data leakage) et les erreurs lors du passage en production

import pandas as pd
import numpy as np
# import ColumnTransformer, RandomForestRegressor, train_test_split, Pipeline, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import time
from sklearn.preprocessing import OneHotEncoder

start_time = time.time()

# Load data
train_data = pd.read_csv("data/data_train.csv")
# remove id column (first one)
train_data = train_data.drop(columns=[train_data.columns[0]])

# get num_col
num_col = train_data.select_dtypes(include=np.number).columns.tolist()
num_col.remove("Final_CGPA")  # remove target
cat_col = train_data.select_dtypes(exclude=np.number).columns.tolist()
print(f'num_col {num_col}')
print(f'cat_col {cat_col}')

print("Defining preprocessing pipeline...")
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_col),
        ('cat', OneHotEncoder(drop='first'), cat_col)
    ]
)

# Define pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(random_state=42))
])

# train and save model
TARGET = "Final_CGPA"
X = train_data.drop(columns=[TARGET])
y = train_data[TARGET]

# split already made don't need to split again, just fit on all data
print("Training model pipeline...")
pipeline.fit(X, y)

# Save model
print("Saving model pipeline...")
import joblib
joblib.dump(pipeline, "model_pipeline.joblib")

end_time = time.time()
print(f"Training completed in {end_time - start_time:.2f} seconds")
print(pipeline)
#print model param
print(joblib.load("model_pipeline.joblib").get_params())
    