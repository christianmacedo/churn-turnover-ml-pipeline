import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, f1_score

app = FastAPI(title="Churn & Turnover Prediction Pipeline", version="1.0.0")

class PredictionPayload(BaseModel):
    tenure_months: int
    monthly_contract_value: float
    support_tickets_90d: int
    usage_drop_pct: float
    segment: str

def build_and_train_pipeline(df: pd.DataFrame):
    X = df.drop("target", axis=1)
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), ["tenure_months", "monthly_contract_value", "support_tickets_90d", "usage_drop_pct"]),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["segment"])
    ])

    model_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=4, random_state=42))
    ])

    mlflow.set_experiment("enterprise_churn_turnover")
    with mlflow.start_run():
        model_pipeline.fit(X_train, y_train)
        probs = model_pipeline.predict_proba(X_test)[:, 1]
        preds = model_pipeline.predict(X_test)
        mlflow.log_metric("roc_auc", roc_auc_score(y_test, probs))
        mlflow.log_metric("f1_score", f1_score(y_test, preds))
        mlflow.sklearn.log_model(model_pipeline, "model")

    return model_pipeline
