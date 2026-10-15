# Predictive ML Pipeline: Churn & Turnover Forecasting

Production-ready Machine Learning pipeline for **B2B Customer Churn** and **Employee Turnover** prediction. Built with modular Python architecture, **MLflow** experiment tracking, and a **FastAPI** real-time inference endpoint.

## Architecture & Features
- **Data Preprocessing & Feature Engineering:** Automated pipeline using `scikit-learn` (`ColumnTransformer`, `StandardScaler`, `OneHotEncoder`).
- **Model Training & Tracking:** Gradient Boosting classifier tracked via **MLflow** (metrics: ROC-AUC, F1-Score).
- **Real-Time Inference API:** REST API built with **FastAPI** and **Pydantic** validation for ERP/CRM integration.
