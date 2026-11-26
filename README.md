# Customer Churn Prediction & Intelligence System

An end-to-end Machine Learning pipeline and deployment service predicting customer churn risk using XGBoost, FastAPI, and Streamlit.

## Features
- **High-Performance Inference**: FastAPI REST API serving real-time predictions with Pydantic schema validation.
- **Dynamic Preprocessing**: Automatic handling of label encoders and training feature alignment.
- **Interactive UI**: Clean Streamlit dashboard for non-technical evaluation and scenario simulation.

## Tech Stack
- **Machine Learning**: Scikit-Learn, XGBoost, Pandas, Joblib
- **API Framework**: FastAPI, Uvicorn, Pydantic
- **Dashboard**: Streamlit

## Quickstart

### 1. Run the FastAPI Backend
```bash
uvicorn main:app --port 8000