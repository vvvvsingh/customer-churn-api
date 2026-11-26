# 📊 Customer Churn Intelligence & ML Inference System

An end-to-end Machine Learning operations (MLOps) project that predicts customer churn risk. This system features a serialized model pipeline, an asynchronous REST API built with FastAPI, and an interactive front-end dashboard powered by Streamlit.

---

## 📌 Table of Contents
1. [Project Overview & Business Context](#-project-overview--business-context)
2. [End-to-End Architecture](#-end-to-end-architecture)
3. [Model Training & Serialization (Colab Pipeline)](#-model-training--serialization-colab-pipeline)
4. [Backend Engineering (FastAPI)](#-backend-engineering-fastapi)
5. [Frontend Dashboard (Streamlit)](#-frontend-dashboard-streamlit)
6. [Key Engineering Hurdles & Troubleshooting Log](#-key-engineering-hurdles--troubleshooting-log)
7. [Repository File Structure](#-repository-file-structure)
8. [Local & Codespaces Execution Guide](#-local--codespaces-execution-guide)
9. [Future Roadmap & Production Hardening](#-future-roadmap--production-hardening)

---

## 🎯 Project Overview & Business Context

In telecommunications, retaining existing subscribers is significantly more cost-effective than acquiring new ones. This project simulates an enterprise churn-retention workflow:
* **Goal**: Predict whether an active customer will churn (cancel contract within the next billing cycle).
* **Input**: 19 demographic, service-subscription, and billing attributes.
* **Output**: Binary classification (`0` = Retained, `1` = Churned), human-readable risk category (`Low` vs. `High`), and an exact calibrated probability score ($P \in [0.0, 1.0]$).

---

## 🏗 End-to-End Architecture

```text
[ Raw Customer Data ]
         │
         ▼
[ Google Colab (Training) ] ──> Data Preprocessing & Alignment
         │                  ──> Model Training & Hyperparameter Tuning
         ▼
[ Serialized Artifacts ]
  ├── customer_churn_model.pkl (Trained Estimator + Feature Column List)
  └── encoders.pkl (Pre-fit Scikit-Learn LabelEncoders)
         │
         ▼
[ FastAPI Inference Engine (`main.py`) ] : Port 8000
  ├── Pydantic Input Validation (`CustomerPayload`)
  ├── Dynamic Feature Alignment
  └── `/predict` POST Endpoint
         ▲
         │ (HTTP / JSON Requests)
         ▼
[ Streamlit Client UI (`streamlit_app.py`) ] : Port 8501
  ├── Interactive Input Dashboard (Demographics, Services, Billing)
  └── Live Real-Time Risk & Probability Display
```

## 🧠 Model Training & Serialization (Colab Pipeline)

* **Dataset**: Telco Customer Churn dataset (~7,043 customer records).
* **Preprocessing**:
  * Categorical columns (`gender`, `Partner`, `InternetService`, `Contract`, etc.) were fitted via `LabelEncoder` and stored as a dictionary mapping (`encoders.pkl`).
  * Continuous features (`tenure`, `MonthlyCharges`, `TotalCharges`) were cast and aligned.
* **Model Packaging**:
  * The final model was packaged alongside its exact feature training order into `customer_churn_model.pkl`:
```python
{
    "model": trained_model_object,
    "features_names": list(X_train.columns)
}
```
## ⚙️ Backend Engineering (FastAPI)
Located in main.py:

Data Validation via Pydantic (CustomerPayload):

Validates every incoming request field, enforces static data typing (int, str, float), and supplies clear Swagger UI documentation examples.

Dynamic Preprocessing Pipeline:

Automatically checks incoming columns against the loaded encoders dictionary.

Re-indexes the inference DataFrame to strictly match features_names before scoring.

Inference Endpoint (/predict):

Returns JSON payload:

```JSON
{
  "churn_prediction": 0,
  "churn_risk": "Low",
  "churn_probability": 0.1200
}
```
Interactive Documentation: Auto-generates OpenAPI specs available at /docs (Swagger UI) and /redoc.

## 💻 Frontend Dashboard (Streamlit)
Located in streamlit_app.py:

Built with a 3-column modular interface:

Demographics: Gender, Senior Citizen status, Partner, Dependents.

Account & Services: Tenure slider (1–72 months), Phone, Fiber/DSL, Security, Support add-ons.

Contract & Billing: Contract structure, Paperless billing, Payment method, Charges inputs.

Decoupled architecture: The UI does not load the ML model into memory. Instead, it issues a standard POST request to http://127.0.0.1:8000/predict over HTTP, isolating client and server dependencies.

## 🛠 Key Engineering Hurdles & Troubleshooting Log
A revision log of real-world bugs encountered and resolved during implementation:

NotFittedError: need to call fit or load_model beforehand:

Cause: Re-instantiating an un-fitted model wrapper or cross-environment unpickling mismatch between notebook runtimes.

Resolution: Inspected Colab artifacts and relied on pre-compiled estimator bundles with saved weights.

AttributeError: 'dict' object has no attribute 'predict':

Cause: The pickle file customer_churn_model.pkl was saved as a nested dictionary ({"model": ..., "features_names": ...}) rather than a bare model instance.

Resolution: Extracted model = data["model"] and assigned feature_columns = data["features_names"].

Execution Ordering NameError: name 'feature_columns' is not defined:

Cause: Attempting .get("features_names", feature_columns) before the fallback list had been declared.

Resolution: Directly referenced the dynamically unpacked key data["features_names"], removing redundant hardcoded fallback lists.

Git Cache Pollution:

Cause: __pycache__ and compiled bytecode .pyc files appearing in git status.

Resolution: Initialized a clean .gitignore to omit Python cache and temporary experimental binaries.

## 📂 Repository File Structure
```text
customer-churn-api/
├── .gitignore               # Ignores __pycache__/, *.pyc, and local artifacts
├── README.md                # Project documentation and revision log
├── requirements.txt         # Pinned project dependencies
├── customer_churn_model.pkl # Pickled model object + feature column metadata
├── encoders.pkl             # Pickled categorical feature label encoders
├── main.py                  # FastAPI REST API implementation
└── streamlit_app.py         # Streamlit UI dashboard
```
## 🚀 Local & Codespaces Execution Guide
1. Environment Setup
Bash
```
# Clone the repository
git clone [https://github.com/vvvvsingh/customer-churn-api.git](https://github.com/vvvvsingh/customer-churn-api.git)
cd customer-churn-api

# Install dependencies
pip install -r requirements.txt
```
2. Launch FastAPI Backend
Bash
```
uvicorn main:app --port 8000 --reload
```
API Root: http://127.0.0.1:8000/

Swagger Interactive Docs: http://127.0.0.1:8000/docs

3. Launch Streamlit Frontend (In a Separate Terminal Tab)
Bash
```
streamlit run streamlit_app.py --server.port 8501
```
Dashboard URL: http://localhost:8501

## 🔮 Future Roadmap & Production Hardening
[ ] Dockerization: Create a multi-stage Dockerfile and docker-compose.yml to orchestrate both API and Streamlit containers simultaneously.

[ ] Cloud Deployment: Deploy the API on Render or AWS ECS and host the Streamlit UI on Streamlit Community Cloud or Hugging Face Spaces.

[ ] Model Observability: Add Prometheus middleware and Grafana for monitoring API latency, request throughput, and data drift.

[ ] CI/CD Pipeline: Add GitHub Actions workflow to run automated unit tests with pytest on every push.
