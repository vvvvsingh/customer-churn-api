import streamlit as st
import requests

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Churn Intelligence Dashboard")
st.markdown("Assess customer risk levels in real-time powered by our FastAPI machine learning service.")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Demographics")
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])

with col2:
    st.subheader("Account & Services")
    tenure = st.slider("Tenure (Months)", min_value=1, max_value=72, value=12)
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])

with col3:
    st.subheader("Contract & Billing")
    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox("Payment Method", [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ])
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=10.0, max_value=150.0, value=65.0, step=0.5)
    total_charges = st.number_input("Total Charges ($)", min_value=10.0, max_value=10000.0, value=780.0, step=5.0)

st.divider()

if st.button("Evaluate Churn Risk", type="primary", use_container_width=True):
    payload = {
        "gender": gender,
        "SeniorCitizen": int(senior),
        "Partner": partner,
        "Dependents": dependents,
        "tenure": int(tenure),
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": float(monthly_charges),
        "TotalCharges": float(total_charges)
    }

    try:
        response = requests.post("http://127.0.0.1:8000/predict", json=payload, timeout=5)
        
        if response.status_code == 200:
            result = response.json()
            prob = result["churn_probability"]
            risk = result["churn_risk"]

            res_col1, res_col2 = st.columns(2)
            with res_col1:
                if risk == "High":
                    st.error(f"### High Churn Risk (Prediction: {result['churn_prediction']})")
                else:
                    st.success(f"### Low Churn Risk (Prediction: {result['churn_prediction']})")
            
            with res_col2:
                st.metric("Estimated Churn Probability", f"{prob * 100:.1f}%")
        else:
            st.error(f"API Error ({response.status_code}): {response.text}")

    except requests.exceptions.ConnectionError:
        st.warning("Could not connect to FastAPI server. Ensure `uvicorn main:app --reload` is running on port 8000.")