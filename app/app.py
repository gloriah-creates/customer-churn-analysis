import streamlit as st
import pandas as pd
import joblib
import os


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# LOAD MODEL
# ============================================================

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "src",
    "churn_model.pkl"
)

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(
        "Unable to load the churn prediction model."
    )
    st.exception(e)
    st.stop()


# ============================================================
# TITLE
# ============================================================

st.title("📊 Customer Churn Predictor")

st.write(
    """
    Enter the customer's information below to estimate
    their likelihood of churning.
    """
)

st.divider()


# ============================================================
# CUSTOMER INFORMATION
# ============================================================

st.subheader("Customer Information")

col1, col2, col3 = st.columns(3)


with col1:

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    senior_citizen = st.selectbox(
        "Senior Citizen",
        [0, 1],
        format_func=lambda x:
            "Yes" if x == 1 else "No"
    )

    partner = st.selectbox(
        "Partner",
        ["Yes", "No"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["Yes", "No"]
    )

    tenure = st.number_input(
        "Tenure (months)",
        min_value=0,
        max_value=72,
        value=12,
        step=1
    )


with col2:

    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )

    multiple_lines = st.selectbox(
        "Multiple Lines",
        [
            "Yes",
            "No",
            "No phone service"
        ]
    )

    internet_service = st.selectbox(
        "Internet Service",
        [
            "DSL",
            "Fiber optic",
            "No"
        ]
    )

    contract = st.selectbox(
        "Contract",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    )

    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )


with col3:

    monthly_charges = st.number_input(
        "Monthly Charges ($)",
        min_value=18.25,
        max_value=118.75,
        value=70.00,
        step=1.00
    )

    total_charges = st.number_input(
        "Total Charges ($)",
        min_value=0.0,
        max_value=10000.0,
        value=840.0,
        step=10.0
    )

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    online_security = st.selectbox(
        "Online Security",
        [
            "Yes",
            "No",
            "No internet service"
        ]
    )

    online_backup = st.selectbox(
        "Online Backup",
        [
            "Yes",
            "No",
            "No internet service"
        ]
    )


# ============================================================
# ADDITIONAL SERVICES
# ============================================================

st.divider()

st.subheader("Support & Additional Services")

service_col1, service_col2, service_col3 = st.columns(3)


with service_col1:

    device_protection = st.selectbox(
        "Device Protection",
        [
            "Yes",
            "No",
            "No internet service"
        ]
    )

    tech_support = st.selectbox(
        "Tech Support",
        [
            "Yes",
            "No",
            "No internet service"
        ]
    )


with service_col2:

    streaming_tv = st.selectbox(
        "Streaming TV",
        [
            "Yes",
            "No",
            "No internet service"
        ]
    )


with service_col3:

    streaming_movies = st.selectbox(
        "Streaming Movies",
        [
            "Yes",
            "No",
            "No internet service"
        ]
    )


# ============================================================
# PREDICTION
# ============================================================

st.divider()

predict_button = st.button(
    "🔍 Predict Churn",
    type="primary",
    use_container_width=True
)


if predict_button:

    # --------------------------------------------------------
    # Create customer dataframe
    # --------------------------------------------------------

    customer = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [senior_citizen],
        "Partner": [partner],
        "Dependents": [dependents],
        "tenure": [tenure],
        "PhoneService": [phone_service],
        "MultipleLines": [multiple_lines],
        "InternetService": [internet_service],
        "OnlineSecurity": [online_security],
        "OnlineBackup": [online_backup],
        "DeviceProtection": [device_protection],
        "TechSupport": [tech_support],
        "StreamingTV": [streaming_tv],
        "StreamingMovies": [streaming_movies],
        "Contract": [contract],
        "PaperlessBilling": [paperless_billing],
        "PaymentMethod": [payment_method],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges]
    })


    # --------------------------------------------------------
    # Generate prediction
    # --------------------------------------------------------

    probability = model.predict_proba(
        customer
    )[0][1]

    prediction = model.predict(
        customer
    )[0]


    probability_percent = probability * 100


    # --------------------------------------------------------
    # Display result
    # --------------------------------------------------------

    st.divider()

    st.subheader("Prediction Result")


    if prediction == 1:

        st.error(
            "⚠️ Customer predicted to churn"
        )

    else:

        st.success(
            "✅ Customer predicted to stay"
        )


    result_col1, result_col2 = st.columns(2)


    with result_col1:

        st.metric(
            "Churn Probability",
            f"{probability_percent:.1f}%"
        )


    with result_col2:

        if probability >= 0.70:

            risk_level = "High Risk"

        elif probability >= 0.40:

            risk_level = "Medium Risk"

        else:

            risk_level = "Low Risk"


        st.metric(
            "Risk Level",
            risk_level
        )


    # --------------------------------------------------------
    # Business interpretation
    # --------------------------------------------------------

    st.write("### What this means")

    if probability >= 0.70:

        st.warning(
            """
            This customer has a high predicted likelihood of
            churning. A retention intervention may be worth
            considering.
            """
        )

    elif probability >= 0.40:

        st.info(
            """
            This customer has a moderate predicted likelihood
            of churning. Consider monitoring the customer and
            evaluating possible retention opportunities.
            """
        )

    else:

        st.success(
            """
            This customer has a relatively low predicted
            likelihood of churning based on the model.
            """
        )


    # --------------------------------------------------------
    # Show submitted information
    # --------------------------------------------------------

    with st.expander("View customer information"):

        st.dataframe(
            customer.T.rename(
                columns={0: "Value"}
            ),
            use_container_width=True
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    """
    Customer Churn Prediction Model | 
    Logistic Regression | 
    Telco Customer Churn Dataset
    """
)
