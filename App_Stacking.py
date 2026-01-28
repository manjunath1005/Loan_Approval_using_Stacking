import streamlit as st
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.pipeline import Pipeline

# ---------------------------------
# PAGE CONFIG
# ---------------------------------
st.set_page_config(
    page_title="Smart Loan Approval System",
    page_icon="🎯",
    layout="wide"
)

# ---------------------------------
# TITLE & DESCRIPTION
# ---------------------------------
st.title("🎯 Smart Loan Approval System – Stacking Model")
st.markdown("""
This system uses a **Stacking Ensemble Machine Learning model**
to predict whether a loan will be approved by combining
multiple ML models for better decision making.
""")

# ---------------------------------
# LOAD DATA & TRAIN MODELS (ONCE)
# ---------------------------------
@st.cache_data
def train_models():
    df = pd.read_csv("train_u6lujuX_CVtuZ9i.csv")

    df.drop(columns=["Loan_ID"], inplace=True)

    categorical_cols = ['Gender','Married','Dependents','Self_Employed']
    for col in categorical_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)

    numerical_cols = ['LoanAmount','Loan_Amount_Term','Credit_History']
    for col in numerical_cols:
        df[col].fillna(df[col].median(), inplace=True)

    df['Loan_Status'] = df['Loan_Status'].map({'Y':1, 'N':0})
    df = pd.get_dummies(df, drop_first=True)

    X = df.drop("Loan_Status", axis=1)
    y = df["Loan_Status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Base Models
    lr = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(max_iter=2000))
    ])

    dt = DecisionTreeClassifier(max_depth=5)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train base models
    lr.fit(X_train, y_train)
    dt.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    # Stacking Model
    stack_model = StackingClassifier(
        estimators=[
            ('lr', lr),
            ('dt', dt),
            ('rf', rf)
        ],
        final_estimator=LogisticRegression(max_iter=2000),
        cv=5
    )

    stack_model.fit(X_train, y_train)

    return stack_model, lr, dt, rf, X.columns

stack_model, lr_model, dt_model, rf_model, feature_cols = train_models()

# ---------------------------------
# SIDEBAR INPUTS
# ---------------------------------
st.sidebar.header("🧾 Applicant Details")

app_income = st.sidebar.number_input("Applicant Income", min_value=0)
co_income = st.sidebar.number_input("Co-Applicant Income", min_value=0)
loan_amount = st.sidebar.number_input("Loan Amount", min_value=0)
loan_term = st.sidebar.number_input("Loan Amount Term (months)", min_value=0)

credit_history = st.sidebar.radio("Credit History", ["Yes", "No"])
employment = st.sidebar.selectbox("Employment Status", ["Salaried", "Self-Employed"])
property_area = st.sidebar.selectbox("Property Area", ["Urban", "Semi-Urban", "Rural"])

# ---------------------------------
# MODEL ARCHITECTURE DISPLAY
# ---------------------------------
st.subheader("🧠 Model Architecture (Stacking Ensemble)")

st.markdown("""
**Base Models**
- Logistic Regression  
- Decision Tree  
- Random Forest  

**Meta Model**
- Logistic Regression  

📌 Predictions from base models are combined and passed to the meta-model.
""")

# ---------------------------------
# PREDICTION
# ---------------------------------
if st.button("🔘 Check Loan Eligibility (Stacking Model)"):

    input_data = pd.DataFrame({
        "ApplicantIncome": [app_income],
        "CoapplicantIncome": [co_income],
        "LoanAmount": [loan_amount],
        "Loan_Amount_Term": [loan_term],
        "Credit_History": [1 if credit_history == "Yes" else 0],
        "Self_Employed_Yes": [1 if employment == "Self-Employed" else 0],
        "Property_Area_Semiurban": [1 if property_area == "Semi-Urban" else 0],
        "Property_Area_Urban": [1 if property_area == "Urban" else 0]
    })

    # Align columns
    for col in feature_cols:
        if col not in input_data.columns:
            input_data[col] = 0

    input_data = input_data[feature_cols]

    # Base model predictions
    lr_pred = lr_model.predict(input_data)[0]
    dt_pred = dt_model.predict(input_data)[0]
    rf_pred = rf_model.predict(input_data)[0]

    # Stacking prediction
    final_pred = stack_model.predict(input_data)[0]
    confidence = stack_model.predict_proba(input_data)[0][final_pred] * 100

    # ---------------------------------
    # OUTPUT
    # ---------------------------------
    st.subheader("📊 Prediction Results")

    if final_pred == 1:
        st.success("✅ **Loan Approved**")
    else:
        st.error("❌ **Loan Rejected**")

    st.markdown("### 📌 Base Model Predictions")
    st.write(f"**Logistic Regression:** {'Approved' if lr_pred == 1 else 'Rejected'}")
    st.write(f"**Decision Tree:** {'Approved' if dt_pred == 1 else 'Rejected'}")
    st.write(f"**Random Forest:** {'Approved' if rf_pred == 1 else 'Rejected'}")

    st.markdown("### 🧠 Final Stacking Decision")
    st.write(f"**Confidence Score:** `{confidence:.2f}%`")

    # ---------------------------------
    # BUSINESS EXPLANATION (MANDATORY)
    # ---------------------------------
    st.subheader("💼 Business Explanation")

    st.info(
        f"Based on income, credit history, employment status, and combined "
        f"predictions from multiple machine learning models, the applicant is "
        f"**{'likely' if final_pred == 1 else 'unlikely'} to repay the loan**. "
        f"Therefore, the stacking model predicts **loan "
        f"{'approval' if final_pred == 1 else 'rejection'}**."
    )
