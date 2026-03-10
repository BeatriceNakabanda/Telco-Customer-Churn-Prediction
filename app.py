import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Load model and feature info
# -----------------------------
model = joblib.load("outputs/models/rf_model.pkl")
feature_columns = joblib.load("outputs/models/feature_columns.pkl")  # list
feature_medians = joblib.load("outputs/models/feature_medians.pkl")  # dict of medians

st.title("Telco Customer Churn Prediction")
st.write("Enter customer details to estimate churn risk.")

# -----------------------------
# User input widgets
# -----------------------------
tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.number_input("Monthly Charges", 0.0, 500.0, 70.0)
total_charges = st.number_input("Total Charges", 0.0, 10000.0, 1000.0)

contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

# -----------------------------
# Build input dictionary with median/defaults
# -----------------------------
user_input = {}

for col in feature_columns:
    if col in ["tenure", "MonthlyCharges", "TotalCharges"]:
        if col == "tenure":
            user_input[col] = tenure
        elif col == "MonthlyCharges":
            user_input[col] = monthly_charges
        elif col == "TotalCharges":
            user_input[col] = total_charges
    elif col.startswith("Contract_"):
        user_input[col] = 1 if col == f"Contract_{contract}" else 0
    elif col.startswith("InternetService_"):
        user_input[col] = 1 if col == f"InternetService_{internet_service}" else 0
    else:
        user_input[col] = feature_medians.get(col, 0)

input_df = pd.DataFrame([user_input], columns=feature_columns)

# -----------------------------
# Predict churn
# -----------------------------
if st.button("Predict Churn Risk"):
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)[:, 1][0]

    st.subheader("Prediction Result")
    if prediction[0] == 1:
        st.error(f"⚠️ High churn risk ({probability:.2%})")
    else:
        st.success(f"✅ Low churn risk ({probability:.2%})")

    # -----------------------------
    # Feature Importance Chart
    # -----------------------------
    st.subheader("Top 10 Feature Importances")
    importance = pd.Series(model.feature_importances_, index=feature_columns)
    top_features = importance.sort_values(ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(8,5))
    top_features.sort_values().plot(kind='barh', ax=ax, color='skyblue')
    ax.set_xlabel("Importance Score")
    ax.set_title("Top Features Influencing Churn")
    st.pyplot(fig)

    # -----------------------------
    # Churn Probability Histogram
    # -----------------------------
    st.subheader("Churn Probability Distribution (Simulated)")

    num_samples = 500
    demo_df = pd.DataFrame([feature_medians]*num_samples)

    # Random variations for numeric features
    demo_df["tenure"] = np.random.randint(0, 72, size=num_samples)
    demo_df["MonthlyCharges"] = np.random.uniform(20, 120, size=num_samples)
    demo_df["TotalCharges"] = demo_df["MonthlyCharges"] * demo_df["tenure"]

    numeric_cols = [col for col in feature_columns if col in feature_medians]
    for col in numeric_cols:
        if col not in ["tenure", "MonthlyCharges", "TotalCharges"]:
            # Add +/- 20% random variation around the median
            median_val = feature_medians[col]
            demo_df[col] = median_val * np.random.uniform(0.8, 1.2, size=num_samples)

    # Ensure all categorical one-hot columns exist
    for col in feature_columns:
        if col not in demo_df.columns:
            demo_df[col] = 0

    # Reorder columns exactly as model expects
    demo_df = demo_df[feature_columns]

    # Predict probabilities
    demo_probs = model.predict_proba(demo_df)[:, 1]

    fig2, ax2 = plt.subplots(figsize=(8,5))
    sns.histplot(demo_probs, bins=30, kde=True, ax=ax2, color='salmon')
    ax2.set_xlabel("Churn Probability")
    ax2.set_ylabel("Number of Customers")
    ax2.set_title("Distribution of Churn Risk")
    st.pyplot(fig2)