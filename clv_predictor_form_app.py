import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib

st.set_page_config(page_title="CLV Predictor", layout="centered")

st.title("Customer Lifetime Value (CLV) Predictor")

st.markdown("Enter the customer attributes below to predict their expected lifetime value.")

# Manual input fields
recency = st.number_input("Recency (days since last purchase)", min_value=0, max_value=365, value=30)
frequency = st.number_input("Frequency (number of purchases)", min_value=1, max_value=100, value=5)
cluster_labels = {
    "0 - Low Value (infrequent, low spenders)": 0,
    "1 - Lost or Dormant": 1,
    "2 - High Value (frequent + high spenders)": 2,
    "3 - Potential Loyalists": 3
}

cluster_label = st.selectbox("Customer Segment", options=list(cluster_labels.keys()))
cluster = cluster_labels[cluster_label]
avg_order_value = st.number_input("Average Order Value", min_value=0.0, max_value=10000.0, value=100.0)
days_active = st.number_input("Days Active (range between first and last purchase)", min_value=0, max_value=365, value=60)
quantity = st.number_input("Total Quantity Purchased", min_value=1, max_value=1000, value=20)

# Assemble input into DataFrame
input_df = pd.DataFrame({
    'Recency': [recency],
    'Frequency': [frequency],
    'Cluster': [cluster],
    'AvgOrderValue': [avg_order_value],
    'DaysActive': [days_active],
    'Quantity': [quantity]
})

# Load trained model
model = joblib.load("xgb_clv_model.pkl")

# Predict log CLV
log_clv_pred = model.predict(input_df)

# Convert back to actual currency value
predicted_clv = np.expm1(log_clv_pred)[0]

# Display result
st.subheader("Predicted Customer Lifetime Value:")
st.success(f"${predicted_clv:,.2f}")
