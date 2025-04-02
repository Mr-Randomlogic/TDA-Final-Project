import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and dataset
model = joblib.load("gb_model.pkl")
df = pd.read_csv("Superstore.csv")

# Create model feature list
df_model = df.drop(columns=["Country", "City", "Postal Code", "Ship Mode"])
df_encoded = pd.get_dummies(df_model, columns=["Category", "Sub-Category", "Segment", "State", "Region"], drop_first=True)
model_features = df_encoded.drop(columns=["Profit"]).columns.tolist()

# ------------------- Prediction Function -------------------
def simulate_profit(
    df,
    trained_model,
    model_features,
    category=None,
    region=None,
    sub_category=None,
    discount=0.15
):
    df_filtered = df.copy()

    # Filters
    if category:
        df_filtered = df_filtered[df_filtered["Category"] == category]
    if region:
        df_filtered = df_filtered[df_filtered["Region"] == region]
    if sub_category:
        df_filtered = df_filtered[df_filtered["Sub-Category"] == sub_category]

    if df_filtered.empty:
        return {"message": "No data matches the given filters."}

    # Apply cap
    df_filtered["Discount"] = df_filtered["Discount"].apply(lambda x: min(x, discount))

    # Drop extras
    df_filtered = df_filtered.drop(columns=["Country", "City", "Postal Code", "Ship Mode"])

    # One-hot encode
    df_encoded = pd.get_dummies(df_filtered, columns=["Category", "Sub-Category", "Segment", "State", "Region"], drop_first=True)

    # Fill missing columns
    for col in model_features:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    df_encoded = df_encoded[model_features]

    # Predict
    predicted = trained_model.predict(df_encoded)

    return {
        "Total Predicted Profit": round(predicted.sum(), 2),
        "Average Profit per Order": round(predicted.mean(), 2),
        "Num Orders": len(predicted)
    }

# ------------------- Streamlit UI -------------------
st.title("🛒 Superstore Profit Estimator")

# Selections
region = st.selectbox("Select Region", df["Region"].unique())
category = st.selectbox("Select Category", df[df["Region"] == region]["Category"].unique())
subcat = st.selectbox("Select Sub-Category", df[(df["Region"] == region) & (df["Category"] == category)]["Sub-Category"].unique())

discount_cap = st.slider("Max Discount Cap", 0.0, 0.5, 0.15, step=0.01)

# Prediction
results = simulate_profit(
    df=df,
    trained_model=model,
    model_features=model_features,
    region=region,
    category=category,
    sub_category=subcat,
    discount=discount_cap
)

# Results
st.subheader("📈 Predicted Profit")
if "message" in results:
    st.warning(results["message"])
else:
    st.metric("Total Predicted Profit", f"${results['Total Predicted Profit']:,.2f}")
    st.metric("Average Profit per Order", f"${results['Average Profit per Order']:,.2f}")
    st.metric("Number of Orders", results["Num Orders"])
