import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("model/waste_model.pkl")
food_encoder = joblib.load("model/food_encoder.pkl")
status_encoder = joblib.load("model/status_encoder.pkl")

st.set_page_config(page_title="AI Food Waste Predictor", page_icon="🍽️")
st.markdown("""
<style>
.stApp {
    background-color: #1e293b;
    color: white;
}

h1, h2, h3, p, label {
    color: white !important;
}

div.stButton > button {
    background-color: #22c55e;
    color: white;
}
</style>
""", unsafe_allow_html=True)

st.title("🍽️ AI Food Waste Predictor")
st.markdown("""
---
### 📋 Food Waste Prediction System

This AI system predicts whether food is:

- 🟢 Fresh
- 🟡 Consume Soon
- 🔴 Waste

Fill in the details below and click **Predict**.
---
""")
st.write("Enter the food details to predict whether it is Fresh or Waste.")

# User Inputs
food = st.selectbox(
    "Select Food",
    food_encoder.classes_
)

quantity = st.number_input(
    "Quantity",
    min_value=1,
    value=1
)

storage_days = st.number_input(
    "Storage Days",
    min_value=0,
    value=1
)

temperature = st.number_input(
    "Storage Temperature (°C)",
    value=25
)

expiry_days = st.number_input(
    "Days Left to Expiry",
    min_value=0,
    value=3
)
if st.button("Predict"):

    # Encode food name
    food_encoded = food_encoder.transform([food])[0]

    # Create input dataframe
    input_data = pd.DataFrame([[
        food_encoded,
        quantity,
        storage_days,
        temperature,
        expiry_days
    ]], columns=[
        "Food",
        "Quantity",
        "Storage_Days",
        "Temperature",
        "Expiry_Days"
    ])

    # Predict
    prediction = model.predict(input_data)
    result = status_encoder.inverse_transform(prediction)[0]

    st.subheader("Prediction Result")

    if result == "Fresh":
        st.success("✅ Food is Fresh")
        st.info("💡 Tip: Store the food in a refrigerator and consume it before the expiry date.")

    elif result == "Consume Soon":
        st.warning("⚠️ Consume Soon")
        st.info("🍽️ Tip: Consume this food as soon as possible to avoid spoilage.")

    else:
        st.error("❌ Food is Waste")
        st.info("♻️ Tip: Do not consume this food. Dispose of it safely or compost it if possible.")
st.markdown("---")
st.caption("Developed by Hasini | B.E AI & DS | AI Food Waste Predictor")