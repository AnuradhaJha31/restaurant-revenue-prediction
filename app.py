import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# -----------------------------
# LOAD DATA
# -----------------------------
data = pd.read_csv("restaurant_data.csv")

# -----------------------------
# DROP NON-NUMERIC COLUMNS
# (Fixes your error)
# -----------------------------
for col in data.columns:
    if data[col].dtype == "object":
        data.drop(col, axis=1, inplace=True)

# -----------------------------
# SPLIT FEATURES & TARGET
# -----------------------------
X = data.drop("Revenue", axis=1)
y = data["Revenue"]

# -----------------------------
# TRAIN MODEL
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

rf = RandomForestRegressor(n_estimators=50, random_state=42)
rf.fit(X_train, y_train)

# -----------------------------
# STREAMLIT APP UI
# -----------------------------
st.title("Restaurant Revenue Prediction App")

meal_price = st.number_input("Average Meal Price")
marketing = st.number_input("Marketing Budget")
seating = st.number_input("Seating Capacity")
weekend = st.number_input("Weekend Reservations")

if st.button("Predict Revenue"):
    data_input = np.array([[meal_price, marketing, seating, weekend]])
    prediction = rf.predict(data_input)
    st.success(f"Predicted Revenue: {prediction[0]:.2f}")

