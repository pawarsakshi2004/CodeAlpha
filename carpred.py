# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =====================================================
# TRAIN MODEL (ONLY FIRST TIME)
# =====================================================

MODEL_FILE = "car_price_model.pkl"
DATA_FILE = "car data.csv"

if not os.path.exists(MODEL_FILE):

    df = pd.read_csv(DATA_FILE)

    # Feature Engineering
    current_year = datetime.now().year
    df["Car_Age"] = current_year - df["Year"]

    # Remove unnecessary column
    df.drop(["Car_Name", "Year"], axis=1, inplace=True)

    # Features and Target
    X = df.drop("Selling_Price", axis=1)
    y = df["Selling_Price"]

    categorical_features = [
        "Fuel_Type",
        "Selling_type",
        "Transmission"
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features
            )
        ],
        remainder="passthrough"
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=200,
            random_state=42
        ))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("\nModel Evaluation")
    print("-----------------------")
    print("MAE :", round(mae, 3))
    print("RMSE:", round(rmse, 3))
    print("R²  :", round(r2, 3))

    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)

# =====================================================
# LOAD MODEL
# =====================================================

with open(MODEL_FILE, "rb") as f:
    model = pickle.load(f)

# =====================================================
# STREAMLIT UI
# =====================================================

st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Car Price Prediction Using Machine Learning")
st.write("Predict the selling price of a used car based on its features.")

st.markdown("---")

col1, col2 = st.columns(2)

with col1:

    present_price = st.number_input(
        "Present Price (Lakhs)",
        min_value=0.0,
        max_value=100.0,
        value=5.0
    )

    driven_kms = st.number_input(
        "Driven Kilometers",
        min_value=0,
        value=20000
    )

    owner = st.selectbox(
        "Number of Previous Owners",
        [0, 1, 2, 3]
    )

with col2:

    year = st.slider(
        "Manufacturing Year",
        2000,
        datetime.now().year,
        2018
    )

    fuel_type = st.selectbox(
        "Fuel Type",
        ["Petrol", "Diesel", "CNG"]
    )

    selling_type = st.selectbox(
        "Seller Type",
        ["Dealer", "Individual"]
    )

    transmission = st.selectbox(
        "Transmission",
        ["Manual", "Automatic"]
    )

car_age = datetime.now().year - year

st.write(f"### Car Age: {car_age} Years")

# =====================================================
# PREDICTION
# =====================================================

if st.button("Predict Price"):

    input_df = pd.DataFrame({
        "Present_Price": [present_price],
        "Driven_kms": [driven_kms],
        "Fuel_Type": [fuel_type],
        "Selling_type": [selling_type],
        "Transmission": [transmission],
        "Owner": [owner],
        "Car_Age": [car_age]
    })

    prediction = model.predict(input_df)[0]

    st.success(
        f"Predicted Selling Price: ₹ {prediction:.2f} Lakhs"
    )

    # ===============================================
    # VISUALIZATION
    # ===============================================

    st.subheader("Price Trend Based on Car Age")

    ages = np.arange(1, 16)

    prices = []

    for age in ages:

        temp = pd.DataFrame({
            "Present_Price": [present_price],
            "Driven_kms": [driven_kms],
            "Fuel_Type": [fuel_type],
            "Selling_type": [selling_type],
            "Transmission": [transmission],
            "Owner": [owner],
            "Car_Age": [age]
        })

        prices.append(model.predict(temp)[0])

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(
        ages,
        prices,
        marker="o",
        color="blue",
        linewidth=2
    )

    ax.set_title("Predicted Price vs Car Age")
    ax.set_xlabel("Car Age (Years)")
    ax.set_ylabel("Predicted Price (Lakhs)")
    ax.grid(True)

    st.pyplot(fig)

st.write("Model trained using Random Forest Regressor")