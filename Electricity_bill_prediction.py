import streamlit as st
import pandas as pd
import joblib

from pathlib import Path


MODEL_PATH = Path(__file__).resolve().parent / "polynomial_regression_electric_bill (1).pkl"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()


st.set_page_config(
    page_title="Electric Bill Prediction",
    page_icon="⚡",
    layout="centered"
)


st.title("⚡ Electric Bill Prediction")

st.write("Polynomial Regression Model")

st.write("Predict the Electric Bill based on AC Units")


st.divider()


ac_units = st.number_input(
    "Enter AC Units",
    min_value=10.0,
    max_value=105.0
    value=10.0,
    step=1.0
)


if st.button("Predict Electric Bill", type="primary"):

    new_data = pd.DataFrame({
        "AC_Units": [ac_units]
    })

    prediction = model.predict(new_data)[0]

    st.success(
        f"Predicted Electric Bill: ₹{prediction:,.2f}"
    )
