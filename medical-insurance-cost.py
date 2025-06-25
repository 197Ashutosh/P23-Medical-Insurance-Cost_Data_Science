import streamlit as st
import pandas as pd
import numpy as np
import joblib as jb

model = jb.load("Medical-Insurance-cost.jb")

st.title("Medical Insurance Cost Predictor")

st.write("Enter the details below to predict the medical insurance cost:")

age = st.number_input("Age", min_value=18, max_value=100, value=30)
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
sex_male = st.selectbox("Sex", options=["Female", "Male"]) == "Male"
smoker_yes = st.selectbox("Smoker", options=["No", "Yes"]) == "Yes"
region = st.selectbox("Region", options=["northeast", "northwest", "southeast", "southwest"])

region_northwest = region == "northwest"
region_southeast = region == "southeast"
region_southwest = region == "southwest"

if st.button("Predict"):
    patient = pd.DataFrame({
        "age": [age],
        "bmi": [bmi],
        "children": [children],
        "sex_male": [sex_male],
        "smoker_yes": [smoker_yes],
        "region_northwest": [region_northwest],
        "region_southeast": [region_southeast],
        "region_southwest": [region_southwest]
    })
    pred = model.predict(patient)
    cost = np.exp(pred[0])
    st.success(f"Predicted Medical Insurance Cost: ${cost:.2f}")

    