import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("💼 Employee Salary Prediction")

st.write("Enter the employee's details to predict their salary:")

# Input from user
experience = st.number_input("Years of Experience", min_value=0.0, max_value=50.0, step=0.1)

education_dict = {
    "High School": 1,
    "Bachelor's": 2,
    "Master's": 3
}
education = st.selectbox("Education Level", list(education_dict.keys()))
education_level = education_dict[education]

# Predict button
if st.button("Predict Salary"):
    features = np.array([[experience, education_level]])
    prediction = model.predict(features)[0]
    st.success(f"Predicted Salary: ${prediction:,.2f}")
