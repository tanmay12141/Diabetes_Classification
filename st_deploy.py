import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Set page title and favicon
st.set_page_config(page_title="Diabetes Prediction", page_icon=":bar_chart:")

# Header
st.title("Diabetes Prediction")

# Sidebar for user input
st.sidebar.title('User Input')

def user_input_features():
    # User input fields
    Pregnancies = st.sidebar.number_input("Number of Pregnancies", 0, 20, 6)
    Glucose = st.sidebar.number_input("Glucose (mg/dL)", 35, 200, 148)
    BloodPressure = st.sidebar.number_input("Blood Pressure (mmHg)", 20, 150, 72)
    SkinThickness = st.sidebar.number_input("Skin Thickness (mm)", 0, 100, 35)
    Insulin = st.sidebar.number_input("Insulin Level (mu U/ml)", 0, 1000, 0)
    BMI = st.sidebar.number_input("Body Mass Index (BMI)", 33.6)
    DiabetesPedigreeFunction = st.sidebar.number_input("Diabetes Pedigree Function", 0.627)
    Age = st.sidebar.number_input("Age (years)", 18, 120, 50)

    # Create DataFrame from user inputs
    data = {'Pregnancies': Pregnancies,
            'Glucose': Glucose,
            'BloodPressure': BloodPressure,
            'SkinThickness': SkinThickness,
            'Insulin': Insulin,
            'BMI': BMI,
            'DiabetesPedigreeFunction': DiabetesPedigreeFunction,
            'Age': Age}

    features = pd.DataFrame(data, index=[0])
    return features

# Get user inputs
df = user_input_features()

# Load the machine learning model
model = pickle.load(open('model.pkl', 'rb'))

# Predictions and probabilities
if st.button('Predict'):
    st.subheader("User Input:")
    st.write(df)

    prediction = model.predict(df)
    proba = model.predict_proba(df)

    st.subheader("Prediction:")
    if prediction[0] == 1:
        st.write("**Has Diabetes**")
    else:
        st.write("**Does not have Diabetes**")

    st.subheader("Probabilities:")
    st.write(proba)