import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load(r"C:/Users/Admin/Downloads/diabetes_linear_model.pkl")

# Title of the app
st.title("Diabetes Prediction App")

# Input fields for user to enter data
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0)
glucose = st.number_input("Glucose", min_value=0, max_value=200, value=100)
insulin = st.number_input("Insulin", min_value=0, max_value=900, value=100)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)  # Added Blood Pressure
bmi = st.number_input("BMI", min_value=0.0, max_value=50.0, value=25.0)
diabetes_pedigree_function = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5)
age = st.number_input("Age", min_value=0, max_value=120, value=30)

# Button to predict
if st.button("Predict"):
    # Create a DataFrame with the input data
    input_data = pd.DataFrame({
        'Pregnancies': [pregnancies],
        'Glucose': [glucose],
        'Insulin': [insulin],
        'BloodPressure': [blood_pressure],  # Include Blood Pressure
        'BMI': [bmi],
        'DiabetesPedigreeFunction': [diabetes_pedigree_function],
        'Age': [age]
    })

    # Predict the outcome
    try:
        prediction = model.predict(input_data)

        # Display the result
        if prediction[0] == 1:
            st.success("The model predicts that the person has diabetes.")
        else:
            st.success("The model predicts that the person does not have diabetes.")
    except ValueError as e:
        st.error(f"Error in prediction: {e}")

# Footer
st.markdown("### Note: Please ensure that the input values are accurate.")