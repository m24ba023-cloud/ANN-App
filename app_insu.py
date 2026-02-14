import streamlit as st
import pandas as pd
import joblib

# Load the trained model and scaler
model = joblib.load('model.joblib')
scaler = joblib.load('scaler.joblib')

st.title('Insurance Charge Prediction App')

st.write('Enter the details below to predict insurance charges:')

# Input fields
age = st.slider('Age', 18, 100, 30)
sex = st.selectbox('Sex', ['male', 'female'])
bmi = st.slider('BMI', 15.0, 50.0, 25.0)
children = st.slider('Number of Children', 0, 5, 1)
smoker = st.selectbox('Smoker', ['yes', 'no'])
region = st.selectbox('Region', ['southwest', 'northwest', 'southeast', 'northeast'])

# Create a DataFrame from inputs
input_data = pd.DataFrame({
    'age': [age],
    'sex': [sex],
    'bmi': [bmi],
    'children': [children],
    'smoker': [smoker],
    'region': [region]
})

# Preprocess the input data
# Align columns with the training data (X.columns obtained during training)
# Recreate original_cols from the notebook state
original_cols = ['age', 'bmi', 'children', 'sex_male', 'smoker_yes', 'region_northwest', 'region_southeast', 'region_southwest'] # Replace with actual X.columns

input_data_processed = pd.get_dummies(input_data, columns=['sex', 'smoker', 'region'], drop_first=True)

# Add missing columns with False and reorder to match training data
for col in original_cols:
    if col not in input_data_processed.columns:
        input_data_processed[col] = False

input_data_processed = input_data_processed[original_cols]

# Scale the input data
input_data_scaled = scaler.transform(input_data_processed)

# Predict charges
if st.button('Predict Charges'):
    prediction = model.predict(input_data_scaled)
    st.success(f'Predicted Insurance Charges: ${prediction[0][0]:.2f}')
