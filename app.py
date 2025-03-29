import streamlit as st
import numpy as np
import joblib

# Load the saved model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# Streamlit App Title
st.title('Wine Quality Prediction App')
st.write('Enter the wine characteristics to predict its quality.')

# Input Fields for the Features
features = []
feature_names = ['Fixed Acidity', 'Volatile Acidity', 'Citric Acid', 'Residual Sugar', 'Chlorides',
                 'Free Sulfur Dioxide', 'Total Sulfur Dioxide', 'Density', 'pH', 'Sulphates', 'Alcohol']

for name in feature_names:
    value = st.number_input(f'{name}', min_value=0.0, step=0.1)
    features.append(value)

# Predict Button
if st.button('Predict Quality'):
    try:
        final_features = np.array(features).reshape(1, -1)
        scaled_features = scaler.transform(final_features)
        prediction = model.predict(scaled_features)
        result = round(prediction[0], 2)
        st.success(f'Predicted Wine Quality: {result}')
    except Exception as e:
        st.error(f'Error: {e}')

