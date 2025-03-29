# Wine-Quality-Prediction-using-ML
## Overview:

This project uses machine learning algorithms to predict the quality of wine based on various physicochemical features. The dataset consists of red and white wine samples with characteristics such as acidity, pH, alcohol content, and sugar levels.

## Dataset:

Source: Wine Quality Dataset (UCI Machine Learning Repository)

Features: 11 physicochemical properties (e.g., fixed acidity, volatile acidity, citric acid, residual sugar, etc.)

Target: Wine Quality Score (0 to 10)

## Models Used:

Random Forest Regressor

Gradient Boosting

XG Boosting Regressor

## Download Model
Download the pre-trained model from the link below:
[Download Model.pkl]https://drive.google.com/file/d/1gwVyvitf98KYg3XZlDq02ovN8X_LDfzn/view?usp=sharing

## Requirements

Ensure you have Python installed. Install necessary libraries using the following command:
```bash
pip install -r requirements.txt
```
## Training the Model

To train the model, run the following command:
```bash
python training_model.py
```
Ensure that model.pkl is created upon successful training.

## Deploying the Model

You can deploy the model using Streamlit. Run the following command:
```bash
streamlit run app.py
```
The app will be available at http://localhost:8501/

## How to Use
1. Open the link in your browser.
2. Enter the wine characteristics using the input fields.
3. Click on the **Predict Quality** button to get the prediction.

## Example Input
Provide the following input values in the app:
```
Fixed Acidity: 7.4
Volatile Acidity: 0.7
Citric Acid: 0.0
Residual Sugar: 1.9
Chlorides: 0.076
Free Sulfur Dioxide: 11.0
Total Sulfur Dioxide: 34.0
Density: 0.9978
pH: 3.51
Sulphates: 0.56
Alcohol: 9.4
```
Click **Predict Quality** to view the wine quality prediction.