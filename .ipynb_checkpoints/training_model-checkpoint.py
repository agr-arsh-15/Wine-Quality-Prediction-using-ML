#!/usr/bin/env python
# coding: utf-8

# # ***Wine Quality Prediction using ML***

# https://archive.ics.uci.edu/dataset/186/wine+quality

# ***

# - Dataset Characteristics: multivariate
# - Subject Area: Business 
# - Total instances: 4898
# - Features: 11

# **Input variables (based on physicochemical tests):**
# 1. fixed acidity
# 2. volatile acidity
# 3. citric acid
# 4. residual sugar
# 5. chlorides
# 6. free sulfur dioxide
# 7. total sulphur dioxide
# 8. density
# 9. pH
# 10. sulphates
# 11. alcohol
# 
# **Output variable (based on sensory data):**
# 
# 12. quality (Score between 0 and 10)   

# ### ***Problem Statement***
# Wine quality is a key factor influencing consumer satisfaction and market value. However, assessing wine quality typically requires expert tasters, which is both time-consuming and expensive.
# 
# This project aims to build a machine learning model that predicts the quality of wine based on its physicochemical properties such as acidity, pH, alcohol content, and sugar levels. Using classification algorithms, the goal is to accurately classify wines on a quality scale (e.g., 0 to 10).
# 
# The model’s predictions can help winemakers optimize production processes, ensure quality consistency, and make data-driven decisions without solely relying on manual testing.

# In[ ]:


# Importing necessary libraries
import joblib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from imblearn.over_sampling import SMOTE
from collections import Counter
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# In[ ]:


# Integrating the Dataset to the notebook
file_path= "Dataset.csv"
data = pd.read_csv(file_path)


# In[ ]:


# Removing Duplicates
data_cleaned = data.drop_duplicates()


# In[ ]:


# Separate features and target
X = data_cleaned.drop(columns=['quality'])
y = data_cleaned['quality']


# In[ ]:


# Splitting the data into Testing and Training
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42, stratify=y)


# ### ***Data Preprocessing***
# 1. Checking for null values
# 2. Standardising the values
# 3. Checking for outliers and removing them using Z-score
# 4. Checking for class imbalance and balance them using SMOTE analysis

# In[ ]:


# Check for missing values in each column
print(data_cleaned.isnull().sum())


# In[ ]:


# Scaling the Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Checking for Standardized values
print("Mean (After Scaling):", X_train_scaled.mean(axis=0))
print("Std (After Scaling):", X_train_scaled.std(axis=0))


# In[ ]:


# Converting scaled data to DataFrame
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)

# Checking for outliers
plt.figure(figsize=(12, 8))
sns.boxplot(data=X_train_scaled_df)
plt.xticks(rotation=90)
plt.title('Box Plot to Detect Outliers')
plt.show()


# In[ ]:


# Removing outliers using Z-Score
z_scores = np.abs(stats.zscore(X_train_scaled_df))
mask = ((z_scores < 3).all(axis=1))
X_train_filtered = X_train_scaled_df[mask]
y_train_filtered = y_train[mask]

# Converting filtered data to Data Frame
X_train_filtered_df = pd.DataFrame(X_train_filtered, columns=X_train_scaled_df.columns)

# Plot after removing outliers
plt.figure(figsize=(12, 6))
sns.boxplot(data=X_train_filtered_df)
plt.title('Boxplot of Scaled Features')
plt.xticks(rotation=90)
plt.show()

# Checking for Standardized values after removing outliers
print("Mean (After Scaling):", X_train_filtered_df.mean(axis=0))
print("Std (After Scaling):", X_train_filtered_df.std(axis=0))


# In[ ]:


# Initial Class Distribution
print("Class Counts Before SMOTE:")
print(y_train_filtered.value_counts())

# Plotting the Class Distribution
y_train_filtered.value_counts().plot(kind='bar', color=['blue','orange'])
plt.title("Class Distribution")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()


# In[ ]:


# Applying SMOTE Analysis to balance the Dataset
smote = SMOTE(sampling_strategy='auto', random_state=42, k_neighbors=3)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_filtered_df, y_train_filtered)


# In[ ]:


# Finalising the Class Distribution
print("Class Counts After SMOTE:")
print(y_train_resampled.value_counts())

# Plotting the Class Distribution
y_train_resampled.value_counts().plot(kind='bar', color=['blue','orange'])
plt.title("Class Distribution")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()


# ### ***Model Evaluation***
# 1. Applying **XGB Regressor** on pre-processed data
# 2. Evaluating through R2 Score, MAE, MSE

# In[ ]:


# Re-Splitting the Data after SMOTE
X_train, X_test, y_train, y_test = train_test_split(X_train_resampled, y_train_resampled, test_size=0.2, random_state=42)


# In[ ]:


# Using XG Boosting Model to predict the Wine quality
xgb_model = XGBRegressor(random_state=42)

# Definining Paramters
param_dist = {
    'n_estimators': [100, 300, 500, 700, 1000],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7, 9],
    'min_child_weight': [1, 3, 5, 7],
    'gamma': [0, 0.1, 0.2, 0.3],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'reg_alpha': [0, 0.01, 0.1, 1, 10],
    'reg_lambda': [1, 10, 50, 100]
}

# Hyperparameter tuning using Randomized Search
random_search = RandomizedSearchCV(estimator=xgb_model, param_distributions=param_dist,
                                    n_iter=50,
                                    cv=5,
                                    verbose=2,
                                    n_jobs=-1,
                                    scoring='r2',
                                    random_state=42)

random_search.fit(X_train, y_train)

print("Best Parameters:", random_search.best_params_)

best_xgb_model = random_search.best_estimator_
y_pred_xgb = best_xgb_model.predict(X_test)

# Evaluating the model
print("R2 Score for XGBoost Model:", r2_score(y_test, y_pred_xgb))
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, y_pred_xgb))
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred_xgb))


# ### ***Feature Engineering***

# In[ ]:


# Getting feature importance
feature_importance = best_xgb_model.feature_importances_

# Converting feature importance to DataFrame
feature_importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

print(feature_importance_df)


# In[ ]:


# Plotting feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importance from Random Forest')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.show()


# In[ ]:


# Compute the correlation matrix
correlation_matrix = data_cleaned.corr()

# Plot the heatmap
plt.figure(figsize=(10,8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# Check the correlation of free sulfur dioxide with quality
correlation_value = correlation_matrix['quality']['alcohol']
print(f"Correlation between Alcohol and Quality: {correlation_value:.2f}")


# **Final Analysis of Feature Engineering**
# 
# Feature engineering and correlation analysis indicate that **alcohol** content is the most influential factor in predicting wine quality, contributing **26.55%** to the model’s decision and showing a moderate positive correlation (0.46) with quality. Additionally, free sulfur dioxide (23.66%) and chlorides (11.32%) significantly impact the quality, with excessive levels negatively affecting taste. While volatile acidity and total sulfur dioxide demonstrate a moderate effect, their correlation with quality highlights the importance of maintaining balanced levels. This combined insight enables more accurate predictions, supporting informed decisions in wine production and quality management.

# ### ***Plot of Actual vs Predicted Value***

# In[ ]:


plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test, y=y_pred_xgb)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.xlabel('Actual Quality')
plt.ylabel('Predicted Quality')
plt.title('Actual vs Predicted Wine Quality')
plt.show()


# In[ ]:





# ### ***Wine Quality Prediction - Model Comparison***
# 
# | **Metric**                  | **Random Forest Regressor** | **Gradient Boosting Regressor** | **XGBoost Regressor** |
# |------------------------------|-----------------------------|---------------------------------|------------------------|
# | **R² Score**                  | 0.91                        | 0.92                            | **0.95**                |
# | **Mean Absolute Error (MAE)** | 0.35                        | 0.31                            | **0.27**                |
# | **Mean Squared Error (MSE)**  | 0.24                        | 0.21                            | **0.19**                |
# | **Training Time**             | Moderate                    | High                            | **Moderate to High**    |
# | **Interpretability**          | High                        | Moderate                        | Moderate                |
# | **Overfitting Risk**          | Moderate                    | Low to Moderate                 | **Low (with tuning)**   |
# | **Hyperparameter Sensitivity**| Moderate                    | High                            | **High**                |
# | **Best for Small Datasets**   | Yes                         | No                              | No                     |
# | **Best for Large Datasets**   | Moderate                    | Yes                             | **Yes**                 |

# ### ***Saving the Model***

# In[25]:


joblib.dump(best_xgb_model, 'model.pkl')
print("Model saved as model.pkl")

joblib.dump(scaler, 'scaler.pkl')
print("Scaler saved as scaler.pkl")


# In[ ]:





# In[ ]:





# ### ***Predicting the Sample Input***

# In[ ]:


sample_input = [[7.0, 0.25, 0.5, 4.0, 0.02, 25.0, 150.0, 0.994, 3.4, 0.65, 15]]
sample_input_scaled = scaler.transform(sample_input)
prediction = best_xgb_model.predict(sample_input_scaled)
print("Predicted Quality:", prediction[0])

