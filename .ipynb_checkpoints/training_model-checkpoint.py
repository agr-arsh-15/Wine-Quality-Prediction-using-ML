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

# In[41]:


# Importing necessary libraries
import joblib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay,roc_curve, precision_score, recall_score, f1_score


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
# 1. Applying **Random Forest** on pre-processed data
# 2. Evaluating through accuracy, F-1 score, precision, recall
# 3. Constructing Confusion Matrix
# 4. Plotting ROC Curve
# 5. Cross-Validation

# In[ ]:


# Re-Splitting the Data after SMOTE
X_train, X_test, y_train, y_test = train_test_split(X_train_resampled, y_train_resampled, test_size=0.2, random_state=42)


# In[ ]:


# Using Random Forest Classifier to predict the Wine quality
rf_model = RandomForestClassifier(random_state=42, max_depth=19, n_estimators=250, min_samples_split=2)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))


# In[ ]:


# Constructing the Confusion Matrix
cm = confusion_matrix(y_test, y_pred_rf)
print("Confusion Matrix:\n", cm)


# In[ ]:


# Binarize the output labels for multi-class ROC
y_test_bin = label_binarize(y_test, classes=rf_model.classes_)
y_score = rf_model.predict_proba(X_test)

# Plot ROC Curve for each class
plt.figure(figsize=(10, 8))
for i in range(len(rf_model.classes_)):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    plt.plot(fpr, tpr, label=f"Class {rf_model.classes_[i]} (AUC = {roc_auc_score(y_test_bin[:, i], y_score[:, i]):.2f})")

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multiclass ROC Curve')
plt.legend(loc="lower right")
plt.grid()
plt.show()


# In[ ]:


# Perform 5-Fold Cross-Validation
cv_scores = cross_val_score(rf_model, X_train_resampled, y_train_resampled, cv=5, scoring='accuracy')

print("Cross-Validation Scores:", cv_scores)
print("Mean Accuracy:", cv_scores.mean())
print("Standard Deviation:", cv_scores.std())


# ### ***Feature Engineering***

# In[ ]:


# Getting feature importance
feature_importance = rf_model.feature_importances_

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
correlation_value = correlation_matrix['quality']['free sulfur dioxide']
print(f"Correlation between Free Sulfur Dioxide and Quality: {correlation_value:.2f}")


# Based on the analysis, alcohol emerged as the most influential feature in predicting wine quality, showing a moderate positive correlation (0.46). This suggests that higher alcohol content is generally associated with better wine quality. In contrast, while free sulfur dioxide showed a negligible correlation (0.01) with quality, its prominence during feature engineering indicates that it may have a non-linear impact on quality.

# ### ***Model Comparison***

# |Model|Accuracy|Precision|Recall|F1-Score|Training Time|Comments|
# |-----|--------|---------|------|--------|-------------|--------|
# |Random Forest|86.33%|85.73%|86.33%|85.88%|1.03 sec|Good balance of accuracy and speed|
# |SVM|76.32%|75.12%|76.32%|75.26%|0.36 sec|May take longer with large datasets|
# |Gradient Boosting|77.83%|77.13%|77.83%|77.35%|9.97 sec|Strong performance, prone to overfitting|
# |Decision Tree|77.61%|76.60%|77.61%|76.99%|0.05 sec|Simple and interpretable, but prone to overfitting.|

# In[43]:


joblib.dump(rf_model, 'rf_model.pkl')
print("Model saved as rf_model.pkl")


# In[ ]:




