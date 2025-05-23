# -*- coding: utf-8 -*-
"""G24AIT2189_Assignment2_Salary_Prediction.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1xrhytr8fbbp57Sq_y67nNXqquN8rinT8

Q4: Select a project from the list provided in the document and implement a complete end-to-end machine learning pipeline for the same Include the major methods, results, discussions, and conclusions in your report. Also, prepare a demo using gradio/streamlit for evaluation.

List of projects: https://docs.google.com/document/d/1HYz5TA1QBhhutvWdKIKvTr4v6SzebjrKxZbklh8zsZo/edit?usp=sharing

Opted -

**Salary Prediction Project**


Downloaded Dataset: Salary Prediction dataset (kaggle.com)

**Description**:
Develop a machine learning model to predict the salaries of employees based on their qualifications, experience, and other relevant factors. The goal is to create a system that can provide accurate salary estimates to help job seekers and employers make informed decisions.
"""

import pandas as pd
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

dataset = pd.read_csv(
    "/content/Salary Data.csv"
)

dataset.head()

print(f"Dataset size: {dataset.shape[0]}")

dataset.isna().sum()
dataset = dataset.dropna(axis=0)

print(f"Total unique values in 'Gender' column: {len(dataset['Gender'].unique())}")
print(f"Total unique values in 'Education Level' column: {len(dataset['Education Level'].unique())}")
print(f"Total unique values in 'Job Title' column: {len(dataset['Job Title'].unique())}")

from sklearn.preprocessing import LabelEncoder

gender_encoder = LabelEncoder()
dataset["Gender"] = gender_encoder.fit_transform(dataset["Gender"])

edu_level_encoder = LabelEncoder()
dataset["Education Level"] = edu_level_encoder.fit_transform(
    dataset["Education Level"]
)

dataset = pd.get_dummies(
    dataset, columns=["Job Title"], prefix="JobTitle",
    drop_first=False, dtype=int
)

dataset.head()

dataset[["Age", "Gender", "Education Level", "Years of Experience", "Salary"]].describe()

from sklearn.preprocessing import StandardScaler

normalize_columns = ["Age", "Years of Experience", "Salary"]
scaler = StandardScaler()
dataset[normalize_columns] = scaler.fit_transform(dataset[normalize_columns])

dataset[normalize_columns].head()

from matplotlib import pyplot as plt
import seaborn as sns

_, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=100)

for i, column in enumerate(["Age", "Years of Experience", "Salary"]):
    sns.histplot(dataset[column], kde=True, ax=axes[i])

_, axes = plt.subplots(2, 3, figsize=(15, 7), dpi=100)

col = 0
row = 0
for column in ["Age", "Gender", "Education Level", "Years of Experience", "Salary"]:
    sns.boxplot(x=column, data=dataset, ax=axes[col, row])
    col, row = (col+1, 0) if row >= 2 else (col, row+1)

from sklearn.model_selection import train_test_split

X = dataset.drop(columns=["Salary"], axis=1, inplace=False)
y = dataset[["Salary"]]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

print(f"Train dataset size: {X_train.shape[0]}")
print(f"Test dataset size: {X_test.shape[0]}")

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(
    n_estimators=30,
    max_depth=10,
    n_jobs=-1,
    random_state=42,
    verbose=0,
    bootstrap=True,
    oob_score=True,
    criterion="absolute_error",
    max_features=1.0,
    min_samples_split=30,
)
model.fit(X_train, y_train)

# Save the model
import pickle
filename = 'my_random_forest_model.pkl'
with open(filename, 'wb') as file:
    pickle.dump(model, file)
print(f"Model saved as {filename}")

# Load the model (in a new session or later)
with open(filename, 'rb') as file:
    loaded_model = pickle.load(file)
print("Model loaded successfully")

print(f"Train score: {model.score(X_train, y_train)}")
print(f"Test score: {model.score(X_test, y_test)}")

from sklearn.metrics import mean_squared_error

y_train_pred = model.predict(X_train)
train_mse = mean_squared_error(y_train, y_train_pred)
print(f"Train dataset Mean Squared Error: {train_mse}")

y_test_pred = model.predict(X_test)
test_mse = mean_squared_error(y_test, y_test_pred)
print(f"Test dataset Mean Squared Error: {test_mse}")

import numpy as np

train_r_mse = np.sqrt(mean_squared_error(y_train, y_train_pred))
print(f"Train dataset Root Mean Squared Error: {train_r_mse}")

test_r_mse = np.sqrt(mean_squared_error(y_test, y_test_pred))
print(f"Train datasetRoot Mean Squared Error: {test_r_mse}")

from sklearn.metrics import r2_score

train_r2 = r2_score(y_train, y_train_pred)
print(f"Train dataset R-squared: {train_r2}")

test_r2 = r2_score(y_test, y_test_pred)
print(f"Test dataset R-squared: {test_r2}")

def adjusted_r2(r2, n, p):
    return 1 - (((1 - r2) * (n - 1)) / (n - p - 1))

train_adj_r2 = adjusted_r2(train_r2, len(y_train), 5)
print(f"Train dataset Adjusted R2 Score: {train_adj_r2}")

test_adj_r2 = adjusted_r2(test_r2, len(y_test), 5)
print(f"Test dataset Adjusted R2 Score: {test_adj_r2}")

