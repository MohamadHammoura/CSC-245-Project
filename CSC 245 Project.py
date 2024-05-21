#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Step 1: Load the data
data = pd.read_csv('true_car_listings.csv')
print("Data loaded successfully.")

# Display basic data information
print("First few rows of the dataset:")
print(data.head())
print("Summary statistics of the dataset:")
print(data.describe())

# Define categorical and numeric features
categorical_features = ['City', 'State', 'Make', 'Model']
numeric_features = ['Year', 'Mileage']

# Define Column Transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Create a pipeline that processes the data and then runs the model
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('regressor', LinearRegression())])

# Separate target variable and features
X = data.drop('Price', axis=1)
y = data['Price']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Data split into training and testing sets.")

# Fit the pipeline
pipeline.fit(X_train, y_train)
print("Model trained successfully.")

# Predict prices on the testing set
predictions = pipeline.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
print(f"Mean Absolute Error: {mae}")
print(f"Accuracy Score: {r2}")


# In[ ]:




