import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# Load the data
train_df = pd.read_csv('./Tasks/house-prices-advanced-regression-techniques/train.csv')

# Explore the data
print(train_df.info())
print(train_df.describe())

# Clean the data
train_df['SalePrice'] = pd.to_numeric(train_df['SalePrice'], errors='coerce')
train_df.dropna(inplace=True)

# Split the data into training and validation sets
X = train_df.drop('SalePrice', axis=1)
y = train_df['SalePrice']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the validation set
y_pred = model.predict(X_val)

# Calculate the mean squared error
mse = mean_squared_error(y_val, y_pred)
print(f'Mean squared error: {mse:.2f}')

# Load the test data
test_df = pd.read_csv('./Tasks/house-prices-advanced-regression-techniques/test.csv')

# Make predictions on the test data
y_pred_test = model.predict(test_df)

# Save the predictions to a submission file
submission_df = pd.DataFrame({'Id': test_df['Id'], 'SalePrice': y_pred_test})
submission_df.to_csv('submission.csv', index=False)
