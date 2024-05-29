import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np

# Load the training data
train_data = pd.read_csv('/home/shiqi/ML-hw/Tasks/Mobile_Price_Classification/train.csv')

# Define the categorical and numerical columns
categorical_cols = ['blue', 'clock_speed', 'dual_sim', 'fc', 'four_g','m_dep','mobile_wt', 'n_cores', 'pc','sc_h','sc_w', 'three_g', 'touch_screen', 'wifi']
numerical_cols = ['battery_power', 'int_memory','m_dep', 'ram', 'px_height', 'px_width', 'talk_time', 'price_range']

# Convert categorical columns to numerical using LabelEncoder
le = LabelEncoder()
for col in categorical_cols:
    train_data[col] = le.fit_transform(train_data[col])

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(train_data.drop('price_range', axis=1), train_data['price_range'], test_size=0.2, random_state=42)

# Train a random forest regressor model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions on the validation set
y_pred_val = rf_model.predict(X_val)

# Evaluate the model on the validation set
print('Mean absolute error on validation set: ', np.mean(np.abs(y_pred_val - y_val)))

# Load the test data
test_data = pd.read_csv('/home/shiqi/ML-hw/Tasks/Mobile_Price_Classification/test.csv')

# Convert categorical columns to numerical using LabelEncoder
for col in categorical_cols:
    test_data[col] = le.fit_transform(test_data[col])

# Split the test data into features and target
X_test = test_data.drop('id', axis=1)

# Make predictions on the test set
y_pred_test = rf_model.predict(X_test)

# Save the predictions to a csv file
submission_data = pd.DataFrame({'id': test_data['id'], 'price_range': y_pred_test})
submission_data.to_csv('submission.csv', index=False)