import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import Imputer, LabelEncoder

# Load the training data
train_df = pd.read_csv('Tasks/house-prices-advanced-regression-techniques/train.csv')

# Define the target variable and the feature variables
target = 'SalePrice'
features = [col for col in train_df.columns if col != target]

# Convert categorical variables to numerical variables
le = LabelEncoder()
for col in features:
    if train_df[col].dtype == 'object':
        train_df[col] = le.fit_transform(train_df[col])

# Impute missing values
imputer = Imputer(strategy='median')
train_df[features] = imputer.fit_transform(train_df[features])

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(train_df[features], train_df[target], test_size=0.2, random_state=42)

# Train a random forest regressor model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Make predictions on the validation set
y_pred = rf.predict(X_val)

# Calculate the mean squared error
mse = mean_squared_error(y_val, y_pred)
print(f'Mean squared error: {mse:.2f}')

# Load the test data
test_df = pd.read_csv('Tasks/house-prices-advanced-regression-techniques/test.csv')

# Convert categorical variables to numerical variables
for col in features:
    if test_df[col].dtype == 'object':
        test_df[col] = le.transform(test_df[col])

# Impute missing values
test_df[features] = imputer.transform(test_df[features])

# Make predictions on the test set
y_pred_test = rf.predict(test_df[features])

# Save the predictions to a submission file
submission_df = pd.DataFrame({'Id': test_df['Id'], 'SalePrice': y_pred_test})
submission_df.to_csv('submission.csv', index=False)