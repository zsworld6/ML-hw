import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from category_encoders import TargetEncoder
from sklearn.preprocessing import OneHotEncoder

# Load the dataset
train_df = pd.read_csv('../house-prices-advanced-regression-techniques/train.csv')

# Load the test dataset
test_df = pd.read_csv('../house-prices-advanced-regression-techniques/test.csv')

# Select the features and target
X = train_df.drop('SalePrice', axis=1)
y = train_df['SalePrice']

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert categorical variables to numerical variables
categorical_cols = X.select_dtypes(include=['object']).columns
le = LabelEncoder()
for col in categorical_cols:
    le.fit(pd.concat([X_train[col], X_val[col]]).values)
    X_train[col] = le.transform(X_train[col])
    X_val[col] = le.transform(X_val[col])
    test_df[col] = le.transform(test_df[col])

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_val_imputed = imputer.transform(X_val)
test_df_imputed = imputer.transform(test_df)

# Target encoding for the target variable
y_train = y_train.values.reshape(-1, 1)
y_val = y_val.values.reshape(-1, 1)
te = TargetEncoder()
y_train = te.fit_transform(y_train)
y_val = te.transform(y_val)

# Create a pipeline with a random forest regressor and a standard scaler
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train the model
pipe.fit(X_train_imputed, y_train)

# Evaluate the model
y_pred = pipe.predict(X_val_imputed)
print(f'Mean squared error: {mean_squared_error(y_val, y_pred)}')

# Make predictions
test_pred = pipe.predict(test_df_imputed)

# Save the predictions
submission_df = pd.DataFrame({'Id': test_df.index, 'SalePrice': test_pred})
submission_df.to_csv('submission.csv', index=False)
