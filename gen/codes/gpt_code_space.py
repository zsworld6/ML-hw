import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the train data
train_data = pd.read_csv('./Tasks/spaceship-titanic/train.csv')

# Handling missing values for numerical columns
imputer = SimpleImputer(strategy='median')
numerical_columns = train_data.select_dtypes(include=['float64', 'int64']).columns
train_data[numerical_columns] = imputer.fit_transform(train_data[numerical_columns])

# Convert categorical variables to numeric using OneHotEncoder
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
categorical_columns = train_data.select_dtypes(include=['object']).columns
train_encoded = encoder.fit_transform(train_data[categorical_columns])
train_encoded_df = pd.DataFrame(train_encoded, columns=encoder.get_feature_names_out())

# Combine encoded columns back to the original dataframe
train_data = train_data.drop(categorical_columns, axis=1)
train_data = pd.concat([train_data, train_encoded_df], axis=1)

# Define X and y
X = train_data.drop(['Transported'], axis=1)
y = train_data['Transported']

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Make predictions on the validation set
y_pred = rf.predict(X_val)

# Evaluate the model
print('Validation accuracy: {:.3f}'.format(accuracy_score(y_val, y_pred)))
print('Validation classification report:')
print(classification_report(y_val, y_pred))
print('Validation confusion matrix:')
print(confusion_matrix(y_val, y_pred))

# Load the test data
test_data = pd.read_csv('./Tasks/spaceship-titanic/test.csv')
passenger_ids = test_data['PassengerId']  # Save PassengerId before any manipulation
test_data[numerical_columns] = imputer.transform(test_data[numerical_columns])
test_encoded = encoder.transform(test_data[categorical_columns])
test_encoded_df = pd.DataFrame(test_encoded, columns=encoder.get_feature_names_out())
test_data = test_data.drop(categorical_columns, axis=1)
test_data = pd.concat([test_data, test_encoded_df], axis=1)

# Make predictions on the test set
y_test_pred = rf.predict(test_data)

# Save the predictions to submission.csv
submission_data = pd.DataFrame({'PassengerId': passenger_ids, 'Transported': y_test_pred})
submission_data.to_csv('submission.csv', index=False)
