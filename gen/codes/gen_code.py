import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load the dataset
train_df = pd.read_csv('/home/shiqi/ML-hw/Tasks/spaceship-titanic/train.csv')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(train_df.drop('Transported', axis=1), train_df['Transported'], test_size=0.2, random_state=42)

# Encode categorical variables
le = LabelEncoder()
for col in X_train.select_dtypes(include=['object']).columns:
    X_train[col] = le.fit_transform(X_train[col])
    X_test[col] = le.transform(X_test[col])

# Train a random forest classifier
clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=42)
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.3f}')

# Save the predictions to a csv file
submission_df = pd.DataFrame({'PassengerId': X_test['PassengerId'], 'Transported': y_pred})
submission_df.to_csv('./submissions/submission.csv', index=False)