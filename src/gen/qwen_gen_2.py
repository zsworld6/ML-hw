import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
# Load the training and testing datasets
data = pd.read_csv("/home/shiqi/Tasks/house-prices-advanced-regression-techniques/train.csv")
X_train, X_test, y_train, y_test = train_test_split(data['SalePrice'], data['LotFrontage'], test_size=0.2, random_state=42)

# Create the RandomForestClassifier object with the appropriate parameters
clf = RandomForestClassifier()

# Train the model using the training data
clf.fit(X_train, y_train)

# Predict the SalePrice for the testing data
y_pred = clf.predict(X_test)

# Calculate the accuracy score
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)