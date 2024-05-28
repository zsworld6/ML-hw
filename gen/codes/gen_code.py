import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load the training data
train_df = pd.read_csv('/home/shiqi/ML-hw/Tasks/Mobile_Price_Classification/train.csv')

# Check if the column 'id' exists
if 'id' in train_df.columns:
    train_df = train_df.drop('id', axis=1)

# Preprocess the data
X = train_df.drop('price_range', axis=1)
y = train_df['price_range']

# Encode categorical variables
X['blue'] = X['blue'].astype(int)
X['dual_sim'] = X['dual_sim'].astype(int)
X['fc'] = X['fc'].astype(int)
X['four_g'] = X['four_g'].astype(int)
X['m_dep'] = X['m_dep'].astype(float)
X['m_dep'] = X['m_dep'].fillna(0)
X['mobile_wt'] = X['mobile_wt'].astype(float)
X['n_cores'] = X['n_cores'].astype(int)
X['px_height'] = X['px_height'].astype(int)
X['px_width'] = X['px_width'].astype(int)
X['ram'] = X['ram'].astype(int)
X['sc_h'] = X['sc_h'].astype(int)
X['sc_w'] = X['sc_w'].astype(int)
X['talk_time'] = X['talk_time'].astype(int)
X['three_g'] = X['three_g'].astype(int)
X['touch_screen'] = X['touch_screen'].astype(int)
X['wifi'] = X['wifi'].astype(int)

# Scale the numerical features
scaler = StandardScaler()
X[['m_dep','mobile_wt', 'n_cores', 'px_height', 'px_width', 'ram','sc_h','sc_w', 'talk_time']] = scaler.fit_transform(X[['m_dep','mobile_wt', 'n_cores', 'px_height', 'px_width', 'ram','sc_h','sc_w', 'talk_time']])

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a random forest classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Make predictions on the validation set
y_pred_val = rf.predict(X_val)

# Evaluate the model
print('Validation accuracy:', accuracy_score(y_val, y_pred_val))
print('Validation classification report:')
print(classification_report(y_val, y_pred_val))
print('Validation confusion matrix:')
print(confusion_matrix(y_val, y_pred_val))

# Load the test data
test_df = pd.read_csv('/home/shiqi/ML-hw/Tasks/Mobile_Price_Classification/test.csv')

# Drop the 'id' column from the test data
test_df = test_df.drop('id', axis=1)

# Scale the numerical features in the test data
test_df[['m_dep','mobile_wt', 'n_cores', 'px_height', 'px_width', 'ram','sc_h','sc_w', 'talk_time']] = scaler.transform(test_df[['m_dep','mobile_wt', 'n_cores', 'px_height', 'px_width', 'ram','sc_h','sc_w', 'talk_time']])

# Make predictions on the test set
y_pred_test = rf.predict(test_df)

# Save the predictions to a csv file
submission_df = pd.DataFrame({'id': range(1, len(y_pred_test)+1), 'price_range': y_pred_test})
submission_df.to_csv('submission.csv', index=False)