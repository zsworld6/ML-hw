from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv('data.csv')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data.drop('target_column_name', axis=1), data['target_column_name'], test_size=0.2)

# Select the most relevant features for the model
X_train = X_train.select_dtypes(include=['object']).dropna()
X_test = X_test.select_dtypes(include=['object']).dropna()

# Fit the model using the selected features
clf = MultinomialNB()

# Train the model
clf.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = clf.predict(X_test)

# Calculate the accuracy score
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")