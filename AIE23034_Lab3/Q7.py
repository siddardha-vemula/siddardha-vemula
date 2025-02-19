import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# Load dataset
df = pd.read_csv('smoking.csv')

# Convert categorical variables to numerical
df['gender'] = df['gender'].map({'M': 1, 'F': 0})
df['oral'] = df['oral'].map({'Y': 1, 'N': 0})
df['tartar'] = df['tartar'].map({'Y': 1, 'N': 0})

# Select only numeric columns
numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
numeric_columns.remove('ID')  # Remove ID column

# Drop rows with missing values
df = df[numeric_columns].dropna()

# Split features and target
X = df.drop(columns=['smoking'])  # Features
y = df['smoking']  # Target variable

# Split dataset into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train classifier
k = 3  # Default k value for kNN
model = KNeighborsClassifier(n_neighbors=k)
model.fit(X_train_scaled, y_train)

# Predict and analyze behavior
y_test_pred = model.predict(X_test_scaled)

print("Predictions for test data:", y_test_pred)

# Perform classification on a single test vector
test_vect = X_test_scaled[0].reshape(1, -1)  # Selecting a single test vector
predicted_class = model.predict(test_vect)
print(f"Predicted class for test vector {test_vect}: {predicted_class[0]}")
