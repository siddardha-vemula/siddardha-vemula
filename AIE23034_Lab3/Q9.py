import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

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

# Train classifier with optimal k (assuming k=3 based on previous evaluation)
k_optimal = 3
model = KNeighborsClassifier(n_neighbors=k_optimal)
model.fit(X_train_scaled, y_train)

# Predictions
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

# Evaluate confusion matrix and performance metrics
conf_matrix_train = confusion_matrix(y_train, y_train_pred)
conf_matrix_test = confusion_matrix(y_test, y_test_pred)

print("Confusion Matrix (Training Data):\n", conf_matrix_train)
print("Confusion Matrix (Test Data):\n", conf_matrix_test)

print("Classification Report (Training Data):\n", classification_report(y_train, y_train_pred))
print("Classification Report (Test Data):\n", classification_report(y_test, y_test_pred))

# Model evaluation for overfitting or underfitting
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

if train_accuracy > test_accuracy + 0.1:
    print("Model is overfitting.")
elif test_accuracy > train_accuracy:
    print("Model might be underfitting.")
else:
    print("Model is well-generalized (regular fit).")
