import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

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

# Evaluate accuracy for different k values
k_values = range(1, 12)
accuracies = []

for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracies.append(accuracy_score(y_test, y_pred))

# Plot accuracy vs k values
plt.plot(k_values, accuracies, marker='o', linestyle='-')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Accuracy')
plt.title('kNN Accuracy for Different k Values')
plt.grid(True)
plt.show()

# Compare NN (k=1) with kNN (k=3)
k1_model = KNeighborsClassifier(n_neighbors=1)
k1_model.fit(X_train_scaled, y_train)
k1_accuracy = accuracy_score(y_test, k1_model.predict(X_test_scaled))

k3_model = KNeighborsClassifier(n_neighbors=3)
k3_model.fit(X_train_scaled, y_train)
k3_accuracy = accuracy_score(y_test, k3_model.predict(X_test_scaled))

print(f"Accuracy with k=1 (NN): {k1_accuracy:.4f}")
print(f"Accuracy with k=3 (kNN): {k3_accuracy:.4f}")
