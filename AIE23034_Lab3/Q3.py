import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('smoking.csv')

# Convert categorical variables to numerical
df['gender'] = df['gender'].map({'M': 1, 'F': 0})
df['oral'] = df['oral'].map({'Y': 1, 'N': 0})
df['tartar'] = df['tartar'].map({'Y': 1, 'N': 0})

# Select only numeric columns for analysis
numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
numeric_columns.remove('ID')  # Remove ID column
numeric_columns.remove('smoking')  # Remove target variable

# Drop rows with missing values
df = df[numeric_columns].dropna()

# Select two feature vectors for Minkowski distance calculation
vector1 = df.iloc[0].values
vector2 = df.iloc[1].values

# Calculate Minkowski distances for r from 1 to 10
distances = []
r_values = range(1, 11)

for r in r_values:
    distance = np.linalg.norm(vector1 - vector2, ord=r)
    distances.append(distance)

# Plot Minkowski distances
plt.plot(r_values, distances, marker='o', color='blue', linestyle='dashed')
plt.title("Minkowski Distance between Two Feature Vectors")
plt.xlabel("Order r")
plt.ylabel("Distance")
plt.grid(True)
plt.show()

print("Minkowski distances for r from 1 to 10:", distances)
