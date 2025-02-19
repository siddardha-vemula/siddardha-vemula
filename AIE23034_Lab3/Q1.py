import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv('smoking.csv')

# Convert categorical variables to numerical
df['gender'] = df['gender'].map({'M': 1, 'F': 0})
df['oral'] = df['oral'].map({'Y': 1, 'N': 0})
df['tartar'] = df['tartar'].map({'Y': 1, 'N': 0})

# Select only numerical columns
df = df.select_dtypes(include=[np.number]).dropna()

# Define features and target variable
X = df.drop(columns=['smoking', 'ID'])  # Drop ID as it's not a feature
y = df['smoking']

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Select two classes for analysis
class_labels = y.unique()
if len(class_labels) >= 2:
    class1, class2 = class_labels[:2]
    
    # Extract feature vectors for two classes
    class1_vecs = X[y == class1]
    class2_vecs = X[y == class2]
    
    # Compute class centroids
    centroid1 = np.mean(class1_vecs, axis=0)
    centroid2 = np.mean(class2_vecs, axis=0)
    
    # Compute spread (standard deviation) for each class
    spread1 = np.std(class1_vecs, axis=0)
    spread2 = np.std(class2_vecs, axis=0)
    
    # Compute Euclidean distance between class centroids
    interclass_distance = np.linalg.norm(centroid1 - centroid2)
    
    print(f"Class {class1} Centroid: {centroid1}")
    print(f"Class {class2} Centroid: {centroid2}")
    print(f"Class {class1} Spread: {spread1}")
    print(f"Class {class2} Spread: {spread2}")
    print(f"Interclass Distance between {class1} and {class2}: {interclass_distance}")
else:
    print("Not enough classes to compute interclass distances.")
