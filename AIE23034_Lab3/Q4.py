import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
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

# Select features and target for binary classification
X = df.drop("smoking", axis=1)  # Features
y = df["smoking"]  # Target (binary classification)

# Split dataset into training and test sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("Training set size:", X_train.shape)
print("Test set size:", X_test.shape)
