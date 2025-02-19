import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('smoking.csv')

# Convert categorical variables to numerical
df['gender'] = df['gender'].map({'M': 1, 'F': 0})
df['oral'] = df['oral'].map({'Y': 1, 'N': 0})
df['tartar'] = df['tartar'].map({'Y': 1, 'N': 0})

# Select numeric columns for analysis (excluding ID and target variable)
numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
numeric_columns.remove('ID')  # Remove ID column
numeric_columns.remove('smoking')  # Remove target variable

# Drop rows with missing values
df = df[numeric_columns].dropna()

# Select a feature for histogram plotting
feature = "age"  # You can change this to any numeric column from the dataset

# Plot histogram
plt.hist(df[feature], bins=10, color='skyblue', edgecolor='black')
plt.title(f"Histogram of {feature}")
plt.xlabel(feature)
plt.ylabel("Frequency")
plt.show()

# Calculate mean and variance
mean_value = df[feature].mean()
variance_value = df[feature].var()

print(f"Mean of {feature}: {mean_value}")
print(f"Variance of {feature}: {variance_value}")
