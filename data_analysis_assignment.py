# Data Analysis Assignment
# To load and analyze a dataset using pandas and create visualizations with matplotlib

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import os

print("=" * 50)
print("DATA ANALYSIS ASSIGNMENT")
print("=" * 50)

# TASK 1: LOAD AND EXPLORE THE DATASET
print("\n" + "=" * 30)
print("TASK 1: LOAD AND EXPLORE THE DATASET")
print("=" * 30)

# Load dataset
print("Loading Iris dataset...")
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target
df['species_name'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

print(f"Dataset shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())

print("\nDataset structure:")
print(f"Rows: {len(df)}, Columns: {len(df.columns)}")
print("\nColumn types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

# TASK 2: BASIC DATA ANALYSIS
print("\n" + "=" * 30)
print("TASK 2: BASIC DATA ANALYSIS")
print("=" * 30)

numerical_columns = ['sepal length (cm)', 'sepal width (cm)', 
                    'petal length (cm)', 'petal width (cm)']

print("\nBasic statistics:")
print(df[numerical_columns].describe())

print("\nGrouped by species (mean):")
species_means = df.groupby('species_name')[numerical_columns].mean()
print(species_means)

print("\nCount per species:")
print(df['species_name'].value_counts())

print("\nKey findings:")
print("- Setosa has smallest petal dimensions")
print("- Virginica has largest measurements")
print("- Strong correlation between petal length and width")
print("- Dataset is balanced (50 samples per species)")

# TASK 3: DATA VISUALIZATION
print("\n" + "=" * 30)
print("TASK 3: DATA VISUALIZATION")
print("=" * 30)

# Create a 2x2 grid of plots
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 1. Line chart
for species in df['species_name'].unique():
    species_data = df[df['species_name'] == species]
    axes[0, 0].plot(species_data.index, species_data['petal length (cm)'], 
                   label=species, marker='o', alpha=0.7)
axes[0, 0].set_title('Petal Length by Sample Index')
axes[0, 0].set_xlabel('Sample Index')
axes[0, 0].set_ylabel('Petal Length (cm)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Bar chart
avg_petal_length = df.groupby('species_name')['petal length (cm)'].mean()
axes[0, 1].bar(avg_petal_length.index, avg_petal_length.values, 
              color=['lightcoral', 'skyblue', 'lightgreen'])
axes[0, 1].set_title('Average Petal Length by Species')
axes[0, 1].set_xlabel('Species')
axes[0, 1].set_ylabel('Average Petal Length (cm)')

# 3. Histogram
axes[1, 0].hist(df['sepal length (cm)'], bins=15, color='purple', alpha=0.7)
axes[1, 0].set_title('Distribution of Sepal Length')
axes[1, 0].set_xlabel('Sepal Length (cm)')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].axvline(df['sepal length (cm)'].mean(), color='red', linestyle='--')

# 4. Scatter plot
colors = {'setosa': 'red', 'versicolor': 'blue', 'virginica': 'green'}
for species in df['species_name'].unique():
    species_data = df[df['species_name'] == species]
    axes[1, 1].scatter(species_data['sepal length (cm)'], 
                      species_data['petal length (cm)'],
                      c=colors[species], label=species, alpha=0.7)
axes[1, 1].set_title('Sepal vs Petal Length')
axes[1, 1].set_xlabel('Sepal Length (cm)')
axes[1, 1].set_ylabel('Petal Length (cm)')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()

# Save the plot
current_dir = os.getcwd()
plot_path = os.path.join(current_dir, 'iris_analysis.png')
plt.savefig(plot_path)
print(f"\nPlot saved as: {plot_path}")

print("\n" + "=" * 50)
print("ASSIGNMENT COMPLETED SUCCESSFULLY!")
print("=" * 50)