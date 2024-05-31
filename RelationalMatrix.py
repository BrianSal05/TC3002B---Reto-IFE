import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('DataSets/normalized.csv')

# Filter the relevant variables
variables_of_interest = [
    'age', 'gender', 'program', 'average.first.period', 'scholarship.perc', 
    'loan.perc', 'retention', 'dropout.semester', 'socioeconomic.level', 'foreign',
    'social.lag'
]
df_filtered = df[variables_of_interest]

# Generate the correlation matrix for the filtered variables
correlation_matrix = df_filtered.corr()

# Plot the correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix for Selected Variables')
plt.show()

# Analyze the correlations
def analyze_correlations(correlation_matrix, threshold):
    correlated_vars = []
    for col in correlation_matrix.columns:
        for row in correlation_matrix.index:
            if col != row and abs(correlation_matrix.loc[row, col]) > threshold:
                correlated_vars.append((row, col, correlation_matrix.loc[row, col]))

    return correlated_vars

# Describe the correlations
def describe_correlations(correlated_vars):
    print(f"Correlated Variables:{correlated_vars}")
    for var1, var2, corr in correlated_vars:
        direction = "positive" if corr > 0 else "negative"
        f.write(f"Variables {var1} and {var2} are {direction}ly correlated with a correlation coefficient of {corr:.2f}."+"\n")

# Define the threshold for significant correlation
threshold = 0.2  # Adjust this threshold based on your requirements

# Analyze and describe correlations
correlated_vars = analyze_correlations(correlation_matrix, threshold)

# Provide a summary for specific variables of interest
with open('Assets/RelationalMatrix.txt', 'w') as f:
    f.write("*************************************\n")
    f.write("Correlation Matrix Summary\n")
    describe_correlations(correlated_vars)
    f.write("\nSummary for variables of interest:")
    for var in variables_of_interest:
        f.write(f"\nVariable: {var}")
        for var1, var2, corr in correlated_vars:
            if var1 == var or var2 == var:
                f.write(f"- Correlated with {var1 if var2 == var else var2} (Correlation: {corr:.2f})")
    f.write("\n*************************************\n")
    f.write(df_filtered.describe().to_string())
