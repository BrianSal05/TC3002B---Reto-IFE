import pandas as pd

# Load the dataset
data = pd.read_csv('DataSets/database.csv')
total_rows_before = len(data)


# Define target variables
target_variables = [
    'age', 'gender', 'program', 'average.first.period', 'scholarship.perc', 
    'loan.perc', 'retention', 'dropout.semester', 'socioeconomic.level', 'foreign',
    'social.lag'
]

# Function to clean the data
def clean_data(df, target_vars):
    for var in target_vars:
        df = df[df[var] != 'No information']
    return df

# Clean the data
cleaned_data = clean_data(data, target_variables)

# Save the cleaned data to a new CSV file
cleaned_data.to_csv('clean.csv', index=False)


print("Total rows before cleaning: ", total_rows_before)
print("Total rows after cleaning: ", len(cleaned_data))
print("Cleaned data saved to 'clean.csv'")