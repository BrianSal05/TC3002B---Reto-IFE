import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Load the data
file_path = 'DataSets/clean.csv'  # Replace with your file path
df = pd.read_csv(file_path)

# Selecting the relevant columns for normalization
columns_to_normalize = [
    'age', 'gender', 'program', 'average.first.period', 
    'scholarship.perc', 'loan.perc', 'retention', 
    'dropout.semester','socioeconomic.level', 'foreign', 'social.lag'
]

# Keeping only the relevant columns
df_selected = df[columns_to_normalize]

# Encoding categorical variables
label_encoders = {}
categorical_columns = ['gender', 'program', 'socioeconomic.level', 'foreign', 'social.lag']

for col in categorical_columns:
    label_encoders[col] = LabelEncoder()
    df_selected[col] = label_encoders[col].fit_transform(df_selected[col])

# Normalizing numerical variables
scaler = MinMaxScaler()
numerical_columns = ['age', 'average.first.period', 'scholarship.perc', 'loan.perc', 'dropout.semester']

df_selected[numerical_columns] = scaler.fit_transform(df_selected[numerical_columns])

# Display the normalized dataframe
print(df_selected.head())

print("Total rows after cleaning: ", len(df_selected))
df_selected.to_csv('normalized_all.csv', index=False)
