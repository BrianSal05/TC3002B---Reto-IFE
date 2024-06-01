import numpy as np
import pandas as pd
from SVC import SupportVC
from GradientBootsting import GB


df = pd.read_csv('DataSets/normalized.csv')

# Filter Dropouts
#df = df[df['retention'] == 0]

# Selección de las características y la variable objetivo
X = df[['program', 'average.first.period', 'scholarship.perc', 
                       'loan.perc', 'socioeconomic.level', 'foreign', 'social.lag']]
y = df['retention']


# Support Vector Classifier
classification_df, confusion_mat = SupportVC(X, y)
print("Support Vector Classifier")
print(classification_df)
print("Confusion Matrix:")
print(confusion_mat)

# Gradient Boosting
classification_df, confusion_mat = GB(X, y)
print("Gradient Boosting")
print(classification_df)
print("Confusion Matrix:")
print(confusion_mat)
