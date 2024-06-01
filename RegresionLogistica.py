# Regresion Logistica

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv('DataSets/normalized.csv')

# Dividir los datos en conjuntos de entrenamiento y prueba
X = df[['age', 'gender', 'program', 'average.first.period', 'scholarship.perc', 'loan.perc', 'socioeconomic.level', 'foreign']]
y = df['retention']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Entrenar el modelo
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Predicciones y evaluación
y_pred = log_reg.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
