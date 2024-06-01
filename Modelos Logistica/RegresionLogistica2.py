import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Cargar los datos normalizados
data = pd.read_csv('DataSets/normalized.csv')

# Selección de características excluyendo 'dropout.semester'
# Basado en las correlaciones y sin 'dropout.semester'
X = data[['average.first.period', 'scholarship.perc', 'socioeconomic.level', 'age', 'loan.perc', 'program']]
y = data['retention']

# División de los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Ajuste de Pesos del Modelo usando 'balanced'
log_reg = LogisticRegression(class_weight='balanced', random_state=42)
log_reg.fit(X_train, y_train)

# Predicciones y evaluación
y_pred = log_reg.predict(X_test)

# Generar el reporte de clasificación y la matriz de confusión
classification_rep = classification_report(y_test, y_pred, output_dict=True)
classification_df = pd.DataFrame(classification_rep).transpose()
confusion_mat = confusion_matrix(y_test, y_pred)

# Imprimir los resultados
print("Classification Report:\n", classification_df)
print("\nConfusion Matrix:\n", confusion_mat)

