import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


df = pd.read_csv('DataSets/normalized.csv')

# Selección de las características y la variable objetivo
X = df[['program', 'average.first.period', 'scholarship.perc', 
                       'loan.perc', 'dropout.semester', 'socioeconomic.level', 'foreign', 'social.lag']]
y = df['retention']


# División de los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Entrenamiento del modelo de regresión lineal
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Predicciones
y_pred = lin_reg.predict(X_test)

# Evaluación del modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Coeficientes del modelo
for var in range (len(X.columns)):
    print(X.columns[var], ":", lin_reg.coef_[var])
print("Intercepto del modelo:", lin_reg.intercept_)


