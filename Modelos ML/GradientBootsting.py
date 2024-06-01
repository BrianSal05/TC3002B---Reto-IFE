import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

def GB(X, y):
    # Datos de prueba y entrenamiento
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Training
    gb_clf = GradientBoostingClassifier(random_state=42)
    gb_clf.fit(X_train, y_train)

    # Predicciones y evaluación
    y_pred = gb_clf.predict(X_test)

    # Reporte de clasificación
    classification_rep = classification_report(y_test, y_pred, output_dict=True)
    classification_df = pd.DataFrame(classification_rep).transpose()
    
    confusion_mat = confusion_matrix(y_test, y_pred)

    return classification_df, confusion_mat
