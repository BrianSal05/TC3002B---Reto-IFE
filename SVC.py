import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

def SupportVC(X, y):
    #   Datos de Prueba y Entrenamiento
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    #   SKLearn Support Vector Machine
    svm_clf = SVC(kernel='linear', random_state=42)
    svm_clf.fit(X_train, y_train)

    y_pred = svm_clf.predict(X_test)

    #   Reporte de Clasificación
    classification_rep = classification_report(y_test, y_pred, output_dict=True)
    classification_df = pd.DataFrame(classification_rep).transpose()

    confusion_mat = confusion_matrix(y_test, y_pred)

    return classification_df, confusion_mat
