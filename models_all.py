import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# Local imports
from analisis import train_set, test_set  # Asegúrate de que target esté definido en analisis
from models import save_models

 
target = 'loan_status'  # Asegúrate de que la variable objetivo esté definida

def logistic_regression_all_features(train_set, target='loan_status'):
    # Eliminar duplicados antes de aplicar la condición
    train_set = train_set.loc[~train_set.index.duplicated(keep='first')]
    filtered_train_set = train_set[train_set[target] < 5]

    # Separar características y objetivo
    X = filtered_train_set.drop(columns=[target])  # Usar todas las características excepto el objetivo
    y = filtered_train_set[target]  # Variable objetivo

    # Dividir en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Escalar características
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Crear y entrenar el modelo
    logistic_model = LogisticRegression(max_iter=300)
    logistic_model.fit(X_train_scaled, y_train)

    # Predecir en el conjunto de prueba
    y_pred = logistic_model.predict(X_test_scaled)

    # Evaluar el modelo
    confusion = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    return confusion, report, logistic_model, scaler  # Retornar el escalador

def random_forest_all_features(train_set, target='loan_status'):
    # Eliminar duplicados antes de aplicar la condición
    train_set = train_set.loc[~train_set.index.duplicated(keep='first')]
    filtered_train_set = train_set[train_set[target] < 5]

    # Separar características y objetivo
    X = filtered_train_set.drop(columns=[target])  # Usar todas las características excepto el objetivo
    y = filtered_train_set[target]

    # Dividir en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Crear y entrenar el modelo Random Forest
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)

    # Predecir en el conjunto de prueba
    y_pred = rf_model.predict(X_test)

    # Evaluar el modelo
    confusion = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    return confusion, report, rf_model

if __name__ == "__main__":
    # Aplicar regresión logística con todas las características
    confusion_logistic_all, report_logistic_all, logistic_model_all, scaler_all = logistic_regression_all_features(train_set)
    print("Confusion Matrix - Logistic Regression (All Features):")
    print(confusion_logistic_all)
    print("\nClassification Report - Logistic Regression (All Features):")
    print(report_logistic_all)

    # Aplicar Random Forest con todas las características
    confusion_rf_all, report_rf_all, rf_model_all = random_forest_all_features(train_set)
    print("Confusion Matrix - Random Forest (All Features):")
    print(confusion_rf_all)
    print("\nClassification Report - Random Forest (All Features):")
    print(report_rf_all)

    # Guardar los modelos entrenados con todas las características
    save_models(logistic_model_all, rf_model_all)
    
    # eh dejado estos modelos bajo el nombre de  ..._featureless para poder reciclar save_models en models_all.py