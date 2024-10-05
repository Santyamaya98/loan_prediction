import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# Local imports
from analisis import train_set, test_set  # Asegúrate de que target esté definido en analisis

target = 'loan_status'  # Asegúrate de que la variable objetivo esté definida

def drop_noisy_columns(relevant_features_train, relevant_features_test, train_set, test_set):
    # Eliminar duplicados en train_set y test_set
    train_set = train_set.loc[~train_set.index.duplicated(keep='first')]
    test_set = test_set.loc[~test_set.index.duplicated(keep='first')]
    
    # Filtrar el conjunto de datos para mantener solo las columnas relevantes
    filtered_train_set = train_set[relevant_features_train + [target]]
    filtered_test_set = test_set[relevant_features_test]
    
    return filtered_train_set, filtered_test_set

def logistic_regression_with_threshold(train_set, features, target='loan_status'):
    # Eliminar duplicados antes de aplicar la condición
    train_set = train_set.loc[~train_set.index.duplicated(keep='first')]
    filtered_train_set = train_set[train_set[target] < 5]

    # Separar características y objetivo
    X = filtered_train_set[features]  # Solo características
    y = filtered_train_set[target]     # Variable objetivo

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

def random_forest_with_threshold(train_set, features, target='loan_status'):
    # Eliminar duplicados antes de aplicar la condición
    train_set = train_set.loc[~train_set.index.duplicated(keep='first')]
    filtered_train_set = train_set[train_set[target] < 5]

    # Separar características y objetivo
    X = filtered_train_set[features]
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

def save_models(logistic_model, rf_model):
    # Guardar los modelos entrenados de regresión logística y Random Forest
    with open('logistic_model.pkl', 'wb') as f:
        pickle.dump(logistic_model, f)

    with open('rf_model.pkl', 'wb') as f:
        pickle.dump(rf_model, f)

if __name__ == "__main__":
    # Definir características relevantes
    relevant_features_train = [
        'loan_grade',
        'loan_percent_income',
        'person_home_ownership',
        'loan_intent_DEBTCONSOLIDATION',
        'loan_intent_HOMEIMPROVEMENT',
    ]

    relevant_features_test = relevant_features_train  # Las mismas para prueba

    # Filtrar los conjuntos de datos para incluir solo columnas relevantes
    train_set, test_set = drop_noisy_columns(relevant_features_train, relevant_features_test, train_set, test_set)

    # Aplicar regresión logística
    confusion_logistic, report_logistic, logistic_model, scaler = logistic_regression_with_threshold(train_set, relevant_features_train)
    print("Confusion Matrix - Logistic Regression:")
    print(confusion_logistic)
    print("\nClassification Report - Logistic Regression:")
    print(report_logistic)

    # Aplicar Random Forest
    confusion_rf, report_rf, rf_model = random_forest_with_threshold(train_set, relevant_features_train)
    print("Confusion Matrix - Random Forest:")
    print(confusion_rf)
    print("\nClassification Report - Random Forest:")
    print(report_rf)

    # Guardar los modelos entrenados
    save_models(logistic_model, rf_model)
