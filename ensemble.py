import pickle
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier  # Importar VotingClassifier para el ensamble
from analisis import train_set

def train_and_save_ensemble_model(train_set, relevant_features_train):
    # Cargar el modelo XGBoost usando pickle
    with open('xgb_model.pkl', 'rb') as f:
        xgb_model = pickle.load(f)

    # Cargar el modelo de regresión logística usando pickle
    with open('logistic_model_featureless.pkl', 'rb') as f:
        logistic_model = pickle.load(f)

    # Preparar el conjunto de características y la variable objetivo
    X = train_set[relevant_features_train]
    y = train_set['loan_status']

    # Dividir el conjunto de datos en entrenamiento y validación
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Escalar las características
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Crear el modelo de ensamble utilizando VotingClassifier
    ensemble_model = VotingClassifier(estimators=[
        ('logistic', logistic_model),
        ('xgb', xgb_model)
    ], voting='soft')

    # Entrenar el modelo de ensamble
    ensemble_model.fit(X_train_scaled, y_train)

    # Guardar el modelo de ensamble usando pickle
    with open('ensemble.pkl', 'wb') as f:
        pickle.dump(ensemble_model, f)
    print("Ensemble model saved as 'ensemble.pkl'.")

    # Hacer predicciones con el conjunto de validación
    y_pred_val = ensemble_model.predict(X_val_scaled)

    # Evaluar las predicciones del ensemble
    confusion_ensemble = confusion_matrix(y_val, y_pred_val)
    report_ensemble = classification_report(y_val, y_pred_val)

    # Imprimir los resultados
    print("Confusion Matrix - Ensemble:")
    print(confusion_ensemble)
    print("\nClassification Report - Ensemble:")
    print(report_ensemble)

# Ejemplo de uso:
relevant_features_train = [
    'loan_grade',
    'loan_percent_income',
    'person_home_ownership',
    'loan_intent_DEBTCONSOLIDATION',
    'loan_intent_HOMEIMPROVEMENT',
]
train_and_save_ensemble_model(train_set, relevant_features_train)
