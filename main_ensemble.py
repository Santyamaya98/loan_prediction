import pickle
from sklearn.preprocessing import StandardScaler
import plotly.express as px

# Local imports
from analisis import train_set, test_set, submission

def load_ensemble_model():
    """
    Carga el modelo de ensamble guardado desde el archivo local.
    """
    with open('ensemble.pkl', 'rb') as f:
        ensemble_model = pickle.load(f)

    return ensemble_model

def make_predictions(ensemble_model, scaler, test):
    # Escalar el conjunto de prueba primero
    test_scaled = scaler.transform(test)  # Usar el escalador ya ajustado

    # Hacer predicciones con el modelo de ensamble
    predictions = ensemble_model.predict(test_scaled)
    
    print("Ensemble Predictions:", predictions)
    
    return predictions

def scalar_test_set(test_set, scaler):
    # Escalar el conjunto de prueba
    test_scaled = scaler.transform(test_set)  # Usar el escalador ya ajustado
    return test_scaled

if __name__ == "__main__":
    # Cargar el modelo de ensamble
    ensemble_model = load_ensemble_model()

    # Características relevantes utilizadas en el entrenamiento
    relevant_features_train = [
        'loan_grade',
        'loan_percent_income',
        'person_home_ownership',
        'loan_intent_DEBTCONSOLIDATION',
        'loan_intent_HOMEIMPROVEMENT',
    ]

    # Filtrar las características relevantes en el conjunto de prueba
    test_set_filtered = test_set[relevant_features_train]

    # Ajustar el escalador al conjunto de entrenamiento
    scaler = StandardScaler()
    
    # Asegúrate de que train_set solo contenga las características relevantes
    train_set_filtered = train_set[relevant_features_train]

    # Ajustar el escalador con el conjunto de entrenamiento
    scaler.fit(train_set_filtered)  # Ajustar el escalador

    # Escalar el conjunto de prueba
    test_scaled = scalar_test_set(test_set_filtered, scaler)

    # Hacer predicciones
    ensemble_predictions = make_predictions(ensemble_model, scaler, test_scaled)

    # Guardar las predicciones en el archivo submission
    submission['loan_status'] = ensemble_predictions
    submission.to_csv('ensemble_predictions.csv', index=False)
    print("Predictions saved as 'ensemble_predictions.csv'.")

    # Visualizar las predicciones
    fig = px.histogram(x=ensemble_predictions, title='Ensemble Predictions')
    fig.show()
