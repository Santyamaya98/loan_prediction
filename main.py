import pickle
from sklearn.preprocessing import StandardScaler
import plotly.express as px

#local 
from analisis import train_set, test_set, submission

def load_models():
    """
    Carga los modelos guardados desde los archivos locales.
    """
    with open('logistic_model.pkl', 'rb') as f:
        logistic_model = pickle.load(f)

    with open('rf_model.pkl', 'rb') as f:
        rf_model = pickle.load(f)

    return logistic_model, rf_model

def make_predictions(logistic_model, rf_model, scaler, test):
    # Scale the test set first
    test_scaled = scaler.transform(test)  # Use the already fitted scaler

    # Make predictions with Logistic Regression
    logistic_predictions = logistic_model.predict(test_scaled)
    rf_predictions = rf_model.predict(test_scaled)
    
    print("Logistic Regression Predictions:", logistic_predictions)
    print("Random Forest Predictions:", rf_predictions)
    
    return logistic_predictions, rf_predictions

def scalar_test_set(test_set, scaler):
    test_scaled = scaler.transform(test_set)  # Use the already fitted scaler
    return test_scaled

def scalar_test_set(test_set, scaler):
    test_scaled = scaler.transform(test_set)  # Use the already fitted scaler
    return test_scaled

if __name__ == "__main__":
    # Cargar los modelos
    logistic_model, rf_model = load_models()

    # Relevant features used in training
    relevant_features_train = [
        'loan_grade',
        'loan_percent_income',
        'person_home_ownership',
        'loan_intent_DEBTCONSOLIDATION',
        'loan_intent_HOMEIMPROVEMENT',
    ]

    # Filtrar las características relevantes en el conjunto de prueba
    test_set = test_set[relevant_features_train]

    # Ajustar el escalador al conjunto de entrenamiento
    scaler = StandardScaler()
    
    # Asegúrate de que train_set solo contenga las características relevantes
    train_set_filtered = train_set[relevant_features_train]

    # Ajustar el escalador con el conjunto de entrenamiento
    scaler.fit(train_set_filtered)  # Ajustar el escalador

    # Escalar el conjunto de prueba
    test_scaled = scalar_test_set(test_set, scaler)

    # Hacer predicciones
    logistic_predictions, rf_predictions = make_predictions(logistic_model, rf_model, scaler, test_scaled)
    '''
    submission['loan_status'] = rf_predictions 
    submission.to_csv('rf_predictions.csv', index = False)
    '''
    submission['loan_status'] = logistic_predictions
    submission.to_csv('log_predictions.csv', index = False)


    # Example usage
    fig = px.histogram(x=logistic_predictions, title='Logistic Regression Predictions')
    fig.show()



    # Example usage
    fig = px.histogram(x=rf_predictions, title='Logistic Regression Predictions')
    fig.show()
