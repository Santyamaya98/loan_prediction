import pickle
from sklearn.preprocessing import StandardScaler
import plotly.express as px

#local 
from analisis import train_set, test_set, submission
from main import load_models

def make_predictions(logistic_model, rf_model, scaler, test):
    # Escalar el conjunto de prueba
    test_scaled = scaler.transform(test)  # Usa el escalador previamente ajustado

    # Hacer predicciones con Logistic Regression y Random Forest
    logistic_predictions = logistic_model.predict(test_scaled)
    rf_predictions = rf_model.predict(test_scaled)
    
    print("Logistic Regression Predictions:", logistic_predictions)
    print("Random Forest Predictions:", rf_predictions)
    
    return logistic_predictions, rf_predictions

def scalar_test_set(test_set, scaler):
    test_scaled = scaler.transform(test_set)  # Usa el escalador previamente ajustado
    return test_scaled

if __name__ == "__main__":
    # Cargar los modelos
    logistic_model, rf_model = load_models()

    # Eliminar duplicados en el conjunto de datos de prueba, si hay
    test_set = test_set.loc[~test_set.index.duplicated(keep='first')]

    # Ajustar el escalador al conjunto de entrenamiento
    scaler = StandardScaler()
    
    # Preparar el conjunto de entrenamiento excluyendo la columna objetivo 'loan_status'
    train_set_filtered = train_set.drop(columns=['loan_status'])

    # Ajustar el escalador con el conjunto de entrenamiento
    scaler.fit(train_set_filtered)  # Ajustar el escalador a todos los features

    # Escalar el conjunto de prueba
    test_scaled = scalar_test_set(test_set, scaler)

    # Hacer predicciones
    logistic_predictions, rf_predictions = make_predictions(logistic_model, rf_model, scaler, test_scaled)

    # Guardar las predicciones en un archivo CSV (puedes elegir cuál quieres usar)
    
    submission['loan_status'] = rf_predictions
    submission.to_csv('rf_predictions_all_features.csv', index=False)
    
    submission['loan_status'] = logistic_predictions
    submission.to_csv('log_predictions_all_features.csv', index=False)

    # Visualización de las predicciones
    fig = px.histogram(x=logistic_predictions, title='Logistic Regression Predictions (All Features)')
    fig.show()

    fig = px.histogram(x=rf_predictions, title='Random Forest Predictions (All Features)')
    fig.show()
