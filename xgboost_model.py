import xgboost as xgb
import pickle
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


from analisis import train_set, test_set

# Definir características relevantes
relevant_features_train = [
    'loan_grade',
    'loan_percent_income',
    'person_home_ownership',
    'loan_intent_DEBTCONSOLIDATION',
    'loan_intent_HOMEIMPROVEMENT',
]

# Filtrar las características relevantes del conjunto de entrenamiento
X = train_set[relevant_features_train]
y = train_set['loan_status']  # Asumiendo que esta es tu variable objetivo

# Dividir el conjunto de entrenamiento en entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar las características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Crear y entrenar el modelo XGBoost
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train_scaled, y_train)

# Hacer predicciones con el conjunto de validación
y_pred_xgb = xgb_model.predict(X_val_scaled)

# Evaluar el modelo XGBoost
confusion_xgb = confusion_matrix(y_val, y_pred_xgb)
report_xgb = classification_report(y_val, y_pred_xgb)

print("Confusion Matrix - XGBoost:")
print(confusion_xgb)
print("\nClassification Report - XGBoost:")
print(report_xgb)

# Guardar el modelo usando pickle
with open('xgb_model.pkl', 'wb') as f:
    pickle.dump(xgb_model, f)

# Hacer predicciones en el conjunto de test
# Filtrar las características relevantes del conjunto de prueba
X_test = test_set[relevant_features_train]
X_test_scaled = scaler.transform(X_test)  # Escalar el conjunto de prueba

# Hacer predicciones
y_pred_test = xgb_model.predict(X_test_scaled)

# Crear un DataFrame para la sumisión
submission = pd.DataFrame({
    'id': test_set['id'],  # Asegúrate de que el test_set tiene una columna 'id'
    'loan_status': y_pred_test
})

# Guardar el DataFrame como un archivo CSV
submission.to_csv('xgb_model_submission.csv', index=False)
print("Submission file 'xgb_model_submission.csv' created successfully.")
