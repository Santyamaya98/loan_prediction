import tensorflow as tf
import xgboost as xgb
import pickle
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import SMOTE  # Para manejar el desbalance de clases
from analisis import train_set, test_set# Asegúrate de importar las variables necesarias


def train_and_predict(train_data, test_data, target_column='loan_status', submission_file='submission.csv'):
    # Preparar el set de entrenamiento completo
    X = train_data.drop(columns=[target_column])
    y = train_data[target_column] 

    # Dividir el conjunto de entrenamiento en entrenamiento y validación
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Escalar las características
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Manejar desbalance de clases usando SMOTE
    smote = SMOTE()
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

    # ======================= MODELO XGBOOST =======================
    # Crear y entrenar el modelo XGBoost
    xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    xgb_model.fit(X_train_resampled, y_train_resampled)

    # Hacer predicciones con el conjunto de validación
    y_pred_xgb = xgb_model.predict(X_val_scaled)

    # Evaluar el modelo XGBoost
    print("XGBoost Evaluation")
    print("Confusion Matrix - XGBoost:")
    print(confusion_matrix(y_val, y_pred_xgb))
    print("\nClassification Report - XGBoost:")
    print(classification_report(y_val, y_pred_xgb))

    # ======================= RED NEURONAL =======================
    # Crear y entrenar el modelo de Red Neuronal
    model_nn = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Para clasificación binaria
    ])

    # Compilar el modelo
    model_nn.compile(optimizer='adam',
                     loss='binary_crossentropy',
                     metrics=['accuracy'])

    # Entrenar el modelo
    model_nn.fit(X_train_resampled, y_train_resampled, epochs=10, validation_data=(X_val_scaled, y_val))

    # Evaluar el modelo de Red Neuronal
    print("Red Neuronal Evaluation")
    y_pred_nn = (model_nn.predict(X_val_scaled) > 0.5).astype(int)
    print("Confusion Matrix - Red Neuronal:")
    print(confusion_matrix(y_val, y_pred_nn))
    print("\nClassification Report - Red Neuronal:")
    print(classification_report(y_val, y_pred_nn))

    # ======================= PREDICCIÓN CON EL TEST SET =======================
    # Preparar el conjunto de prueba
    X_test = test_data
    X_test_scaled = scaler.transform(X_test)

    # Predicciones con el modelo de XGBoost
    y_pred_test_xgb = xgb_model.predict(X_test_scaled)

    # Predicciones con el modelo de Red Neuronal
    y_pred_test_nn = (model_nn.predict(X_test_scaled) > 0.5).astype(int)

    # Crear un DataFrame para la predicción de XGBoost
    submission_xgb = pd.DataFrame({
        'id': test_data['id'],
        'loan_status': y_pred_test_xgb
    })

    # Crear un DataFrame para la predicción de Red Neuronal
    submission_nn = pd.DataFrame({
        'id': test_data['id'],
        'loan_status': y_pred_test_nn.flatten()  # Aplano para tener un formato adecuado
    })

    # Guardar ambas predicciones como archivos CSV
    submission_xgb.to_csv(submission_file.replace('.csv', '_xgb.csv'), index=False)
    submission_nn.to_csv(submission_file.replace('.csv', '_nn.csv'), index=False)
    print(f"Archivos de predicción creados: {submission_file.replace('.csv', '_xgb.csv')} y {submission_file.replace('.csv', '_nn.csv')}")

# Llamar a la función con el conjunto de entrenamiento y prueba
train_and_predict(train_set, test_set)

'''(env) usuario@usuario-System-Product-Name:~/hoja_vida/kaggle_projects/loan_prediction$ python3 nn_xgboost.py
2024-10-17 17:23:19.268873: I external/local_xla/xla/tsl/cuda/cudart_stub.cc:32] Could not find cuda drivers on your machine, GPU will not be used.
2024-10-17 17:23:19.271756: I external/local_xla/xla/tsl/cuda/cudart_stub.cc:32] Could not find cuda drivers on your machine, GPU will not be used.
2024-10-17 17:23:19.279103: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:485] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
2024-10-17 17:23:19.290765: E external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:8454] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
2024-10-17 17:23:19.294064: E external/local_xla/xla/stream_executor/cuda/cuda_blas.cc:1452] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
2024-10-17 17:23:19.303244: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2024-10-17 17:23:19.958244: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
          id  person_age  ...  loan_intent_PERSONAL  loan_intent_VENTURE
0          0          37  ...                 False                False
1          1          22  ...                 False                False
2          2          29  ...                  True                False
3          3          30  ...                 False                 True
4          4          22  ...                 False                False
...      ...         ...  ...                   ...                  ...
58640  58640          34  ...                 False                False
58641  58641          28  ...                 False                False
58642  58642          23  ...                 False                False
58643  58643          22  ...                 False                False
58644  58644          31  ...                 False                 True

[58645 rows x 18 columns]
          id  person_age  ...  loan_intent_PERSONAL  loan_intent_VENTURE
0      58645          23  ...                 False                False
1      58646          26  ...                  True                False
2      58647          26  ...                 False                 True
3      58648          33  ...                 False                False
4      58649          26  ...                 False                False
...      ...         ...  ...                   ...                  ...
39093  97738          22  ...                 False                False
39094  97739          22  ...                 False                False
39095  97740          51  ...                  True                False
39096  97741          22  ...                  True                False
39097  97742          31  ...                 False                False

[39098 rows x 17 columns]
Missing values in training set:
Series([], dtype: int64)

Missing values in test set:
Series([], dtype: int64)
T-statistic: -41.76473773511475, P-value: 0.0
/home/usuario/hoja_vida/kaggle_projects/env/lib/python3.12/site-packages/xgboost/core.py:158: UserWarning: [17:23:21] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "use_label_encoder" } are not used.

  warnings.warn(smsg, UserWarning)
XGBoost Evaluation
Confusion Matrix - XGBoost:
[[9938  149]
 [ 410 1232]]

Classification Report - XGBoost:
              precision    recall  f1-score   support

           0       0.96      0.99      0.97     10087
           1       0.89      0.75      0.82      1642

    accuracy                           0.95     11729
   macro avg       0.93      0.87      0.89     11729
weighted avg       0.95      0.95      0.95     11729

/home/usuario/hoja_vida/kaggle_projects/env/lib/python3.12/site-packages/keras/src/layers/core/dense.py:87: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
  super().__init__(activity_regularizer=activity_regularizer, **kwargs)
Epoch 1/10
   1/2513 ━━━━━━━━━━━━━━━━━━━━ 22:23 535ms/step - accuracy: 0.3750 - loss: 0.791   2/2513 ━━━━━━━━━━━━━━━━━━━━ 3s 1ms/step - accuracy: 0.3984 - loss: 0.7919    2513/2513 ━━━━━━━━━━━━━━━━━━━━ 2s 470us/step - accuracy: 0.8177 - loss: 0.4033 - val_accuracy: 0.8949 - val_loss: 0.3088
Epoch 2/10
2513/2513 ━━━━━━━━━━━━━━━━━━━━ 1s 445us/step - accuracy: 0.8568 - loss: 0.3359 - val_accuracy: 0.9061 - val_loss: 0.3020
Epoch 3/10
2513/2513 ━━━━━━━━━━━━━━━━━━━━ 1s 448us/step - accuracy: 0.8588 - loss: 0.3217 - val_accuracy: 0.9083 - val_loss: 0.2913
Epoch 4/10
2513/2513 ━━━━━━━━━━━━━━━━━━━━ 1s 450us/step - accuracy: 0.8624 - loss: 0.3120 - val_accuracy: 0.9085 - val_loss: 0.2942
Epoch 5/10
2513/2513 ━━━━━━━━━━━━━━━━━━━━ 1s 445us/step - accuracy: 0.8653 - loss: 0.3077 - val_accuracy: 0.9122 - val_loss: 0.2880
Epoch 6/10
2513/2513 ━━━━━━━━━━━━━━━━━━━━ 1s 445us/step - accuracy: 0.8691 - loss: 0.3010 - val_accuracy: 0.9041 - val_loss: 0.2913
Epoch 7/10
2513/2513 ━━━━━━━━━━━━━━━━━━━━ 1s 445us/step - accuracy: 0.8685 - loss: 0.3000 - val_accuracy: 0.9131 - val_loss: 0.2848
Epoch 8/10
2513/2513 ━━━━━━━━━━━━━━━━━━━━ 1s 443us/step - accuracy: 0.8698 - loss: 0.2968 - val_accuracy: 0.9160 - val_loss: 0.2763
Epoch 9/10
2513/2513 ━━━━━━━━━━━━━━━━━━━━ 1s 446us/step - accuracy: 0.8710 - loss: 0.2931 - val_accuracy: 0.9128 - val_loss: 0.2698
Epoch 10/10
2513/2513 ━━━━━━━━━━━━━━━━━━━━ 1s 454us/step - accuracy: 0.8734 - loss: 0.2880 - val_accuracy: 0.9194 - val_loss: 0.2688
Red Neuronal Evaluation
367/367 ━━━━━━━━━━━━━━━━━━━━ 0s 344us/step
Confusion Matrix - Red Neuronal:
[[9507  580]
 [ 365 1277]]

Classification Report - Red Neuronal:
              precision    recall  f1-score   support

           0       0.96      0.94      0.95     10087
           1       0.69      0.78      0.73      1642

    accuracy                           0.92     11729
   macro avg       0.83      0.86      0.84     11729
weighted avg       0.92      0.92      0.92     11729

1222/1222 ━━━━━━━━━━━━━━━━━━━━ 0s 295us/step
Archivos de predicción creados: submission_xgb.csv y submission_nn.csv
'''