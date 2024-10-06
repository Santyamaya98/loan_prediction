import xgboost as xgb
import pickle
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import SMOTE  # To handle class imbalance
from sklearn.model_selection import cross_val_score

from analisis import train_set, test_set

# Define relevant features for the model
relevant_features_train = [
    'loan_grade',
    'loan_percent_income',
    'person_home_ownership',
    'loan_intent_DEBTCONSOLIDATION',
    'loan_intent_HOMEIMPROVEMENT',
]

# Filter relevant features from the training set
X = train_set[relevant_features_train]
y = train_set['loan_status']  # Assuming this is the target variable

# Split the training set into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features to have a mean of 0 and a standard deviation of 1
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit and transform
X_val_scaled = scaler.transform(X_val)  # Only transform using the fitted scaler

# Handle class imbalance using SMOTE
smote = SMOTE()
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

# Create and train the XGBoost model with default parameters
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train_resampled, y_train_resampled)  # Train the model with the resampled dataset

# Make predictions with the validation set
y_pred_xgb = xgb_model.predict(X_val_scaled)

# Evaluate the XGBoost model
confusion_xgb = confusion_matrix(y_val, y_pred_xgb)  # Confusion matrix
report_xgb = classification_report(y_val, y_pred_xgb)  # Classification report

# Print evaluation results
print("Confusion Matrix - XGBoost:")
print(confusion_xgb)
print("\nClassification Report - XGBoost:")
print(report_xgb)

# Save the model using pickle
with open('xgb_model.pkl', 'wb') as f:
    pickle.dump(xgb_model, f)

# Make predictions on the test set
X_test = test_set[relevant_features_train]  # Filter relevant features from the test set
X_test_scaled = scaler.transform(X_test)  # Scale the test set

# Make predictions
y_pred_test = xgb_model.predict(X_test_scaled)

# Create a DataFrame for submission
submission = pd.DataFrame({
    'id': test_set['id'],  # Ensure the test_set has an 'id' column
    'loan_status': y_pred_test  # Predictions of loan status
})

# Save the DataFrame as a CSV file
submission.to_csv('xgb_model_submission.csv', index=False)
print("Submission file 'xgb_model_submission.csv' created successfully.")

# Define the parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [100, 200],  # Number of trees
    'max_depth': [3, 5, 7],       # Maximum depth of the tree
    'learning_rate': [0.01, 0.1, 0.2],  # Learning rate
    'subsample': [0.8, 1.0],      # Proportion of samples to train each tree
    'colsample_bytree': [0.8, 1.0]  # Proportion of features to train each tree
}

# Perform hyperparameter search with cross-validation
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring='accuracy', cv=3)
grid_search.fit(X_train_resampled, y_train_resampled)  # Use the resampled dataset

# Get the best model and parameters
best_model = grid_search.best_estimator_
print("Best parameters:", grid_search.best_params_)

# Cross-validation with the best model
scores = cross_val_score(best_model, X_train_resampled, y_train_resampled, cv=5)
print("Cross-validation scores:", scores)
