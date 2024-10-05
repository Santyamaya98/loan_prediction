import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Local import (Assuming train_set is a DataFrame and target is a string with the target column name)
from analisis import train_set, target

def evaluate_logistic_regression_with_thresholds(train_set, target, features, start_threshold=5, end_threshold=9, step=1):
    """
    Evaluates a logistic regression model across a range of thresholds.

    Parameters:
    - train_set (pd.DataFrame): The training dataset.
    - target (str): The target variable name.
    - features (list): List of feature names.
    - start_threshold (int): Start of the threshold range (inclusive, multiplied by 0.1).
    - end_threshold (int): End of the threshold range (inclusive, multiplied by 0.1).
    - step (int): Step size for the threshold increment (in tenths).

    Returns:
    - dict: A dictionary with threshold as key and F1 score as value, along with the best threshold.
    - np.ndarray: Scaled training features.
    - np.ndarray: Scaled testing features.
    """

    # Prepare data
    X = train_set[features]
    y = train_set[target]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize the Logistic Regression model
    model = LogisticRegression(max_iter=300)

    # Fit the model on the training data
    model.fit(X_train_scaled, y_train)

    # Get predicted probabilities for the test data
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]  # Probabilities for class 1

    best_threshold = None
    best_f1_score = 0
    results = {}

    # Iterate over thresholds
    for i in range(start_threshold, end_threshold + 1, step):
        threshold = i / 10  # Convert to float
        # Apply the custom threshold
        y_pred_custom = (y_pred_proba >= threshold).astype(int)

        # Calculate the F1 score
        current_f1_score = f1_score(y_test, y_pred_custom)

        # Store the result
        results[threshold] = current_f1_score

        # Update the best threshold
        if current_f1_score > best_f1_score:
            best_f1_score = current_f1_score
            best_threshold = threshold

    # Final evaluation using the best threshold
    y_pred_final = (y_pred_proba >= best_threshold).astype(int)
    confusion = confusion_matrix(y_test, y_pred_final)
    report = classification_report(y_test, y_pred_final)

    return results, best_threshold, confusion, report, X_train_scaled, y_train

def plot_feature_importance(model, feature_names, title):
    """Plots feature importance based on the model."""
    feature_importances = pd.Series(model.feature_importances_, index=feature_names)
    feature_importances = feature_importances.sort_values(ascending=False)

    plt.figure(figsize=(12, 6))
    feature_importances.plot(kind='bar')
    plt.title(title)
    plt.ylabel('Importance Score')
    plt.show()
    
    return feature_importances  # Return feature importances for further use

if __name__ == '__main__':
    # Example usage of the evaluate function
    features = ['person_age', 'person_income', 'person_home_ownership',
                'person_emp_length', 'loan_grade', 'loan_amnt', 'loan_int_rate',
                'loan_percent_income', 'cb_person_default_on_file',
                'cb_person_cred_hist_length',
                'loan_intent_DEBTCONSOLIDATION', 'loan_intent_EDUCATION',
                'loan_intent_HOMEIMPROVEMENT', 'loan_intent_MEDICAL',
                'loan_intent_PERSONAL', 'loan_intent_VENTURE']

    results, best_threshold, confusion, report, X_train_scaled, y_train = evaluate_logistic_regression_with_thresholds(train_set, target, features)
    print("Thresholds and their corresponding F1 scores:", results)
    print("Best threshold based on F1 score:", best_threshold)
    print("\nConfusion Matrix:")
    print(confusion)
    print("\nClassification Report:")
    print(report)

    # Coefficients for Logistic Regression
    logistic_model = LogisticRegression(max_iter=300)
    logistic_model.fit(X_train_scaled, y_train)  # Using the scaled training data
    logistic_coefs = pd.Series(logistic_model.coef_[0], index=features)
    logistic_coefs = logistic_coefs.sort_values(ascending=False)

    # Plotting Logistic Regression Feature Importance with positive coefficients
    plt.figure(figsize=(12, 6))
    logistic_coefs[logistic_coefs > 0].plot(kind='bar')
    plt.title('Logistic Regression Feature Importance (Positive Coefficients)')
    plt.ylabel('Coefficient Value')
    plt.show()

    # Random Forest for Feature Importance
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train_scaled, y_train)

    # Plotting Random Forest Feature Importance
    rf_feature_importances = plot_feature_importance(rf_model, features, 'Random Forest Feature Importance')

    # Print the top 5 most important features for logistic regression
    print("\nTop 5 Most Relevant Features from Log Regression:")
    print(logistic_coefs.head(5))

    # Print the top 5 most important features for Random Forest
    print("\nTop 5 Most Relevant Features from Random Forest:")
    print(rf_feature_importances.head(5))


'''
most important values
Top 5 Most Relevant Features from Log Regression:
loan_grade                       1.269336
loan_percent_income              1.149557
person_home_ownership            0.506831
loan_intent_DEBTCONSOLIDATION    0.209082
loan_intent_HOMEIMPROVEMENT      0.157785
dtype: float64

Top 5 Most Relevant Features from Random Forest:
loan_percent_income      0.238804
loan_int_rate            0.130108
loan_grade               0.126709
person_income            0.108314
person_home_ownership    0.090805
dtype: float64
'''