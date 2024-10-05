from analisis import train_set, test_set

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.express as px

#local 
from main import logistic_predictions, rf_predictions

def plot_age_distribution(dataframe, age_column):
    """
    This function takes a DataFrame and the name of the age column,
    and plots a pie chart representing the distribution of each unique age.
    
    Parameters:
    dataframe (pd.DataFrame): The dataset containing the age data.
    age_column (str): The column name representing the age of the individuals.
    """
    # Count the occurrences of each unique age
    age_counts = dataframe[age_column].value_counts()

    # Plot the pie chart
    plt.figure(figsize=(10, 10))
    plt.pie(age_counts, labels=age_counts.index, autopct='%1.1f%%', startangle=90)
    plt.title('Age Distribution by Unique Age')
    plt.show()


def plot_loan_grade_income_vs_status(train_set, limit_amount=500000):
    """
    This function plots the relationship between loan_grade, person_income, and loan_status in the training set.
    
    Parameters:
    train_set (pd.DataFrame): The dataset used for training.
    limit_amount (int): The maximum income to filter the dataset.
    """
    # Filter the dataset based on the limit_amount
    filtered_data = train_set[train_set['person_income'] <= limit_amount]

    # Plotting with Seaborn for better visualization
    plt.figure(figsize=(12, 8))
    
    # Create a scatter plot showing loan_grade on x-axis, person_income on y-axis, and color by loan_status
    sns.scatterplot(x='loan_grade', y='person_income', hue='loan_status', data=filtered_data, palette='coolwarm', s=100)
    
    plt.title('Loan Grade vs Person Income and Loan Status (Income <= {})'.format(limit_amount))
    plt.xlabel('Loan Grade')
    plt.ylabel('Person Income')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()

def plot_approved_loans_by_grade_and_income(train_set):
    """
    This function plots the number of approved loans by loan_grade and income ranges in the training set.
    
    Parameters:
    train_set (pd.DataFrame): The dataset used for training.
    """
    # Create income bins with a range of 50,000
    bins = range(0, 200001, 50000)
    labels = [f'{i}-{i + 50000 - 1}' for i in bins[:-1]]  # Adjust label for better readability
    
    # Bin the person_income into categories
    train_set['income_group'] = pd.cut(train_set['person_income'], bins=bins, labels=labels, right=False)
    
    # Filter approved loans
    approved_loans = train_set[train_set['loan_status'] == 1]
    
    # Count approved loans by loan_grade and income_group
    counts = approved_loans.groupby(['loan_grade', 'income_group']).size().reset_index(name='approved_count')

    # Print the summary of approved loans
    print("Summary of Approved Loans by Loan Grade and Income Group:")
    print(counts.to_string(index=False))  # Print without index for better readability
    
    # Plotting the results
    plt.figure(figsize=(12, 8))
    sns.barplot(data=counts, x='income_group', y='approved_count', hue='loan_grade', palette='coolwarm')
    
    plt.title('Approved Loans by Loan Grade and Income Group')
    plt.xlabel('Income Group')
    plt.ylabel('Number of Approved Loans')
    plt.xticks(rotation=45)
    plt.legend(title='Loan Grade')
    
    plt.tight_layout()
    plt.show()

'''
 loan_grade  income_group  approved_count
          0       0-49999             724
          0   50000-99999             284
          0 100000-149999              20
          0 150000-199999               3
          1       0-49999            1382
          1   50000-99999             666
          1 100000-149999              32
          1 150000-199999               4
          2       0-49999            1016
          2   50000-99999             459
          2 100000-149999              17
          2 150000-199999               1
          3       0-49999            1591
          3   50000-99999            1266
          3 100000-149999             114
          3 150000-199999              14
          4       0-49999             321
          4   50000-99999             291
          4 100000-149999              18
          4 150000-199999               1
          5       0-49999              48
          5   50000-99999              42
          5 100000-149999               1
          5 150000-199999               0
          6       0-49999              19
          6   50000-99999               8
          6 100000-149999               0
          6 150000-199999               0
'''
def plot_histograms(train_set):
    train_set.hist(bins=30, figsize=(15, 10))
    plt.tight_layout()
    plt.show()


def plot_correlation_matrix(train_set):
    plt.figure(figsize=(12, 8))
    sns.heatmap(train_set.corr(), annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar=True)
    plt.title('Correlation Matrix')
    plt.show()

def plot_boxplots(train_set, column):
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='loan_grade', y=column, data=train_set)
    plt.title(f'Boxplot of {column} by Loan Grade')
    plt.show()

def plot_scatter(train_set, x_column, y_column):
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x=x_column, y=y_column, data=train_set, alpha=0.5)
    plt.title(f'Scatter Plot of {y_column} vs {x_column}')
    plt.show()



def plot_missing_values(train_set):
    missing_values = train_set.isnull().sum()
    missing_values = missing_values[missing_values > 0]
    plt.figure(figsize=(12, 6))
    sns.barplot(x=missing_values.index, y=missing_values.values)
    plt.title('Missing Values Count')
    plt.xticks(rotation=45)
    plt.show()




# Example usage
fig = px.histogram(x=logistic_predictions, title='Logistic Regression Predictions')
fig.show()



# Example usage
fig = px.histogram(x=rf_predictions, title='Logistic Regression Predictions')
fig.show()



