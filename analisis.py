# here I'll explore the dataset 
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from scipy import stats
from scipy.stats import chi2_contingency
#lets take a deep look into train data

train_set = pd.read_csv('train.csv')
test_set = pd.read_csv('test.csv')
submission = pd.read_csv('sample_submission.csv')
target = 'loan_status'
'''
print(train_set.head())
print(train_set.describe())
print(train_set.info())
print(train_set.columns)

print(train_set[['person_age', 'person_income', 'person_home_ownership',
                  'person_emp_length', 'loan_intent', 'loan_grade', 'loan_amnt',
                  'loan_int_rate', 'loan_percent_income', 'cb_person_default_on_file',
                  'cb_person_cred_hist_length', 'loan_status']].describe())


   id  person_age  person_income person_home_ownership  person_emp_length  ... loan_int_rate loan_percent_income  cb_person_default_on_file  cb_person_cred_hist_length  loan_status
0   0          37          35000                  RENT                0.0  ...         11.49                0.17                          N                          14            0
1   1          22          56000                   OWN                6.0  ...         13.35                0.07                          N                           2            0
2   2          29          28800                   OWN                8.0  ...          8.90                0.21                          N                          10            0
3   3          30          70000                  RENT               14.0  ...         11.11                0.17                          N                           5            0
4   4          22          60000                  RENT                2.0  ...          6.92                0.10                          N                           3            0

[5 rows x 13 columns]
                 id    person_age  person_income  person_emp_length     loan_amnt  loan_int_rate  loan_percent_income  cb_person_cred_hist_length   loan_status
count  58645.000000  58645.000000   5.864500e+04       58645.000000  58645.000000   58645.000000         58645.000000                58645.000000  58645.000000
mean   29322.000000     27.550857   6.404617e+04           4.701015   9217.556518      10.677874             0.159238                    5.813556      0.142382
std    16929.497605      6.033216   3.793111e+04           3.959784   5563.807384       3.034697             0.091692                    4.029196      0.349445
min        0.000000     20.000000   4.200000e+03           0.000000    500.000000       5.420000             0.000000                    2.000000      0.000000
25%    14661.000000     23.000000   4.200000e+04           2.000000   5000.000000       7.880000             0.090000                    3.000000      0.000000
50%    29322.000000     26.000000   5.800000e+04           4.000000   8000.000000      10.750000             0.140000                    4.000000      0.000000
75%    43983.000000     30.000000   7.560000e+04           7.000000  12000.000000      12.990000             0.210000                    8.000000      0.000000
max    58644.000000    123.000000   1.900000e+06         123.000000  35000.000000      23.220000             0.830000                   30.000000      1.000000
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 58645 entries, 0 to 58644
Data columns (total 13 columns):
 #   Column                      Non-Null Count  Dtype  
---  ------                      --------------  -----  
 0   id                          58645 non-null  int64  
 1   person_age                  58645 non-null  int64  
 2   person_income               58645 non-null  int64  
 3   person_home_ownership       58645 non-null  object 
 4   person_emp_length           58645 non-null  float64
 5   loan_intent                 58645 non-null  object 
 6   loan_grade                  58645 non-null  object 
 7   loan_amnt                   58645 non-null  int64  
 8   loan_int_rate               58645 non-null  float64
 9   loan_percent_income         58645 non-null  float64
 10  cb_person_default_on_file   58645 non-null  object 
 11  cb_person_cred_hist_length  58645 non-null  int64  
 12  loan_status                 58645 non-null  int64  
dtypes: float64(3), int64(6), object(4)
memory usage: 5.8+ MB
None
Index(['id', 'person_age', 'person_income', 'person_home_ownership',
       'person_emp_length', 'loan_intent', 'loan_grade', 'loan_amnt',
       'loan_int_rate', 'loan_percent_income', 'cb_person_default_on_file',
       'cb_person_cred_hist_length', 'loan_status'],
      dtype='object')
         person_age  person_income  person_emp_length     loan_amnt  loan_int_rate  loan_percent_income  cb_person_cred_hist_length   loan_status
count  58645.000000   5.864500e+04       58645.000000  58645.000000   58645.000000         58645.000000                58645.000000  58645.000000
mean      27.550857   6.404617e+04           4.701015   9217.556518      10.677874             0.159238                    5.813556      0.142382
std        6.033216   3.793111e+04           3.959784   5563.807384       3.034697             0.091692                    4.029196      0.349445
min       20.000000   4.200000e+03           0.000000    500.000000       5.420000             0.000000                    2.000000      0.000000
25%       23.000000   4.200000e+04           2.000000   5000.000000       7.880000             0.090000                    3.000000      0.000000
50%       26.000000   5.800000e+04           4.000000   8000.000000      10.750000             0.140000                    4.000000      0.000000
75%       30.000000   7.560000e+04           7.000000  12000.000000      12.990000             0.210000                    8.000000      0.000000


# lets work with object data type 
print(train_set[['person_home_ownership','loan_intent','loan_grade','cb_person_default_on_file']])
prepare categorical values 
      person_home_ownership loan_intent loan_grade cb_person_default_on_file
0                      RENT   EDUCATION          B                         N
1                       OWN     MEDICAL          C                         N
2                       OWN    PERSONAL          A                         N
3                      RENT     VENTURE          B                         N
4                      RENT     MEDICAL          A                         N
...                     ...         ...        ...                       ...
58640              MORTGAGE   EDUCATION          D                         Y
58641                  RENT     MEDICAL          C                         N
58642                  RENT   EDUCATION          D                         N
58643                  RENT   EDUCATION          A                         N
58644              MORTGAGE     VENTURE          B                         N
'''

# binary categorial to boolean
train_set['cb_person_default_on_file'] = train_set['cb_person_default_on_file'].map({'Y':True, 'N':False}) 
test_set['cb_person_default_on_file'] = test_set['cb_person_default_on_file'].map({'Y':True, 'N':False}) 
# label encoder to assign a numeric etiquet on each item on ['person_home_ownership']
label_encoder = LabelEncoder()
train_set['person_home_ownership'] = label_encoder.fit_transform(train_set['person_home_ownership'])
test_set['person_home_ownership'] =  label_encoder.fit_transform(test_set['person_home_ownership'])
# assign a numerical value for loan_intent
# Assuming there is a logical order in the 'loan_grade', The data was exp[lored in pwer Bi and we find 7 diferentes loan_grades that caused missing values 
train_set['loan_grade'] = train_set['loan_grade'].map({'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E' : 4, 'F' : 5, 'G': 6})
test_set['loan_grade'] = test_set['loan_grade'].map({'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E' : 4, 'F' : 5, 'G': 6})
# one hot encoding for loan_intent
train_set = pd.get_dummies(train_set, columns=['loan_intent'])
test_set = pd.get_dummies(test_set, columns=['loan_intent'])


print(train_set)
print(test_set)

'''
          id  person_age  person_income  person_home_ownership  ...  loan_intent_HOMEIMPROVEMENT  loan_intent_MEDICAL  loan_intent_PERSONAL  loan_intent_VENTURE
0          0          37          35000                      3  ...                        False                False                 False                False
1          1          22          56000                      2  ...                        False                 True                 False                False
2          2          29          28800                      2  ...                        False                False                  True                False
3          3          30          70000                      3  ...                        False                False                 False                 True
4          4          22          60000                      3  ...                        False                 True                 False                False
...      ...         ...            ...                    ...  ...                          ...                  ...                   ...                  ...
58640  58640          34         120000                      0  ...                        False                False                 False                False
58641  58641          28          28800                      3  ...                        False                 True                 False                False
58642  58642          23          44000                      3  ...                        False                False                 False                False
58643  58643          22          30000                      3  ...                        False                False                 False                False
58644  58644          31          75000                      0  ...                        False                False                 False                 True

[58645 rows x 18 columns]
          id  person_age  person_income  person_home_ownership  ...  loan_intent_HOMEIMPROVEMENT  loan_intent_MEDICAL  loan_intent_PERSONAL  loan_intent_VENTURE
0      58645          23          69000                      3  ...                         True                False                 False                False
1      58646          26          96000                      0  ...                        False                False                  True                False
2      58647          26          30000                      3  ...                        False                False                 False                 True
3      58648          33          50000                      3  ...                        False                False                 False                False
4      58649          26         102000                      0  ...                         True                False                 False                False
...      ...         ...            ...                    ...  ...                          ...                  ...                   ...                  ...
39093  97738          22          31200                      0  ...                        False                False                 False                False
39094  97739          22          48000                      0  ...                        False                False                 False                False
39095  97740          51          60000                      0  ...                        False                False                  True                False
39096  97741          22          36000                      0  ...                        False                False                  True                False
39097  97742          31          45000                      3  ...                        False                False                 False                False

[39098 rows x 17 columns]
'''

# lets search for missing data 
# Check for missing data in the training set
missing_train = train_set.isnull().sum()
print("Missing values in training set:")
print(missing_train[missing_train > 0])  # Print only columns with missing values

# Check for missing data in the test set
missing_test = test_set.isnull().sum()
print("\nMissing values in test set:")
print(missing_test[missing_test > 0])  # Print only columns with missing values

'''
[39098 rows x 17 columns]
Missing values in training set:
Series([], dtype: int64)

Missing values in test set:
Series([], dtype: int64)
'''

# t-students proof 

group1 = train_set[train_set['loan_status'] == 1]['person_income']  # Ingresos de aprobados
group2 = train_set[train_set['loan_status'] == 0]['person_income']  # Ingresos de no aprobados

# Realizar la prueba t
t_stat, p_value = stats.ttest_ind(group1, group2)

print(f'T-statistic: {t_stat}, P-value: {p_value}')


'''
loan_status      0     1
loan_grade              
0            19952  1032
1            18313  2087
2             9542  1494
3             2046  2988
4              378   631
5               58    91
6                6    27
T-statistic: -41.76473773511475, P-value: 0.0
T-statistic:

The T-statistic of -41.76 indicates a very strong difference between the two groups.
 A negative value suggests that the average income in the group with loan status 0 (not approved)
 is significantly higher than in the group with loan status 1 (approved).


P-value:

The P-value of 0.0 indicates that the result is highly statistically significant. 
Since the P-value is much less than the typical significance level of 0.05, you can reject the null hypothesis.
This means there is strong evidence to conclude that there is a 
significant difference in person_income between those whose loans were approved and those whose loans were not approved.
'''


# Create a sample dataset
data = {
    'loan_grade': ['0', '0', '1', '1', '2', '2', '3', '3', '4', '4', '5', '5', '6', '6'],
    'loan_status': ['Approved', 'Not Approved', 'Approved', 'Not Approved',
                   'Approved', 'Not Approved', 'Approved', 'Not Approved',
                   'Approved', 'Not Approved', 'Approved', 'Not Approved',
                   'Approved', 'Not Approved'],
    'count': [19952, 1032, 18313, 2087, 9542, 1494, 2046, 2988, 378, 631, 58, 91, 6, 27]
}





'''
chi cuadrado confirma matriz de correlacion 
Chi-Squared Statistic: 12476.25

This is a large value, indicating a significant difference between the observed and expected frequencies in your contingency table. A higher Chi-Squared statistic generally suggests that the two categorical variables are associated.
P-Value: 0.0

The P-value being 0.0 (often represented as less than 0.0001) indicates that the probability of observing such a difference (or a more extreme difference) between the observed and expected frequencies by chance is virtually zero. This strongly suggests that there is a significant association between the variables you're testing (in this case, loan_grade and loan_status).
Degrees of Freedom: 6

Degrees of freedom for the test are calculated as 
(number of rows−1)×(number of columns−1). In your case, it indicates that you have a well-defined contingency table structure.
Expected Frequencies:

These values represent the counts you would expect in each cell of your contingency table if there were no association between the loan_grade and loan_status. For instance, if the observed count in loan_grade 0 and loan_status Approved is significantly higher than the expected frequency, it indicates a stronger association in that category.
Conclusion
Given the Chi-Squared statistic and P-value, you can conclude that there is a significant relationship between loan_grade and loan_status. This suggests that the grade of a loan is likely influencing whether it gets approved or not.
'''

