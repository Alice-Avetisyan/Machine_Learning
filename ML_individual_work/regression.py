import pandas as pd

insurance_data = pd.read_csv('insurance.csv')

'''
Problem setting ->> Using data, check how much a person is charged for the insurance 
'''

# Retrieving the data
insurance_data = pd.read_csv('insurance.csv')

# Checking the data - phase 1
print('-----Insurance data columns-----\n', insurance_data.columns)
print('-----Insurance data shape-----\n', insurance_data.shape)
print('-----Insurance data head (first 20 elements)-----\n', insurance_data.head(20))
print('-----Insurance data tail (last 10 elements)-----\n', insurance_data.tail(10))
# Checking the data - phase 2
print('-----Missing Data-----\n', insurance_data.isna().sum())
# if there are missing data use ->>  insurance_data.fillna(insurance_data.mean(), inplace=True)

print('-----Insurance data info-----\n', insurance_data.info())
print('-----Insurance data types-----\n', insurance_data.dtypes)
# noted columns with object type ->> sex, smoker, region

# converting object types
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
insurance_data['sex'] = le.fit_transform(insurance_data['sex'])
insurance_data['smoker'] = le.fit_transform(insurance_data['smoker'])
insurance_data['region'] = le.fit_transform(insurance_data['region'])

import matplotlib.pyplot as plt

# insurance_data.hist()
# plt.show()
#
# plt.matshow(insurance_data.corr())
# plt.xticks(range(insurance_data.shape[1]), insurance_data.columns, fontsize=12, rotation=90)
# plt.yticks(range(insurance_data.shape[1]), insurance_data, fontsize=12)
# clb = plt.colorbar()
# clb.ax.tick_params(labelsize=12)
# plt.show()

# X = insurance_data.drop('charges', axis=1) -->> og features
X = insurance_data.drop(['charges', 'sex', 'children', 'region'], axis=1)
y = insurance_data['charges']

# Splitting the data ->> train set and test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101, shuffle=True)

# Implementing Support Vector Regression
from sklearn.svm import SVR
# creating a mode with non-linear function
svr = SVR(kernel='poly', degree=5)  # default kernel is 'rbf'
# model training
svr.fit(X_train, y_train, sample_weight=y_train)
# The weights force the classifier to put more emphasis on C(regularization param)
y_pred = svr.predict(X_test)  # this is an unsupervised learning model, therefor we do not have labels/y data
pred_score = svr.score(X_test, y_test)

'''
The coefficient R^2 is defined as (1 - u/v), where u is the residual sum of squares ((y_true - y_pred) ** 2).sum() 
and v is the total sum of squares ((y_true - y_true.mean()) ** 2).sum(). 
The best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse). 
'''

from sklearn.metrics import max_error

print('Score: ', pred_score)
print('Max error: ', max_error(y_test, y_pred))
# The max_error function computes the maximum residual error ,
# a metric that captures the worst case error between the predicted value and the true value
# closer to 0, better the model

# Original model (no additional params)
# score: ~ -1.89
# max error: ~ 51100.89

# Verdict 1: ->>
# svr = SVR(kernel='poly', degree=5)
# svr.fit(X_train, y_train, sample_weight=y_train)
# score: 0.39560349256147065
# max error: 29936.53552973828

# Verdict 2: ->>
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)
# score: 0.49567286600622507
# max error: 23705.59629384602

# Final verdict ->> as already noted in the previous script,
# the data is either to small, or has redundant data

from sklearn.ensemble import RandomForestRegressor
rfreg = RandomForestRegressor()
rfreg.fit(X_train, y_train)
feature_imp = pd.Series(rfreg.feature_importances_)
print('Checking feature importance of the data:\n', feature_imp)
# 0    0.127357 ->> age  ->> 3
# 1    0.006275 ->> sex  ->> 0
# 2    0.209271 ->> bmi  ->> 4
# 3    0.018685 ->> children  ->> 2
# 4    0.625287 ->> smoker  ->> 5
# 5    0.013124 ->> region  ->> 1

# after removing 'sex' column
# score:  0.44648670301045545
# max error:  27091.45510139222

# after removing 'sex' and 'children' column
# score:  0.5568796739254404
# max error:  22142.71723131895

# after removing 'sex', 'children' and 'region' column
# score:  0.6380263354369701
# max error:  24685.833342752623
