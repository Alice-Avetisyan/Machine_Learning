import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


income_data_dict = {'Previous Day_Inc': [100, 450, 340, 234, 180, 190],  # x > 200 = true; x < 200 = false
                    'Month': [1, 2, 3, 4, 5, 6], 'Revenue': [0, 1, 1, 1, 0, 0]}

income_data_df = pd.DataFrame(income_data_dict)
print(income_data_df)

X = income_data_df[['Previous Day_Inc', 'Month']]
y = income_data_df['Revenue']
print("--------------Printing X and y--------------")
print(X)
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
print("--------------Printing X test and train--------------")
print(X_test)
print(X_train)
print("--------------Printing y test and train--------------")
print(y_train)
print(y_train)

logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)

prediction = logmodel.predict(X_test)
print("--------------Printing prediction--------------")
print(prediction)
print("--------------Printing X and y test--------------")
print(X_test)
print(y_test)

classf_report = classification_report(y_test, prediction)
print("--------------Printing classification_report--------------")
print(classf_report)

conf_matrix = confusion_matrix(y_test, prediction)
print("--------------Printing confusion_matrix--------------")
print(conf_matrix)

