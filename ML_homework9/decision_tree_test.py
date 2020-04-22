import pandas as pd

heart_df = pd.read_csv('heart.csv')

# print('The shape of the data set\n', heart_df.shape)
# print('The data set columns\n', heart_df.columns)
# print('The first 10 elements of the data set\n', heart_df.head(10))
# print('The last 5 elements of the data set\n', heart_df.tail(5))
#
# # print(heart_df.isna().sum()) -> 0 found
#
# print(heart_df.describe())
# print(heart_df.info())
#
# print(heart_df['target'].value_counts())

X = heart_df.drop('target', axis=1)
y = heart_df['target']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101, stratify=y)

from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier(criterion='gini', max_depth=100, splitter='random')
dtc.fit(X_train, y_train)
y_pred = dtc.predict(X_test)

from sklearn.metrics import accuracy_score, confusion_matrix

print("Model accuracy: ", accuracy_score(y_test, y_pred))

import matplotlib.pyplot as plt
from sklearn import tree

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 4), dpi=1000)
fn = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
cn = ['present', 'absent']

tree.plot_tree(dtc,
               feature_names=fn,
               class_names=cn,
               filled=True)

fig.savefig('tree_test.png')