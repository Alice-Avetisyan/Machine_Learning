from sklearn.datasets import make_classification
# Generate a random n-class classification problem
X, y = make_classification(100, 3, 2, 1, class_sep=0.5)  # 2 of 3 features are informative and 1 is redundant
# 100 -> number of samples/rows,
# 3 -> number of features/columns,
# 2 -> number of informative features,
# 1 -> number of redundant features (useless data)
# class_sep -> the complexity if the model

import matplotlib.pyplot as plt

# plt.hist(X[:, 1])  # all rows of the second column
# plt.show()
# plt.scatter(X[:, 0], X[:, 1])
# plt.show()

fig = plt.figure()
axis1 = fig.add_subplot(1, 2, 1)
axis1.hist(X[:, 1])
axis2 = fig.add_subplot(1, 2, 2)
axis2.scatter(X[:, 0], X[:, 1])
plt.show()

# plots the class distribution
for i in range(len(X)):
    if y[i] == 0:
        plt.scatter(X[i, 0], X[i, 1], marker='*', color='b')
    else:
        plt.scatter(X[i, 0], X[i, 1], marker='D', color='r')
plt.show()

from sklearn.svm import SVC

svc_model = SVC(kernel='rbf')

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=101)
svc_model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score

y_pred = svc_model.predict(X_test)
print("Model Accuracy: ", accuracy_score(y_test, y_pred))

# converting the data into DataFrame
import pandas as pd

custom_df = pd.DataFrame(X, columns=['X1', 'X2', 'X3'])
custom_df.insert(len(custom_df.columns), 'y', pd.DataFrame(y))
print(custom_df)
# turning the data into a csv file
custom_df.to_csv('custom_data.csv', index=False)
csv = pd.read_csv('custom_data.csv')
print(csv)