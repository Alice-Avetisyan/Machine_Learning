# from sklearn import datasets
#
# iris_data = datasets.load_iris()
#
# print('Iris feature names: ', iris_data.feature_names)
# print('Iris data labels: ', iris_data.target_names)
# print('Iris data shape: ', iris_data.data.shape)
# print('Iris data first 10 elements: ', iris_data.data[0:11])

iris_data_url = 'https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv'

import pandas as pd

iris_df = pd.read_csv(iris_data_url)
print('Iris data first 10 elements: ', iris_df.head(10))
print('\n', iris_df.columns)

print(iris_df.dtypes)

# X = iris_df[['sepal length', 'petal length']]
X = iris_df.drop('species', axis=1)

from sklearn.cluster import KMeans

model = KMeans(n_clusters=3)
model.fit(X)
print(model.predict(X))
print(model.score(X))
