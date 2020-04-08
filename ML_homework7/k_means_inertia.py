iris_data_url = 'https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv'

import pandas as pd

iris_df = pd.read_csv(iris_data_url)
#print('Iris data first 10 elements: ', iris_df.head(10))
#print('\n', iris_df.columns)

print(iris_df.dtypes)

# X = iris_df[['sepal length', 'petal length']]
print(set(iris_df['species']))
X = iris_df.drop('species', axis=1)


from sklearn.cluster import KMeans

inertias = []

ns = list(range(1, 10)) # n_clusters to check

for n in ns:
    model = KMeans(n_clusters=n)
    model.fit(X)
    inertias.append(model.inertia_)

import matplotlib.pyplot as plt

plt.plot(range(1, len(inertias)+1), inertias)
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Inertia Based On Cluster Numbers')
plt.show()

