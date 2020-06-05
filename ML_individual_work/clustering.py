import pandas as pd

insurance_data = pd.read_csv('insurance.csv')

'''
Problem setting ->> Using data, to divide into groups 
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

X = insurance_data

from sklearn.cluster import KMeans

inertia = []  # I use inertia to implement elbow method

ns = list(range(1, 10))  # number of clusters

# iterating kmeans model to see the best/optimal number of clusters
for n in ns:
    kmeans = KMeans(n_clusters=n)
    #kmeans.fit(X)
    kmeans.fit_transform(X)  # computes clustering and transforms X to cluster-distance space
    inertia.append(kmeans.inertia_)

plt.plot(range(1, len(inertia)+1), inertia)
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Inertia Based On Cluster Numbers')
plt.show()

# This shows that after the 4-5 clusters there wont be a big effect inside of the model

model = KMeans(n_clusters=3, max_iter=250)
# max iter number of iterations for a single run
model.fit(X)
pred_model = model.predict(X, sample_weight=2)
print("Predicted model: ", pred_model)

print("Coherence of the model: ", model.score(X))
# opposite of the value of X on the K-means objective->> negative; less is better
# objective - to reduce the sum of squares of the distances of points from their respective cluster centroids
