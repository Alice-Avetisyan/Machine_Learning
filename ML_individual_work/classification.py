import pandas as pd

'''
Problem setting ->> We are checking if the person is a smoker, with the use of data
After checking the data for correlation, we can see that charges and smoker have the highest correlation
->> Smokes when stressed 
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

# sex ->> 0 | 1
# print('-----Checking data: Sex-----\n', insurance_data['sex'].head(10))

# smoker ->> 0 | 1
# print('-----Checking data: Smoker-----\n', insurance_data['smoker'].head(10))

# region ->> 0 | 1 | 2 | 3
# print('-----Checking data: Region-----\n', insurance_data['region'].head(10))

import matplotlib.pyplot as plt

# insurance_data.hist()
# plt.show()

plt.matshow(insurance_data.corr())
plt.xticks(range(insurance_data.shape[1]), insurance_data.columns, fontsize=12, rotation=90)
plt.yticks(range(insurance_data.shape[1]), insurance_data, fontsize=12)
clb = plt.colorbar()
clb.ax.tick_params(labelsize=12)
plt.show()

# import seaborn as sb
# sb.countplot(insurance_data['smoker'])
# plt.show()

# Preparing data ->> features and labels
# X = insurance_data.drop('smoker', axis=1) -->> og features
X = insurance_data.drop(['smoker', 'sex', 'children'], axis=1)
y = insurance_data['smoker']

# Splitting the data ->> train set and test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101, stratify=y)

# from sklearn.linear_model import LogisticRegression
# log_r = LogisticRegression()
# log_r.fit(X_train, y_train)
# checking for an unnatural error

# tensorflow cannot use dataframe format ??? current tensorflow==2.0.0; python==3.7.6
X_train = X_train.to_numpy()
X_test = X_test.to_numpy()
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

# Preparing the model
import tensorflow as tf

model = tf.keras.models.Sequential()  # choosing a Neural Network
# creating the Neural Network
model.add(tf.keras.layers.Dense(32, activation=tf.nn.relu, input_shape=(4,)))
model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu))  # creating the hidden layer
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))  # creating the hidden layer
model.add(tf.keras.layers.Dense(1, activation=tf.nn.sigmoid))  # creating the output layer

# compiling the model
adam = tf.keras.optimizers.Adam(lr=0.01)
# default lr(learning rate) 0.001 ->> defines how the gradients affects weight update
model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
# training the model
model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test), verbose=1)
# validation_data checks performance of the model (overfitting)
model.summary()  # param - weight
y_pred = model.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
# confusion_matrix and accuracy_score accept discrete inputs
cm = confusion_matrix(y_test, y_pred > 0.5)  # initial y_pred is continuous/ is not discrete
print('accuracy score: \n', accuracy_score(y_test, y_pred > 0.5))  # initial y_pred is continuous/ is not discrete
print('accuracy score: \n', confusion_matrix(y_test, y_pred > 0.5))  # initial y_pred is continuous/ is not discrete
print('classification report: \n', classification_report(y_test, y_pred > 0.5))  # initial y_pred is continuous/ is not discrete

# confusion matrix visualization
plt.matshow(cm, cmap='Pastel1')
for x in range(0, 2):
    for y in range(0, 2):
        plt.text(x, y, cm[x, y])
plt.ylabel('expected label')
plt.xlabel('predicted label')
plt.show()

# Verdict 1: ->>
# model.fit(X_train, y_train) accuracy ->> 0.76
# model.fit(X_train, y_train, epoch=5) accuracy ->> 0.91
# the training process ranges from 0.60 to 0.91

# Verdict 2: ->>
'''
adam = tf.keras.optimizers.Adam(lr=0.01) 
model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
'''
# The accuracy is: ~0.7947761194029851 - 0.9365671641791045

# Final verdict ->>
# 1. The model isn't the best;
# 2. The data is small for the NN;
# 3. The calculation inside of the training are always different












