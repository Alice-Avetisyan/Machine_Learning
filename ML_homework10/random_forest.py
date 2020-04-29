import pandas as pd

bank_data = pd.read_csv('bank.csv')

print('----Bank columns----\n', bank_data.columns)
print('----Bank data shape----\n', bank_data.shape)
print('----First 20 rows----\n', bank_data.head(20))
print('----Last 10 rows----\n', bank_data.tail(10))

print('----Bank data description----\n', bank_data.describe())
print('----Bank data info----\n', bank_data.info)

print('----Checking missing data----\n', bank_data.isna().sum())
# For other checking purposes
# print('Bank columns\n', bank_data.columns)
# print('Bank data info\n', bank_data.info())
# job, marital, education, default, housing, loan, contact, month, poutcome, deposit

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
bank_data['job'] = le.fit_transform(bank_data['job'])
bank_data['marital'] = le.fit_transform(bank_data['marital'])
bank_data['education'] = le.fit_transform(bank_data['education'])
bank_data['default'] = le.fit_transform(bank_data['default'])
bank_data['housing'] = le.fit_transform(bank_data['housing'])
bank_data['loan'] = le.fit_transform(bank_data['loan'])
bank_data['contact'] = le.fit_transform(bank_data['contact'])
bank_data['month'] = le.fit_transform(bank_data['month'])
bank_data['poutcome'] = le.fit_transform(bank_data['poutcome'])
bank_data['deposit'] = le.fit_transform(bank_data['deposit'])

print(bank_data.head(20))
print(bank_data.tail(7))

import matplotlib.pyplot as plt

# bank_data.hist()
# plt.show()

from sklearn.preprocessing import StandardScaler

sts = StandardScaler()
bank_data[['balance', 'duration', 'pdays']] = sts.fit_transform(bank_data[['balance', 'duration', 'pdays']])

print(bank_data.head(20))
print(bank_data.tail(7))

# plt.matshow(bank_data.corr())
# plt.xticks(range(bank_data.shape[1]), bank_data.columns, fontsize=12, rotation=90)
# plt.yticks(range(bank_data.shape[1]), bank_data, fontsize=12)
# clb = plt.colorbar()
# clb.ax.tick_params(labelsize=12)
# plt.show()

import seaborn as sns
# sns.countplot(bank_data['deposit'])
# plt.show()

X = bank_data.drop('deposit', axis=1)
y = bank_data['deposit']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101, stratify=y)

from  sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=100, max_depth=10)
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)

from sklearn.metrics import accuracy_score, confusion_matrix

print('Accuracy score: ', accuracy_score(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.matshow(cm, cmap='Pastel1')

for x in range(0, 2):
    for y in range(0, 2):
        plt.text(x, y, cm[x, y])
plt.ylabel('expected label')
plt.xlabel('predicted label')
plt.show()

feature_imp = pd.Series(rfc.feature_importances_)
print('----Feature importance----\n', feature_imp)




