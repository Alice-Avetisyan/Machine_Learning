import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

# movie price expensive mood will go

data_dict = {'price': ['expensive', 'cheap', 'cheap', 'expensive', 'expensive', 'cheap'],
             'mood': ['happy', 'sad', 'indifferent', 'sad', 'indifferent', 'happy'],
             'will_go': ['yes', 'no', 'yes', 'no', 'yes', 'yes']}

movie_df = pd.DataFrame(data_dict)
print('Printing the data: \n', movie_df)

le = preprocessing.LabelEncoder()
movie_df['price'] = le.fit_transform(movie_df['price'])
movie_df['mood'] = le.fit_transform(movie_df['mood'])
movie_df['will_go'] = le.fit_transform(movie_df['will_go'])

print('\nprice encoded: \n', movie_df['price'])
print('\nmood encoded: \n', movie_df['mood'])
print('\nwill go to the movie encoded: \n', movie_df['will_go'])

print(movie_df.head(3))

X = movie_df[['price', 'mood']]
y = movie_df['will_go']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

model = GaussianNB()
model.fit(X_train, y_train)
print('\nGo to the movie: ', model.predict([[0, 2]]))
