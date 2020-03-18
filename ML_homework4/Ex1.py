import pandas as pd
import numpy as np

dict = {'First score': [100, 90, np.nan, 95],
        'Second score': [30, 45, 56, np.nan],
        'Third score': [np.nan, 40, 80, 98]}

index = ['row1', 'row2', 'row3', 'row4']

dict_df = pd.DataFrame(dict, index=index)
print(dict_df.fillna(0))
print("Calculating mean: ", dict_df.mean())
print("Calculating sum: ", dict_df.sum())

#  print(dir(pd.DataFrame))

a_df = pd.DataFrame([[2, 1, -5, 4], [4, -6, 2, 5]])
s_df = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
other_df = pd.DataFrame([[1, 2], [2, 2], [3, 2], [3, 2]])

print(a_df.dot(s_df))
print(a_df.dot(other_df))
print(a_df.add(2))
print(other_df.div(10))

