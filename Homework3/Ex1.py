import numpy as np

one_d = np.array([23, 44, 5, 1])
print("One dimensional array shape", one_d.shape)
print(one_d[1])
print("------------------------------------------------")
two_d = np.array([[2, 6, 3, 6], [6, 2, 4, 5]])
two_d_2 = np.array([[3, 5], [2, 2], [0, 3], [4, 2]])
print("Two dimensional array shape", two_d.shape)
print(two_d[1, 3])
print(np.dot(two_d, two_d_2))
print(two_d @ two_d_2)
print(two_d @ one_d)  # multiplies last axis (axis 1, columns) element wise, and then sums everything up

print("------------------------------------------------")
three_d = np.array([[[2, 4], [3, 4]], [[3, 4], [5, 5]], [[5, 5], [6, 5]]])
# three_d = np.array([[[2, 4]], [[3, 4]], [[5, 5]]])
print("Three dimensional array shape", three_d.shape)

print(three_d @ two_d)

print(three_d[0][0][0])  # 1. Nested 2D array, 2. Nested 2D array row, 3. Nested 2D array element

print(three_d[0] @ two_d)  # 2D array multiplication

print("------------------------------------------------")
four_d = np.array([[[[2, 4], [0, 0]], [[0, 1], [5, 1]]]])
print("Four dimensional array shape", four_d.shape)
print(np.dot(four_d, three_d))
