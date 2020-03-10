import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

curr_dollar_value = np.array([478.98, 479.05, 479.29, 478.60, 479.05, 478.95])
nextd_dollar_value = np.array([479.05, 479.29, 478.60, 479.05, 478.95, 478.63])
# print(curr_dollar_value.shape)

curr_dollar_value = curr_dollar_value.reshape(-1, 1)
print("Reshaped array: ", curr_dollar_value)
print("Mean: ", np.mean(curr_dollar_value))
# print(curr_dollar_value.shape)
# creating a model base on linear regression
model = LinearRegression()
# training a model through complete Linear regression (finding coefficient)
model.fit(curr_dollar_value, nextd_dollar_value)

print("a: ", model.coef_)
print("b: ", model.intercept_)

print("R squared metrics: ", model.score(curr_dollar_value, nextd_dollar_value))

predicted = model.predict([[478.63]])  # 478.63 target value
print("Tomorrow's dollar value would be: ", predicted)

plt.scatter(curr_dollar_value, nextd_dollar_value, color='red')
plt.xlabel('current dollar value')
plt.ylabel('next day dollar value')
plt.plot(curr_dollar_value, model.predict(curr_dollar_value))
plt.show()
