import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from numpy.core.fromnumeric import mean
from sklearn.model_selection import train_test_split

################################################################################################################
# Here we use MSE loss function for linear regression!
################################################################################################################
# PreProcessing :

# main_dataset = pd.read_csv("train.csv")
# X = np.array(main_dataset['GrLivArea']).reshape((-1, 1))
# y = np.array(main_dataset['SalePrice']).reshape((-1, 1))
# X_train, X_test, y_train, y_test = train_test_split(X, y)
# train = [X_train, y_train]
# test = [X_test, y_test]
# data = train

# ################################################################################################################
# Main funcs:

def loss_function(m, b, points):
    total_error = 0
    n = len(points[0])
    for i in range(n):
        X = points[0][i]
        y = points[1][i]
        total_error += (y - (m * X + b)) ** 2
    # total_error = int(total_error)/int(n)
    return total_error

def gradient_descent(m_now, b_now, points, L):
    m_gradient = 0
    b_gradient = 0
    n = len(points[0])
    for i in range(n):
        X = points[0][i]
        y = points[1][i]
        m_gradient += -(2/n) * X * (y- (m_now * X + b_now))
        b_gradient += -(2/n) * (y- (m_now * X + b_now))

    m = m_now - m_gradient * L
    b = b_now - b_gradient * L
    return m ,b


m = 0
b = 0
L = 0.0001
epoches = 1000
data = [
    [1, 2, 2, 3, 4, 5, 5, 5, 6, 7, 8, 9, 9, 9, 10, 11, 11, 12, 12, 12, 12, 12, 13],
    [10, 12, 12, 20, 30, 40, 40, 45, 50, 55, 57, 60, 61, 62, 65, 70, 71, 75, 75, 76, 78, 80, 90]
]
errors = list()

for i in range(epoches):
    errors.append(loss_function(m, b, data))
    m, b = gradient_descent(m, b, data, L)


print('m and b:',m, b)
print('first Error:', errors[0])
print('last  Error:', errors[len(errors)-1])
if min(errors) == errors[0]:
    print('So Its Bad!')
elif min(errors) == errors[len(errors)-1]:
    print('So Its Good!')

plt.scatter(data[0], data[1], color="black")
plt.plot(list(range(1, 15)), [m * X + b for X in range(1, 15)], color="red")
plt.show()

