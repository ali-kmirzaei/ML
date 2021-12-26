################################################################################################################
# Libraries :

import numpy as np
import matplotlib.pyplot as plt

################################################################################################################
# Functions :

def train(points, w, t):
    landa = 0.1
    for X in points:
        I = np.dot(X, w)
        y = 0
        if I>=0:
            y = 1

        f = list()
        for x in X:
            f.append(2*landa * (t-y) * x)

        w_new = list()
        for i in range(len(w)):
            w_new.append(w[i]+f[i])

        w = w_new
        # print('point =',X)
        # print('I =',I)
        # print('y =',y)
        # print('f =',f)
        # print('w_new =',w)
        # print('--------------------------------------------------')
    return w


def test(points, w, beta):
    cnt = 0
    for X in points:
        I = np.dot(X, w)
        if I*beta > 0:
            cnt += 1
    if cnt == len(points):
        return 1
    return 0

################################################################################################################
# Use :

reds = [
    [1, 4, 0, 16, 0],
    [1, 6, -1, 36, 1],
    [1, 5, 1, 25, 1],
    [1, 5, 2, 25, 4],
    [1, 7, 1, 49, 1],
]
greens = [
    [1, 2, 4, 4, 16],
    [1, 3, 3, 9, 9],
    [1, 2, -2, 4, 4],
    [1, 9, 2, 81, 4],
    [1, 8, -3, 64, 9],
    [1, 12, -4, 144, 16]
]
w = [20, -10, -2, 2, 2] # random vector
epoche_number = 10

# TRAIN
for i in range(epoche_number):
    w = train(reds, w, t=0)
    w = train(greens, w, t=1)
    print(w)
    print('--------------------------------------------------')


# TEST
red_status = test(reds, w, beta=-1)
green_status = test(greens, w, beta=1)
if red_status==1 and green_status==1:
    print("Result : TRUE")
else:
    print("Result : FALSE")
print('--------------------------------------------------')


# PLOT RESULT
reds1 = [4, 6, 5, 5, 7]
reds2 = [0, -1, 1, 2, 1]

greens1 = [2, 3, 2, 9, 8, 12]
greens2 = [4, 3, -2, 2, -3, -4]

plt.plot(reds1, reds2, 'ro')
plt.plot(greens1, greens2, 'go')
x1 = np.linspace(-12.0, 12.0, 100)
x2 = np.linspace(-12.0, 12.0, 100)
X1, X2 = np.meshgrid(x1,x2)
F = w[0] + w[1]*X1 + w[2]*X2 + w[3]*(X1**2) + w[4]*(X2**2)
plt.contour(X1, X2, F,[0])
plt.axis('equal')
plt.show()
