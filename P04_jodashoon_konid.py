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
epoche_means = list()

# TRAIN
for i in range(epoche_number):
    w = train(reds, w, t=0)
    w = train(greens, w, t=1)
    epoche_means.append(np.mean(w))
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


# PLOT EPOCHES
x = np.arange(epoche_number)
y = epoche_means

plt.plot(x, y)
plt.xlabel(['epoche mean', 'epoches number'], fontdict=None, labelpad=None)
plt.show()
