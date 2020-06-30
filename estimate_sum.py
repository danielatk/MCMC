import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
import statistics
from random import *
from matplotlib import style
import socket
import time

style.use('ggplot')

#-------------estimating a target value with importance sampling---------------

N = 10**6

target = 0

g = lambda x: x*np.log(x)

for i in range(1, N+1):
    target += g(i)

print("target value = " + str(target))

#K = np.sum(np.arange(1,N+1))
K = N*(N+1)/2

#proponent distribution function (probability scales linearly with x)
h = lambda x: x/K

#calculating the second moment of g(Y)/h(Y)
second_moment = 0
for i in range(1, N+1):
    second_moment += (g(i)**2)/h(i)

print("Second moment: " + str(second_moment))

#number of samples (10**6 instead of 10**7 because of memory limitations)
n = 10**6

F_inverse = lambda u: (-1 + np.sqrt(1 + 8*K*u))/2

#sanity check
print("max value = " + str((-1 + np.sqrt(1 + 8*K))/2))

sum_samples = 0
counter = 0
errors = []

for i in range(1, n+1):
    u = random()
    Y = F_inverse(u)
    sum_samples += g(Y)/h(Y)
    counter += 1
    errors.append(np.log(np.abs(target - (sum_samples/counter))/target))

y_pos = []
for k in range(1, n+1):
    y_pos.append(math.log(k))

plt.plot(y_pos, errors, color='r')
plt.xlabel("Ammount of samples (log scale)", fontsize=18)
plt.ylabel("Relative error (log scale)", fontsize=18)
plt.title("Relative error to estimate " + r'$G_N$' + ", N = "+str(N), fontsize=18)
plt.show()