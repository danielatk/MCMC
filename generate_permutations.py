import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
import statistics
from random import *
from matplotlib import style
import time

style.use('ggplot')

#------------efficient algorithm for generation of random permutations-------------

r = 1000
k = [10, 100, 1000, 10000]
n = [10**4, 10**6, 10**8]

times = np.empty((len(k), len(n)))

for i in range(len(k)):
    for j in range(len(n)):
        times_local = []
        elems = np.arange(n[j])
        for l in range(r):
            start_time = time.time()
            for m in range(k[i]):
                s = randint(0,n[j]-(m+1))
                aux = elems[s]
                elems[s] = elems[n[j]-(m+1)]
                elems[n[j]-(m+1)] = aux
            times_local.append(time.time() - start_time)
        times[i,j] = np.mean(times_local)

colors = ['r', 'g', 'b']

for i in range(len(n)):
    plt.scatter(k, times[:,i], color=colors[i], label="n = "+str(n[i]))
plt.xlabel("k value", fontsize=18)
plt.ylabel("Mean execution time (seconds)", fontsize=18)
plt.title("Mean time for permutation generation, rounds = "+str(r) , fontsize=18)
plt.legend()
plt.show()