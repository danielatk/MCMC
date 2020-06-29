import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
import statistics
from random import *
from matplotlib import style

style.use('ggplot')

#-------------generating samples of X ~ Exp(lambda) with inverse transform---------------

param = [1/2, 1, 3/2]
colors = ['r', 'g', 'b']

n_samples = 1000000

for i in range(len(param)):
    F_inverse = lambda u: -math.log(1-u)/param[i]

    samples = []

    for j in range(n_samples):
        u = random()
        samples.append(F_inverse(u))

    plt.hist(x=samples, density=True, color=colors[i])
    plt.ylabel("Probability density", fontsize=18)
    plt.xlabel("Values of x", fontsize=18)
    plt.title("Histogram of X~Exp({})".format(param[i]))
    plt.show()

#-------------generating samples of X ~ Pareto(x0, alpha) with inverse transform---------------

x0 = 1
param = [6, 10, 14]
colors = ['r', 'g', 'b']

n_samples = 1000000

for i in range(len(param)):
    F_inverse = lambda u: x0*((1-u)**(-1/param[i]))

    samples = []

    for j in range(n_samples):
        u = random()
        samples.append(F_inverse(u))

    print(len(samples))

    plt.hist(x=samples, density=False, color=colors[i])
    plt.ylabel("Number of samples", fontsize=18)
    plt.xlabel("Values of x", fontsize=18)
    plt.title("Histogram of X~Pareto({},{})".format(x0, param[i]))
    plt.show()