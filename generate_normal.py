import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
import statistics
from random import *
from matplotlib import style

style.use('ggplot')

#-------------generating samples of N(0,1) with rejection sampling---------------

#range of the uniform distribution (with r=10 the error is negligible) 
r = 10

#normal pdf
f = lambda x: ((math.e)**(-x/2))/(math.sqrt(2*math.pi))

#maximum point value of N(0,1)
max_prob = 0.39894228040

#three step method (making use of the standardized normal distribution symmetry)
c = max_prob*r

#we want n_accepted_samples_exp samples
n_accepted_samples_exp = 1000000
n_samples = int(n_accepted_samples_exp*c)

accepted_samples = []

#seed(111)

for i in range(n_samples):
    sample = random() #uniform(0,1)
    sample = r*sample
    accept_unif = random()
    if accept_unif <= f(sample)/max_prob: #accepts
        pos_or_neg = random()
        if pos_or_neg < 0.5:
            sample = - sample
        accepted_samples.append(sample)

plt.hist(x=accepted_samples, density=False)
plt.ylabel("Number of samples", fontsize=18)
plt.xlabel("Values of x", fontsize=18)
plt.title("Histogram of X~N(0,1), sampled until x = "+str(r))
plt.show()