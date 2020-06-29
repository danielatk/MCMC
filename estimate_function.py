import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
import statistics
from random import *
from matplotlib import style

style.use('ggplot')

#-------------estimating a target function---------------

a = 0
b = [1, 2, 4]
alpha = [1, 2, 3]
colors = ['r', 'g', 'b']

for j in range(len(alpha)):
    for i in range(len(b)):

        #what we want to estimate
        target = (b[i]**(alpha[j]+1) - a**(alpha[j]+1))/(alpha[j]+1)

        n_samples = 1000000

        #indicator function
        def g(x, y):
            if y <= x**alpha[j]:
                return 1
            else:
                return 0

        #our estimate, a function of the expected value of the indicator function
        estimate = lambda x: (b[i]-a)*(b[i]**alpha[j])*x

        counter = 0
        errors = []
        errors_log = []
        for k in range(n_samples):
            x = uniform(a,b[i])
            y = uniform(0,b[i]**alpha[j])
            counter += g(x,y)
            errors.append(np.abs(estimate(float(counter/(k+1))) - target)/target)
            if np.abs(estimate(float(counter/(k+1))) - target) > 0:
                errors_log.append(math.log(np.abs(estimate(float(counter/(k+1))) - target)/target))
            else:
                errors_log.append(-22) #log(0) is undefined so the log of a small value is chosen instead

        y_pos = []
        for k in range(1, n_samples+1):
            y_pos.append(math.log(k))

        plt.plot(y_pos, errors_log, color=colors[i], label="b = "+str(b[i]))
    plt.xlabel("Ammount of samples (log scale)", fontsize=18)
    plt.ylabel("Relative error (log scale)", fontsize=18)
    plt.title("Relative error to estimate g with \u03B1 = " + str(alpha[j]) + ", a = " + str(a) , fontsize=18)
    plt.legend()
    plt.show()