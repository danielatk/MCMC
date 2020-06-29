import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
import statistics
from random import *
from matplotlib import style

style.use('ggplot')

#-------------estimating square root of 2---------------

#what we want to estimate
target = math.sqrt(2)

n_samples = 1000000

#indicator function
def g(x, y):
    if y <= 2 - x**2:
        return 1
    else:
        return 0

#our estimate, a function of the expected value of the indicator function
estimate = lambda x: 3*x

counter = 0
errors = []
errors_log = []
for i in range(n_samples):
    x = uniform(0,2)
    y = uniform(0,2)
    counter += g(x,y)
    errors.append(np.abs(estimate(float(counter/(i+1))) - target)/target)
    errors_log.append(math.log(np.abs(estimate(float(counter/(i+1))) - target)/target))

y_pos = []
for k in range(1, n_samples+1):
    y_pos.append(math.log(k))

plt.plot(y_pos, errors_log, color='r')
plt.xlabel("Ammount of samples (log scale)", fontsize=18)
plt.ylabel("Relative error (log scale)", fontsize=18)
plt.title("Relative error to estimate " + r'$\sqrt{2}$', fontsize=18)
plt.show()

'''plt.plot(y_pos, errors, color='r')
plt.xlabel("Ammount of samples", fontsize=18)
plt.ylabel("Relative error", fontsize=18)
plt.title("Relative error to estimate " + r'$\sqrt{2}$', fontsize=18)
plt.show()'''