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

#-------------  ---------------

k = 4
characters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i','j', 'k', 'l', 'm', 'n', 'o', 'p',
                'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
c = len(characters)

n = 10**4 #number of samples
counter = 0
n_urls = 0
estimates = []

mapping_n_characters = [] #list with number of characters mapped by number of domains with that number of characters
for i in range(1, k+1):
    for j in range(c**i):
        mapping_n_characters.append(i)

size_d = len(mapping_n_characters) #number of domains with k or less characters

print("number of possible domains with k or less characters = " + str(size_d))

domains = []

for i in range(n):
    n_characters = mapping_n_characters[randint(0, size_d-1)] #size of domain to be tested
    exists = True
    #url = 'http://www.'
    for j in range(n_characters):
        if j == 0:
            url = characters[randint(0, c-1)]
        else:
            url = url + characters[randint(0, c-1)]
    url = url + '.ufrj.br'
    try:
        socket.gethostbyname(url.strip())
    except socket.gaierror:
        exists = False
    if exists == True:
        n_urls += 1
        domains.append(url)
    counter += 1
    estimate = (n_urls/counter)*size_d
    estimates.append(estimate)

print("Domains found:")
print(domains)

y_pos = np.arange(1, n+1)

plt.plot(y_pos, estimates, color='b')
plt.xlabel("Number of samples", fontsize=18)
plt.ylabel("Estimate of number of domains", fontsize=18)
plt.title("Estimated number of domains in UFRJ", fontsize=18)
plt.show()