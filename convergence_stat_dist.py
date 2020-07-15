import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
import statistics
from random import *
from matplotlib import style

style.use('ggplot')

#-------------convergence to stationary distribution for ring graph-------------

n = 100 #number of states (vertices)
p = 0.5 #lazy random walk parameter
P = np.zeros((n,n))
stat_dist = [1/n]*n

type_graph = "ring graph"

for i in range(1,n-1):
    P[i,i] = p
    P[i,i-1] = (1-p)/2
    P[i,i+1] = (1-p)/2

P[0,0] = p
P[0,1] = (1-p)/2
P[0,n-1] = (1-p)/2

P[n-1,n-1] = p
P[n-1,n-2] = (1-p)/2
P[n-1,0] = (1-p)/2

print("transition matrix")
print(P)

pi0 = [0]*n
pi0[0] = 1

rounds = 1000
total_variation = []
total_variation_log = []
pi = pi0

print("initial dist")
print(pi)
print("stationary dist")
print(stat_dist)

for i in range(rounds):
    pi = np.dot(pi,P)
    tv = 0.5*np.sum(np.abs(stat_dist - pi))
    total_variation.append(tv)
    total_variation_log.append(math.log(tv))

y_pos = []
for k in range(1, rounds+1):
    y_pos.append(math.log(k))

plt.plot(y_pos, total_variation_log)
plt.xlabel("t (log scale)", fontsize=18)
plt.ylabel("Total variation distance (log scale)", fontsize=18)
plt.title("Total variation distance between \u03C0 and \u03C0(t), "+type_graph , fontsize=18)
plt.show()

#-------------convergence to stationary distribution for full binary tree-------------

k = 7 #levels in the full binary tree
n = 2**k - 1 #number of states (vertices)
p0 = 3/(4*(2**(k-1)-1)) #parameter for the stationary distribution
stat_dist = []
stat_dist.append((2/3)*p0)
for i in range(1,2**(k-1)-1):
    stat_dist.append(p0)
for i in range(2**(k-1)-1, n):
    stat_dist.append(p0/3)

p = 0.5 #lazy random walk parameter
P = np.zeros((n,n))

P[0,0] = p
P[0,1] = (1-p)/2
P[0,2] = (1-p)/2

for i in range(2,k):
    for j in range(2**(i-1) - 1,2**i - 1):
        P[j,j] = p
        if j%2 == 1:
            P[j,(2**(i-2))-1 + (int((j+1)/2) - 2**(i-2))] = (1-p)/3
        else:
            P[j,2**(i-2)-1 + (int(j/2) - 2**(i-2))] = (1-p)/3
        P[j,(2**i)-1 + 2*(j+1 - 2**(i-1))] = (1-p)/3
        P[j,(2**i) + 2*(j+1 - 2**(i-1))] = (1-p)/3

for j in range(2**(k-1) - 1,2**k - 1):
        P[j,j] = p
        if j%2 == 1:
            P[j,2**(k-2)-1 + (int((j+1)/2) - 2**(k-2))] = 1-p
        else:
            P[j,2**(k-2)-1 + (int(j/2) - 2**(k-2))] = 1-p

print("transition matrix")
print(P)

pi0 = np.zeros((1,n))
pi0[0,0] = 1 #initial distribution

rounds = 1000
total_variation = []
total_variation_log = []
pi = pi0

print("initial dist")
print(pi)
print("stationary dist")
print(stat_dist)

for i in range(rounds):
    pi = np.dot(pi,P)
    tv = 0.5*np.sum(np.abs(stat_dist - pi))
    total_variation.append(tv)
    total_variation_log.append(math.log(tv))

y_pos = []
for k in range(1, rounds+1):
    y_pos.append(math.log(k))

type_graph = "full binary tree"

plt.plot(y_pos, total_variation_log)
plt.xlabel("t (log scale)", fontsize=18)
plt.ylabel("Total variation distance (log scale)", fontsize=18)
plt.title("Total variation distance between \u03C0 and \u03C0(t), "+type_graph , fontsize=18)
plt.show()

#-------------convergence to stationary distribution for 2D grid-------------

n = 100 #number of states (vertices)
p = 0.5 #lazy random walk parameter
P = np.zeros((n,n))
stat_dist = []
p0 = 1/((np.sqrt(n)-2)**2 + 3*(np.sqrt(n)-2) + 2) #parameter for the stationary distribution
for i in range(int(np.sqrt(n))):
    for j in range(int(np.sqrt(n))):
        if i == 0 or i == np.sqrt(n)-1:
            if j > 0 and j < np.sqrt(n) - 1:
                stat_dist.append((3/4)*p0)
            else:
                stat_dist.append((1/2)*p0)
        else:
            if j > 0 and j < np.sqrt(n) - 1:
                stat_dist.append(p0)
            else:
                stat_dist.append((3/4)*p0)

type_graph = "2D grid graph"

for i in range(int(np.sqrt(n))):
    for j in range(int(np.sqrt(n))):
        P[int(i*np.sqrt(n)) + j, int(i*np.sqrt(n)) + j] = p
        if i == 0:
            if j == 0:
                P[0,1] = (1-p)/2
                P[0,int(np.sqrt(n))] = (1-p)/2
            elif j == np.sqrt(n)-1:
                P[int(np.sqrt(n)-1),int(np.sqrt(n)-2)] = (1-p)/2
                P[int(np.sqrt(n)-1),int(2*np.sqrt(n)-1)] = (1-p)/2
            else:
                P[j,j-1] = (1-p)/3
                P[j,j+1] = (1-p)/3
                P[j,int(np.sqrt(n) + j)] = (1-p)/3
        elif i == np.sqrt(n)-1:
            if j == 0:
                P[int((np.sqrt(n)-1)*np.sqrt(n)),int((np.sqrt(n)-1)*np.sqrt(n))+1] = (1-p)/2
                P[int((np.sqrt(n)-1)*np.sqrt(n)),int((np.sqrt(n)-2)*np.sqrt(n))] = (1-p)/2
            elif j == np.sqrt(n)-1:
                P[n-1,n-2] = (1-p)/2
                P[n-1,int((np.sqrt(n)-1)*np.sqrt(n))-1] = (1-p)/2
            else:
                P[int((np.sqrt(n)-1)*np.sqrt(n)) + j, int((np.sqrt(n)-1)*np.sqrt(n))+j-1] = (1-p)/3
                P[int((np.sqrt(n)-1)*np.sqrt(n))+j,int((np.sqrt(n)-1)*np.sqrt(n))+j+1] = (1-p)/3
                P[int((np.sqrt(n)-1)*np.sqrt(n))+j,int((np.sqrt(n)-2)*np.sqrt(n)) + j] = (1-p)/3
        else:
            if j == 0:
                P[int(i*np.sqrt(n)),int(i*np.sqrt(n))+1] = (1-p)/3
                P[int(i*np.sqrt(n)),int((i-1)*np.sqrt(n))] = (1-p)/3
                P[int(i*np.sqrt(n)),int((i+1)*np.sqrt(n))] = (1-p)/3
            elif j == np.sqrt(n)-1:
                P[int((i+1)*np.sqrt(n))-1,int((i+1)*np.sqrt(n))-2] = (1-p)/3
                P[int((i+1)*np.sqrt(n))-1,int((i)*np.sqrt(n))-1] = (1-p)/3
                P[int((i+1)*np.sqrt(n))-1,int((i+2)*np.sqrt(n))-1] = (1-p)/3
            else:
                P[int(i*np.sqrt(n))+j,int(i*np.sqrt(n))+j-1] = (1-p)/4
                P[int(i*np.sqrt(n))+j,int(i*np.sqrt(n))+j+1] = (1-p)/4
                P[int(i*np.sqrt(n))+j,int((i+1)*np.sqrt(n))+j] = (1-p)/4
                P[int(i*np.sqrt(n))+j,int((i-1)*np.sqrt(n))+j] = (1-p)/4

print("transition matrix")
print(P)

pi0 = [0]*n
pi0[0] = 1

rounds = 1000
total_variation = []
total_variation_log = []
pi = pi0

print("initial dist")
print(pi)
print("stationary dist")
print(stat_dist)

for i in range(rounds):
    pi = np.dot(pi,P)
    tv = 0.5*np.sum(np.abs(stat_dist - pi))
    total_variation.append(tv)
    total_variation_log.append(math.log(tv))

y_pos = []
for k in range(1, rounds+1):
    y_pos.append(math.log(k))

plt.plot(y_pos, total_variation_log)
plt.xlabel("t (log scale)", fontsize=18)
plt.ylabel("Total variation distance (log scale)", fontsize=18)
plt.title("Total variation distance between \u03C0 and \u03C0(t), "+type_graph , fontsize=18)
plt.show()