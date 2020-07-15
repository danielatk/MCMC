import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
import statistics
from random import *
from matplotlib import style

style.use('ggplot')

#number of states per graph
n_ring = [10, 50, 100, 300, 700, 1000]
n_grid = [9, 64, 100, 289, 729, 1024]
k_tree = [3, 6, 7, 8, 9, 10]
n_tree = []
for k in k_tree:
    n_tree.append(2**k -1)

n_ring_log = []
for n in n_ring:
    n_ring_log.append(np.log(n))
n_grid_log = []
for n in n_grid:
    n_grid_log.append(np.log(n))
n_tree_log = []
for n in n_tree:
    n_tree_log.append(np.log(n))

ns = [n_ring, n_grid, n_tree]
ns_log = [n_ring_log, n_grid_log, n_tree_log]

colors = ['r', 'g', 'b']

tol = 10**-4

#mixture times per graph
mt_ring = []
mt_grid = []
mt_tree = []

mt_ring_log = []
mt_grid_log = []
mt_tree_log = []

p = 0.5 #lazy random walk parameter

for l in range(len(n_ring)):

    #-------------ring graph mixture time--------------
    n = n_ring[l]
    P = np.zeros((n,n))
    stat_dist = [1/n]*n

    for j in range(1,n-1):
        P[j,j] = p
        P[j,j-1] = (1-p)/2
        P[j,j+1] = (1-p)/2

    P[0,0] = p
    P[0,1] = (1-p)/2
    P[0,n-1] = (1-p)/2

    P[n-1,n-1] = p
    P[n-1,n-2] = (1-p)/2
    P[n-1,0] = (1-p)/2

    pi0 = [0]*n
    pi0[0] = 1

    pi = pi0

    tv = 0.5*np.sum(np.abs(np.asarray(stat_dist) - np.asarray(pi)))
    counter = 0

    while(tv > tol):
        pi = np.dot(pi,P)
        tv = 0.5*np.sum(np.abs(stat_dist - pi))
        counter += 1
    
    mt_ring.append(counter)
    mt_ring_log.append(np.log(counter))

    #-------------full binary tree mixture time--------------
    k = k_tree[l]
    n = n_tree[l] #number of states (vertices)
    p0 = 3/(4*(2**(k-1)-1)) #parameter for the stationary distribution
    stat_dist = []
    stat_dist.append((2/3)*p0)
    for j in range(1,2**(k-1)-1):
        stat_dist.append(p0)
    for j in range(2**(k-1)-1, n):
        stat_dist.append(p0/3)

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

    pi0 = np.zeros((1,n))
    pi0[0,0] = 1 #initial distribution

    pi = pi0

    tv = 0.5*np.sum(np.abs(stat_dist - pi))
    counter = 0

    while(tv > tol):
        pi = np.dot(pi,P)
        tv = 0.5*np.sum(np.abs(stat_dist - pi))
        counter += 1
    
    mt_tree.append(counter)
    mt_tree_log.append(np.log(counter))

    #-------------2d grid mixture time--------------
    n = n_grid[l]
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

    pi0 = [0]*n
    pi0[0] = 1

    pi = pi0

    tv = 0.5*np.sum(np.abs(np.asarray(stat_dist) - np.asarray(pi)))
    counter = 0

    while(tv > tol):
        pi = np.dot(pi,P)
        tv = 0.5*np.sum(np.abs(stat_dist - pi))
        counter += 1
    
    mt_grid.append(counter)
    mt_grid_log.append(np.log(counter))

    print("round {} finished".format(l))

mts = []
mts.append(mt_ring)
mts.append(mt_grid)
mts.append(mt_tree)

mts_log = [mt_ring_log, mt_grid_log, mt_tree_log]

types_graphs = ['ring graph', '2d grid', 'full binary tree']

for i in range(len(colors)):
    plt.plot(ns_log[i], mts_log[i], color = colors[i], label = types_graphs[i])
    plt.xlabel("Number of states (log scale)", fontsize=18)
    plt.ylabel("Mixture time (log_scale)", fontsize=18)
plt.title("Mixture time per type of graph and number of states" , fontsize=18)
plt.legend()
plt.show()