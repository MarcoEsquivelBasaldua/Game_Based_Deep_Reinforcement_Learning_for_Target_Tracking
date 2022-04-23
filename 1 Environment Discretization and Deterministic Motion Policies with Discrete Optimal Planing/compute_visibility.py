'''
Code name: compute_visibility.py
Author: Marco Antonio Esquivel Basaldua

Description:
This code computes whether there exists or not visibility between every pair of cells in a discrete environment given as argument.

Input arguments:
    - Env<env>_res<res>.txt
        format:
            first row: w, l, env, res
                w: width of the environment (total of rows)
                l: length of the environment (total of columns)
                env: 1, 2, 3 or 4 (any of the environments used in simulation)
                res: resolution used in the environment
            next w rows: l columns of 0 and 1 where 0 is an obstacle and 1 is free space

Output:
    - visual_Env<env>_res<res>.txt
        format: visibility matrix (dimensions nxn) where 1 means visibility and 0 means no visibility according to arranged positions in the free space. 
        n is the total number of cells in the free space (Workspace)
'''

import numpy as np
import sys
from sympy import Polygon
from obstacles import obstacles_per_env

sel_env = int(sys.argv[1][3])
Obs = obstacles_per_env(sel_env)

def to_continuous(x):
    x = np.array(x, dtype=np.float)
    return (x + 0.5)/res

def isVisual(e,p):
    e = to_continuous(e)
    p = to_continuous(p)

    for obstacle in Obs:
        if len(obstacle.intersection(Polygon(e,p))) > 1:
            return 0

    return 1

map_file = open(sys.argv[1], 'r')
map_head = np.fromstring(map_file.readline(), dtype = np.int, sep = ' ')
map_file.close()

width = map_head[0]
length = map_head[1]
res = map_head[-1]

# Grab environment from input file
MAP = np.loadtxt(sys.argv[1], skiprows=1, dtype = int)
n = sum(sum(MAP))

# Cpmute workspace
i = 0
W = []

for row in MAP:
    j = 0
    for col in row:
        if col == 1:
            W.append([i,j])
        j += 1
    i += 1


# For every location in the workspace, compute visibility matrix
visMatrix = np.zeros((n,n), dtype=np.int)
for i in range(n):
    for j in range(n):
        if j == i:
            visMatrix[i][j] = 1
        elif j < i:
            visMatrix[i][j] = visMatrix[j][i]
        else:
            visMatrix[i][j] = isVisual(W[i],W[j])


# Save visibility matrix for Workspace
vis_file = open('visual_Env'+str(sel_env)+'_res'+str(res)+'.txt', 'w')
for row in visMatrix:
    for col in row:
        vis_file.write(str(col) + ' ')
    vis_file.write('\n')
vis_file.close()