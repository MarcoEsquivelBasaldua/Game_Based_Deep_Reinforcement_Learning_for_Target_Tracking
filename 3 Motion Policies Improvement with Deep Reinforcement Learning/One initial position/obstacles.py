'''
Code name: obstacles.py
Author: Marco Antonio Esquivel Basaldua

Description:
Defines every obstacle vertices in the environment

Input arguments:
    - sel: Env we are interested in (1, 2, 3, 4)

Output:
    - obstacles: list of Polygons representing every obstacle
    - border: border of the environmnet as a Polygon
'''

from sympy import Polygon

def obstacles_per_env(sel):
    obstacles = []
    
    if sel == 1:
        obstacles.append(Polygon((2,2), (2,7), (3,7), (3,2)))
    elif sel == 2:
        obstacles.append(Polygon((2,2), (2,9), (6,9), (6,2)))
        obstacles.append(Polygon((2,10), (2,19), (6,19), (6,10)))
    elif sel == 3:
        obstacles.append(Polygon((2,2), (2,3), (3,3), (3,2)))
        obstacles.append(Polygon((2,10), (2,12), (1,12), (1,15), (2,15), (2,13), (4,13), (4,11), (3,11), (3,10)))
        obstacles.append(Polygon((4,1), (4,3), (5,3), (5,1)))
        obstacles.append(Polygon((8,0), (8,1), (9,1), (9,0)))
        obstacles.append(Polygon((8,14), (8,15), (10,15), (10,14)))
        obstacles.append(Polygon((10,5), (10,9), (12,9), (12,5)))
    elif sel == 4:
        obstacles.append(Polygon((0,0), (0,1), (2,1), (2,0)))
        obstacles.append(Polygon((0,9), (0,11), (2,11), (2,10), (1,10), (1,9)))
        obstacles.append(Polygon((0,13), (0,15), (4,15), (4,14), (1,14), (1,13)))
        obstacles.append(Polygon((1,4), (1,5), (2,5), (2,7), (3,7), (3,6), (4,6), (4,4), (3,4), (3,3), (4,3), (4,2), (2,2), (2,4)))
        obstacles.append(Polygon((3,0), (3,1), (5,1), (5,0)))
        obstacles.append(Polygon((4,7), (4,10), (9,10), (9,15), (15,15), (15,12), (14,12), (14,13), (13,13), (13,14), (10,14), (10,12), (11,12), (11,11), (10,11), (10,10), (12,10), (12,9), (13,9), (13,8), (11,8), (11,9), (8,9), (8,7), (7,7), (7,9), (5,9), (5,7)))
        obstacles.append(Polygon((4,12), (4,13), (7,13), (7,12)))
        obstacles.append(Polygon((5,2), (5,6), (6,6), (6,4), (7,4), (7,3), (6,3), (6,2)))
        obstacles.append(Polygon((6,0), (6,1), (9,1), (9,0)))
        obstacles.append(Polygon((6,14), (6,15), (8,15), (8,14)))
        obstacles.append(Polygon((9,5), (9,6), (10,6), (10,7), (13,7), (13,6), (14,6), (14,5), (15,5), (15,4), (13,4), (13,5)))
        obstacles.append(Polygon((10,1), (10,4), (11,4), (11,3), (12,3), (12,2), (11,2), (11,1)))
        #obstacles.append(Polygon((10,14), (10,15), (15,15), (15,12), (14,12), (14,13), (13,13), (13,14)))
        obstacles.append(Polygon((14,2), (14,3), (15,3), (15,2)))

    return obstacles

def border_per_env(sel):
    if sel == 1:
        return Polygon((0,0), (0,10), (5,10), (5,0)) 
    elif sel == 2:
        return Polygon((0,0), (0,21), (8,21), (8,0)) 
    elif sel == 3 or sel == 4:
        return Polygon((0,0), (0,15), (15,15), (15,0))