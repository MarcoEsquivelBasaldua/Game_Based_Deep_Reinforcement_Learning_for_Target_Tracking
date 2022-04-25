'''
Code name: policies_approximation.py
Author: Marco Antonio Esquivel Basaldua

Description:
Simulates the Tracking Problem using the trained model for the evader and the pursuer

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
    - none

Required files:
    - evader_NN_Env<env>_res<res>.pt
    - pursuer_NN_Env<env>_res<res>.pt
'''

import torch
import sys
import numpy as np
import matplotlib.pyplot as plt
import pygame

from sympy import Polygon
from obstacles import obstacles_per_env
from models import evader_NN, pursuer_NN
from environment import Environment
from Agents_continuous import Evader, Pursuer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

############# Load Map environment #################
env_file = open(sys.argv[1], 'r')
env_head = np.fromstring(env_file.readline(), dtype=np.int32, sep=' ')
env_file.close()

WIDTH = env_head[0]
LENGTH = env_head[1]
sel_env = env_head[2]
res = env_head[3]

ENV = np.loadtxt(sys.argv[1], skiprows=1, dtype=np.int)

if WIDTH > LENGTH: escale = 1000//WIDTH
else: escale = 1000//LENGTH

env_Length = LENGTH * escale
env_Width = WIDTH * escale

Obstacles = obstacles_per_env(sel_env)
##################################################

########### Load  pre-trained models ######################
evader_model = torch.load('evader_NN_Env' + str(sel_env) + '_res' + str(res) + '.pt')
evader_model.eval().to(device = device)

pursuer_model = torch.load('pursuer_NN_Env' + str(sel_env) + '_res' + str(res) + '.pt')
pursuer_model.eval().to(device = device)

########### Load environment #######################
env = Environment(env_head, 'cpu')

############### Initialize pygame ################
pygame.init()
pygame.display.set_caption('Target tracking - policies approximation')
screen = pygame.display.set_mode((env_Length,env_Width))

WHITE = (255,255,255)
BLACK = (0,0,0)
GRAY = (150,150,150)
GRAY2 = (75,75,75)
RED = (255,0,0)
BLUE = (0,0,255)
GREEN = (190,230,90)
PURPLE = (148,0,188)
#################################################


############## Functions ####################
# Draw environment and grid
def draw_map(grid=True):
    screen.fill(WHITE)

    if grid:
        m = int(env_Width/escale)
        n = int(env_Length/escale)

        for i in range(1,m):
            pygame.draw.line(screen, (150,150,150), (0,i*escale), (env_Length-1, i*escale), 3)
        for i in range(1,n):
            pygame.draw.line(screen, (150,150,150), (i*escale,0), (i*escale,env_Width-1), 3)

    for i in range(WIDTH):
        for j in range(LENGTH):
            if ENV[i][j] == 0: pygame.draw.rect(screen, BLACK,[j*escale,i*escale, escale,escale])

def env_to_screen_coordinates(pos):
    return escale * np.flipud(pos) * res

def screen_to_env_coordinates(pos):
    return np.flipud(pos)/(escale * res)

def isVisual(x,y):
    for obstacle in Obstacles:
        if len(obstacle.intersection(Polygon(x,y))) > 1:
            return False
    return True

##########################################


running = True
game =  0
time = 1000
state = None
evader_total_reward = 0
pursuer_total_reward = 0
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT: running = False
        if pygame.key.get_pressed()[13] == 1:
            game = 0
            evader_total_reward = 0
            pursuer_total_reward = 0
            evader_done = False
            pursuer_done = False
    
    if game == 0:
        state = env.reset()
        e_t = state[0:2]
        p_t = state[2:4]
        evader = Evader(e_t, env_head, escale, screen)
        pursuer = Pursuer(p_t, env_head, escale, screen)

        game = 1

    elif game == 1:
        evader_dist = evader_model(state)
        state, evader_reward, evader_done, _ = env.step_evader_best_action(evader_dist)
        evader_total_reward += evader_reward

        if evader_done:
            print('evader has steped out the workspace')
            game = 2
            continue


        pursuer_dist = pursuer_model(state)
        state, pursuer_reward, pursuer_done, _ = env.step_pursuer_best_action(pursuer_dist)
        pursuer_total_reward += pursuer_reward

        if pursuer_done:
            print('Pursuer lost visibility')
            game = 2
            continue        

    elif game == 2:
        print('End of the game')
        print('evader total reward', evader_total_reward)
        print('pursuer total reward', pursuer_total_reward)
    

    draw_map()
    
    e_t = state[0:2]
    p_t = state[2:4]

    evader.move_to(e_t)
    pursuer.move_to(p_t)

    evader.draw()
    pursuer.draw()
    pygame.draw.line(screen, (0, 255, 0), env_to_screen_coordinates(evader.pos), env_to_screen_coordinates(pursuer.pos), 3)
    

    pygame.display.update()
    pygame.time.delay(time)