'''
Code name: simulation_evader_fixed_trajectory_fixed_start_location.py
Author: Marco Antonio Esquivel Basaldua

Description:
Simulates the Tracking Problem using the pursuer trained model with Rl. The evader follows its predifined path.

Input arguments:
    - Env<env>_res<res>.txt
    - pursuer trained model

Output:
    - none
'''

import torch
import sys
import numpy as np
import pygame

from sympy import Polygon
from obstacles import obstacles_per_env
#from models import pursuer_NN
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
pursuer_model = torch.load(sys.argv[2])
#pursuer_model = torch.load('pursuer_RLwithE_Env' + str(sel_env) + '_res' + str(res) + '.pt')
pursuer_model.eval().to(device = device)

########### Load environment #######################
env = Environment(env_head, 'cpu')

############### Initialize pygame ################
pygame.init()
pygame.display.set_caption('Target tracking - fixed initial locations - pursuer trained with PG')
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

def select_evader_action(state):
        e = tuple(state[0:2])
        e = tuple(e)

        return env.evader_actions[e]
##########################################


running = True
game =  0
time = 1000
state = None
pursuer_total_reward = 0
t = 0

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT: running = False
        if pygame.key.get_pressed()[13] == 1:
            game = 0
            pursuer_total_reward = 1
            t = 0
            evader_done = False
            pursuer_done = False
    
    if game == 0:
        state = env.reset_fixed_initial_position()
        e_t = state[0:2]
        p_t = state[2:4]
        evader = Evader(e_t, env_head, escale, screen)
        pursuer = Pursuer(p_t, env_head, escale, screen)

        game = 1

        pygame.time.delay(3000)

    elif game == 1:
        a_e = select_evader_action(state)
        state = env.evader_step(a_e)

        p_dist_before = pursuer_model(state)
        p_dist = pursuer_model(state).detach().cpu().numpy()
        
        state, reward, pursuer_done, _ = env.step_best_action(p_dist, t)
        t += 1
        pursuer_total_reward += reward

        if pursuer_done:
            print('Pursuer lost visibility')
            game = 2
            continue        

    elif game == 2:
        print('End of the game')
        print('pursuer total reward', pursuer_total_reward)
    

    draw_map(grid=False)
    
    e_t = state[0:2]
    p_t = state[2:4]

    evader.move_to(e_t)
    pursuer.move_to(p_t)

    evader.draw()
    pursuer.draw()
    pygame.draw.line(screen, (0, 255, 0), env_to_screen_coordinates(evader.pos), env_to_screen_coordinates(pursuer.pos), 3)
    

    pygame.display.update()
    pygame.time.delay(time)