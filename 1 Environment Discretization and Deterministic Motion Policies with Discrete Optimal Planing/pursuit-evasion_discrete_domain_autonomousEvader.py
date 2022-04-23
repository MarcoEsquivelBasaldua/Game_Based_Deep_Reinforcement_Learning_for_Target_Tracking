'''
Code name: pursuit-evasion_discrete_domain_autonomousEvader.py
Author: Marco Antonio Esquivel Basaldua

Description:
Simulates the Tracking Problem by placing both players in the environment with an autonomous and antagonistic evader

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
    - E_Env<env>_res<res>
    - W_Env<env>_res<res>
    - visual_Env<env>_res<res>.txt
'''

import pygame
import numpy as np
import sys
from Agents_discrete import Evader, Pursuer

############# Load ENV environment #################
env_file = open(sys.argv[1], 'r')
env_head = np.fromstring(env_file.readline(), dtype = np.int32, sep = ' ')
env_file.close()

WIDTH = env_head[0]
LENGTH = env_head[1]
sel_env = str(env_head[-2])
res = str(env_head[-1])

ENV = np.loadtxt(sys.argv[1], skiprows=1, dtype = int)

if WIDTH > LENGTH: escale = 1000//WIDTH
else: escale = 1000//LENGTH

env_Length = LENGTH * escale
env_Width = WIDTH * escale

#### Load Workspace, E table and visMatrix #######
E = np.loadtxt('E_Env' + sel_env + '_res' + res + '.txt')
W = np.loadtxt('W_Env' + sel_env + '_res' + res + '.txt').tolist()
visMatrix = np.loadtxt('visual_Env' + sel_env + '_res' + res + '.txt')

############### Initialize pygame ################
pygame.init()
pygame.display.set_caption('pursuit-evation game on discrete domain')
screen = pygame.display.set_mode((env_Length,env_Width))
BLACK = (0,0,0)

############# Used Functions ###################
# Get value in table E for given evader and pursuer positions
def getValue_E(e,p):
    return E[W.index(e.tolist())][W.index(p.tolist())]

# Draw environment and grid
def draw_map(grid=True):
    screen.fill((255,255,255))

    if grid:
        m = int(env_Width/escale)
        n = int(env_Length/escale)

        for i in range(1,m):
            pygame.draw.line(screen, (150,150,150), (0,i*escale), (env_Length-1, i*escale), 3)
        for i in range(1,n):
            pygame.draw.line(screen, (150,150,150), (i*escale,0), (i*escale,env_Width-1), 3)

    for i in range(WIDTH):
        for j in range(LENGTH):
            if ENV[i][j] == 0: pygame.draw.rect(screen, BLACK ,[j*escale,i*escale, escale,escale])

############# Game Loop #################
running = True
place_pursuer = True
place_evader = False
start_game = prepare_game = False
valid_placeForPursuer = valid_placeForEvader = False
steps = 0
game_time = 1000
time = 10
t=0

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT: running = False
        if pygame.mouse.get_pressed()[0] and place_pursuer and valid_placeForPursuer:
            place_pursuer = False
            place_evader = True
        if pygame.mouse.get_pressed()[0] and place_evader and valid_placeForEvader:
            place_evader = False
            prepare_game = True

        if pygame.key.get_pressed()[13] == 1: # Start a new game
            start_game = prepare_game = place_evader = False
            place_pursuer = True
            steps = 0
            valid_placeForPursuer = valid_placeForEvader = False
            time = 10
            t = 0

    # Environment
    draw_map()
    if valid_placeForPursuer: pursuer.draw()
    if valid_placeForEvader: evader.draw()

    if place_pursuer:
        p_t = np.array(pygame.mouse.get_pos())
        if screen.get_at(p_t) != BLACK:
            p_t = np.flipud(p_t)//escale
            pursuer = Pursuer(p_t, escale, LENGTH, WIDTH, screen, W, visMatrix)
            valid_placeForPursuer = True
        else:
            valid_placeForPursuer = False
    
    elif place_evader:
        e_t = np.array(pygame.mouse.get_pos())
        if screen.get_at(e_t) != BLACK:
            e_t = np.flipud(e_t)//escale
            evader = Evader(e_t, escale, LENGTH, WIDTH, screen, W)
            valid_placeForEvader = True
        else:
            valid_placeForEvader = False

    elif prepare_game:
        pygame.display.update()
        pygame.time.delay(time)
        #pygame.image.save(screen, 'env'+str(sel_env)+'_t0.png')
        print('New game starting...')
        escape_lenght = getValue_E(evader.pos, pursuer.pos)

        if escape_lenght == np.inf: print('There is not escape')
        elif escape_lenght == 0: print('Evader is out of sight')
        else: print('Escape path lenght:', escape_lenght)

        prepare_game = False
        start_game = True

        e_last = evader.pos
    elif start_game:
        time = game_time

        if escape_lenght == np.inf:
            # Evader autonomous mouvement
            valE_in_Ne = []
            for e_prime in evader.N_e():
                if (e_prime == e_last).all():
                    valE_in_Ne.append(np.inf)
                else:
                    valE_in_Ne.append(np.nan_to_num(getValue_E(e_prime, pursuer.pos)))
            e_last = evader.pos

            min_valE = np.argwhere(valE_in_Ne == np.min(valE_in_Ne))
            if min_valE.shape[0] * min_valE.shape[1] == 1:
                evader.move_to(evader.N_e()[min_valE[0][0]])
            else:
                dist_to_pursuer = []
                for i in min_valE:
                    dist_to_pursuer.append(np.linalg.norm(evader.N_e()[i[0]] - pursuer.pos))
                evader.move_to(evader.N_e()[min_valE[np.argmax(dist_to_pursuer)][0]])
            
            # pursuer moves
            if getValue_E(evader.pos, pursuer.pos) != np.inf:
                for p_prime in pursuer.N_p():
                    if getValue_E(evader.pos, p_prime) == np.inf:
                        pursuer.move_to(p_prime)
                        break

            escape_lenght = getValue_E(evader.pos, pursuer.pos)

            #pygame.image.save(screen, 'env'+str(sel_env)+'_t'+str(t)+'.png')
            t += 1

        elif escape_lenght != 0.0:
            while 0.0 < escape_lenght < np.inf:
                # evader moves
                maxes = []
                for e_prime in evader.N_e():
                    max_val = getValue_E(e_prime, pursuer.pos)
                    for p_prime in pursuer.N_p():
                        max_val = max(max_val, getValue_E(e_prime, p_prime))
                        # if getValue_E(e_prime, p_prime) > max_val:
                        #     max_val = getValue_E(e_prime, p_prime)
                    maxes.append(max_val)
                evader.move_to(evader.N_e()[np.argmin(maxes)])

                # pursuer moves
                S = []
                max_val = getValue_E(evader.pos, pursuer.pos)
                for p_prime in pursuer.N_p():
                    max_val = max(max_val, getValue_E(evader.pos, p_prime))
                    # if getValue_E(evader.pos, p_prime) > max_val:
                    #     max_val = getValue_E(evader.pos, p_prime)
                if getValue_E(evader.pos, pursuer.pos) == max_val:
                    S.append(pursuer.pos)
                for p_prime in pursuer.N_p():
                    if getValue_E(evader.pos, p_prime) == max_val:
                        S.append(p_prime)
                
                dist_to_max = []
                for s in S:
                    dist_to_max.append(np.linalg.norm(evader.pos - s))
                pursuer.move_to(S[np.argmin(dist_to_max)])

                draw_map()
                pursuer.draw()
                evader.draw()
                pygame.display.update()
                pygame.time.delay(time)

                t += 1
                #pygame.image.save(screen, 'env'+str(sel_env)+'_t'+str(t)+'.png')

                escape_lenght = getValue_E(evader.pos, pursuer.pos)
        
        elif escape_lenght == 0.0:
            print('evasion done')
            start_game = False

    pygame.display.update()
    pygame.time.delay(time)
