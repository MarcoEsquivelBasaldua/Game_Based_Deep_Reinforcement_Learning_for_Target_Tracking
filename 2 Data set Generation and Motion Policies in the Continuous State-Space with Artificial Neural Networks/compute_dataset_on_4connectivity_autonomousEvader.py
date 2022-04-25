'''
Code name: compute_dataset_on_4connectivity_autonomousEvader.py
Author: Marco Antonio Esquivel Basaldua

Description:
Computes train data set for pursuer and evader, considering an autonomous evader.

Input arguments:
    - Env<env>_res<res>.txt
    - E_Env<env>_res<res>.txt
    - W_Env<env>_res<res>.txt
    - visual_Env<env>_res<res>.txt

Output:
    - train_set_Env<env>_res<res>.csv
        Format:
        evader_y, evader_x, pursuer_y, pursuer_x, evader_action, pursuer_action, escape_length
        Note: If an evader_action value is set to -1, this is an invalid entry for the evader
'''

import numpy as np
import sys
from time import time
import copy

############# Load Map environment #################
env_file = open(sys.argv[1], 'r')
env_head = np.fromstring(env_file.readline(), dtype = np.int, sep = ' ')
env_file.close()

WIDTH = env_head[0]
LENGTH = env_head[1]
sel_env = str(env_head[-2])
res = str(env_head[-1])

########## Load Workspace and E table ############
E = np.loadtxt('E_Env' + sel_env + '_res' + res + '.txt')
W = np.loadtxt('W_Env' + sel_env + '_res' + res + '.txt').tolist()
visMatrix = np.loadtxt('visual_Env' + sel_env + '_res' + res + '.txt')
res = int(res)

############# Used Functions ###################
# Get value in table E for given evader and pursuer positions
def getValue_E(e,p):
    return E[W.index(e.tolist())][W.index(p.tolist())]

# Find the valid neighbors of a given point (x) in W
def neighborhood(x):
    neighbors = []
    x1 = x + np.array([1, 0])
    x2 = x + np.array([0, 1])
    x3 = x + np.array([-1, 0])
    x4 = x + np.array([0, -1])

    all_x = [x1,x2,x3,x4]

    for x_i in all_x:
        if 0 <= x_i[1] < LENGTH and 0 <= x_i[0] < WIDTH:
            if x_i.tolist() in W: neighbors.append(x_i)
    return neighbors

class Pursuer:
    def __init__(self, pos):
        self.pos = pos

    def N_p(self):
        return neighborhood(self.pos)
    
    def move_to(self, new_pos):
        action = np.array(new_pos - self.pos)
        if (action == np.array([0, 1])).all(): pursuer_action = 1.
        elif (action == np.array([-1, 0])).all(): pursuer_action = 2.
        elif (action == np.array([0, -1])).all(): pursuer_action = 3.
        elif (action == np.array([1, 0])).all(): pursuer_action = 4.
        else: pursuer_action = 0.

        return pursuer_action

class Evader:
    def __init__(self, pos):
        self.pos = pos

    def N_e(self):
        return neighborhood(self.pos)

    def move_to(self, new_pos):
        action = np.array(new_pos - self.pos)
        if (action == np.array([0, 1])).all(): evader_action = 0.
        elif (action == np.array([-1, 0])).all(): evader_action = 1.
        elif (action == np.array([0, -1])).all(): evader_action = 2.
        elif (action == np.array([1, 0])).all(): evader_action = 3.
        else: evader_action = -1.

        return evader_action

############# Game Loop #################
train_set = []
start_time = time()
for p in W:
    pursuer = Pursuer(np.array(p))
    for e in W:
        evader = Evader(np.array(e))
        state = (np.array([evader.pos, pursuer.pos, np.zeros(2)]).flatten() + 0.5) / res

        escape_lenght = getValue_E(evader.pos, pursuer.pos)

        # Split samples by cases
        if escape_lenght == np.inf:
            # Evader autonomous mouvement
            e_last = copy.copy(evader.pos)
            valE_in_Ne = []
            for e_prime in evader.N_e():
                if (e_prime == e_last).all():
                    valE_in_Ne.append(np.inf)
                else:
                    valE_in_Ne.append(np.nan_to_num(getValue_E(e_prime, pursuer.pos)))
            e_last = copy.copy(evader.pos)

            min_valE = np.argwhere(valE_in_Ne == np.min(valE_in_Ne))
            if min_valE.shape[0] * min_valE.shape[1] == 1:
                state[4] = evader.move_to(evader.N_e()[min_valE[0][0]])
            else:
                dist_to_pursuer = []
                for i in min_valE:
                    dist_to_pursuer.append(np.linalg.norm(evader.N_e()[i[0]] - pursuer.pos))
                state[4] = evader.move_to(evader.N_e()[min_valE[np.argmax(dist_to_pursuer)][0]])
            
            state[5] = 0.0

            train_set.append(state)

        elif escape_lenght != 0.0:
            # evader moves
            maxes = []
            for e_prime in evader.N_e():
                max_val = getValue_E(e_prime, pursuer.pos)
                for p_prime in pursuer.N_p():
                    if getValue_E(e_prime, p_prime) > max_val:
                        max_val = getValue_E(e_prime, p_prime)
                maxes.append(max_val)
            evader_action = evader.move_to(evader.N_e()[np.argmin(maxes)])

            # pursuer moves
            S = []
            max_val = getValue_E(evader.pos, pursuer.pos)
            for p_prime in pursuer.N_p():
                if getValue_E(evader.pos, p_prime) > max_val:
                    max_val = getValue_E(evader.pos, p_prime)
            if getValue_E(evader.pos, pursuer.pos) == max_val:
                S.append(pursuer.pos)
            for p_prime in pursuer.N_p():
                if getValue_E(evader.pos, p_prime) == max_val:
                    S.append(p_prime)
            
            dist_to_max = []
            for s in S:
                dist_to_max.append(np.linalg.norm(evader.pos - s))
            pursuer_action = pursuer.move_to(S[np.argmin(dist_to_max)])

            state[4] = evader_action
            state[5] = pursuer_action

            train_set.append(state)
        else:
            # pursuer moves
            S = []
            max_val = getValue_E(evader.pos, pursuer.pos)
            for p_prime in pursuer.N_p():
                if getValue_E(evader.pos, p_prime) > max_val:
                    max_val = getValue_E(evader.pos, p_prime)
            if getValue_E(evader.pos, pursuer.pos) == max_val:
                S.append(pursuer.pos)
            for p_prime in pursuer.N_p():
                if getValue_E(evader.pos, p_prime) == max_val:
                    S.append(p_prime)
            
            dist_to_max = []
            for s in S:
                dist_to_max.append(np.linalg.norm(evader.pos - s))
            new_pos = S[np.argmin(dist_to_max)]
            pursuer_action = pursuer.move_to(new_pos)

            if visMatrix[W.index(evader.pos.tolist())][W.index(new_pos.tolist())]:
            #if visual(new_pos, evader.pos):
                state[4] = -1.
                state[5] = pursuer_action

                train_set.append(state)

train_set_file = open('train_set_Env' + sys.argv[1][3] + '_res' + sys.argv[1][8] + '.csv', 'w')
train_set_file.write("evader_y, evader_x, pursuer_y, pursuer_x, evader_action, pursuer_action\n")
for sample in train_set:
    for i in range(6):
        train_set_file.write(str(sample[i]))
        if i < 5:
            train_set_file.write(',')
    train_set_file.write('\n')
train_set_file.close()


end_time = time()

print('Computation time:', end_time - start_time, 'seconds')