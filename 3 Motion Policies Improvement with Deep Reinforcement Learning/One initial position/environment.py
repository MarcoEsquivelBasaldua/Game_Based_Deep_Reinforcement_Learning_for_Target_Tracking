'''
Code name: environment.py
Author: Marco Antonio Esquivel Basaldua

Description:
Creates the class Environment to define an environment to work with.

Input arguments:
    - none

Output:
    - none
'''

import numpy as np
from random import choice, choices, sample

from obstacles import obstacles_per_env, border_per_env
from evader_actions import evader_actions
from sympy import Polygon

class Environment:
    def __init__(self, env_head, device):
        self.state      = None
        res             = env_head[-1]
        sel_env         = env_head[-2]
        self.res        = res
        self.sel_env    = sel_env
        self.Width      = env_head[0]/res
        self.Length     = env_head[1]/res
        self.Obs        = obstacles_per_env(sel_env)
        self.border     = border_per_env(sel_env)
        self.device     = device

        self.evader_actions = evader_actions(sel_env, res)
        self.t_end          = 2*len(self.evader_actions)

    def set_state(self, state):
        self.state = state

    def reset_fixed_initial_position(self):
        self.state = np.array([1.5, 1.5, 0.5, 0.5])

        return self.state

    def reset(self):
        e = sample(self.evader_actions.keys(),1)[0]
        p = np.random.rand(2)
        p[0] *= self.Width
        p[1] *= self.Length
        state = np.array([e,p]).flatten()

        while not self.isVisual(state):
            p = np.random.rand(2)
            p[0] *= self.Width
            p[1] *= self.Length
            state = np.array([e,p]).flatten()

        self.state = state

        return state

    def train_configurations(self, samples_per_evader_position):
        initial_configurations = []

        for e in self.evader_actions.keys():
            for _ in range(samples_per_evader_position):
                p = np.random.rand(2)
                p[0] *= self.Width
                p[1] *= self.Length
                state = np.array([e,p]).flatten()

                while not self.isVisual(state):
                    p = np.random.rand(2)
                    p[0] *= self.Width
                    p[1] *= self.Length
                    state = np.array([e,p]).flatten()

                initial_configurations.append(state)
        return initial_configurations
        

    def select_evader_action(self, state):
        e = tuple(state[0:2])
        e = tuple(e)

        return self.evader_actions[e]


    def isVisual(self, s):
        x = s[0:2]
        y = s[2:4]
        for obstacle in self.Obs:
            if len(obstacle.intersection(Polygon(x,y))) > 1 or obstacle.encloses_point(x) or obstacle.encloses_point(y):
                return False
        return True

    def evader_step(self, action):
        new_state = None
        if action == 0:
            new_state = self.state + np.array([0., 1./self.res, 0., 0.])
        elif action == 1:
            new_state = self.state + np.array([-1./self.res, 0., 0., 0.])
        elif action == 2:
            new_state = self.state + np.array([0., -1./self.res, 0., 0.])
        elif action == 3:
            new_state = self.state + np.array([1./self.res, 0., 0., 0.])

        return new_state

    def pursuer_step(self, action):
        new_state = None
        if action == 1:
            new_state = self.state + np.array([0., 0., 0., 1./self.res])
        elif action == 2:
            new_state = self.state + np.array([0., 0., -1./self.res, 0.])
        elif action == 3:
            new_state = self.state + np.array([0., 0., 0., -1./self.res])
        elif action == 4:
            new_state = self.state + np.array([0., 0., 1./self.res, 0.])
        else:
            new_state = self.state

        return new_state

    def step_training(self, pursuer_dsit, t, epsilon=None):
        a_p = 0
        valid_actions = []
        probs = []
        obstacle_found = False
        for prob in pursuer_dsit:
            new_state = self.pursuer_step(a_p)
            p = new_state[2:4]

            if not self.border.encloses_point(p):
                a_p += 1
                continue

            for obstacle in self.Obs:
                if obstacle.encloses_point(p):
                    obstacle_found = True
                    continue
            
            if obstacle_found:
                obstacle_found = False
                a_p += 1
                continue
            else:
                valid_actions.append(a_p)
                probs.append(prob)

            a_p += 1

        if epsilon is not None:
        # Epsilon greedy exploration
            if np.random.uniform(0,1) > epsilon:
                a_p = choices(valid_actions, weights=probs, k=1)[0]
            else:
                a_p = choice(valid_actions)
        else:
            a_p = choices(valid_actions, weights=probs, k=1)[0]

        #print(a_p)

        a_e = self.select_evader_action(self.state)

        self.state = self.evader_step(a_e)
        self.state = self.pursuer_step(a_p)

        # # Players positions
        # e = self.state[0:2]
        # p = self.state[2:4]

        reward = 0
        done = False

        if self.isVisual(self.state):
            reward = 1
        else:
            done = True

        if t == self.t_end: done = True

        return self.state, reward, done, a_p


    def step_best_action(self, pursuer_dist, t):
        new_state = None

        a_e = self.select_evader_action(self.state)
        self.state = self.evader_step(a_e)

        ordered_actions = np.argsort(pursuer_dist, axis=-1)
        ordered_actions = np.flipud(ordered_actions)

        obstacle_found = False

        for action in ordered_actions:
            action = action.item()
            new_state = self.pursuer_step(action)
            p = new_state[2:4]

            if not self.border.encloses_point(p):
                continue

            for obstacle in self.Obs:
                if obstacle.encloses_point(p):
                    obstacle_found = True
                    break
            
            if obstacle_found:
                obstacle_found = False
                continue
            else:
                action_applied = action
                self.state = new_state
                break

        reward = 0
        done = False
        if self.isVisual(self.state): reward = 1
        else: done = True

        if t == self.t_end: done =  True

        return self.state, reward, done, action_applied