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

import random
from torch.distributions.categorical import Categorical
import numpy as np

from obstacles import obstacles_per_env, border_per_env
from sympy import Polygon

class Environment:
    def __init__(self, env_head, device):
        self.state      = None
        res             = env_head[3]
        sel_env         = env_head[2]
        self.res        = res
        self.sel_env    = sel_env
        self.W          = env_head[0]/res
        self.L          = env_head[1]/res
        self.Obs        = obstacles_per_env(sel_env)
        self.border     = border_per_env(sel_env)
        self.device     = device

    def set_state(self, state):
        self.state = state

    def reset(self):
        state_ = np.random.rand(4)
        state_[0] *= self.W
        state_[2] *= self.W
        state_[1] *= self.L
        state_[3] *= self.L

        while not self.isVisual(state_):
            state_ = np.random.rand(4)
            state_[0] *= self.W
            state_[2] *= self.W
            state_[1] *= self.L
            state_[3] *= self.L

        self.state =  state_
        return self.state

    def isVisual(self, s):
        x = s[0:2]#.numpy()
        y = s[2:4]#.numpy()
        for obstacle in self.Obs:
            if len(obstacle.intersection(Polygon(x,y))) > 0 or obstacle.encloses_point(x) or obstacle.encloses_point(y):
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


    def step_evader_prob_action(self, evader_dist):
        evader_dist = Categorical(evader_dist).probs

        action = 0
        valid_actions = []
        probs = []
        for prob in evader_dist:
            new_state = self.evader_step(action)

            for obstacle in self.Obs:
                if not obstacle.encloses_point(new_state[0:2]) and self.border.encloses_point(new_state[0:2]):
                    valid_actions.append(action)
                    probs.append(prob)

            action += 1

        action = random.choices(valid_actions, weights=probs, k=1)
        new_state = self.evader_step(action)

        
        reward = 0
        done = False
        if self.isVisual(self.state):
            reward = -1
        else:
            done = True

        return self.state, reward, done, action

    def step_evader_best_action(self, evader_dist):
        ordered_actions = np.argsort(evader_dist, axis=-1)
        ordered_actions = np.flipud(ordered_actions)

        obstacle_found = False
        action_applied = None
        for action in ordered_actions:
            action = action.item()
            new_state = self.evader_step(action)

            if not self.border.encloses_point(new_state[0:2]):
                continue

            for obstacle in self.Obs:
                if obstacle.encloses_point(new_state[0:2]):
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
        if self.isVisual(self.state):
            reward = -1
        else:
            done = True

        return self.state, reward, done, action_applied

    def step_pursuer_prob_action(self, pursuer_dist):
        pursuer_dist = Categorical(pursuer_dist).probs

        action = 0
        valid_actions = []
        probs = []
        for prob in pursuer_dist:
            new_state = self.pursuer_step(action)

            for obstacle in self.Obs:
                if not obstacle.encloses_point(new_state[2:4]) and self.border.encloses_point(new_state[2:4]):
                    valid_actions.append(action)
                    probs.append(prob)

            action += 1

        action = random.choices(valid_actions, weights=probs, k=1)
        new_state = self.pursuer_step(action)

        reward = 0
        done = False
        if self.isVisual(self.state):
            reward = 1
        else:
            done = True

        return self.state, reward, done, action



    def step_pursuer_best_action(self, pursuer_dist):
        ordered_actions = np.argsort(pursuer_dist, axis=-1)
        ordered_actions = np.flipud(ordered_actions)

        obstacle_found = False
        action_applied = None
        for action in ordered_actions:
            action = action.item()
            new_state = self.pursuer_step(action)

            if not self.border.encloses_point(new_state[2:4]):
                continue

            for obstacle in self.Obs:
                if obstacle.encloses_point(new_state[2:4]):
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
        if self.isVisual(self.state):
            reward = 1
        else:
            done = True

        return self.state, reward, done, action_applied