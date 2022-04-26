'''
Code name: Agents_continuous.py
Author: Marco Antonio Esquivel Basaldua

Description:
Defines evader and pursuer Classes to create the agents in the continuous state-space. Methods in every class:
    - draw:         draws the agent in the screen
    - move_to:      defines next agent location

Input arguments:
    - none

Output:
    - none
'''

import numpy as np
import pygame

from obstacles import obstacles_per_env, border_per_env


def myFunc(e):
  return e[0]

# Given pos in discrete grid coordinates, returns location on the screen (pixels) coordinates
def env_to_screen_coordinates(pos, escale, res):
    return escale * np.flipud(pos) * res


class Evader:
    def __init__(self, pos, env_head, escale, screen):
        self.pos    = pos
        self.WIDTH  = env_head[0]
        self.LENGTH = env_head[1]
        self.res    = env_head[3]
        self.escale = escale
        self.screen = screen

    def draw(self):
        esc = self.escale
        l = esc // 2
        pos_pixels = env_to_screen_coordinates(self.pos, esc, self.res) - l//2 * np.ones(2, dtype = int)
        pygame.draw.rect(self.screen, (255,0,0), [pos_pixels[0],pos_pixels[1], l,l], 0)
        pygame.draw.circle(self.screen, (148,0,188), env_to_screen_coordinates(self.pos, esc, self.res).astype(int), esc//16)

        pos_round =  np.around(self.pos, 2)
        myfont = pygame.font.SysFont("Comic Sans MS", esc//3)
        label_pos = myfont.render('(' + str(pos_round[0]) + ',' + str(pos_round[1]) + ')', 1, (75,75,75))
        self.screen.blit(label_pos, env_to_screen_coordinates(self.pos + np.array([0.1, 0.1]), esc, self.res))

    def move_to(self, new_pos):
        self.pos = new_pos

class Pursuer:
    def __init__(self, pos, env_head, escale, screen):
        self.pos        = pos
        self.escale     = escale
        self.WIDTH      = env_head[0]
        self.LENGTH     = env_head[1]
        self.res        = env_head[3]
        self.screen     = screen

        self.border     = border_per_env(env_head[2])
        self.obstacles  = obstacles_per_env(env_head[2])
        self.obstacles.append(self.border)
    
    def draw(self):
        esc = self.escale
        h = esc//3
        l = esc//5

        pos_pixels = env_to_screen_coordinates(self.pos, esc, self.res)
        pygame.draw.polygon(self.screen, (0,0,255), [pos_pixels + np.array([0,-h]), pos_pixels + np.array([l,0]), pos_pixels + np.array([0,h]), pos_pixels + np.array([-l,0])], 0)
        pygame.draw.circle(self.screen, (148,0,188), env_to_screen_coordinates(self.pos, esc, self.res).astype(int), esc//16)

        pos_round =  np.around(self.pos, 2)
        myfont = pygame.font.SysFont("Comic Sans MS", esc//3)
        label_pos = myfont.render('(' + str(pos_round [0]) + ',' + str(pos_round[1]) + ')', 1, (75,75,75))
        self.screen.blit(label_pos, env_to_screen_coordinates(self.pos + np.array([0.1, 0.1]), esc, self.res))

    def move_to(self, new_pos):
        self.pos = new_pos