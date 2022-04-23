'''
Code name: Agents_discrete.py
Author: Marco Antonio Esquivel Basaldua

Description:
Defines evader and pursuer Classes to create the agents. Methods in every class:
    - N_e or N_p:   valid agent's neighborhood
    - draw:         draws the agent in the screen
    - move_to:      defines next agent location

Input arguments:
    - none

Output:
    - none
'''

import numpy as np
import pygame


# Given pos in discrete grid coordinates, returns location on the screen (pixels) coordinates
def env_to_screen_coordinates(pos, escale):
    return escale * np.flipud(pos) + escale//2

# Find the valid neighbors of a given point (x) in W for a 4-connectivity neighborhood
def neighborhood(x, LENGTH, WIDTH, W):
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

# Evader Class
class Evader:
    def __init__(self, pos, escale, LENGTH, WIDTH, screen, W=None):
        self.pos    = pos
        self.escale = escale
        self.LENGTH = LENGTH
        self.WIDTH  = WIDTH
        self.W      = W
        self.screen = screen

    def N_e(self):
        return neighborhood(self.pos, self.LENGTH, self.WIDTH, self.W)

    def draw(self):
        esc = self.escale
        l = esc // 2
        pos_pixels = env_to_screen_coordinates(self.pos, esc) - l//2 * np.ones(2, dtype = int)
        pygame.draw.rect(self.screen, (255,0,0), [pos_pixels[0],pos_pixels[1], l,l], 0)

        pos_round =  np.around(self.pos, 2)
        myfont = pygame.font.SysFont("Comic Sans MS", esc//3)
        label_pos = myfont.render('(' + str(pos_round[0]) + ',' + str(pos_round[1]) + ')', 1, (75,75,75))
        self.screen.blit(label_pos, env_to_screen_coordinates(self.pos + np.array([0.1, 0.1]), esc))

    def move_to(self, new_pos):
        self.pos = new_pos

# Pursuer Class
class Pursuer:
    def __init__(self, pos, escale, LENGTH, WIDTH, screen, W=None, visMatrix=None):
        self.pos        = pos
        self.escale     = escale
        self.LENGTH     = LENGTH
        self.WIDTH      = WIDTH
        self.W          = W
        self.screen     = screen
        self.visMatrix  = visMatrix
        
    def N_p(self):
        return neighborhood(self.pos, self.LENGTH, self.WIDTH, self.W)
    
    def draw(self, vis_Region = True):
        esc = self.escale
        h = esc//3
        l = esc//5

        if vis_Region:
            for w in self.W:
                if self.visMatrix[self.W.index(self.pos.tolist())][self.W.index(w)]:
                    w_map = env_to_screen_coordinates(w, esc)
                    pygame.draw.rect(self.screen, (0,255,0), [w_map[0]-esc//2+2, w_map[1]-esc//2+2, esc-2, esc-2])

        pos_pixels = env_to_screen_coordinates(self.pos, esc)
        pygame.draw.polygon(self.screen, (0,0,255), [pos_pixels + np.array([0,-h]), pos_pixels + np.array([l,0]), pos_pixels + np.array([0,h]), pos_pixels + np.array([-l,0])], 0)

        pos_round =  np.around(self.pos, 2)
        myfont = pygame.font.SysFont("Comic Sans MS", esc//3)
        label_pos = myfont.render('(' + str(pos_round[0]) + ',' + str(pos_round[1]) + ')', 1, (75,75,75))
        self.screen.blit(label_pos, env_to_screen_coordinates(self.pos + np.array([0.1, 0.1]), esc))

    def move_to(self, new_pos):
        self.pos = new_pos