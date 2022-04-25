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
from sympy.geometry import Polygon, Ray

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
        myfont = pygame.font.SysFont("Comic Sans MS", 2*esc//3)
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
    
    def draw(self, vis_Region = False):
        esc = self.escale
        h = esc//3
        l = esc//5

        p = self.pos

        if vis_Region:
            vis_polygon = []
            for obs in self.obstacles:
                for vertice in obs.args:
                    intersections = []
                    segment = Polygon(p, vertice)

                    for obs2 in self.obstacles:
                        inter = obs2.intersection(segment)
                        if len(inter) > 0:
                            for i in inter:
                                i = np.array(i, dtype=np.float64)
                                d = np.linalg.norm(i - p)
                                intersections.append((d, i))

                    if len(intersections) == 1:
                        inter = intersections[0][1]
                        angle = np.arctan2(inter[1]-p[1], inter[0]-p[0])
                        vis_polygon.append((angle, inter))

                        intersections2 = []
                        line = Ray(p, vertice)

                        for obs3 in self.obstacles:
                            inter = obs3.intersection(line)

                            for i in inter:
                                i = np.array(i, dtype=np.float64)
                                d = np.linalg.norm(i - p)
                                intersections2.append((d, i))
                        
                        if len(intersections2) > 1:
                            intersections2.sort()
                            inter = intersections2[1][1]
                            angle = np.arctan2(inter[1]-p[1], inter[0]-p[0])
                            vis_polygon.append((angle, inter))
            
            vis_polygon.sort(key=myFunc)

            VP = []
            for vp in vis_polygon:
                i = env_to_screen_coordinates(vp[1], esc, self.res)
                VP.append(i)
            
            pygame.draw.polygon(self.screen, (0, 255, 0), VP, 0)

            #print(VP)
            pygame.time.delay(10000)

        pos_pixels = env_to_screen_coordinates(self.pos, esc, self.res)
        pygame.draw.polygon(self.screen, (0,0,255), [pos_pixels + np.array([0,-h]), pos_pixels + np.array([l,0]), pos_pixels + np.array([0,h]), pos_pixels + np.array([-l,0])], 0)
        pygame.draw.circle(self.screen, (148,0,188), env_to_screen_coordinates(self.pos, esc, self.res).astype(int), esc//16)

        pos_round =  np.around(self.pos, 2)
        myfont = pygame.font.SysFont("Comic Sans MS", 2*esc//3)
        label_pos = myfont.render('(' + str(pos_round [0]) + ',' + str(pos_round[1]) + ')', 1, (75,75,75))
        self.screen.blit(label_pos, env_to_screen_coordinates(self.pos + np.array([0.1, 0.1]), esc, self.res))

    def move_to(self, new_pos):
        self.pos = new_pos