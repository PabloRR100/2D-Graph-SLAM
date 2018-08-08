#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 08:54:29 2018
@author: pabloruizruiz
"""

import numpy as np


def initialize_brain(N, num_landmarks, world_size):
    ''' Initialize O and Sigma with only knowing the initial position '''
    rows, cols = (2*N + 2*num_landmarks), (2*N + 2*num_landmarks)
    cx, cy = world_size / 2, world_size / 2
    
    O = np.zeros((rows, cols))
    xi = np.zeros((rows, 1))
    
    # Initialize with 100% confidence at the center of the world
    O[0][0], O[1][1] = 1, 1
    xi[0][0], xi[1][0] = cx, cy
    return O, xi


def update_beliefs(N, O, xi, n:int, i:int, x, y, noise, act: str = 'sense'):
    ''' Recalculate matrices based on new perception ''' 
    i = int(i)
    n = int(n)        
    
    # Measurement steps
    if act == 'sense':
        
        x_pos,y_pos = n*2, n*2+1   
        l_x_pos,l_y_pos = (N+i)*2, (((N+i)*2)+1)

        O[x_pos,x_pos] -= 1/noise
        O[x_pos,l_x_pos] += 1/noise
        O[l_x_pos,x_pos] += 1/noise
        O[l_x_pos,l_x_pos] -= 1/noise

        O[y_pos,y_pos] -= 1/noise
        O[y_pos,l_y_pos] += 1/noise
        O[l_y_pos,y_pos] += 1/noise
        O[l_y_pos,l_y_pos] -= 1/noise

        xi[x_pos] += x/noise
        xi[l_x_pos] -= x/noise
        xi[y_pos] += y/noise
        xi[l_y_pos] -= y/noise
    
    # Motion Steps
    elif act == 'move':
        
        x_pos,y_pos = n*2, n*2+1   
        x_pos_m, y_pos_m = (n+1)*2, ((n+1)*2)+1

        O[x_pos,x_pos] -= 1/noise
        O[x_pos,x_pos_m] += 1/noise
        O[x_pos_m,x_pos] += 1/noise
        O[x_pos_m,x_pos_m] -= 1/noise

        O[y_pos,y_pos] -= 1/noise
        O[y_pos,y_pos_m] += 1/noise
        O[y_pos_m,y_pos] += 1/noise
        O[y_pos_m,y_pos_m] -= 1/noise

        xi[x_pos] += x/noise
        xi[x_pos_m] -= x/noise
        xi[y_pos] += y/noise
        xi[y_pos_m] -= y/noise
    
    # print('|O| = ', np.linalg.det(O))
    # mu = np.dot(np.linalg.inv(O), xi)
    return O, xi, 0