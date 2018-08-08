#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 08:54:29 2018
@author: pabloruizruiz
"""

import numpy as np


def initialize_brain(N, num_landmarks, world_size):
    ''' Initialize Omega and Sigma with only knowing the initial position '''
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
    
    def noisy(v):  
        if noise == 0:
            val = v
        else:    
            val = v/noise
        return val
    
    # The v = 1 should be replaced by the confidence of the measure
    def update_idx(M, idx, v=1, vector = False): 
        for j, i in enumerate(idx):
            if not vector:
                c = [1, -1, -1, 1]
                M[i[0]][i[1]] += c[j] * noisy(v)
            else:
                c = [-1, 1]
                M[i[0],0] += c[j] * noisy(v)
        return M
    
    # Measurement steps
    if act == 'sense':
        
        # Indices to update in Omega
        Ox = np.array([[2*n, 2*n], [2*n, 2*N + 2*i], [2*N + 2*i, 2*n], [2*N + 2*i, 2*N + 2*i]])
        Oy = Ox + np.array([[1,1]]*4)
        # Indices to update in Xi
        Xx = np.array([[2*n,0], [2*i + 2*N, 0]])
        Xy = Xx + np.array([[1,0]]*2)
        
        O = update_idx(O, Ox)       # Update O with the x-measurement
        O = update_idx(O, Oy)       # Update O with the y-measurement
        xi = update_idx(xi, Xx, x, vector = True)  # Update Xi with the x-measurement
        xi = update_idx(xi, Xy, y, vector = True)  # Update Xi with the y-measurement
    
    # Motion Steps
    elif act == 'move':
        
        # Indices to update in Omega
        Ox = np.array([[2*n, 2*n], [2*n, 2*n + 2], [2*n + 2, 2*n], [2*n + 2, 2*n + 2]])
        Oy = Ox + np.array([[1,1]]*4)
        # Indices to update in Xi
        Xx = np.array([[2*n, 0], [2*i + 2, 0]])
        Xy = Xx + np.array([[1,0]]*2)
        
        O = update_idx(O, Ox)       # Update O with the x-measurement
        O = update_idx(O, Oy)       # Update O with the y-measurement
        xi = update_idx(xi, Xx, x, vector = True)  # Update Xi with the x-measurement
        xi = update_idx(xi, Xy, y, vector = True)  # Update Xi with the y-measurement
    
    # print('|O| = ', np.linalg.det(O))
    # mu = np.dot(np.linalg.inv(O), xi)
    return O, xi, 0