#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 18:51:21 2018
@author: pabloruizruiz
"""

import random
from random import randint
import numpy as np

from brain import update_beliefs
from brain import initialize_brain


class landmark:
    
    def __init__(self, x = 0, y = 0, visible=False):
        self.position = (x, y)
        self.visible = visible
        
        
class control:
    
    def __init__(self):
        self.self_ = list()
        self.landmarks_ = list()
        

class agent:

    def __init__(self, timesteps = 3, world_size = 100.0, 
                 measurement_range = 5.0, motion_noise = None, measurement_noise = None, comments=False):
        self.lifetime = timesteps
        self.age = 0.0
        self.measurement_noise = 0.0
        self.world_size = world_size
        self.measurement_range = measurement_range
        self.x = world_size / 2.0
        self.y = world_size / 2.0
        self.motion_noise = motion_noise
        self.measurement_noise = measurement_noise
        self.landmarks = []
        self.num_landmarks = 0
        self.omega = None
        self.xi = None
        self.mu = None
        self.control = control()
        self.control.self_.append([self.x, self.y])
        self.comments = comments
        
        
    
    def wakeup(self, N):
        ''' Wake up the agent with this time step life span and the world he is in '''
        self.omega, self.xi = initialize_brain(N, self.num_landmarks, self.world_size)
        self.control.landmarks_ = [(0,0) for _ in range(self.num_landmarks)]
   
     
    def get_old(self):
        ''' Time pass also for robots '''
        self.age += 1
        
        
    def think(self, i, x, y, noise, act:str):
        ''' Brain of the agent that store its belief states space '''
        err = 'Hey, I only know how to move and see... Teach me more things!'
        assert act == 'sense' or act == 'move', err
        self.omega, self.xi, _ = update_beliefs(self.lifetime, self.omega, self.xi,
                                                      self.age, i, x, y, noise, act)
            
    def move(self):
        ''' Move the agent. Assert we are not leaving the word! '''      
        # Ensure the new position would be out of the world
        s = 3
        dx = randint(max(-s, int(0 - self.x + 1)), min(int(self.world_size - self.x -1), s))
        dy = randint(max(-s, int(0 - self.y + 1)), min(int(self.world_size - self.y -1), s))
        if self.comments: print('Moving: ', (dx, dy))
        
        # Change the new position and introduce some real world inaccuracy
        x = self.x + dx + random.uniform(-1, 1) * self.motion_noise
        y = self.y + dy + random.uniform(-1, 1) * self.motion_noise
        
        # Conclude, update beliefs and update position
        if self.comments: 
            print('* Sensing my movement...')
            print('* Thinking, updating my matrices...')
            print('-'*len('* Thinking, updating my matrices...'))
        self.think(0, dx, dy, self.measurement_noise, act='move')
        self.x = x
        self.y = y
        
        # Append to the control of the position
        self.control.self_.append([self.x, self.y])
    
    
    def sense(self):
        ''' Look at the landmarks and stimate the distance '''
        measurements = []
        for i, l in enumerate(self.landmarks):
            
            # Calculate the distance to landmark
            lx, ly = l.position[0], l.position[1]                   
            dx, dy = self.x - lx, self.y - ly                   
            d = np.sqrt(dx**2 + dy**2)
            
            # Add noise to landmark estimation(sensors are not perfect!)
            lx += random.uniform(-1, 1) * self.measurement_noise    
            ly += random.uniform(-1, 1) * self.measurement_noise
            
            if d > self.measurement_range:
                self.landmarks[i].visible = False
                
            # Ensure the measure is within the measurable range
            if d < self.measurement_range: 
                
                if self.comments: print('* Calculating distance to landmark #{}...'.format(i+1))                
                measurements.append([lx, ly]) # The measurement is the noised position of the landmark
                
                if self.comments: print('* Thinking, updating my matrices...')
                self.think(i, dx, dy, self.measurement_noise, act='sense') # The omega-xi matrices get the distance to landmaks
                
                # Update the visible field of the landmark
                self.landmarks[i].visible = True
                
        # Have a conclusion
        if self.comments: 
            conclusion = '\n\n Alright! I am detecting {} tree(s) on my sensor range'
            print(conclusion.format(len(measurements)))   
            print('-'*len(conclusion))
        return measurements

    
    def __repr__(self):
        ''' Overwrite printting function to print agent location '''
        return '[x=%.5f y=%.5f]'  % (self.x, self.y)
    
    
    

