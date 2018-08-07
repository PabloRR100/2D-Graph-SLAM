#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 19:19:51 2018
@author: pabloruizruiz
"""

import os
import numpy as np
from objects import agent
from matplotlib import pyplot as plt

os.chdir('/Users/pabloruizruiz/OneDrive/Proyectos/Udacity CVNP/E3_SLAM')
from utils import display_world, make_landmarks, plot_estimations


# 1 - Construct the world, create the agent and place the landmarks
# -----------------------------------------------------------------

''' Parameters of the world '''
timesteps          = 50                      # Number of time steps the agent will live
world_size         = 30                     # Size of world (square)
measurement_range  = 10                      # Range at which we can sense landmarks
motion_noise       = 0.2                    # Noise in robot motion
measurement_noise  = 0.2                    # Noise in the measurements
num_landmarks = 3                           # Number of Landmarks to include in the world
plt.rcParams["figure.figsize"] = (10,10)    # Size of the visualization of the simulation


# 1.1 - Creates an Agent

r = agent(timesteps, world_size, measurement_range, motion_noise, measurement_noise)
#display_world(agent = r)


# 1.2 - Create any number of landmarks
make_landmarks(r, num_landmarks)
#display_world(agent = r)


# 2 - Agent starts to move and sense the world 
# --------------------------------------------

# 2.1 - Wake up agent
r.wakeup(timesteps)
print('Waking up, initializing system...: ', '\n', r.omega, '\n', r.xi, '\n', r.mu, '\n')
print('Good, I am certain I am at time 0, and position', r)


# 2.2 - Sense any surrounding landmarks
measurements = r.sense()
display_world(agent = r, estimates = measurements)


# 3 - Move the agent
# ------------------
for t in range(timesteps - 1):
    
    # 3.1 - Explore the world
    r.move()
    
    # 3.2 - Sense the landmarks within the agent range
    measurements = r.sense()
    
    # 3.3 - Age the agent
    r.get_old()
    
    # 3.4 - Display current picture of the world
    if t % 10 == 0:
        display_world(agent = r, estimates = measurements)


mu = np.dot(np.linalg.inv(r.omega), r.xi).reshape(-1,2)[:-3,:]
re = np.array(r.control.self_)    
plot_estimations(mu, re)
