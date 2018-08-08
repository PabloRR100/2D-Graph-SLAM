#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 19:22:40 2018
@author: pabloruizruiz
"""

from math import *
import numpy as np
import pandas as pd
from random import random

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from objects import landmark
robot = image.imread('imgs/robot.png')
tree  = image.imread('imgs/tree.png')


def make_landmarks(agent, num_landmarks):
    ''' Create random landmarks on the world '''
    landmarks = []
    for i in range(num_landmarks):
        l = landmark(x = round(random() * (agent.world_size - 2)),
                     y = round(random() * (agent.world_size - 2)))
        landmarks.append(l)
    agent.landmarks = landmarks
    agent.num_landmarks = len(agent.landmarks)
    

def print_beliefs(agent):
    
    fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw = {'width_ratios':[8, 1]}, figsize = (15,5))
    sns.heatmap(pd.DataFrame(agent.omega), cmap='Oranges', annot=True, linewidths=.5, ax=ax1)
    sns.heatmap(pd.DataFrame(agent.xi), cmap='Blues', annot=True, linewidths=.5, ax=ax2)
    ax1.set_title('Omega')
    ax2.set_title('Xi')
    plt.show()


def display_world(agent, estimates=None):
    ''' Display the world, with the estimation if any '''   
    world_size = agent.world_size
    position = (agent.x, agent.y)
    mrange = agent.measurement_range
    landmarks = agent.landmarks
    
    sns.set_style("dark")
    fig, ax = plt.subplots()
    
    # Create the robot
    oi = OffsetImage(robot, zoom = 0.04)
    box = AnnotationBbox(oi, (position[0], position[1]), frameon=False)
    ax.add_artist(box)
    ax.plot(position[0], position[1])
    
    # Plot the measurement_range
    m_range = plt.Circle((position[0], position[1]), radius=mrange, alpha=0.15, color='blue')
    ax.add_artist(m_range)
    
    # Draw landmarks if they exists
    if(landmarks is not None):
        oit = OffsetImage(tree, zoom = 0.7)
        lands = [l.position for l in landmarks] 
        # loop through all path indices and draw a tree (unless it's at the robot's location)
        for pos in lands:
            if(pos != position):
                boxt = AnnotationBbox(oit, (pos[0], pos[1]), frameon=False)
                ax.add_artist(boxt)
                
    # Draw estimates
    if estimates is not None and len(estimates) > 0:
        
        # This measurments are the actual positions of the landmarks with the added noise
#        land = [l.position for l in landmarks if l.visible == True] # Only the visibles
#        ests = np.array(estimates)
#        ests = np.array([tuple(m) for m in estimates])
#        estimates = (land + ests).tolist()
        cc = position # Put the estimate position here were corrected
        points = [(x,y) for x,y in estimates]   
        x_p, y_p = zip(*points)    
        for p in points:
            x, y = zip(*[p,cc])
            ax.plot(x, y, color = 'green')
#        ax.scatter(cc[0], cc[1], s=20, color='black')
        ax.scatter(x_p, y_p, s=30, color='green', marker = 'x')
    
    # Format the grid
    cols = world_size + 1
    rows = world_size + 1
    
    ax.set_xticks([x for x in range(0, cols)], minor=False)
    ax.set_yticks([y for y in range(0, rows)], minor=False)
    
    ax.grid(which='minor',ls='-',lw=1, color='white')
    ax.grid(which='major',ls='-',lw=2, color='white')
    
    plt.show()


def plot_estimations(mu, re):
    
    di = mu - re
    er = np.zeros((di.shape[0])).reshape(-1,1)    
    for i, p in enumerate(er):
        er[i] = np.sqrt(di[i][0]**2 + di[i][1]**2)

    mu = pd.DataFrame(mu, columns = ['x', 'y'])
    re = pd.DataFrame(re, columns = ['x', 'y'])
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, gridspec_kw = {'height_ratios':[1, 1, 3]}, figsize = (15,10))
    sns.heatmap(mu.T, cmap='Blues', annot=True, linewidths=.5, cbar=False, xticklabels=(np.arange(50) + 1), ax=ax1)
    sns.heatmap(re.T, cmap='Blues', annot=True, linewidths=.5, cbar=False, xticklabels=(np.arange(50) + 1), ax=ax2)
    sns.barplot(x=(np.arange(50) + 1), y = er.reshape(-1,), color = 'red')
    ax1.set_title('Predictions')
    ax2.set_title('Real position')
    ax3.set_title('Errors')
    plt.show()



#    
#
#def make_data(N, num_landmarks, world_size, measurement_range, motion_noise, 
#              measurement_noise, distance):
#
#    # check that data has been made
#    try:
#        check_for_data(num_landmarks, world_size, measurement_range, motion_noise, measurement_noise)
#    except ValueError:
#        print('Error: You must implement the sense function in robot_class.py.')
#        return []
#    
#    complete = False
#    
#    r = robot(world_size, measurement_range, motion_noise, measurement_noise)
#    r.make_landmarks(num_landmarks)
#
#    while not complete:
#
#        data = []
#
#        seen = [False for row in range(num_landmarks)]
#    
#        # guess an initial motion
#        orientation = random() * 2.0 * pi
#        dx = cos(orientation) * distance
#        dy = sin(orientation) * distance
#            
#        for k in range(N-1):
#    
#            # collect sensor measurements in a list, Z
#            Z = r.sense()
#
#            # check off all landmarks that were observed 
#            for i in range(len(Z)):
#                seen[Z[i][0]] = True
#    
#            # move
#            while not r.move(dx, dy):
#                # if we'd be leaving the robot world, pick instead a new direction
#                orientation = random() * 2.0 * pi
#                dx = cos(orientation) * distance
#                dy = sin(orientation) * distance
#
#            # collect/memorize all sensor and motion data
#            data.append([Z, [dx, dy]])
#
#        # we are done when all landmarks were observed; otherwise re-run
#        complete = (sum(seen) == num_landmarks)
#
#    print(' ')
#    print('Landmarks: ', r.landmarks)
#    print(r)
#
#
#    return data
#
#
#def check_for_data(num_landmarks, world_size, measurement_range, motion_noise, measurement_noise):
#    # make robot and landmarks
#    r = robot(world_size, measurement_range, motion_noise, measurement_noise)
#    r.make_landmarks(num_landmarks)
#
#    # check that sense has been implemented/data has been made
#    test_Z = r.sense()
#    if(test_Z is None):
#        raise ValueError




# Backup definition of display world
#def display_world(agent):
#    
#    world_size = agent.world_size
#    position = (agent.x, agent.y)
#    mrange = agent.measurement_range
#    landmarks = agent.landmarks
#    
#    sns.set_style("dark")
#    fig, ax = plt.subplots()
#    
#    # Create the robot
#    oi = OffsetImage(robot, zoom = 0.04)
#    box = AnnotationBbox(oi, (position[0], position[1]), frameon=False)
#    ax.add_artist(box)
#    ax.plot(position[0], position[1])
#    
#    # Plot the measurement_range
#    m_range = plt.Circle((position[0], position[1]), radius=mrange, alpha=0.15, color='blue')
#    ax.add_artist(m_range)
#    
#    # Draw landmarks if they exists
#    if(landmarks is not None):
#        landmarks = [l.position for l in landmarks] 
#        oit = OffsetImage(tree, zoom = 0.7)
#        # loop through all path indices and draw a tree (unless it's at the car's location)
#        for pos in landmarks:
#            if(pos != position):
#                boxt = AnnotationBbox(oit, (pos[0], pos[1]), frameon=False)
#                ax.add_artist(boxt)
#    
#    # Format the grid
#    cols = world_size + 1
#    rows = world_size + 1
#    
#    ax.set_xticks([x for x in range(0, cols)],minor=False )
#    ax.set_yticks([y for y in range(0, rows)],minor=False)
#    
#    ax.grid(which='minor',ls='-',lw=1, color='white')
#    ax.grid(which='major',ls='-',lw=2, color='white')
#    
#    plt.show()