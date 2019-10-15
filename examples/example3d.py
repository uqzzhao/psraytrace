# -*- coding: utf-8 -*-

"""
A 3D raytracing example typically seen in surface monitoring monitoring scenario.

@author:     Zhengguang Zhao
@copyright:  Copyright 2016-2019, Zhengguang Zhao.
@license:    MIT
@contact:    zg.zhao@outlook.com

"""
import sys 
sys.path.append(".") 

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
from copy import deepcopy
from psmodules.psraytrace import raytrace
import matplotlib.animation as animation

def run():

    # Receivers
    filePath = './demo_data/rcv_surface.csv'
    csvData = pd.read_csv(filePath)
    rcv = np.array(csvData)

    # Sources
    filePath = './demo_data/src_surface.csv'
    csvData = pd.read_csv(filePath)
    src = np.array(csvData)
    nsrc = src.shape[0]

    # Display sources and receivers locations
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(rcv[:,1], rcv[:,2], rcv[:,3], c='r', marker='v')
    ax.scatter(src[:,2], src[:,3], src[:,4], c='b', marker='o')
    ax.invert_zaxis()
    ax.set_xlabel('Easting (m)')
    ax.set_ylabel('Northing (m)')
    ax.set_zlabel('Depth (m)')
    
    #plt.show()

    # Velocity Model
    filePath = './demo_data/vm_layered.csv'
    csvData = pd.read_csv(filePath)
    vm = np.array(csvData)
    vp = vm[:,1]
    vs = vm[:,2]
    vz = vm[:,0]

    # Raytracing 
    dg = 10
    times, rays, _ = raytrace(vp, vs, vz, dg, src[:,2:], rcv[:,1:])
    

    FLAG = 1 # 1= animated, 0= static
    # Display rays
    if FLAG == 1:

        # animation
        ani, = ax.plot([],[],[],'k-', linewidth=0.5)
        ni = len(rays[0, :])
        
        def init():
            ani.set_data([], [])
            return ani,

        def animate(i):
            dx = rays[0, :i+2]
            dy = rays[1, :i+2]
            zh = rays[2, :i+2]
            ani.set_data(dx, dy)
            ani.set_3d_properties(zh)
            return ani,

        plot = animation.FuncAnimation(fig, animate, frames=ni-2, 
                                    interval=10, blit=True, init_func = init)

    else:
        for i in range(nsrc):
            dx = deepcopy(rays[0, :])
            dy = deepcopy(rays[1, :])
            zh = deepcopy(rays[2, :])
            
            ax.plot(dx, dy, zh,'k-', linewidth=0.5)
            fig.canvas.draw()
        
   
    plt.show()

    