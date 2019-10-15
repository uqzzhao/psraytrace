# -*- coding: utf-8 -*-

"""
A 2D raytracing example typically seen in downhole monitoring scenario.

@author:     Zhengguang Zhao
@copyright:  Copyright 2016-2019, Zhengguang Zhao.
@license:    MIT
@contact:    zg.zhao@outlook.com

"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from copy import deepcopy
from psmodules.psraytrace import raytrace
import matplotlib.animation as animation

def run():
    
    # Receivers
    filePath = './demo_data/rcv_downhole.csv'
    csvData = pd.read_csv(filePath)
    rcv = np.array(csvData)

    # Sources
    filePath = './demo_data/src_downhole.csv'
    csvData = pd.read_csv(filePath)
    src = np.array(csvData)
    nsrc = src.shape[0]

    # Display sources and receivers locations
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111)

    ax.scatter(rcv[:,2], rcv[:,3], c='r', marker='v')
    ax.scatter(src[:,3], src[:,4], c='b', marker='o')
    ax.invert_yaxis()
    ax.set_xlabel('Y (m)')
    ax.set_ylabel('Z (m)')
    
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
        ani, = ax.plot([], [],'k-', linewidth=0.5)

        ni = len(rays[1,:])
        def animate(i):
            dy = rays[1, :i+2]
            zh = rays[2, :i+2]
            ani.set_data(dy, zh)
            return ani,

        plot = animation.FuncAnimation(fig, animate, frames=ni-2, 
                                    interval=100, blit=True)

    else:
        
        for i in range(nsrc):
            dx = deepcopy(rays[0, :])
            dy = deepcopy(rays[1, :])
            zh = deepcopy(rays[2, :])
            
            ax.plot(dy, zh,'k-', linewidth=0.5)
            fig.canvas.draw()
        
   
    plt.show()


