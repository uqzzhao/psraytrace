# -*- coding: utf-8 -*-

"""
Core functions for ray tracing

@author:     Zhengguang Zhao
@copyright:  Copyright 2016-2019, Zhengguang Zhao.
@license:    MIT
@contact:    zg.zhao@outlook.com

"""


import copy

import numpy as np
import matplotlib.pyplot as plt
from math import exp, floor, ceil
from numpy.lib.scimath import sqrt
from numpy import array, linspace, ones, zeros, empty, repeat, \
    transpose, diff, where, real, cumsum, append, multiply, \
    arcsin, finfo, concatenate, square, flipud, max, ceil



def raytrace(vp, vs, zlayer, dg, src, rcv):
    #  Input geometry
    sourcex = src[:,0]
    sourcey = src[:,1]
    sourcez = src[:,2]
    receiverx = rcv[:,0]
    receivery = rcv[:,1]
    receiverz = rcv[:,2]

    max_x = max(sourcex)
    max_y = max(sourcey)
    topright_dis = ceil(sqrt(max_x*max_x + max_y*max_y)) + dg
    xmin = 0
    xmax = topright_dis
    # Make strata layer
    xx = np.arange(xmin, xmax, dg)
   

    # P wave velocity
    vp = vp.reshape(-1,1)
    vs = vs.reshape(-1,1)

    # Source-Receiver Groups
    # Source
    zs = sourcez
    xs = sourcex - sourcex
    ns = len(xs)
    # Receiver
    zr = receiverz
    nr = len(zr)
    xr = empty((nr, 1), dtype="float32")

    nray = ns * nr

    times = np.zeros((nray, 1), dtype="float32").flatten()
    thetas = np.zeros((nray, 1), dtype="float32").flatten()

    dxList = []
    dyList = []
    zhList = []

    # Run Ray tracing
    # # Loop over for number of souce
    for i in range(ns):   
                
        # Loop over for number of receiver
        for j in range(nr):
            
            xr[j] = sqrt((sourcex[i] - receiverx[j]) * (sourcex[i] - receiverx[j]) +
                         (sourcey[i] - receivery[j]) * (sourcey[i] - receivery[j]))
           
            # Compare zs and zr to determine downgoing or upgoing shooting
            if zs[i] > zr[j]:
                #  Upgoing path
                ind = where(zlayer < zs[i])
                u = array(ind[0]).reshape(-1)
               

                if len(u) == 0:
                    eup = len(zlayer)
                else:
                    eup = u[-1]
                ind = where(zlayer > zr[j])
                u = array(ind[0]).reshape(-1)
               

                if len(u) == 0:
                    sup = len(zlayer)
                else:
                    sup = u[0]

                if sup > eup:                    
                    zu = append(zr[j], zs[i])
                   
                elif sup == eup:
                    zuu = append(zr[j], zlayer[sup])
                    zu = append(zuu, zs[i])
                    
                else:
                    zuu = append(zr[j], zlayer[sup:eup+1])
                    zu = append(zuu, zs[i])
                    
                zu = zu.reshape(-1,1)
                nu = len(zu)
                zn = flipud(zu)

                # Upgoing elastic parameter
                if sup - 1 == eup:
                    vpu = vp[sup - 1]
                    vsu = vs[sup - 1]
                else:
                    vpu = vp[sup - 1:eup + 1]
                    vsu = vs[sup - 1:eup + 1]

                # Combine model elastic parameter
                vpp = flipud(vpu)
                vps = flipud(vsu)

                # Start Raytracing (P-P, S-S, or P-S mode)
                ops = 1
                sx = array([xs[i]])
                rx = array([xr[j]])
                # ops=1 for PP mode; ops=2 for PS mode
                xh, zh, vh, pp, teta, ttime = shooting(
                    vpp, vps, zn, xx, sx, rx,  ops)
                if xs[i] == xr[j]:
                    zv = abs(diff(zn, axis=0))
                    tv = zv / vpp
                    tt = cumsum(tv)
                    tt_size = tt.size
                    ttime = tt[tt_size - 1]

            elif zs[i] == zr[j]:

                # Horizontal path
                ind = where(zlayer < zs[i])
                h = array(ind[0]).reshape(-1)
                

                if len(h) == 0:
                    hor = 0
                else:
                    hor = h[-1]

                zhor = append(zs[i], zr[j])
                nu = len(zhor)
                zn = zhor.reshape(-1,1)

                # Upgoing elastic parameter
                vph = vp[hor]
                vsh = vs[hor]

                # Combine model elastic parameter
                vpp = vph
                vps = vsh

                # Start Raytracing(P - P, S - S, or P - S mode)
                ops = 1
                # ops = 1 for PP mode, ops = 2 for PS mode
                sx = array([xs[i]])
                rx = array([xr[j]])
                xh, zh, vh, pp, teta, ttime = directshooting(
                    vpp, vps, zn, xx, sx, rx, ops)

            else:
                # Downgoing path
                ind = where(zlayer > zs[i])
                d = array(ind[0]).reshape(-1)
                

                if len(d) == 0:
                    sdown = len(zlayer)
                else:
                    sdown = d[0]

                ind = where(zlayer < zr[j])
                d = array(ind[0]).reshape(-1)
               

                if len(d) == 0:
                    edown = len(zlayer)
                else:
                    edown = d[-1]

                if sdown > edown:
                    zd = append(zs[i], zr[j])
                    
                elif sdown == edown:
                    zdd = append(zs[i], zlayer[sdown])
                    zd = append(zdd, zr[j])
                    
                else:
                    zdd = append(zs[i], zlayer[sdown:edown+1])
                    zd = append(zdd, zr[j])
                   
                zd = zd.reshape(-1,1)
                nd = len(zd)
                zn = zd

                # Downgoing elastic parameter
                if sdown - 1 == edown:
                    vpd = vp[sdown - 1]
                    vsd = vs[sdown - 1]
                else:
                    vpd = vp[sdown - 1: edown + 1]
                    vsd = vs[sdown - 1: edown + 1]

                # Combine model elastic parameter
                vpp = vpd
                vps = vsd

                # Start Raytracing(P - P, S - S, or P - S mode)
                ops = 1
                # ops = 1 for PP mode, ops = 2 for PS mode
                sx = array([xs[i]])
                rx = array([xr[j]])
                xh, zh, vh, pp, teta, ttime = shooting(
                    vpp, vps, zn, xx, sx, rx, ops)
                
                if xs[i] == xr[j]:
                    zv = abs(diff(zn, axis=0))
                    tv = zv / vpp
                    tt = cumsum(tv)
                    tt_size = tt.size
                    ttime = tt[tt_size - 1]

            # Store traveltimes and incidence angles
 
            times[i * nr + j] = ttime
            thetas[i * nr + j] = abs(teta[len(teta) - 1])


            # Plot Ray
            
            L = sqrt((sourcex[i] - receiverx[j]) * (sourcex[i] - receiverx[j]) + (sourcey[i] - receivery[j]) * (sourcey[i] - receivery[j]))
            X = sourcex[i] - receiverx[j]
            Y = sourcey[i] - receivery[j]

            
            if L == 0.0:
                L = 1

            if X <= 0:
                dx = sourcex[i] + xh/L * abs(X)
            else:
                dx = sourcex[i] - xh/L * abs(X)
            
            
            if Y <= 0:
                dy = sourcey[i] + xh/L * abs(Y)
            else:
                dy = sourcey[i] - xh/L * abs(Y)

            dxList.extend(copy.deepcopy(dx.flatten()))
            dyList.extend(copy.deepcopy(dy.flatten()))
            zhList.extend(copy.deepcopy(zh.flatten()))
            
    
    rays = np.array(dxList+dyList+zhList).reshape(3,-1)                 
                
                

    return times, rays, thetas


def directshooting(vpp, vps, zn, xx, xs, xr, ops):
    # Horizontal path
    if xs < xr:
        xh = np.array([xs[0], xr[0]])
    else:
        xh = np.array([xr[0], xs[0]])

    zh = zn
    vh = vpp
    teta = np.array([0.0])
    

    if ops == 1:
        pp = 1 / vpp
        time = abs(np.diff(xh)) / vpp[0]
    else:
        pp = 1 / vps
        time = abs(np.diff(xh)) / vps[0]

    return xh, zh, vh, pp, teta, time[0]


def verticalshooting(vpp, vps, zn, xx, xs, xr, ops):
    # Horizontal path
    if xs < xr:
        xh = np.array([xs[0], xr[0]])
    else:
        xh = np.array([xr[0], xs[0]])

    zh = zn
    vh = vpp
    teta = np.array([0.0])
    

    if ops == 1:
        pp = 1 / vpp
        time = abs(np.diff(xh)) / vpp[0]
    else:
        pp = 1 / vps
        time = abs(np.diff(xh)) / vps[0]

    return xh, zh, vh, pp, teta, time[0]


def shooting(vpp, vps, zn, xx, xs, xr, ops):
    # some constants
    itermax = 50
    offset = abs(xs.reshape(-1) - xr.reshape(-1))
    
    xc = 10

    # determin option
    if (ops == 1):
        vh = vpp
    elif (ops == 2):
        vh = vps

    # initial guess of the depth & time
    zh = zn - finfo("float32").eps 
  
    
    t = float("inf") * ones((len(offset),), dtype=np.float32)
    # t = exp(100) * ones((len(offset),), dtype=np.float32)
    
    p = float("inf") * ones((len(offset),), dtype=np.float32)
    # p = exp(100) * ones((len(offset),), dtype=np.float32)

    # start raytracing
    # trial shooting
    pmax = 1 / min(vh)   
    pp = np.linspace(0, 1 / max(vh), len(xx)).reshape(1,-1)    
    temp = np.array(vh[0:len(zh)]).reshape(-1,1)   
    sln = temp.dot(pp) - exp(-20)
  
    vel = temp.dot(ones((1, np.size(pp,1)), dtype= np.float32)) # a.dot(b) matrix multiply
 
   
    if len(zh) >2:
        dz = np.array(abs(diff(zh, axis=0))).dot(ones((1, np.size(pp,1))))
    elif len(zh) == 2:
        temp1 = np.array(abs(diff(zh, axis=0))).reshape(-1,1)
        
        dz = temp1.dot(ones((1, np.size(pp,1))))
    else:
        temp2 = np.array(abs(zh)).reshape(-1,1)
        
        dz = temp2.dot(ones((1, np.size(pp,1))))
    
    
    dim_sln = sln.shape
    
    if (dim_sln[0] > 1):
        xn = np.sum((dz * sln) / sqrt(1 - sln**2), axis=0) # need to assign axis
       
        tt = np.sum(dz / (vel * sqrt(1 - sln**2)), axis=0)
        
    else:
        xn =(dz * sln) / sqrt(1 - sln**2) # a * b element-wise multiply
        tt = dz / (vel * sqrt(1 - sln**2)) # a / b element-wise divide
 
    
    if xn.ndim>1:
        xn = xn[0,:]
        tt = tt[0,:]
    pp = pp[0,:]

    
    xmax = xn.max()
   

    # bisection method
    # start bisection method

    for k in range(len(offset)):
        
        # analyze the radius of target
        n = len(xn)        
        xa = xn[0:n - 1].flatten()       
        xb = xn[1:n].flatten()
       
        opt1 = empty((1, n - 1)).flatten()
        opt2 = empty((1, n - 1)).flatten()
        opt = empty((1, n - 1)).flatten()
        for i in range(n - 1):
            if xa[i] <= offset[k] and xb[i] > offset[k]:
                opt1[i] = 1
            else:
                opt1[i] = 0

            if xa[i] >= offset[k] and xb[i] < offset[k]:
                opt2[i] = 1
            else:
                opt2[i] = 0

        opt = opt1 + opt2      

        
        ind = where(opt == 1)

      
        if len(ind) == 0:
            if (offset(k) >= xmax):
                a = n
                b = []
            else:
                a = []
                b = 1
        else:
            a = ind[0]
            b = ind[0] + 1
       

        x1 = xn[a]
        x2 = xn[b]
        t1 = tt[a]
        t2 = tt[b]
        p1 = pp[a]
        p2 = pp[b]
        iter = 0
        err = (b - a) / 2
        
        # Minimize the error & intersect the reflector        
        while((iter < itermax) and abs(err) < 1):
            
            iter = iter + 1
            xt1 = abs(offset[k] - x1)
            xt2 = abs(offset[k] - x2)
            if (xt1 < xc) and (xt1 <= xt2):
                # linear interpolation
                t[k] = t1 + (offset[k] - x1) * (t2 - t1) / (x2 - x1)
                p[k] = p1 + (offset[k] - x1) * (p2 - p1) / (x2 - x1)
            elif (xt2 < xc) and (xt2 <= xt1):
                t[k] = t2 + (offset[k] - x2) * (t1 - t2) / (x1 - x2)
                p[k] = p2 + (offset[k] - x2) * (p1 - p2) / (x1 - x2)
            # set new ray parameter
            if a.size == 0:
                p2 = p1
                p1 = 0
            elif b.size == 0:
                p1 = p2
                p2 = pmax
           
            pnew = linspace(array([p1, p2]).min(), array([p1, p2]).max(), 3)
            pnew = pnew.reshape(1,-1)
            pnew2 = (pnew[:, 1]).reshape(1,1)
            
           

            # do shooting by new ray parameter
            temp = np.array(vh[0:len(zh)]).reshape(-1,1)
                        
            sln = temp.dot(pnew2) 
            
            
            vel = temp.dot(ones((1, len(pnew2)))) # a.dot(b) matrix multiply
            
           
            if len(zh) >2:
                dz = np.array(abs(diff(zh, axis=0))).dot(ones((1, len(pnew2))))
            elif len(zh) == 2:
                temp1 = (abs(diff(zh, axis=0))).reshape(1,1)                
                dz = temp1.dot(ones((1, len(pnew2))))
            else:
                temp2 = (abs(zh)).reshape(1,1)
                
                dz = temp2.dot(ones((1, len(pnew2))))
            
            
            dim_sln = sln.shape
            
            if (dim_sln[0] > 1):
                xtemp = sum((dz * sln) / sqrt(1 - sln**2))                
                ttemp = sum(dz / (vel * sqrt(1 - sln**2)))
                
            else:
                xtemp = (dz * sln) / sqrt(1 - sln**2)
                ttemp = dz / (vel * sqrt(1 - sln**2))
            xnew = array([x1, xtemp[0], x2]).flatten()          
            tnew = array([t1, ttemp, t2]).flatten()
            
          
            xmax = xnew.max()
          
            # analyze the radius of target
            n = len(xnew)           
            xa = xnew[0:n - 1]          
            xb = xnew[1:n]
          
            opt1 = empty((1, n - 1)).flatten()
            opt2 = empty((1, n - 1)).flatten()
            opt = empty((1, n - 1)).flatten()

            for i in range(n - 1):
                
                if xa[i] <= offset[k] and xb[i] > offset[k]:
                    opt1[i] = 1
                else:
                    opt1[i] = 0
                if xa[i] >= offset[k] and xb[i] < offset[k]:
                    opt2[i] = 1
                else:
                    opt2[i] = 0

            opt = opt1 + opt2            
            
            ind = where(opt == 1)            
            
            a = ind[0]
            b = ind[0] + 1
            
            pnew = pnew[0]            
            
            x1 = xnew[a]
            x2 = xnew[b]
           
            t1 = tnew[a]
            t2 = tnew[b]
            p1 = pnew[a]
            p2 = pnew[b]
            err = (b - a) / 2

            # declare ray parameter
            if xr[0] > xs[0]:
                pp = p
            else:
                pp = -p
            # compute travel time & angle
            dx = real((pp * vh * dz) / sqrt(1 - pp * pp * vh * vh))
           
            xx = xs + cumsum(dx)
           
            xh = append(xs, xx)
            xh.reshape(-1,1)
           
            dz = real(dx * sqrt(1 - pp * pp * vh * vh)) / (pp * vh)
            dt = dz / (vh * sqrt(1 - pp * pp * vh * vh))
            tt = cumsum(dt)
            tt_size = tt.size
            time = tt[tt_size - 1]
        

            teta = real(arcsin(multiply(pp, vh)))
          
    
    return xh, zh, vh, pp, teta, time

