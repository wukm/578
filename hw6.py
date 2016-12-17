#!/usr/bin/env python3

import numpy as np

def vcycle(l,b,e0):

    #u = np.zeros(N,1)
    if l == 1: 
        #return stored result of Ax=b for small system
    else:
        e = smooth(v,b,e0)
        r = b - k 
    return e

def restrict(x):

    nf = np.shape[0]

    x[1::-1] = 0.5*x[1:-1:2] + 0.5*(x[2::2] + x[:-1:2])

    return x[1:-1:2]

def interpolate(x):

    nc = x.shape[0]
    nf = 2*nc - 1

    y = np.empty(nf )

    return y

def  

def full_multigrid(l,r):

    if l == 1:
        # this will call vcycle in the base case
        # i.e. apply the exact solution
        u = vcycle(l,b,0)
    else:
        u = interpolate(full_multigrid(l-1,4*restrict(b)))
        u += vcycle(l,b-Al



