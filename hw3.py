#!/usr/bin/env python3

import numpy as np

def deflate(T, tol):
    """
    deflates a triagonal matrix into its largest irreducible component,
    starting from the bottom-right corner.
    
    returns T,p,and q
    """

    n = T.shape[0]
    a = np.diag(T)
    b = (np.diag(T,1)+np.diag(T,-1))/2
    b[np.abs(b) < tol*abs(a[1:-1]) + tol*np.abs(a[2:])] = 0
    q,p = n,0
    
    try:
        indf = np.nonzero(b[::-1])[0][0]
    except IndexError:
        pass
    else:
        indb = indf
        while (indb >= 0 and b[indb] !=0):
            indb -= 1
        else:
            indb = 0
        p,q = indb, n-(indf+1)

    T = diag(a,0) + diag(b,1) + diag(b,-1)

    return T,p,q

def tridiagonalize(A):
    """
    return the triangularization of a matrix
    """
    T = A.copy()
    n = A.shape

    Q = np.eye(n)
    
    for k in range(n-1):
        
        v = T[k+1:,k] 
        if v[0] < 0:
            v[0] -= np.norm(v)
        else:
            v[0] += np.norm(v)

        v /= np.norm(v)

        T[k+1:,k:] -= 2*v@v.T[k+1:,k:]
        T[:,k+1:] -= 2*(T[:,k+1:]*v)*v.T
        Q[k+1:,:] -= 2*v*v.T * Q[k+1:,:]
