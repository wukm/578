#!/usr/bin/env python3

"""
USED FOR HW 2 PROBLEM 2
MATH 578 Luke Wukmer Fall 2016
blah blah blah
"""

import numpy as np

def tridiagonal(A):
    """
    A must be square. or an integer
    returns a tridiagonal matrix (ones and zeros) if input is an
    integer. if input is a (dense) matrix,
    then zero out what is not on the tridiagonal
    """
    try:
        N = int(A)
    except TypeError:
        N = A.shape[0]
    else:
        A = np.ones((N,N))
                
    trid = np.eye(N) + np.eye(N,k=1) + np.eye(N,k=-1)
    
    return trid*A


def givens(x):
    """
    returns the components c,s of the rotation matrix

    G = ( c   -s )
        ( s    c )

    such that the 2x1 input x = [a b]^T would be rotated to align with e_1:

    G x = [ ~, 0]^T for some number ~
    """

    a, b = x

    # this equals 2**(-52) exactly when x is float64
    eps = np.finfo(x.dtype).eps

    if np.abs(a) < eps:
        c, s = 1, 0
    elif np.abs(b) > np.abs(a):
        tau = -a/b
        
        s = 1 / ( np.sqrt(1+tau*tau))
        c = tau*s
    else:
        tau = -b/a
        c = 1 / ( np.sqrt(1+tau*tau))
        s = tau*c

    return c,s

def givens_qr(A):
    """ applies givens QR"""
    
    # A needs to be at least float64 dtype
    A = A.astype('float64')
    R = A.copy()
    n = A.shape[0]
    V = np.zeros((n-1,2))
    
    # for each element on diagonal except for last
    G = lambda c, s : np.array([[c,-s],[s,c]]) 
    for k in range(n-1):
        x = R[k:k+2,k] # diagonal and entry below
        
        g  = givens(x) # givens coefficients

        V[k,:] = g
        # apply to whole subblock
        #R[k:k+2,:] = G(*g) @ R[k:k+2,:]
    
        # apply to 2x3 subblock (assuming tridiagonal)
        R[k:k+2,k:k+3] = G(*g) @ R[k:k+2,k:k+3]
        
    # now obtain Q by backwards accumulation
    Q = np.eye(n)

    for k in range(n-2,-1,-1):
        c,s = V[k,:]     
        qk = np.eye(n)
        qk[k:k+2,k:k+2] = G(c,s)
        Q = qk.T @ Q
    return Q,R
    
if __name__ == "__main__":

    from numpy.linalg import norm
    
    np.set_printoptions(precision=3)
    
    for i in range(20):
        A = 100*np.random.random((100,100))
        A = tridiagonal(A)
        Q,R = givens_qr(A)
        rel_error = norm(Q@R - A,2)
        ortho_error = norm(Q@Q.T - np.eye(100),2)
        
        print('i=',i, '--'*30, sep=' ')
        print('\tfactor error', rel_error, sep='\t')
        print('\tortho error', ortho_error, sep='\t')
