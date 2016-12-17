#!/usr/bin/env python3

import numpy as np
from numpy.linalg import norm

from scipy import sparse

from givens import givens

def backsolve(R,b):
    """
    solves a system Rx=b via back substituation where R is an
    upper-triangular matrix
    """
    n = R.shape[0]

    x = np.zeros(b.shape)

    x[-1] = b[-1] / R[-1,-1] # last entry to initialize
    for i in range(n-2,-1,-1):
        x[i] = (b[i] - R[i,i+1:]@x[i+1:]) / R[i,i]
    
    return x

def apply_givens(v,h):
    print(h.shape) 
    if v.size == 0:
        pass  # apply no rotations; v is empty
    else:
        for j in range(h.size-2):
            print(j)
            c,s = list(v[j,:]) # jth givens coefficients

            G = np.array([[c, -s],
                          [s,  c]])

            h[j:j+2] = G @ h[j:j+2] 
    return h

def pcgmres(A, b, tol, M_inv):
    """
    preconditioned GMRES
    inputs
        A   input system A
        b   inital conditions
        tol
        M_inv inverse of preconditioner M

    outputs
        x   solution to Ax=b
        it  iteration count

    """

    b = M_inv@b
    β = norm(b,2)
    
    # will store Arnoldi vectors basis (expands in cols)
    Q = b / β

    R = np.empty(0) # expands in size
    V = np.empty(0) # expands in size

    r = β # always a scalar
    t = β # a scalar, but becomes a vector

    n = A.shape[0]
    
    # make sure to check & fix iteration count
    for it in range(1,n+1):
        print('iteration ', it)
        print('\tr=',r)
        print('\tβ=',β)
        if (r <= tol*β):
            break

        z = M_inv @ (A @ Q[:,-1]) # Aq_k (i.e. latest Q vector)
        h = (Q.T @ z)
        #h = np.expand_dims(h,-1)
        h_tilde = np.sqrt(norm(z) - norm(h))
        # will be empty on first pass, hope that's okay
        h_hat = apply_givens(V,h)

        c, s = givens(np.array((h_hat[-1],h_tilde)))

        h_hat[-1] = c*h_hat[-1] - s*h_tilde

        # form upper triangular matrix R
        try:
            R = np.hstack((R, np.expand_dims(h_hat,-1)))
        except ValueError:
            #R = h_hat.copy()
            R = np.expand_dims(h_hat,-1)
            #print("R is now ", R)
            #import sys
            #sys.exit()
            

        # apply and store givens rotations, etc.
        if V.size == 0:
            V = np.array([c,s])
        else:
            V = np.vstack((V, np.array([c,s]))) # k x 2
            


        # fix last two elements t grows in size by 1
        t = np.hstack((t,0)) # i don't believe this is memory efficient
        try:
            t[-2:] = V[-1,:]*t[-2] # apply c & s to second to last t to get last t
        except IndexError:
            if it == 1:
                t[-2:] = V[-1:] * t[-2] 
            else:
                raise 
        print(it,"-------------")
        print(t)
        r = np.abs(t[-1])


        
        # if there will be another iteration
        if r >= tol*β and it < n:
            # compute next Arnoldi vector
            q = z - Q@h
            q = q / norm(q) # or w / h_tilde (same number)
    
            # add on additional basis vector
            q = np.expand_dims(q,-1)
            Q = np.hstack((Q,q))
    # solve Ry = t[:]
    y = backsolve(R,t)
    # form approximation x
    x = Q@y
    
    return x

def pcg(A,b,tol, Minv):
    pass

def p1_system(n):
    """
    A_{i,j} = { 2 + (1.1)^i     if j = i
              { -1              if j = i+1 , i-1
              { 0               otherwise
    """
    A = 2*np.eye(n) - np.eye(n,k=-1) - np.eye(n,k=1)
    A += np.diag(np.fromfunction(lambda i: 1.1**(i+1) , (n,)))
    return A

if __name__ == "__main__":
    n = 6
    A = p1_system(n)
    b = np.ones((n,1))

    x, endit = pcgmres(A,b,tol=10e-12, M_inv=np.eye(n))

    

