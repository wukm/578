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
    """
    apply the givens transformation to a vector
    """
    print('starting apply_givens with v.shape ==',v.shape, 'and',
            'h.shape==',h.shape)
    
    if v.size == 0:
        pass  # apply no rotations; v is empty
    else:
        #h = np.resize(h,(h.shape[0]+1,1))
        #h[-1,-1] = 0
        for j in range(v.shape[0]-1):
            c,s = v[j,:] # jth givens coefficients

            G = np.array([[c, -s],
                          [s,  c]])

            h[j:j+2] = G @ h[j:j+2]

    return h

def pcgmres(A, b, tol, M_inv):
    """
    preconditioned GMRES
    inputs
        A   input system A nxn
        b   inital conditions
        tol
        M_inv inverse of preconditioner M

    outputs
        x   solution to Ax=b
        it  iteration count

    """
    if b.ndim == 1:
        b = np.expand_dims(b,-1)

    b = M_inv@b
    β = norm(b,2)
    
    Q = b / β    # will store Arnoldi vectors basis (expands in cols)

    R = np.empty((0,0)) # expands in size
    V = np.empty((0,2)) # expands in size

    r = β # always a scalar
    t = np.array([[β]]) # expanding vector

    n = A.shape[0]
    
    # make sure to check & fix iteration count
    for it in range(1,n+1):
        print('iteration ', it)
        print('\tr=',r)
        print('\ttol*β=',tol*β)
        if (r <= tol*β):
            break

        z = M_inv @ (A @ Q[:,-1:]) # Aq_k (i.e. latest Q vector)
        h = (Q.T @ z)
        h_tilde = np.sqrt(np.abs(np.vdot(z,z) - np.vdot(h,h)))

        #h_tilde = norm(z-Q@h,2)
        # will be empty on first pass, hope that's okay
        h_hat = apply_givens(V,h)

        c, s = givens(np.array([h_hat[-1,-1] , h_tilde]))
        if np.isnan(c) or np.isnan(s):
            raise

        #h_hat[-1,-1] = c*h_hat[-1,-1] - s*h_tilde
        h_hat[-1,-1] = c*h_hat[-1,-1] - s*h_tilde
        
        h_new = np.vstack((h_hat,h_tilde))
        # apply and store givens rotations, etc.
        V = np.vstack((V, np.array([c,s]))) # k x 2

        # form upper triangular matrix R
        R = np.hstack((np.vstack((R,np.zeros((1,R.shape[1])))),h_hat))
        print(R)

        # t grows in size by one. can't figure out how to do this cleanly
        t = np.resize(t, (t.size+1,1))
        # just to be safe--this isn't actually used until it's overwritten
        t[-1,-1] = 0

        # apply c & s to second to last t to get last t
        t[-2:] = V[-1:].T @ t[-2:-1] 
        print(t)
        r = np.abs(t[-1,-1])

        
        # if there will be another iteration
        if r >= tol*β and it < n:
            # compute next Arnoldi vector
            q = z - Q@h
            q = q / norm(q,2) # or w / h_tilde (same number)
    
            # add on additional basis vector
            Q = np.hstack((Q,q))

    y = backsolve(R,t[:-1])
    #y = np.linalg.solve(R,t[:-1])
    # form approximation x
    x = Q@y
    
    return x, it, locals()

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

def make_preconditioner(n,alpha):

    W = -1*np.eye(n,k=-1) - np.eye(n,k=1)
    W[0,-1] = -1
    W[-1,0] = -1

    return .5*np.eye(n) + 0.25*alpha*W

if __name__ == "__main__":
    
    n = 100
    A = p1_system(n)
    b = np.ones((n,1))
    
    prec = make_preconditioner(n, .99)
    #prec = np.eye(n)

    x, endit,loc = pcgmres(A,b,tol=10e-12, M_inv=prec)

    real_solution = np.linalg.solve(A,b)

    

