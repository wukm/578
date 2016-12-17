#!/usr/bin/env python3

import numpy as np
from numpy.linalg import norm, qr, cond

def mgs_qr(A):
    """
    (Thin) QR Factorization of an mxn matrix A via modified Gram-Schmidt
    
    input
    
    A - an mxn np.ndarray


    output

    Q - a mxn np.ndarray
    R - an nxn np.ndarray
    """

    m,n = A.shape

    assert m >= n

    Q = np.zeros((m,n)) # thin/reduced QR
    R = np.zeros((n,n))

    for j in range(n):
        
        Q[:,j] = A[:,j] # set jth col of Q to that of A

        for i in range(j):
            # classical uses: R[i,j] = Q[:,i] @ A[:,j]
            R[i,j] = Q[:,i] @ Q[:,j]
            Q[:,j] = Q[:,j] - R[i,j] * Q[:,i]

        R[j,j] = norm(Q[:,j]) # what? is this BLAS?
        Q[:,j] = Q[:,j] / R[j,j]

    return Q, R

def householder_qr(A, show_iterates=False):
    """
    (Thin) QR Factorization of an mxn matrix A via Householder
    transformations. Note that this function *does*
    automatically and invariably return Q in explicit form.
    This requires some extra overhead after R has
    been computed. speedier options exist.

    """
    # bad things happen with assignment when A (thus R or V) is of type int
    A = A.astype('float64')

    m,n = A.shape
    assert m >= n
    
    R = A.copy() # not sure what will happen by default
    V = np.zeros((m,n)) # mxn empty matrix to store Householder vectors
    if show_iterates:
        A_k = list()

    for k in range(n):
        x = R[k:,k] # kth (current) column from diagonal to end; will be zeroed
        # otherwise outer product below 
        v = x.copy()
        v = np.expand_dims(v,-1) # make it an honest to god column matrix
        
        v[0] = x[0] + np.sign(x[0])*norm(x) # householder vector

        v = v/norm(v)
        V[k:,k] = v.flatten() # store it so that Q can be generated
        print("my v is: ", v.flatten())
        # apply Q_k := I - 2vv'
        # note implicit transpose of v inside parens (broadcasting)

        # note on use of np.outer:
        # broadcasting rules would cause 2*v@(v.T @ R[k:,k:]) to be a dot
        # product in the case that v is 1 dimensional (i.e. of shape (k,) ).
        # setting v = np.expand_dims(v,1) would fix this as well

        R[k:,k:] = R[k:,k:] - 2*np.outer(v , v.T @ R[k:,k:])
        #R[k:,k:] -= 2*np.outer(v , v.T @ R[k:,k:])

        if show_iterates:
            print('R after {} iterations:'.format(k+1))
            print(R)
            temp = R[k:,k:].copy()
            A_k.append(temp)

    # now explicitly form Q by multiplying together Q_ks in reverse order
    Q = np.eye(m,n) # an mxn identity matrix (rows after nth are all zero)

    for k in reversed(range(n)):
        v = V[k:,k] 
        Q[k:,k:] = Q[k:,k:] - 2*np.outer(v, v @ Q[k:,k:])

    if show_iterates:
        return Q,R,A_k
    else:
        return Q,R


def orthogonality(Q):
    """
    a measure of how orthogonal a matrix is, gauged by
    e_h = || Q'Q - I ||_F
    """
    n = Q.shape[-1] # column size

    eyelike = Q.T @ Q

    return norm(eyelike - np.eye(n), ord='fro') 

def deviation(A, B):
    """
    returns ||A - B ||_F
    """
    return norm(A-B, ord='fro')

def lotkin(n):
    """
    with n>1, the matrix L_n given by

    L_n(i,j) = { 1  if i=1
               { 1 / (i+j-1) otherwise

    In this definition, the matrix is 1-indexed. The 0-indexed version is:

    L_n(i,j) = { 1 if i=0
               { 1 / (i+j+1) otherwise
    """
    
    assert n > 1

    # build denominators
    seeds = [[i+j+1 for i in range(n)] for j in range(n)]

    L = np.array(seeds)
    L[0,:] = 1 # per definition

    return 1/L

# also do measure || A - QR ||_F

if __name__ == "__main__":
    
    import matplotlib.pyplot as plt 

    # to agree with HW pdf question 1a
    np.set_printoptions(precision=2)
    test = np.array([[2,1,0],[1,2,1],[0,1,2]])
    kf = [      
    2.650891787262261 ,
    6.184870438760594 ,
    9.749450746898903 ,
    13.298809766667933,
    16.838678039737061,
    20.373204464310181,
    23.904588988942542,
    27.434028478463464,
    30.962317672268533,
    34.492511119474557,
    38.035973766472360,
    ]
    e_h = list()
    e_mgs = list()
    f_h = list()
    f_mgs = list()
    kf_alt = list()
    Ns = list(range(2,13))
    for N in Ns:
        print('N={}'.format(N), '*'*20)
        L = lotkin(N)
        lkf = np.log(cond(L, p='fro'))
        kf.append(lkf)
        print('log of (frobenius) condition number for L_N:', lkf)
        print('running householder QR.', '.'*20)
        Q, R = householder_qr(L)
        e = np.log(orthogonality(Q))
        f = np.log(deviation(Q@R, L))
        print('e_N = ', e) 
        print('f_N = ', f)
        e_h.append(e)
        f_h.append(f)
        print('running mGS QR.', '.'*20)
        q, r = mgs_qr(L)
        e = np.log(orthogonality(q))
        f = np.log(deviation(q@r, L))
        print('e_N = ', e) 
        print('f_N = ', f)
        e_mgs.append(e)
        f_mgs.append(f)

    # to answer part d questions
    for i, ers in enumerate((e_mgs, e_h)):
        plt.figure(i)
        ax = plt.scatter(kf, ers)
        plt.xlabel('$\log(\kappa_F{(L_n)})$')
        plt.ylabel('$\log(e_n)$')
        plt.title('log-condition number v.s. log-deviation for Lotkin matrices with mGS-QR')

        m, b = np.polyfit(kf, ers, 1)
        plt.plot(m*np.arange(*plt.xlim())+b)

        print('estimated slope:', m)

    # to answer question 3
    q3 = np.eye(10) + np.eye(10,k=1) + np.fliplr(np.eye(10))
    q3[q3.nonzero()] = 1 # get rid of single entry > 1
    q,k,Ak = householder_qr(q3,show_iterates=True)
    B = [(np.count_nonzero(a), (10-k)**2) for k,a in enumerate(Ak)]
