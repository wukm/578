#!/usr/bin/env python3

import numpy as np
import scipy.sparse as sparse
from numpy.linalg import norm

from itertools import count
from functools import partial

def build_p1(n):
    """
    n is size of matrix

    A_{i,j} = 3     if j = i
            = -1    if j = i + 1
            = -1    if j = i - 1
            = -1    if j = n - i + 1
    """
    
    # takes care of first three cases. make as a linked-list
    # so we can set the antidiagonal by indexing
    A = sparse.diags([-1,3,-1], offsets=(-1,0,1),
            shape=(n,n),format='lil')

    # antidiagonal indices
    adi = (np.ogrid[0:n],np.ogrid[n-1:-1:-1])

    A[adi] = -1 # this sets the antidiagonnal

    # convert to a more reasonable data-type
    return A.tocsc()
    
def cholesky(A):
    """
    computes the cholesky decomposition for symmetric, positive definite
    matrices. returns a lower-triangular matril L with positive diagonal
    entries so that A=LL^T. also returns an integer nzl that gives the number of
    nonzero entries in the Cholesky factor.

    INPUT:

    A - a positive definite matrix nxn

    OUTPUT:

    L - the cholesky factor L s.t. A = LL^T
    nzl - number of nonzero entries in L i.e. where |L_{ij}| > 0

    """
    
    if sparse.issparse(A):
        G = sparse.tril(A)
        G = G.tocsc() # a sparse matrix that still allows elementwise access
    else:
        G = np.tril(A)
    
    n = A.shape[0]

    for k in range(n):
        G[k:,k] -= G[k:,:k] @ G[k,:k].T
        G[k:,k] /= np.sqrt(G[k,k])
    
    if sparse.issparse(G):
        nzl = G.count_nonzero()
    else:
        nzl = np.count_nonzero(G)

    return G, nzl

def multiply(x, diagonal=None):
    """
    implicitly performs the matrix multiplication Ax
    for the matrix given in (i)

    # test multiply function
    
    n = 60
    x = np.expand_dims(np.random.randn(n),-1)
    A = build_p1(n)

    np.all(np.isclose(A@x,multiply(x))) -> True
    """ 
    if diagonal is None:

        d = 3
    else:
        d = diagonal

    y = np.empty_like(x)
    n = x.size - 1
    n2 = x.size // 2 - 1
    
    # just overwrite the different entries later (ignore 0,n)
    for i in range(1,n):
        y[i] = d*x[i] - x[i+1] - x[i-1] - x[n-i]

    y[0] = d*x[0] - x[1] - x[-1] # first entry
    y[-1] = d*x[-1] - x[-2] - x[0]  # nth (last) entry

    # then overwrite the following:
    y[n2] = d*x[n2] - x[n2+1] - x[n2-1]
    y[n2+1] = d*x[n2+1] - x[n2+2] - x[n2]

    return y 

def multiply2(x):
    """
    multiply implicitly by A = I + BB.T
    where B = np.tril(np.ones((n,3))) and n = x.size
    i.e. B = array([[ 1,  0,  0],
                    [ 1,  1,  0],
                    [ 1,  1,  1],
                    [ .   .   .],
                    [ .   .   .],
                    [ .   .   .],
                    [ 1,  1,  1]])

    this faster method is shown by first considering (B.T @ x) itself,
    which yields the 3-vector:
    [ S , S -x[0] , S - x[0] - x[1] ], where S = x.sum()
    """

    S = x.sum()
    
    # do BB.T mult first
    BBx = np.zeros_like(x)
    BBx[0] = S
    BBx[1] = 2*S - x[0]
    BBx[2:] = 3*S - 2*x[0] - x[1]
    
    # now add I part and return
    return x + BBx

def cg(b=None, n=None, mult=None, tol=1e-6):
    """
    non-preconditioned CG
    initial estimate is b (defaults to normalized ones vector)
    size of system n (can be inferred from b or vice versa)     

    using multiplication method mult
    """
    # infer b or n as needed 

    assert mult is not None

    if b is None:
        try:
            # normalize vector of ones
            b = np.ones((n,1)) / np.sqrt(n)
        except NameError:
            raise Exception('must specify system size or initial guess')
    else:
        n = b.size

    # make sure initial guess is a column vector ala matlab
    if b.ndim == 1:
        b = np.expand_dims(b,-1)
    
    # not sure if explicit copy is needed
    x = np.zeros_like(b)
    r = b.copy()
    p = b.copy()
    
    d = mult(p)

    alpha = np.vdot(r,r) / np.vdot(p,d)

    for iterations in count(1):

        x += alpha*p
        r_new = r - alpha*d

        beta = np.vdot(r_new,r_new) / np.vdot(r,r)

        p = r_new + beta*p

        d = mult(p)
    
        alpha = np.vdot(r_new, r_new) / np.vdot(p,d)
        
        r = r_new

        err = norm(mult(x) - b) / norm(b) 

        #print(iterations, err, sep='\t| ')
        if err <= tol:
            break

    return x, iterations, err

if __name__ == "__main__":
    
    import matplotlib.pyplot as plt 

    ## PART A ################################################################
    print("part a", "_"*30)
    ## test cholesky
    test_part_a = cholesky(np.array([[2,1,0],[1,2,1],[0,1,2]],dtype='f'))
    print(test_part_a)
        

    ## PART 1B ########################################################
    print("part b", "_"*30)
    #ns = (10,20,100,200,1000,2000)
    #ns = (10,20,30,40,50)
    #nz = list()

    #for n in ns: 

    #    A = build_p1(n)
    #    L, nzl = cholesky(A) 
    #    nz.append(nzl)

    #    print(n, nzl)

    #coefs = np.polyfit(ns,nz, 2) # coefficients for  nzl ~ an^2 + bn + c
    #
    #plt.figure(1)
    #plt.scatter(ns,nz)
    #ngrid = np.linspace(0,max(ns))
    #plt.plot(ngrid, [coefs[0]*n**2 + coefs[1]*n + coefs[2] for n in ngrid])

    #plt.title(r'$nzl \;\propto\; an^2 + bn + c  ,\quad a \approx 0.25$') 

    #plt.figure(2)
    #plt.spy(L)
    
    ## PART 1C ########################################################
    print("part c", "_"*30)
    #np.roots([coefs[0],coefs[1],coefs[2]-10**9])
   
    ## PART 1D ########################################################
    print("part d", "_"*30)
    ## for part e-f uncomment one of the following
    ##multmethod = multiply
    #multmethod = partial(multiply, diagonal=3.0001) 

    ## PART 1E-F ######################################################
    print("part e/f", "_"*30)
    #Ns = (10,50,100,5000,15000,30000)
    #its = list()
    #errs = list()

    #for n in Ns:
    #    x, iterations, err = cg(n=n, mult=multmethod)
    #    its.append(iterations)
    #    errs.append(err)
    #    print(n, iterations, err)
 

    ##sample output part (e)
    ##10 3 1.13492816813e-15
    ##50 13 7.26595797191e-12
    ##100 26 2.12606603652e-08
    ##5000 1544 6.25257025384e-07
    ##15000 4616 7.0254248905e-07
    ##30000 9216 9.29915260829e-07
    
    #plt.scatter(Ns, its)
    
    ##sample output part (f)
    ##10 3 1.88574575416e-15
    ##50 13 3.00488927641e-12
    ##100 26 9.88035379143e-07
    ##5000 1519 8.48965754023e-07
    ##15000 1835 9.97779149178e-07
    ##30000 1797 9.90061636977e-07
    
    ##PART 1G ###########################################################
    print("part g", "_"*30)
    tol = 1e-12

    for n in [10000, 50000, 100000]:
        b = np.ones([n,1]) / np.sqrt(n)
        x, iterations, err = cg(b=b, mult=multiply2, tol=tol)
        print(n, iterations, err)

    
    ##sample output of part (g)
    ##10000 4 2.06562011245e-15
    ##50000 4 1.82194210335e-14
    ##100000 4 1.15615944448e-13
