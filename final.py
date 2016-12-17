#!/usr/bin/env python3

import numpy as np
from scipy.linalg import norm
from hw4 import cholesky
from itertools import count

def make_system(M):
    """
    form the (2^M) x (2^M) system described in (1) with entries

    A_{ij} =    3   if j = i
                -1  if j = i+1
                -1  if j = i-1
                -1  if j = N - i + 1, i != N/2, i != N/2 + 1

    where N = 2^N

    That is A has tridiagonal bands of (-1,3,1) and an
    antidiagonal band of -1.
    
    Note:   In the context of the problem, this function is used to
            create the top level (finest) system, A_{L}

    INPUT:
        M - a positive integer denoting the size of the system (log base 2)
            i.e. the returned system will be (2^M, 2^M)
            
    OUTPUT:
        A - a dense nd.array of shape (2^M, 2^M)

    TODO:
    *   consider allowing this to deal with sparse matrices
        note: see p1 system for hw4 for sparse implementation
    """

    N = 2**M
    
    # indicate positions of all nonzero elements except the main diagonal
    A = np.flipud(np.eye(N)) #antidiagonal
    A += np.eye(N, k=-1) + np.eye(N, k=1) # subdiag & superdiag

    # overwrite these with -1 as desired
    A[A.nonzero()] = -1

    # calculate indices of main diagonal and *overwrite* them with 3
    A[np.diag_indices(N)] = 3

    return A

def pcg(A,b, Minv, tol=1e-8, x_init=None, return_iterations=False,
        return_error=False, verbose=False):
    """
    preconditioned conjugate gradient method
    solves Ax = b by preconditioning

    INPUT:

    A       - an NxN nd.array describing the system
    b       - initial conditions (can be 1D-array or 2D row vector)
    Minv    - preconditioner, which should be a function handle
    tol     - stopping tolerance (returns sol if ||A*sol - b}||_2 < tol ) 
              (optional) default is 1e-8
    x_init  - initial guess (default is None, in which case the zero vector
                is used)

    return_iterations   (optional) return iteration count (default False)
    return_error        (optional) return error (will be below tolerance
                        if converged)

    OUTPUT:
    x           - solution (an Nx1 nd.array)
    iterations  -(if return_iterations=True above) iterations to run
    err         -(||A*sol-b|| of solution calculated error of residual
    """

    # make sure initial guess is a column vector ala matlab
    if b.ndim == 1:
        b = np.expand_dims(b,-1)
    
    if x_init is None:
        x = np.zeros_like(b) # default to zero vector as initial guess
    else:
        x = x_init

    tol *= norm(b)  # for stopping check (save some divisions)

    r = b - A@x     # initial residual
    z = Minv(r)     # residual of preconditioned system
    p = z.copy()    # initial search direction
    d = A@p         # initial A-projected search direction


    for iterations in count(1):

        alpha = np.vdot(r,z) / np.vdot(p,d)

        x += alpha*p
        r_new = r - alpha*d
        
        # equivalent to norm(b - A@x) / norm(b)
        err = norm(r_new)

        if verbose:
            print(iterations, err, sep='\t| ')

        if err <= tol:
            break
        
        z_new = Minv(r_new)
        beta = np.vdot(z_new,r_new) / np.vdot(r,z)

        p = z_new + beta*p

        d = A@p
    
        r = r_new
        z = z_new


        #if err <= tol:
        #    break
    
    # return statement boogaloo
    if return_iterations:

        if return_error:
            return x, iterations, err / norm(b)
        else:
            return x, iterations

    elif return_error:
        return x, err / norm(b)

    else:
        return x

def interpolation_matrix(M,L,el):
    """
    The interpolation matrix I_el between levels el and (el-1)
    INPUT:
    M -     refers to size of system 2^M x 2^M 
    L -     number of levels/grids
    el-     interpolate between el and el-1 

    OUTPUT:
    I_el    a 2^{ M + el - L } by 2^{M + el - (L + 1) }
            interpolation matrix
    """

    n = 2**(M + el - (L+1))

    # identity matrix but repeat each row twice
    # (equivalent to given system)
    return np.repeat(np.eye(n), 2, axis=0)

def interpolation_matrix_2(M,L,el):
    """
    The interpolation matrix I_el between levels el and (el-1)
    for example if M = 2, this function would return
         array([[ 1.,  0.,  0.,  0.],
                [ 0.,  1.,  0.,  0.],
                [ 0.,  0.,  1.,  0.],
                [ 0.,  0.,  0.,  1.],
                [ 0.,  0.,  0.,  1.],
                [ 0.,  0.,  1.,  0.],
                [ 0.,  1.,  0.,  0.],
                [ 1.,  0.,  0.,  0.]])

    for the given question, this is only to be used for the
    (L-1) to (L-2)th level
    """
    if el != L-1:
        return interpolation_matrix(M,L,el)
    else:
        n = 2**(M-2)
        return np.concatenate((np.eye(n),np.flipud(np.eye(n))))

def smooth(A, omega, nu, b, x0, tol=None):
    """
    smoothing function via ω-weighted Jacobi iteration
    this is also a standard iterative method on a
    diagonally dominant system



    AA = np.array([[10., -1., 2., 0.],
                   [-1., 11., -1., 3.],
                   [2., -1., 10., -1.],
                   [0.0, 3., -1., 8.]])

    bb = np.array([6., 25., -11., 15.])
    x00 = np.zeros_like(bb)
    ans = smooth(AA,1., 50, bb, x00)

    converges after 24 iterations
    
    if omega is changed to 2/3 in the above, converges in 35 iterations
    """
    if x0.ndim == 1:
        x0 = np.expand_dims(x0,-1)
    if b.ndim == 1:
        b = np.expand_dims(b,-1)
    
    x = x0.copy()
    D = np.diag(A) # diagonal of system (as a Nx1 vector)
    # must be same shape as b or will broadcast to a matrix under division
    D = D.reshape(x.shape)

    W = np.tril(A, k=-1) + np.triu(A,k=1) #deleted diagonal

    for i in range(nu):
        x = (1-omega)*x + ((omega*(b- (W@x))) / D)
        if tol is not None and np.allclose(b,A@x, 1e-12):
            break
    else:
        if tol is not None:
            print("Warning, did not converge within tolerance", tol)

    return x

def vcycle(l,b,e0, A, I, Lchol):
    
    omega = 2/3
    nu1 = 1

    # base case
    if l == 1:
        e_base = linalg.solve(Lchol.T,linalg.solve(Lchol,b))
        return e_base
    else:
        a = A[-(l-1)]
        i = I[-(l-1)]
        e = smooth(a, omega, nu1, b, e0)

        # compute and restrict error
        res = i.T @ (b - a@e)

        # correct error
        e = e + i @ vcycle(l-1,res, np.zeros_like(res), A,I,Lchol)
         
        # smooth nu1 times on a x = b with initial guess e
        e = smooth(a,omega,nu1,b,e)

    return e

class MGCG():
    """
    Multigrid preconditioned Conjugate Gradient Method
    
    This solves the system Ax=b.
    """
    pass

if __name__ == "__main__":

    # for cheating :-)
    from scipy import linalg
    import matplotlib.pyplot as plt
    from sys

    M = 13
    max_level = 6
    interpolation_method = interpolation_matrix_2
    N = 2**M

    A_L = make_system(M)
    # build *all* interpolation matrices and store
    I = tuple((interpolation_method(M,max_level,l)
            for l in range(max_level,1,-1)))

    a = A_L
    A = list()
    for interp in I:
        A.append(a) #append last system matrix

        # only an issue if L is larger than M
        if not interp.size:
            break
        
        a = interp.T @ (a @ interp)
    # now base_case
    A_1 = a.copy()
    A = tuple(A)
    # you may check np.allclose(Lchol, linalg.cholesky(A_1,lower=True))
    Lchol, _ = cholesky(A_1)
    
    Minv = lambda b: vcycle(max_level,b,np.zeros_like(b),A,I,Lchol)    
    # the following is to be used for unpreconditioned CG
    #Minv = lambda b: b

    # normalized one vector
    bee = np.ones((N,1)) / 2**(M/2)
    
    pcg_sol, its = pcg(A_L,bee,Minv, return_iterations=True, verbose=False)

    ### this shows effect of preconditioning, looks like two arches
    plt.scatter(np.arange(pcg_sol.size), pcg_sol)
    #plt.show()
