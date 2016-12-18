#!/usr/bin/env python3

import numpy as np
from scipy.linalg import norm, solve

from functools import partial

from mgcg import *

def get_minv(M,n_levels, interpolation_method):
    """
    Same code as for mgcg, but simply returns the preconditioner.
    This is very redundant and wasteful. hail satan.
    """
    
    # build *all* interpolation matrices and store
    I = tuple((interpolation_method(M,n_levels,el)
            for el in range(n_levels,1,-1)))

    a = make_system(M)

    A_levels = list() # store all systems
    for interp in I:
        A_levels.append(a) # append last system matrix

        # yer done if interp is 0x0 (only an issue if L is larger than M)
        if not interp.size:
            break

        a = interp.T @ (a @ interp)

    A_1 = a # now base_case
    A_levels = tuple(A_levels) # make static

    # you may check np.allclose(Lchol, linalg.cholesky(A_1,lower=True))
    Lchol, it_chol = cholesky(A_1)

    Minv = lambda b: vcycle(n_levels,b,np.zeros_like(b),A_levels,I,Lchol)    

    return Minv

if __name__ == "__main__":

    import matplotlib.pyplot as plt

    M = 11
    L = 3
    L2 = 6

    bee = make_initial_conditions(M)

    m1 = get_minv(M,L, interpolation_matrix)
    m2 = get_minv(M,L, interpolation_matrix_2)
    
    m1a = get_minv(M,L2, interpolation_matrix)
    m1b = get_minv(M,L2, interpolation_matrix_2)
    sol, _,_,_ = mgcg(make_system(M), bee, n_levels=L2,
            interpolation_method=None)
    ax = plt.subplot(111)

    plt.set_cmap('Accent')
    plt.plot(np.arange(bee.size), bee, label='initial conditions', linewidth=2)
    plt.plot(np.arange(bee.size), m1(bee), '-', label='SC 1 (L=3)', linewidth=2)
    plt.plot(np.arange(bee.size), m2(bee), '-', label='SC 2 (L=3)', linewidth=2)
    plt.plot(np.arange(bee.size), m1(bee), '.', label='SC 1 (L=6)', linewidth=2)
    plt.plot(np.arange(bee.size), m2(bee), '.', label='SC 2 (L=6)', linewidth=2)
    plt.plot(np.arange(bee.size), sol, '-', label='solution (CG)', linewidth=2)

    plt.show()
