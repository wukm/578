#!/usr/bin/env python3

from mgcg import *
import matplotlib.pyplot as plt
import numpy as np

M = 7
levels = 6
A_L = make_system(M)
b = np.ones((2**M,1))/2**(M/2)


sol, its, As, Is = mgcg(A_L,b,n_levels=levels,
        interpolation_method=interpolation_matrix_2)

fig, ax = plt.subplots(2,2)

ax = ax.ravel()

for i, a in enumerate(ax):

    ax[i].spy(As[i])
    if i > 0:
        ax[i].set_title(r'$A_{\mathscr{l}},\;\mathscr{l}=L - $'+str(i),
                fontsize=20)
    else:
        ax[i].set_title(r'$A_L$', fontsize=20)
    ax[i].axis('off')

fig.tight_layout()
plt.show()

