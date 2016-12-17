#!/usr/bin/env python3
import numpy as np
def demo_system(n):
    
    A = np.eye(n) - 2*np.eye(n,k=-1)
    b = np.ones(n) + np.sqrt(6)*(2**-52)

    return A, b


