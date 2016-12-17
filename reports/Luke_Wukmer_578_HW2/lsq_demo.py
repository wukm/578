#!/usr/bin/env python3
import numpy as np
from numpy.linalg import norm

def demo_system(n):
    """
    as described in problem 2(b)
    """
    A = np.eye(n) - 2*np.eye(n,k=1)
    b = np.ones(n) + np.sqrt(6)*(2**-52)

    b = np.expand_dims(b,-1)
    return A, b

def back_substitution(R,b):
    """
    Solves a system Rx=b via backsubstitution where R is an nxn upper-triangular
    matrix
    """
    n = R.shape[0]
    
    x = np.zeros(b.shape)
    
    x[-1] = b[-1] / R[-1,-1] # last entry to initialize
    for i in range(n-1,-1,-1):
        x[i] = (b[i] - R[i,i+1:]@x[i+1:]) /R[i,i]

    return x

def relative_residual(A,x,b):

    return norm(A@x-b,2) / norm(b,2)

if __name__ == "__main__":

    from qr import householder_qr
    from givens import givens_qr
    from numpy.linalg import qr
    
    import matplotlib.pyplot as plt

    print('n', 'e_n (backsub)', 'f_n (normal eqs)', sep='\t')
    print('-'*80)
    
    Ns = np.arange(5,26)

    errs = list()

    for n in Ns:

        A,b = demo_system(n)
        
        # backsub only -> e_n
        #Q,R = givens_qr(A)
        x = back_substitution(A,b)
        e = relative_residual(A,x,b)
        
        # normal_eqs -> f_n
        N = A.T @ A
        c = A.T @ b
        Q,R = givens_qr(N)
        #Q, R = qr(N)
        x = back_substitution(R,Q.T@c)
        f = relative_residual(A,x,b)
        

        #   # builtin -> g_n
        #   Q,R = qr(A)
        #   x = back_substitution(R,Q.T@b)
        #   g = relative_residual(A,x,b)
        #   
        #   
        #   # householder -> h_n
        #   
        #   Q,R = householder_qr(A) 
        #   x = back_substitution(R,Q.T@b)
        #   h = relative_residual(A,x,b)
        #   
        #   print(n, e, f, g, h, sep='\t')
    
        print(n,e,f, sep='\t')
        errs.append((e,f))

    C = np.log([ f / e for e,f in errs])
    
    plt.scatter(Ns, C, marker='d')

    m, yint = np.polyfit(Ns,C,1)
    xs = np.linspace(5,25)

    plt.plot(xs,m*xs + yint)
    
    plt.title('relative residual errors in least-squares problems')
    plt.xlabel('n (size of system)')
    plt.minorticks_on()
    plt.ylabel('$\log(f_n/e_n)$')
    plt.axis('equal')
    plt.tight_layout()
    
    print('esimated slope of line:', m)

    plt.show()


