import numpy as np
from numpy import linalg as LA

def harmonic_pot(omega,mass,x,x0):
    """
    harmonic oscillator potential
    """
    k=mass * (omega**2)
    return 0.5e+0 * k * (x - x0)**2

def harmonic_eigenval(omega,n):
    """
    eigenvalues of a harmonic oscillator
    """
    return omega * (n + 0.5e+0)

#---------------------------------------------------------------------------------

def kinetic_op_1d(m,deltax,n):
    """
    4th order centered difference scheme for the kinetic energy operator in 1D
    f"(x0) = (-f(-2) +16f(-1) -30f(0) +16f(+1) -f(+2))/12Dx^2
    """
    T = np.zeros((n,n),dtype=float)
    
    #--- upper edge
    i=0
    T[i,i]   = -30.0e+0 
    T[i,i+1] = +16.0e+0
    T[i,i+2] = -1.00e+0

    i=1
    T[i,1-i] = +16.0e+0
    T[i,i]   = -30.0e+0 
    T[i,i+1] = +16.0e+0
    T[i,i+2] = -1.00e+0

    #--- matrix bulk
    for i in range(2,(n-2)):
        T[i,i-2] = -1.00e+0
        T[i,i-1] = +16.0e+0
        T[i,i]   = -30.0e+0 
        T[i,i+1] = +16.0e+0
        T[i,i+2] = -1.00e+0
    
    #--- lower edge
    i=n-2
    T[i,i-2] = -1.00e+0
    T[i,i-1] = +16.0e+0
    T[i,i]   = -30.0e+0 
    T[i,i+1] = +16.0e+0

    i=n-1
    T[i,i-2] = -1.00e+0
    T[i,i-1] = +16.0e+0
    T[i,i]   = -30.0e+0 
    
    #---------------------------
    T=T/(-12.0e+0 * 2.0e+0 * m * (deltax**2))

    #----------------------------
    return T

#------------------------------------------------------------

def hamiltonian_1d(T,V):
    """
    Generates 1D Hamiltonian given the kinetic energy operator T and the diagonal potential operator V
    """
    H=np.empty_like(T)
    H[:,:] = T
    for i in range(V.size):
        H[i,i] = H[i,i] + V[i]
    return H

#------------------------------------------------------------


def solve_eigenstates(H):
    """
    performs diagonalization of Hamiltonian
    """
    eigen_val,eigen_vec=np.linalg.eigh(H)
    return eigen_val,eigen_vec


#------------------------------------------------------------

def hamiltonian_diag_1d(mu,V,deltax):
    """
     Diagonalizes a 1D hamiltonian with 4th order centered differenece kinetic energy operator
    
     mu - mass in a.u.
     V - potential in a.u.
     deltax - discretization step to be used in the kinetic energy operator
    """


    #constructs kinetic energy operator
    T=kinetic_op_1d(mu,deltax,V.size)
    #constructs hamiltonian
    H=hamiltonian_1d(T,V)
    #performs diagonalization
    eigen_val,eigen_vec=solve_eigenstates(H)

    return eigen_val,eigen_vec

#---------------------------------------------------------------------------------
