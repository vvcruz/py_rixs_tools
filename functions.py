import numpy as np
from numpy import linalg as LA
from scipy.integrate import simps

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

def franck_condon(x,u,w,dipole=None):
    """
    computes the franck-condon amplitude between real vectors u and w FC = < u | w >
    """
    if(dipole is None):
        dipole = np.ones(x.shape,dtype=float)

    deltax=x[1]-x[0]
    fc=simps((u * dipole * w),x)/deltax
    return fc

#---------------------------------------------------------------------------------

def all_franck_condon(x,u,w,dipole=None):
    """
    computes the franck condon amplitude matrix between two sets of vectors u and w
    """
    n=u.shape[1]
    m=w.shape[1]

    if(dipole == None):
        dipole = np.ones(x.shape,dtype=float)

    fc=np.zeros((n,m),dtype=float)
    for i in range(n):
        for j in range(m):
            fc[i,j]=franck_condon(x,u[:,i],w[:,j],dipole)
    return fc

#---------------------------------------------------------------------------------

def lorentzian(x,y):
    """
    Lorentzian function with witdh y
    """
    return y/(np.pi*(x**2 + y**2))

#---------------------------------------------------------------------------------

def xas_cross_section(omega,fc_gc,omega_gc,e0,ec,Gamma):
    """
    computes the absorption cross section from all the previously computed quantities

    omega : desired incoming photon energies (in a.u.)
    fc_gc : franck-condon amplitude array between electronic states <c|d|g>
    omega_gc : energy difference between the minimum of the |g> and |c> PES (in a.u.)
    e0 : initial state energy (in a.u.)
    ec : array with the vibrational energies of state |c> (in a.u.)
    Gamma : lifetime broadening of state |c>
    """

    
    sig_xas=np.zeros_like(omega)
    
    for i in range(fc_gc.size):
        sig_xas=sig_xas+ (fc_gc[i]**2)*lorentzian(omega -omega_gc-(ec[i]-e0),Gamma)
    
    return sig_xas
    

#---------------------------------------------------------------------------------

def rixs_cross_section(omega_in,omega_out,fc_gc,fc_fc,omega_gc,omega_gf,e0,ec,ef,Gamma,Gammaf):
    """
    computes the absorption cross section

    omega_in  : desired incoming photon energy (in a.u.)
    omega_out : desired outgoing photon energies (in a.u.)
    fc_gc : franck-condon amplitude array between electronic states <c|d|g>
    fc_fc : franck-condon amplitude array between electronic states <c|d|f>
    omega_gc : energy difference between the minimum of the |g> and |c> PES (in a.u.)
    omega_gf : energy difference between the minimum of the |g> and |f> PES (in a.u.)
    e0 : initial state energy (in a.u.)
    ec : array with the vibrational energies of state |c> (in a.u.)
    ef : array with the vibrational energies of state |f> (in a.u.)
    Gamma : lifetime broadening of state |c>
    Gammaf : lifetime broadening of state |f> (default 0.02 eV ~ 7e-4 au
    """

    nc=fc_fc.shape[0]
    nf=fc_fc.shape[1]
    sig_rixs=np.zeros_like(omega_out)
    F_re=np.zeros(nf,dtype=float)
    F_im=np.zeros(nf,dtype=float)


    for k in range(nf):
        for i in range(nc):
            F_wk=np.pi*(fc_gc[i] * fc_fc[i,k])*lorentzian(omega_in -omega_gc-(ec[i]-e0),Gamma)/Gamma
            F_re[k]=F_re[k] +  F_wk * (omega_in - omega_gc + e0 -ec[i])
            F_im[k]=F_im[k] -  F_wk * Gamma
 
        sig_rixs=sig_rixs + (F_re[k]**2 + F_im[k]**2)*lorentzian(omega_in - omega_out - omega_gf -ef[k] + e0,Gammaf)
        
    return sig_rixs

#-------------------------------------------------------------------------------
def compute_xas(mu,x,V_g,V_c,Gamma,omega=None,nc=15):
    """
    computes the absorption cross section

    mu       : mass in a.u.
    x        : coordinate space vector 
    V_g      : PES for state |g>
    V_c      : PES for state |c>
    Gamma    : lifetime broadening of state |c> (in a.u.)
optional:
    omega    : desired incoming photon energies (in a.u.)
    nc       : desired number of vibrational levels considereg for |c>
    """

    omega_gc=V_c.min() - V_g.min()
    if(omega == None):
        omega = np.linspace(omega_gc-0.5,omega_gc + 0.5,1024)

    
    deltax=(x[1] - x[0])
    eg,psi_g=hamiltonian_diag_1d(mu,V_g - V_g.min(),deltax)
    ec,psi_c=hamiltonian_diag_1d(mu,V_c - V_c.min(),deltax)
    fc_gc=all_franck_condon(x,psi_g[:,0:3],psi_c[:,0:nc])
    sig_xas=xas_cross_section(omega,fc_gc[0,:],omega_gc,eg[0],ec[0:nc],Gamma)
    return omega,sig_xas


#-------------------------------------------------------------------------------
def compute_rixs(omega_in,mu,x,V_g,V_c,V_f,Gamma,Gammaf=7e-4,omega_out=None,nc=15,nf=15):
    """
    computes the absorption cross section

    mu       : mass in a.u.
    x        : coordinate space vector 
    V_g      : PES for state |g>
    V_c      : PES for state |c>
    V_f      : PES for state |f>
    omega_gc : energy difference between the minimum of the |g> and |c> PES (in a.u.)
    omega_gf : energy difference between the minimum of the |g> and |f> PES (in a.u.)
    Gamma    : lifetime broadening of state |c> (in a.u.)
optional:
    omega    : desired incoming photon energies (in a.u.)
    nc       : desired number of vibrational levels considereg for |c>
    """

    omega_gc=V_c.min() - V_g.min()
    omega_gf=V_f.min() - V_g.min()
    if(omega_out == None):
        omega_out = np.linspace(omega_in-0.5,omega_in + 0.1,1024)

    deltax=(x[1] - x[0])
    eg,psi_g=hamiltonian_diag_1d(mu,V_g - V_g.min(),deltax)
    ec,psi_c=hamiltonian_diag_1d(mu,V_c - V_c.min(),deltax)
    ef,psi_f=hamiltonian_diag_1d(mu,V_f - V_f.min(),deltax)
    fc_gc=all_franck_condon(x,psi_g[:,0:3],psi_c[:,0:nc])
    fc_fc=all_franck_condon(x,psi_c[:,0:nc],psi_f[:,0:nf])
    
    sig_rixs=rixs_cross_section(omega_in,omega_out,fc_gc[0,:],fc_fc[:,:],omega_gc,omega_gf,eg[0],ec[0:nc],ef[0:nf],Gamma,Gammaf)
    return omega_out,sig_rixs
    

#-----------------------------------------------------------

def dump_spec(f,header,omega,sigma):
    print("#"+header,file=f)
    for i in range(omega.size):
        print(omega[i],sigma[i],file=f)
    return
