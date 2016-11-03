#test functions.py
#

import numpy as np
import numpy.testing as test
import functions as fun
import matplotlib.pyplot as plt


def test_harmonic_pot():
    V=fun.harmonic_pot(1.0,1.0,1.0,0.0)
    test.assert_equal(V,0.5e+0)

def test_harmonic_eigenval():
    E0=fun.harmonic_eigenval(1.0,0)
    deltaE=fun.harmonic_eigenval(0.45e+0,10)-fun.harmonic_eigenval(0.45,9)
    test.assert_almost_equal(E0,0.5e+0,10)
    test.assert_almost_equal(deltaE,0.45e+0,10)

def test_solve_eigenstates():
    H=np.array([[0, 1], [1, 0]])
    Eval_ref=np.array([-1,1])
    Evect_ref=np.array([[-0.70710678,0.70710678],[0.70710678,0.70710678]])

    Eval,Evect=fun.solve_eigenstates(H)
    
    test.assert_array_almost_equal(Eval,Eval_ref,8)
    test.assert_array_almost_equal(Evect,Evect_ref,8)

def test_hamiltonian_diag_1d():
    """
    compares with the analytical harmonic oscillator solution
    """
    n=512
    x=np.linspace(-2.0,2.0,n)
    omega=0.45e0/27.2114e0
    mu=1713.52e0
    deltax=(x[-1] - x[0])/(n - 1)

    Eval_harm=fun.harmonic_eigenval(omega,np.linspace(0,4,5))

    N=np.power(mu*omega/np.pi,0.25e0)
    psi_0=N*np.exp(-(0.5e0 * mu * omega)*x**2)
    psi_1=N*x*np.sqrt(2.0e0 * mu * omega)*np.exp(-(0.5e0 * mu * omega)*x**2)
    
    V_harm=fun.harmonic_pot(omega,mu,x,0.0e0)
    Eval,Evect=fun.hamiltonian_diag_1d(mu,V_harm,deltax)
    
    test.assert_array_almost_equal(Eval[0:5],Eval_harm,6)
    test.assert_array_almost_equal(np.fabs(psi_0/psi_0.max()),np.fabs(Evect[:,0]/Evect[:,0].max()),5)
    test.assert_array_almost_equal(np.fabs(psi_1/psi_1.max()),np.fabs(Evect[:,1]/Evect[:,1].max()),5)

def test_franck_condon():
    x=np.linspace(-2.0,2.0,256)
    omega=0.45e0/27.2114e0
    mu=1713.52e0
    deltax=(x[-1] - x[0])/(256 - 1)
    
    N=np.power(mu*omega/np.pi,0.25e0)*np.sqrt(deltax) #euclidian norm
    psi_0=N*np.exp(-(0.5e0 * mu * omega)*x**2)
    psi_1=N*x*np.sqrt(2.0e0 * mu * omega)*np.exp(-(0.5e0 * mu * omega)*x**2)


    FC00=fun.franck_condon(x,psi_0,psi_0)
    FC01=fun.franck_condon(x,psi_0,psi_1)
    FC11=fun.franck_condon(x,psi_1,psi_1)
    test.assert_almost_equal(FC00,1.0e0)
    test.assert_almost_equal(FC01,0.0e0)
    test.assert_almost_equal(FC11,1.0e0)

    V_harm=fun.harmonic_pot(omega,mu,x,0.0e0)
    Eval,Evect=fun.hamiltonian_diag_1d(mu,V_harm,deltax)
    
    nFC00=fun.franck_condon(x,Evect[:,0],Evect[:,0])
    nFC01=fun.franck_condon(x,Evect[:,0],Evect[:,1])
    nFC11=fun.franck_condon(x,Evect[:,1],Evect[:,1])
    nFC12=fun.franck_condon(x,Evect[:,1],Evect[:,2])
    test.assert_almost_equal(nFC00,1.0e0)
    test.assert_almost_equal(nFC01,0.0e0)
    test.assert_almost_equal(nFC11,1.0e0)
    test.assert_almost_equal(nFC12,0.0e0)

#------------------------------------------------------------

def test_all_franck_condon():
    x=np.linspace(-2.0,2.0,256)
    omega=0.45e0/27.2114e0
    mu=1713.52e0
    deltax=(x[-1] - x[0])/(256 - 1)

    V_harm=fun.harmonic_pot(omega,mu,x,0.0e0)
    Eval,Evect=fun.hamiltonian_diag_1d(mu,V_harm,deltax)

    FCij=fun.all_franck_condon(x,Evect[:,0:5],Evect[:,0:3])
    FC_check=np.zeros((5,3),dtype=float)
    for i in range(FC_check.shape[0]):
        if(i<FC_check.shape[1]): FC_check[i,i] = 1.0e0
            
    assert FCij.shape == (5,3)
    test.assert_array_almost_equal(FC_check,FCij,6)

    FC_check=np.zeros((5,3),dtype=float)
    FC_check[0,1]=np.sqrt(1.0/(2.0*mu*omega))
    for i in range(1,FC_check.shape[0]):
        if(i<FC_check.shape[1]-1): 
            FC_check[i,i+1] = np.sqrt((i+1)/(2.0*mu*omega))
            FC_check[i-1,i] = np.sqrt((i)/(2.0*mu*omega))
        if(i<FC_check.shape[1]): 
            FC_check[i,i-1] = np.sqrt(i/(2.0*mu*omega))
            FC_check[i+1,i] = np.sqrt((i+1)/(2.0*mu*omega))

    FCij=fun.all_franck_condon(x,Evect[:,0:5],Evect[:,0:3],dipole=x)
    test.assert_array_almost_equal(np.abs(FC_check),np.abs(FCij),5)

#----------------------------------------------------------------------

def test_xas_cross_section():
    x=np.linspace(-2.0,2.0,256)
    omega=0.45e0/27.2114e0
    mu=1713.52e0
    deltax=(x[-1] - x[0])/(256 - 1)

    V_g=fun.harmonic_pot(omega,mu,x,0.0e0)
    eg,psi_g=fun.hamiltonian_diag_1d(mu,V_g,deltax)

    V_c=fun.harmonic_pot(omega,mu,x,0.3e0)
    ec,psi_c=fun.hamiltonian_diag_1d(mu,V_c,deltax)

    fc_gc=fun.all_franck_condon(x,psi_g[:,0:6],psi_c[:,0:6])
    
    sig_xas=fun.xas_cross_section(500.0,fc_gc[0,:],300.0/27.2114,eg[0],ec[0:6],0.08/27.2114)
    test.assert_almost_equal(sig_xas,0.0)
    

    om=np.linspace(295.0/27.2114,305.0/27.2114,512)
    sig_xas=fun.xas_cross_section(om,fc_gc[0,:],300.0/27.2114,eg[0],ec[0:6],0.08/27.2114)
    #plt.plot(om*27.2114,sig_xas,'-')
    #plt.show()

    assert sig_xas.size == om.size

#----------------------------------------------------------------------

def test_rixs_cross_section():
    x=np.linspace(-2.0,2.0,256)
    omega=0.45e0/27.2114e0
    mu=1713.52e0
    deltax=(x[-1] - x[0])/(256 - 1)

    V_g=fun.harmonic_pot(omega,mu,x,0.0e0)
    eg,psi_g=fun.hamiltonian_diag_1d(mu,V_g,deltax)

    V_c=fun.harmonic_pot(omega,mu,x,0.3e0)
    ec,psi_c=fun.hamiltonian_diag_1d(mu,V_c,deltax)

    fc_gc=fun.all_franck_condon(x,psi_g[:,0:6],psi_c[:,0:3])
    fc_cg=fun.all_franck_condon(x,psi_c[:,0:3],psi_g[:,0:6])
    
    om_in=300.0/27.2114
    om_out=np.linspace(295.0/27.2114,300.5/27.2114,512)
    sig_rixs=fun.rixs_cross_section(om_in,om_out,fc_gc[0,:],fc_cg[:,:],300.0/27.2114,0.0,eg[0],ec[0:6],eg[0:6],0.08/27.2114,7e-4)
    #plt.plot((om_in - om_out)*27.2114,sig_rixs,'-')
    #plt.show()

    assert sig_rixs.size == om_out.size
    

def test_xas_cross_section2():
    x=np.linspace(-2.0,2.0,256)
    omega=0.45e0/27.2114e0
    mu=1713.52e0
    deltax=(x[-1] - x[0])/(256 - 1)

    V_g=fun.harmonic_pot(omega,mu,x,0.0e0)
    V_c=fun.harmonic_pot(omega,mu,x,0.3e0)+ 300.0/27.2114
    
    om,sig_xas=fun.compute_xas(mu,x,V_g,V_c,0.08/27.2114)
    plt.plot(om*27.2114,sig_xas,'-')
    plt.show()

    assert sig_xas.size == om.size

def test_rixs_cross_section2():
    x=np.linspace(-2.0,2.0,256)
    omega=0.45e0/27.2114e0
    mu=1713.52e0
    deltax=(x[-1] - x[0])/(256 - 1)

    V_g=fun.harmonic_pot(omega,mu,x,0.0e0)
    V_c=fun.harmonic_pot(omega,mu,x,0.3e0) + 300.0/27.2114
    
    om_in=300.0/27.2114
    om_out,sig_rixs=fun.compute_rixs(om_in,mu,x,V_g,V_c,V_g,0.08/27.2114)
    plt.plot((om_in - om_out)*27.2114,sig_rixs,'-')
    plt.show()

    assert sig_rixs.size == om_out.size
