#test functions.py
#

import numpy as np
import numpy.testing as test
import functions as fun

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
    
    x=np.linspace(-2.0,2.0,256)
    omega=0.45e0/27.2114e0
    mu=1713.52e0
    deltax=(x[-1] - x[0])/(256 - 1)

    Eval_harm=fun.harmonic_eigenval(omega,np.linspace(0,4,5))
    
    V_harm=fun.harmonic_pot(omega,mu,x,0.0e0)
    Eval,Evect=fun.hamiltonian_diag_1d(mu,V_harm,deltax)
    
    test.assert_array_almost_equal(Eval[0:5],Eval_harm,6)
