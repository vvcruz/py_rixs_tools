#test functions.py
#

import numpy as np
import numpy.testing as test
import functions as fun

def test_harmonic_pot():
    V=fun.harmonic_pot(1.0,1.0,0.0)
    test.assert_equal(V,0.5e+0)

def test_solve_eigenstates():
    H=np.array([[0, 1], [1, 0]])
    Eval_ref=np.array([-1,1])
    Evect_ref=np.array([[-0.70710678,0.70710678],[0.70710678,0.70710678]])

    Eval,Evect=fun.solve_eigenstates(H)
    
    test.assert_array_almost_equal(Eval,Eval_ref,6)
    test.assert_array_almost_equal(Evect,Evect_ref,6)

def test_hamiltonian_diag_1d():
    """
    compares with the analytical harmonic oscillator solution
    """
    
    Eval,Evect=hamiltonian_diag_1d(mu,V,deltax)
    
    test.assert_array_almost_equal(Eval,Eval_ref,6)
    test.assert_array_almost_equal(Evect,Evect_ref,6)
