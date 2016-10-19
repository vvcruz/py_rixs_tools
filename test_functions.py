#test functions.py
#

import unittest
import functions as fun

class TestFunctions(unittest.TestCase):

    def testharmonic_pot(self):
        V=fun.harmonic_pot(1.0,1.0,0.0)
        self.assertEqual(V,0.5)

    
if __name__=="__main__":
    unittest.main()
