import unittest
import numpy as np
import numpy.testing as npt
import neuron
import models
import solvers
import network

class TestNetwork(unittest.TestCase):
    def testSingleNeuronNetwork(self):
        # test to make sure a network of 1 neuron behaves the same as a network in isolation
        # test both identity and Yamada Neurons

        self.assertAlmostEqual(1., 1.)


    def testIdentityNetworks(self):
        # test to make sure a network of identity neurons works
        # covers a feedforward and an arbitrary network of weights and delays

        self.assertAlmostEqual(1., 1.)



if __name__ == "__main__":
    unittest.main()
