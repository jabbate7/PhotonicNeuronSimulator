import unittest
import numpy as np
import numpy.testing as npt
from neuron import Neuron
from network import Network

class SimpleTestCase(unittest.TestCase):
    def setUp(self):
        # List of Neuron objects
        neur_1=Neuron()
        neur_2=Neuron()
        self.neurons=[neur_1,neur_2]
        # Weight matrix
        # Input goes 1, 1 goes to 2, 2 goes to 3, 3 goes to output
        self.weights=[[1,0,0],[0,1,0]]
        
    def testSingleNeuronNetwork(self):
        # test to make sure a network of 1 neuron behaves the same as a network in isolation
        # test both identity and Yamada Neurons
        
        self.assertAlmostEqual(1., 1.)


    def testIdentityNetwork(self):
        # test to make sure a network of identity neurons works
        # covers a feedforward and an arbitrary network of weights and delays

        net = Network(self.neurons, self.weights)
        
        input_1=1
        output = net.network_step(1)
        self.assertAlmostEqual(output[0], input_1) # the input
        self.assertAlmostEqual(output[1][0], input_1) # neuron 1 output
        self.assertAlmostEqual(output[1][1], 0.) # neuron 2 output

        input_2=2
        output = net.network_step(2)
        self.assertAlmostEqual(output[0], input_2) # the input
        self.assertAlmostEqual(output[1][0], input_2) # neuron 1 output
        self.assertAlmostEqual(output[1][1], input_1) # neuron 2 output

    def testDelayedIdentityNetwork(self):
        delays = [[0,0],[1,0]]
        net = Network(self.neurons, self.weights, delays=delays)
        
        input_1=1
        net.network_step(input_1)
        input_2=2
        output=net.network_step(input_2)
        
        self.assertAlmostEqual(output[0], input_2) # the input
        self.assertAlmostEqual(output[1][0], input_2) # neuron 1 output
        self.assertAlmostEqual(output[1][1], 0.0) # neuron 2 output

        input_3=3
        output=net.network_step(input_3)
        self.assertAlmostEqual(output[0], input_3)
        self.assertAlmostEqual(output[1][0], input_3)
        self.assertAlmostEqual(output[1][1], input_1)

if __name__ == "__main__":
    unittest.main()
