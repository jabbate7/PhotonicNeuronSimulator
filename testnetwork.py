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
        # Input goes 1, 1 goes to 2, 2 goes to output
        self.weights=np.array([[1,0,0],[0,1,0]])

    def testSetup(self):
        #ensure when create a network, updates neurons properties accordingly
        old_dt=self.neurons[0].dt
        dt=10*old_dt #use different timestep than neuron
        neur1_hist_length=12
        delays = np.array([[0,0],[neur1_hist_length*dt,0]]) #need longer history than default (10)
        net=Network(self.neurons, self.weights, delays=delays, dt=dt)
        self.assertAlmostEqual(dt, self.neurons[0].dt) #check if dts are updates
        self.assertAlmostEqual(dt, self.neurons[1].dt)
        self.assertEqual(neur1_hist_length+1, self.neurons[0].hist_len) #should store 13 elements
        self.assertEqual(1, self.neurons[1].hist_len) #should only store 1

        num_inputs=self.weights.shape[1]-len(self.neurons)
        self.assertEqual(num_inputs, net.num_inputs)
        self.assertEqual(len(self.neurons), net.num_neurons)

        #the delay matrix in net should be an array of the number of timesteps to step back
        expected_delay_mat=np.array([[0, 0], [neur1_hist_length, 0]]) 
        npt.assert_array_equal(expected_delay_mat, net.delays)


    def testIdentityNetwork(self):
        # test to make sure a network of identity neurons works
        # covers a feedforward and an arbitrary network of weights and delays

        net = Network(self.neurons, self.weights)
        
        input_1=1
        output = net.network_step(input_1)
        self.assertAlmostEqual(output[0], input_1) # neuron 1 output
        self.assertAlmostEqual(output[1], 0.) # neuron 2 output

        input_2=2
        output = net.network_step(input_2)
        self.assertAlmostEqual(output[0], input_2) # neuron 1 output
        self.assertAlmostEqual(output[1], input_1) # neuron 2 output


    def testDelayedIdentityNetwork(self):
        dt=1
        delays = np.array([[0,0],[dt,0]])
        net = Network(self.neurons, self.weights, delays=delays, dt=dt)
        
        input_1=1
        net.network_step(input_1)
        input_2=2
        output=net.network_step(input_2)
        
        self.assertAlmostEqual(output[0], input_2) # neuron 1 output
        self.assertAlmostEqual(output[1], 0.0) # neuron 2 output

        input_3=3
        output=net.network_step(input_3)
        self.assertAlmostEqual(output[0], input_3)
        self.assertAlmostEqual(output[1], input_1)

class YamadaTestCase(unittest.TestCase):
    def setUp(self):
        params={'model': "Yamada_1", 'dt': 5.e-3}
        params["y0"]=Neuron(params).steady_state([0., 6.5, -6.])
        self.neuron=Neuron(params)
        self.neurons=[Neuron(params), Neuron(params)]
        self.params=params

    def testSolve(self):
        #solve a network of a single neuron, should output the same as neuron.solve
        #should work for any neuron so if change defaults this should still pass
        weights=[1., 0]
        net = Network([self.neuron], weights)
        tlength=1000
        input1=0.5*np.ones(tlength) #create an input signal
        #if plotted, this will cause default neuron to spike twice  

        output=net.network_solve(input1) #solve for the network

        neuron2=Neuron(self.params) #create an identical neuron
        output2=neuron2.solve(input1)[:, 0] #solve for same input signal

        #assert means of signals are the same
        self.assertAlmostEqual(np.mean(output), np.mean(output2)) 
        #assert entire output signals are the same!
        npt.assert_array_almost_equal(output.squeeze(),output2) 





if __name__ == "__main__":
    unittest.main()
