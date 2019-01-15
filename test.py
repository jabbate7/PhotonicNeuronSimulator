import unittest
import numpy as np
import numpy.testing as npt
from neuron import Neuron
from network import Network
import neuron
import models
import scipy.signal as sig

class TestNeuronSetUp(unittest.TestCase):
    def setUp(self):
        self.neur = neuron.Neuron()
        self.neur2=neuron.Neuron({'model': "Yamada_1", 'y0': [0., 0., 0.]})
        
    def testHist(self):
        hist_len_1=10
        self.neur.set_history(hist_len_1)
        self.assertEqual(hist_len_1, len(self.neur.hist))
        hist_len_2=20
        self.neur.set_history(hist_len_2)
        self.assertEqual(hist_len_2, len(self.neur.hist))
    
    def testDT(self):
        dt_1=10
        self.neur.set_dt(dt_1)
        self.assertEqual(dt_1, self.neur.dt)
        dt_2=20
        self.neur.set_dt(dt_2)
        self.assertEqual(dt_2, self.neur.dt)
        
    # make sure the history array kept by the neuron (1) works
    # and (2) properly "circulates", i.e. kicks values out
    def testHist(self):
        self.neur.set_history(2)
        input_1=1
        self.neur.step(input_1)
        self.assertAlmostEqual(self.neur.hist[0],input_1)

        input_2=2
        self.neur.step(input_2)
        self.assertAlmostEqual(self.neur.hist[0],input_2)
        self.assertAlmostEqual(self.neur.hist[1],input_1)

        input_3=3
        self.neur.step(input_3)
        self.assertAlmostEqual(self.neur.hist[0],input_3)
        self.assertAlmostEqual(self.neur.hist[1],input_2)

    def testHist3D(self):
        # make sure hist still works as expected for higher dim neurons
        #should only store y[0] variable
        hist_len_1=10
        self.neur2.set_history(hist_len_1)
        #first element of history should be y0[0]
        self.assertAlmostEqual(self.neur2.hist[0],self.neur2.y0[0])


        input_1=1
        #take several steps 
        steps=3
        for ind in range(0, steps):
            self.neur2.step(input_1)
        #make sure history has updated
        self.assertAlmostEqual(self.neur2.hist[0],self.neur2.y[0])
        # after taking n steps, history should have n+1 entries
        # or 10 entries if n+1>10
        self.assertEqual(len(self.neur2.hist), min(steps+1, hist_len_1))

    def testY0(self):
        #test that resetting initial conditions works
        #should also clear history and update y
        hist_len_1=10
        self.neur2.set_history(hist_len_1)
        input_1=5.e-2
        #take several steps 
        steps=12
        for ind in range(0, steps):
            self.neur2.step(input_1)
        #set new initial state    
        newIC=[1.e-2, 6.5, -6.]
        self.neur2.set_initial_state(newIC)
        #y0, hist and y should all be this now
        self.assertAlmostEqual(self.neur2.y0,newIC)
        self.assertAlmostEqual(self.neur2.y,newIC)
        self.assertAlmostEqual(self.neur2.hist[0],newIC[0])
        self.assertEqual(len(self.neur2.hist), 1)

    def testModelError(self):
        #makes sure get exception if initialize without implemented model
        with self.assertRaises(Exception):
            neuron.Neuron({'model': "NotaNeuron"})

    def testHistError(self):
        self.neur.set_history(1)
        with self.assertRaises(Exception):
            self.neur.hist[1]

class TestNeuronBasic(unittest.TestCase):
    def testIdentity(self):
        # test to make sure neuron step and neuron solve works for identity neuron
        # input=output
        Idparams={"model" : "identity", "y0": 0., "dt": 1.e-6}
        IdNeuron=neuron.Neuron(Idparams)
        DCin=2.
        DCout=IdNeuron.step(DCin)
        self.assertAlmostEqual(DCin, DCout)

        #this should work for any step size or initial state
        Idparams2={"model" : "identity", "y0": np.pi, "dt": 1.e2}
        IdNeuron2=neuron.Neuron(Idparams2)
        DCin=2.
        DCout2=IdNeuron2.step(DCin)
        self.assertAlmostEqual(DCout, DCout2)

        #test  neuron.solve for IdNeuron
        Inlength=1e5
        Idin =np.sin(np.linspace(0, 2.*np.pi, int(Inlength))) 
        Idout=IdNeuron.solve(Idin)
        npt.assert_array_almost_equal(Idin[:-1,np.newaxis], Idout[1:])

    def testSteady(self):
        # test toverify that Neuron evolves to steady state, 
        # and verify that this is predicted by steady_state method
        # work with Yamada0 first
        Y0mpars={"P": 0.9, "gamma": 1e-1, "kappa": 2, "beta": 1e-2 }
        #use completely random initial state
        Y0params={"model" : "Yamada_0", "y0": np.random.random(2) ,
             "dt": 1.e-2, 'mpar': Y0mpars}
        Y0Neuron=neuron.Neuron(Y0params)
        # have state decay a bunch
        N=int(np.ceil(100/Y0Neuron.dt))
        x=np.zeros(N)
        y_out=Y0Neuron.solve(x)
        # also tests that steady state solver works
        y_steady=Y0Neuron.steady_state([Y0mpars['beta']/Y0mpars['kappa'], Y0mpars['P']])
        npt.assert_array_almost_equal(y_out[-1, :], y_steady)

        #try with another neuron model
        Y1mpars={"a": 2, "A": 6.3, "B":-6.,
            "gamma1": 1e-1, "gamma2": 1e-1, "kappa": 2, "beta": 1e-3 }
        #use completely random initial state
        Y1params={"model" : "Yamada_1", "y0": np.random.random(3) ,
             "dt": 1.e-2, 'mpar': Y1mpars}
        Y1Neuron=neuron.Neuron(Y1params)
        # have state decay a bunch
        N1=int(np.ceil(500/Y1Neuron.dt))
        x1=np.zeros(N1)
        y1_out=Y1Neuron.solve(x1)
        # this should be close to the steady state,
        # note that Yamada neuron has 3 fixed points (2 unstable) in this region
        y1_steady_est=[Y1mpars['beta']/Y1mpars['kappa'],
               Y1mpars['A'],Y1mpars['B'] ]
        y1_steady=Y1Neuron.steady_state(y1_steady_est)

        npt.assert_array_almost_equal(y1_out[-1, :], y1_steady, decimal=3)

class TestNeuronDynamics(unittest.TestCase):
    def testYamadaSpike(self):
        # check if ode stepper is working by seeing if neuron evolves as predicted
        # test to verify Yamada neuron spikes if given an input above threshold
        Gaussian_pulse= lambda x, mu, sig: np.exp(-np.power(x - mu, 2.) 
            / (2 * np.power(sig, 2.)))/(np.sqrt(2*np.pi)*sig)

        Y1mpars={"a": 2, "A": 6.5, "B":-6., "gamma1": 1e-1,
         "gamma2": 1e-1, "kappa": 2, "beta": 1e-2 }
        y1_steady_est=[Y1mpars['beta']/Y1mpars['kappa'],
            Y1mpars['A'],Y1mpars['B'] ]
        Y1params={"model" : "Yamada_1", "y0": y1_steady_est,
        "dt": 1.e-2, 'mpar': Y1mpars} #close enough to steady state
        Y1Neuron=neuron.Neuron(Y1params)
        y1_steady=Y1Neuron.steady_state(y1_steady_est)

        #create time signal
        t1_end=10./Y1mpars["gamma1"]; #atleast this long
        N1=int(np.ceil(t1_end/Y1Neuron.dt))
        time1=np.linspace(0.,(N1-1)*Y1Neuron.dt, num=N1 )
        x1=Gaussian_pulse(time1, 0.5/Y1mpars["gamma1"], 1.)

        # create neuron, solve
        y1_out=Y1Neuron.solve(x1)
        #peak height scales roughly as kappa/gamma1
        # so roughly spikes if max of the signal> ~kappa/gamma2
        # also make sure returns to steady state
        self.assertGreaterEqual(np.max(y1_out[:,0]), 0.5*Y1mpars["kappa"]/Y1mpars["gamma1"])
        npt.assert_array_almost_equal(y1_out[-1, :], y1_steady, decimal=2)

    def test_RK4_vs_Euler(self):
        # check if RK4 stepper works in the same way as the Euler stepper
        Gaussian_pulse= lambda x, mu, sig: np.exp(-np.power(x - mu, 2.) 
            / (2 * np.power(sig, 2.)))/(np.sqrt(2*np.pi)*sig)

        Y1mpars={"a": 2, "A": 6.5, "B":-6., "gamma1": 1e-1,
         "gamma2": 1e-1, "kappa": 2, "beta": 1e-2 }
        y1_steady_est=[Y1mpars['beta']/Y1mpars['kappa'],
            Y1mpars['A'],Y1mpars['B'] ]
        Y1params={"model" : "Yamada_1", "y0": y1_steady_est,
        "dt": 1.e-2, 'mpar': Y1mpars, 'solver': 'Euler'} #close enough to steady state
        Y1Neuron=neuron.Neuron(Y1params)
        y1_steady=Y1Neuron.steady_state(y1_steady_est)

        Y1params['solver'] = 'RK4'
        Y2Neuron=neuron.Neuron(Y1params)
        y2_steady=Y1Neuron.steady_state(y1_steady_est)

        #create time signal
        t1_end=10./Y1mpars["gamma1"]; #atleast this long
        N1=int(np.ceil(t1_end/Y1Neuron.dt))
        time1=np.linspace(0.,(N1-1)*Y1Neuron.dt, num=N1 )
        x1=Gaussian_pulse(time1, 0.5/Y1mpars["gamma1"], 1.)

        # create neuron, solve
        y1_out=Y1Neuron.solve(x1)
        y2_out=Y2Neuron.solve(x1)

        # calculate L2 norm of the difference of the two 
        L2_err = np.sum((y1_out[3:] - y2_out[:-3])**2) / np.sum((y1_out)**2)

        # should throw an error if the outputs are significantly different
        self.assertTrue(L2_err < 1e-5)


    def testYamadaPulsing(self):
        # test to verify Yamada pulses if given continuous input above threshold
        # also increase input and verify pulse period decreases
        Y1mpars={"a": 1.8, "A": 5.7, "B":-5., "gamma1": 1e-2,
         "gamma2": 1e-2, "kappa": 1, "beta": 1e-3 }
        y1_steady_est=[Y1mpars['beta']/Y1mpars['kappa'],
            Y1mpars['A'],Y1mpars['B'] ]
        Y1params={"model" : "Yamada_1", "y0": y1_steady_est,
        "dt": 1.e-2, 'mpar': Y1mpars} #close enough to steady state
        Y1Neuron=neuron.Neuron(Y1params)
        y1_steady=Y1Neuron.steady_state(y1_steady_est)

        #create time signal
        t1_end=10./Y1mpars["gamma1"]; #atleast this long
        N1=int(np.ceil(t1_end/Y1Neuron.dt))
        time1=np.linspace(0.,(N1-1)*Y1Neuron.dt, num=N1 )
        x1=np.zeros(N1)
        #make sure x1 amplitude is sufficient for spiking
        switchtime=8./Y1mpars["gamma1"] #increase drive at this point
        x1+=(0.5*Y1mpars["gamma1"])*np.heaviside(time1-0.5/Y1mpars["gamma1"], 0.5)
        x1+=(1.5*Y1mpars["gamma1"])*np.heaviside(time1-switchtime, 0.5)
        
        y1_out=Y1Neuron.solve(x1)

        (peaks, props) = sig.find_peaks(y1_out[:,0], height=1e-2*Y1mpars["kappa"]/Y1mpars["gamma1"])
        peaktimes=time1[peaks]
        self.assertGreaterEqual(peaktimes.size, 2) #assert spiked atleast twice

        (peaktimes1, peaktimes2)=(np.array([]), np.array([]))
        for (i, time) in enumerate(peaktimes):
            if time <= switchtime:
                peaktimes1=np.append(peaktimes1,time)
            else:
                peaktimes2=np.append(peaktimes2,time)
        if peaktimes1.size<2 or peaktimes2.size<2:
            raise Exception("Not enough spikes to determine period")

        (per1, per2)=(np.mean(np.diff(peaktimes1)), np.mean(np.diff(peaktimes2)))

        self.assertGreaterEqual(peaktimes.size, 2) #assert spiked faster in part 2

class TestNetworkBasics(unittest.TestCase):
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

class TestNetworkYamada(unittest.TestCase):
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
    def testFullNetwork(self):
        #create, solve, and visualize small random network
        #ensures main methods run and work together without producing errors

        num_inputs=2
        num_neurons=3
        neurons=[]
        for ind in range(num_neurons): #create neuron list
            neurons.append(Neuron(self.params))

        #weights and delays are random matrices [0, 1)
        delays=np.random.rand(num_neurons, num_neurons)
        #shift weights so inputs are positive, connections random
        weights=np.concatenate((np.random.rand(num_neurons, num_inputs),
                       np.random.rand(num_neurons, num_neurons)-0.5), axis=1 )

        network=Network(neurons, weights, delays)

        tlength=2000
        #random external input as well
        external_input=0.3*(1.+np.random.rand(tlength, num_inputs) )
        time=np.arange(0., tlength*network.dt, network.dt)

        outputs = network.network_solve(external_input)

        total_inputs = network.network_inputs(outputs, external_input)

        testfig = network.visualize_plot(total_inputs, outputs, time)

        testan = network.visualize_animation(inputs=total_inputs, outputs=outputs)



if __name__ == "__main__":
    unittest.main()




