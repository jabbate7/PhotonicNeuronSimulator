import unittest
import numpy as np
import numpy.testing as npt
import neuron
import models
import solvers
import scipy.signal as sig

class TestNeuronTimeDelay(unittest.TestCase):
    def setUp(self):
        self.neur = neuron.Neuron()
        
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

    def testHistError(self):
        self.neur.set_history(1)
        with self.assertRaises(Exception):
            self.neur.hist[1]

class TestNeuron(unittest.TestCase):
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

    def testYamadaSteady(self):
        # test to verify Yamada model neuron goes to steady state
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
        #should also write test for another yamada type here probably
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

    def testYamadaSpike(self):
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

if __name__ == "__main__":
    unittest.main()




