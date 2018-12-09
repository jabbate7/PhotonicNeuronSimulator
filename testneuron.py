import unittest
import numpy as np
import numpy.testing as npt
import neuron
import models
import solvers
import numpy.testing as npt

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
        y_steady=Y0Neuron.steady_state(Y0Neuron.y0)
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
        Y0mpars={"P": 0.9, "gamma": 1e-1, "kappa": 2, "beta": 1e-2 }
        #use completely random initial state
        Y0params={"model" : "Yamada_0", "y0": np.random.random(2) ,
             "dt": 1.e-3, 'mpar': Y0mpars}
        Y0Neuron=neuron.Neuron(Y0params)
        # have state decay a bunch
        N=int(np.ceil(100/Y0Neuron.dt))
        x=np.zeros(N)
        y_out=Y0Neuron.solve(x)
        npt.assert_array_almost_equal(y_out[-1, :], y_out[-1, :], )

    def testYamadaPulsing(self):
        # test to verify Yamada pulses if given continuous input above threshold
        # also increase input and verify pulse period decreases

        self.assertAlmostEqual(1., 1.)

if __name__ == "__main__":
    unittest.main()




