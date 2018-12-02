import unittest
import numpy as np
import numpy.testing as npt
import neuron
import models
import solvers

class TestNeuron(unittest.TestCase):
    def testIdentity(self):
        # test to make sure neuron step and neuron solve works for itentity neuron
        # input=output
        Idparams={"model" : "identity", "y0": 0., "dt": 1.e-6}
        IdNeuron=neuron.Neuron(Idparams)
        DCin=2.
        DCout=IdNeuron.step(Dcin)
        self.assertAlmostEqual(DCin, DCout)

        #this should work for any step size or initial state
        Idparams2={"model" : "identity", "y0": np.pi, "dt": 1.e2}
        IdNeuron2=neuron.Neuron(Idparams2)
        DCin=2.
        DCout2=Idneuron2.step(Dcin)
        self.assertAlmostEqual(DCout, DCout2)

        #test  neuron.solve for IdNeuron
        Inlength=1e5
        Idin =np.sin(np.linspace(0, 2.*np.pi, Inlength)) 
        Idout=IdNeuron.solve(Idin)
        self.assertAlmostEqual(Idin[1:], Idout[:-1])

    def testYamadaSteady(self):
        # test to verify Yamada model neuron goes to steady state

        self.assertAlmostEqual(1., 1.)

    def testYamadaSpike(self):
        # test to verify Yamada neuron spikes if given an input above threshold

        self.assertAlmostEqual(1., 1.)

    def testYamadaPulsing(self):
        # test to verify Yamada pulses if given continuous input above threshold
        # also increase input and verify pulse period decreases

        self.assertAlmostEqual(1., 1.)

if __name__ == "__main__":
    unittest.main()




