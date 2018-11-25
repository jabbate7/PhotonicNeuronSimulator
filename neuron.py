# NEURON CLASS

import numpy as np
import models
import solvers

class Neuron(object):
    """This is a Neuron object
    Initialize it as 
    myneuron = Neuron(params)
    where params is a dict with parameter values, for example
    params = {'x0': 0.0, 'model': }

    """
    
    def __init__(self, params):
        # constructor to initialize a Neuron
        self.x0 = params.get("x0", 0.0) # initial state
        self.model = params.get("model", "FitzHughNagamo")
        self.dt = params.get("dt", 1.0e-6)
        self.y = 0.0 # output
        self.history_len = params.get("dt", 1.0e-6)


        # set model ...
        if self.model == 'FitzHughNagamo':
            # read in the parameters, do some stuff
            self.f = models.FitzHughNagamo(...)
            pass

        elif self.model == 'Yamada':
            # do sonething else
            pass

        else:
            # raise an exception (not implemented)
            pass 


        # set solver
        if self.solver == 'Euler':
            pass

        elif self.solver == 'RK4':
            pass

        else:
            # raise an exception (not implemented)
            pass


    def __repr__(self):
        # how to print a Neuron
        pass

    def step(self, x):
        """ get the output at the next time step, given an input x
            update history ...
        """
        return x
        #pass