# NEURON CLASS

import numpy as np
import models
import solvers

class Neuron(object):
    """This is a Neuron object
    Initialize it as 
    myneuron = Neuron(params)
    where params is a dict with parameter values, for example
    params = {'model': "identity", 'x0': 0.0}

    """
    
    def __init__(self, params):
        # constructor to initialize a Neuron
        self.model = params.get("model", "identity")
        self.dt = params.get("dt", 1.0e-6)
        self.dim # number of dimensions of phase space
        self.y
        self.history_len = params.get("history_len", 10)


        # set model ...
        if self.model == 'identity':
            self.dim = 1
            self.f = models.identity
            


        elif self.model == 'FitzHughNagumo':
            self.x0 = params.get("x0", 0.0) # initial state
            # read in the parameters, do some stuff
            self.f = models.FitzHughNagamo

        elif self.model == 'Yamada':
            self.x0 = params.get("x0", 0.0) # initial state
            self.f = models.Yamada

        else:
            raise ValueError("Not implemented")


        self.y = params.get("initial_state", np.zeros(self.dim)) # default initial state, all zeros
        if len(self.y) != self.dim
            raise ValueError(
                "The initial state has {0:d} dimensions but the {1:s} model has a {2:d}-dim phase space".
                format(len(self.y), self.model, self.dim))

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
            y_{n+1} = y_n + h f(x_n, y_n)
        """
        y = x
        return y # return output y (t+dt)

    def solve(self, x):
        """ get the entire output time series of a neuron with input time series x
        """
        pass

        #pass