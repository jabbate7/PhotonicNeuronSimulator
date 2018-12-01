# NEURON CLASS

import numpy as np
import models
import solvers

class Neuron(object):
    """This is a Neuron object
    Initialize it as 
    myneuron = Neuron(params)
    where params is a dict with parameter values, for example
    params = {'model': "identity", 'y0': [0.0]}

    """
    
    def __init__(self, params):
        # constructor to initialize a Neuron
        # attributes:
        
        self.model = params.get("model", "identity")
        self.dt = params.get("dt", 1.0e-6)
        self.history_len = params.get("history_len", 10)


        # set model ...
        if self.model == 'identity':
            self.dim = 1
            self.fun = models.identity
            # for the identity, the step parameter h should be the same as dt
            params['mpar'] = {'h': self.dt} 

        elif self.model == 'FitzHughNagumo':
            self.dim = 2
            self.fun = models.FitzHughNagamo

        elif self.model == 'Yamada':
            self.dim = 3
            self.fun = models.Yamada

        else:
            raise ValueError("Not implemented")


        # read initial state
        # default initial state, all zeros
        self.y = params.get("y0", np.zeros(self.dim)) 
        if np.isscalar(self.y): self.y = np.array([self.y])
        if len(self.y) != self.dim:
            raise ValueError(
                "The initial state has {0:d} dimensions but the {1:s} model has a {2:d}-dim phase space".
                format(len(self.y), self.model, self.dim))

        mkwargs = params.get('mpar') # read model specific parameters such as tau
        self.f = lambda x, y : self.fun(x, y, **mkwargs)

        # set solver
        #if self.solver == 'Euler':
        #    pass



    def __repr__(self):
        # how to print a Neuron
        pass

    def step(self, x):
        """ get the output at the next time step, given an input x
            update history ...
            y_{n+1} = y_n + h f(x_n, y_n)
        """
        self.y = self.y + self.dt * self.f(x, self.y)
        return self.y # return output y (t+dt)

    def solve(self, x):
        """ get the entire output time series of a neuron with input time series x
        """
        pass

        #pass