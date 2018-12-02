# NEURON CLASS

import numpy as np
import models
import solvers

class Neuron(object):
    """This is a Neuron object
    Initialize it as 
    myneuron = Neuron() # default, or
    myneuron = Neuron(params)
    where params is a dict with parameter values, for example
    params = {'model': "FitzHughNagumo", 'y0': [0.0, 0.0]}

    """
    
    def __init__(self, params={}):
        # constructor to initialize a Neuron
        # attributes:
        # -- model    : name of function implemented by neuron
        # -- dim      : dimensionality of phase space of model
        # -- dt       : time step
        # -- y        : current (most recent) state  
        # -- hist     : list of most recent states 
        #               hist[0] = y(t); hist[1] = y(t-dt); and so on
        # -- hist_len : length of time history to be stored

        self.model = params.get("model", "identity")
        self.dt = params.get("dt", 1.0e-6)
        self.hist_len = params.get("hist_len", 10)
        self.hist = []

        # set model ...
        if self.model == 'identity':
            self.dim = 1
            self.fun = models.identity
            params['mpar'] = {'h': self.dt} # for the identity, the step parameter h should be the same as dt

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
        if np.isscalar(self.y):
            self.y = np.array([self.y])
        if len(self.y) != self.dim:
            raise ValueError(
                "The initial state has {0:d} dimensions but the {1:s} model has a {2:d}-dim phase space".
                format(len(self.y), self.model, self.dim))

        self.hist.insert(0, self.y.copy())

        # read model specific parameters such as tau
        mkwargs = params.get('mpar') 
        # set parameter-agnostic stepping function 
        self.f = lambda x, y : self.fun(x, y, **mkwargs)

        # set solver
        #if self.solver == 'Euler':
        #    pass

    def __repr__(self):
        return "Neuron of type {0:s}".format(self.type)

    def step(self, x):
        """ get the output at the next time step, given an input x
            update history ...
            y_{n+1} = y_n + h f(x_n, y_n)
        """
        self.y = self.y + self.dt * self.f(x, self.y)

        self.hist.append(self.y.copy())
        # trim the history from the back if it grows too big
        if len(self.hist > self.hist_len): 
            _ = self.hist.pop()

        return self.y # return output y (t+dt)

    def solve(self, x):
        """ get the entire output time series of a neuron with input time series x
        """
        y_out = np.arange(len(x)+1)
        y_out[0] = self.y # initial state
        for i in np.arange(len(x)):
            y_out[i] = self.step(x[i])

        return y_out
