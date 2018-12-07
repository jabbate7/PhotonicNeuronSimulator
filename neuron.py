# NEURON CLASS

import numpy as np
import models
import solvers
import scipy as sp

class Neuron(object):
    """This is a Neuron object
    Initialize it as 
    myneuron = Neuron() # default, or
    myneuron = Neuron(params)
    where params is a dict with parameter values, for example
    params = {'model': "FitzHughNagumo", 'y0': [0.0, 0.0]}
    params = {'model': "Yamada0", 'mpar': {'P': 0.8, 'gamma': 1.e-2, 'kappa': 1, 'beta': 1e-3} }

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
        self.solver = params.get("solver", "Euler")
        self.dt = params.get("dt", 1.0e-6)
        self.hist_len = params.get("hist_len", 10)
        self.hist = []
        self.y0 = params.get("y0", np.zeros(self.dim))

        # set model ...
        if self.model == 'identity':
            self.dim = 1
            self.fun = models.identity
            # for the identity, the step parameter h should be the same as dt
            params['mpar'] = {'h': self.dt} 

        elif self.model == 'FitzHughNagumo':
            self.dim = 2
            self.fun = models.FitzHughNagamo

        elif self.model == 'Yamada_0':
            self.dim = 2
            self.fun = models.Yamada_0

        elif self.model == 'Yamada_1':
            self.dim = 3
            self.fun = models.Yamada_1

        elif self.model == 'Yamada_2':
            self.dim = 3
            self.fun = models.Yamada_2

        else:
            raise ValueError("Not implemented")


        # read initial state
        # default initial state, all zeros
        self.y = params.get("y0", np.zeros(self.dim)) 
        if np.isscalar(self.y):
            self.y = np.array([self.y])
        if len(self.y) != self.dim:
            raise ValueError(
                """The initial state has {0:d} dimensions but the {1:s} model 
                has a {2:d}-dim phase space
                """.format(len(self.y), self.model, self.dim))

        self.hist.insert(0, self.y.copy())

        # read model specific parameters such as tau
        mkwargs = params.get('mpar', {}) 
        # set parameter-agnostic stepping function 
        self.f = lambda x, y : self.fun(x, y, **mkwargs)

        # set solver ...
        if self.solver == 'Euler':
            self.step = self.step_Euler

        elif self.solver == 'RK4':
            self.step = self.step_RK4

        else:
            raise ValueError("Not implemented")

    def __repr__(self):
        return "Neuron of type {0:s}".format(self.type)

    def step_Euler(self, x):
        """ get the output at the next time step, given an input x
            update history ...
            y_{n+1} = y_n + h f(x_n, y_n)
        """
        self.y = self.y + self.dt * self.f(x, self.y)

        self.hist.append(self.y.copy())
        # trim the history from the back if it grows too big
        if len(self.hist) > self.hist_len: 
            _ = self.hist.pop()

        return self.y # return output y (t+dt)

    def step_RK4(self, x):
        """
        RK4 stepper insted of the Euler stepper above
        """
        k1 = self.f(x, self.y)
        k2 = 0
        self.y = self.y + self.dt * self.f(x, self.y)

        self.hist.append(self.y.copy())
        # trim the history from the back if it grows too big
        if len(self.hist) > self.hist_len: 
            _ = self.hist.pop()

        return self.y # return output y (t+dt)


    def solve(self, x):
        """ get the entire output time series of a neuron with input
        time series x
        """
        y_out = np.zeros((len(x), self.dim))
        y_out[0,:] = self.y # initial state
        for i in np.arange(len(x)-1):
            y_out[i,:] = self.step(x[i])

        return y_out

    def steady_state(self):
        """
        solve for the no-input steady state of the neuron
        """
        ODEs=lambda y: self.f(0, y) #assume zero input
        yguess=self.y0
        Root=sp.optimize.fsolve(ODEs, Root)
        #should probably write stuff here at one point so catches unphysical roots
        #also many of these systems have multistability so may not return the root we want
        return Root

    def step_t_to_n(self):
        """
        convert time to discrete units by dividing a time interval by the 
        size of a time step
        """
        pass

    def set_dt(self, dt):
        self.dt = dt

    def set_history(self, t_hist):
        self.hist_len = self.t_hist