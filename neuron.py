# NEURON CLASS

import numpy as np
import models
from scipy import optimize
import pdb
import inspect
import matplotlib.pyplot as plt

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
        # -- y0       : initial
        # -- y        : current (most recent) state  
        # -- hist     : list of most recent states 
        #               hist[0] = y(t); hist[1] = y(t-dt); and so on
        # -- hist_len : length of time history to be stored

        self.model = params.get("model", "identity")
        self.solver = params.get("solver", "Euler")

        # set model ...
        self.set_model()
        #pdb.set_trace()

        # set initial state, default: all zeros

        # some BS merge conflict here
        ## self.set_initial_state(np.array(params.get("y0", np.zeros(self.dim))))
        self.set_initial_state(params.get("y0", np.zeros(self.dim)))
        # set history, do it after creating y0 and model
        self.set_history(params.get("hist_len", 10))


        self.set_dt(params.get("dt", 1.0e-4))
        
        # read model specific parameters such as tau
        if self.model != 'identity':
            self.set_model_params(params.get('mpar', {}))

        # set solver ...
        if self.solver == 'Euler':
            self.step = self.step_Euler
        elif self.solver == 'RK4':
            self.step = self.step_RK4
        else:
            raise ValueError("Not implemented")

    def __repr__(self):
        return "Neuron of type {0:s}".format(self.model)

    #####  CONSTRUCTOR HELPER FUNCTIONS  #####

    def set_dt(self, dt):
        """ 
        Set timestep for Neuron
        """
        self.dt = dt
        # for the identity, the step parameter h should be the same as dt
        if self.model == "identity":
            self.set_model_params({'h': self.dt})

    def set_model(self):
        """
        constructor helper function, sets Neuron model and model's key properties
        """
        if self.model == 'identity':
            self.dim = 1
            self.fun = models.identity

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

    def set_initial_state(self, y0=None):
        """
        set initial conditions
        """
        if y0 is None:
            y0 = np.zeros(self.dim)

        self.y0 = y0

        if np.isscalar(self.y0):
            self.y = np.array([self.y0])
            #wipe history as well, replace with initial state
            self.hist = [self.y0] 
        else:
            self.y = self.y0.copy() 
            #same thing, but history must be list of scalars
            self.hist=[self.y0[0]]

        if len(self.y) != self.dim:
            raise ValueError(
                """The initial state has {0:d} dimensions but the {1:s} model 
                has a {2:d}-dim phase space
                """.format(len(self.y), self.model, self.dim))

        # for cnt in np.arange(self.hist_len):
        #     self.hist.insert(0, self.y.copy())

    def set_history(self, hist_len):
        """
        Constructs empty history list of length hist_len,
        to be populated with previous neuron states.
        For multidemensional Neurons, history only stores output/state variable.
        """
        self.hist_len = hist_len
        if np.isscalar(self.y0):
            self.hist = [self.y0] 
        else:
            self.hist = [self.y0[0]]

    def set_model_params(self, mkwargs):
        """
        set parameter-agnostic stepping function, also saves model parameters
        """
        self.f = lambda x, y : self.fun(x, y, **mkwargs)
        self.mpars=mkwargs 
        if len(self.mpars)==0: #if dont pass, mpars is empty
            signature = inspect.signature(self.fun) #need to read default parameters from model
            self.mpars= {
                k: v.default
                for k, v in signature.parameters.items()
                if v.default is not inspect.Parameter.empty
            }



    #### STEPPER FUNCTIONS (THE HEART OF THE ODE SOLVER) ####

    def step_Euler(self, x):
        """ get the output at the next time step, given an input x
            update history ...
            y_{n+1} = y_n + h f(x_n, y_n)
        """
        self.y = self.y + self.dt * self.f(x, self.y)
        if self.dim>1:
            self.hist.insert(0,self.y[0] )#first element is "state" variable 
        else:
            self.hist.insert(0,self.y)

        # trim the history from the back if it grows too big
        if len(self.hist) > self.hist_len: 
            _ = self.hist.pop()

        return self.y # return output y (t+dt)

    def step_RK4(self, x):
        """
        RK4 stepper insted of the Euler stepper above
        [right now its still just Euler]
        """
        k1 = self.f(x, self.y)
        k2 = self.f(x + 0, self.y + 0.5*self.dt*k1)
        k3 = self.f(x + 0, self.y + 0.5*self.dt*k1)
        k4 = self.f(x + 0, self.y + 0.5*self.dt*k1)

        self.y = self.y + self.dt * self.f(x, self.y)

        if self.dim>1:
            self.hist.insert(0,self.y[0].copy()) #first element is "state" variable 
            #also dont think i need .copy() anymore
        else:
            self.hist.insert(0,self.y)

        # trim the history from the back if it grows too big
        if len(self.hist) > self.hist_len: 
            _ = self.hist.pop()

        return self.y # return output y (t+dt)


    def solve(self, x):
        """ 
        Calculate the entire dynamics of the neuron with input x(t).
        """
        y_out = np.zeros((len(x), self.dim))
        y_out[0,:] = self.y # initial state
        for i in np.arange(len(x)-1):
            y_out[i+1,:] = self.step(x[i])

        return y_out

    def steady_state(self, yguess=None):
        """
        solve for the no-input steady state of the neuron.
        Choose yguess "wisely" to avoid unphysical roots.
        Recommended to test steady state by ruling solve(np.zeros(N))
        and seeing if state stays stationary.
        """
        if yguess is None:
            yguess=self.y0
        ODEs=lambda y: self.f(0, y)
        Root=optimize.fsolve(ODEs, yguess)
        return Root

    def visualize_plot(self, x_in, output, time=None, ysteady=None, plotparams=None):
        """
        Generate a simple and easy to read plot of the neuron dynamics. 

        After solving a neuron for a given input signal, pass this and computed
        dynamics to generate a plot. 
        Use returned figure handle to update plot parameters from defaults 
        if desired.

        Parameters
        -----------
        time
            an array of time points which input and output are plotted over
        x_in
            the time-dependant input signal (1d array)
        outputs
            the resultant neuron phase space dynamics
            as a (neuron.dim X num_timesteps) array
        ysteady
            Optional Neuron steady state to include in plot
        plotparams
            A dictionary of plot parameters to be used

        Returns
        ----------
        np.array
            A 3D array (num_timesteps X num_neuron X neuron.dim)
             of the full state of each neuron in the network at each time.
        """

        #havent implimented default parameters yet, sue me

        Len_t=output.shape[0] #length of time vector
        msg2="input and output should both the same temporal length"
        assert ( len(x_in)==int(Len_t) ), msg2
        if time is None: #didnt pass time, so compute from dt and signal length TL
            time=np.linspace(0., (Len_t-1)*self.dt, num=Len_t)
        colors=['b', 'r', 'g', 'c'] #use these when plotting

        fig=plt.figure()
        ax1=fig.add_axes([0,0.0, 1, 0.6])
        ax2=ax1.twinx()
        ax3=fig.add_axes([0,.7, 1, 0.3])

        if ysteady is not None: #if have steady states, plot them first
            ax1.plot(time, ysteady[0]*np.ones(Len_t), '--'+colors[0], linewidth=2)
            for ind in range(1, self.dim):
                if ind==2 and (self.model == 'Yamada_1' or self.model == 'Yamada_2' ):
                    ax2.plot(time, -ysteady[ind]*np.ones(Len_t), '--'+colors[ind], linewidth=2)
                else:
                    ax2.plot(time, ysteady[ind]*np.ones(Len_t), '--'+colors[ind], linewidth=2)


        # plot Neuron state and input current
        ax1.plot(time, output[:,0], 'b')
        for ind in range(1, self.dim):
            if ind==2 and (self.model == 'Yamada_1' or self.model == 'Yamada_2' ):
                 #want to flip Q->-Q in this case
                ax2.plot(time, -output[:, ind], '-.'+colors[ind])
            else:
                ax2.plot(time, output[:, ind], '-.'+colors[ind])


        ax3.plot(time, x_in, '-k')



        ax1.set_xlabel('t [$1/\gamma$]')
        ax1.set_ylabel('$I$ [arb units]')
        ax3.set_ylabel('$i_{in}$ [$\gamma$]')
        ax2.set_ylabel('$J$ [arb units]')

        ax1.set_xlim(time[0], time[-1])
        ax2.set_xlim(time[0], time[-1])
        ax3.set_xlim(time[0], time[-1])

        if (self.model == 'Yamada_1' or self.model == 'Yamada_2' ): #want to flip Q->-Q in this case
            ax2.set_ylabel('$G,\,-Q$ [arb units]')


        return fig

    def step_t_to_n(self):
        """
        convert time to discrete units by dividing a time interval by the 
        size of a time step
        """
        pass
