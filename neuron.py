# NEURON CLASS

import numpy as np
import models
from scipy import optimize
import inspect
import matplotlib.pyplot as plt

class Neuron(object):
    """
    A Neuron object is a specific instance of a neuron model, 
    which can exits on its own or as a member of a network.

    A Neuron is initialized with ``myneuron = Neuron(params)``, where 
    params is an optional dict of parameter values for myneuron. 
    Once initialized, the primary method of a neuron object is to calculate 
    its evolution given an input.

    Parameters
    ----------
    params
        A dictionary of parameter values, elements described below:
    model
        name of the evolution function :math:`\\dot{y}=f(x,y)` 
        implemented by neuron and described in ``models.py``.
        defaults to "identity"
    solver
        name of ODE solver used to update neuron state and 
        solve its dynamics, default is "Euler"
    dt
        time-step to use in ODE solver, defaults to 1.e-4
    hist_len
        length of list of previous neuron states to store. 
        default is 10
    y0
        initial state of neuron, defaults to zeros
    mpar
        a dictionary of model specific parameters describing 
        a specific neuron instance.  See ``models.py`` for a
        given model's parameters

    Attributes
    ----------
    y
        current neuron state
    y0
        initial sate of neuron
    hist
        list of previous neuron states:  
        hist[0] = y(t); hist[1] = y(t-dt); and so on.
        For multidimensional neurons, history only stores output/state variable.
    hist_len
        length of hist list, 
        number of previous states neuron stores
    dt
        time step
    model
        name of the evolution function :math:`\\dot{y}=f(x,y)` 
    mpars
        list of parameters of evolution function
    dim
        dimensionality of neuron phase space

    """
    
    def __init__(self, params={}):

        self.model = params.get("model", "identity")
        self.solver = params.get("solver", "Euler")

        # set model ...
        self.set_model()
        #pdb.set_trace()

        # set initial state, default: all zeros
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
        Constructor helper function, sets time-step for neuron

        Parameters
        ----------
        dt
            time-step for neuron ODE solver
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
        Sets neuron initial state and clears neuron history

        Parameters
        ----------
        y0
            new initial state for neuron
            
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
        Constructor helper function, 
        initializes empty history list of previous neuron states

        Parameters
        ----------
        hist_len
            length of history list
        """
        self.hist_len = hist_len
        if np.isscalar(self.y0):
            self.hist = [self.y0] 
        else:
            self.hist = [self.y0[0]]

    def set_model_params(self, mkwargs):
        """
        Constructor helper function, sets neuron's evolution function

        Parameters
        ----------
        mkwargs
            model specific list of parameters for evolution function
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
        """
        Steps the neuron forward one time step using Euler's method

        Computes the state of the neuron at the next time step, given
        the current input signal and neuron state: 
        :math:`y(n+1)=y(n)+dt\\cdot f(x(n), y(n))`

        Parameters
        ----------
        x
            the input signal at the current time (a scalar)

        Returns
        ----------
        np.array
            The state of the neuron at t+dt
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
        Steps the neuron forward one time step using 
        the fourth-order Runge-Kutta method

        Computes the state of the neuron at the next time step, given
        the current input signal and neuron state: 
        :math:`y(n+1)=y(n)+dt\\cdot f(x(n), y(n))`

        Parameters
        ----------
        x
            the input signal at the current time (a scalar)

        Returns
        ----------
        np.array
            The state of the neuron at t+dt        
        """
        # RK4 needs to know previous inputs as well
        if not hasattr(self, 'x_prev'): self.x_prev = x
        k1 = self.f(self.x_prev, self.y)
        k2 = self.f(0.5*(x + self.x_prev), self.y + 0.5*self.dt*k1)
        k3 = self.f(0.5*(x + self.x_prev), self.y + 0.5*self.dt*k2)
        k4 = self.f(x, self.y + self.dt*k3)
        self.x_prev = x

        self.y = self.y + (k1/6 + k2/3 + k3/3 + k4/6) *self.dt

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
        Calculates the neuron's dynamics in response to an input signal x(t).

        Uses the solver defined in ``solver`` to step trough each element of x(t)
        and compute the resultant neuron evolution.


        Parameters
        ----------
        x
            the time-dependent input signal (1d array)


        Returns
        ----------
        np.array
            the resultant neuron phase space dynamics
            as a (neuron.dim X num_timesteps) array                

        """
        y_out = np.zeros((len(x), self.dim))
        y_out[0,:] = self.y # initial state
        for i in np.arange(len(x)-1):
            y_out[i+1,:] = self.step(x[i])

        return y_out

    def steady_state(self, yguess=None):
        """
        solve for the no-input steady-state of the neuron.

        Choose yguess "wisely" to avoid unphysical roots.  We recommend
        testing steady state by running ``myneuron.solve(np.zeros(N))``
        and verifying the dynamics converge to the calculated steady-state

        Parameters
        ----------
        yguess
            an initial guess of the steady-state; method will return fixed-point
            closest to yguess.  Default is ``myneuron.y0``

        Returns
        ----------
        np.array
            The steady state of the neuron as calculated by setting 
            :math:`f(0, y)=0` and solving for y nearest to yguess               

        """
        if yguess is None:
            yguess=self.y0
        ODEs=lambda y: self.f(0, y)
        Root=optimize.fsolve(ODEs, yguess)
        return Root

    def visualize_plot(self, x_in, output, time=None, ysteady=None):
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
            the time-dependent input signal (1d array)
        outputs
            the resultant neuron phase space dynamics
            as a (neuron.dim X num_timesteps) array
        ysteady
            Optional neuron steady state to include in plot

        Returns
        ----------
        matplotlib.figure.Figure
            A matplotlib figure instance showing the network dynamics
        """

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
