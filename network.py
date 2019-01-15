from neuron import Neuron
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

# A network is a specific topology of Neurons
class Network:
    """
    A Network is a specific topology of connected Neuron objects. 
    Connections between neurons can be weighted, and also can carry a 
    time delay. There can be multiple input sources, which can be 
    connected to any neurons. These connections also have associated weights
    and time delays. 

    Once initialized, a Network's primary method is to take in a set of signals
    at a given time-step and reveal the updated neuron states. 

    Parameters
    ----------
    neurons
        A list of neuron objects.
    weights
        An np.array matrix of weights.
    delays
        An np.array matrix of time delays, in absolute time.
    dt
        The time step for propagating the network.

    Attributes
    ----------
    neurons
        a list of the neuron objects which make up the network
    weights
        the np.array matrix of weights for the network 
        (num_neuron X num_neuron+num_inputs)
    delays
        The np.array matrix of delays for the network, in units
        of dt (num_neuron X num_neuron)
    dt
        the time step for propagating the network.
    num_inputs
        the number of external inputs in the network
    num_neurons
        the number of neurons in the network
    """
        
    def __init__(self, neurons, weights, delays=None, dt=None):
        self.neurons=neurons

        weights = np.atleast_2d(weights) #asserts 2d array in case only 1 neuron

        self.weights=weights
        if (dt is not None):
            self.dt=dt
        else:
            self.dt=neurons[0].dt
        
        self.num_neurons = np.shape(weights)[0]
        self.num_inputs = np.shape(weights)[1]-self.num_neurons

        if (delays is not None):
            #delays is a list of times, need to convert to units of dt
            # divide by dt, so assumes its an np.array and not just a list
            assert isinstance(delays, np.ndarray), "Delays must be np.ndarray"
            msg="Delays must have shape ({}, {})".format(self.num_neurons, self.num_neurons)
            assert (delays.shape==(self.num_neurons,self.num_neurons)), msg
            self.delays=(delays/self.dt).astype(int)
        else:
            self.delays = np.zeros((self.num_neurons,self.num_neurons),dtype=int)
        
        max_times = np.apply_along_axis(lambda x: max(x)+1, 0, self.delays)
        #calculates longest time each neurons output is delayed
#        import pdb; pdb.set_trace()
        # set up each neuron with dt and necessary history
        for i,neuron in enumerate(neurons):
            #each neuron needs to store history=longest its output gets delayed
            neuron.set_history(max_times[i])
            neuron.set_dt(self.dt)
            
    def generate_neuron_inputs(self, external_inputs):
        """
        Builds input array for next network step.

        Each element in the array corresponds to a neuron in the network,
        calculated by summing all weighted and delayed inputs to that neuron.
        For each element, first adds weighted external inputs,
        then adds appropriately delayed weighted outputs from other neurons.
        
        Parameters
        ----------
        external_inputs
            A 1D array with the external inputs (i.e. non-neuron inputs).

        Returns
        ----------
        np.array
            array of summed inputs to each neuron at current time
        """
        def get_prev_output(row, col):
            # an internal function to get the inputs needed to update neuron states 
            if col < self.num_inputs:
                return external_inputs[col]
            else:
                col = col - self.num_inputs
                if (self.delays[row][col] >= len(self.neurons[col].hist)):
                    #to save space, history starts almost empty and is populated until full
                    # assumption is that neuron was in initial state for all times before calculation
                    return self.neurons[col].hist[-1] #return final element, which is initial state
                else:
                    return self.neurons[col].hist[self.delays[row][col]] #assumes 1d hist
        inputs=np.zeros(self.num_neurons)
        for row in range(self.num_neurons):
            for col in range(self.num_neurons+self.num_inputs):
                if self.weights[row][col] != 0:
                    inputs[row] += self.weights[row][col]*get_prev_output(row,col)
        return inputs


    def network_step(self,external_inputs):
        """
        Steps the network forward in time by dt.

        Each Neuron updates based on its state and its input,
        using its internal ode solver to compute its next time step.

        Parameters
        -----------
        external_inputs
            A (num_inputs X 1) array with the external inputs at the current time
            (i.e. non-neuron inputs).

        Returns
        ----------
        np.array
            The state (output) of each neuron after stepping forward by dt
        """
        external_inputs = np.atleast_1d(external_inputs)
        msg="Please specify {} inputs in an array".format(self.num_inputs)
        assert (len(external_inputs)==int(self.num_inputs)), msg
        # update the state of each neuron 
        neuron_inputs = self.generate_neuron_inputs(external_inputs)
        neuron_outputs = np.zeros(self.num_neurons)
        for i,neuron in enumerate(self.neurons):
            neuron.step(neuron_inputs[i]) 
            neuron_outputs[i]=neuron.hist[0] #ensures get scalar result
            #hist[0] is current state of output variable of a given neuron
        return neuron_outputs

    def __repr__(self):
        """
        Prints the number of inputs and number of neurons in the network. 
        """
        return '{}-input, {}-neuron network'.format(self.num_inputs, self.num_neurons)

    def return_states(self, dims=None):
        """
         Return the state of the network (by querying each neuron).

         If want the state of all of neurons internal variables,
         include argument dims=dimension of neuron phase space.

         Parameters
         ----------
         dims
             Dimension of neuron

        Returns
        ----------
        np.array
            The state of each neuron at the current time (num_neurons X dims)  
         """
        if dims is None: 
            states = np.zeros(self.num_neurons)
            for i,neuron in enumerate(self.neurons):
                states[i]=neuron.hist[0]
            return states
        else:
            states.np.zeros([self.num_neurons, dims])
            for i,neuron in enumerate(self.neurons):
                states[i, :]=neuron.y
            return states            
    
    def network_solve(self,external_inputs):
        """
        Solve the network given an array of time dependent external inputs.
        
        Steps the network forward in time for each successive row of external_inputs
        using network_step, with the columns defining the input to each neuron.
        External inputs is a num_timesteps X num_inputs array
        Outputs a  num_timesteps X num_neuron array which is the network state 
        at each time-step.

        Parameters
        -----------
        external_inputs
            A 2D array (num_timesteps X num_inputs) with the 
            external inputs (i.e. non-neuron inputs).  

        Returns
        ----------
        np.array
            A 2D array (num_timesteps X num_neuron) of the state of each 
            neuron in the network at each time.
        """
        external_inputs = np.atleast_2d(external_inputs) #asserts 2d array 
        #above converts 1d array (ie 1 input) to shape (1, Len_t), so catch and transpose
        if (external_inputs.shape[0]==1): #dont use solve with single timestep anyways
            external_inputs=external_inputs.transpose() #flip so now has shape (Len_t, 1)
        Len_t=external_inputs.shape[0]#first dimension is time, this is num_timesteps
        msg="External input matrix needs shape ({}, {})".format(Len_t, self.num_inputs)
        assert (external_inputs.shape[1]==int(self.num_inputs)), msg
        network_outputs=np.zeros([Len_t, self.num_neurons])
        network_outputs[0, :]=self.return_states()
        for i in range(Len_t-1):
            network_outputs[i+1, :]=self.network_step(external_inputs[i, :].squeeze()) #step network forward
        return network_outputs

    def network_inputs(self, network_outputs, external_inputs):
        """
        Calculate the time-dependent input to each neuron in the network

        Uses the previously solved for time dependent neuron states and 
        given external inputs (so call after network_solve).

        Parameters
        -----------
        network_outputs
            A 2D array (num_timesteps X num_neuron) of the state of each 
            neuron in the network at each time, as calculated from network_solve.
            external_inputs
            A 2D array (num_timesteps X num_inputs) of the external inputs (i.e. non-neuron inputs).  

        Returns
        ----------
        np.array
            A 2D array (num_timesteps X num_neuron) of the total wieghter and delayed
            input (integrated internal and external) to each neuron at each time.
        """

        def get_prev_outputv2(t, row, col):
            # an internal function to get the inputs needed to update neuron states 
            if col < self.num_inputs:
                return external_inputs[t, col]
            else:
                col = col - self.num_inputs
#                import pdb; pdb.set_trace()
                if (self.delays[row][col] >= t):
                    return network_outputs[0,col] 
                else:
                    return network_outputs[t-self.delays[row, col], col]

        external_inputs = np.atleast_2d(external_inputs) #asserts 2d array 
        #above converts 1d array (ie 1 input) to shape (1, Len_t), so catch and transpose
        if (external_inputs.shape[0]==1): #dont use solve with single timestep anyways
            external_inputs=external_inputs.transpose() #flip so now has shape (Len_t, 1)

        if np.amax(self.delays)==0: #no delays, so this is easy
            Inputs_raw=np.concatenate((external_inputs, network_outputs), axis=1).transpose()
            Inputs=np.dot(self.weights, Inputs_raw).transpose()
        else:
            Len_t=network_outputs.shape[0]
            Inputs=np.zeros([Len_t, self.num_neurons])
            for t in range(Len_t):
                for row in range(self.num_neurons):
                    for col in range(self.num_neurons+self.num_inputs):
                        Inputs[t, row] += self.weights[row,col]*get_prev_outputv2(t,row,col)

        return Inputs

    def network_step_full(self,external_inputs, dim=1):
        """
        Steps the network forward in time by dt, just as network_step,
         with option to return full neuron state.

        Use with network_solve_full or to see full phase space of each neuron as evolve.
        To return state of all of neurons internal variables (num_neurons X neuron.dim), 
        include argument dims=dimension of neuron phase space

        Parameters
        -----------
        external_inputs
            A (num_inputs X 1) array with the external inputs at the current time
            (i.e. non-neuron inputs).
        dims
            Dimension of neuron state (1 if only want output)

        Returns
        ----------
        np.array
            The dims-dimensional state of each neuron after stepping forward by dt
        """
        external_inputs = np.atleast_1d(external_inputs)
        msg="Please specify {} inputs in an array".format(self.num_inputs)
        assert (len(external_inputs)==int(self.num_inputs)), msg
        # update the state of each neuron 
        neuron_inputs = self.generate_neuron_inputs(external_inputs)
        neuron_outputs = np.zeros([self.num_neurons, dim])
        for i,neuron in enumerate(self.neurons):
           neuron_outputs[i, :] = neuron.step(neuron_inputs[i]) 
        return neuron_outputs    

    def network_solve_full(self,external_inputs):
        """
        Solve the network given an array of time dependent external inputs,
        as in network_solve, but returns the full phase space of each neuron, 
        rather than just the state variable.

        Parameters
        -----------
        external_inputs
            A 2D array (num_timesteps X num_inputs) with the external inputs (i.e. non-neuron inputs).  

        Returns
        ----------
        np.array
            A 3D array (num_timesteps X num_neuron X neuron.dim)
            of the full state of each neuron in the network at each time.
        """
        Len_t=external_inputs.shape[0]#first dimension is time
        msg="External input matrix needs shape ({}, {})".format(Len_t, self.num_inputs)
        assert (external_inputs.shape[1]==int(self.num_inputs)), msg
        dim=self.neurons[0].dim
        #maybe assert all neurons have this many dims
        Net=np.zeros(Len_t, self.num_neurons, dim)
        Net[0, :, :]=self.return_states(dim)
        for i in range(Len_t-1):
            Net[i, :, :]=self.network_step_full(external_inputs[i, :].squeeze(), dim) #step network forward
        return Net

    def visualize_plot(self, inputs, outputs, time=None):
        """
        Generate a simple and easy to read plot of the network dynamics. 

        After solving a network for a given set of inputs, pass these inputs
        and the computed result from network_solve to generate a plot. 
        Use returned figure handle to update plot parameters from defaults 
        if desired.

        Parameters
        -----------
        time
            an array of time points which inputs and outputs are plotted over
        inputs
            the 2D array (num_timesteps X num_neurons) array of total inputs
             to each neuron in the network
        outputs
            the 2D array (num_timesteps X num_neurons) of the state of each 
            neuron in the network as a function of time
       
        Returns
        ----------
        matplotlib.figure.Figure
            A matplotlib figure instance showing the network dynamics
        """

        msg1="outputs expected to have {} Neurons".format(self.num_neurons)
        assert (outputs.shape[1]==int(self.num_neurons) ), msg1
        Len_t=outputs.shape[0] #length of time vector
        msg2="inputs and outputs should both the same temporal length"
        assert (inputs.shape[0]==int(Len_t) ), msg2
        if time is None: #didnt pass time, so compute from dt and signal length TL
            time=np.linspace(0., (Len_t-1)*self.dt, num=Len_t)

        fig=plt.figure()
        ax1=fig.add_axes([0,0.0, 1, 0.6])
        ax2=fig.add_axes([0,.7, 1, 0.3])

        # plot Neuron state and input current
        for ind in range(self.num_neurons):
            if 7<=ind<15: #default color cycle only has 7 colors
                ax1.plot(time, outputs[:,ind], '-.', label='Neuron {}'.format(ind+1))
                ax2.plot(time, inputs[:,ind], '-.')
            else:
                ax1.plot(time, outputs[:,ind],label='Neuron {}'.format(ind+1))
                ax2.plot(time, inputs[:,ind])
            
        ax1.set_xlabel('t [$1/\gamma$]')
        ax1.set_ylabel('$I$ [arb units]')
        ax2.set_ylabel('$i_{in}$ [$\gamma$]')

        ax1.set_xlim(time[0], time[-1])
        ax2.set_xlim(time[0], time[-1])
        ax1.legend()

        return fig


    def visualize_animation(self, inputs=None, outputs=None, t_mov=10):
        """
        Visualize the neural network as a graph. The colors of the nodes are 
        animated in response to their outputs as a function of time
        Parameters
        -----------
        inputs
            The inputs to the neural network
        outputs
            The outputs of each neuron in the network
        t_mov
            approximate length of movie in seconds

        Returns
        ----------
        matplotlib.animation.FuncAnimation
            A matplotlib animation of the network dynamics
        """

        import networkx as nx
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation

        fig, ax = plt.subplots(figsize=(6,6))
        ax.set_axis_off()
        
        # if inputs not given, assume a default input of ones
        if inputs is None:
            inputs = np.ones([10000, self.num_inputs])
        # if outputs not given, solve for outputs first
        if outputs is None:
            outputs = self.network_solve(inputs)

        if inputs.ndim == 1: inputs = inputs[:, np.newaxis]
        if outputs.ndim == 1: outputs = outputs[:, np.newaxis]


        g = nx.DiGraph()
        node_names = []
        edge_weights = []

        # create node labels
        for n1 in range(self.num_inputs):
            node_names.append('$I_{0:d}$'.format(n1 + 1))
            g.add_node('$I_{0:d}$'.format(n1 + 1))
        for n1 in range(self.num_neurons):
            node_names.append('$N_{0:d}$'.format(n1 + 1))
            g.add_node('$N_{0:d}$'.format(n1 + 1))
            #g.add_node('$N_{'+str(n1+1)+'}$')

        # set edge weights for thicknesses of lines    
        for n1 in range(self.num_neurons):
            for n2 in range(self.num_inputs + self.num_neurons):
                if self.weights[n1, n2] != 0:
                    g.add_edges_from([(node_names[n2], node_names[self.num_inputs + n1])])
                    edge_weights.append(self.weights[n1, n2])

        # scale edge weights appropriately
        edge_weights = np.array(edge_weights)
        edge_weights = 3.0 * np.abs(edge_weights) / np.max(edge_weights)

        # set node positions 
        g_pos = nx.fruchterman_reingold_layout(g)

        max_input = np.max(inputs) # largest value of all inputs over all time
        max_output = np.max(outputs) # largest value of all outputs over all time
        
        n_cols = np.concatenate([inputs[0,:]/max_input, outputs[0,:]/max_output])

        # draw edges, node labels and the nodes themselves
        edges = nx.draw_networkx_edges(g, pos=g_pos, width=edge_weights, node_size=1200, arrowsize=20)
        node_labels = nx.draw_networkx_labels(g, pos=g_pos, font_size=14, font_color='w')
        nodes_inp = nx.draw_networkx_nodes(g, pos=g_pos, nodelist=list(g.nodes())[:self.num_inputs],
            cmap = plt.cm.viridis, node_color=n_cols[:self.num_inputs], vmin=0.0, vmax=1.0, node_size=1200, node_shape='s') 
        nodes_neu = nx.draw_networkx_nodes(g, pos=g_pos, nodelist=list(g.nodes())[self.num_inputs:],
            cmap = plt.cm.viridis, node_color=n_cols[self.num_inputs:], vmin=0.0, vmax=1.0, node_size=1200, node_shape='o')

        interval = 10 # milliseconds between successive frames in the animation
        #title = ax.set_title('$t = 0.000$')

        time_arr = np.arange(len(inputs)) * self.dt

        #number of frames to skip
        frame_skip_ratio = max(1, int(len(inputs) * interval / (t_mov * 1000.0) ))
        num_frames = int(len(inputs) / frame_skip_ratio)

        def update(i):
            # The i^th frame needs to have the updated color
            n_cols = np.concatenate([1.0*inputs[i * frame_skip_ratio,:]/max_input, 
                1.0*outputs[i * frame_skip_ratio,:]/max_output])
            nodes_inp.set_array(n_cols[:self.num_inputs])
            nodes_neu.set_array(n_cols[self.num_inputs:])
            #title.set_text('$t = {0:.3f}$'.format(time_arr[i * frame_skip_ratio]))

            return nodes_inp, nodes_neu#, title

        anim = FuncAnimation(fig, update, frames=num_frames, interval=10, blit=True)
        return anim


def main():
    """
    If this python script is called directly, create a 16-neuron reservoir computer network and 
    run it. This is useful for profiling purposes
    """

    Gaussian_pulse= lambda x, mu, sig: np.exp(-np.power(x - mu, 2.) 
    / (2 * np.power(sig, 2.)))/(np.sqrt(2*np.pi)*sig)

    Y1mpars={"a": 2, "A": 6.5, "B":-6., "gamma1": 1,
             "gamma2": 1, "kappa": 50, "beta": 5e-1 }#these are the model parameters
    y1_steady_est=[Y1mpars['beta']/Y1mpars['kappa'],
                   Y1mpars['A'],Y1mpars['B'] ] #quick estimate of ss
    Y1params={"model" : "Yamada_1", "y0": y1_steady_est,
        "dt": 1e-2, 'mpar': Y1mpars} #neuron parameters
    y1_steady=Neuron(Y1params).steady_state(y1_steady_est)
    Y1params["y0"]=y1_steady #change model parameters so that starts w this ss
    
    num_neurons=16 #number of neurons in reservoir computer
    num_inputs=4
    #sparsity of reservoir-reservoir coupling matrix 0=empty, 1=full

    #make a list of 16 neurons
    neuronsR=[]
    for ind in range(num_neurons):
        neuronsR.append(Neuron(Y1params))
        
    #set up coupling matrix between neurons
    WRR_density=0.25
    WRR_maxeig=0.3 #scale largest eigenvalue of WRR matrix
    #pull from random matrix centered at 0
    WRR=sp.sparse.random(num_neurons, num_neurons, WRR_density,
                         data_rvs=lambda s: np.random.uniform(-0.5, 0.5, size=s))
    eigWRR, eigvecWRR = sp.sparse.linalg.eigs(WRR)
    eigMax = np.max(np.abs(eigWRR))
    WRR = (WRR_maxeig/eigMax)*WRR
    # set up WIR (input to neuron matrix)
    #also choose randomly distributed matrix
    WIR =-1 + (1+1)*np.random.rand(num_neurons,num_inputs)
    weightsR=sp.sparse.hstack((WIR, WRR)).toarray()
    #also use random delay matrix: [0,1)
    delaysR=np.random.rand(num_neurons, num_neurons)
    #create network
    networkR=Network(neuronsR, weightsR, delaysR, dt=0.001)
    tR_end=20./Y1mpars["gamma1"]; #atleast this long
    NR=int(np.ceil(tR_end/networkR.dt)) #this many points
    timeR=np.linspace(0.,(NR-1)*networkR.dt, num=NR )
    inR=np.zeros([NR, num_inputs])
    inR[:, 0]=0.2*(np.heaviside(timeR, 2))*(0.9+0.2*np.random.rand(NR))
    inR[:, 1]+=0.3*(np.heaviside(timeR, 6))*(0.8+0.4*np.random.rand(NR))
    inR[:, 2]+=0.3*(np.heaviside(timeR, 10))*(-0.5+1.*np.random.rand(NR))
    inR[:, 3]+=0.2*(np.heaviside(timeR, 14))

    # the solution step
    outputR=networkR.network_solve(inR)
    # compute inputs to each neuron
    inputR=networkR.network_inputs(network_outputs=outputR, external_inputs=inR)
    an3 = networkR.visualize_animation(inputs=inR, outputs=outputR)


    

if __name__ == '__main__':
    main()