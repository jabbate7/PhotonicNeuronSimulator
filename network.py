from neuron import Neuron
import numpy as np



# A network is a specific topology of Neurons
class Network:
    """
    A Network is a specific topology of connected Neuron objects. 
    Connections between neurons can be weighted, and also can carry a 
    time delay. There can be multiple input sources, which can be 
    connected to any neurons. These connections also have associated weights
    and time delays. 

    Once initialized, a Network's primary method is to take in a set of signals
    at a given timestep and reveal the updated neuron states. 

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
        Each element in the array corresponds to a neuron in the network.
        Calculated by summing weighted and delayed inputs to that neuron
        For each element, first add weighted external inputs
        Then add appropriatly delayed weighted outputs from other neurons.
        
        Parameters
        ----------
        external_inputs
            A 1D array with the external inputs (i.e. non-neuron inputs).
        """
        def get_prev_output(row, col):
            # an internal function to get the inputs needed to 
            # update neuron states 
            if col < self.num_inputs:
                return external_inputs[col]
            else:
                col = col - self.num_inputs
                if (self.delays[row][col] >= len(self.neurons[col].hist)):
                    #to save space, history starts almost empty and is populated untill full
                    # assumption is that neuron was in initial state for all times before calculation
                    return self.neurons[col].hist[-1] #return final element, which is initial state
                else:
                    return self.neurons[col].hist[self.delays[row][col]] #assumes 1d hist
        # did some baby profiling and this bit is actually slower, what the fuck, why?
        # if np.amax(self.delays)==0: #no delays,use matrix multiplication
        #     inputs_raw=external_inputs
        #     for i,neuron in enumerate(self.neurons):
        #         inputs_raw=np.append(inputs_raw, neuron.hist[0])
        #     inputs=np.dot(self.weights, inputs_raw)
        # else: 
        inputs=np.zeros(self.num_neurons)
        for row in range(self.num_neurons):
            for col in range(self.num_neurons+self.num_inputs):
                if self.weights[row][col] != 0:
                    inputs[row] += self.weights[row][col]*get_prev_output(row,col)
        return inputs


    def network_step(self,external_inputs):
        """
        Steps the network forward in time by dt.
        Each Neuron updates based on its state and its input.

        Parameters
        -----------
        external_inputs
            A (num_inputs X 1) array with the external inputs at the current time
            (i.e. non-neuron inputs).
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
         Return the state of the network (by querying each neuron)
         If want the state of all of neurons internal variables,
         include argument dims=dimension of neuron phase space

         Parameters
         ----------
         dims
             Dimension of neuron
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
        Solve the network given an array of time dependent external inputs
        External inputs is a num_timesteps X num_inputs array.
        Outputs a  num_timesteps X num_neuron array which is the network state 
        at each timestep.

        Parameters
        -----------
        external_inputs
            A 1D array with the external inputs (i.e. non-neuron inputs).
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
        Returns the integrated input to each Neuron as a (num_timesteps X num_neuron) array
        attributes are network_outputs, the (num_timesteps X num_neuron) array containing each neurons state,
        as comupted from network_solve, and external_inputs the (num_timesteps X num_inputs) array of external inputs
        """

        def get_prev_outputv2(t, row, col):
            # an internal function to get the inputs needed to 
            # update neuron states 
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
        Steps the network forward in time by dt, with option to return full neuron state.
        Use with network_solve_full or to see full phase space of each neuron as evolve
        external_inputs is input array at the current time (num_inputs X 1)
        Each Neuron updates based on its state and its input
        Returns array of each neuron's output variable (num_neurons X 1)
        To return state of all of neurons internal variables (num_neurons X neuron.dim)
        include argument dims=dimension of neuron phase space
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
        and return the entire dynamical phase space of each neuron, rather 
        than just the state variable.
        External inputs is a num_timesteps X num_inputs array.
        Outputs a num_timesteps X num_neuron X neuron.dim array which is the
        network state at each timestep

        Parameters
        -----------
        external_inputs
            A 1D array with the external inputs (i.e. non-neuron inputs).
        """
        #skeleton version for now
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

    def visualize(self):
        # visualize the network
        import networkx as nx
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6,6))
        ax.set_axis_off()
        
        g = nx.DiGraph()
        node_names = []
        edge_weights = []

        for n1 in range(self.num_inputs):
            node_names.append('$I_{0:d}$'.format(n1 + 1))
        for n1 in range(self.num_neurons):
            node_names.append('$N_{0:d}$'.format(n1 + 1))
                
        for n1 in range(self.num_neurons):
            for n2 in range(self.num_inputs + self.num_neurons):
                if self.weights[n1, n2] != 0:
                    g.add_edges_from([(node_names[n2], node_names[self.num_inputs + n1])])
                    edge_weights.append(self.weights[n1, n2])

        edge_weights = np.array(edge_weights)
        edge_weights = 3.0 * np.abs(edge_weights) / np.max(edge_weights)
        g_pos = nx.spectral_layout(g)
        n_cols = []
        for n1 in range(self.num_inputs + self.num_neurons):
            n_cols.append(plt.cm.cool(np.random.rand()))

        #nx.draw(g, with_labels=True, ax=ax, 
        #    node_color=n_cols, node_size=1200, pos=g_pos, width=edge_weights, arrowsize=20, fontsize=18)
        nx.draw_networkx_nodes(g, with_labels=True, ax=ax, 
            node_color=n_cols, node_size=1200, pos=g_pos, width=edge_weights, arrowsize=20, fontsize=18)
        nx.draw_networkx_edges(g, with_labels=True, ax=ax, 
            node_color=n_cols, node_size=1200, pos=g_pos, width=edge_weights, arrowsize=20, fontsize=18)